# 03. The +1.0R Dynamic Breakeven Engine

A pervasive psychological and mathematical pitfall in trading is allowing a winning position to reverse into a terminal stop-out. In prop firm evaluations, where daily equity is benchmarked against the 00:00 UTC snapshot, a position that achieves +100 points of profit and subsequently reverses to hit a -60 point stop loss inflicts a **160-point net equity swing** on the account.

**FinRL-X** implements an autonomous **+1.0R Dynamic Breakeven Trailing Engine** directly within the MetaTrader 5 position monitor loop.

---

## ⚙️ Mathematical Trigger Mechanics

For every open position monitored on the live tick stream:

### 1. Initial Risk Distance ($R$)
$$R = |\text{Open Price} - \text{Original Stop Loss}|$$

*Example on NAS100.x BUY order:*
- Open Price: `29,423.32`
- Stop Loss: `29,243.92`
- Initial Risk $R = 29,423.32 - 29,243.92 = \mathbf{179.40 \text{ points}}$

### 2. Floating Profit Calculation ($P_{\text{float}}$)
$$\text{For BUY:} \quad P_{\text{float}} = \text{Current Bid} - \text{Open Price}$$
$$\text{For SELL:} \quad P_{\text{float}} = \text{Open Price} - \text{Current Ask}$$

### 3. The Breakeven Activation Condition
$$\text{Trigger Condition:} \quad P_{\text{float}} \ge 1.0 \times R$$

The moment market price achieves a favorable excursion equal to or greater than the original dollar risk taken, the order modification protocol is dispatched.

---

## 🛡️ The +10-Point Spread Buffer (Avoiding Retail Traps)

Moving a stop-loss to the exact open price (`SL = Open Price`) is a classic retail flaw:
1. Broker spread fluctuates dynamically during high-volatility tick bursts.
2. Even if price touches entry, closing at Bid means the trade absorbs commission and spread, resulting in a net negative dollar loss.
3. Microscopic liquidity noise often re-tests the breakout point before continuation.

To guarantee that a breakeven exit is **strictly profit-positive**, FinRL-X offsets the new stop loss by an institutional buffer:

$$\text{New SL (BUY)} = \text{Open Price} + (10.0 \times \text{Point Value})$$
$$\text{New SL (SELL)} = \text{Open Price} - (10.0 \times \text{Point Value})$$

*Numerical execution on NAS100.x:*
$$\text{New SL} = 29,423.32 + 10.0 \times 0.01 = 29,423.32 + 0.10 = \mathbf{29,423.42}$$

This guarantees that if the position is stopped out after achieving +1.0R, the execution covers broker commission and yields a micro-positive balance, completely preserving daily drawdown limits.

---

## 🔄 Position State Progression Diagram

```text
[ ORDER OPENED ] ──► Initial Risk: -0.50% Capital ($50.00 at SL)
        │
        ▼ (Price expands favorably in direction of trend)
[ FLOATING PROFIT REACHES +1.0R ] (e.g. +179.40 points)
        │
        ▼ (Autonomous MT5 OrderSend modification request)
[ STOP LOSS MODIFIED TO ENTRY + 10 PTS ]
        │
        ├─────────────────────────────┬─────────────────────────────┐
        ▼                             ▼                             ▼
[ SCENARIO A: CONTINUATION ]  [ SCENARIO B: REVERSAL ]      [ SCENARIO C: TRAIL ]
Reaches Take Profit (+1.5R)   Stops out at Breakeven + Buffer Trailing stop locks
Net Result: +$75.00 (+0.75%)  Net Result: +$0.50 (FREE ROLL)  additional profit at
Account equity surges.        Daily drawdown 100% immune.   +1.5R and +2.0R steps.
```

---

## 💻 Engine Implementation in `MT5Executor`

The logic is executed on every tick cycle inside [`src/trading/mt5_executor.py`](file:///c:/Users/aitsi/Desktop/FinRL-X-MT5/src/trading/mt5_executor.py):

```python
def check_and_apply_breakeven(self, pos, info) -> bool:
    """Modify open position stop loss to entry + buffer when profit >= 1.0R."""
    point = info.point
    digits = info.digits
    open_price = pos.price_open
    current_sl = pos.sl
    
    # BUY POSITION
    if pos.type == mt5.ORDER_TYPE_BUY:
        initial_risk = open_price - current_sl
        if initial_risk <= 0:
            return False  # Already in profit or invalid SL
            
        floating_profit = pos.price_current - open_price
        
        # Check 1.0R threshold
        if floating_profit >= 1.0 * initial_risk:
            new_sl = round(open_price + (10.0 * point), digits)
            if new_sl > current_sl:
                return self.modify_order_sl(pos.ticket, new_sl)

    # SELL POSITION
    elif pos.type == mt5.ORDER_TYPE_SELL:
        initial_risk = current_sl - open_price
        if initial_risk <= 0:
            return False
            
        floating_profit = open_price - pos.price_current
        
        if floating_profit >= 1.0 * initial_risk:
            new_sl = round(open_price - (10.0 * point), digits)
            if new_sl < current_sl:
                return self.modify_order_sl(pos.ticket, new_sl)
                
    return False
```
