# 02. Risk Management Architecture & The 0.50% Capital Ceiling

In institutional proprietary trading, capital preservation supersedes win rate. A strategy with an 80% win rate will inevitably fail if its position sizing allows loss clustering to violate a 5.0% daily drawdown limit or a 10.0% trailing maximum drawdown threshold.

**FinRL-X** enforces a mathematically inviolable **0.50% Capital Risk Ceiling** across all live order routing.

---

## 📐 Mathematical Formulation of Position Sizing

For any trade, the maximum permissible position volume is strictly determined by:
1. Current account equity ($E$)
2. Distance from entry price to stop-loss in points ($\Delta P_{\text{pts}}$)
3. Broker contract multiplier ($C_{\text{size}}$)
4. Broker minimum lot step ($S_{\text{lot}}$)
5. Capital risk percentage limit ($R_{\text{pct}} = 0.0050$)

### Step 1: Dollar Loss Budget
$$\text{Max Risk USD} = E \times R_{\text{pct}}$$
*Example on $10,000 equity:* $\text{Max Risk USD} = \$10,000 \times 0.0050 = \mathbf{\$50.00}$

### Step 2: Loss Incurred Per 1.00 Standard Lot
$$\text{Loss Per Lot} = \Delta P_{\text{pts}} \times C_{\text{size}}$$
*On NAS100.x with a 60-point stop and contract size of 10.0:*
$$\text{Loss Per Lot} = 60.0 \times 10.0 = \mathbf{\$600.00 \text{ per lot}}$$

### Step 3: Floor Quantization (Never Round Up)
Retail bots often use `round()` which can round volume up, inadvertently inflating risk above the ceiling. FinRL-X enforces strict floor-quantization:
$$\text{Max Risk Lots} = \left\lfloor \frac{\text{Max Risk USD}}{\text{Loss Per Lot} \times S_{\text{lot}}} \right\rfloor \times S_{\text{lot}}$$

*Numerical execution:*
$$\text{Max Risk Lots} = \left\lfloor \frac{\$50.00}{\$600.00 \times 0.01} \right\rfloor \times 0.01 = \left\lfloor \frac{50}{6} \right\rfloor \times 0.01 = 8 \times 0.01 = \mathbf{0.08 \text{ lots}}$$

### Step 4: Minimum Lot Budget Overrun Protection
If the broker's minimum volume step ($0.01$ lots) at the current stop distance would risk more than the permitted $\$50.00$, the trade is **immediately rejected with 0.0 lots**:
```python
if max_risk_lots < broker_min_lot:
    logger.warning(f"Stop distance too wide ({sl_dist} pts) for 0.50% budget. Order BLOCKED.")
    return 0.0
```

---

## 🔍 Post-Mortem & Architecture Hardening: The 0.41 Lot Bug

During live forward testing, an anomaly occurred where an order opened at **0.41 lots** on a $10k account, creating an unexpected risk of $251.04 instead of $50.00. 

A rigorous quantitative audit uncovered two structural flaws that have since been permanently resolved:

### Flaw 1: Order of Operations Decoupling
- **What happened:** The Chief Actuary computed a tight raw stop of `12.8 points`. Sizing volume against 12.8 points produced `0.41 lots` ($52.93 risk). Later in the execution bridge, an index volatility buffer unilaterally expanded the stop to `61.2 points` without re-quantizing the lot size. The monetary risk multiplied 4.8x.
- **The Permanent Fix:** Stop sanitization was decoupled and moved **prior** to lot sizing. Volume is now sized against the final sanitized stop that the broker will actually receive.

### Flaw 2: Unsafe Contract Size Fallback
- **What happened:** If MT5's `symbol_info.trade_contract_size` returned `None`, legacy code defaulted to `1.0`. On NAS100 (where real contract size is `10.0`), dividing by 1.0 resulted in a 10x oversized volume calculation.
- **The Permanent Fix:** The unsafe `1.0` fallback was eliminated completely. If the broker contract size is invalid or $\le 0$, the order is blocked immediately:
```python
contract_size = info.trade_contract_size if (info and info.trade_contract_size > 0) else None
if not contract_size or contract_size <= 0:
    logger.error(f"Cannot execute order: missing trade_contract_size for {symbol}")
    return 0.0
```

---

## 🛡️ Rule 8: Pre-Trade Monetary Risk Budget Verification

Even after volume calculation, [`RiskManager.validate_trade`](file:///c:/Users/aitsi/Desktop/FinRL-X-MT5/src/trading/risk_manager.py) performs an independent pre-flight sanity check before any order payload is dispatched to MetaTrader 5:

```python
# Rule 8: Monetary Risk Budget Verification
actual_dollar_risk = lots * sl_distance_pts * contract_size
max_allowed_dollar_risk = equity * default_risk_pct * 1.01  # 1% sub-cent buffer

if actual_dollar_risk > max_allowed_dollar_risk:
    return False, (
        f"Order rejected: Monetary risk (${actual_dollar_risk:.2f}) "
        f"exceeds 0.50% ceiling (${max_allowed_dollar_risk:.2f})"
    )

# Hard Physical Lot Ceiling Check
if lots > config_max_lot:
    return False, f"Order rejected: Lots ({lots}) exceed hard cap ({config_max_lot})"
```

---

## 📊 Live Verification Audit: Ticket #40558123

Under active live deployment on institutional server `GoatFunded-Server3`:
- **Account Equity:** $\$9,805.56$
- **Asset:** `NAS100.x` (Contract Size: `10.0`)
- **Volume Opened:** `0.02 Lots`
- **Entry Price:** `29,423.32`
- **Stop Loss:** `29,243.92` (Distance: `179.40 points`)
- **Monetary Exposure at SL:**
  $$\text{Actual Risk} = 0.02 \times 179.40 \times 10.0 = \mathbf{\$35.88}$$
- **Percentage Risk:**
  $$\text{Risk \%} = \frac{\$35.88}{\$9,805.56} = \mathbf{0.366\%}$$

The trade risk strictly obeyed the 0.50% ceiling with zero overrun.
