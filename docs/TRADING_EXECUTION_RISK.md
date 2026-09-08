# 🛡️ Trading Layer: Execution, Lot Sizing & Risk Management

Algorithmic trading across disparate asset classes (Commodities, Indices, Equities) requires rigorous **contract-size normalization** and institutional risk guardrails. A $1.00 move in Silver (`XAGUSD.x`, contract size 5,000) represents \$5,000 per lot, whereas a $1.00 move in `US30.x` (contract size 1.0) represents only \$1.00 per lot.

**FinRL-X-MT5** implements dynamic position sizing and multi-tiered pre-trade risk controls to guarantee capital preservation.

---

## 📐 1. Contract Size Normalized Position Sizing

Position volume is never calculated as static lots. The `MT5LotSizer` calculates volume based on dollar risk, stop-loss distance, and the Council's confidence score:

$$\text{DollarRisk} = \text{Equity} \times \text{RiskPctPerTrade} \times \text{PositionSize} \times \text{HalfKellyMultiplier}$$

$$\text{Volume}_{\text{raw}} = \frac{\text{DollarRisk}}{|\text{EntryPrice} - \text{SL}| \times \text{ContractSize}}$$

The raw volume is then quantized to the broker's execution constraints:
$$\text{Volume} = \text{clamp}\left( \text{round}\left(\frac{\text{Volume}_{\text{raw}}}{\text{LotStep}}\right) \times \text{LotStep}, \; \text{MinLot}, \; \text{MaxLot} \right)$$

### Broker Contract Size Table

| Symbol | Description | Contract Size | 1-Lot Point Value | Min Lot | Lot Step |
|---|---|---|---|---|---|
| **`NAS100.x`** | US Tech 100 | **10.0** | \$0.10 per point | 0.01 | 0.01 |
| **`WTI.x`** | WTI Crude Oil | **100.0** | \$1.00 per point | 0.01 | 0.01 |
| **`XAGUSD.x`** | Silver Spot | **5,000.0** | \$5.00 per point | 0.01 | 0.01 |
| **`US30.x`** | Wall Street 30 | **1.0** | \$1.00 per point | 0.01 | 0.01 |
| **`GER40.x`** | Germany 40 | **1.0** | €0.10 per point | 0.01 | 0.01 |
| **`SPX500.x`** | US 500 | **10.0** | \$1.00 per point | 0.10 | 0.10 |
| **`JAP225.x`** | Japan 225 | **100.0** | ¥100 per point | 0.01 | 0.01 |
| **Equities** | US Stocks | **1.0** | \$0.01 per point | 0.10 | 0.10 |

---

## 🛑 2. Pre-Trade Risk Guardrails

Before any order is dispatched to the MT5 trade queue, `RiskManager` evaluates five institutional criteria:

```
 Incoming Trade Signal
          │
          ▼
 ┌───────────────────────────────────┐     FAIL
 │ 1. Account Drawdown < 10%?        │ ──────────► [HALT ALL TRADING]
 └─────────────────┬─────────────────┘
          │ PASS
          ▼
 ┌───────────────────────────────────┐     FAIL
 │ 2. Free Margin > 30%?             │ ──────────► [BLOCK ORDER]
 └─────────────────┬─────────────────┘
          │ PASS
          ▼
 ┌───────────────────────────────────┐     FAIL
 │ 3. Current Spread < 3× Typical?   │ ──────────► [DELAY ORDER (News Spike)]
 └─────────────────┬─────────────────┘
          │ PASS
          ▼
 ┌───────────────────────────────────┐     FAIL
 │ 4. Council Confidence ≥ 0.40?     │ ──────────► [REJECT WEAK CONVICTION]
 └─────────────────┬─────────────────┘
          │ PASS
          ▼
 ┌───────────────────────────────────┐     FAIL
 │ 5. Expected RR ≥ 1.50?            │ ──────────► [REJECT POOR ASYMMETRY]
 └─────────────────┬─────────────────┘
          │ PASS
          ▼
    DISPATCH ORDER TO MT5
```

---

## ⚡ 3. Order Execution Protocol

Orders are transmitted using MetaTrader 5's native `ORDER_TYPE_BUY` or `ORDER_TYPE_SELL` structures:
- **Type Filling:** `ORDER_FILLING_IOC` (Immediate-or-Cancel) or `ORDER_FILLING_FOK` (Fill-or-Kill) to prevent partial fills at adverse prices.
- **Magic Number:** `20260908` tags all orders created by the Council, preventing interference with manual positions.
- **Dynamic TP/SL:** Take-Profit and Stop-Loss are set at order creation, ensuring broker-side execution even in the event of local disconnection.
- **Drawdown Circuit Breaker:** If equity falls by $> 10\%$ from high-water mark, an emergency trigger liquidates open positions and halts trading until manual review.
