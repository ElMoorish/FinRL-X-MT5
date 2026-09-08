# 🧪 MetaTrader 5 Strategy Tester Guide
### *Real-Tick Backtesting with the K-Dense Council*

The standard industry backtest mode using bar open/close prices or interpolated "1-minute OHLC" ticks produces unrealistic fill rates, ignores spread spikes, and hides slippage.

**FinRL-X-MT5** operates under the institutional standard: **100% Every Tick Based on Real Ticks**.

---

## 🏗️ The Bridge Architecture

```
 Python K-Dense Council                                 MetaTrader 5 Terminal
┌──────────────────────────────────────┐               ┌──────────────────────────────────────┐
│ 1. Pull historical M5 & Ticks        │               │ 4. Open Strategy Tester (Ctrl + R)   │
│ 2. Council runs inference on bars    │               │ 5. Load FinRL_X_MT5.ex5              │
│ 3. Export signals to CSV:            │               │ 6. Model: Every tick based on        │
│    • consensus_signal, direction     │               │    real ticks                        │
│    • position_size, tp, sl, conf     │               │ 7. EA reads finrl_x_signals.csv      │
│    Direct copy to:                   │──────────────►│ 8. Real-time lot sizing, spread      │
│    MQL5\Files\finrl_x_mt5\           │               │    filters, and TP/SL execution      │
└──────────────────────────────────────┘               └──────────────────────────────────────┘
```

---

## 📋 Step-by-Step Backtesting Procedure

### Step 1: Export Real-Tick Signals from Python
Run the export command for your target instrument and backtest window:

```powershell
$env:PYTHONPATH="."
& "C:\Users\aitsi\AppData\Local\Programs\Python\Python311\python.exe" -m src.main_mt5 export-signals --symbol NAS100.x --days 60
```

This command automatically:
1. Gathers the latest M5 bars and tick features from MT5.
2. Fuses the lagged cross-asset fundamental data (QQQ, Mag-7, macro yields).
3. Loads the trained Council models (`models/nas100.x/`).
4. Generates consensus decisions (`consensus_signal`, `direction`, `position_size`, `tp`, `sl`, `conf`, `regime`).
5. **Automatically writes** the signal file to:
   - `<TerminalDataPath>\MQL5\Files\finrl_x_mt5\finrl_x_signals.csv`
   - `<CommonDataPath>\Files\finrl_x_mt5\finrl_x_signals.csv`

---

### Step 2: Open MT5 Strategy Tester
1. In MetaTrader 5, press `Ctrl + R` (or go to `View` $\to$ `Strategy Tester`).
2. Switch to the **Settings** tab.

### Step 3: Configure Tester Inputs
Configure the following fields:

| Setting | Recommended Value | Notes |
|---|---|---|
| **Expert** | `FinRL_X_MT5` | Compiled `.ex5` in `MQL5\Experts\` |
| **Symbol** | `NAS100.x` (or `WTI.x`, `US30.x`, etc.) | Must match broker `.x` symbol |
| **Timeframe** | `M5` | 5-Minute execution timeframe |
| **Date** | `Custom period` | Must match the exported signal window |
| **Forward** | `No` | Python walk-forward handles OOS validation |
| **Delays** | `Zero latency` or `10 ms` | Mimics real broker execution delay |
| **Model** | **`Every tick based on real ticks`** | **CRITICAL:** Do NOT use 1-minute OHLC! |
| **Deposit** | `10,000 USD` (or match account) | Matches your target prop firm account |
| **Leverage** | `1:100` | As provided by broker |

---

### Step 4: Configure EA Parameters
In the **Inputs** tab of the Strategy Tester:
- `SignalFile`: `finrl_x_signals.csv`
- `RiskPctPerTrade`: `2.0` (Risk 2% of equity per trade)
- `MaxDrawdownPct`: `10.0` (Hard stop at 10% drawdown)
- `MinFreeMarginPct`: `30.0` (Block new orders if margin $< 30\%$)
- `SpreadMultiplier`: `3.0` (Filters news spread widening)
- `MinSignalStrength`: `0.20` (Minimum $|signal|$ required to enter)
- `MinConfidence`: `0.40` (Minimum Council confidence threshold)
- `MinExpectedRR`: `1.50` (Minimum Risk:Reward ratio)

---

### Step 5: Run & Analyze
Click **Start**.

During the backtest:
- The EA matches tick timestamps with the nearest confirmed bar in `finrl_x_signals.csv`.
- Order volumes are dynamically computed using the live broker contract size and Half-Kelly scaling.
- Trailing stops and dynamic Take-Profit targets update at each bar close.
- Review the **Graph** tab for real-tick equity curve, drawdown spikes, and Sharpe ratio.
