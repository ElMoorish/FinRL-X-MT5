# 01. The 5-Agent Council Architecture (Mixture-of-Experts)

The fundamental failure point of retail algorithmic trading is **single-model non-stationarity**. Financial time series exhibit evolving statistical distributions: an LSTM or XGBoost model trained on trending bull markets suffers catastrophic drawdown the moment the market shifts into high-volatility sideways mean-reversion.

To overcome single-model fragility, **FinRL-X** implements a **5-Expert Mixture-of-Experts (MoE)** architecture. Every 5-minute candle close, five decoupled specialists evaluate market state independently before casting weighted votes into an NSGA-III Pareto consensus gate.

---

## 🧠 The 5 Council Specialists

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       THE 5 SPECIALIST AGENTS                               │
├───────────────────┬───────────────────────────────────┬─────────────────────┤
│ Specialist        │ Underlying Engine                 │ Primary Mandate     │
├───────────────────┼───────────────────────────────────┼─────────────────────┤
│ E1: DRL Trader    │ Soft Actor-Critic (SAC)           │ Direction Conviction│
│ E2: Regime Master │ 3-State Gaussian HMM              │ Market Regime Gate  │
│ E3: Prophet       │ TimesFM 2.5 Zero-Shot Transformer │ Volatility Corridor │
│ E4: Analyst       │ XGBoost + SHAP Tree Explainability│ Order-Flow Delta    │
│ E5: Chief Actuary │ PyMC Bayesian Credible Intervals  │ Risk & Stop Pricing │
└───────────────────┴───────────────────────────────────┴─────────────────────┘
```

---

### Expert 1: DRL Trader (Soft Actor-Critic)
- **Engine:** PyTorch continuous-action Soft Actor-Critic (SAC) with twin Q-networks and automated entropy temperature adjustment ($\alpha$).
- **State Space:** 48 normalized features including M5 return autocorrelation, normalized ATR, Bollinger Band bandwidth, RSI momentum, and tick volume velocity.
- **Action Space:** Continuous directional conviction $a_t \in [-1.0, +1.0]$:
  - $a_t > +0.30 \implies \text{Long Conviction}$
  - $a_t < -0.30 \implies \text{Short Conviction}$
  - $-0.30 \le a_t \le +0.30 \implies \text{Neutral Hold}$
- **Reward Formulation:**
  $$R_t = \frac{r_t}{\sigma_t} - \lambda_{\text{DD}} \cdot \mathbb{I}_{\text{DD}} - \lambda_{\text{cost}} \cdot (\text{spread} + \text{slippage})$$
  The agent is explicitly penalized for drawdown duration and trade friction, discouraging over-trading in noisy sessions.

---

### Expert 2: Regime Master (Gaussian Hidden Markov Model)
- **Engine:** 3-state Gaussian HMM trained on rolling log returns and normalized high-low spread variance.
- **States & Position Sizing Multipliers:**
  1. **State 0 (Bullish Momentum):** Low variance, positive mean drift. Multiplier: **`1.00x`** (full allocation).
  2. **State 1 (High-Volatility Bear/Chop):** High variance, negative skew. Multiplier: **`0.25x`** (defensive dampening).
  3. **State 2 (Sideways Mean-Reversion):** Low volatility, zero drift. Multiplier: **`0.60x`** (tight targets).
- **Veto Authority:** If the posterior state probability $P(\text{State } 1) > 0.65$, all breakout expansion trades are vetoed to protect against bull-traps.

---

### Expert 3: Prophet (TimesFM 2.5 Zero-Shot Transformer)
- **Engine:** Google Research TimesFM 2.5 foundation time-series transformer.
- **Function:** Generates multi-horizon forecasts for the next 12 M5 intervals (1 hour) without task-specific fine-tuning.
- **Volatility Band Calibration:** Projects 90th and 10th percentile expected price envelopes:
  $$\text{Corridor Upper} = \hat{y}_{t+k}^{(90)}, \quad \text{Corridor Lower} = \hat{y}_{t+k}^{(10)}$$
- Prevents entries when price is already extended into the 90th percentile exhaustion zone.

---

### Expert 4: Quantitative Analyst (XGBoost + SHAP)
- **Engine:** Extreme Gradient Boosting (XGBoost) classifier with real-time TreeSHAP contribution values.
- **Input Vectors:** Bid/ask volume delta, CVD (Cumulative Volume Delta) divergence, Tick Imbalance Ratios, and VWAP displacement.
- **Explainability Filter:** Trades require positive SHAP attribution from order-flow momentum. If technical indicators indicate long but SHAP reveals institutional volume distribution, the trade is suppressed.

---

### Expert 5: Chief Risk Actuary (PyMC Bayesian Inference)
- **Engine:** Markov Chain Monte Carlo (MCMC) sampling via PyMC.
- **Function:** Calibrates dynamic Stop Loss and Take Profit levels by computing the 95% Bayesian Highest Density Interval (HDI) of adverse excursion.
- **Mandatory Enforcement:**
  1. Applies an upfront **60.0-point index breathing floor** on NAS100/US30.
  2. Computes the maximum volume allowed under the strict **0.50% capital risk ceiling**.
  3. Clamps volume with `math.floor` and blocks trades where minimum broker lot exceeds the risk budget.

---

## ⚖️ The Consensus Deliberation Protocol

At each M5 candle close, the `Council` engine executes the following voting algorithm in [`src/council/council.py`](file:///c:/Users/aitsi/Desktop/FinRL-X-MT5/src/council/council.py):

```python
# Council Consensus Score Calculation
consensus_score = (
    w_sac * e1_action +
    w_regime * e2_direction * e2_confidence +
    w_prophet * e3_direction +
    w_analyst * e4_signal * e4_prob
)

# High Conviction Gate
if abs(consensus_score) >= 0.70:
    candidate_direction = "BUY" if consensus_score > 0 else "SELL"
    
    # Rule 7: Macro Trend Governor Check
    if candidate_direction == "BUY" and current_bid < h1_ema50:
        logger.warning("VETO: Long trade below H1 EMA 50 (Rule 7)")
        return None
    if candidate_direction == "SELL" and current_ask > h1_ema50:
        logger.warning("VETO: Short trade above H1 EMA 50 (Rule 7)")
        return None
        
    return approve_trade(candidate_direction, consensus_score)
```

---

## 🛡️ Rule 7: The H1 Macro Trend Governor

Even if short-term M5 momentum displays a strong signal, counter-trend trades against the higher timeframe structural trend have a failure rate exceeding 68%. 

The Council enforces **Rule 7**:
- **BUY Trades:** Permitted **only** if current Bid price is strictly **above** the 50-period Exponential Moving Average on the H1 timeframe (`Bid > H1_EMA50`).
- **SELL Trades:** Permitted **only** if current Ask price is strictly **below** the 50-period Exponential Moving Average on the H1 timeframe (`Ask < H1_EMA50`).

This single rule eliminates whipsaw fakeouts during macro trend days.
