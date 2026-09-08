# 🏛️ K-Dense Council: System Architecture

The **FinRL-X-MT5** trading platform is built on the **K-Dense Council**, a multi-agent **Mixture-of-Experts (MoE)** architecture designed specifically for high-frequency algorithmic trading of Indices, Commodities, and Equities via MetaTrader 5.

Traditional single-model reinforcement learning systems suffer from severe sample inefficiency, non-stationarity breakdown, and catastrophic forgetting when market regimes shift. The K-Dense Council addresses this by decomposing the trading problem into **five specialized institutional experts** orchestrated by an **NSGA-III Pareto-optimal gating optimizer**.

---

## 📐 Architecture Topology

```
                               ┌──────────────────────────────────────────────┐
                               │             MT5 High-Frequency Ticks         │
                               │          + Yahoo/Macro Correlation Baskets   │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                           ┌──────────▼──────────┐
                                           │  Correlation Fuser  │
                                           │  (Polars M5 Matrix) │
                                           └──────────┬──────────┘
                                                      │
                       ┌──────────────────────────────┼──────────────────────────────┐
                       │                              │                              │
             ┌─────────▼─────────┐          ┌─────────▼─────────┐          ┌─────────▼─────────┐
             │    Expert 2       │          │    Expert 4       │          │    Expert 3       │
             │   HMM Regime      │          │  SHAP + XGBoost   │          │ TimesFM / EWMA    │
             │ (Market State)    │          │  (Signal Scorer)  │          │(Volatility Bands) │
             └─────────┬─────────┘          └─────────┬─────────┘          └─────────┬─────────┘
                       │                              │                              │
                       │                              │                              │
             ┌─────────▼─────────┐                    │                    ┌─────────▼─────────┐
             │    Expert 1       │                    │                    │    Expert 5       │
             │    SAC DRL        │                    │                    │ Bayesian Actuary  │
             │ (Action / Conv)   │                    │                    │ (Dynamic TP / SL) │
             └─────────┬─────────┘                    │                    └─────────┬─────────┘
                       │                              │                              │
                       └──────────────────────────────┼──────────────────────────────┘
                                                      │
                                           ┌──────────▼──────────┐
                                           │   NSGA-III Gate     │
                                           │ (Pareto α Weighting)│
                                           └──────────┬──────────┘
                                                      │
                                           ┌──────────▼──────────┐
                                           │   Consensus Signal  │
                                           │  (Direction & Size) │
                                           └──────────┬──────────┘
                                                      │
                                 ┌────────────────────┴────────────────────┐
                                 │                                         │
                       ┌─────────▼─────────┐                     ┌─────────▼─────────┐
                       │  Strategy Tester  │                     │    Live Socket    │
                       │ (Real-Tick CSV)   │                     │    Order Engine   │
                       └───────────────────┘                     └───────────────────┘
```

---

## 🧠 The 5 Council Specialists

### 1. Expert 1 — The Trader (Soft Actor-Critic DRL)
- **Engine:** `stable-baselines3` (GPU-accelerated PyTorch / CUDA).
- **Mathematical Basis:** Maximum Entropy Reinforcement Learning:
  $$\max_{\pi} \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t \left( R(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right) \right]$$
- **State Representation:** Rolling tensor of normalized microstructure and technical features over a 50-bar lookback window:
  $$s_t \in \mathbb{R}^{50 \times d}$$
- **Action Space:** Continuous directional conviction:
  $$a_t \in [-1.0, 1.0]$$
  where $+1.0$ represents maximum long conviction, $-1.0$ represents maximum short conviction, and $0.0$ represents neutral/cash.
- **Reward Formulation:** Online differential Sharpe ratio penalized for drawdown excursions exceeding 5%:
  $$R_t = \frac{\mu_{pnl}}{\sigma_{pnl} + \epsilon} \cdot \text{PnL}_t - 2.0 \cdot \max(0, \text{Drawdown}_t - 0.05)$$

---

### 2. Expert 2 — The Oracle (Gaussian Hidden Markov Model)
- **Engine:** `hmmlearn` + `aeon`.
- **Role:** Discrete market regime classification and risk throttling.
- **States:** 3 distinct hidden states:
  - **State 0 (Bear):** High volatility, negative mean return drift.
  - **State 1 (Sideways / Consolidation):** Mean-reverting, low directional momentum, compressed ATR.
  - **State 2 (Bull):** Positive mean return drift, sustained buying pressure.
- **Inputs:** Return distribution, volume Z-score, normalized ATR, and tick imbalance.
- **Regime Modifier:** Gates Council sizing dynamically:
  $$\text{Modifier} = \begin{cases} 
  1.0 & \text{if regime matches signal direction} \\ 
  0.25 & \text{if sideways regime} \\ 
  0.0 & \text{if conflicting regime (hard block)} 
  \end{cases}$$

---

### 3. Expert 3 — The Prophet (TimesFM / Analytical EWMA Quantiles)
- **Engine:** Google TimesFM 2.5 (200M parameter zero-shot foundation model) with analytical EWMA volatility fallback.
- **Role:** High-frequency forward volatility projection over a 60-minute horizon ($12 \times \text{M5}$ bars).
- **Outputs:**
  - Forward projected drift $\hat{\mu}_{t+h}$
  - Expected upper and lower volatility envelopes:
    $$P_{\text{upper}} = P_t \cdot \left(1 + z_{0.95} \cdot \hat{\sigma}_{t+h}\right)$$
    $$P_{\text{lower}} = P_t \cdot \left(1 - z_{0.95} \cdot \hat{\sigma}_{t+h}\right)$$
  - Volatility expansion/compression probability.

---

### 4. Expert 4 — The Analyst (SHAP + XGBoost)
- **Engine:** `xgboost` + `shap` TreeExplainer.
- **Role:** Feature importance attribution and non-linear signal quality verification.
- **Functionality:**
  - Trains an ensemble of gradient-boosted decision trees to predict forward bar returns.
  - Computes exact Shapley values for all incoming features:
    $$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} (v(S \cup \{i\}) - v(S))$$
  - Verifies whether current price movement is driven by genuine fundamental cross-asset flows (e.g. QQQ/NVDA lead flows for NAS100) or spurious noise.

---

### 5. Expert 5 — The Actuary (Bayesian Dynamic Risk Engine)
- **Engine:** `pymc` (No-U-Turn Sampler / NUTS) with high-speed analytical Student's t / Extreme Value Theory (EVT) fallback.
- **Role:** Calculates dynamic Take-Profit (TP) and Stop-Loss (SL) price targets based on posterior return distributions rather than static pip levels.
- **Parameters:**
  - **Stop-Loss:** Tail value-at-risk ($\text{CVaR}_{99\%}$) derived from posterior predictive distributions.
  - **Take-Profit:** Upper credible interval calibrated to deliver a minimum expected Risk:Reward ratio:
    $$\mathbb{E}[\text{RR}] \ge 1.50$$

---

## ⚖️ The Gate: NSGA-III Pareto Multi-Objective Optimization

Rather than using arbitrary static weights, the Council's blending weights $\boldsymbol{\alpha} = [\alpha_1, \alpha_2, \alpha_3, \alpha_4, \alpha_5]$ are solved using **NSGA-III** (Non-dominated Sorting Genetic Algorithm III via `pymoo`).

### Optimization Objectives:
$$\min_{\boldsymbol{\alpha}} \mathbf{F}(\boldsymbol{\alpha}) = \begin{bmatrix} 
-\text{SharpeRatio}(\boldsymbol{\alpha}) \\ 
\text{MaxDrawdown}(\boldsymbol{\alpha}) \\ 
\text{Turnover}(\boldsymbol{\alpha}) 
\end{bmatrix}$$

### Constraints:
$$\sum_{k=1}^{5} \alpha_k = 1.0, \quad 0 \le \alpha_k \le 0.60 \quad \forall k$$

The Council maintains diversity and prevents any single expert from dominating the trading decision.

---

## ⚡ Execution Hot Path

During live trading or backtesting:
1. New M5 bar closes or tick arrives.
2. Microstructure features and lagged Yahoo EOD metrics are computed in memory.
3. **Expert 2 (HMM)** identifies the regime and assigns the throttle modifier.
4. **Experts 1, 3, 4, 5** evaluate the feature vector in parallel or sequential pipeline.
5. **NSGA-III Gate** applies the optimal $\boldsymbol{\alpha}$ weights:
   $$S_{\text{consensus}} = \sum_{k=1}^{5} \alpha_k \cdot s_k$$
6. **Pre-Trade Risk Manager** verifies:
   - Max portfolio risk & account drawdown circuit breaker.
   - Spread spike filter ($< 3\times$ typical spread).
   - Minimum Council confidence threshold ($\ge 0.40$).
7. **Lot Sizer** computes order volume scaled by Half-Kelly confidence and broker contract size.
8. Trade dispatched to MT5 terminal or recorded to Strategy Tester CSV.
