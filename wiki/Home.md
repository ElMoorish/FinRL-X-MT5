# FinRL-X Prime Quant &bull; Technical Knowledge Base

[![Platform](https://img.shields.io/badge/Platform-primeclub--quant.vercel.app-C5A059?style=flat-square&logo=vercel)](https://primeclub-quant.vercel.app/)
[![MT5](https://img.shields.io/badge/MetaTrader-Build_6180-blue?style=flat-square&logo=metaquotes)](https://www.metatrader5.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python)](https://python.org)
[![Telegram](https://img.shields.io/badge/Telegram-Signals_Channel-229ED9?style=flat-square&logo=telegram)](https://t.me/primeclubsignals_public)
[![Discord](https://img.shields.io/badge/Discord-Prime_Quant_Desk-5865F2?style=flat-square&logo=discord)](https://discord.gg/Ch3DxJ2Bb)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://github.com/ElMoorish/FinRL-X-MT5/blob/main/LICENSE)

Welcome to the **FinRL-X-MT5** technical wiki. This documentation provides deep mathematical formulations, engineering blueprints, execution teardowns, and setup guides for our open-source 5-Agent Mixture-of-Experts (MoE) quantitative council operating on **MetaTrader 5**.

> 🌐 **Turn-key Institutional Access:** Don't want to manage local Python processes and MetaTrader 5 server daemons? Access our low-latency webhook dispatch at **[primeclub-quant.vercel.app](https://primeclub-quant.vercel.app/)** or subscribe to our [VIP Alpha Telegram Channel](https://t.me/primeclubsignals_public).

---

## 🏛️ System Architecture Blueprint

Unlike traditional retail Expert Advisors that rely on static heuristics, single-layer LSTMs, or dangerous Martingale/Grid sizing, **FinRL-X** decouples trade generation, market regime classification, volatility forecasting, order-flow momentum, and capital governance into specialized decoupled agents:

```text
               ┌─────────────────────────────────────────────────────────┐
               │              M5 CANDLE CLOSE EVENT DISPATCH             │
               └───────────────────────────┬─────────────────────────────┘
                                           │
         ┌───────────────────┬─────────────┴───────┬───────────────────┐
         ▼                   ▼                     ▼                   ▼
 ┌───────────────┐   ┌───────────────┐     ┌───────────────┐   ┌───────────────┐
 │   EXPERT 1    │   │   EXPERT 2    │     │   EXPERT 3    │   │   EXPERT 4    │
 │  DRL Trader   │   │ Regime Master │     │    Prophet    │   │    Analyst    │
 │ (Soft Actor-  │   │ (Gaussian HMM │     │  (TimesFM 2.5 │   │  (XGBoost +   │
 │   Critic)     │   │   3-State)    │     │  Transformer) │   │     SHAP)     │
 └───────┬───────┘   └───────┬───────┘     └───────┬───────┘   └───────┬───────┘
         │                   │                     │                   │
         └───────────────────┼─────────────────────┴───────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────────────────────┐
         │             NSGA-III PARETO CONSENSUS GATE            │
         │   • High Conviction Threshold: Consensus ≥ 0.70       │
         │   • Rule 7: H1 Macro Trend Governor (EMA 50 Filter)   │
         └───────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────────┐
         │               EXPERT 5: CHIEF RISK ACTUARY            │
         │  • PyMC Bayesian Credible Stop Loss / Take Profit     │
         │  • 60.0-Point Volatility Breathing Floor on Indices   │
         │  • Floor-Quantized 0.50% Capital Risk Ceiling         │
         │  • Rule 8 Pre-Trade Monetary Risk Budget Guard        │
         └───────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────────┐
         │           MT5 TICK EXECUTION & TRAILING ENGINE        │
         │  • Low-Latency Market Order Execution (< 85ms)        │
         │  • Dynamic +1.0R Breakeven Stop Modification          │
         │  • Autonomous 3-Stage Portfolio Circuit Breaker       │
         └───────────────────────────────────────────────────────┘
```

---

## 📚 Technical Wiki Index

| Chapter | Topic | Description |
| :--- | :--- | :--- |
| **[01. The 5-Agent Council](01-Council-Mixture-of-Experts)** | Neural Architecture | Deep dive into SAC, HMM, TimesFM, XGBoost, and Actuary models. |
| **[02. 0.50% Risk Ceiling & Sizing](02-Risk-Management-&-0.50%-Ceiling)** | Mathematical Risk | Formal sizing equations, index contract multipliers, and Rule 8 ceiling. |
| **[03. +1.0R Dynamic Breakeven](03-Dynamic-Breakeven-Engine)** | Order Lifecycle | Automating the risk-free free-roll transition at +1.0R profit. |
| **[04. MetaTrader 5 Bridge Setup](04-MetaTrader-5-Bridge-Setup)** | Installation & Bridge | Setting up Python MT5, environment flags, and background daemons. |
| **[05. Prop Firm Passkeeper Guide](05-Prop-Firm-Evaluation-Protocol)** | Funded Challenges | Geometric drawdown math, 3-stage circuit breakers for FTMO & GoatFunded. |
| **[06. Local Dashboard & Audits](06-Local-Dashboard-&-Monitoring)** | Telemetry & UI | Monitoring live tickets, WebSockets, COT sentiment, and memory audits. |

---

## ⚙️ Key Quantitative Specifications

- **Primary Instrument:** `NAS100.x` (Nasdaq 100 Index CFD)
- **Secondary Baskets:** `US30.x` (Dow Jones), `GER40.x` (DAX), `XAUUSD` (Gold), `EURUSD`
- **Base Execution Timeframe:** M5 (5-minute candle close synchronization)
- **Macro Trend Horizon:** H1 (50 Exponential Moving Average trend gate)
- **Capital Risk Budget:** Strictly $\le 0.50\%$ of account equity per trade
- **Index Volatility Floor:** 60.0 points minimum SL on Nasdaq / US30
- **Contract Multiplier Awareness:** 10.0 for NAS100 index contracts
- **Trailing Mechanics:** +1.0R Breakeven Lock with +10-point spread buffer
- **Circuit Breakers:** Stage 1 ($\ge 1.5\%$ DD), Stage 2 ($\ge 2.5\%$ DD), Stage 3 Freeze ($\ge 3.8\%$ DD)

---

## 🚀 Quick Local Terminal Launch

To launch the live trading engine locally on your MetaTrader 5 terminal:

```powershell
# 1. Clone repository and activate environment
git clone https://github.com/ElMoorish/FinRL-X-MT5.git
cd FinRL-X-MT5
.\.venv\Scripts\Activate.ps1

# 2. Configure credentials in .env
cp .env.example .env

# 3. Launch live engine with hard 0.04 lot cap and 0.50% risk ceiling
.\run_live_trader.ps1 -Symbols "NAS100.x" -RiskPct 0.0050

# 4. Launch institutional monitoring dashboard (http://127.0.0.1:8000)
python -m src.dashboard.app --port 8000
```
