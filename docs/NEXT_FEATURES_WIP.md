# FinRL-X Institutional Next Features & Work-in-Progress (WIP) Roadmap

**Document Version:** `2.4.0-WIP`  
**Classification:** Quantitative Architecture & Strategic Milestone Specification  
**Target Repository:** `github.com/ElMoorish/FinRL-X-MT5`

---

## Executive Overview

This document specifies the technical architecture, implementation schedules, and quantitative engineering workflows for upcoming releases of **FinRL-X**. The objective is to evolve the open-source MetaTrader 5 multi-agent framework into an institutional-grade, continuous-learning algorithmic fund infrastructure capable of autonomous capital management and accredited allocator integration.

---

## 1. Step-by-Step Technical Implementation Plan

### Feature 1: Automated Continual RL & Walk-Forward Engine (`src/council/continual_learner.py`)
To prevent policy obsolescence without catastrophic forgetting, the Council implements a two-stage rolling retraining lifecycle.

#### Step 1.1: Reward Function Shaping for Asymmetric Expectancy
* **File:** [`src/council/expert_trader.py`](file:///c:/Users/aitsi/Desktop/FinRL-X-MT5/src/council/expert_trader.py)
* **Mathematical Specification:**
  Replace simple Sharpe return scaling with a three-component institutional utility function:
  $$R_t = R_{\text{Sortino}} + R_{\text{Breakeven}} + R_{\text{Patience}}$$
  Where:
  - **Sortino Downside Penalty:** Downside volatility is penalized $3\times$ heavier than upside gains:
    $$R_{\text{Sortino}} = \frac{\mu_r}{\sigma_{\text{downside}} + \epsilon} \cdot \text{pnl}_t - 3.0 \cdot \min(0, \text{pnl}_t)^2$$
  - **Breakeven Milestone Reward ($+1.0R$):** If an open trade trajectory reaches $+1.0R$ (triggering breakeven arming), award $+0.25$ utility to reinforce swift trend alignment.
  - **Capital Preservation Reward:** When volatility is in pure chop ($P(\text{Consolidation}) > 0.70$), neutral actions ($|a_t| < 0.15$) receive a $+0.05$ preservation reward.

#### Step 1.2: Monthly Regime & Gate Recalibration Daemon
* **Trigger:** Last Sunday of each month at 22:00 UTC.
* **Actions:**
  1. Fit Gaussian Hidden Markov Model (E2) on the rolling 60-day feature set to capture updated regime transition matrix $A_{ij}$ and covariance matrices $\Sigma_k$.
  2. Optimize NSGA-III Pareto gating weights $\vec{\alpha} = [\alpha_1, \dots, \alpha_5]$ against the latest 30-day validation matrix.
  3. Save calibrated weights to `weights/council_gate_latest.json`.

#### Step 1.3: Quarterly SAC DRL Warm-Start Fine-Tuning
* **Trigger:** Quarterly rollover (End of Q1, Q2, Q3, Q4).
* **Actions:**
  1. Load existing champion weights from `weights/expert_trader_drl.zip`.
  2. Warm-start the Soft Actor-Critic network with a reduced learning rate ($\eta = 7 \times 10^{-5}$).
  3. Train for 50,000 timesteps across a rolling 12-month window (~15,000 M5 bars).
  4. Run the **Champion-Challenger Gate** using [`src/backtest/walk_forward.py`](file:///c:/Users/aitsi/Desktop/FinRL-X-MT5/src/backtest/walk_forward.py):
     - Require Out-of-Sample (OOS) Win Rate $\ge 60.0\%$.
     - Require OOS Profit Factor $\ge 1.75$.
     - Require OOS Sharpe Degradation ($Sharpe_{\text{OOS}} / Sharpe_{\text{IS}}$) $\ge 0.65$.
  5. Only promote challenger weights to live production if all acceptance criteria are met.

---

### Feature 2: Multi-Asset Council Expansion (`src/data/correlation_fuser.py`)
Expanding beyond `NAS100.x` to a balanced cross-asset macro basket.

1. **Phase 1 Assets:**
   - **`NAS100.x` / `USTEC`**: High-beta tech index (current production flagship).
   - **`US30.x` / `DJ30`**: Cyclical, value-weighted industrial index.
   - **`XAUUSD.x` / `GOLD`**: Sovereign hedge and real-yield macro diversifier.
   - **`EURUSD`**: High-liquidity FX benchmark for session-boundary order flow.
2. **Implementation Steps:**
   - Extend [`src/data/mt5_tick_fetcher.py`](file:///c:/Users/aitsi/Desktop/FinRL-X-MT5/src/data/mt5_tick_fetcher.py) to ingest multi-symbol tick books simultaneously.
   - Upgrade `CorrelationFuser` to compute rolling cross-asset correlation matrices $\rho_{ij}$ between Nasdaq, US Yields (`^TNX`), Gold, and Dollar Index (`DX-Y.NYB`).
   - Implement Portfolio-Level Heat Limiter: Cap total simultaneous correlated risk across all pairs at $\le 2.0\%$ aggregate equity.

---

### Feature 3: Institutional Health Watchdog & Cloud Enclave (`src/infra/`)
1. **Containerized Linux/Windows Headless Daemon:**
   - Headless Wine/MetaTrader 5 Docker container specification for zero-maintenance deployment on high-uptime cloud VPS (Equinix NY4 / LD4 cross-connect).
2. **Process Watchdog & Dead-Man's Switch (`src/infra/watchdog.py`):**
   - Monitors live loop heartbeats every 60 seconds.
   - If an M5 bar closes without a Council evaluation, triggers automated MT5 terminal reconnect and dispatches emergency alerts to Telegram/Discord.
   - Local sqlite state cache prevents state desynchronization in the event of VPS reboots.

---

### Feature 4: Audited Institutional Investor Reporting & Tear-Sheets
1. **Automated QuantStats Tear-Sheet Engine:**
   - Generates weekly and monthly institutional PDF/HTML tear-sheets containing:
     - Cumulative equity curves, rolling Sharpe ratio (6-month), Sortino ratio, Omega ratio.
     - Maximum drawdown underwater chart with duration and recovery metrics.
     - Trade expectancy distribution, win rate realization, and session-by-session alpha attribution.
2. **Third-Party Track Record Verification:**
   - Automated sync webhook to verified third-party audit portals (Myfxbook, FXBlue, and Darwinex Zero).
   - Read-only investor audit endpoint displaying verified broker-signed history deals.

---

## 2. Institutional Roadmap Horizon

| Phase | Milestone | Scope & Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **Phase I** | **Core Multi-Agent Architecture** | 5-Expert Council, SAC DRL, HMM Regimes, Risk-Reward Actuary, MT5 execution engine. | **COMPLETED** |
| **Phase II** | **Commercial Alpha Hub & Verification** | Telegram/Discord VIP bot, automated daily/weekly recaps, luxury commercial portal. | **COMPLETED** |
| **Phase III** | **Continual RL & Walk-Forward Gating** | Rolling HMM recalibration, warm-start quarterly SAC fine-tuning, automated OOS challenger gate. | **IN PROGRESS** |
| **Phase IV** | **Cross-Asset Expansion & Cloud Enclave** | Multi-symbol portfolio hedging (`US30`, `XAUUSD`), headless Docker containerization, 24/5 watchdog. | **UPCOMING (Q4)** |
| **Phase V** | **Institutional Managed Accounts & Fund** | MAM/PAMM allocator infrastructure, BVI/Cayman incubator structure, accredited LP capital raising. | **HORIZON** |

---

## 3. Investor Acquisition Strategy: *From Algo Framework to Hedge Fund*

To successfully attract private capital, High-Net-Worth Individuals (HNWIs), and Family Offices into a dedicated quantitative vehicle, institutional allocators require specific evidence, legal frameworks, and custody structures:

### A. The Three Pillars Institutional Allocators Require

1. **Audited, Unmanipulated Track Record (The "Skin in the Game" Rule):**
   - Allocators do not fund backtests or theoretical curves. They require **6 to 12 months of live broker-verified trading** on live capital.
   - The developers trading their own capital under identical sizing demonstrates sovereign alignment.
   - Third-party verified read-only links (e.g. Myfxbook Auto-Track, Darwinex, or interactive broker statement audits).

2. **The Exact Quantitative Thresholds Allocators Look For:**
   - **Sharpe Ratio:** $\ge 1.80$ (Annualized).
   - **Sortino Ratio:** $\ge 2.50$ (Zero penalty for upside volatility).
   - **Maximum Historical Drawdown:** $< 5.0\%$ (Strict solvency preservation).
   - **Monthly Return Target:** $+3.0\%$ to $+6.0\%$ steady net return (Institutions prefer consistent 40–60% annualized over volatile 200% swings).
   - **Absence of High-Risk Mechanics:** Zero Martingale, zero unbounded grid, guaranteed stop loss on every position, strictly limited overnight exposure.

3. **Custodial Separation & Capital Safety:**
   - Professional investors will **never** send funds directly to an unregulated third party.
   - Capital must reside in the investor's own segregated account with an institutional prime broker (e.g., Swissquote, Interactive Brokers, Saxo Bank, or regulated MT5 Prime of Primes).

---

### B. Scalable Legal Structuring Options

| Structure | Typical Jurisdiction | Minimum Capital | Target Investor Profile | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Managed Account / MAM / PAMM** | Regulated Broker (UK, Switzerland, Cyprus, Australia) | $25,000 – $250,000 | Private HNWIs, Early Angels, VIP Members | Algorithm connects via Multi-Account Manager (MAM). Investor maintains 100% custody; fund manager earns 20-30% performance fee via high-water mark. |
| **BVI / Cayman Incubator Fund** | British Virgin Islands / Cayman Islands | $500,000 – $5,000,000 | Accredited Investors, Angel Syndicates | Low-cost offshore vehicle ($15k–$25k setup). No full licensing required up to 20 accredited investors / $20M AUM. Fast-track 2-year runway. |
| **UAE / DIFC / ADGM Quant Enclave** | Dubai (DIFC) / Abu Dhabi (ADGM) | $1,000,000 – $20,000,000+ | Regional Family Offices, Tech Entrepreneurs | Emerging global hub for algorithmic trading. Favorable tax incentives, institutional digital asset/FX recognition. |

---

### C. The 4-Step Capital Acquisition Funnel

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: VIP Membership & Community Incubation             │
│  Offer VIP Alpha alerts and model licensing to early users. │
│  Build a loyal cohort that witnesses daily/weekly profits.   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Private Managed Accounts (MAM / PAMM Pilot)        │
│  Invite top-tier VIP community members to connect capital   │
│  via segregated broker accounts with institutional custody. │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: 12-Month Audited Track Record & Tear-Sheet Pack    │
│  Compile verified broker statements, QuantStats analytics,   │
│  and mathematical MoE papers into a clean Investor Deck.    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Incubator Fund Launch (Family Offices & LPs)       │
│  Launch BVI Incubator / Cayman SPC. Onboard external        │
│  accredited LP capital with 2/20 fee structure.              │
└─────────────────────────────────────────────────────────────┘
```
