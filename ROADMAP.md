# FinRL-X-MT5 Engineering Roadmap

> This document tracks deliberate design deferrals — improvements that are architecturally
> correct but require model retraining or larger-scope changes before deployment.
> Last updated: 2026-09-11

---

## Phase 1 — Completed Fixes ✅

These were identified in the Architecture Audit (2026-09-11) and applied without retraining.

| Fix | File | Description |
|---|---|---|
| GAP-M3 | expert_trader.py | VecNormalize stats restored at DRL inference |
| GAP-X2 | expert_prophet.py | detect_regime_change() None-guard added |
| GAP-X3 | mt5_executor.py | Missing import math added |
| GAP-X1 | correlation_fuser.py | DataStore caching wired into feature pipeline |
| GAP-X7 | acktest_engine.py | Breakeven management added (mirrors live executor) |
| OBS-2  | isk_manager.py + cot_generator.py | H1 EMA50 extracted to shared get_h1_trend() |

---

## Phase 2 — Requires Model Retraining 🔄

### GAP-X6: Real Rolling Pearson Cross-Correlation
**Priority: HIGH**
**File:** src/data/correlation_fuser.py

**Current state (line 260-261):**
The _add_correlation_metrics() uses a simple rolling mean as a proxy for correlation.
The actual Pearson correlation between MT5 returns and Yahoo basket returns is not computed.

**Why deferred:**
Expert 4 (XGBoost) and Expert 1 (DRL) were trained without the real correlation feature.
Adding it changes the feature matrix — both models require full retraining and
out-of-sample walk-forward validation before deployment.

**Implementation approach:**
Rolling Pearson via z-score product (native Polars, no external deps):
`python
def _rolling_pearson(self, df, col_a, col_b, window=20):
    za = (pl.col(col_a) - pl.col(col_a).rolling_mean(window)) / (pl.col(col_a).rolling_std(window) + 1e-8)
    zb = (pl.col(col_b) - pl.col(col_b).rolling_mean(window)) / (pl.col(col_b).rolling_std(window) + 1e-8)
    return (za * zb).rolling_mean(window).alias(f'{col_a}_{col_b}_corr20')
`

**Acceptance criteria:**
- [ ] Implement _rolling_pearson() in CorrelationFuser
- [ ] Add {symbol}_basket_corr20 to fused feature matrix
- [ ] Retrain ExpertAnalyst (XGBoost) — confirm correlation feature in SHAP top-10
- [ ] Retrain ExpertTrader (SAC) — verify Sharpe improvement on OOS walk-forward
- [ ] Paper trade minimum 2 weeks before promoting to live

---

## Phase 3 — Future Enhancements 💡

### FE-01: Entropy-Based DRL Confidence (ExpertTrader)
Replace min(1.0, |weight| / 0.5) with SAC log-probability entropy measure.

### FE-02: Feature Schema Versioning in DataStore
Invalidate cache automatically when feature column set changes post-GAP-X6.

### FE-03: MT5 Connection Pool
Share a single MT5 connection across MT5TickFetcher and MT5Manager.

### FE-04: detect_regime_change() Live Integration
Wire ExpertProphet.detect_regime_change() into the cmd_live() bar loop
as an emergency circuit-breaker (after Phase 2 retraining).
