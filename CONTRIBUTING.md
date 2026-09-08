# Contributing to FinRL-X-MT5

Thank you for your interest in contributing to **FinRL-X-MT5**!

This project is an institutional Mixture-of-Experts (MoE) algorithmic trading architecture designed for MetaTrader 5, focusing on Indices, Commodities, and Equities using 100% real-tick data.

---

## 🏛️ Development Guidelines

### 1. Adding a New Council Expert
All experts must adhere to the Council contract:
- Implement `fit(features)` for historical training.
- Implement `get_council_signal(features, symbol)` returning a standardized dictionary:
  ```python
  {
      "expert": "expert_name",
      "signal": float,       # Continuous signal in [-1.0, 1.0]
      "confidence": float,   # Normalized certainty in [0.0, 1.0]
      "metadata": dict,      # Domain-specific metrics (e.g. regime state, forecast bands)
  }
  ```
- Register the new expert's objective in `gating_optimizer.py` for NSGA-III Pareto weighting.

### 2. Adding New Instruments
When adding a new broker CFD or stock symbol:
1. Query the broker's specifications directly in MetaTrader 5:
   ```python
   info = mt5.symbol_info("SYMBOL.x")
   # Record: trade_contract_size, point, digits, volume_min, volume_step
   ```
2. Update `instrument_config` in `src/config/settings.py`.
3. Add the corresponding fundamental Yahoo Finance correlation basket in `yahoo_symbols`.

### 3. Data Integrity & Anti-Lookahead Standard
- **Never** use future data or unconfirmed bar closes in feature calculation.
- EOD equity baskets and macroeconomic indicators must be shifted by at least **1 calendar day** (`pl.duration(days=1)`) relative to intraday M5 bars.
- Vectorized operations should be implemented using **Polars** to preserve memory bandwidth and execution speed.

### 4. Code Style & Typing
- Code must be formatted using `black` and type-annotated with Python 3.10+ syntax (`|` union syntax, `from __future__ import annotations`).
- Log all operational events through `loguru.logger`.

---

## 🔒 Security & Privacy Policy
- **NEVER** commit `.env`, private account numbers, passwords, broker IPs, or SQLite databases.
- Verify `git status` before submitting Pull Requests to ensure no local model weights (`models/`) or CSV signal logs are staged.
