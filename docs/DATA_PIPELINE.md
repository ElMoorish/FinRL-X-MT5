# 📊 FinRL-X-MT5: Data Pipeline & Feature Fusion

The primary competitive edge of **FinRL-X-MT5** lies in its **hybrid data fusion architecture**. Rather than relying solely on lagging technical indicators computed on pure price bars, the system fuses **high-frequency MT5 order-book microstructure** with **EOD cross-asset fundamental flows** from US equities and macro benchmarks.

---

## 🔄 Dual-Source Data Architecture

```
 MT5 Live Terminal / Broker Server                     Yahoo Finance / FMP API
┌───────────────────────────────────┐               ┌───────────────────────────────┐
│ • High-Frequency Real Ticks       │               │ • ETF Baskets (QQQ, XLE, DIA) │
│ • Bid / Ask / Last / Volume       │               │ • Mag-7 & Constituent Returns │
│ • Microsecond Timestamps          │               │ • Macro Yields (^TNX, DXY)    │
└─────────────────┬─────────────────┘               └───────────────┬───────────────┘
                  │                                                 │
                  ▼                                                 ▼
     ┌─────────────────────────┐                       ┌─────────────────────────┐
     │   MT5TickFetcher        │                       │   YahooFetcher          │
     │   (Polars Resampler)    │                       │   (Cross-Asset Pivot)   │
     └────────────┬────────────┘                       └────────────┬────────────┘
                  │                                                 │
                  ▼                                                 ▼
     ┌─────────────────────────┐                       ┌─────────────────────────┐
     │  TickFeatureEngineer    │                       │   1-Day Lag Alignment   │
     │  (53 Microstructure Feat│                       │   (Strict Anti-Lookahead│
     └────────────┬────────────┘                       └────────────┬────────────┘
                  │                                                 │
                  └────────────────────────┬────────────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │    CorrelationFuser     │
                              │ (Unified Feature Matrix)│
                              └─────────────────────────┘
```

---

## 1. High-Frequency Microstructure Engineering

Historical or live ticks are extracted via `MT5TickFetcher` and aggregated into M5 bars with continuous microstructural metrics computed via **Polars** for maximum vectorization:

### Microstructure Indicators
- **Spread Dynamics:**
  $$\text{Spread}_t = \frac{\text{Ask}_t - \text{Bid}_t}{\text{Point}} \quad (\text{Z-score normalized over 50 bars})$$
- **Tick / Order Flow Imbalance:**
  $$\text{Imbalance} = \frac{V_{\text{buy}} - V_{\text{sell}}}{V_{\text{buy}} + V_{\text{sell}} + \epsilon}$$
- **Parkinson High-Low Volatility:**
  $$\sigma_{\text{parkinson}} = \sqrt{\frac{1}{4 \ln 2} \sum_{i=1}^{n} \ln\left(\frac{H_i}{L_i}\right)^2}$$
- **Realized Microstructure Volatility:**
  $$\text{RV} = \sqrt{\sum_{k} r_k^2}$$
- **Volume Z-Score & OBV:** Measures institutional accumulation vs distribution.
- **Calendar & Session Seasonality:** Cyclical encoding ($\sin/\cos$) of Hour-of-Day, Day-of-Week, US Market Open ($09:30\text{ EST}$), and European Session Open.

---

## 2. Cross-Asset Correlation Baskets

Each MT5 CFD instrument is mapped to its underlying equity constituents and sector ETFs:

| MT5 Instrument | Correlation Basket | Economic Edge Provided |
|---|---|---|
| **`NAS100.x`** | `QQQ`, `AAPL`, `MSFT`, `NVDA`, `AMZN`, `META`, `GOOGL` | Institutional ETF inflows/outflows lead futures & CFD index momentum. |
| **`WTI.x`** | `CL=F` (NYMEX front month), `USO`, `XLE`, `XOM`, `CVX` | Energy equity performance and spot futures curves confirm physical crude demand. |
| **`XAGUSD.x`** | `SLV`, `SI=F` (Comex Silver), `GLD`, `PAAS` | Safe-haven monetary rotation vs industrial silver demand. |
| **`US30.x`** | `DIA`, `BA`, `GS`, `JPM`, `UNH`, `CAT` | Industrial & banking breadth indicators confirming Dow trends. |
| **`GER40.x`** | `EWG`, `SAP` | European equity capital flows vs transatlantic market shifts. |
| **`SPX500.x`** | `SPY`, `VOO`, `IVV`, `XLK`, `XLF` | Broad market liquidity and sector rotation balance. |
| **`JAP225.x`** | `EWJ`, `DXJ`, `TM`, `SONY` | Yen currency correlation and export equity strength. |
| **`UK100.x`** | `EWU`, `SHEL`, `AZN`, `HSBC` | UK commodity-heavy blue chip momentum. |
| **`AUS200.x`** | `EWA`, `BHP` | Asia-Pacific resource and mining demand lead indicators. |
| **Equities** (`AAPL.x`, `NVDA.x`, etc.) | Direct stock ticker, Sector ETF (`XLK`, `SOXX`), `SPY`, `QQQ` | Sub-sector momentum and beta sensitivity vs broad market indexes. |

In addition, global macro drivers are tracked across all instruments:
- `DX-Y.NYB` — US Dollar Index (DXY)
- `^VIX` — CBOE Market Volatility Index
- `^TNX` — US 10-Year Treasury Yield
- `GC=F` — Spot Gold Futures
- `^GSPC` — S&P 500 Index

---

## 3. Strict Anti-Lookahead Information Barrier

A common flaw in algorithmic trading backtests is leaking end-of-day information into intraday bars.

In `CorrelationFuser`:
1. All Yahoo Finance and macro data are stamped with their closing date $T$.
2. Before merging with intraday M5 bars on day $T$, the correlation data is **explicitly shifted forward by 1 calendar day**:
   $$\text{SignalDate} = T + 1\text{ day}$$
3. Intraday M5 bars on Monday morning trade **strictly on Friday's confirmed closing fundamental signals**.
4. Missing weekend or holiday values are forward-filled without peering into the future.

---

## 4. Feature Store & Cache Architecture

- **SQLite Database:** `data/cache/finrl_mt5.db` caches ingested tick and bar data to prevent duplicate broker queries.
- **Parquet Storage:** Exported feature datasets are saved as columnar Apache Parquet files for instant loading and training throughput.
