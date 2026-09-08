"""
Yahoo / FMP Correlation Data Fetcher
======================================
Pulls US equity EOD data (QQQ, XLE, GLD, DIA and constituents) from
Yahoo Finance and optionally FMP. This data forms the FUNDAMENTAL EDGE
layer that is correlated with MT5 tick data by the Council's Expert 4 (SHAP).

Edge examples:
  - QQQ constituent fund flows → NAS100/USTEC positioning edge
  - XLE/CL=F sector momentum   → USOIL directional bias
  - GLD/DXY inverse            → XAUUSD safe-haven regime signal
  - DIA breadth signals        → US30 macro confirmation

Usage:
    fetcher = YahooFetcher()
    df = fetcher.get_correlation_features("USTEC", days=365)
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
from loguru import logger

from src.config.settings import settings

warnings.filterwarnings("ignore", category=FutureWarning)


class YahooFetcher:
    """
    Fetches US equity EOD data from Yahoo Finance for cross-asset correlation.

    Each MT5 instrument is mapped to a basket of Yahoo tickers:
        USTEC  → QQQ + Mag7 (AAPL, MSFT, NVDA, AMZN, META, GOOGL)
        USOIL  → XLE + XOM, CVX, COP, CL=F
        XAUUSD → GLD + GDX, GDXJ
        US30   → DIA + BA, GS, JPM, CAT
    """

    def __init__(self):
        self._cache: dict[str, pl.DataFrame] = {}
        cfg = settings.data
        self.yahoo_symbols = cfg.yahoo_symbols
        self.interval       = cfg.yahoo_interval
        self.lookback_days  = cfg.yahoo_lookback_days

    # ─── Core Fetch ──────────────────────────────────────────────────────────

    def fetch_tickers(
        self,
        tickers: list[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: str = "1d",
    ) -> pl.DataFrame:
        """
        Download OHLCV for a list of Yahoo tickers.

        Returns a Polars DataFrame with columns:
            [date, ticker, open, high, low, close, volume, return_1d, return_5d]
        """
        if start is None:
            start = datetime.now() - timedelta(days=self.lookback_days)
        if end is None:
            end = datetime.now()

        logger.info(f"⬇️  Fetching Yahoo: {tickers} | {start.date()} → {end.date()}")

        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        frames = []
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df_t = raw.copy()
                else:
                    df_t = raw[ticker].copy() if ticker in raw.columns.get_level_values(0) else pd.DataFrame()

                if df_t.empty:
                    logger.warning(f"No data for {ticker}")
                    continue

                df_t = df_t.dropna(subset=["Close"])
                df_t["ticker"]     = ticker
                df_t["return_1d"]  = df_t["Close"].pct_change(1)
                df_t["return_5d"]  = df_t["Close"].pct_change(5)
                df_t["return_20d"] = df_t["Close"].pct_change(20)
                df_t["vol_20d"]    = df_t["return_1d"].rolling(20).std() * np.sqrt(252)
                df_t["rsi_14"]     = self._compute_rsi(df_t["Close"], 14)
                df_t.index.name    = "date"
                df_t = df_t.reset_index()
                frames.append(df_t)

            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                continue

        if not frames:
            return pl.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined.columns = [c.lower().replace(" ", "_") for c in combined.columns]
        combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)

        return pl.from_pandas(combined[
            ["date", "ticker", "open", "high", "low", "close", "volume",
             "return_1d", "return_5d", "return_20d", "vol_20d", "rsi_14"]
        ].dropna(subset=["close"]))

    # ─── Correlation Features ─────────────────────────────────────────────────

    def get_correlation_features(
        self,
        mt5_symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Build a feature-rich correlation DataFrame for a given MT5 instrument.

        Features returned per date:
          - Correlation basket returns (QQQ, XLE, etc.)
          - Rolling cross-correlations
          - Sector momentum Z-scores
          - Lead-lag signals (equity market → CFD price tomorrow)

        These features are merged with MT5 tick features by correlation_fuser.py.
        """
        if days is not None and start is None:
            end = end or datetime.now()
            start = end - timedelta(days=days)

        cache_key = f"{mt5_symbol}_{start}_{end}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        tickers = self.yahoo_symbols.get(mt5_symbol, [])
        if not tickers:
            base_ticker = mt5_symbol.replace(".x", "").replace(".cash", "").upper()
            tickers = [base_ticker, "SPY", "QQQ"]
            logger.info(f"Auto-mapped equity {mt5_symbol} to Yahoo basket: {tickers}")

        raw = self.fetch_tickers(tickers, start=start, end=end)
        if raw.is_empty():
            return pl.DataFrame()

        # Pivot: one row per date, one column per ticker feature
        # Returns: ticker_return_1d, ticker_vol_20d, ticker_rsi_14
        pivoted = self._pivot_features(raw, mt5_symbol)
        self._cache[cache_key] = pivoted
        return pivoted

    def _pivot_features(self, df: pl.DataFrame, mt5_symbol: str) -> pl.DataFrame:
        """Pivot long-format ticker data to wide-format date-indexed features."""
        frames = []
        for ticker in df["ticker"].unique().to_list():
            t_df = (
                df.filter(pl.col("ticker") == ticker)
                .select(["date",
                         pl.col("return_1d").alias(f"{ticker}_ret1d"),
                         pl.col("return_5d").alias(f"{ticker}_ret5d"),
                         pl.col("vol_20d").alias(f"{ticker}_vol20d"),
                         pl.col("rsi_14").alias(f"{ticker}_rsi14"),
                         pl.col("close").alias(f"{ticker}_close"),
                ])
                .sort("date")
            )
            frames.append(t_df)

        if not frames:
            return pl.DataFrame()

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.join(frame, on="date", how="full", coalesce=True).sort("date")

        # Add composite signals
        ret_cols = [c for c in merged.columns if c.endswith("_ret1d")]
        if ret_cols:
            merged = merged.with_columns(
                pl.concat_list([pl.col(c) for c in ret_cols])
                  .list.mean()
                  .alias(f"{mt5_symbol}_basket_return"),
                pl.concat_list([pl.col(c) for c in ret_cols])
                  .list.std()
                  .alias(f"{mt5_symbol}_basket_dispersion"),
            )

        return merged.with_columns(pl.col("date").dt.date().alias("date"))

    # ─── DXY / VIX Context ────────────────────────────────────────────────────

    def get_macro_context(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """
        Fetch macro regime indicators:
          DX-Y.NYB → US Dollar Index (inverse gold signal)
          ^VIX     → Volatility index (risk-off regime signal)
          ^TNX     → 10Y Treasury yield (rate regime)
          GC=F     → Gold futures (confirmation for XAUUSD)
        """
        macro_tickers = ["DX-Y.NYB", "^VIX", "^TNX", "GC=F", "^GSPC"]
        return self.fetch_tickers(macro_tickers, start=start, end=end)

    # ─── Helper: RSI ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
