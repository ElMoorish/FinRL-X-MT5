"""
Data Fusion — MT5 Ticks ↔ Yahoo/FMP EOD Correlations
=======================================================
This is where the EDGE is created.

MT5 tick data (intraday M5) is merged with Yahoo Finance EOD correlation
data to create a unified feature matrix that contains BOTH:
  - High-frequency microstructure signals (tick level)
  - Fundamental / cross-asset correlation signals (daily)

The forward-fill of daily signals into M5 bars gives the DRL agent
awareness of macro context at every bar — something purely technical
systems cannot achieve.

Edge mechanism:
  QQQ daily return → leads USTEC intraday by ~1 session
  XLE momentum    → confirms USOIL direction before MT5 ticks reflect it
  VIX spike       → risk-off signal before NAS100 drops
  DXY strength    → inverse XAUUSD signal
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import polars as pl
from loguru import logger

from src.config.settings import settings
from src.data.mt5_tick_fetcher import MT5TickFetcher
from src.data.yahoo_fetcher import YahooFetcher
from src.data.tick_feature_engineer import TickFeatureEngineer
from src.data.data_store import DataStore


class CorrelationFuser:
    """
    Fuses MT5 tick features with Yahoo/FMP daily correlation data.

    Output: Unified feature DataFrame at M5 frequency with both
    microstructure and fundamental signals available at each bar.
    """

    def __init__(
        self,
        tick_fetcher: Optional[MT5TickFetcher] = None,
        yahoo_fetcher: Optional[YahooFetcher] = None,
    ):
        self.tick_fetcher  = tick_fetcher or MT5TickFetcher(auto_connect=False)
        self.yahoo_fetcher = yahoo_fetcher or YahooFetcher()
        self.engineer      = TickFeatureEngineer(settings.mt5.timeframe_minutes)

    # ─── Main Fusion Pipeline ─────────────────────────────────────────────────

    def build_feature_store(
        self,
        symbol: str,
        date_from: datetime,
        date_to: datetime,
    ) -> pl.DataFrame:
        """
        Build the complete fused feature matrix for a single instrument.

        Pipeline:
          1. Pull MT5 real ticks for the date range
          2. Engineer M5 tick features (microstructure + technical)
          3. Pull Yahoo EOD correlation data
          4. Forward-fill daily signals into M5 bars
          5. Merge → unified feature matrix

        Args:
            symbol:    MT5 symbol (e.g. 'USTEC', 'USOIL')
            date_from: Start date
            date_to:   End date

        Returns:
            Fused feature DataFrame ready for Council experts.
        """
        logger.info(
            f"🔀 Building fused features for {symbol} | "
            f"{date_from.date()} → {date_to.date()}"
        )

        # ── Cache check (GAP-X1 fix) ──────────────────────────────────────────
        # Check DataStore before issuing any MT5 / Yahoo API calls.
        # Cache key is (symbol, date_from, date_to) — same as DataStore schema.
        store  = DataStore()
        cached = store.load_tick_features(symbol, date_from, date_to)
        if cached is not None and not cached.is_empty():
            logger.info(
                f"  📂 Cache hit: {len(cached):,} fused bars for {symbol} "
                f"— skipping MT5/Yahoo fetch"
            )
            return cached

        # ── Step 1: MT5 Features (Ticks or OHLCV) ─────────────────────────────
        ticks = self.tick_fetcher.get_ticks_range(symbol, date_from, date_to)
        if not ticks.is_empty():
            tick_features = self.engineer.ticks_to_features(ticks)
            logger.info(f"  ✅ Tick features: {len(tick_features)} M5 bars")
        else:
            logger.info(f"  Raw ticks not cached for {symbol}. Fetching M5 bars directly...")
            days_diff = max(1, (date_to - date_from).days)
            bars = self.tick_fetcher.get_ohlcv(symbol, timeframe="M5", n_bars=days_diff * 288)
            if bars.is_empty():
                logger.error(f"No MT5 data retrieved for {symbol} — cannot build feature store")
                return pl.DataFrame()
            tick_features = self.engineer.compute_features(bars)
            logger.info(f"  ✅ Features computed from M5 bars: {len(tick_features)} bars")

        # ── Step 2: Yahoo Correlation Features ────────────────────────────────
        # Pull extra days ahead to avoid lookahead bias after daily join
        yahoo_start = date_from - timedelta(days=30)
        corr_features = self.yahoo_fetcher.get_correlation_features(
            symbol, start=yahoo_start, end=date_to
        )

        if corr_features.is_empty():
            logger.warning(f"No Yahoo correlation data for {symbol} — using tick features only")
            return tick_features

        logger.info(f"  ✅ Yahoo corr features: {len(corr_features)} days")

        # ── Step 3: Macro Context ─────────────────────────────────────────────
        macro = self.yahoo_fetcher.get_macro_context(start=yahoo_start, end=date_to)
        if not macro.is_empty():
            macro_pivot = self._pivot_macro(macro)
            corr_features = corr_features.join(macro_pivot, on="date", how="left")

        # ── Step 4: Forward-Fill Daily → M5 ──────────────────────────────────
        fused = self._forward_fill_daily_to_m5(tick_features, corr_features)

        # ── Step 5: Add Cross-Asset Correlation Metrics ───────────────────────
        fused = self._add_correlation_metrics(fused, symbol)

        logger.info(
            f"  🎯 Fused feature matrix: {len(fused)} bars × "
            f"{len(fused.columns)} features"
        )

        # ── Persist to cache (GAP-X1 fix) ────────────────────────────────────
        try:
            store.save_tick_features(symbol, fused, date_from, date_to)
        except Exception as e:
            logger.warning(f"DataStore cache write failed ({e}) — continuing without cache")

        return fused

    def build_multi_symbol_store(
        self,
        symbols: Optional[list[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Build fused feature stores for all configured instruments.

        Returns a dict: {symbol → feature_DataFrame}
        """
        symbols   = symbols   or settings.mt5.symbols
        date_from = date_from or datetime.now() - timedelta(days=settings.data.tick_lookback_days)
        date_to   = date_to   or datetime.now()

        stores = {}
        for symbol in symbols:
            logger.info(f"━━━ Processing {symbol} ━━━")
            try:
                df = self.build_feature_store(symbol, date_from, date_to)
                if not df.is_empty():
                    stores[symbol] = df
            except Exception as e:
                logger.error(f"Failed to build features for {symbol}: {e}")

        logger.info(
            f"✅ Feature stores built for: {list(stores.keys())} | "
            f"Sizes: { {k: len(v) for k,v in stores.items()} }"
        )
        return stores

    def fuse(
        self,
        bar_features: pl.DataFrame,
        corr_features: pl.DataFrame,
        symbol: str = "NAS100.x",
    ) -> pl.DataFrame:
        """
        Fuse existing bar features directly with correlation features.
        """
        if corr_features.is_empty():
            return bar_features
        fused = self._forward_fill_daily_to_m5(bar_features, corr_features)
        return self._add_correlation_metrics(fused, symbol)

    # ─── Internal Helpers ─────────────────────────────────────────────────────

    def _forward_fill_daily_to_m5(
        self,
        m5_df: pl.DataFrame,
        daily_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Forward-fill daily signals into M5 bar timestamps.

        Method:
          1. Add 'date' column to M5 dataframe
          2. Left-join on date
          3. Forward-fill NaN values within each date group

        This ensures no lookahead bias — each M5 bar only sees
        the PREVIOUS day's EOD signals.
        """
        # Add date column to M5 features (truncate to day)
        m5_with_date = m5_df.with_columns(
            pl.col("time").dt.date().alias("date")
        )

        # Shift daily signals by 1 day (previous day's data at current bar)
        # This is critical — prevents lookahead bias
        daily_shifted = daily_df.with_columns(
            (pl.col("date").cast(pl.Date) + pl.duration(days=1)).cast(pl.Date).alias("date")
        )

        # Join on date
        fused = m5_with_date.join(daily_shifted, on="date", how="left")

        # Forward-fill any remaining NaNs (weekends, holidays)
        daily_cols = [c for c in daily_df.columns if c != "date"]
        fused = fused.with_columns([
            pl.col(c).forward_fill() for c in daily_cols if c in fused.columns
        ])

        return fused.drop("date")

    def _pivot_macro(self, macro: pl.DataFrame) -> pl.DataFrame:
        """Pivot macro DataFrame (VIX, DXY, TNX) to wide format per date."""
        frames = []
        for ticker in macro["ticker"].unique().to_list():
            # Clean ticker name for column prefix
            clean = ticker.replace("^", "").replace("=", "").replace("-", "_").lower()
            t_df = (
                macro.filter(pl.col("ticker") == ticker)
                .select([
                    pl.col("date").dt.date().alias("date"),
                    pl.col("return_1d").alias(f"{clean}_ret1d"),
                    pl.col("close").alias(f"{clean}_close"),
                    pl.col("vol_20d").alias(f"{clean}_vol20d"),
                ])
            )
            frames.append(t_df)

        if not frames:
            return pl.DataFrame()

        merged = frames[0]
        for f in frames[1:]:
            merged = merged.join(f, on="date", how="full", coalesce=True).sort("date")
        return merged

    def _add_correlation_metrics(
        self, df: pl.DataFrame, symbol: str
    ) -> pl.DataFrame:
        """
        Compute rolling cross-asset correlation metrics.

        For each instrument:
          - Rolling 20-bar correlation: close_return vs basket_return
          - Relative strength vs basket (alpha signal)
        """
        basket_col = f"{symbol}_basket_return"
        if basket_col not in df.columns:
            return df

        return df.with_columns([
            # Relative performance: MT5 return vs equity basket
            (pl.col("return_pct") / 100 - pl.col(basket_col))
              .alias(f"{symbol}_relative_strength"),

            # Rolling correlation proxy (simplified — full Pearson needs custom expr)
            pl.col(basket_col).rolling_mean(20).alias(f"{symbol}_basket_ma20"),
        ])
