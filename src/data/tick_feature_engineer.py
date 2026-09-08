"""
Tick Feature Engineer
======================
Converts raw MT5 tick data into structured M5 OHLCV bars enriched with
microstructure features. Uses Polars for high-performance processing.

Feature categories produced:
  1. OHLCV aggregates (open, high, low, close, volume)
  2. Microstructure (spread, tick count, tick imbalance, VWAP)
  3. Momentum (EMA, RSI, MACD, ROC)
  4. Volatility (ATR, Bollinger Bands, realized vol)
  5. Volume (OBV, volume Z-score)
  6. Calendar (hour_of_day, day_of_week, is_london_session, is_ny_session)

These features form the state space for:
  - Expert 1 (SAC DRL) observation vector
  - Expert 3 (TimesFM) input series
  - Expert 4 (SHAP XGBoost) feature set
"""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

from src.config.settings import settings


class TickFeatureEngineer:
    """
    Transforms raw MT5 ticks → M5 OHLCV bars + rich feature matrix.

    All processing done in Polars for maximum throughput on large tick datasets.
    Typical performance: ~1M ticks → M5 features in < 2 seconds.
    """

    # MT5 session windows (UTC)
    LONDON_OPEN  = 7    # 07:00 UTC
    LONDON_CLOSE = 16   # 16:00 UTC
    NY_OPEN      = 13   # 13:00 UTC
    NY_CLOSE     = 21   # 21:00 UTC

    def __init__(self, timeframe_minutes: int = 5):
        self.tf_minutes = timeframe_minutes
        self.tf_str     = f"{timeframe_minutes}m"

    # ─── Main Pipeline ────────────────────────────────────────────────────────

    def ticks_to_features(self, ticks: pl.DataFrame) -> pl.DataFrame:
        """
        Full pipeline: raw ticks → feature matrix.

        Args:
            ticks: Raw tick DataFrame from MT5TickFetcher.get_ticks()
                   Required columns: [time, bid, ask, volume, spread_pts]

        Returns:
            Feature DataFrame indexed by [time] with all engineered features.
            One row per M5 bar. Ready for Council experts.
        """
        if ticks.is_empty():
            logger.warning("Empty tick DataFrame — returning empty features")
            return pl.DataFrame()

        logger.info(f"Engineering features from {len(ticks):,} ticks → {self.tf_str} bars")

        ohlcv    = self._aggregate_to_ohlcv(ticks)
        ohlcv    = self._add_microstructure(ohlcv, ticks)
        ohlcv    = self._add_momentum(ohlcv)
        ohlcv    = self._add_volatility(ohlcv)
        ohlcv    = self._add_volume_features(ohlcv)
        ohlcv    = self._add_calendar_features(ohlcv)
        features = ohlcv.drop_nulls()  # Drop leading NaNs from rolling windows

        logger.info(f"✅ Feature matrix: {len(features)} bars × {len(features.columns)} features")
        return features

    def compute_features(self, ohlcv: pl.DataFrame) -> pl.DataFrame:
        """
        Compute rich feature matrix directly from pre-aggregated OHLCV bars.
        Useful when bars are fetched via get_ohlcv() or during fast backtesting.
        """
        if ohlcv.is_empty():
            return pl.DataFrame()

        df = ohlcv.sort("time")
        if "volume" not in df.columns and "tick_volume" in df.columns:
            df = df.with_columns(pl.col("tick_volume").alias("volume"))

        cols_to_add = []
        if "price_change" not in df.columns:
            cols_to_add.append((pl.col("close") - pl.col("close").shift(1)).alias("price_change"))
        if "return_pct" not in df.columns:
            cols_to_add.append((pl.col("close").pct_change() * 100).alias("return_pct"))
        if "bar_range_pct" not in df.columns:
            cols_to_add.append(((pl.col("high") - pl.col("low")) / (pl.col("close") + 1e-8) * 100).alias("bar_range_pct"))

        if "tick_imbalance" not in df.columns:
            cols_to_add.append(pl.lit(0.0).alias("tick_imbalance"))
        if "tick_density" not in df.columns:
            cols_to_add.append(pl.lit(1.0).alias("tick_density"))
        if "close_vs_vwap" not in df.columns:
            cols_to_add.append(pl.lit(0.0).alias("close_vs_vwap"))
        if "tick_count" not in df.columns:
            cols_to_add.append(pl.lit(100).alias("tick_count"))
        if "avg_spread" not in df.columns:
            cols_to_add.append(pl.col("spread").alias("avg_spread") if "spread" in df.columns else pl.lit(10.0).alias("avg_spread"))
        if "max_spread" not in df.columns:
            cols_to_add.append(pl.col("spread").alias("max_spread") if "spread" in df.columns else pl.lit(15.0).alias("max_spread"))
        if "spread_std" not in df.columns:
            cols_to_add.append(pl.lit(0.0).alias("spread_std"))
        if "vwap" not in df.columns:
            cols_to_add.append(pl.col("close").alias("vwap"))

        if cols_to_add:
            df = df.with_columns(cols_to_add)

        df = self._add_momentum(df)
        df = self._add_volatility(df)
        df = self._add_volume_features(df)
        df = self._add_calendar_features(df)
        return df.drop_nulls()

    # ─── Step 1: OHLCV Aggregation ────────────────────────────────────────────

    def _aggregate_to_ohlcv(self, ticks: pl.DataFrame) -> pl.DataFrame:
        """Aggregate tick stream to M5 OHLCV bars using Polars group_by_dynamic."""
        mid = (pl.col("bid") + pl.col("ask")) / 2.0

        return (
            ticks
            .sort("time")
            .group_by_dynamic("time", every=self.tf_str, period=self.tf_str)
            .agg([
                mid.first().alias("open"),
                mid.max().alias("high"),
                mid.min().alias("low"),
                mid.last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("spread_pts").mean().alias("avg_spread"),
                pl.col("spread_pts").max().alias("max_spread"),
                pl.col("spread_pts").std().alias("spread_std"),
                pl.len().alias("tick_count"),
                pl.col("ask").last().alias("ask_close"),
                pl.col("bid").last().alias("bid_close"),
                # Tick imbalance: (buy ticks - sell ticks) / total ticks
                # MT5 flag bit 2 = trade (ask hit), bit 4 = buy
                pl.col("flags").filter(pl.col("flags").is_in([4, 6])).len()
                  .alias("buy_ticks"),
            ])
            .with_columns([
                (pl.col("close") - pl.col("close").shift(1))
                  .alias("price_change"),
                (pl.col("close").pct_change() * 100)
                  .alias("return_pct"),
                # Tick imbalance ratio ∈ [-1, 1]
                ((2 * pl.col("buy_ticks") - pl.col("tick_count")) /
                 (pl.col("tick_count") + 1e-8))
                  .alias("tick_imbalance"),
            ])
            .sort("time")
        )

    # ─── Step 2: Microstructure Features ─────────────────────────────────────

    def _add_microstructure(
        self, ohlcv: pl.DataFrame, ticks: pl.DataFrame
    ) -> pl.DataFrame:
        """Add VWAP and high-frequency microstructure metrics."""
        # VWAP per bar: sum(price × volume) / sum(volume)
        mid = (pl.col("bid") + pl.col("ask")) / 2.0
        vwap_df = (
            ticks
            .sort("time")
            .group_by_dynamic("time", every=self.tf_str)
            .agg(
                (mid * pl.col("volume")).sum().alias("_pv"),
                pl.col("volume").sum().alias("_v"),
            )
            .with_columns(
                (pl.col("_pv") / (pl.col("_v") + 1e-8)).alias("vwap")
            )
            .select(["time", "vwap"])
        )

        return (
            ohlcv.join(vwap_df, on="time", how="left")
            .with_columns([
                # Price vs VWAP
                (pl.col("close") - pl.col("vwap")).alias("close_vs_vwap"),
                # Bar range as % of close
                ((pl.col("high") - pl.col("low")) / (pl.col("close") + 1e-8) * 100)
                  .alias("bar_range_pct"),
                # Tick density (ticks per second)
                (pl.col("tick_count") / (self.tf_minutes * 60))
                  .alias("tick_density"),
            ])
        )

    # ─── Step 3: Momentum Features ────────────────────────────────────────────

    def _add_momentum(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add EMA, RSI, MACD, ROC momentum indicators."""
        return df.with_columns([
            # EMAs
            pl.col("close").ewm_mean(span=9,   adjust=False).alias("ema_9"),
            pl.col("close").ewm_mean(span=21,  adjust=False).alias("ema_21"),
            pl.col("close").ewm_mean(span=50,  adjust=False).alias("ema_50"),
            pl.col("close").ewm_mean(span=200, adjust=False).alias("ema_200"),

            # Rate of Change
            pl.col("close").pct_change(6).alias("roc_6"),    # 30min
            pl.col("close").pct_change(12).alias("roc_12"),   # 60min
            pl.col("close").pct_change(48).alias("roc_48"),   # 4h

            # MACD (12, 26, 9) on M5
            (pl.col("close").ewm_mean(span=12, adjust=False) -
             pl.col("close").ewm_mean(span=26, adjust=False))
              .alias("macd_line"),
        ]).with_columns([
            # MACD signal
            pl.col("macd_line").ewm_mean(span=9, adjust=False).alias("macd_signal"),
            # EMA crossover signals
            (pl.col("ema_9") - pl.col("ema_21")).alias("ema_9_21_diff"),
            (pl.col("ema_21") - pl.col("ema_50")).alias("ema_21_50_diff"),
        ]).with_columns([
            (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_histogram"),
        ])

    # ─── Step 4: Volatility Features ─────────────────────────────────────────

    def _add_volatility(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add ATR, Bollinger Bands, realized volatility."""
        return df.with_columns([
            # True Range
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low")  - pl.col("close").shift(1)).abs(),
            ).alias("true_range"),

            # Bollinger Bands (20, 2σ)
            pl.col("close").rolling_mean(20).alias("bb_mid"),
            pl.col("close").rolling_std(20).alias("bb_std"),

            # Realized volatility (annualized)
            pl.col("return_pct").rolling_std(20)
              .alias("realized_vol_20"),
            pl.col("return_pct").rolling_std(60)
              .alias("realized_vol_60"),

        ]).with_columns([
            # ATR (14-period)
            pl.col("true_range").ewm_mean(span=14, adjust=False).alias("atr_14"),

            # BB bands
            (pl.col("bb_mid") + 2 * pl.col("bb_std")).alias("bb_upper"),
            (pl.col("bb_mid") - 2 * pl.col("bb_std")).alias("bb_lower"),
        ]).with_columns([
            # BB width & %B
            ((pl.col("bb_upper") - pl.col("bb_lower")) /
             (pl.col("bb_mid") + 1e-8)).alias("bb_width"),
            ((pl.col("close") - pl.col("bb_lower")) /
             (pl.col("bb_upper") - pl.col("bb_lower") + 1e-8)).alias("bb_pct_b"),

            # ATR as % of close (normalized across instruments)
            (pl.col("atr_14") / (pl.col("close") + 1e-8) * 100)
              .alias("atr_pct"),
        ])

    # ─── Step 5: Volume Features ──────────────────────────────────────────────

    def _add_volume_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add OBV, volume momentum, volume Z-score."""
        vol_f = pl.col("volume").cast(pl.Float64)
        return df.with_columns([
            # Volume Z-score (20-bar)
            ((vol_f - vol_f.rolling_mean(20)) /
             (vol_f.rolling_std(20) + 1e-8))
              .alias("volume_zscore"),

            # Volume ratio (current / 20-bar average)
            (vol_f / (vol_f.rolling_mean(20) + 1e-8))
              .alias("volume_ratio"),

            # OBV (On-Balance Volume)
            pl.when(pl.col("close") > pl.col("close").shift(1))
              .then(vol_f)
              .when(pl.col("close") < pl.col("close").shift(1))
              .then(-vol_f)
              .otherwise(0.0)
              .cum_sum()
              .alias("obv"),
        ])

    # ─── Step 6: Calendar / Session Features ─────────────────────────────────

    def _add_calendar_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add time-of-day and session overlap flags — critical for indices/commodities."""
        return df.with_columns([
            pl.col("time").dt.hour().alias("hour_utc"),
            pl.col("time").dt.weekday().alias("day_of_week"),  # 0=Mon, 6=Sun
            pl.col("time").dt.minute().alias("minute"),

            # Session flags (UTC)
            (pl.col("time").dt.hour().is_between(self.LONDON_OPEN, self.LONDON_CLOSE))
              .alias("is_london_session"),
            (pl.col("time").dt.hour().is_between(self.NY_OPEN, self.NY_CLOSE))
              .alias("is_ny_session"),
            # London-NY overlap (highest liquidity for NAS100/US30)
            (pl.col("time").dt.hour().is_between(self.NY_OPEN, self.LONDON_CLOSE))
              .alias("is_overlap_session"),

            # NAS100 pre-market (US 09:30 = 13:30 UTC)
            pl.col("time").dt.hour().is_between(13, 14).alias("is_us_open_hour"),
        ])

    # ─── Normalization ────────────────────────────────────────────────────────

    def normalize_for_rl(self, features: pl.DataFrame) -> np.ndarray:
        """
        Convert feature DataFrame to normalized numpy array for DRL state space.
        Returns shape: (n_bars, n_features) with Z-score normalization.
        """
        # Columns to use as RL state (exclude time and raw price levels)
        state_cols = [
            "return_pct", "tick_imbalance", "close_vs_vwap", "bar_range_pct",
            "tick_density", "roc_6", "roc_12", "roc_48",
            "macd_histogram", "ema_9_21_diff", "ema_21_50_diff",
            "atr_pct", "bb_pct_b", "bb_width", "realized_vol_20",
            "volume_zscore", "volume_ratio",
            "hour_utc", "day_of_week",
            "is_london_session", "is_ny_session", "is_overlap_session",
        ]

        cols_to_select = [
            pl.col(c) if c in features.columns else pl.lit(0.0).alias(c)
            for c in state_cols
        ]
        arr = features.select(cols_to_select).to_numpy().astype(np.float32)

        # Z-score normalization (per column)
        mean = arr.mean(axis=0, keepdims=True)
        std  = arr.std(axis=0, keepdims=True) + 1e-8
        return (arr - mean) / std
