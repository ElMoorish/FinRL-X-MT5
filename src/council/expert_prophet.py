"""
Expert 3 — The Prophet: TimesFM 2.5 Volatility & Price Band Forecaster
=======================================================================
Uses Google's TimesFM 2.5 (200M parameter foundation model) to generate
zero-shot 1-hour ahead price forecasts with calibrated prediction intervals.

These quantile bands serve two purposes:
  1. Dynamic SL/TP levels → passed to Expert 5 (PyMC Bayesian) as priors
  2. Breakout probability → contributed as a trading signal to Council gate

No model training required — zero-shot inference directly from tick data.
TimesFM 2.5 supports 16,384 context points, far exceeding our 512-bar window.

⚠️ Run .agents/skills/timesfm-forecasting/scripts/check_system.py first!
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger

from src.config.settings import settings

warnings.filterwarnings("ignore")


class ExpertProphet:
    """
    Expert 3: TimesFM 2.5 zero-shot volatility band forecaster.

    Input:  M5 close price series (last N bars)
    Output: 12-bar ahead (60min) price forecast with 10th–90th percentile bands

    The quantile bands translate directly to:
      - upper_90 → Take-Profit target
      - lower_90 → Stop-Loss bound
      - Forecast confidence → position size scaling
    """

    def __init__(self):
        self._model   = None
        self._loaded  = False
        cfg           = settings.council
        self.horizon  = cfg.timesfm_horizon_bars     # 12 × M5 = 60 minutes
        self.context  = cfg.timesfm_context_bars     # 512 M5 bars ≈ 42 hours
        self.checkpoint = cfg.timesfm_checkpoint

    # ─── Lazy Model Loading ───────────────────────────────────────────────────

    def _ensure_loaded(self) -> bool:
        """Load TimesFM model on first use. Returns True if available, False otherwise."""
        if self._loaded:
            return self._model is not None

        try:
            import torch
            import timesfm

            logger.info("⏳ Loading TimesFM 2.5 (200M params)...")
            torch.set_float32_matmul_precision("high")

            self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.checkpoint
            )
            self._model.compile(timesfm.ForecastConfig(
                max_context=self.context,
                max_horizon=self.horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
                force_flip_invariance=True,
                infer_is_positive=False,
            ))
            self._loaded = True
            logger.info("✅ TimesFM 2.5 loaded and compiled")
            return True

        except Exception as e:
            logger.warning(
                f"TimesFM 2.5 not installed or loading failed ({e}) — using analytical EWMA volatility band forecaster."
            )
            self._loaded = True
            self._model = None
            return False

    # ─── Forecasting ─────────────────────────────────────────────────────────

    def forecast_price_bands(
        self, features: pl.DataFrame, symbol: str = ""
    ) -> dict:
        """
        Generate 60-minute ahead price bands for the given M5 feature series.

        Args:
            features: M5 feature DataFrame with at least 'close' column
            symbol:   Symbol name for logging

        Returns:
            dict with quantile bounds and confidence
        """
        is_timesfm = self._ensure_loaded()

        close = features["close"].tail(self.context).to_numpy().astype(np.float32)

        if len(close) < 20:
            logger.warning(f"Too few bars for forecasting ({len(close)} < 20)")
            return self._empty_forecast()

        if is_timesfm and self._model is not None:
            point, quantiles = self._model.forecast(
                horizon=self.horizon,
                inputs=[close],
            )
            result = {
                "point":    point[0],
                "lower_90": quantiles[0, :, 1],
                "upper_90": quantiles[0, :, 9],
                "lower_80": quantiles[0, :, 2],
                "upper_80": quantiles[0, :, 8],
                "lower_50": quantiles[0, :, 4],
                "upper_50": quantiles[0, :, 6],
            }
        else:
            # High-performance analytical EWMA Volatility Quantile Projection
            current_p = float(close[-1])
            returns = np.diff(np.log(close[-60:])) if len(close) >= 60 else np.diff(np.log(close))
            mu = float(np.mean(returns))
            sigma = float(np.std(returns) + 1e-6)

            steps = np.arange(1, self.horizon + 1)
            drift = np.exp(mu * steps)
            point = current_p * drift
            vol_steps = sigma * np.sqrt(steps) * current_p

            result = {
                "point":    point,
                "lower_90": point - (1.282 * vol_steps),
                "upper_90": point + (1.282 * vol_steps),
                "lower_80": point - (0.842 * vol_steps),
                "upper_80": point + (0.842 * vol_steps),
                "lower_50": point - (0.253 * vol_steps),
                "upper_50": point + (0.253 * vol_steps),
            }

        # Forecast confidence = tightness of prediction interval relative to ATR
        current_price = float(close[-1])
        band_width    = float(result["upper_90"][-1] - result["lower_90"][-1])
        atr_proxy     = float(np.std(np.diff(close[-20:])) * np.sqrt(self.horizon))

        result["band_width_pct"] = band_width / (current_price + 1e-8) * 100
        result["confidence"]     = 1.0 / (1.0 + band_width / (atr_proxy + 1e-8))

        # Directional bias from point forecast
        forecast_return = (result["point"][-1] - current_price) / (current_price + 1e-8)
        result["forecast_return"] = float(forecast_return)
        result["forecast_signal"] = float(np.tanh(forecast_return * 100))  # ∈ (-1, 1)

        logger.debug(
            f"🔭 {symbol} Prophet | signal={result['forecast_signal']:.3f} | "
            f"conf={result['confidence']:.3f} | "
            f"band={result['band_width_pct']:.2f}%"
        )
        return result

    def forecast_volatility_only(
        self, close_prices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Lightweight call — returns only point forecast and spread.
        Used internally by Expert 5 (PyMC) for prior construction.
        """
        is_timesfm = self._ensure_loaded()
        if is_timesfm and self._model is not None:
            point, quantiles = self._model.forecast(
                horizon=self.horizon, inputs=[close_prices.astype(np.float32)]
            )
            return point[0], quantiles[0]
        else:
            sigma = float(np.std(np.diff(close_prices[-40:])) if len(close_prices) > 40 else 1.0)
            point = np.full(self.horizon, close_prices[-1])
            spread = np.sqrt(np.arange(1, self.horizon + 1)) * sigma * 2.56
            return point, spread

    # ─── Council Signal ───────────────────────────────────────────────────────

    def get_council_signal(self, features: pl.DataFrame, symbol: str = "") -> dict:
        """
        Return Expert 3 formatted signal for the Council gate.

        Signal: tanh-scaled forecast return (directional bias)
        Confidence: prediction interval tightness
        Metadata: raw quantile bands for Expert 5 (PyMC prior)
        """
        forecast = self.forecast_price_bands(features, symbol)

        return {
            "expert":     "prophet",
            "signal":     forecast.get("forecast_signal", 0.0),
            "confidence": forecast.get("confidence", 0.5),
            "metadata": {
                "forecast_return": forecast.get("forecast_return", 0.0),
                "band_width_pct":  forecast.get("band_width_pct", 0.0),
                "point_forecast":  forecast.get("point", np.zeros(self.horizon)).tolist(),
                "upper_90":        forecast.get("upper_90", np.zeros(self.horizon)).tolist(),
                "lower_90":        forecast.get("lower_90", np.zeros(self.horizon)).tolist(),
            },
        }

    # ─── Anomaly / Regime Change Detection ───────────────────────────────────

    def detect_regime_change(
        self, features: pl.DataFrame, lookback: int = 48
    ) -> bool:
        """
        Detect potential regime change: current price outside 90% CI of forecast.
        Returns True if anomaly detected (current price statistically rare).

        Note: This method is available for future integration into the Council's
        live loop as an emergency circuit-breaker. Currently not called by default.
        Requires TimesFM to be installed — returns False gracefully if unavailable.
        """
        self._ensure_loaded()

        # Guard: analytical fallback does not support this check
        if self._model is None:
            logger.debug("detect_regime_change: TimesFM unavailable — returning False")
            return False

        close = features["close"].to_numpy().astype(np.float32)

        # Forecast from `lookback` bars ago
        if len(close) < lookback + self.horizon:
            return False

        context_end = len(close) - self.horizon
        context     = close[context_end - lookback: context_end]
        actual      = close[context_end:]

        _, quantiles = self._model.forecast(horizon=self.horizon, inputs=[context])
        lower_90     = quantiles[0, :len(actual), 1]
        upper_90     = quantiles[0, :len(actual), 9]

        outside_ci   = np.any((actual < lower_90) | (actual > upper_90))
        if outside_ci:
            logger.warning(f"⚠️ Regime change detected — price outside 90% CI")
        return bool(outside_ci)

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _empty_forecast(self) -> dict:
        h = self.horizon
        return {
            "point":           np.zeros(h),
            "lower_90":        np.zeros(h),
            "upper_90":        np.zeros(h),
            "lower_80":        np.zeros(h),
            "upper_80":        np.zeros(h),
            "band_width_pct":  0.0,
            "confidence":      0.0,
            "forecast_return": 0.0,
            "forecast_signal": 0.0,
        }
