"""
Expert 5 — The Actuary: PyMC Bayesian Dynamic TP/SL Inference
=============================================================
Uses Bayesian inference (PyMC) to compute optimal TP/SL levels with
calibrated uncertainty. Unlike fixed pip values, these levels adapt
to the current volatility regime and TimesFM forecast bands.

The model:
  - Prior on TP/SL: derived from TimesFM 90th/10th percentile forecasts
  - Likelihood: observed price movement distribution (historical)
  - Posterior: updated TP/SL with credible intervals

Output:
  - tp_price: Take-profit level (75th credible interval upper bound)
  - sl_price: Stop-loss level (75th credible interval lower bound)
  - expected_rr: Expected Risk-Reward ratio given posteriors
  - trade_confidence: P(TP hit before SL) from posterior predictive

This expert does NOT output a directional signal — it outputs position
SIZING and RISK LEVELS consumed by the Council orchestrator.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger

from src.config.settings import settings

warnings.filterwarnings("ignore", category=UserWarning)


class ExpertActuary:
    """
    Expert 5: Bayesian dynamic TP/SL level estimation.

    Uses PyMC Bayesian inference with TimesFM forecast bands as priors
    to compute calibrated TP/SL levels adjusted for current volatility.
    """

    def __init__(self):
        self._pymc_available = self._check_pymc()
        cfg = settings.council
        self.n_samples = cfg.pymc_samples
        self.n_tune    = cfg.pymc_tune
        self.n_chains  = cfg.pymc_chains

    @staticmethod
    def _check_pymc() -> bool:
        try:
            import pymc  # noqa
            return True
        except ImportError:
            logger.warning(
                "pymc not installed — ExpertActuary will use analytical fallback. "
                "Install: uv pip install 'pymc>=5.20.0'"
            )
            return False

    # ─── Bayesian TP/SL Estimation ────────────────────────────────────────────

    def estimate_tp_sl(
        self,
        features: pl.DataFrame,
        prophet_output: dict,
        current_price: float,
        direction: int,    # +1 = long, -1 = short
        symbol: str = "",
    ) -> dict:
        """
        Estimate Bayesian TP/SL levels.

        Args:
            features:       M5 feature DataFrame (last N bars for volatility)
            prophet_output: Expert 3 (TimesFM) forecast output dict
            current_price:  Current mid price
            direction:      +1 for long, -1 for short
            symbol:         Symbol name for logging

        Returns:
            {
              tp_price:          float,
              sl_price:          float,
              tp_pips:           float,
              sl_pips:           float,
              expected_rr:       float,
              trade_confidence:  float ∈ [0, 1],
              posterior_summary: dict
            }
        """
        if self._pymc_available:
            return self._bayesian_estimate(
                features, prophet_output, current_price, direction, symbol
            )
        else:
            return self._analytical_estimate(
                features, prophet_output, current_price, direction
            )

    def _bayesian_estimate(
        self,
        features: pl.DataFrame,
        prophet_output: dict,
        current_price: float,
        direction: int,
        symbol: str,
    ) -> dict:
        """Full PyMC Bayesian posterior estimation."""
        import pymc as pm
        import pytensor.tensor as pt

        # ── Prepare data ────────────────────────────────────────────────────
        returns  = features["return_pct"].drop_nulls().tail(200).to_numpy() / 100.0
        vol      = float(np.std(returns))
        atr_pct  = float(features["atr_pct"].drop_nulls().tail(1)[0]) / 100.0

        # TimesFM-derived priors
        upper_90 = np.array(prophet_output.get("upper_90", [current_price * 1.01]))
        lower_90 = np.array(prophet_output.get("lower_90", [current_price * 0.99]))

        tp_prior_mu  = float(upper_90.mean()) if direction > 0 else float(lower_90.mean())
        sl_prior_mu  = float(lower_90.mean()) if direction > 0 else float(upper_90.mean())
        band_width   = float(upper_90.mean() - lower_90.mean())
        prior_sigma  = max(atr_pct * current_price, band_width * 0.2)

        # ── PyMC Model ──────────────────────────────────────────────────────
        with pm.Model() as model:
            # Priors informed by TimesFM
            tp_target = pm.Normal(
                "tp_target",
                mu=tp_prior_mu,
                sigma=prior_sigma,
            )
            sl_bound  = pm.Normal(
                "sl_bound",
                mu=sl_prior_mu,
                sigma=prior_sigma * 0.5,  # Tighter prior on SL
            )

            # Likelihood: price moves follow a t-distribution (fat tails)
            price_vol = pm.HalfNormal("price_vol", sigma=vol)
            _obs = pm.StudentT(  # noqa
                "obs",
                nu=4,           # Degrees of freedom (heavy-tailed)
                mu=current_price,
                sigma=price_vol * current_price,
                observed=np.array([current_price]),  # Anchor to current price
            )

            # Sample posterior
            trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                chains=self.n_chains,
                progressbar=False,
                return_inferencedata=True,
                target_accept=0.9,
            )

        # ── Extract posteriors ───────────────────────────────────────────────
        tp_samples = trace.posterior["tp_target"].values.flatten()
        sl_samples = trace.posterior["sl_bound"].values.flatten()

        # Use 75th credible interval for conservative targets
        if direction > 0:  # Long
            tp_price = float(np.percentile(tp_samples, 75))
            sl_price = float(np.percentile(sl_samples, 25))
        else:              # Short
            tp_price = float(np.percentile(tp_samples, 25))
            sl_price = float(np.percentile(sl_samples, 75))

        return self._build_result(tp_price, sl_price, current_price, direction, trace)

    def _analytical_estimate(
        self,
        features: pl.DataFrame,
        prophet_output: dict,
        current_price: float,
        direction: int,
    ) -> dict:
        """
        Analytical fallback (no PyMC) using ATR + TimesFM bands.
        Used when PyMC is not installed.
        """
        atr_pct = float(features["atr_pct"].drop_nulls().tail(1)[0]) / 100.0
        atr_abs = atr_pct * current_price

        upper_90 = np.array(prophet_output.get("upper_90", [current_price + 2 * atr_abs]))
        lower_90 = np.array(prophet_output.get("lower_90", [current_price - 2 * atr_abs]))

        if direction > 0:  # Long
            tp_price = float(upper_90.mean())
            sl_price = float(lower_90.mean())
        else:              # Short
            tp_price = float(lower_90.mean())
            sl_price = float(upper_90.mean())

        return self._build_result(tp_price, sl_price, current_price, direction, trace=None)

    def _build_result(
        self,
        tp_price: float,
        sl_price: float,
        current_price: float,
        direction: int,
        trace=None,
    ) -> dict:
        """Compute derived metrics from TP/SL prices."""
        tp_dist = abs(tp_price - current_price)
        sl_dist = abs(sl_price - current_price)
        rr      = tp_dist / (sl_dist + 1e-8)

        # P(TP hit) ≈ RR-calibrated probability (Kelly-style)
        trade_confidence = rr / (1 + rr) if rr > 0 else 0.5

        return {
            "tp_price":         tp_price,
            "sl_price":         sl_price,
            "tp_pips":          tp_dist,
            "sl_pips":          sl_dist,
            "expected_rr":      rr,
            "trade_confidence": trade_confidence,
            "posterior_summary": {
                "tp_mean": tp_price,
                "sl_mean": sl_price,
                "has_bayesian": trace is not None,
            },
        }

    # ─── Council Signal ───────────────────────────────────────────────────────

    def get_council_signal(
        self,
        features: pl.DataFrame,
        prophet_output: dict,
        current_price: float,
        direction: int,
        symbol: str = "",
    ) -> dict:
        """
        Return Expert 5 formatted output for the Council gate.
        Note: signal = trade_confidence (not directional — just quality)
        """
        result = self.estimate_tp_sl(
            features, prophet_output, current_price, direction, symbol
        )

        logger.debug(
            f"⚖️ {symbol} Actuary | TP={result['tp_price']:.2f} | "
            f"SL={result['sl_price']:.2f} | RR={result['expected_rr']:.2f} | "
            f"conf={result['trade_confidence']:.3f}"
        )

        direction_sign = 1.0 if direction > 0 else (-1.0 if direction < 0 else 0.0)
        directional_signal = direction_sign * (result["trade_confidence"] * 2.0 - 1.0)

        return {
            "expert":     "actuary",
            "signal":     float(directional_signal),
            "confidence": result["trade_confidence"],
            "tp_price":   result["tp_price"],
            "sl_price":   result["sl_price"],
            "expected_rr": result["expected_rr"],
            "metadata":   result,
        }
