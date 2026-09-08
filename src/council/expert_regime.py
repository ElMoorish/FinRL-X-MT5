"""
Expert 2 — The Oracle: HMM Market Regime Filter
=================================================
Uses a 3-state Gaussian HMM to classify the current market regime.
This gating signal modulates ALL other expert outputs.

States:
  0 = Bear / Risk-Off    → scale down all positions by 75%
  1 = Sideways / Range   → mean-reversion bias, reduced size
  2 = Bull / Trending    → full position size, momentum bias

Inputs:
  - Returns (M5 log returns)
  - Volume Z-score
  - ATR (realized volatility proxy)
  - Tick imbalance (order flow)

This expert runs FIRST in the Council pipeline and its output
(regime_weight_modifier) scales every other expert's signal.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM
from loguru import logger

from src.config.settings import settings


class ExpertRegime:
    """
    Expert 2: Gaussian HMM Market Regime Filter.

    The regime state is the first filter applied by the Council.
    A Bear regime will reduce any long signal; a Bull regime amplifies it.
    """

    # Position size multipliers per regime
    REGIME_MULTIPLIERS = {
        0: 0.25,  # Bear: 25% of normal size
        1: 0.60,  # Sideways: 60% of normal size
        2: 1.00,  # Bull: 100% of normal size
    }

    REGIME_NAMES = {0: "BEAR", 1: "SIDEWAYS", 2: "BULL"}

    def __init__(self):
        cfg = settings.council
        self.hmm = GaussianHMM(
            n_components=cfg.hmm_n_states,
            covariance_type=cfg.hmm_covariance_type,
            n_iter=cfg.hmm_n_iter,
            min_covar=1e-3,
            random_state=42,
            verbose=False,
        )
        self._fitted       = False
        self._state_map: dict[int, int] = {}  # maps raw HMM state → ordered (0=Bear)
        self.model_path    = settings.council.models_dir / "expert_regime_hmm.pkl"

    # ─── Feature Extraction ───────────────────────────────────────────────────

    def _extract_hmm_features(self, features: pl.DataFrame) -> np.ndarray:
        """
        Extract HMM input features from the M5 feature DataFrame.
        Returns shape (n_bars, 4).
        """
        required = ["return_pct", "volume_zscore", "atr_pct", "tick_imbalance"]
        cols_to_select = [
            pl.col(c) if c in features.columns else pl.lit(0.0).alias(c)
            for c in required
        ]

        arr = features.select(cols_to_select).to_numpy().astype(np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize features (Z-score) to prevent numerical likelihood underflow
        for col_idx in range(arr.shape[1]):
            std = np.std(arr[:, col_idx])
            mean = np.mean(arr[:, col_idx])
            if std < 1e-6:
                arr[:, col_idx] = np.random.normal(0, 1e-4, size=arr.shape[0])
            else:
                arr[:, col_idx] = np.clip((arr[:, col_idx] - mean) / (std + 1e-8), -5.0, 5.0)
        return arr

    # ─── Training ────────────────────────────────────────────────────────────

    def fit(self, features: pl.DataFrame) -> "ExpertRegime":
        """
        Fit the HMM on historical M5 features.

        After fitting, states are re-ordered by mean return so that:
          state 0 = lowest return (Bear)
          state 1 = middle return (Sideways)
          state 2 = highest return (Bull)
        """
        X = self._extract_hmm_features(features)
        logger.info(f"🔮 Training HMM on {len(X):,} bars...")

        try:
            self.hmm.fit(X)
        except Exception as e:
            logger.warning(f"Initial HMM fit failed ({e}). Retrying with diag covariance...")
            self.hmm = GaussianHMM(
                n_components=settings.council.hmm_n_states,
                covariance_type="diag",
                n_iter=settings.council.hmm_n_iter,
                min_covar=1e-2,
                random_state=42,
                verbose=False,
            )
            self.hmm.fit(X)

        n_states = settings.council.hmm_n_states

        # Sanitize startprob_ and transmat_ against NaNs
        if not hasattr(self.hmm, "startprob_") or np.isnan(self.hmm.startprob_).any():
            self.hmm.startprob_ = np.full(n_states, 1.0 / n_states)
        else:
            self.hmm.startprob_ = np.nan_to_num(self.hmm.startprob_, nan=1.0 / n_states)
            self.hmm.startprob_ = np.maximum(self.hmm.startprob_, 1e-4)
            self.hmm.startprob_ /= self.hmm.startprob_.sum()

        if not hasattr(self.hmm, "transmat_") or np.isnan(self.hmm.transmat_).any():
            self.hmm.transmat_ = np.full((n_states, n_states), 1.0 / n_states)
        else:
            self.hmm.transmat_ = np.nan_to_num(self.hmm.transmat_, nan=1.0 / n_states)
            self.hmm.transmat_ = np.maximum(self.hmm.transmat_, 1e-4)
            self.hmm.transmat_ /= self.hmm.transmat_.sum(axis=1, keepdims=True)

        self._fitted = True

        # Re-order states by mean return (column 0 = return_pct)
        if hasattr(self.hmm, "means_") and not np.isnan(self.hmm.means_).any():
            mean_returns = self.hmm.means_[:, 0]
            ordered_idx  = np.argsort(mean_returns)
        else:
            ordered_idx = np.arange(n_states)
            mean_returns = np.zeros(n_states)

        self._state_map = {int(raw): int(ordered) for ordered, raw in enumerate(ordered_idx)}

        logger.info(
            f"✅ HMM fitted | Regime means (return_pct): "
            f"Bear={mean_returns[ordered_idx[0]]:.3f}% | "
            f"Sideways={mean_returns[ordered_idx[1]]:.3f}% | "
            f"Bull={mean_returns[ordered_idx[2]]:.3f}%"
        )
        return self

    # ─── Inference ────────────────────────────────────────────────────────────

    def predict_regime_sequence(self, features: pl.DataFrame) -> np.ndarray:
        """
        Predict the regime state for each bar in the feature DataFrame.
        Returns array of shape (n_bars,) with values in {0, 1, 2}.
        """
        if not self._fitted:
            raise RuntimeError("ExpertRegime must be fitted before prediction")

        try:
            X      = self._extract_hmm_features(features)
            states = self.hmm.predict(X)
            return np.array([self._state_map.get(int(s), 1) for s in states])
        except Exception as e:
            logger.warning(f"HMM predict fallback triggered ({e})")
            ret = features["return_pct"].to_numpy() if "return_pct" in features.columns else np.zeros(len(features))
            states = np.where(ret > 0.05, 2, np.where(ret < -0.05, 0, 1))
            return states

    def predict_current_regime(self, features: pl.DataFrame) -> Tuple[int, float]:
        """
        Predict the current market regime (last bar of features).

        Returns:
            (regime_id, weight_modifier)
            regime_id: 0=Bear, 1=Sideways, 2=Bull
            weight_modifier: position size scalar [0.25, 0.60, 1.00]
        """
        regimes  = self.predict_regime_sequence(features)
        regime   = int(regimes[-1])
        modifier = self.REGIME_MULTIPLIERS.get(regime, 0.5)

        logger.debug(
            f"🌡️ Current regime: {self.REGIME_NAMES.get(regime, 'SIDEWAYS')} "
            f"(modifier: {modifier:.2f})"
        )
        return regime, modifier

    def predict_regime_probabilities(self, features: pl.DataFrame) -> np.ndarray:
        """
        Return posterior probabilities for each regime state (last bar).
        Shape: (n_states,) — soft signal for MoE blending.
        """
        if not self._fitted:
            raise RuntimeError("ExpertRegime must be fitted before prediction")

        try:
            X          = self._extract_hmm_features(features)
            log_probs  = self.hmm.score_samples(X)[1]
            probs      = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
            probs     /= (probs.sum(axis=1, keepdims=True) + 1e-8)
            last_bar   = probs[-1]

            ordered = np.zeros(3)
            for raw_state, ordered_state in self._state_map.items():
                if int(raw_state) < len(last_bar):
                    ordered[ordered_state] = last_bar[int(raw_state)]

            if np.isnan(ordered).any() or ordered.sum() == 0:
                return np.array([0.33, 0.34, 0.33])
            return ordered / ordered.sum()
        except Exception as e:
            logger.warning(f"HMM probabilities fallback triggered ({e})")
            return np.array([0.33, 0.34, 0.33])

    # ─── Council Signal ───────────────────────────────────────────────────────

    def get_council_signal(self, features: pl.DataFrame) -> dict:
        """
        Return the Expert 2 signal for the Council gate.

        Output format expected by council.py:
          {
            "expert": "regime",
            "signal": float ∈ [-1, 1],   (−1=max bearish, +1=max bullish)
            "confidence": float ∈ [0, 1],
            "metadata": dict
          }
        """
        regime, modifier = self.predict_current_regime(features)
        probs = self.predict_regime_probabilities(features)

        # Signal: map Bull→+1, Sideways→0, Bear→-1 weighted by probability
        signal     = probs[2] - probs[0]  # P(Bull) - P(Bear) ∈ [-1, 1]
        confidence = float(probs.max())   # certainty of the regime call

        return {
            "expert":    "regime",
            "signal":    float(signal),
            "modifier":  modifier,
            "confidence": confidence,
            "metadata": {
                "regime":       self.REGIME_NAMES[regime],
                "p_bear":       float(probs[0]),
                "p_sideways":   float(probs[1]),
                "p_bull":       float(probs[2]),
            },
        }

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        path = path or self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"hmm": self.hmm, "state_map": self._state_map}, f)
        logger.info(f"💾 Regime HMM saved → {path}")

    def load(self, path: Path | None = None) -> "ExpertRegime":
        path = path or self.model_path
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.hmm        = data["hmm"]
        self._state_map = data["state_map"]
        self._fitted    = True
        logger.info(f"📂 Regime HMM loaded ← {path}")
        return self
