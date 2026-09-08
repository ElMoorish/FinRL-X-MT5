"""
Expert 4 — The Analyst: SHAP + XGBoost Feature Attribution Signal Scorer
=========================================================================
Trains an XGBoost classifier on the fused feature matrix (MT5 ticks +
Yahoo correlation) to predict directional movement. SHAP values are used
to:
  1. Score the quality of the current signal (confidence weighting)
  2. Identify WHICH features are driving the prediction (interpretability)
  3. Prune irrelevant features to prevent overfitting

This expert provides the Council with a data-driven, interpretable signal
that explicitly leverages the MT5 ↔ Yahoo correlation edge.

SHAP interaction plots reveal: "QQQ outflowing → NAS100 signal weakens"
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl
import xgboost as xgb
import shap
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from src.config.settings import settings


class ExpertAnalyst:
    """
    Expert 4: SHAP-explained XGBoost directional signal scorer.

    Label: y = 1 if close[t+1] > close[t] * (1 + min_move_pct) else -1 (or 0)
    Prediction: Calibrated probability of upward move → signal ∈ [-1, 1]
    """

    # Minimum price move to count as directional (avoids noise labels)
    MIN_MOVE_PCT = 0.0005   # 0.05% per M5 bar

    # Feature importance threshold (SHAP) — features below this are dropped
    MIN_SHAP_IMPORTANCE = 0.001

    def __init__(self):
        cfg = settings.council
        self.xgb_params = {
            "n_estimators":    cfg.xgb_n_estimators,
            "max_depth":       cfg.xgb_max_depth,
            "learning_rate":   cfg.xgb_learning_rate,
            "subsample":       0.80,
            "colsample_bytree": 0.75,
            "min_child_weight": 3,
            "reg_alpha":       0.1,
            "reg_lambda":      1.0,
            "use_label_encoder": False,
            "eval_metric":     "logloss",
            "tree_method":     "hist",
            "random_state":    42,
        }
        self.model:   Optional[xgb.XGBClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self._feature_cols: list[str] = []
        self._shap_importance: Optional[np.ndarray] = None
        self.model_path = settings.council.models_dir / "expert_analyst_xgb.pkl"

    # ─── Feature Selection ────────────────────────────────────────────────────

    def _select_features(self, df: pl.DataFrame) -> list[str]:
        """
        Select numerical feature columns for XGBoost.
        Excludes raw price levels (to avoid non-stationarity).
        """
        exclude = {
            "time", "date", "open", "high", "low", "close",
            "bid_close", "ask_close", "buy_ticks",
            "vwap",  # keep close_vs_vwap (normalized)
        }
        # Keep boolean (session flags) and all float/int features
        numeric_types = {pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Boolean, pl.UInt32}
        cols = [
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in numeric_types
        ]
        return cols

    # ─── Label Construction ───────────────────────────────────────────────────

    def _build_labels(self, features: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build forward-looking binary labels:
          y = +1 if next-bar return > +MIN_MOVE_PCT
          y = -1 if next-bar return < -MIN_MOVE_PCT
          y =  0 (flat — excluded from training)

        Returns (X, y) arrays with flat samples removed.
        """
        close   = features["close"].to_numpy()
        returns = np.diff(close) / close[:-1]

        y_raw = np.where(
            returns > self.MIN_MOVE_PCT,   1,
            np.where(returns < -self.MIN_MOVE_PCT, -1, 0)
        )

        # Align features (drop last row — no label) and remove flat
        feature_cols = self._select_features(features)
        X_all        = features.select(feature_cols)[:-1].to_numpy().astype(np.float32)

        mask = y_raw != 0
        X    = X_all[mask]
        y    = y_raw[mask]

        # Re-label for XGBoost: -1 → 0, +1 → 1
        y_binary = (y == 1).astype(np.int32)
        return X, y_binary, feature_cols

    # ─── Training ────────────────────────────────────────────────────────────

    def fit(self, features: pl.DataFrame) -> "ExpertAnalyst":
        """
        Train XGBoost on historical M5 feature matrix.
        Uses TimeSeriesSplit cross-validation to avoid lookahead bias.
        """
        X, y, feature_cols = self._build_labels(features)
        self._feature_cols = feature_cols

        logger.info(
            f"🔬 Training XGBoost Analyst | "
            f"{len(X):,} samples | "
            f"{len(feature_cols)} features | "
            f"class balance: {y.mean():.3f}"
        )

        self.model = xgb.XGBClassifier(**self.xgb_params)

        # Time-series CV for early stopping evaluation
        tscv = TimeSeriesSplit(n_splits=5)
        eval_set_X, eval_set_y = X[-len(X)//5:], y[-len(y)//5:]

        self.model.fit(
            X, y,
            eval_set=[(eval_set_X, eval_set_y)],
            verbose=100,
        )

        # Build SHAP explainer
        logger.info("🔍 Computing SHAP values...")
        self.explainer = shap.TreeExplainer(self.model)
        sample_idx     = np.random.choice(len(X), min(2000, len(X)), replace=False)
        shap_values    = self.explainer.shap_values(X[sample_idx])

        # Mean absolute SHAP importance per feature
        self._shap_importance = np.abs(shap_values).mean(axis=0)

        # Log top-10 most important features
        top_idx = np.argsort(self._shap_importance)[::-1][:10]
        for i in top_idx:
            logger.info(
                f"  SHAP | {feature_cols[i]:40s} | {self._shap_importance[i]:.5f}"
            )

        logger.info("✅ XGBoost Analyst trained")
        return self

    # ─── Inference ────────────────────────────────────────────────────────────

    def predict_signal(self, features: pl.DataFrame) -> dict:
        """
        Predict directional signal for current bar.

        Returns:
          signal:     ∈ [-1, 1] — (2 × P(up)) - 1, calibrated probability
          confidence: feature quality score (top SHAP features active)
          top_drivers: list of (feature_name, shap_value) top 5 drivers
        """
        if self.model is None or self.explainer is None:
            raise RuntimeError("ExpertAnalyst must be fitted before prediction")

        # Use last available bar, ensuring all expected feature columns exist
        cols_to_select = [
            pl.col(c) if c in features.columns else pl.lit(0.0).alias(c)
            for c in self._feature_cols
        ]
        X_latest = (
            features.select(cols_to_select)
            .tail(1)
            .to_numpy()
            .astype(np.float32)
        )
        X_latest = np.nan_to_num(X_latest, nan=0.0)

        # Probability of upward move
        prob_up  = float(self.model.predict_proba(X_latest)[0, 1])
        signal   = 2 * prob_up - 1.0   # re-center ∈ [-1, 1]

        # SHAP explanation for current bar
        shap_vals = self.explainer.shap_values(X_latest)[0]

        # Top 5 feature drivers
        top_idx  = np.argsort(np.abs(shap_vals))[::-1][:5]
        drivers  = [
            (self._feature_cols[i], float(shap_vals[i]))
            for i in top_idx
        ]

        # Confidence = proportion of high-importance features that are non-zero
        important_mask = self._shap_importance > self.MIN_SHAP_IMPORTANCE
        active         = np.abs(X_latest[0][important_mask]) > 0.01
        confidence     = float(active.mean()) if active.size > 0 else 0.5

        return {
            "prob_up":    prob_up,
            "signal":     signal,
            "confidence": confidence,
            "top_drivers": drivers,
        }

    # ─── Council Signal ───────────────────────────────────────────────────────

    def get_council_signal(self, features: pl.DataFrame, symbol: str = "") -> dict:
        """Return Expert 4 formatted signal for the Council gate."""
        prediction = self.predict_signal(features)

        logger.debug(
            f"📊 {symbol} Analyst | signal={prediction['signal']:.3f} | "
            f"conf={prediction['confidence']:.3f} | "
            f"P(up)={prediction['prob_up']:.3f}"
        )

        return {
            "expert":     "analyst",
            "signal":     prediction["signal"],
            "confidence": prediction["confidence"],
            "metadata": {
                "prob_up":     prediction["prob_up"],
                "top_drivers": prediction["top_drivers"],
            },
        }

    # ─── Feature Importance Report ────────────────────────────────────────────

    def get_feature_importance_df(self) -> pl.DataFrame:
        """Return full SHAP importance table as a Polars DataFrame."""
        if self._shap_importance is None:
            return pl.DataFrame()
        return pl.DataFrame({
            "feature":    self._feature_cols,
            "shap_importance": self._shap_importance.tolist(),
        }).sort("shap_importance", descending=True)

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        path = path or self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":            self.model,
                "feature_cols":     self._feature_cols,
                "shap_importance":  self._shap_importance,
            }, f)
        logger.info(f"💾 Analyst XGBoost saved → {path}")

    def load(self, path: Path | None = None) -> "ExpertAnalyst":
        path = path or self.model_path
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model            = data["model"]
        self._feature_cols    = data["feature_cols"]
        self._shap_importance = data["shap_importance"]
        self.explainer        = shap.TreeExplainer(self.model)
        logger.info(f"📂 Analyst XGBoost loaded ← {path}")
        return self
