"""
The K-Dense Council — MoE Orchestrator
========================================
The Council is the top-level controller that:
  1. Runs all 5 experts in sequence
  2. Applies the HMM regime modifier first (gates all signals)
  3. Calls the NSGA-III gate to blend expert signals
  4. Outputs a final trading decision with TP/SL levels

Call flow per bar (live trading):
  features → E2 (Regime) → E1 (DRL) → E3 (TimesFM) → E4 (SHAP) → E5 (PyMC)
                         → Gate (NSGA-III blend) → TradingDecision

Training flow:
  1. Fit E1 (DRL)      on full feature history
  2. Fit E2 (HMM)      on full feature history
  3. Fit E4 (XGBoost)  on full feature history
  4. Run all experts on validation window → collect signal matrix
  5. Run Gate (NSGA-III) on signal matrix to find optimal α weights
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger

from src.config.settings import settings
from src.council.expert_trader   import ExpertTrader
from src.council.expert_regime   import ExpertRegime
from src.council.expert_prophet  import ExpertProphet
from src.council.expert_analyst  import ExpertAnalyst
from src.council.expert_actuary  import ExpertActuary
from src.council.gating_optimizer import GatingOptimizer
from src.data.tick_feature_engineer import TickFeatureEngineer


@dataclass
class TradingDecision:
    """Final Council trading decision for one bar."""
    symbol:           str
    consensus_signal: float          # ∈ [-1, 1] blended Council signal
    direction:        int            # +1=long, -1=short, 0=flat
    position_size:    float          # ∈ [0, 1] of max risk
    tp_price:         Optional[float] = None
    sl_price:         Optional[float] = None
    expected_rr:      float          = 0.0
    regime:           str            = "UNKNOWN"
    regime_modifier:  float          = 1.0
    council_confidence: float        = 0.0
    expert_signals:   dict           = field(default_factory=dict)
    expert_weights:   list           = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return self.council_confidence

    @property
    def is_tradeable(self) -> bool:
        """True if decision meets minimum signal threshold."""
        return (
            abs(self.consensus_signal) >= 0.20
            and self.council_confidence >= 0.40
            and self.expected_rr >= 1.5
            and self.direction != 0
        )

    def __repr__(self) -> str:
        dir_str = "LONG" if self.direction > 0 else "SHORT" if self.direction < 0 else "FLAT"
        return (
            f"Decision({self.symbol} | {dir_str} | "
            f"signal={self.consensus_signal:.3f} | "
            f"conf={self.council_confidence:.3f} | "
            f"RR={self.expected_rr:.2f} | "
            f"regime={self.regime} | "
            f"tradeable={'✅' if self.is_tradeable else '❌'})"
        )


class Council:
    """
    The K-Dense Council — MoE trading system orchestrator.

    Manages the lifecycle of all 5 expert agents and the NSGA-III gate.
    Provides unified train() and decide() interfaces.
    """

    def __init__(self):
        self.expert_trader  = ExpertTrader()
        self.expert_regime  = ExpertRegime()
        self.expert_prophet = ExpertProphet()
        self.expert_analyst = ExpertAnalyst()
        self.expert_actuary = ExpertActuary()
        self.gate           = GatingOptimizer()
        self.engineer       = TickFeatureEngineer(settings.mt5.timeframe_minutes)

        self._fitted = False

    # ─── Training Pipeline ────────────────────────────────────────────────────

    def train(
        self,
        features: pl.DataFrame,
        symbol: str,
        val_split: float = 0.2,
        timesteps: Optional[int] = None,
    ) -> "Council":
        """
        Full Council training pipeline.

        Args:
            features:   Fused feature DataFrame (MT5 + Yahoo correlation)
            symbol:     MT5 symbol name
            val_split:  Fraction of data held out for gate optimization
            timesteps:  Override total training timesteps for DRL agent

        Steps:
            1. Split train/val
            2. Fit E2 (HMM Regime) — fast, goes first
            3. Fit E4 (SHAP Analyst) — needs labels
            4. Fit E1 (DRL Trader) — slowest, most compute
            5. Run all experts on val set → signal matrix (T_val, 5)
            6. Run NSGA-III Gate → optimal α weights
        """
        logger.info(f"🏛️ Convening the Council for {symbol} training...")

        # ── Train/Val split (time-series — no shuffle) ────────────────────────
        n_total = len(features)
        n_val   = int(n_total * val_split)
        n_train = n_total - n_val

        train_df = features[:n_train]
        val_df   = features[n_train:]

        logger.info(f"   Train: {n_train:,} bars | Val: {n_val:,} bars")

        # ── Fit Expert 2 (HMM Regime) ─────────────────────────────────────────
        logger.info("🔮 [1/4] Fitting Expert 2: HMM Regime...")
        self.expert_regime.fit(train_df)

        # ── Fit Expert 4 (SHAP Analyst) ───────────────────────────────────────
        logger.info("🔬 [2/4] Fitting Expert 4: SHAP Analyst...")
        self.expert_analyst.fit(train_df)

        # ── Fit Expert 1 (DRL Trader) ─────────────────────────────────────────
        logger.info("🤖 [3/4] Fitting Expert 1: DRL Trader...")
        features_norm = self.engineer.normalize_for_rl(train_df)
        self.expert_trader.fit(features_norm, symbol=symbol, total_timesteps=timesteps)

        # ── Build Validation Signal Matrix ────────────────────────────────────
        logger.info("📊 [4/4] Optimizing Council Gate (NSGA-III)...")
        val_signals, val_returns = self._build_signal_matrix(val_df, symbol)

        # ── Run NSGA-III Gate ─────────────────────────────────────────────────
        self.gate.optimize(val_signals, val_returns)

        # ── Save all experts ──────────────────────────────────────────────────
        self._save(symbol)
        self._fitted = True

        logger.info(f"🏛️ Council training complete for {symbol} ✅")
        return self

    def _build_signal_matrix(
        self,
        features: pl.DataFrame,
        symbol: str,
        window: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run all experts bar-by-bar on the feature DataFrame to build
        the signal matrix for NSGA-III gate optimization.

        Returns:
            signals: (T, 5) signal matrix
            returns: (T,)   actual bar returns
        """
        signals_list = []
        n            = len(features)
        returns_arr  = features["return_pct"].to_numpy() / 100.0

        for t in range(window, n):
            bar_features = features[:t]
            current_price = float(bar_features["close"][-1])

            # Expert 2 (Regime) — fastest, run first
            e2_sig = self.expert_regime.get_council_signal(bar_features)

            # Expert 1 (DRL) — needs normalized array
            feat_norm = self.engineer.normalize_for_rl(bar_features)
            e1_sig    = self.expert_trader.get_council_signal(feat_norm, symbol)

            # Expert 3 (TimesFM) — zero-shot
            e3_sig = self.expert_prophet.get_council_signal(bar_features, symbol)

            # Expert 4 (SHAP) — XGBoost
            e4_sig = self.expert_analyst.get_council_signal(bar_features, symbol)

            # Expert 5 (PyMC) — Bayesian TP/SL
            direction = +1 if e1_sig["signal"] > 0 else -1
            e5_sig = self.expert_actuary.get_council_signal(
                bar_features,
                e3_sig.get("metadata", {}),
                current_price,
                direction,
                symbol,
            )

            signals_list.append([
                e1_sig["signal"],
                e2_sig["signal"],
                e3_sig["signal"],
                e4_sig["signal"],
                e5_sig["signal"],
            ])

        signals = np.array(signals_list, dtype=np.float32)
        returns = returns_arr[window:n]
        return signals, returns

    # ─── Live Decision Making ─────────────────────────────────────────────────

    def decide(
        self,
        features: pl.DataFrame,
        symbol: str,
    ) -> TradingDecision:
        """
        Make a trading decision for the current bar.

        This is the hot path — called every M5 bar during live trading.
        All experts run sequentially; regime modifier applied first.

        Args:
            features: Most recent M5 feature DataFrame (last N bars)
            symbol:   MT5 symbol

        Returns:
            TradingDecision with consensus signal, direction, TP/SL, etc.
        """
        if not self._fitted:
            logger.warning("Council not trained — returning neutral decision")
            return TradingDecision(symbol=symbol, consensus_signal=0.0,
                                   direction=0, position_size=0.0)

        current_price = float(features["close"][-1])

        # ── Expert 2: Regime (first — gates all others) ───────────────────────
        e2_out  = self.expert_regime.get_council_signal(features)
        regime  = e2_out["metadata"]["regime"]
        modifier = e2_out["modifier"]

        # ── Expert 1: DRL Trader ───────────────────────────────────────────────
        feat_norm = self.engineer.normalize_for_rl(features)
        e1_out    = self.expert_trader.get_council_signal(feat_norm, symbol)

        # ── Expert 3: TimesFM Prophet ─────────────────────────────────────────
        e3_out  = self.expert_prophet.get_council_signal(features, symbol)
        prophet_meta = e3_out.get("metadata", {})

        # ── Expert 4: SHAP Analyst ────────────────────────────────────────────
        e4_out  = self.expert_analyst.get_council_signal(features, symbol)

        # ── Expert 5: PyMC Actuary ────────────────────────────────────────────
        direction_hint = +1 if e1_out["signal"] > 0 else -1
        e5_out  = self.expert_actuary.get_council_signal(
            features, prophet_meta, current_price, direction_hint, symbol
        )

        # ── Council Gate: Blend signals ────────────────────────────────────────
        signals_1d = np.array([
            e1_out["signal"],
            e2_out["signal"],
            e3_out["signal"],
            e4_out["signal"],
            e5_out["signal"],
        ], dtype=np.float32)

        raw_signal = self.gate.blend_signals(signals_1d)

        # Apply regime modifier (dampens signal in Bear/Sideways)
        final_signal = raw_signal * modifier

        # Confidence = weighted mean of expert confidences
        confidences = np.array([
            e1_out["confidence"],
            e2_out["confidence"],
            e3_out["confidence"],
            e4_out["confidence"],
            e5_out["confidence"],
        ])
        council_confidence = float(np.dot(self.gate.optimal_weights, confidences))

        # Direction
        direction = +1 if final_signal > 0.05 else -1 if final_signal < -0.05 else 0

        # Position size = |signal| × regime_modifier × confidence
        position_size = abs(final_signal) * modifier * council_confidence

        decision = TradingDecision(
            symbol            = symbol,
            consensus_signal  = float(final_signal),
            direction         = direction,
            position_size     = min(1.0, position_size),
            tp_price          = e5_out.get("tp_price"),
            sl_price          = e5_out.get("sl_price"),
            expected_rr       = e5_out.get("expected_rr", 0.0),
            regime            = regime,
            regime_modifier   = modifier,
            council_confidence = council_confidence,
            expert_signals    = {
                "E1_trader":  e1_out["signal"],
                "E2_regime":  e2_out["signal"],
                "E3_prophet": e3_out["signal"],
                "E4_analyst": e4_out["signal"],
                "E5_actuary": e5_out["signal"],
            },
            expert_weights = self.gate.optimal_weights.tolist(),
        )

        logger.info(f"🏛️ {decision}")
        return decision

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _save(self, symbol: str) -> None:
        base = settings.council.models_dir / symbol.lower()
        base.mkdir(parents=True, exist_ok=True)
        self.expert_regime.save(base / "expert_regime.pkl")
        self.expert_analyst.save(base / "expert_analyst.pkl")
        self.expert_trader.save(base / "expert_trader_drl")
        self.gate.save(base / "council_gate.pkl")
        logger.info(f"💾 All Council models saved → {base}")

    def load(self, symbol: str) -> "Council":
        base = settings.council.models_dir / symbol.lower()
        self.expert_regime.load(base / "expert_regime.pkl")
        self.expert_analyst.load(base / "expert_analyst.pkl")
        drl_path = base / "expert_trader_drl"
        if not (drl_path.exists() or drl_path.with_suffix(".zip").exists()):
            drl_path = settings.council.best_model_dir / "expert_trader_drl"
        self.expert_trader.load(features_shape=(settings.council.drl_lookback_bars, 1), path=drl_path)
        self.gate.load(base / "council_gate.pkl")
        self._fitted = True
        logger.info(f"🏛️ Council loaded for {symbol} ← {base}")
        return self

    def evaluate(self, symbol: str, features: pl.DataFrame | pd.DataFrame) -> TradingDecision:
        """Convenience wrapper for decide() accepting either Polars or Pandas."""
        if hasattr(features, "to_pandas"):
            feat_pl = features
        else:
            feat_pl = pl.from_pandas(features)
        return self.decide(feat_pl, symbol=symbol)

    def fit(self, symbol: str, features: pl.DataFrame | pd.DataFrame) -> "Council":
        """Convenience wrapper for train() accepting either Polars or Pandas."""
        if hasattr(features, "to_pandas"):
            feat_pl = features
        else:
            feat_pl = pl.from_pandas(features)
        return self.train(feat_pl, symbol=symbol)


# Export alias
KDenseCouncil = Council
