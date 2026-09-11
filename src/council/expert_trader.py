"""
Expert 1 — The Trader: PPO/SAC Deep Reinforcement Learning Agent
=================================================================
Trains a Stable Baselines 3 SAC/PPO agent on the fused MT5+Yahoo
feature matrix. The agent learns a continuous weight vector ∈ [-1, 1]
representing directional conviction per instrument.

Environment design (FinRL-X weight-centric):
  - State:  lookback window of normalized M5 features (microstructure + corr)
  - Action: continuous weight ∈ [-1, 1] per instrument
  - Reward: Sharpe-adjusted PnL - spread cost - slippage penalty

The DRL agent is the primary signal generator. Other experts (HMM, SHAP,
TimesFM, PyMC) act as FILTERS and SCALERS on this base signal.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import polars as pl
from loguru import logger
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from src.config.settings import settings


# ─── Custom Gymnasium Environment ─────────────────────────────────────────────

class MT5TradingEnv(gym.Env):
    """
    FinRL-X-MT5 custom Gymnasium trading environment.

    Observation: (lookback, n_features) window of normalized M5 features
    Action:      Continuous weight ∈ [-1, 1] per instrument
    Reward:      Sharpe-scaled PnL after spread costs
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,    # (T, n_features) normalized
        symbol:   str = "USTEC",
        lookback: int = 50,
        spread_pct: float = 0.0002,  # 2 bps round-trip
        initial_equity: float = 10_000.0,
        symmetric_training: bool = True,
    ):
        super().__init__()
        self.features      = features
        self.symbol        = symbol
        self.lookback      = lookback
        self.spread_pct    = spread_pct
        self.initial_equity = initial_equity
        self.symmetric_training = symmetric_training
        self._invert_episode = False

        n_features = features.shape[1]

        # Action: single instrument weight ∈ [-1, 1]
        # (Multi-instrument: extend to Box of shape (n_instruments,))
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation: lookback × features
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0,
            shape=(lookback, n_features), dtype=np.float32
        )

        self._reset_state()

    def _reset_state(self):
        self._step         = self.lookback
        self._equity       = self.initial_equity
        self._prev_equity  = self.initial_equity
        self._position     = 0.0         # current weight
        self._returns_hist = []
        self._peak_equity  = self.initial_equity

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        if self.symmetric_training:
            self._invert_episode = bool(np.random.rand() > 0.5)
        else:
            self._invert_episode = False
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action[0], -1.0, 1.0)

        # Simulate price return for this bar
        if self._step >= len(self.features) - 1:
            return self._get_obs(), 0.0, True, False, {}

        # Approximate price return from features (index 0 = return_pct after normalization)
        bar_return = float(self.features[self._step, 0]) * 0.01  # denormalize approx
        if self._invert_episode:
            bar_return = -bar_return

        # PnL = weight × return - |Δweight| × spread
        delta_weight = abs(action - self._position)
        pnl          = action * bar_return - delta_weight * self.spread_pct

        self._equity    *= (1 + pnl)
        self._position   = action
        self._peak_equity = max(self._peak_equity, self._equity)

        # Reward: Sharpe-adjusted (online running estimate)
        self._returns_hist.append(pnl)
        if len(self._returns_hist) > 252:
            self._returns_hist.pop(0)

        mean_r = np.mean(self._returns_hist)
        std_r  = np.std(self._returns_hist) + 1e-8
        sharpe_reward = mean_r / std_r * pnl

        # Drawdown penalty
        dd = (self._peak_equity - self._equity) / (self._peak_equity + 1e-8)
        dd_penalty = -max(0, dd - 0.05) * 2.0  # Penalize DD > 5%

        reward = float(sharpe_reward + dd_penalty)

        self._step += 1
        terminated = bool(
            self._step >= len(self.features) - 1
            or self._equity < self.initial_equity * 0.7   # Bust condition
        )

        return self._get_obs(), reward, terminated, False, {
            "equity": self._equity,
            "position": self._position,
            "pnl": pnl,
        }

    def _get_obs(self) -> np.ndarray:
        start = max(0, self._step - self.lookback)
        obs   = self.features[start: self._step]
        # Pad if at start
        if len(obs) < self.lookback:
            pad = np.zeros((self.lookback - len(obs), obs.shape[1]), dtype=np.float32)
            obs = np.vstack([pad, obs])
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        if self._invert_episode:
            obs = -obs
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def render(self):
        logger.info(
            f"Step={self._step} | Equity=${self._equity:,.2f} | "
            f"Position={self._position:.3f}"
        )


# ─── Expert 1: The Trader ─────────────────────────────────────────────────────

class ExpertTrader:
    """
    Expert 1: SAC/PPO DRL agent trained on fused MT5+Yahoo features.

    Produces a continuous weight signal ∈ [-1, 1] for each instrument.
    This is the primary directional signal; other experts modulate it.
    """

    ALGORITHMS = {"SAC": SAC, "PPO": PPO}

    def __init__(self, algorithm: str | None = None):
        cfg = settings.council
        self.algorithm   = (algorithm or cfg.drl_algorithm).upper()
        self.n_envs      = cfg.drl_n_envs
        self.lookback    = cfg.drl_lookback_bars
        self.total_steps = cfg.drl_total_timesteps
        self.model_path  = cfg.best_model_dir / "expert_trader_drl"
        self._model: Optional[SAC | PPO] = None
        self._vec_normalize_path = cfg.best_model_dir / "expert_trader_vecnorm.pkl"
        # Restored VecNormalize running statistics (mean/var per feature)
        # Populated by load() — ensures inference distribution matches training
        self._obs_rms = None

    # ─── Training ────────────────────────────────────────────────────────────

    def fit(
        self,
        features_normalized: np.ndarray,
        symbol: str = "NAS100.x",
        total_timesteps: Optional[int] = None,
    ) -> "ExpertTrader":
        """
        Train the DRL agent on the normalized feature matrix.

        Args:
            features_normalized: (T, n_features) Z-score normalized array
            symbol:              MT5 symbol name for environment label
            total_timesteps:     Override total training timesteps
        """
        AlgoCls = self.ALGORITHMS.get(self.algorithm, SAC)

        def make_env():
            env = MT5TradingEnv(
                features    = features_normalized,
                symbol      = symbol,
                lookback    = self.lookback,
                spread_pct  = 0.0002,
            )
            return env

        # Validate environment
        check_env(make_env(), warn=True)

        # Vectorized environments
        vec_env = make_vec_env(make_env, n_envs=self.n_envs)
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

        # Eval env (single, deterministic)
        eval_env = VecNormalize(
            make_vec_env(make_env, n_envs=1),
            norm_obs=True, norm_reward=False, training=False
        )

        steps = total_timesteps or self.total_steps
        logger.info(
            f"🤖 Training {self.algorithm} DRL on {symbol} | "
            f"{steps:,} timesteps | {self.n_envs} envs"
        )

        algo_kwargs = {
            "policy":        "MlpPolicy",
            "env":           vec_env,
            "verbose":       1,
            "tensorboard_log": str(settings.council.logs_dir / "tensorboard"),
        }

        if self.algorithm == "SAC":
            algo_kwargs.update({
                "learning_rate": settings.council.drl_learning_rate,
                "buffer_size":   settings.council.drl_buffer_size,
                "batch_size":    settings.council.drl_batch_size,
                "gradient_steps": -1,   # Off-policy best practice (SB3 skill)
                "gamma": 0.99,
                "tau":   0.005,
            })
        elif self.algorithm == "PPO":
            algo_kwargs.update({
                "learning_rate": settings.council.drl_learning_rate,
                "n_steps":       2048,
                "batch_size":    64,
                "n_epochs":      10,
                "gamma":         0.99,
                "gae_lambda":    0.95,
                "clip_range":    0.2,
            })

        self._model = AlgoCls(**algo_kwargs)

        eval_freq = max(1000, steps // 20)
        save_freq = max(2000, steps // 10)
        callbacks = CallbackList([
            EvalCallback(
                eval_env,
                best_model_save_path=str(settings.council.best_model_dir),
                log_path=str(settings.council.logs_dir),
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
            ),
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(settings.council.checkpoint_dir),
                name_prefix=f"drl_{symbol.lower()}",
            ),
        ])

        self._model.learn(
            total_timesteps=steps,
            callback=callbacks,
            progress_bar=True,
        )

        # Save VecNormalize statistics
        vec_env.save(str(self._vec_normalize_path))
        logger.info(f"✅ DRL training complete → {self.model_path}")
        return self

    # ─── Inference ────────────────────────────────────────────────────────────

    def predict_weight(self, features_normalized: np.ndarray) -> tuple[float, float]:
        """
        Predict current position weight.

        Returns:
            (weight, confidence)
            weight:     ∈ [-1, 1] — directional conviction
            confidence: entropy-based uncertainty measure
        """
        if self._model is None:
            raise RuntimeError("ExpertTrader must be fitted before prediction")

        # Get last lookback window
        obs = features_normalized[-self.lookback:].astype(np.float32)
        if obs.shape[0] < self.lookback:
            pad = np.zeros((self.lookback - obs.shape[0], obs.shape[1]))
            obs = np.vstack([pad, obs])

        # ── GAP-M3 Fix: apply VecNormalize statistics saved during training ──
        # During training: normalize_for_rl() → VecNormalize (double normalization)
        # At inference:    normalize_for_rl() only (mis-match) → fixed here
        if self._obs_rms is not None:
            try:
                obs_mean = np.asarray(self._obs_rms.mean, dtype=np.float32)
                obs_var  = np.asarray(self._obs_rms.var,  dtype=np.float32)
                # Apply per-feature normalization matching VecNormalize behaviour
                # obs shape: (lookback, n_features); rms shape: (n_features,)
                if obs_mean.shape == obs.shape[1:]:
                    obs = (obs - obs_mean) / np.sqrt(obs_var + 1e-8)
                    obs = np.clip(obs, -10.0, 10.0).astype(np.float32)
            except Exception as e:
                logger.debug(f"VecNormalize application skipped: {e}")

        obs_tensor  = obs[np.newaxis, ...]  # (1, lookback, features)
        action, _   = self._model.predict(obs_tensor, deterministic=True)
        weight      = float(np.clip(action[0], -1.0, 1.0))

        # Confidence: how far from 0 is the weight (certainty proxy)
        confidence = min(1.0, abs(weight) / 0.5)

        return weight, confidence

    # ─── Council Signal ───────────────────────────────────────────────────────

    def get_council_signal(
        self, features_normalized: np.ndarray, symbol: str = ""
    ) -> dict:
        """Return Expert 1 formatted signal for the Council gate."""
        weight, confidence = self.predict_weight(features_normalized)

        logger.debug(
            f"🤖 {symbol} DRL Trader | weight={weight:.3f} | conf={confidence:.3f}"
        )

        return {
            "expert":     "trader",
            "signal":     weight,
            "confidence": confidence,
            "metadata":   {"weight": weight, "algorithm": self.algorithm},
        }

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Path | str | None = None) -> None:
        target = Path(path) if path else self.model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if self._model:
            self._model.save(str(target))
            logger.info(f"💾 DRL model saved → {target}")

    def load(self, features_shape: tuple | None = None, path: Path | str | None = None) -> "ExpertTrader":
        import pickle
        AlgoCls = self.ALGORITHMS.get(self.algorithm, SAC)
        target = Path(path) if path else self.model_path
        if not target.exists() and target.with_suffix(".zip").exists():
            target = target.with_suffix(".zip")
        self._model = AlgoCls.load(str(target))

        # ── GAP-M3 Fix: restore VecNormalize running statistics ──────────────
        # VecNormalize saves obs_rms (running mean/var per feature dim).
        # Without restoring these, inference observations are on a different
        # scale than what the policy was trained on, causing degraded Q-values.
        self._obs_rms = None
        if self._vec_normalize_path.exists():
            try:
                with open(str(self._vec_normalize_path), "rb") as f:
                    vec_norm_obj = pickle.load(f)
                self._obs_rms = vec_norm_obj.obs_rms
                logger.info(
                    f"📊 VecNormalize stats restored ← {self._vec_normalize_path.name} "
                    f"| obs_mean μ={float(self._obs_rms.mean.mean()):.4f} "
                    f"σ²={float(self._obs_rms.var.mean()):.4f}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not restore VecNormalize stats ({e}). "
                    "DRL inference will run without observation re-normalization. "
                    "Re-run council.train() to regenerate vecnorm file."
                )
        else:
            logger.warning(
                f"VecNormalize stats not found at {self._vec_normalize_path}. "
                "This file is generated by council.train(). Re-train to fix."
            )

        logger.info(f"📂 DRL model loaded ← {target}")
        return self
