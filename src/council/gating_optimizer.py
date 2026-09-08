"""
Council Gate — NSGA-III Pareto-Optimal Expert Weighting (pymoo)
================================================================
The Council orchestrates all 5 experts and finds the optimal blending
of their signals using NSGA-III multi-objective optimization.

NSGA-III simultaneously optimizes 3 conflicting objectives:
  f1: Maximize Sharpe Ratio
  f2: Maximize Win Rate
  f3: Minimize Max Drawdown

Decision variables: α = [α₁, α₂, α₃, α₄, α₅] — expert weights
Constraints: Σαᵢ ≈ 1, αᵢ ≥ 0

Result: Pareto front of optimal expert weight vectors.
The Council selects the solution maximizing Sharpe from the Pareto front.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions


class CouncilGatingProblem(ElementwiseProblem):
    """
    NSGA-III optimization problem for expert weight discovery.

    Objective:
      Minimize: [-Sharpe, -WinRate, MaxDrawdown]
      (pymoo always minimizes, so we negate Sharpe and WinRate)

    Constraint:
      |Σαᵢ - 1| < ε  (weights sum to ~1)
    """

    def __init__(
        self,
        expert_signals: np.ndarray,  # (T, 5) — signal per expert per bar
        bar_returns:    np.ndarray,   # (T,)   — actual M5 returns
    ):
        super().__init__(
            n_var=5,           # α₁...α₅
            n_obj=3,           # Sharpe, WinRate, MaxDD
            n_ieq_constr=1,    # |Σα - 1| ≤ ε
            xl=np.zeros(5),    # αᵢ ≥ 0
            xu=np.ones(5),     # αᵢ ≤ 1
        )
        self.signals = expert_signals
        self.returns = bar_returns

    def _evaluate(self, alpha: np.ndarray, out: dict, *args, **kwargs):
        # Normalize weights to sum to 1
        alpha_norm = alpha / (alpha.sum() + 1e-8)

        # Blended signal: (T,)
        blended   = self.signals @ alpha_norm  # shape (T,)
        pnl       = blended * self.returns

        # Objective 1: Negative Sharpe (minimize → maximize Sharpe)
        sharpe = (pnl.mean() / (pnl.std() + 1e-8)) * np.sqrt(252 * 12)  # annualized M5
        f1     = -sharpe

        # Objective 2: Negative Win Rate
        win_rate = float((pnl > 0).mean())
        f2       = -win_rate

        # Objective 3: Max Drawdown
        equity   = np.cumprod(1 + pnl)
        peak     = np.maximum.accumulate(equity)
        drawdown = ((peak - equity) / (peak + 1e-8)).max()
        f3       = float(drawdown)

        out["F"] = [f1, f2, f3]
        out["G"] = [abs(alpha_norm.sum() - 1.0) - 0.005]  # ≈ sum-to-one


class GatingOptimizer:
    """
    Runs NSGA-III to find Pareto-optimal expert weights.
    Called during training / periodic re-calibration.
    """

    def __init__(self):
        cfg              = settings_lazy()
        self.pop_size    = cfg.nsga3_pop_size
        self.n_gen       = cfg.nsga3_n_gen
        self.n_partitions = cfg.nsga3_n_partitions
        self.seed        = cfg.nsga3_seed
        self._optimal_weights: Optional[np.ndarray] = None
        self._pareto_front:    Optional[np.ndarray] = None
        self.save_path   = None

    def optimize(
        self,
        expert_signals: np.ndarray,
        bar_returns:    np.ndarray,
    ) -> np.ndarray:
        """
        Run NSGA-III to find optimal expert weights.

        Args:
            expert_signals: (T, 5) expert signal matrix
            bar_returns:    (T,)   actual M5 price returns

        Returns:
            optimal_weights: (5,) sum-to-1 expert weights
        """
        logger.info(
            f"⚙️ Running NSGA-III Council Gate | "
            f"pop={self.pop_size}, gen={self.n_gen} | "
            f"{len(bar_returns):,} bars"
        )

        problem  = CouncilGatingProblem(expert_signals, bar_returns)
        ref_dirs = get_reference_directions(
            "das-dennis", 3, n_partitions=self.n_partitions
        )
        algorithm = NSGA3(pop_size=self.pop_size, ref_dirs=ref_dirs)

        result = minimize(
            problem, algorithm,
            ("n_gen", self.n_gen),
            seed=self.seed,
            verbose=False,
            save_history=False,
        )

        # result.X: (n_pareto, 5) — all Pareto-optimal weight vectors
        # result.F: (n_pareto, 3) — [−Sharpe, −WinRate, MaxDD]
        pareto_weights = result.X
        pareto_F       = result.F

        # Select solution with best (lowest) negative Sharpe from Pareto front
        best_idx       = int(np.argmin(pareto_F[:, 0]))
        best_weights   = pareto_weights[best_idx] / pareto_weights[best_idx].sum()

        self._optimal_weights = best_weights
        self._pareto_front    = pareto_F

        logger.info(
            f"✅ NSGA-III complete | Optimal weights: "
            f"Trader={best_weights[0]:.3f} | "
            f"Regime={best_weights[1]:.3f} | "
            f"Prophet={best_weights[2]:.3f} | "
            f"Analyst={best_weights[3]:.3f} | "
            f"Actuary={best_weights[4]:.3f}"
        )
        logger.info(
            f"   Best solution: Sharpe={-pareto_F[best_idx,0]:.3f} | "
            f"WinRate={-pareto_F[best_idx,1]:.3f} | "
            f"MaxDD={pareto_F[best_idx,2]:.3f}"
        )
        return best_weights

    @property
    def optimal_weights(self) -> np.ndarray:
        if self._optimal_weights is None:
            logger.warning("No weights optimized yet — using equal weights")
            return np.ones(5) / 5
        return self._optimal_weights

    def blend_signals(self, expert_signals_1d: np.ndarray) -> float:
        """
        Apply optimal weights to blend a single-bar signal vector.

        Args:
            expert_signals_1d: (5,) current bar signals from each expert

        Returns:
            blended: float ∈ [-1, 1] — council consensus signal
        """
        w       = self.optimal_weights
        blended = float(np.dot(w, expert_signals_1d))
        return np.clip(blended, -1.0, 1.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "optimal_weights": self._optimal_weights,
                "pareto_front":    self._pareto_front,
            }, f)
        logger.info(f"💾 Council gate weights saved → {path}")

    def load(self, path: Path) -> "GatingOptimizer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._optimal_weights = data["optimal_weights"]
        self._pareto_front    = data.get("pareto_front")
        logger.info(
            f"📂 Council gate loaded ← {path} | "
            f"weights={self._optimal_weights.round(3).tolist()}"
        )
        return self


def settings_lazy():
    """Lazy import to avoid circular dependency."""
    from src.config.settings import settings
    return settings.council
