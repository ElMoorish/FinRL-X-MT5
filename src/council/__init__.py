"""
The K-Dense Council — Mixture of Experts Architecture
======================================================
5-Expert Council with NSGA-III Pareto Gating.
"""

from src.council.council import Council, KDenseCouncil, TradingDecision
from src.council.expert_trader import ExpertTrader
from src.council.expert_regime import ExpertRegime
from src.council.expert_prophet import ExpertProphet
from src.council.expert_analyst import ExpertAnalyst
from src.council.expert_actuary import ExpertActuary
from src.council.gating_optimizer import GatingOptimizer

__all__ = [
    "Council",
    "KDenseCouncil",
    "TradingDecision",
    "ExpertTrader",
    "ExpertRegime",
    "ExpertProphet",
    "ExpertAnalyst",
    "ExpertActuary",
    "GatingOptimizer",
]
