"""
Council Chain-of-Thought (CoT) Generator
=========================================
Synthesizes the mathematical and probabilistic outputs of all 5 Council
specialists into a transparent, step-by-step institutional reasoning narrative.
"""

from __future__ import annotations
from typing import Any
from src.council.council import TradingDecision


class ChainOfThoughtGenerator:
    """
    Translates raw specialist signals into structured, human-readable CoT traces.
    """

    @classmethod
    def generate(cls, decision: TradingDecision) -> dict[str, Any]:
        """
        Generate detailed Chain-of-Thought breakdown for a TradingDecision.
        """
        sym = decision.symbol
        sig = decision.consensus_signal
        conf = decision.council_confidence
        regime = decision.regime
        rr = decision.expected_rr
        is_trade = decision.is_tradeable
        direction = "LONG (BUY)" if decision.direction > 0 else "SHORT (SELL)" if decision.direction < 0 else "FLAT (NEUTRAL)"
        
        e_signals = decision.expert_signals or {}
        e_weights = decision.expert_weights or [0.2, 0.2, 0.2, 0.2, 0.2]

        s_trader = e_signals.get("E1_trader", 0.0)
        s_regime = e_signals.get("E2_regime", 0.0)
        s_prophet = e_signals.get("E3_prophet", 0.0)
        s_analyst = e_signals.get("E4_analyst", 0.0)
        s_actuary = e_signals.get("E5_actuary", 0.0)

        steps = []

        # Step 1: Regime Assessment
        if regime == "BULL":
            regime_desc = f"Gaussian HMM identifies an expansionary **BULL** regime (1.00x modifier). Trend-following long exposures favored."
        elif regime == "BEAR":
            regime_desc = f"Gaussian HMM flags a contractionary **BEAR** regime (0.25x modifier). Exposure dampened; long signals suppressed."
        else:
            regime_desc = f"Gaussian HMM detects choppy **SIDEWAYS** consolidation (0.50x modifier). Mean-reversion constraints applied."
        steps.append({
            "expert": "E2: Regime Specialist (HMM)",
            "icon": "🌡️",
            "signal": s_regime,
            "weight": e_weights[1] if len(e_weights) > 1 else 0.2,
            "narrative": regime_desc
        })

        # Step 2: Analyst Basket & Microstructure
        if s_analyst > 0.15:
            analyst_desc = f"XGBoost factor attribution shows positive cross-asset tailwind (+{s_analyst:.2f}). Underlying ETF basket and order flow favor upside expansion."
        elif s_analyst < -0.15:
            analyst_desc = f"XGBoost factor attribution signals structural selling pressure ({s_analyst:.2f}). ETF basket divergence and order flow skew negative."
        else:
            analyst_desc = f"XGBoost detects balanced order flow ({s_analyst:+.2f}). Macro correlation basket indicates market equilibrium."
        steps.append({
            "expert": "E4: Cross-Asset Analyst (SHAP / XGB)",
            "icon": "📊",
            "signal": s_analyst,
            "weight": e_weights[3] if len(e_weights) > 3 else 0.2,
            "narrative": analyst_desc
        })

        # Step 3: Prophet Volatility Horizon
        if s_prophet > 0.05:
            prophet_desc = f"TimesFM projection forecasts positive price trajectory ({s_prophet:+.2f}) with expanding upper volatility envelopes."
        elif s_prophet < -0.05:
            prophet_desc = f"TimesFM projection indicates mean-reverting downward slope ({s_prophet:+.2f}) testing lower band boundaries."
        else:
            prophet_desc = f"TimesFM projects price stabilization within neutral band boundaries ({s_prophet:+.2f})."
        steps.append({
            "expert": "E3: Prophet Forecasting (TimesFM)",
            "icon": "🔮",
            "signal": s_prophet,
            "weight": e_weights[2] if len(e_weights) > 2 else 0.2,
            "narrative": prophet_desc
        })

        # Step 4: DRL Policy
        if s_trader > 0.10:
            trader_desc = f"SAC Deep RL agent selects Buy policy action ({s_trader:+.2f}), exploiting positive state-action Q-value trajectory."
        elif s_trader < -0.10:
            trader_desc = f"SAC Deep RL agent selects Short policy action ({s_trader:+.2f}), seeking alpha in downward acceleration."
        else:
            trader_desc = f"SAC Deep RL agent holds conservative position ({s_trader:+.2f}), minimizing policy entropy cost in neutral chop."
        steps.append({
            "expert": "E1: Execution Trader (SAC DRL)",
            "icon": "🤖",
            "signal": s_trader,
            "weight": e_weights[0] if len(e_weights) > 0 else 0.2,
            "narrative": trader_desc
        })

        # Step 5: Actuary Risk-to-Reward
        rr_status = "PASS" if rr >= 1.50 else "FAIL"
        actuary_desc = (
            f"Bayesian VaR model calculates Expected RR = **{rr:.2f}** ({'≥ 1.50 minimum target' if rr >= 1.50 else '< 1.50 insufficient reward'}). "
            f"Optimal SL: {decision.sl_price or 'N/A'} | TP: {decision.tp_price or 'N/A'}. "
            f"Half-Kelly allocation factor: {decision.position_size:.2f}."
        )
        steps.append({
            "expert": "E5: Actuary & Risk Solvency (Bayesian VaR)",
            "icon": "⚖️",
            "signal": s_actuary,
            "weight": e_weights[4] if len(e_weights) > 4 else 0.2,
            "narrative": actuary_desc
        })

        # Final Consensus Synthesis
        checkpoints = [
            {"name": "Signal Conviction", "passed": abs(sig) >= 0.15, "value": f"|{sig:.3f}| ≥ 0.15"},
            {"name": "Council Confidence", "passed": conf >= 0.40, "value": f"{conf:.1%} ≥ 40.0%"},
            {"name": "Risk / Reward Threshold", "passed": rr >= 1.50, "value": f"{rr:.2f} ≥ 1.50"},
            {"name": "Regime Concurrence", "passed": decision.regime_modifier > 0.20, "value": f"Modifier {decision.regime_modifier:.2f} > 0.20"},
        ]

        if is_trade:
            verdict = f"APPROVED — The Council reaches strong consensus for a {direction} trade. All 4 risk and conviction checkpoints satisfied."
        else:
            failed = [c["name"] for c in checkpoints if not c["passed"]]
            failed_str = ", ".join(failed) if failed else "Signal magnitude neutral"
            verdict = f"WITHHELD — Council preserves capital. Entry rejected due to: {failed_str}."

        return {
            "symbol": sym,
            "direction": direction,
            "consensus_signal": round(sig, 3),
            "confidence": round(conf, 3),
            "regime": regime,
            "expected_rr": round(rr, 2),
            "is_tradeable": is_trade,
            "position_size": round(decision.position_size, 3),
            "sl_price": decision.sl_price,
            "tp_price": decision.tp_price,
            "steps": steps,
            "checkpoints": checkpoints,
            "verdict": verdict,
        }
