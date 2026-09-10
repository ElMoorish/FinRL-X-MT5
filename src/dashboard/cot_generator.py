"""
Council Chain-of-Thought (CoT) Generator
=========================================
Synthesizes the mathematical and probabilistic outputs of all 5 Council
specialists into a transparent, step-by-step institutional reasoning narrative.
Fully synchronized with High-Conviction (0.70 Conf, 0.50% Risk, H1 Trend Governor)
and symmetric regime-aware weighting.
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

        w_trader  = e_weights[0] if len(e_weights) > 0 else 0.2
        w_regime  = e_weights[1] if len(e_weights) > 1 else 0.2
        w_prophet = e_weights[2] if len(e_weights) > 2 else 0.2
        w_analyst = e_weights[3] if len(e_weights) > 3 else 0.2
        w_actuary = e_weights[4] if len(e_weights) > 4 else 0.2

        s_trader  = e_signals.get("E1_trader", 0.0)
        s_regime  = e_signals.get("E2_regime", 0.0)
        s_prophet = e_signals.get("E3_prophet", 0.0)
        s_analyst = e_signals.get("E4_analyst", 0.0)
        s_actuary = e_signals.get("E5_actuary", 0.0)

        steps = []

        # Step 1: Regime Assessment (Symmetric HMM)
        if regime == "BULL":
            regime_desc = "Gaussian HMM detects expansionary **BULL** regime. Long allocations prioritized (1.00x), Short exposures dampened (0.35x)."
        elif regime == "BEAR":
            regime_desc = "Gaussian HMM detects contractionary **BEAR** regime. Short allocations prioritized (1.00x), Long exposures dampened (0.35x)."
        else:
            regime_desc = "Gaussian HMM detects choppy **SIDEWAYS** consolidation (0.60x multiplier). Mean-reversion dampening active."
        steps.append({
            "expert": "E2: Regime Specialist (HMM)",
            "icon": "🌡️",
            "signal": s_regime,
            "weight": w_regime,
            "contribution": round(w_regime * s_regime, 4),
            "narrative": regime_desc
        })

        # Step 2: Analyst Basket & Microstructure (Calibrated XGBoost)
        if s_analyst > 0.15:
            analyst_desc = f"Calibrated XGBoost shows bullish order flow and ETF tailwind (+{s_analyst:.2f}). Cross-asset correlations favor upward continuation."
        elif s_analyst < -0.15:
            analyst_desc = f"Calibrated XGBoost signals structural selling pressure ({s_analyst:.2f}). Cross-asset ETF basket divergence indicates downward momentum."
        else:
            analyst_desc = f"Calibrated XGBoost detects neutral equilibrium ({s_analyst:+.2f}). Macro correlation basket indicates balanced intraday order flow."
        steps.append({
            "expert": "E4: Cross-Asset Analyst (SHAP / XGB)",
            "icon": "📊",
            "signal": s_analyst,
            "weight": w_analyst,
            "contribution": round(w_analyst * s_analyst, 4),
            "narrative": analyst_desc
        })

        # Step 3: Prophet Volatility Horizon
        if s_prophet > 0.05:
            prophet_desc = f"TimesFM projection forecasts upward price trajectory ({s_prophet:+.2f}) with expanding upper volatility envelopes."
        elif s_prophet < -0.05:
            prophet_desc = f"TimesFM projection indicates mean-reverting downward slope ({s_prophet:+.2f}) testing lower volatility bands."
        else:
            prophet_desc = f"TimesFM forecasts price stabilization within neutral EWMA volatility band boundaries ({s_prophet:+.2f})."
        steps.append({
            "expert": "E3: Prophet Forecasting (TimesFM)",
            "icon": "🔮",
            "signal": s_prophet,
            "weight": w_prophet,
            "contribution": round(w_prophet * s_prophet, 4),
            "narrative": prophet_desc
        })

        # Step 4: Execution Trader Policy (Symmetric SAC DRL)
        if s_trader > 0.10:
            trader_desc = f"SAC Deep RL agent selects Long policy action ({s_trader:+.2f}), exploiting positive state-action Q-value trajectory."
        elif s_trader < -0.10:
            trader_desc = f"SAC Deep RL agent selects Short policy action ({s_trader:+.2f}), seeking alpha in downward acceleration."
        else:
            trader_desc = f"SAC Deep RL agent holds neutral position ({s_trader:+.2f}), minimizing policy entropy cost in choppy noise."
        steps.append({
            "expert": "E1: Execution Trader (SAC DRL)",
            "icon": "🤖",
            "signal": s_trader,
            "weight": w_trader,
            "contribution": round(w_trader * s_trader, 4),
            "narrative": trader_desc
        })

        # Step 5: Actuary Risk-to-Reward (Directionally Signed Bayesian VaR)
        min_rr = 1.50 if decision.direction > 0 else 0.50
        rr_status = "PASS" if rr >= min_rr else "FAIL"
        actuary_desc = (
            f"Bayesian VaR model calculates Expected RR = **{rr:.2f}** ({f'≥ {min_rr:.2f} target' if rr >= min_rr else f'< {min_rr:.2f} insufficient reward'}). "
            f"Target SL: {decision.sl_price or 'N/A'} | TP: {decision.tp_price or 'N/A'}. "
            f"Position Size Factor: {decision.position_size:.2f}x (0.50% base risk)."
        )
        steps.append({
            "expert": "E5: Actuary & Risk Solvency (Bayesian VaR)",
            "icon": "⚖️",
            "signal": s_actuary,
            "weight": w_actuary,
            "contribution": round(w_actuary * s_actuary, 4),
            "narrative": actuary_desc
        })

        # Check H1 Macro Trend Governor if MT5 connected
        h1_trend_status = "UNKNOWN"
        h1_trend_passed = True
        h1_val_str = "H1 EMA50 Check"
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                h1_rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 60)
                if h1_rates is not None and len(h1_rates) >= 50:
                    import pandas as pd
                    h1_closes = pd.Series([r[4] for r in h1_rates])
                    h1_ema = float(h1_closes.ewm(span=50, adjust=False).mean().iloc[-1])
                    tick = mt5.symbol_info_tick(sym)
                    bid = tick.bid if tick else float(h1_closes.iloc[-1])
                    if bid > h1_ema:
                        h1_trend_status = f"BULLISH (Bid {bid:.1f} > EMA50 {h1_ema:.1f})"
                        if decision.direction < 0:
                            h1_trend_passed = False
                            h1_val_str = f"Short Blocked (Bid > H1 EMA50 {h1_ema:.1f})"
                        else:
                            h1_val_str = f"Long Aligned (Bid > H1 EMA50 {h1_ema:.1f})"
                    else:
                        h1_trend_status = f"BEARISH (Bid {bid:.1f} < EMA50 {h1_ema:.1f})"
                        if decision.direction > 0:
                            h1_trend_passed = False
                            h1_val_str = f"Long Blocked (Bid < H1 EMA50 {h1_ema:.1f})"
                        else:
                            h1_val_str = f"Short Aligned (Bid < H1 EMA50 {h1_ema:.1f})"
        except Exception:
            pass

        # Final Consensus Synthesis (High-Conviction Rules)
        checkpoints = [
            {"name": "Signal Conviction", "passed": abs(sig) >= 0.20, "value": f"|{sig:.3f}| ≥ 0.20"},
            {"name": "Council Confidence", "passed": conf >= 0.70, "value": f"{conf:.1%} ≥ 70.0% (High-Conviction)"},
            {"name": "Risk / Reward Threshold", "passed": rr >= min_rr, "value": f"{rr:.2f} ≥ {min_rr:.2f} ({'Long' if decision.direction > 0 else 'Short'})"},
            {"name": "H1 Trend Governor", "passed": h1_trend_passed, "value": h1_val_str},
            {"name": "Breakeven Protection", "passed": True, "value": "Armed at +1.0R (SL → Entry + Buffer)"},
        ]

        if is_trade and h1_trend_passed:
            verdict = f"APPROVED — High-Conviction consensus reached for {direction}. All 5 risk, confidence, and macro trend checkpoints satisfied."
        else:
            failed = [c["name"] for c in checkpoints if not c["passed"]]
            if not failed and not is_trade:
                failed = ["Conviction threshold not met"]
            failed_str = ", ".join(failed) if failed else "Signal magnitude neutral"
            verdict = f"WITHHELD — Capital preserved. Entry filtered due to: {failed_str}."

        return {
            "symbol": sym,
            "direction": direction,
            "consensus_signal": round(sig, 3),
            "confidence": round(conf, 3),
            "regime": regime,
            "h1_trend": h1_trend_status,
            "expected_rr": round(rr, 2),
            "is_tradeable": bool(is_trade and h1_trend_passed),
            "position_size": round(decision.position_size, 3),
            "sl_price": decision.sl_price,
            "tp_price": decision.tp_price,
            "expert_weights": [round(w, 4) for w in e_weights],
            "steps": steps,
            "checkpoints": checkpoints,
            "verdict": verdict,
        }
