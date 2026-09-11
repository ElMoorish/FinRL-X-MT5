"""
Python-Side Fast Backtest Engine
================================
Simulates bar-by-bar execution with real bid/ask spreads, slippage,
broker contract sizes, and commission models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config.settings import settings
from src.council.council import KDenseCouncil, TradingDecision
from src.trading.performance_analyzer import PerformanceAnalyzer, PerformanceMetrics


@dataclass
class SimulatedTrade:
    entry_time: Any
    exit_time: Optional[Any] = None
    symbol: str = ""
    direction: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    volume: float = 0.0
    pnl: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    exit_reason: str = ""
    breakeven_armed: bool = False  # True once SL moved to entry + buffer
    initial_sl: float = 0.0       # Original SL for 1R distance calculation


class BacktestEngine:
    """
    Simulates high-fidelity execution on historical bars.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        commission_rate: float = 0.0001,
        slippage_points: float = 1.0,
    ):
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_points = slippage_points
        self.cfg = settings.mt5

    def run(
        self,
        symbol: str,
        bars_df: pd.DataFrame,
        council: KDenseCouncil,
        warmup_bars: int = 100,
    ) -> tuple[PerformanceMetrics, pd.DataFrame, list[SimulatedTrade]]:
        """
        Run backtest simulation over bar dataframe.

        Returns
        -------
        tuple[PerformanceMetrics, pd.DataFrame, list[SimulatedTrade]]
            (metrics, equity_curve_df, trades)
        """
        spec = self.cfg.instrument_config.get(symbol, {"contract_size": 1.0, "point": 0.01})
        contract_size = spec.get("contract_size", 1.0)
        point = spec.get("point", 0.01)

        balance = self.initial_balance
        equity = balance
        active_trade: Optional[SimulatedTrade] = None
        closed_trades: list[SimulatedTrade] = []
        equity_records = []

        logger.info(f"🚀 Starting backtest simulation for {symbol} on {len(bars_df)} bars...")

        for i in range(warmup_bars, len(bars_df)):
            window = bars_df.iloc[max(0, i - 120):i + 1]
            bar = bars_df.iloc[i]
            bar_time = bar["time"]
            open_p = bar["open"]
            high_p = bar["high"]
            low_p = bar["low"]
            close_p = bar["close"]

            # 1. Manage Active Trade (SL / TP checks)
            if active_trade is not None:
                hit_sl = False
                hit_tp = False
                exit_price = close_p
                exit_reason = "BAR_CLOSE"

                if active_trade.direction > 0:  # Long
                    if active_trade.sl > 0 and low_p <= active_trade.sl:
                        hit_sl = True
                        exit_price = active_trade.sl - (self.slippage_points * point)
                        exit_reason = "SL"
                    elif active_trade.tp > 0 and high_p >= active_trade.tp:
                        hit_tp = True
                        exit_price = active_trade.tp
                        exit_reason = "TP"
                else:  # Short
                    if active_trade.sl > 0 and high_p >= active_trade.sl:
                        hit_sl = True
                        exit_price = active_trade.sl + (self.slippage_points * point)
                        exit_reason = "SL"
                    elif active_trade.tp > 0 and low_p <= active_trade.tp:
                        hit_tp = True
                        exit_price = active_trade.tp
                        exit_reason = "TP"

                if hit_sl or hit_tp:
                    pnl_points = (exit_price - active_trade.entry_price) * active_trade.direction
                    raw_pnl = pnl_points * contract_size * active_trade.volume
                    comm = (active_trade.entry_price + exit_price) * active_trade.volume * contract_size * self.commission_rate
                    net_pnl = raw_pnl - comm

                    active_trade.exit_time = bar_time
                    active_trade.exit_price = exit_price
                    active_trade.pnl = net_pnl
                    active_trade.exit_reason = exit_reason

                    balance += net_pnl
                    equity = balance
                    closed_trades.append(active_trade)
                    active_trade = None
                else:
                    # ── GAP-X7 Fix: Breakeven management ───────────────────────────
                    # Mirrors live executor: once floating P&L >= 1R, move SL
                    # to entry + buffer (protecting the trade from full loss).
                    if not active_trade.breakeven_armed and active_trade.initial_sl > 0:
                        initial_risk  = abs(active_trade.entry_price - active_trade.initial_sl)
                        floating_pts  = (close_p - active_trade.entry_price) * active_trade.direction
                        if floating_pts >= initial_risk:   # price moved +1R
                            be_buffer = point  # move SL to entry + 1 point buffer
                            if active_trade.direction > 0:
                                active_trade.sl = active_trade.entry_price + be_buffer
                            else:
                                active_trade.sl = active_trade.entry_price - be_buffer
                            active_trade.breakeven_armed = True
                            logger.debug(
                                f"BE armed [{active_trade.symbol}]: SL moved to "
                                f"{active_trade.sl:.2f} (+1R at {close_p:.2f})"
                            )

            # 2. Council Evaluation
            decision: TradingDecision = council.evaluate(symbol, window)

            # 3. New Entry if tradeable and no position
            if decision.is_tradeable and active_trade is None:
                direction = decision.direction
                entry_price = close_p + ((self.slippage_points * point) * direction)

                # Compute position size
                risk_usd = equity * self.cfg.default_risk_pct * decision.position_size
                sl_dist = abs(entry_price - decision.sl_price) if decision.sl_price else (entry_price * 0.01)
                vol_raw = risk_usd / (sl_dist * contract_size + 1e-8)
                step = spec.get("lot_step", 0.01)
                vol = max(spec.get("min_lot", 0.01), round(vol_raw / step) * step)

                active_trade = SimulatedTrade(
                    entry_time=bar_time,
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    volume=vol,
                    sl=decision.sl_price or 0.0,
                    tp=decision.tp_price or 0.0,
                    initial_sl=decision.sl_price or 0.0,  # preserve original for BE calc
                )

            # Calculate floating PnL if trade open
            floating = 0.0
            if active_trade is not None:
                floating = (close_p - active_trade.entry_price) * active_trade.direction * contract_size * active_trade.volume

            equity = balance + floating
            equity_records.append({
                "time": bar_time,
                "balance": balance,
                "equity": equity,
                "drawdown": balance - equity if balance > equity else 0.0,
            })

        # Close any lingering trade at backtest end
        if active_trade is not None:
            last_close = bars_df.iloc[-1]["close"]
            pnl_pts = (last_close - active_trade.entry_price) * active_trade.direction
            net_pnl = pnl_pts * contract_size * active_trade.volume
            active_trade.exit_time = bars_df.iloc[-1]["time"]
            active_trade.exit_price = last_close
            active_trade.pnl = net_pnl
            active_trade.exit_reason = "END_OF_TEST"
            balance += net_pnl
            equity = balance
            closed_trades.append(active_trade)

        equity_df = pd.DataFrame(equity_records)
        trade_pnls = [t.pnl for t in closed_trades]
        metrics = PerformanceAnalyzer.calculate_metrics(trade_pnls)

        logger.info(
            f"🏁 Backtest complete: {len(closed_trades)} trades | "
            f"Net PnL: ${metrics.net_profit:,.2f} | Sharpe: {metrics.sharpe_ratio:.2f} | "
            f"Win Rate: {metrics.win_rate:.1%}"
        )
        return metrics, equity_df, closed_trades
