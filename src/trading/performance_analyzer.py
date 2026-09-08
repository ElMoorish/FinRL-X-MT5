"""
Trading Performance Analyzer
============================
Calculates institutional-grade risk and performance metrics for both
live MT5 trade logs and backtest results:
Sharpe, Sortino, Calmar, Max Drawdown, Profit Factor, Win Rate, and Expectancy.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from loguru import logger


@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    expectancy_usd: float
    avg_win_usd: float
    avg_loss_usd: float


class PerformanceAnalyzer:
    """
    Computes performance and risk metrics from MT5 deal history or trade lists.
    """

    @classmethod
    def from_mt5_history(cls, days: int = 30) -> PerformanceMetrics:
        """Analyze trades closed in MT5 terminal within the last N days."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days)

        deals = mt5.history_deals_get(start, now)
        if not deals:
            return cls._empty_metrics()

        # Filter closing deals
        closed_trades = [
            d.profit + d.swap + d.commission
            for d in deals
            if d.entry == mt5.DEAL_ENTRY_OUT
        ]

        return cls.calculate_metrics(closed_trades)

    @classmethod
    def calculate_metrics(cls, pnl_series: list[float] | np.ndarray) -> PerformanceMetrics:
        """Compute all key ratios from an array of trade PnLs."""
        if len(pnl_series) == 0:
            return cls._empty_metrics()

        pnl = np.array(pnl_series, dtype=float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        total_trades = len(pnl)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
        net_profit = float(np.sum(pnl))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)

        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 0.0
        expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)

        # Cumulative equity curve & drawdown
        cum_pnl = np.cumsum(pnl)
        cum_max = np.maximum.accumulate(cum_pnl)
        drawdowns = cum_max - cum_pnl
        max_dd_usd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Percent drawdown (assuming $10k initial benchmark if equity curve starts at 0)
        base_equity = 10_000.0
        equity_curve = base_equity + cum_pnl
        peak_equity = np.maximum.accumulate(equity_curve)
        pct_drawdowns = (peak_equity - equity_curve) / peak_equity
        max_dd_pct = float(np.max(pct_drawdowns)) if len(pct_drawdowns) > 0 else 0.0

        # Sharpe & Sortino (assuming ~252 trades/year normalization)
        mean_pnl = np.mean(pnl)
        std_pnl = np.std(pnl, ddof=1) if len(pnl) > 1 else 1e-6
        downside_std = np.std(losses, ddof=1) if len(losses) > 1 else 1e-6

        ann_factor = np.sqrt(252)
        sharpe = float((mean_pnl / std_pnl) * ann_factor) if std_pnl > 0 else 0.0
        sortino = float((mean_pnl / downside_std) * ann_factor) if downside_std > 0 else 0.0

        # Calmar ratio
        calmar = (net_profit / max_dd_usd) if max_dd_usd > 0 else 0.0

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=net_profit,
            profit_factor=profit_factor,
            max_drawdown_usd=max_dd_usd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            expectancy_usd=expectancy,
            avg_win_usd=avg_win,
            avg_loss_usd=avg_loss,
        )

    @classmethod
    def format_report(cls, metrics: PerformanceMetrics, title: str = "Council Performance Report") -> str:
        """Generate a formatted markdown report."""
        return (
            f"### 📊 {title}\n\n"
            f"| Metric | Value |\n"
            f"|---|---|\n"
            f"| **Total Trades** | {metrics.total_trades} |\n"
            f"| **Win Rate** | {metrics.win_rate:.1%} ({metrics.winning_trades}W / {metrics.losing_trades}L) |\n"
            f"| **Profit Factor** | **{metrics.profit_factor:.2f}** |\n"
            f"| **Net Profit** | **${metrics.net_profit:,.2f}** |\n"
            f"| **Gross Profit / Loss** | +${metrics.gross_profit:,.2f} / -${metrics.gross_loss:,.2f} |\n"
            f"| **Expectancy / Trade** | ${metrics.expectancy_usd:,.2f} |\n"
            f"| **Max Drawdown ($)** | ${metrics.max_drawdown_usd:,.2f} ({metrics.max_drawdown_pct:.1%}) |\n"
            f"| **Sharpe Ratio** | **{metrics.sharpe_ratio:.2f}** |\n"
            f"| **Sortino Ratio** | {metrics.sortino_ratio:.2f} |\n"
            f"| **Calmar Ratio** | {metrics.calmar_ratio:.2f} |\n"
        )

    @staticmethod
    def _empty_metrics() -> PerformanceMetrics:
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
            gross_profit=0.0, gross_loss=0.0, net_profit=0.0, profit_factor=0.0,
            max_drawdown_usd=0.0, max_drawdown_pct=0.0, sharpe_ratio=0.0,
            sortino_ratio=0.0, calmar_ratio=0.0, expectancy_usd=0.0,
            avg_win_usd=0.0, avg_loss_usd=0.0,
        )
