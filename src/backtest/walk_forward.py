"""
Walk-Forward Optimization and Out-of-Sample Validator
=====================================================
Performs rolling walk-forward validation (e.g., 12-month train / 3-month test)
to verify Council gating resilience and protect against data snooping bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config.settings import settings
from src.council.council import KDenseCouncil
from src.backtest.backtest_engine import BacktestEngine, SimulatedTrade
from src.trading.performance_analyzer import PerformanceAnalyzer, PerformanceMetrics


@dataclass
class WalkForwardWindow:
    window_id: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    out_of_sample_pnl: float
    out_of_sample_trades: int


class WalkForwardValidator:
    """
    Executes walk-forward cross-validation across rolling market regimes.
    """

    def __init__(
        self,
        train_bars: int = 15_000,   # ~50-60 trading days on M5
        test_bars: int = 4_000,     # ~14 trading days on M5
        step_bars: int = 4_000,
    ):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.engine = BacktestEngine()

    def run_validation(
        self,
        symbol: str,
        bars_df: pd.DataFrame,
    ) -> tuple[PerformanceMetrics, list[WalkForwardWindow], pd.DataFrame]:
        """
        Execute walk-forward cycles.

        Returns
        -------
        tuple[PerformanceMetrics, list[WalkForwardWindow], pd.DataFrame]
            (combined_oos_metrics, window_records, combined_oos_equity)
        """
        total_bars = len(bars_df)
        if total_bars < (self.train_bars + self.test_bars):
            raise ValueError(
                f"Dataframe too short for walk-forward: {total_bars} bars "
                f"< required {self.train_bars + self.test_bars}"
            )

        windows: list[WalkForwardWindow] = []
        all_oos_trades: list[SimulatedTrade] = []
        oos_equity_segments = []

        start_idx = 0
        window_id = 1

        logger.info(
            f"🔄 Starting Walk-Forward Validation for {symbol} | "
            f"Train: {self.train_bars} bars | Test: {self.test_bars} bars | Step: {self.step_bars} bars"
        )

        while (start_idx + self.train_bars + self.test_bars) <= total_bars:
            train_start_idx = start_idx
            train_end_idx   = start_idx + self.train_bars
            test_start_idx  = train_end_idx
            test_end_idx    = min(total_bars, test_start_idx + self.test_bars)

            train_df = bars_df.iloc[train_start_idx:train_end_idx].copy()
            test_df  = bars_df.iloc[test_start_idx:test_end_idx].copy()

            # 1. Train / Fit Council on In-Sample
            council = KDenseCouncil()
            logger.info(f"Window {window_id}: Training Council on IS window ({len(train_df)} bars)...")
            council.fit(symbol, train_df)

            # 2. Evaluate In-Sample
            is_metrics, _, _ = self.engine.run(symbol, train_df, council)

            # 3. Evaluate Out-of-Sample (unseen data)
            logger.info(f"Window {window_id}: Evaluating Out-of-Sample ({len(test_df)} bars)...")
            oos_metrics, oos_equity, oos_trades = self.engine.run(symbol, test_df, council)

            windows.append(
                WalkForwardWindow(
                    window_id=window_id,
                    train_start=train_df.iloc[0]["time"],
                    train_end=train_df.iloc[-1]["time"],
                    test_start=test_df.iloc[0]["time"],
                    test_end=test_df.iloc[-1]["time"],
                    in_sample_sharpe=is_metrics.sharpe_ratio,
                    out_of_sample_sharpe=oos_metrics.sharpe_ratio,
                    out_of_sample_pnl=oos_metrics.net_profit,
                    out_of_sample_trades=oos_metrics.total_trades,
                )
            )

            all_oos_trades.extend(oos_trades)
            oos_equity_segments.append(oos_equity)

            start_idx += self.step_bars
            window_id += 1

        # Combine all out-of-sample trades
        combined_pnls = [t.pnl for t in all_oos_trades]
        combined_oos_metrics = PerformanceAnalyzer.calculate_metrics(combined_pnls)

        combined_oos_equity = (
            pd.concat(oos_equity_segments, ignore_index=True)
            if oos_equity_segments
            else pd.DataFrame()
        )

        logger.info(
            f"✅ Walk-Forward Complete ({len(windows)} windows) | "
            f"Combined OOS Net PnL: ${combined_oos_metrics.net_profit:,.2f} | "
            f"OOS Sharpe: {combined_oos_metrics.sharpe_ratio:.2f} | "
            f"OOS Win Rate: {combined_oos_metrics.win_rate:.1%}"
        )

        return combined_oos_metrics, windows, combined_oos_equity
