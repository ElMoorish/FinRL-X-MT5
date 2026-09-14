"""
Python-Side Fast Backtest Engine
================================
Simulates bar-by-bar execution with real bid/ask spreads, slippage,
broker contract sizes, and commission models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
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
    initial_volume: float = 0.0
    pnl: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    exit_reason: str = ""
    breakeven_armed: bool = False  # True once SL moved to entry + buffer
    initial_sl: float = 0.0       # Original SL for 1R distance calculation
    tp1_hit: bool = False
    tp1_pnl: float = 0.0
    peak_price: float = 0.0


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

        # Precompute 14-period True Range / ATR across bars
        tr_series = np.maximum(
            bars_df["high"] - bars_df["low"],
            np.maximum(
                (bars_df["high"] - bars_df["close"].shift(1)).abs(),
                (bars_df["low"] - bars_df["close"].shift(1)).abs()
            )
        )
        atr_series = tr_series.rolling(14).mean().bfill()

        inst_cfg = self.cfg.instrument_config.get(symbol, {})
        tp1_r = inst_cfg.get("tp1_r", self.cfg.default_tp1_r)
        tp1_ratio = inst_cfg.get("tp1_ratio", self.cfg.default_tp1_ratio)
        trail_trigger_r = inst_cfg.get("trail_trigger_r", self.cfg.default_trail_trigger_r)
        trail_atr_mult = inst_cfg.get("trail_atr_mult", self.cfg.default_trail_atr_mult)

        for i in range(warmup_bars, len(bars_df)):
            window = bars_df.iloc[max(0, i - 120):i + 1]
            bar = bars_df.iloc[i]
            bar_time = bar["time"]
            open_p = bar["open"]
            high_p = bar["high"]
            low_p = bar["low"]
            close_p = bar["close"]

            # 1. Manage Active Trade (Partial TP1, ATR Trailing Stop, SL/TP checks)
            if active_trade is not None:
                step = spec.get("lot_step", 0.01)
                min_lot = spec.get("min_lot", 0.01)

                initial_risk = abs(active_trade.entry_price - active_trade.initial_sl) if active_trade.initial_sl > 0 else (active_trade.entry_price * 0.005)

                # Update peak favorable price
                if active_trade.direction > 0:
                    active_trade.peak_price = max(active_trade.peak_price, high_p)
                    fav_reach = high_p - active_trade.entry_price
                else:
                    active_trade.peak_price = min(active_trade.peak_price, low_p) if active_trade.peak_price > 0 else low_p
                    fav_reach = active_trade.entry_price - low_p

                # ── Step A: Check Partial Take-Profit 1 (+1.25R) ──
                if not active_trade.tp1_hit and initial_risk > 0 and fav_reach >= (tp1_r * initial_risk):
                    target_tp1 = active_trade.entry_price + (tp1_r * initial_risk * active_trade.direction)
                    close_vol = round(math.floor((active_trade.volume * tp1_ratio) / step) * step, 4)
                    rem_vol = round(active_trade.volume - close_vol, 4)

                    if close_vol >= min_lot and rem_vol >= min_lot:
                        pnl_pts = (target_tp1 - active_trade.entry_price) * active_trade.direction
                        raw_pnl = pnl_pts * contract_size * close_vol
                        comm = (active_trade.entry_price + target_tp1) * close_vol * contract_size * self.commission_rate
                        net_tp1 = raw_pnl - comm

                        active_trade.tp1_pnl += net_tp1
                        balance += net_tp1
                        active_trade.volume = rem_vol
                        active_trade.tp1_hit = True

                        # Move SL to Breakeven (+1 point buffer)
                        active_trade.sl = active_trade.entry_price + (point * active_trade.direction)
                        active_trade.breakeven_armed = True
                    else:
                        # Cannot split minimum lot (e.g. 0.01 lot) -> Lock Breakeven
                        active_trade.sl = active_trade.entry_price + (point * active_trade.direction)
                        active_trade.breakeven_armed = True
                        active_trade.tp1_hit = True

                # ── Step B: Dynamic ATR Trailing Stop (Chandelier Exit) ──
                if active_trade.tp1_hit or (initial_risk > 0 and fav_reach >= trail_trigger_r * initial_risk):
                    current_atr = float(atr_series.iloc[i]) if i < len(atr_series) else (50.0 * point)
                    atr_dist = trail_atr_mult * current_atr

                    if active_trade.direction > 0:
                        candidate_sl = active_trade.peak_price - atr_dist
                        be_min = active_trade.entry_price + point
                        target_sl = max(candidate_sl, be_min)
                        if target_sl > active_trade.sl:
                            active_trade.sl = target_sl
                    else:
                        candidate_sl = active_trade.peak_price + atr_dist
                        be_max = active_trade.entry_price - point
                        target_sl = min(candidate_sl, be_max)
                        if active_trade.sl <= 0 or target_sl < active_trade.sl:
                            active_trade.sl = target_sl

                # ── Step C: Check SL or TP2 Exit on this bar ──
                hit_sl = False
                hit_tp = False
                exit_price = close_p
                exit_reason = "BAR_CLOSE"

                if active_trade.direction > 0:  # Long
                    if active_trade.sl > 0 and low_p <= active_trade.sl:
                        hit_sl = True
                        exit_price = active_trade.sl - (self.slippage_points * point)
                        exit_reason = "TRAIL_SL" if active_trade.breakeven_armed else "SL"
                    elif active_trade.tp > 0 and high_p >= active_trade.tp:
                        hit_tp = True
                        exit_price = active_trade.tp
                        exit_reason = "TP2"
                else:  # Short
                    if active_trade.sl > 0 and high_p >= active_trade.sl:
                        hit_sl = True
                        exit_price = active_trade.sl + (self.slippage_points * point)
                        exit_reason = "TRAIL_SL" if active_trade.breakeven_armed else "SL"
                    elif active_trade.tp > 0 and low_p <= active_trade.tp:
                        hit_tp = True
                        exit_price = active_trade.tp
                        exit_reason = "TP2"

                if hit_sl or hit_tp:
                    pnl_points = (exit_price - active_trade.entry_price) * active_trade.direction
                    raw_pnl = pnl_points * contract_size * active_trade.volume
                    comm = (active_trade.entry_price + exit_price) * active_trade.volume * contract_size * self.commission_rate
                    net_pnl = raw_pnl - comm

                    active_trade.exit_time = bar_time
                    active_trade.exit_price = exit_price
                    active_trade.pnl = active_trade.tp1_pnl + net_pnl
                    active_trade.exit_reason = exit_reason

                    balance += net_pnl
                    equity = balance
                    closed_trades.append(active_trade)
                    active_trade = None

            # 2. Council Evaluation
            decision: TradingDecision = council.evaluate(symbol, window)

            # 3. New Entry if tradeable and no position
            if decision.is_tradeable and active_trade is None:
                direction = decision.direction
                entry_price = close_p + ((self.slippage_points * point) * direction)

                # Compute position size (Strict fixed risk per trade based on default_risk_pct)
                risk_usd = equity * self.cfg.default_risk_pct
                sl_dist = abs(entry_price - decision.sl_price) if decision.sl_price else (entry_price * 0.01)
                vol_raw = risk_usd / (sl_dist * contract_size + 1e-8)
                step = spec.get("lot_step", 0.01)
                min_lot = spec.get("min_lot", 0.01)
                max_lot = spec.get("max_lot", 0.50)
                quantized = math.floor(vol_raw / step) * step
                if quantized < min_lot:
                    if (min_lot * sl_dist * contract_size) > risk_usd:
                        continue  # Skip entry: even min_lot exceeds 0.5% risk budget
                    vol = min_lot
                else:
                    vol = min(round(quantized, 4), max_lot)

                active_trade = SimulatedTrade(
                    entry_time=bar_time,
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    volume=vol,
                    initial_volume=vol,
                    sl=decision.sl_price or 0.0,
                    tp=decision.tp_price or 0.0,
                    initial_sl=decision.sl_price or 0.0,  # preserve original for BE calc
                    peak_price=entry_price,
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
            active_trade.pnl = active_trade.tp1_pnl + net_pnl
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
