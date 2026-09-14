"""
Fast Council Gating & Risk Calibrator
====================================
Precomputes bar-by-bar Council expert decisions once, then runs an exhaustive
grid search across gating parameters, risk allocations, and regime filters in
milliseconds to find the optimal live/prop-firm configuration.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from src.config.settings import settings
from src.council.council import Council, TradingDecision
from src.trading.performance_analyzer import PerformanceAnalyzer, PerformanceMetrics


@dataclass
class SimTrade:
    entry_time: Any
    exit_time: Optional[Any] = None
    direction: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    volume: float = 0.0
    pnl: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    initial_sl: float = 0.0
    breakeven_armed: bool = False
    exit_reason: str = ""


def precompute_decisions(
    symbol: str,
    bars_df: pd.DataFrame,
    cache_path: Path,
    warmup_bars: int = 100,
) -> pd.DataFrame:
    """Evaluate Council once per bar and cache the raw decision stream."""
    if cache_path.exists():
        logger.info(f"📂 Loading precomputed Council decisions from {cache_path.name}...")
        return pd.read_parquet(cache_path)

    logger.info(f"⚡ Precomputing Council decisions for {len(bars_df)} bars (this runs once)...")
    council = Council().load(symbol)
    records = []

    for i in range(warmup_bars, len(bars_df)):
        if i % 2500 == 0:
            logger.info(f"  Processed {i:,} / {len(bars_df):,} bars ({(i/len(bars_df)):.1%})...")
        window = bars_df.iloc[max(0, i - 120):i + 1]
        dec = council.evaluate(symbol, window, verbose=False)
        records.append({
            "bar_idx": i,
            "time": bars_df.iloc[i]["time"],
            "direction": dec.direction,
            "consensus_signal": dec.consensus_signal,
            "council_confidence": dec.council_confidence,
            "expected_rr": dec.expected_rr,
            "sl_price": dec.sl_price or 0.0,
            "tp_price": dec.tp_price or 0.0,
            "position_size": dec.position_size,
            "regime": dec.regime,
        })

    dec_df = pd.DataFrame(records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dec_df.to_parquet(cache_path, index=False)
    logger.info(f"💾 Precomputed decisions cached to {cache_path} ({len(dec_df):,} records)")
    return dec_df


def fast_simulate_core(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    times: list,
    dec_list: list,
    contract_size: float,
    point: float,
    step: float,
    min_lot: float,
    max_lot: float,
    min_signal: float,
    min_conf: float,
    min_rr: float,
    risk_pct: float,
    strict_regime: bool = True,
    slippage_points: float = 1.0,
    commission_rate: float = 0.0001,
    initial_balance: float = 10_000.0,
) -> Dict[str, Any]:
    """Microsecond trade simulation on pre-extracted NumPy arrays."""
    balance = initial_balance
    active_trade: Optional[SimTrade] = None
    trades: List[SimTrade] = []
    n_bars = len(highs)

    for i in range(n_bars):
        high_p = highs[i]
        low_p = lows[i]
        close_p = closes[i]
        bar_time = times[i]

        # 1. Manage Active Trade
        if active_trade is not None:
            hit_sl = False
            hit_tp = False
            exit_price = close_p
            exit_reason = "BAR_CLOSE"

            if active_trade.direction > 0:  # Long
                if active_trade.sl > 0 and low_p <= active_trade.sl:
                    hit_sl = True
                    exit_price = active_trade.sl - (slippage_points * point)
                    exit_reason = "SL"
                elif active_trade.tp > 0 and high_p >= active_trade.tp:
                    hit_tp = True
                    exit_price = active_trade.tp
                    exit_reason = "TP"
            else:  # Short
                if active_trade.sl > 0 and high_p >= active_trade.sl:
                    hit_sl = True
                    exit_price = active_trade.sl + (slippage_points * point)
                    exit_reason = "SL"
                elif active_trade.tp > 0 and low_p <= active_trade.tp:
                    hit_tp = True
                    exit_price = active_trade.tp
                    exit_reason = "TP"

            if hit_sl or hit_tp:
                pnl_pts = (exit_price - active_trade.entry_price) * active_trade.direction
                raw_pnl = pnl_pts * contract_size * active_trade.volume
                comm = (active_trade.entry_price + exit_price) * active_trade.volume * contract_size * commission_rate
                net_pnl = raw_pnl - comm
                active_trade.exit_time = bar_time
                active_trade.exit_price = exit_price
                active_trade.pnl = net_pnl
                active_trade.exit_reason = exit_reason
                balance += net_pnl
                trades.append(active_trade)
                active_trade = None
            else:
                # Breakeven check (+1.0R move)
                if not active_trade.breakeven_armed and active_trade.initial_sl > 0:
                    init_risk = abs(active_trade.entry_price - active_trade.initial_sl)
                    float_pts = (close_p - active_trade.entry_price) * active_trade.direction
                    if float_pts >= init_risk:
                        if active_trade.direction > 0:
                            active_trade.sl = active_trade.entry_price + point
                        else:
                            active_trade.sl = active_trade.entry_price - point
                        active_trade.breakeven_armed = True

        # 2. Check for New Entry
        dec = dec_list[i]
        if active_trade is None and dec is not None:
            # Gating logic
            is_valid = (
                abs(dec.consensus_signal) >= min_signal
                and dec.council_confidence >= min_conf
                and dec.expected_rr >= min_rr
                and dec.direction != 0
            )

            # Strict regime filter: no longs in Bear, no shorts in Bull
            if is_valid and strict_regime:
                if dec.regime == "BEAR" and dec.direction > 0:
                    is_valid = False
                elif dec.regime == "BULL" and dec.direction < 0:
                    is_valid = False

            if is_valid:
                direction = dec.direction
                entry_price = close_p + ((slippage_points * point) * direction)
                risk_usd = balance * risk_pct * dec.position_size
                sl_dist = abs(entry_price - dec.sl_price) if dec.sl_price > 0 else (entry_price * 0.01)
                vol_raw = risk_usd / (sl_dist * contract_size + 1e-8)
                quantized = math.floor(vol_raw / step) * step
                if quantized < min_lot:
                    if (min_lot * sl_dist * contract_size) > risk_usd:
                        continue
                    vol = min_lot
                else:
                    vol = min(round(quantized, 4), max_lot)

                active_trade = SimTrade(
                    entry_time=bar_time,
                    direction=direction,
                    entry_price=entry_price,
                    volume=vol,
                    sl=dec.sl_price,
                    tp=dec.tp_price,
                    initial_sl=dec.sl_price,
                )

    # Close lingering trade
    if active_trade is not None:
        last_close = closes[-1]
        pnl_pts = (last_close - active_trade.entry_price) * active_trade.direction
        raw_pnl = pnl_pts * contract_size * active_trade.volume
        comm = (active_trade.entry_price + last_close) * active_trade.volume * contract_size * commission_rate
        net_pnl = raw_pnl - comm
        active_trade.exit_time = times[-1]
        active_trade.exit_price = last_close
        active_trade.pnl = net_pnl
        active_trade.exit_reason = "END_OF_TEST"
        trades.append(active_trade)

    pnls = [t.pnl for t in trades]
    metrics = PerformanceAnalyzer.calculate_metrics(pnls)
    return {
        "min_signal": min_signal,
        "min_conf": min_conf,
        "min_rr": min_rr,
        "risk_pct": risk_pct,
        "strict_regime": strict_regime,
        "total_trades": metrics.total_trades,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "net_profit": metrics.net_profit,
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "max_drawdown_usd": metrics.max_drawdown_usd,
        "sharpe_ratio": metrics.sharpe_ratio,
        "expectancy": metrics.expectancy_usd,
        "prop_safe": metrics.max_drawdown_pct <= 0.035,  # ≤ 3.5% DD safely below 5.0% limit
    }


def run_calibration():
    symbol = "BTCUSD.x"
    start_dt = datetime(2026, 6, 1, tzinfo=timezone.utc)
    end_dt = datetime(2026, 9, 12, tzinfo=timezone.utc)

    # 1. Load Parquet Data
    parquet_path = Path("data/cache/parquet/BTCUSD.x_M5_2021-11-06_2026-09-12.parquet")
    logger.info(f"Loading M5 feature store from {parquet_path}...")
    full_df = pl.read_parquet(parquet_path)
    slice_df = full_df.filter((pl.col("time") >= start_dt) & (pl.col("time") <= end_dt))
    bars_df = slice_df.to_pandas()
    logger.info(f"Loaded {len(bars_df):,} bars for calibration window ({start_dt.date()} → {end_dt.date()})")

    # 2. Precompute decisions stream
    cache_path = Path("data/cache/btcusd_precomputed_decisions_2026-06-01_2026-09-12.parquet")
    dec_df = precompute_decisions(symbol, bars_df, cache_path)

    # Extract high-speed NumPy arrays and direct-index list
    highs = bars_df["high"].to_numpy(dtype=float)
    lows = bars_df["low"].to_numpy(dtype=float)
    closes = bars_df["close"].to_numpy(dtype=float)
    times = bars_df["time"].tolist()

    dec_list = [None] * len(bars_df)
    for row in dec_df.itertuples(index=False):
        if row.bar_idx < len(dec_list):
            dec_list[row.bar_idx] = row

    spec = settings.mt5.instrument_config.get(symbol, {"contract_size": 1.0, "point": 0.01})
    contract_size = spec.get("contract_size", 1.0)
    point = spec.get("point", 0.01)
    step = spec.get("lot_step", 0.01)
    min_lot = spec.get("min_lot", 0.01)
    max_lot = spec.get("max_lot", 0.50)

    # 3. Parameter Grid Search
    signal_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]
    conf_thresholds   = [0.70, 0.75, 0.80]
    rr_thresholds     = [1.25, 1.50, 1.75, 2.00]
    risk_pcts         = [0.0025, 0.0035, 0.0050]
    regime_filters    = [True, False]

    logger.info(f"🔍 Running grid search over {len(signal_thresholds)*len(conf_thresholds)*len(rr_thresholds)*len(risk_pcts)*len(regime_filters)} combinations...")
    results = []

    for min_sig in signal_thresholds:
        for min_conf in conf_thresholds:
            for min_rr in rr_thresholds:
                for risk_pct in risk_pcts:
                    for strict_regime in regime_filters:
                        res = fast_simulate_core(
                            highs=highs,
                            lows=lows,
                            closes=closes,
                            times=times,
                            dec_list=dec_list,
                            contract_size=contract_size,
                            point=point,
                            step=step,
                            min_lot=min_lot,
                            max_lot=max_lot,
                            min_signal=min_sig,
                            min_conf=min_conf,
                            min_rr=min_rr,
                            risk_pct=risk_pct,
                            strict_regime=strict_regime,
                        )
                        if res["total_trades"] >= 15:  # Statistically meaningful
                            results.append(res)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.error("No parameter combinations generated sufficient trades")
        return

    # Filter for Prop-Safe & Positive Expectancy
    safe_profitable = results_df[
        (results_df["prop_safe"] == True) &
        (results_df["profit_factor"] > 1.0) &
        (results_df["net_profit"] > 0)
    ].sort_values(by=["profit_factor", "sharpe_ratio", "net_profit"], ascending=False)

    print("\n" + "="*95)
    print("TOP 10 PROP-FIRM COMPLIANT & PROFITABLE CONFIGURATIONS (Max DD <= 3.5%, Net > $0)")
    print("="*95)
    if not safe_profitable.empty:
        cols_show = [
            "min_signal", "min_conf", "min_rr", "risk_pct", "strict_regime",
            "total_trades", "win_rate", "profit_factor", "net_profit", "max_drawdown_pct", "sharpe_ratio"
        ]
        print(safe_profitable[cols_show].head(10).to_string(index=False))
    else:
        print("No combinations achieved Profit Factor > 1.0 with Max DD <= 3.5%. Showing Top 10 by Profit Factor overall:")
        best_overall = results_df.sort_values(by="profit_factor", ascending=False).head(10)
        cols_show = [
            "min_signal", "min_conf", "min_rr", "risk_pct", "strict_regime",
            "total_trades", "win_rate", "profit_factor", "net_profit", "max_drawdown_pct", "sharpe_ratio"
        ]
        print(best_overall[cols_show].to_string(index=False))
    print("="*95 + "\n")

    # Save all results to CSV for record
    results_csv = Path("data/cache/btcusd_calibration_grid_results.csv")
    results_df.to_csv(results_csv, index=False)
    logger.info(f"📊 Full grid search saved to {results_csv}")


if __name__ == "__main__":
    run_calibration()
