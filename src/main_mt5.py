"""
FinRL-X-MT5 Main Entry Point
==============================
CLI for training, backtesting, and live trading with the K-Dense Council.

Usage:
    # Train Council for NAS100
    python -m src.main_mt5 train --symbol USTEC --days 365

    # Backtest (Python-side fast iteration)
    python -m src.main_mt5 backtest --symbol USTEC --start 2023-01-01 --end 2025-01-01

    # Live trading (paper or real)
    python -m src.main_mt5 live --symbols USTEC USOIL XAUUSD US30

    # Check system before TimesFM
    python .agents/skills/timesfm-forecasting/scripts/check_system.py
"""

from __future__ import annotations

import sys
import time
import signal
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger

# ─── Configure logging ────────────────────────────────────────────────────────
from src.config.settings import settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level=settings.log_level,
    colorize=True,
)
logger.add(
    settings.log_dir / "finrl_x_mt5_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="1 day",
    retention="30 days",
    encoding="utf-8",
)


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_train(args):
    """Train the K-Dense Council for the specified symbol(s)."""
    from src.data.mt5_tick_fetcher   import MT5TickFetcher
    from src.data.correlation_fuser  import CorrelationFuser
    from src.council.council         import Council

    if getattr(args, "all_symbols", False):
        symbols = settings.mt5.symbols
    elif hasattr(args, "symbols") and args.symbols:
        symbols = args.symbols
    else:
        symbols = [args.symbol]

    days      = args.days
    timesteps = getattr(args, "timesteps", 50000)
    date_to   = datetime.now()
    date_from = date_to - timedelta(days=days)

    logger.info(f"🏛️ Convening Council Training Batch for {len(symbols)} symbols: {symbols}")
    logger.info(f"   Period: {date_from.date()} → {date_to.date()} ({days} days) | DRL steps: {timesteps:,}")

    with MT5TickFetcher() as fetcher:
        fuser = CorrelationFuser(tick_fetcher=fetcher)
        for idx, symbol in enumerate(symbols, 1):
            logger.info(f"\n[{idx}/{len(symbols)}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"🏛️ Building features & training Council for {symbol}...")
            try:
                features = fuser.build_feature_store(symbol, date_from, date_to)
                if features.is_empty():
                    logger.warning(f"Feature store is empty for {symbol} — skipping")
                    continue

                logger.info(f"Feature store: {len(features):,} bars × {len(features.columns)} features")

                council = Council()
                council.train(features, symbol=symbol, timesteps=timesteps)
                logger.info(f"✅ Council trained and saved for {symbol}")
            except Exception as e:
                logger.error(f"Failed training for {symbol}: {e}")

    logger.info("\n🎉 All requested symbols processed!")


def cmd_live(args):
    """Run live/paper trading with the K-Dense Council."""
    from src.data.mt5_tick_fetcher   import MT5TickFetcher
    from src.data.correlation_fuser  import CorrelationFuser
    from src.council.council         import Council
    from src.trading.mt5_executor    import MT5Executor

    symbols = args.symbols or settings.mt5.symbols
    councils = {}
    executor = MT5Executor()

    # Load trained councils
    with MT5TickFetcher() as fetcher:
        for symbol in symbols:
            try:
                council = Council().load(symbol)
                councils[symbol] = council
                logger.info(f"✅ Loaded Council for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load Council for {symbol}: {e}")

    if not councils:
        logger.error("No councils loaded — run 'train' first")
        sys.exit(1)

    # Graceful shutdown handler
    _running = [True]
    def _stop(sig, frame):
        logger.info("⏹  Stop signal received — shutting down...")
        _running[0] = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    logger.info(
        f"🚀 Council live trading started | "
        f"Symbols: {symbols} | "
        f"Timeframe: M{settings.mt5.timeframe_minutes}"
    )

    from src.trading.recap_generator import RecapGenerator
    recap_gen = RecapGenerator()
    last_daily_recap_date = ""
    last_weekly_recap_week = ""

    with MT5TickFetcher() as fetcher:
        fuser = CorrelationFuser(tick_fetcher=fetcher)

        while _running[0]:
            for symbol, council in councils.items():
                try:
                    # Manage open positions (Breakeven +1.0R lock)
                    executor.manage_active_positions(symbol)

                    # Fetch latest features
                    date_to   = datetime.now()
                    date_from = date_to - timedelta(hours=48)  # 2 days of M5 bars
                    features  = fuser.build_feature_store(symbol, date_from, date_to)

                    if features.is_empty():
                        continue

                    # Get account equity
                    acc    = fetcher.get_account_state()
                    equity = acc.get("equity", 10_000.0)

                    # Council decision
                    decision = council.decide(features, symbol)

                    # Manage active positions (Dynamic Breakeven at +1.0R)
                    try:
                        executor.manage_active_positions(symbol)
                    except Exception as be_err:
                        logger.debug(f"Breakeven management error: {be_err}")

                    # Persist live state for real-time dashboard synchronization
                    try:
                        from src.dashboard.cot_generator import ChainOfThoughtGenerator
                        cot = ChainOfThoughtGenerator.generate(decision)
                        cot["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                        cot["source"] = "LIVE_TRADER"
                        state_path = Path("data") / f"live_council_state_{symbol}.json"
                        state_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(state_path, "w", encoding="utf-8") as f:
                            json.dump(cot, f, indent=2)
                    except Exception as cot_err:
                        logger.debug(f"Dashboard state sync error: {cot_err}")

                    # Execute if tradeable
                    if decision.is_tradeable:
                        executor.execute_decision(decision, equity)

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            # ─── Automated Daily & Weekly Performance Recap ────────────────────────
            now_utc = datetime.now(timezone.utc)
            today_str = now_utc.strftime("%Y-%m-%d")
            week_str = now_utc.strftime("%Y_W%W")

            # Daily Recap at 23:55 UTC
            if now_utc.hour == 23 and now_utc.minute >= 50 and today_str != last_daily_recap_date:
                try:
                    logger.info("📊 Generating and dispatching automated End-of-Day Performance Recap...")
                    recap_gen.generate_daily_recap()
                    last_daily_recap_date = today_str
                except Exception as e:
                    logger.warning(f"Automated daily recap failed: {e}")

            # Weekly Recap at Friday 21:55 UTC (market close)
            if now_utc.weekday() == 4 and now_utc.hour == 21 and now_utc.minute >= 50 and week_str != last_weekly_recap_week:
                try:
                    logger.info("📊 Generating and dispatching automated Weekly Performance Recap...")
                    recap_gen.generate_weekly_recap()
                    last_weekly_recap_week = week_str
                except Exception as e:
                    logger.warning(f"Automated weekly recap failed: {e}")

            # Sleep precisely until the next M5 bar close (plus 1.5s buffer)
            now = datetime.now()
            tf_min = settings.mt5.timeframe_minutes
            sec_into_bar = (now.minute % tf_min) * 60 + now.second + (now.microsecond / 1_000_000.0)
            sleep_sec = max(2.0, (tf_min * 60) - sec_into_bar + 1.5)
            logger.info(f"💤 Sleeping {sleep_sec:.1f}s until next M{tf_min} bar close...")
            time.sleep(sleep_sec)

    logger.info("Council live trading stopped.")


def cmd_backtest(args):
    """Run Python-side institutional backtest simulation."""
    from src.data.mt5_tick_fetcher   import MT5TickFetcher
    from src.data.correlation_fuser  import CorrelationFuser
    from src.council.council         import Council
    from src.backtest.backtest_engine import BacktestEngine
    from src.trading.performance_analyzer import PerformanceAnalyzer

    if getattr(args, "all_symbols", False):
        symbols = settings.mt5.symbols
    elif getattr(args, "symbols", None):
        symbols = args.symbols
    else:
        symbols = [args.symbol]

    if getattr(args, "days", None):
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=args.days)
    else:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt   = datetime.strptime(args.end,   "%Y-%m-%d")

    logger.info(f"📊 Backtesting universe ({len(symbols)} symbols) | {start_dt.date()} → {end_dt.date()}")
    logger.info("💡 For MT5 real-tick backtest, use the MQL5 EA in mql5/FinRL_X_MT5.mq5")

    results = []
    with MT5TickFetcher() as fetcher:
        fuser = CorrelationFuser(tick_fetcher=fetcher)
        for sym in symbols:
            try:
                features = fuser.build_feature_store(sym, start_dt, end_dt)
                if features.is_empty():
                    logger.warning(f"No feature data retrieved for {sym}")
                    continue

                council = Council().load(sym)
                engine = BacktestEngine(initial_balance=10_000.0)
                features_pd = features.to_pandas()

                metrics, equity_df, trades = engine.run(sym, features_pd, council)
                report = PerformanceAnalyzer.format_report(metrics, f"Backtest Results — {sym}")
                logger.info("\n" + report)

                results.append({
                    "Symbol": sym,
                    "Trades": metrics.total_trades,
                    "Win Rate": f"{metrics.win_rate:.1%}",
                    "Profit Factor": f"{metrics.profit_factor:.2f}",
                    "Net Profit": f"${metrics.net_profit:,.2f}",
                    "Max DD": f"{metrics.max_drawdown_pct:.1%}",
                    "Sharpe": f"{metrics.sharpe_ratio:.2f}",
                    "Sortino": f"{metrics.sortino_ratio:.2f}",
                })
            except Exception as e:
                logger.error(f"Failed backtesting {sym}: {e}")

    if len(results) > 1:
        import pandas as pd
        summary_df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("🏛️ K-DENSE COUNCIL UNIVERSE BACKTEST SUMMARY")
        print("="*80)
        print(summary_df.to_markdown(index=False))
        print("="*80 + "\n")


def shift(arr):
    result = arr.copy()
    result[1:] = arr[:-1]
    result[0]  = 0
    return result


def max_drawdown(pnl):
    equity = (1 + pnl).cumprod()
    peak   = __import__("numpy").maximum.accumulate(equity)
    return ((peak - equity) / (peak + 1e-8)).max()


def cmd_export_signals(args):
    """Generate real-tick signals CSV and deploy to MT5 Strategy Tester."""
    from src.backtest.mt5_tick_backtest import MT5TickBacktestBridge

    symbol = args.symbol
    days = args.days
    output = args.output

    logger.info(f"🏛️ Preparing MT5 Strategy Tester signals for {symbol} ({days} days)...")
    bridge = MT5TickBacktestBridge()
    csv_path = bridge.generate_strategy_tester_signals(symbol=symbol, days=days, output_filename=output)
    logger.info(f"✅ Strategy Tester signals generated and deployed: {csv_path}")
    logger.info("👉 You can now run FinRL_X_MT5.mq5 in MT5 Strategy Tester under 'Every tick based on real ticks'!")


def cmd_walk_forward(args):
    """Run rolling walk-forward validation."""
    from src.data.mt5_tick_fetcher import MT5TickFetcher
    from src.data.correlation_fuser import CorrelationFuser
    from src.backtest.walk_forward import WalkForwardValidator
    from src.trading.performance_analyzer import PerformanceAnalyzer

    symbol = args.symbol
    days = args.days
    date_to = datetime.now()
    date_from = date_to - timedelta(days=days)

    logger.info(f"🔄 Walk-Forward Validation for {symbol} ({days} days)...")
    with MT5TickFetcher() as fetcher:
        fuser = CorrelationFuser(tick_fetcher=fetcher)
        features = fuser.build_feature_store(symbol, date_from, date_to)

    if features.is_empty():
        logger.error("No features generated for walk-forward")
        return

    features_pd = features.to_pandas()
    validator = WalkForwardValidator(train_bars=args.train_bars, test_bars=args.test_bars)
    oos_metrics, windows, _ = validator.run_validation(symbol, features_pd)

    report = PerformanceAnalyzer.format_report(oos_metrics, f"Walk-Forward Results — {symbol}")
    logger.info("\n" + report)


def cmd_dashboard(args):
    """Launch the FinRL-X-MT5 interactive localhost dashboard."""
    import uvicorn
    from src.dashboard.app import app
    port = getattr(args, "port", 8000)
    host = getattr(args, "host", "127.0.0.1")
    logger.info(f"🏛️ Starting FinRL-X-MT5 Council Terminal on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def cmd_recap(args):
    """Generate and dispatch automated Daily or Weekly Performance Recap."""
    from src.trading.recap_generator import RecapGenerator
    gen = RecapGenerator()

    if getattr(args, "weekly", False):
        logger.info("📊 Generating Weekly Performance Recap...")
        res = gen.generate_weekly_recap(weeks_back=args.weeks, symbol=args.symbol, dispatch=not args.no_dispatch)
        logger.info(f"Weekly Recap complete: {res['total_trades']} trades | Net PnL: ${res['net_pnl']:,.2f} | Win Rate: {res['win_rate']:.1f}%")
    else:
        target_date = datetime.now(timezone.utc) - timedelta(days=args.days) if args.days > 0 else None
        logger.info(f"📊 Generating Daily Performance Recap for {'today' if not target_date else target_date.date()}...")
        res = gen.generate_daily_recap(date=target_date, symbol=args.symbol, dispatch=not args.no_dispatch)
        logger.info(f"Daily Recap complete: {res['total_trades']} trades | Net PnL: ${res['net_pnl']:,.2f} | Win Rate: {res['win_rate']:.1f}%")


# ─── CLI Parser ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="finrl-x-mt5",
        description="K-Dense Council — MoE Trading System for MT5 Indices & Commodities",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train the K-Dense Council")
    p_train.add_argument("--symbol", default="NAS100.x", help="Single MT5 symbol (e.g. NAS100.x, WTI.x, XAGUSD.x)")
    p_train.add_argument("--symbols", nargs="+", help="Multiple MT5 symbols to train in sequence")
    p_train.add_argument("--all-symbols", action="store_true", help="Train all symbols configured in settings")
    p_train.add_argument("--days",   type=int, default=90, help="Training history in days")
    p_train.add_argument("--timesteps", type=int, default=50000, help="DRL timesteps per instrument")

    # live
    p_live = sub.add_parser("live", help="Live/paper trading")
    p_live.add_argument("--symbols", nargs="+", default=["NAS100.x"], help="MT5 symbols to trade (default: NAS100.x)")

    # recap
    p_recap = sub.add_parser("recap", help="Generate and dispatch Daily or Weekly performance recap")
    p_recap.add_argument("--daily", action="store_true", help="Generate today's Daily Recap")
    p_recap.add_argument("--weekly", action="store_true", help="Generate current Weekly Recap")
    p_recap.add_argument("--days", type=int, default=0, help="Days back for Daily Recap (0=today)")
    p_recap.add_argument("--weeks", type=int, default=0, help="Weeks back for Weekly Recap (0=current week)")
    p_recap.add_argument("--symbol", default=None, help="Filter by specific symbol (e.g. NAS100.x)")
    p_recap.add_argument("--no-dispatch", action="store_true", help="Preview only, do not send to Telegram/Discord")

    # backtest
    p_bt = sub.add_parser("backtest", help="Python-side institutional backtest")
    p_bt.add_argument("--symbol", default="NAS100.x", help="Single symbol to backtest")
    p_bt.add_argument("--symbols", nargs="+", help="Multiple symbols to backtest in sequence")
    p_bt.add_argument("--all-symbols", action="store_true", help="Backtest all symbols in configured universe")
    p_bt.add_argument("--days",   type=int, default=None, help="Days of history to backtest")
    p_bt.add_argument("--start",  default="2026-07-01")
    p_bt.add_argument("--end",    default="2026-09-08")

    # export-signals (MT5 Strategy Tester)
    p_exp = sub.add_parser("export-signals", help="Generate & deploy signals for MT5 Strategy Tester (Every tick based on real ticks)")
    p_exp.add_argument("--symbol", default="NAS100.x", help="Symbol for Strategy Tester")
    p_exp.add_argument("--days",   type=int, default=60, help="History lookback in days")
    p_exp.add_argument("--output", default="finrl_x_signals.csv", help="Output filename")

    # walk-forward
    p_wf = sub.add_parser("walk-forward", help="Run rolling walk-forward validation")
    p_wf.add_argument("--symbol", default="NAS100.x")
    p_wf.add_argument("--days", type=int, default=180)
    p_wf.add_argument("--train-bars", type=int, default=12000)
    p_wf.add_argument("--test-bars", type=int, default=3000)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch interactive localhost terminal dashboard")
    p_dash.add_argument("--port", type=int, default=8000, help="Port to run dashboard on (default: 8000)")
    p_dash.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "live":
        cmd_live(args)
    elif args.command == "recap":
        cmd_recap(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "export-signals":
        cmd_export_signals(args)
    elif args.command == "walk-forward":
        cmd_walk_forward(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)


if __name__ == "__main__":
    main()
