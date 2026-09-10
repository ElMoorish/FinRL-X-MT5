"""
MT5 Tick Backtest & Strategy Tester Bridge
==========================================
Generates aligned Council signal CSVs for MT5 Strategy Tester
in 'Every tick based on real ticks' mode.

Automatically copies signals to MT5's MQL5/Files directory so
the FinRL_X_MT5.mq5 EA can be backtested directly.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import polars as pl
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from loguru import logger

from src.config.settings import settings
from src.data.mt5_tick_fetcher import MT5TickFetcher
from src.data.tick_feature_engineer import TickFeatureEngineer
from src.data.yahoo_fetcher import YahooFetcher
from src.data.correlation_fuser import CorrelationFuser
from src.council.council import KDenseCouncil, TradingDecision


class MT5TickBacktestBridge:
    """
    Orchestrates real-tick data pulling, feature fusion, Council inference,
    and export into the MQL5 Strategy Tester format.
    """

    def __init__(self, council: Optional[KDenseCouncil] = None):
        self.council = council or KDenseCouncil()
        self.fetcher = MT5TickFetcher()
        self.feature_eng = TickFeatureEngineer()
        self.yahoo = YahooFetcher()
        self.fuser = CorrelationFuser()

    def generate_strategy_tester_signals(
        self,
        symbol: str = "NAS100.x",
        days: int = 60,
        output_filename: str = "finrl_x_signals.csv",
    ) -> Path:
        """
        Pull real MT5 historical data, run Council, and produce the CSV
        needed by FinRL_X_MT5.mq5 in MT5 Strategy Tester.

        Returns
        -------
        Path
            Path to the generated CSV.
        """
        logger.info(f"🏛️ Generating Strategy Tester signals for {symbol} (past {days} days)...")

        # 1. Fetch M5 bars from MT5
        n_bars = days * 24 * 12  # ~288 M5 bars/day
        ohlcv_pl = self.fetcher.get_ohlcv(symbol, timeframe=mt5.TIMEFRAME_M5, n_bars=n_bars)

        if ohlcv_pl.is_empty():
            raise RuntimeError(f"No OHLCV data retrieved for {symbol}")

        logger.info(f"Retrieved {len(ohlcv_pl)} M5 bars for {symbol}")

        # 2. Build tick / technical features
        bars_with_feat = self.feature_eng.compute_features(ohlcv_pl)

        # 3. Fetch Yahoo correlation features & fuse
        try:
            yahoo_df = self.yahoo.get_correlation_features(symbol, days=days + 30)
            fused_df = self.fuser.fuse(bars_with_feat, yahoo_df)
        except Exception as e:
            logger.warning(f"Could not fuse Yahoo features ({e}). Using technical features only.")
            fused_df = bars_with_feat

        # 4. Generate Council signals
        signals_records = []
        df_pd = fused_df.to_pandas()
        # Ensure Council is loaded from saved model files if available
        if not getattr(self.council, "_fitted", False):
            try:
                self.council.load(symbol)
                logger.info(f"✅ Loaded trained Council models for {symbol}")
            except Exception as e:
                logger.warning(f"Could not load pre-trained Council for {symbol} ({e}) — evaluating with default policy")

        # Iterate over bars (start after warm-up lookback)
        warmup = min(100, len(df_pd) // 4)
        logger.info(f"Running Council evaluation across {len(df_pd) - warmup} bars...")

        for i in range(warmup, len(df_pd) - 1):
            window = df_pd.iloc[max(0, i - 120):i + 1]
            current_bar = df_pd.iloc[i]
            # Anti-lookahead: signal evaluated on bar i is executed at the open of bar i+1
            next_bar = df_pd.iloc[i + 1]
            exec_time = next_bar["time"]

            # Format datetime for MT5: YYYY.MM.DD HH:MM:SS
            if isinstance(exec_time, str):
                dt_obj = pd.to_datetime(exec_time)
            elif isinstance(exec_time, (int, float)):
                dt_obj = datetime.fromtimestamp(exec_time, tz=timezone.utc)
            else:
                dt_obj = exec_time

            time_str = dt_obj.strftime("%Y.%m.%d %H:%M:%S")

            # Evaluate through Council on completed bar window
            decision: TradingDecision = self.council.evaluate(symbol, window)

            signals_records.append({
                "time": time_str,
                "consensus_signal": round(decision.consensus_signal, 4),
                "direction": decision.direction,
                "position_size": round(decision.position_size, 4),
                "tp": round(decision.tp_price, 4) if decision.tp_price else 0.0,
                "sl": round(decision.sl_price, 4) if decision.sl_price else 0.0,
                "rr": round(decision.expected_rr, 2),
                "conf": round(decision.confidence, 4),
                "regime": decision.regime,
            })

        signals_df = pd.DataFrame(signals_records)

        # 5. Save locally in project data dir
        local_path = settings.data.data_dir / output_filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        signals_df.to_csv(local_path, index=False)
        logger.info(f"✅ Local signal CSV saved: {local_path} ({len(signals_df)} rows)")

        # 6. Copy to MT5 terminal Common and Data files directories
        self._deploy_to_mt5_folders(local_path, output_filename)

        return local_path

    def _deploy_to_mt5_folders(self, source_csv: Path, filename: str) -> None:
        """Copy CSV directly to MT5 terminal folders for seamless testing."""
        info = mt5.terminal_info()
        if not info:
            logger.warning("Could not query MT5 terminal info for file deployment")
            return

        deployed_destinations = []

        # Terminal Data Path (MQL5\Files\finrl_x_mt5 and MQL5\Files)
        if hasattr(info, "data_path") and info.data_path:
            terminal_files = Path(info.data_path) / "MQL5" / "Files" / "finrl_x_mt5"
            terminal_files.mkdir(parents=True, exist_ok=True)
            dest = terminal_files / filename
            with open(source_csv, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())
            deployed_destinations.append(str(dest))

            # Also deploy directly to root MQL5\Files for direct Strategy Tester access
            dest_root = Path(info.data_path) / "MQL5" / "Files" / filename
            with open(source_csv, "rb") as src, open(dest_root, "wb") as dst:
                dst.write(src.read())
            deployed_destinations.append(str(dest_root))

        # Terminal Common Path (Common\Files\finrl_x_mt5 and Common\Files)
        if hasattr(info, "commondata_path") and info.commondata_path:
            common_files = Path(info.commondata_path) / "Files" / "finrl_x_mt5"
            common_files.mkdir(parents=True, exist_ok=True)
            dest_common = common_files / filename
            with open(source_csv, "rb") as src, open(dest_common, "wb") as dst:
                dst.write(src.read())
            deployed_destinations.append(str(dest_common))

            dest_common_root = Path(info.commondata_path) / "Files" / filename
            with open(source_csv, "rb") as src, open(dest_common_root, "wb") as dst:
                dst.write(src.read())
            deployed_destinations.append(str(dest_common_root))

        # Tester Agent Paths (Tester\<hash>\Agent-*\MQL5\Files)
        try:
            terminal_id = Path(info.data_path).name
            tester_root = Path(info.data_path).parent.parent / "Tester" / terminal_id
            if tester_root.exists():
                for agent_dir in tester_root.glob("Agent-*"):
                    agent_files = agent_dir / "MQL5" / "Files"
                    agent_files.mkdir(parents=True, exist_ok=True)
                    agent_sub = agent_files / "finrl_x_mt5"
                    agent_sub.mkdir(parents=True, exist_ok=True)

                    with open(source_csv, "rb") as src:
                        content = src.read()
                    with open(agent_files / filename, "wb") as dst:
                        dst.write(content)
                    with open(agent_sub / filename, "wb") as dst:
                        dst.write(content)
                    deployed_destinations.append(str(agent_files / filename))
        except Exception as e:
            logger.debug(f"Could not deploy to tester agent folders: {e}")

        for dest in deployed_destinations:
            logger.info(f"🚀 Deployed signals to MT5: {dest}")
