"""
MT5 Real-Tick Data Fetcher
===========================
Pulls real tick data from MetaTrader 5 terminal via the Python API.
This is the SAME tick data used by MT5 Strategy Tester in
"Every tick based on real ticks" mode — ensuring backtest/live consistency.

Usage:
    fetcher = MT5TickFetcher()
    ticks = fetcher.get_ticks("USTEC", days=90)
    ohlcv = fetcher.get_ohlcv("USTEC", timeframe=mt5.TIMEFRAME_M5, n_bars=500)
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
import polars as pl
import numpy as np
from loguru import logger

from src.config.settings import settings


class MT5TickFetcher:
    """
    Fetches real tick and OHLCV data from MetaTrader 5.

    Tick data pulled here is byte-for-byte identical to what the MT5
    Strategy Tester uses in 'Every tick based on real ticks' mode.
    """

    # MT5 Timeframe constants for convenience
    TF = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }

    def __init__(self, auto_connect: bool = True):
        self._connected = False
        if auto_connect:
            self.connect()

    # ─── Connection Management ────────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialize MT5 connection using settings or attaching to active terminal."""
        cfg = settings.mt5

        # First attempt: attach to already running MT5 terminal
        initialized = mt5.initialize()

        # Second attempt: try with configured path/timeout if direct attach failed
        if not initialized:
            init_kwargs: dict = {"timeout": cfg.mt5_timeout}
            if cfg.mt5_path:
                init_kwargs["path"] = cfg.mt5_path
            initialized = mt5.initialize(**init_kwargs)

        if not initialized:
            logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return False

        # Authenticate if explicit credentials provided
        if cfg.mt5_login and cfg.mt5_password and cfg.mt5_server:
            authorized = mt5.login(
                login=cfg.mt5_login,
                password=cfg.mt5_password,
                server=cfg.mt5_server,
            )
            if not authorized:
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False

        self._connected = True
        info = mt5.terminal_info()
        account = mt5.account_info()
        acc_str = f"Account: {account.login} | Equity: ${account.equity:,.2f}" if account else "Account: Not logged in"
        logger.info(
            f"✅ MT5 Connected | Build {info.build if info else 'Unknown'} | {acc_str}"
        )
        return True

    def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        mt5.shutdown()
        self._connected = False
        logger.info("MT5 disconnected")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()

    # ─── Symbol Info ─────────────────────────────────────────────────────────

    def get_symbol_info(self, symbol: str) -> dict | None:
        """Return symbol metadata: point, digits, spread, contract_size, etc."""
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Symbol {symbol} not found in MT5")
            return None
        # Enable symbol for data access if not already enabled
        if not info.visible:
            mt5.symbol_select(symbol, True)
            time.sleep(0.1)
            info = mt5.symbol_info(symbol)
        return {
            "symbol":        info.name,
            "description":   info.description,
            "point":         info.point,
            "digits":        info.digits,
            "spread":        info.spread,
            "contract_size": info.trade_contract_size,
            "min_lot":       info.volume_min,
            "max_lot":       info.volume_max,
            "lot_step":      info.volume_step,
            "currency_base": info.currency_base,
            "currency_profit": info.currency_profit,
        }

    # ─── OHLCV Data ──────────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "M5",
        n_bars: int = 500,
    ) -> pl.DataFrame:
        """
        Pull OHLCV bars from MT5.

        Args:
            symbol:    MT5 symbol name (e.g. 'USTEC', 'USOIL')
            timeframe: Timeframe string ('M1','M5','M15','H1','H4','D1')
            n_bars:    Number of bars to pull (from current time backwards)

        Returns:
            Polars DataFrame with columns:
            [time, open, high, low, close, tick_volume, spread, real_volume]
        """
        tf = self.TF.get(timeframe.upper(), mt5.TIMEFRAME_M5) if isinstance(timeframe, str) else timeframe
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)

        if rates is None or len(rates) == 0:
            logger.error(f"No OHLCV data for {symbol} {timeframe}: {mt5.last_error()}")
            return pl.DataFrame()

        df = (
            pl.DataFrame({
                "time":        [r[0] for r in rates],
                "open":        [r[1] for r in rates],
                "high":        [r[2] for r in rates],
                "low":         [r[3] for r in rates],
                "close":       [r[4] for r in rates],
                "tick_volume": [r[5] for r in rates],
                "spread":      [r[6] for r in rates],
                "real_volume": [r[7] for r in rates],
            })
            .with_columns(
                pl.from_epoch("time", time_unit="s")
                  .dt.replace_time_zone("UTC")
                  .alias("time")
            )
        )
        logger.debug(f"Pulled {len(df)} {timeframe} bars for {symbol}")
        return df

    def get_ohlcv_range(
        self,
        symbol: str,
        date_from: datetime,
        date_to: datetime,
        timeframe: str = "M5",
    ) -> pl.DataFrame:
        """Pull OHLCV bars for a specific date range (used for backtesting)."""
        tf = self.TF.get(timeframe.upper(), mt5.TIMEFRAME_M5)

        # MT5 expects UTC datetimes
        utc_from = date_from.replace(tzinfo=timezone.utc)
        utc_to   = date_to.replace(tzinfo=timezone.utc)

        rates = mt5.copy_rates_range(symbol, tf, utc_from, utc_to)
        if rates is None or len(rates) == 0:
            logger.error(f"No OHLCV range data for {symbol}: {mt5.last_error()}")
            return pl.DataFrame()

        return (
            pl.from_numpy(rates)
            .with_columns(
                pl.from_epoch("time", time_unit="s")
                  .dt.replace_time_zone("UTC")
                  .alias("time")
            )
        )

    # ─── Real Tick Data ───────────────────────────────────────────────────────

    def get_ticks(
        self,
        symbol: str,
        days: int = 30,
        tick_type: int = mt5.COPY_TICKS_ALL,
    ) -> pl.DataFrame:
        """
        Pull real tick data — same source as MT5 Strategy Tester real ticks.

        Args:
            symbol:    MT5 symbol
            days:      How many calendar days back to pull
            tick_type: mt5.COPY_TICKS_ALL | COPY_TICKS_INFO | COPY_TICKS_TRADE

        Returns:
            Polars DataFrame with tick data including bid/ask/volume/flags
        """
        utc_to   = datetime.now(timezone.utc)
        utc_from = utc_to - timedelta(days=days)

        ticks = mt5.copy_ticks_range(symbol, utc_from, utc_to, tick_type)
        if ticks is None or len(ticks) == 0:
            logger.error(f"No tick data for {symbol}: {mt5.last_error()}")
            return pl.DataFrame()

        df = (
            pl.from_numpy(ticks)
            .with_columns([
                pl.from_epoch("time", time_unit="s")
                  .dt.replace_time_zone("UTC")
                  .alias("time"),
                (pl.col("ask") - pl.col("bid"))
                  .alias("spread_pts"),
                pl.col("volume_real")
                  .alias("real_volume"),
            ])
            .sort("time")
        )

        logger.info(
            f"📈 {symbol}: {len(df):,} ticks over {days}d "
            f"({df['time'].min()} → {df['time'].max()})"
        )
        return df

    def get_ticks_range(
        self,
        symbol: str,
        date_from: datetime,
        date_to: datetime,
    ) -> pl.DataFrame:
        """Pull tick data for a specific date range (for backtest feature engineering)."""
        utc_from = date_from.replace(tzinfo=timezone.utc)
        utc_to   = date_to.replace(tzinfo=timezone.utc)

        ticks = mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return pl.DataFrame()

        return (
            pl.from_numpy(ticks)
            .with_columns([
                pl.from_epoch("time", time_unit="s")
                  .dt.replace_time_zone("UTC")
                  .alias("time"),
                (pl.col("ask") - pl.col("bid")).alias("spread_pts"),
            ])
            .sort("time")
        )

    # ─── Live Account Data ────────────────────────────────────────────────────

    def get_account_state(self) -> dict:
        """Return current account equity, balance, margin, drawdown."""
        acc = mt5.account_info()
        if acc is None:
            return {}
        return {
            "login":       acc.login,
            "balance":     acc.balance,
            "equity":      acc.equity,
            "margin":      acc.margin,
            "free_margin": acc.margin_free,
            "margin_level": acc.margin_level,
            "profit":      acc.profit,
            "drawdown_pct": 1.0 - (acc.equity / acc.balance) if acc.balance > 0 else 0.0,
        }

    def get_open_positions(self, symbol: Optional[str] = None) -> pl.DataFrame:
        """Return current open positions, optionally filtered by symbol."""
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if not positions:
            return pl.DataFrame()

        rows = [{
            "ticket":     p.ticket,
            "symbol":     p.symbol,
            "type":       "buy" if p.type == mt5.ORDER_TYPE_BUY else "sell",
            "volume":     p.volume,
            "open_price": p.price_open,
            "sl":         p.sl,
            "tp":         p.tp,
            "profit":     p.profit,
            "magic":      p.magic,
            "comment":    p.comment,
        } for p in positions]
        return pl.DataFrame(rows)
