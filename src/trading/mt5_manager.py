"""
MT5 Connection and Session Manager
==================================
Manages the MetaTrader 5 terminal lifecycle, account state inspection,
market watch symbol subscriptions, and position queries.

Connects seamlessly to an already active MT5 terminal or launches it
using configured terminal path.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Any
from dataclasses import dataclass

import MetaTrader5 as mt5
from loguru import logger

from src.config.settings import settings


@dataclass
class AccountSnapshot:
    login: int
    server: str
    currency: str
    leverage: int
    balance: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    floating_profit: float
    trade_allowed: bool


class MT5Manager:
    """
    High-level MT5 lifecycle and account state manager.
    """

    def __init__(self, auto_connect: bool = True):
        self._connected = False
        if auto_connect:
            self.connect()

    def connect(self) -> bool:
        """
        Connect to MetaTrader 5 terminal.
        Prioritizes attaching to the currently open/logged-in terminal.
        """
        cfg = settings.mt5

        # 1. Direct attach to running terminal
        initialized = mt5.initialize()

        # 2. Fallback to path if not running
        if not initialized:
            init_kwargs: dict = {"timeout": cfg.mt5_timeout}
            if cfg.mt5_path:
                init_kwargs["path"] = cfg.mt5_path
            initialized = mt5.initialize(**init_kwargs)

        if not initialized:
            err = mt5.last_error()
            logger.error(f"MT5 initialize failed: {err}")
            self._connected = False
            return False

        # 3. Optional credential login if specified
        if cfg.mt5_login and cfg.mt5_password and cfg.mt5_server:
            if not mt5.login(login=cfg.mt5_login, password=cfg.mt5_password, server=cfg.mt5_server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                self._connected = False
                return False

        self._connected = True
        snap = self.get_account_snapshot()
        if snap:
            logger.info(
                f"✅ MT5Manager Online | Login: {snap.login} @ {snap.server} | "
                f"Equity: ${snap.equity:,.2f} | Free Margin: ${snap.margin_free:,.2f}"
            )
        return True

    def disconnect(self) -> None:
        """Shutdown MT5 terminal connection."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5Manager: Terminal connection closed.")

    def is_connected(self) -> bool:
        """Check if terminal is connected and responding."""
        if not self._connected:
            return False
        term = mt5.terminal_info()
        return term is not None and term.connected

    def ensure_connected(self) -> bool:
        """Ensure connection is healthy, reconnecting if dropped."""
        if not self.is_connected():
            logger.warning("MT5 connection lost. Reconnecting...")
            return self.connect()
        return True

    # ─── Account Inspection ───────────────────────────────────────────────────

    def get_account_snapshot(self) -> Optional[AccountSnapshot]:
        """Fetch current account balance, equity, and margin levels."""
        acc = mt5.account_info()
        if acc is None:
            return None
        return AccountSnapshot(
            login=acc.login,
            server=acc.server,
            currency=acc.currency,
            leverage=acc.leverage,
            balance=acc.balance,
            equity=acc.equity,
            margin=acc.margin,
            margin_free=acc.margin_free,
            margin_level=acc.margin_level if acc.margin_level is not None else 0.0,
            floating_profit=acc.profit,
            trade_allowed=acc.trade_allowed and acc.trade_expert,
        )

    # ─── Symbol Management ────────────────────────────────────────────────────

    def enable_symbol(self, symbol: str) -> bool:
        """Add symbol to Market Watch and enable real-time quote streaming."""
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Symbol {symbol} not found on broker server")
            return False
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol} into Market Watch")
                return False
            time.sleep(0.05)
        return True

    def get_symbol_spec(self, symbol: str) -> Optional[dict[str, Any]]:
        """Fetch broker specifications for lot calculation and trade limits."""
        if not self.enable_symbol(symbol):
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            "name": info.name,
            "point": info.point,
            "digits": info.digits,
            "spread": info.spread,
            "contract_size": info.trade_contract_size,
            "min_lot": info.volume_min,
            "max_lot": info.volume_max,
            "lot_step": info.volume_step,
            "trade_mode": info.trade_mode,
            "currency_base": info.currency_base,
            "currency_profit": info.currency_profit,
        }

    def is_market_open(self, symbol: str) -> bool:
        """Check if trading is currently allowed for this symbol."""
        info = mt5.symbol_info(symbol)
        if info is None:
            return False
        # trade_mode: 0=disabled, 1=close only, 2=full
        return info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL

    # ─── Positions & Deals ────────────────────────────────────────────────────

    def get_open_positions(self, symbol: Optional[str] = None) -> list[dict[str, Any]]:
        """Return list of currently open positions."""
        if symbol:
            raw_pos = mt5.positions_get(symbol=symbol)
        else:
            raw_pos = mt5.positions_get()

        if raw_pos is None:
            return []

        positions = []
        for p in raw_pos:
            positions.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": p.volume,
                "price_open": p.price_open,
                "price_current": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "swap": p.swap,
                "magic": p.magic,
                "comment": p.comment,
                "time": datetime.fromtimestamp(p.time, tz=timezone.utc),
            })
        return positions

    def get_daily_realized_pnl(self) -> float:
        """Calculate realized PnL closed today."""
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(today_start, datetime.now(timezone.utc))
        if deals is None:
            return 0.0
        return sum(d.profit + d.swap + d.commission for d in deals if d.entry == mt5.DEAL_ENTRY_OUT)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
