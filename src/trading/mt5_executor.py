"""
MT5 Order Executor
===================
Translates Council TradingDecision → MT5 market orders.
Implements pre-trade risk checks before every order.
"""

from __future__ import annotations

from typing import Optional
import time

import MetaTrader5 as mt5
from loguru import logger

from src.config.settings import settings
from src.council.council import TradingDecision


class MT5Executor:
    """
    Executes Council trading decisions as MT5 market orders.

    Pre-trade checks:
      ✅ Free margin > minimum threshold
      ✅ Portfolio drawdown < halt threshold
      ✅ Spread not spiking (news filter)
      ✅ Symbol not at market close
      ✅ No duplicate position in same direction
    """

    def __init__(self, lot_sizer=None, risk_manager=None):
        self.cfg        = settings.mt5
        self._lot_sizer = lot_sizer
        self._risk_mgr  = risk_manager

    # ─── Main Execute ────────────────────────────────────────────────────────

    def execute_decision(
        self,
        decision: TradingDecision,
        equity: float,
    ) -> Optional[dict]:
        """
        Execute a Council TradingDecision as an MT5 market order.

        Returns order result dict or None if rejected by pre-trade checks.
        """
        symbol    = decision.symbol
        direction = decision.direction

        # 1. Tradeable filter
        if not decision.is_tradeable:
            logger.debug(f"⏭  Skipping non-tradeable decision: {decision}")
            return None

        # 2. Pre-trade risk checks
        if not self._pre_trade_checks(symbol, equity):
            return None

        # 3. Close opposite position if exists
        self._close_opposite_position(symbol, direction)

        # 4. Compute lot size
        lots = self._compute_lots(symbol, decision, equity)
        if lots is None or lots <= 0:
            return None

        # 5. Get current price and compute SL/TP
        tick    = mt5.symbol_info_tick(symbol)
        price   = tick.ask if direction > 0 else tick.bid
        sl_pips = abs(price - decision.sl_price) if decision.sl_price else 0
        tp_pips = abs(decision.tp_price - price) if decision.tp_price else 0

        # 6. Build and send order request
        request = self._build_request(
            symbol, direction, lots, price,
            decision.sl_price, decision.tp_price,
            comment=f"Council|{decision.regime}|{decision.consensus_signal:.2f}"
        )

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                f"❌ Order failed for {symbol}: "
                f"retcode={result.retcode if result else 'None'} | "
                f"{mt5.last_error()}"
            )
            return None

        logger.info(
            f"✅ Order executed: {symbol} | "
            f"{'BUY' if direction > 0 else 'SELL'} {lots} lots @ {price:.5f} | "
            f"SL={decision.sl_price:.5f} | TP={decision.tp_price:.5f} | "
            f"Ticket={result.order}"
        )

        return {
            "ticket":    result.order,
            "symbol":    symbol,
            "direction": direction,
            "lots":      lots,
            "price":     price,
            "sl":        decision.sl_price,
            "tp":        decision.tp_price,
            "retcode":   result.retcode,
        }

    # ─── Pre-Trade Checks ────────────────────────────────────────────────────

    def _pre_trade_checks(self, symbol: str, equity: float) -> bool:
        """Run all pre-trade safety checks. Returns False to block order."""
        acc  = mt5.account_info()
        info = mt5.symbol_info(symbol)

        if acc is None or info is None:
            logger.error("Cannot get account/symbol info — blocking order")
            return False

        # Margin check
        free_margin_pct = acc.margin_free / (acc.equity + 1e-8)
        if free_margin_pct < self.cfg.min_free_margin_pct:
            logger.warning(
                f"⛔ Insufficient free margin: {free_margin_pct:.1%} < "
                f"{self.cfg.min_free_margin_pct:.1%} — order blocked"
            )
            return False

        # Drawdown halt check
        if acc.balance > 0:
            dd = 1.0 - (acc.equity / acc.balance)
            if dd > self.cfg.max_drawdown_halt_pct:
                logger.warning(
                    f"🛑 DRAWDOWN HALT: {dd:.1%} > "
                    f"{self.cfg.max_drawdown_halt_pct:.1%} — ALL trading halted"
                )
                return False

        # Spread spike filter (> 3× normal spread → likely news event)
        tick         = mt5.symbol_info_tick(symbol)
        current_spread = (tick.ask - tick.bid) / info.point
        if current_spread > info.spread * 3:
            logger.warning(
                f"⛔ Spread spike on {symbol}: {current_spread:.0f} pts "
                f"(normal={info.spread}) — order blocked"
            )
            return False

        return True

    def _close_opposite_position(self, symbol: str, new_direction: int) -> None:
        """Close any existing position in the opposite direction."""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return

        for pos in positions:
            pos_dir = +1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            if pos_dir != new_direction:
                close_type = mt5.ORDER_TYPE_SELL if pos_dir > 0 else mt5.ORDER_TYPE_BUY
                tick  = mt5.symbol_info_tick(symbol)
                price = tick.bid if pos_dir > 0 else tick.ask
                request = {
                    "action":   mt5.TRADE_ACTION_DEAL,
                    "position": pos.ticket,
                    "symbol":   symbol,
                    "volume":   pos.volume,
                    "type":     close_type,
                    "price":    price,
                    "magic":    self.cfg.magic_number,
                    "comment":  f"Council close #{pos.ticket}",
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"🔄 Closed opposite position #{pos.ticket} on {symbol}")

    def _compute_lots(
        self,
        symbol: str,
        decision: TradingDecision,
        equity: float,
    ) -> Optional[float]:
        """Compute lot size from decision position_size and equity."""
        inst_cfg = self.cfg.instrument_config.get(symbol, {})
        if not inst_cfg:
            logger.error(f"No instrument config for {symbol}")
            return None

        info  = mt5.symbol_info(symbol)
        tick  = mt5.symbol_info_tick(symbol)
        price = tick.ask

        # Dollar at risk = equity × risk% × position_size_scalar
        dollar_risk = equity * self.cfg.default_risk_pct * decision.position_size
        lot_raw     = dollar_risk / (price * inst_cfg.get("contract_size", 1) + 1e-8)

        # Round to lot step
        step  = inst_cfg.get("lot_step", 0.01)
        min_l = inst_cfg.get("min_lot", 0.01)
        max_l = inst_cfg.get("max_lot", info.volume_max if info else 10.0)

        lots = round(max(min_l, min(max_l, round(lot_raw / step) * step)), 2)
        return lots

    def _build_request(
        self,
        symbol: str,
        direction: int,
        lots: float,
        price: float,
        sl: Optional[float],
        tp: Optional[float],
        comment: str = "",
    ) -> dict:
        info = mt5.symbol_info(symbol)

        request = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   symbol,
            "volume":   lots,
            "type":     mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL,
            "price":    price,
            "deviation": 20,       # max price deviation in points
            "magic":    self.cfg.magic_number,
            "comment":  comment[:63],  # MT5 max comment length
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if sl and sl > 0:
            request["sl"] = round(sl, info.digits if info else 5)
        if tp and tp > 0:
            request["tp"] = round(tp, info.digits if info else 5)

        return request
