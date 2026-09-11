"""
MT5 Order Executor
===================
Translates Council TradingDecision → MT5 market orders.
Implements pre-trade risk checks before every order.
"""

from __future__ import annotations

from typing import Optional
import math
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

    def __init__(self, lot_sizer=None, risk_manager=None, notifier=None):
        from src.trading.mt5_lot_sizer import MT5LotSizer
        from src.trading.risk_manager import RiskManager
        from src.trading.trade_notifier import TradeNotifier
        self.cfg        = settings.mt5
        self._lot_sizer = lot_sizer or MT5LotSizer()
        self._risk_mgr  = risk_manager or RiskManager()
        self._notifier  = notifier or TradeNotifier()

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

        # 2. Get current price and sanitize SL/TP stops upfront BEFORE lot sizing
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Cannot fetch tick for {symbol} — blocking order")
            return None
        price = tick.ask if direction > 0 else tick.bid

        sanitized_sl, sanitized_tp = self.sanitize_stops(
            symbol=symbol,
            direction=direction,
            price=price,
            sl=decision.sl_price,
            tp=decision.tp_price,
        )

        # 3. Compute lot size based on SANITIZED stop distance
        lots = self._compute_lots(
            symbol=symbol,
            decision=decision,
            equity=equity,
            entry_price=price,
            sl_price=sanitized_sl,
        )
        if lots is None or lots <= 0:
            return None

        # 4. Pre-trade risk checks (including strict monetary risk budget validation)
        if not self._pre_trade_checks(
            symbol=symbol,
            equity=equity,
            decision=decision,
            lots=lots,
            entry_price=price,
            sl_price=sanitized_sl,
        ):
            return None

        # 5. Close opposite position if exists
        self._close_opposite_position(symbol, direction)

        # 6. Build and send order request
        request = self._build_request(
            symbol=symbol,
            direction=direction,
            lots=lots,
            price=price,
            sl=sanitized_sl,
            tp=sanitized_tp,
            comment=f"Council|{decision.regime}|{decision.consensus_signal:.2f}",
        )

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode_val = result.retcode if result else "None"
            comment_val = result.comment if result else ""
            logger.error(
                f"❌ Order failed for {symbol}: "
                f"retcode={retcode_val} ({comment_val}) | "
                f"{mt5.last_error()}"
            )
            return None

        actual_sl = request.get("sl", sanitized_sl)
        actual_tp = request.get("tp", sanitized_tp)
        logger.info(
            f"✅ Order executed: {symbol} | "
            f"{'BUY' if direction > 0 else 'SELL'} {lots} lots @ {price:.5f} | "
            f"SL={actual_sl:.5f} | TP={actual_tp:.5f} | "
            f"Ticket={result.order}"
        )

        # Dispatch real-time external alerts (Telegram / Discord)
        self._notifier.notify_trade_executed(
            symbol=symbol,
            direction=direction,
            lots=lots,
            price=price,
            sl=actual_sl,
            tp=actual_tp,
            ticket=result.order,
            decision=decision,
        )

        return {
            "ticket":    result.order,
            "symbol":    symbol,
            "direction": direction,
            "lots":      lots,
            "price":     price,
            "sl":        actual_sl,
            "tp":        actual_tp,
            "retcode":   result.retcode,
        }

    # ─── Pre-Trade Checks ────────────────────────────────────────────────────

    def _pre_trade_checks(
        self,
        symbol: str,
        equity: float,
        decision: Optional[TradingDecision] = None,
        lots: Optional[float] = None,
        entry_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ) -> bool:
        """Run all pre-trade safety checks. Returns False to block order."""
        if self._risk_mgr is not None and decision is not None:
            ok, reason = self._risk_mgr.validate_trade(
                decision,
                lots=lots,
                entry_price=entry_price,
                sl_price=sl_price,
            )
            if not ok:
                logger.warning(f"⛔ RiskManager blocked {symbol}: {reason}")
                return False

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
                    self._notifier.notify_position_closed(
                        symbol=symbol,
                        ticket=pos.ticket,
                        reason=f"Council flipped to {'BUY' if new_direction > 0 else 'SELL'}"
                    )

    def manage_active_positions(self, symbol: str) -> None:
        """Dynamic Breakeven management (+1.0R profit triggers BE lock)."""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return

        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if not info or not tick:
            return

        min_stop_pts = max(info.trade_stops_level, 20) * info.point

        for pos in positions:
            if pos.magic != self.cfg.magic_number:
                continue

            open_price = pos.price_open
            current_sl = pos.sl
            current_tp = pos.tp

            if current_sl <= 0:
                continue

            if pos.type == mt5.ORDER_TYPE_BUY:
                initial_risk = open_price - current_sl
                if initial_risk > 0:
                    profit_dist = tick.bid - open_price
                    if profit_dist >= 1.0 * initial_risk:
                        new_sl = round(open_price + (10.0 * info.point), info.digits)
                        if new_sl > current_sl and (tick.bid - new_sl) >= min_stop_pts:
                            req = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "symbol": symbol,
                                "sl": new_sl,
                                "tp": current_tp,
                            }
                            res = mt5.order_send(req)
                            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                                logger.info(f"🛡️ Breakeven activated for BUY #{pos.ticket} on {symbol} | New SL={new_sl}")
            elif pos.type == mt5.ORDER_TYPE_SELL:
                initial_risk = current_sl - open_price
                if initial_risk > 0:
                    profit_dist = open_price - tick.ask
                    if profit_dist >= 1.0 * initial_risk:
                        new_sl = round(open_price - (10.0 * info.point), info.digits)
                        if new_sl < current_sl and (new_sl - tick.ask) >= min_stop_pts:
                            req = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "symbol": symbol,
                                "sl": new_sl,
                                "tp": current_tp,
                            }
                            res = mt5.order_send(req)
                            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                                logger.info(f"🛡️ Breakeven activated for SELL #{pos.ticket} on {symbol} | New SL={new_sl}")

    def sanitize_stops(
        self,
        symbol: str,
        direction: int,
        price: float,
        sl: Optional[float],
        tp: Optional[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Dynamically validate and sanitize stops against broker trade_stops_level,
        index breathing room buffer, tick size, and order direction.
        """
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if not info or not tick:
            return sl, tp

        broker_stop_pts = max(info.trade_stops_level, 20) * info.point
        is_index = any(idx in symbol.upper() for idx in ["NAS100", "USTEC", "US30", "SPX", "GER40"])
        min_stop_pts = max(broker_stop_pts, 60.0 if is_index else broker_stop_pts)
        tick_size = info.trade_tick_size if info.trade_tick_size > 0 else info.point

        sanitized_sl = sl
        sanitized_tp = tp

        if direction > 0:  # BUY: SL below Bid, TP above Bid
            if sl and sl > 0:
                sl_dist = abs(sl - price)
                raw_sl = min(tick.bid - max(sl_dist, min_stop_pts), tick.bid - min_stop_pts)
                sanitized_sl = round(round(raw_sl / tick_size) * tick_size, info.digits)
            if tp and tp > 0:
                tp_dist = abs(tp - price)
                raw_tp = max(tick.bid + max(tp_dist, min_stop_pts * 1.5), tick.bid + min_stop_pts)
                sanitized_tp = round(round(raw_tp / tick_size) * tick_size, info.digits)

        elif direction < 0:  # SELL: SL above Ask, TP below Ask
            if sl and sl > 0:
                sl_dist = abs(sl - price)
                raw_sl = max(tick.ask + max(sl_dist, min_stop_pts), tick.ask + min_stop_pts)
                sanitized_sl = round(round(raw_sl / tick_size) * tick_size, info.digits)
            if tp and tp > 0:
                tp_dist = abs(price - tp)
                raw_tp = min(tick.ask - max(tp_dist, min_stop_pts * 1.5), tick.ask - min_stop_pts)
                sanitized_tp = round(round(raw_tp / tick_size) * tick_size, info.digits)

        return sanitized_sl, sanitized_tp

    def _compute_lots(
        self,
        symbol: str,
        decision: TradingDecision,
        equity: float,
        entry_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ) -> Optional[float]:
        """Compute lot size from decision position_size and equity."""
        if self._lot_sizer is not None:
            return self._lot_sizer.calculate_lots(
                symbol, decision, equity, entry_price=entry_price, sl_price=sl_price
            )

        inst_cfg = self.cfg.instrument_config.get(symbol, {})
        if not inst_cfg:
            logger.error(f"No instrument config for {symbol}")
            return None

        info  = mt5.symbol_info(symbol)
        contract_size = inst_cfg.get("contract_size", info.trade_contract_size if info else 1.0)
        price = entry_price or (mt5.symbol_info_tick(symbol).ask if mt5.symbol_info_tick(symbol) else 1.0)

        # Monetary risk calculation
        dollar_risk = equity * self.cfg.default_risk_pct * decision.position_size
        target_sl = sl_price if sl_price is not None else decision.sl_price
        sl_dist = abs(price - target_sl) if target_sl and target_sl > 0 else (price * 0.01)
        loss_per_lot = sl_dist * contract_size
        lot_raw = dollar_risk / (loss_per_lot + 1e-8)

        # Round to lot step
        step  = inst_cfg.get("lot_step", info.volume_step if info else 0.01)
        min_l = inst_cfg.get("min_lot", info.volume_min if info else 0.01)
        max_l = inst_cfg.get("max_lot", info.volume_max if info else 10.0)

        lots = round(max(min_l, min(max_l, math.floor(lot_raw / step) * step)), 2)
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

        # Detect broker-supported filling mode dynamically
        filling_mode = mt5.ORDER_FILLING_IOC
        if info and hasattr(info, "filling_mode"):
            if (info.filling_mode & 2) != 0:
                filling_mode = mt5.ORDER_FILLING_IOC
            elif (info.filling_mode & 1) != 0:
                filling_mode = mt5.ORDER_FILLING_FOK
            else:
                filling_mode = mt5.ORDER_FILLING_RETURN

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lots,
            "type":         mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL,
            "price":        price,
            "deviation":    20,       # max price deviation in points
            "magic":        self.cfg.magic_number,
            "comment":      comment[:63],  # MT5 max comment length
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        # Apply pre-sanitized stops directly to request
        if sl is not None and sl > 0:
            request["sl"] = round(sl, info.digits if info else 5)
        if tp is not None and tp > 0:
            request["tp"] = round(tp, info.digits if info else 5)

        return request
