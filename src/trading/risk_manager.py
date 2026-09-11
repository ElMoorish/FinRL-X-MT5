"""
Portfolio & Pre-Trade Risk Manager
==================================
Enforces portfolio-level risk rules, daily drawdown circuit breakers,
spread-spike filters, and cross-asset correlation exposure limits.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5
from loguru import logger

from src.config.settings import settings
from src.council.council import TradingDecision


class RiskManager:
    """
    Portfolio-level guardian for live MT5 execution.
    """

    def __init__(self):
        self.cfg = settings.mt5
        self._initial_day_equity: Optional[float] = None
        self._last_day: Optional[int] = None

    def _sync_day_equity(self, current_equity: float) -> None:
        """Reset starting daily equity at UTC midnight for daily drawdown tracking."""
        today = datetime.now(timezone.utc).day
        if self._last_day != today or self._initial_day_equity is None:
            self._initial_day_equity = current_equity
            self._last_day = today
            logger.info(f"RiskManager: New day initialized. Daily benchmark equity: ${current_equity:,.2f}")

    def validate_trade(
        self,
        decision: TradingDecision,
        lots: Optional[float] = None,
        entry_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ) -> tuple[bool, str]:
        """
        Validate whether an intended trade complies with all risk guidelines.

        Returns
        -------
        tuple[bool, str]
            (True, "Approved") or (False, "Reason for rejection")
        """
        symbol = decision.symbol
        acc = mt5.account_info()
        if acc is None:
            return False, "Failed to retrieve account info from MT5"

        self._sync_day_equity(acc.equity)

        # 1. Trading permission checks
        if not acc.trade_allowed or not acc.trade_expert:
            return False, "Automated algorithmic trading is disabled in MT5 terminal"

        # 2. Daily Drawdown Circuit Breaker
        if self._initial_day_equity and self._initial_day_equity > 0:
            daily_dd = (self._initial_day_equity - acc.equity) / self._initial_day_equity
            daily_limit = getattr(self.cfg, "max_daily_loss_pct", 0.025)
            if daily_dd >= daily_limit:
                return False, f"Daily drawdown halt triggered ({daily_dd:.2%} >= {daily_limit:.2%})"

        # 2b. Total Account Drawdown Circuit Breaker (Peak-to-Trough)
        if acc.balance > 0:
            total_dd = 1.0 - (acc.equity / acc.balance)
            if total_dd >= self.cfg.max_drawdown_halt_pct:
                return False, f"Total drawdown halt triggered ({total_dd:.2%} >= {self.cfg.max_drawdown_halt_pct:.2%})"

        # 3. Free margin check
        if acc.equity > 0:
            free_margin_ratio = acc.margin_free / acc.equity
            if free_margin_ratio < self.cfg.min_free_margin_pct:
                return False, f"Insufficient free margin ratio: {free_margin_ratio:.2%} < {self.cfg.min_free_margin_pct:.2%}"

        # 4. Spread spike filter (news or illiquidity protection)
        info = mt5.symbol_info(symbol)
        if info is None:
            return False, f"Symbol {symbol} not found"

        spec = self.cfg.instrument_config.get(symbol, {})
        base_digits = spec.get("digits", info.digits)
        # Check current spread in points
        current_spread_pts = info.spread
        # If spread is more than 3x normal baseline, reject
        max_allowed_spread = spec.get("max_spread_pts", current_spread_pts * 3.5)
        if current_spread_pts > max_allowed_spread:
            return False, f"Spread too wide ({current_spread_pts} > {max_allowed_spread}) - news volatility filter"

        # 5. Position duplication & correlation check
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for p in positions:
                # If already holding a position in the same direction, don't double down
                existing_dir = 1 if p.type == mt5.POSITION_TYPE_BUY else -1
                if existing_dir == decision.direction:
                    return False, f"Position already exists in same direction on {symbol}"

        # 6. Correlated index exposure limit (e.g. NAS100 + SPX500 + US30)
        us_indices = {"NAS100.x", "SPX500.x", "US30.x"}
        if symbol in us_indices:
            all_positions = mt5.positions_get() or []
            active_us_index_trades = sum(
                1 for p in all_positions if p.symbol in us_indices and (1 if p.type == mt5.POSITION_TYPE_BUY else -1) == decision.direction
            )
            if active_us_index_trades >= 2:
                return False, f"Correlated US index limit reached ({active_us_index_trades} active positions in same direction)"

        # 7. H1 Macro Trend Governor: block counter-trend trades against H1 EMA 50
        h1_ema, bid = RiskManager.get_h1_trend(symbol)
        if h1_ema is not None and bid is not None:
            if decision.direction > 0 and bid < h1_ema:
                return False, f"H1 Macro Trend Governor: Long blocked (Bid {bid:.2f} < H1 EMA50 {h1_ema:.2f})"
            if decision.direction < 0 and bid > h1_ema:
                return False, f"H1 Macro Trend Governor: Short blocked (Bid {bid:.2f} > H1 EMA50 {h1_ema:.2f})"

        # 8. Strict Monetary Risk Budget Enforcement (0.50% hard risk ceiling)
        if lots is not None and lots > 0:
            config_max_lot = spec.get("max_lot")
            if config_max_lot and lots > config_max_lot:
                return False, f"Lot size {lots} exceeds configured max_lot ceiling ({config_max_lot}) for {symbol}"

            eff_sl = sl_price if sl_price is not None else decision.sl_price
            if eff_sl and eff_sl > 0 and entry_price and entry_price > 0 and acc.equity > 0:
                contract_size = info.trade_contract_size if (info and hasattr(info, "trade_contract_size") and info.trade_contract_size > 0) else spec.get("contract_size")
                if not contract_size or contract_size <= 0:
                    return False, f"Contract size unknown for {symbol} — trade blocked for safety"
                sl_dist = abs(entry_price - eff_sl)
                monetary_risk = lots * sl_dist * contract_size
                max_risk_pct = getattr(self.cfg, "default_risk_pct", 0.005)
                max_allowed_risk = acc.equity * max_risk_pct * 1.01  # Strict 0.50% ceiling (+1% micro-buffer for tick precision)
                if monetary_risk > max_allowed_risk:
                    return False, (
                        f"Monetary risk ceiling exceeded: ${monetary_risk:.2f} > "
                        f"${max_allowed_risk:.2f} ({max_risk_pct:.2%} max budget on ${acc.equity:,.2f} equity)"
                    )

        return True, "Approved"

    # ─── Shared Utilities ─────────────────────────────────────────────────────

    @staticmethod
    def get_h1_trend(symbol: str) -> tuple[float | None, float | None]:
        """
        Single source of truth for the H1 EMA50 macro trend check.

        Returns (h1_ema, current_bid) if data is available, else (None, None).
        Both the trade validator (Rule 7) and the CoT generator call this method
        to guarantee a consistent result from the same computation path.

        Returns
        -------
        tuple[float | None, float | None]
            (h1_ema50, bid_price)  — both None if MT5 data unavailable
        """
        try:
            h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 60)
            if h1_rates is not None and len(h1_rates) >= 50:
                import pandas as pd
                h1_closes = pd.Series([r[4] for r in h1_rates])
                h1_ema    = float(h1_closes.ewm(span=50, adjust=False).mean().iloc[-1])
                tick = mt5.symbol_info_tick(symbol)
                bid  = tick.bid if tick else float(h1_closes.iloc[-1])
                return h1_ema, bid
        except Exception as e:
            logger.debug(f"get_h1_trend ({symbol}): {e}")
        return None, None
