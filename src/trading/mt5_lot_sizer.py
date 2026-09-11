"""
MT5 Dynamic Lot Sizer
=====================
Computes precision position sizes for indices, commodities, and equities
based on account equity, stop-loss distance, broker contract size, and
Council confidence weighting (Half-Kelly scaling).
"""

from __future__ import annotations

import math
from typing import Optional

import MetaTrader5 as mt5
from loguru import logger

from src.config.settings import settings
from src.council.council import TradingDecision


class MT5LotSizer:
    """
    Calculates exact lot size for MT5 orders respecting broker constraints.
    """

    def __init__(self):
        self.cfg = settings.mt5

    def calculate_lots(
        self,
        symbol: str,
        decision: TradingDecision,
        equity: float,
        entry_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ) -> float:
        """
        Calculate appropriate lot size for a Council TradingDecision.

        Parameters
        ----------
        symbol : str
            Trading symbol, e.g. 'NAS100.x', 'WTI.x'
        decision : TradingDecision
            Council output with direction, confidence, sl_price, tp_price
        equity : float
            Current account equity in USD
        entry_price : float, optional
            Expected entry price (defaults to current market ask/bid)
        sl_price : float, optional
            Sanitized stop loss price (defaults to decision.sl_price)

        Returns
        -------
        float
            Quantized lot size conforming to broker min/max/step.
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"LotSizer: Could not fetch symbol info for {symbol}")
            return 0.0

        spec = self.cfg.instrument_config.get(symbol, {})
        min_lot  = spec.get("min_lot", info.volume_min)
        max_lot  = min(info.volume_max, spec.get("max_lot", info.volume_max))
        lot_step = info.volume_step

        # Validate contract size strictly — never guess 1.0 for indices/commodities
        contract_size = info.trade_contract_size if (info and info.trade_contract_size > 0) else spec.get("contract_size")
        if not contract_size or contract_size <= 0:
            logger.error(f"LotSizer [{symbol}]: Contract size is unknown or non-positive ({contract_size}) — blocking order for capital safety")
            return 0.0

        if entry_price is None or entry_price <= 0:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return min_lot
            entry_price = tick.ask if decision.direction > 0 else tick.bid

        # 1. Determine effective minimum stop distance (broker limit + index buffer)
        broker_stop_pts = max(info.trade_stops_level, 20) * info.point
        is_index = any(idx in symbol.upper() for idx in ["NAS100", "USTEC", "US30", "SPX", "GER40"])
        min_stop_pts = max(broker_stop_pts, 60.0 if is_index else broker_stop_pts)

        target_sl = sl_price if sl_price is not None else decision.sl_price
        if target_sl is None or target_sl <= 0:
            sl_dist = max(entry_price * 0.01, min_stop_pts)
        else:
            sl_dist = max(abs(entry_price - target_sl), min_stop_pts)

        # 2. Risk capital in USD (Strict 0.50% hard dollar ceiling)
        base_risk_pct = self.cfg.default_risk_pct
        max_risk_usd = equity * base_risk_pct

        # 3. Half-Kelly scaling by Council confidence
        # Scales risk DOWN on lower confidence (min 0.5x), but strictly CAPPED at 1.0x (never exceeds 0.50%)
        confidence = max(0.1, min(1.0, decision.confidence))
        kelly_factor = min(1.0, 0.5 + (confidence * 0.5))  # Range [0.55, 1.00]
        adjusted_risk_usd = min(max_risk_usd, max_risk_usd * kelly_factor)

        # 4. Compute raw volume
        # Monetary loss for 1 standard lot = sl_dist * contract_size
        loss_per_lot = sl_dist * contract_size

        if loss_per_lot <= 0:
            return min_lot

        raw_lots = adjusted_risk_usd / loss_per_lot

        # 5. Margin limit safety check
        acc = mt5.account_info()
        leverage = acc.leverage if acc and acc.leverage > 0 else 100
        free_margin = acc.margin_free if acc else equity

        margin_per_lot = (contract_size * entry_price) / leverage
        if margin_per_lot > 0:
            max_affordable_lots = (free_margin * 0.50) / margin_per_lot
            raw_lots = min(raw_lots, max_affordable_lots)

        # 6. Quantize lots to broker step and enforce strict dollar risk ceiling
        quantized_lots = math.floor(raw_lots / lot_step) * lot_step
        max_risk_lots = math.floor(max_risk_usd / loss_per_lot / lot_step) * lot_step
        quantized_lots = min(quantized_lots, max_risk_lots)
        quantized_lots = round(quantized_lots, self._step_to_digits(lot_step))

        # 7. Broker minimum lot safety check: never clamp to min_lot if it exceeds 0.5% risk
        if quantized_lots < min_lot:
            min_lot_loss = min_lot * loss_per_lot
            if min_lot_loss > max_risk_usd:
                logger.warning(
                    f"LotSizer [{symbol}]: Minimum broker lot {min_lot} risks ${min_lot_loss:.2f} > "
                    f"max allowed 0.50% risk (${max_risk_usd:.2f}) on ${equity:,.2f} equity — trade rejected for safety"
                )
                return 0.0
            final_lots = min_lot
        else:
            final_lots = min(max_lot, quantized_lots)

        # Final sanity assertion: effective loss must never exceed 0.50% ceiling
        effective_loss = final_lots * loss_per_lot
        if effective_loss > max_risk_usd * 1.001:  # negligible precision threshold
            final_lots = math.floor(max_risk_usd / loss_per_lot / lot_step) * lot_step
            effective_loss = final_lots * loss_per_lot
            if final_lots < min_lot:
                return 0.0

        logger.debug(
            f"LotSizer [{symbol}]: max_budget=${max_risk_usd:.2f} | target_risk=${adjusted_risk_usd:.2f} | "
            f"sl_dist={sl_dist:.2f}pts | contract={contract_size} | "
            f"lots={final_lots} | effective_loss=${effective_loss:.2f} ({(effective_loss / equity):.3%})"
        )
        return final_lots

    @staticmethod
    def _step_to_digits(step: float) -> int:
        """Derive decimal places from broker volume step (e.g., 0.01 -> 2, 0.1 -> 1)."""
        if step >= 1.0:
            return 0
        s = f"{step:.6f}".rstrip("0")
        return len(s.split(".")[1]) if "." in s else 2
