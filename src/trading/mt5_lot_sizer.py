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

        Returns
        -------
        float
            Quantized lot size conforming to broker min/max/step.
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"LotSizer: Could not fetch symbol info for {symbol}")
            return 0.0

        min_lot  = info.volume_min
        max_lot  = info.volume_max
        lot_step = info.volume_step
        contract_size = info.trade_contract_size

        if entry_price is None or entry_price <= 0:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return min_lot
            entry_price = tick.ask if decision.direction > 0 else tick.bid

        # 1. Determine SL distance in price units
        sl_price = decision.sl_price
        if sl_price is None or sl_price <= 0:
            # Fallback: estimate 1% price distance if no SL specified
            sl_dist = entry_price * 0.01
        else:
            sl_dist = abs(entry_price - sl_price)

        if sl_dist <= 0:
            sl_dist = entry_price * 0.005

        # 2. Risk capital in USD
        # Base risk: e.g. 2% of equity
        base_risk_pct = self.cfg.default_risk_pct
        risk_usd = equity * base_risk_pct

        # 3. Half-Kelly scaling by Council confidence (0.40 - 1.00)
        # Higher council consensus scales risk slightly up (max 1.25x), low confidence scales down (min 0.5x)
        confidence = max(0.1, min(1.0, decision.confidence))
        kelly_factor = 0.5 + (confidence * 0.75)  # Range ~ [0.8, 1.25]
        adjusted_risk_usd = risk_usd * kelly_factor

        # 4. Compute raw volume
        # Monetary loss for 1 standard lot = sl_dist * contract_size
        loss_per_lot = sl_dist * contract_size

        if loss_per_lot <= 0:
            return min_lot

        raw_lots = adjusted_risk_usd / loss_per_lot

        # 5. Margin limit safety check
        # Approximate margin required: (lots * contract_size * entry_price) / leverage
        acc = mt5.account_info()
        leverage = acc.leverage if acc and acc.leverage > 0 else 100
        free_margin = acc.margin_free if acc else equity

        margin_per_lot = (contract_size * entry_price) / leverage
        if margin_per_lot > 0:
            max_affordable_lots = (free_margin * 0.50) / margin_per_lot
            raw_lots = min(raw_lots, max_affordable_lots)

        # 6. Quantize lots to broker step and clamp to [min_lot, max_lot]
        quantized_lots = math.floor(raw_lots / lot_step) * lot_step
        quantized_lots = round(quantized_lots, self._step_to_digits(lot_step))

        final_lots = max(min_lot, min(max_lot, quantized_lots))

        logger.debug(
            f"LotSizer [{symbol}]: risk=${adjusted_risk_usd:.2f} | "
            f"sl_dist={sl_dist:.4f} | raw={raw_lots:.4f} -> final={final_lots}"
        )
        return final_lots

    @staticmethod
    def _step_to_digits(step: float) -> int:
        """Derive decimal places from broker volume step (e.g., 0.01 -> 2, 0.1 -> 1)."""
        if step >= 1.0:
            return 0
        s = f"{step:.6f}".rstrip("0")
        return len(s.split(".")[1]) if "." in s else 2
