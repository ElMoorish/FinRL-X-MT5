"""
FinRL-X External Trade Notifier
================================
Dispatches real-time trade signals and execution alerts to Discord Webhook
and Telegram Bot directly from the FinRL-X Model Infrastructure.

Execution is asynchronous (runs in background daemon threads) to guarantee
zero latency impact on order submission to MetaTrader 5.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional, Any

from loguru import logger
from src.config.settings import settings


class TradeNotifier:
    """
    Non-blocking notifier for Telegram Bot and Discord Webhook.
    """

    def __init__(self):
        self.cfg = settings.notifications
        # Fallback to standard environment variables if not nested
        self.discord_url = (
            self.cfg.discord_webhook_url
            or os.environ.get("DISCORD_WEBHOOK_URL", "")
        ).strip()

        self.tg_token = (
            self.cfg.telegram_bot_token
            or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        ).strip()

        self.tg_chat_id = (
            self.cfg.telegram_chat_id
            or os.environ.get("TELEGRAM_CHAT_ID", "")
        ).strip()

        self.enabled = self.cfg.enable_notifications

    @property
    def has_discord(self) -> bool:
        return bool(self.discord_url and self.discord_url.startswith("http"))

    @property
    def has_telegram(self) -> bool:
        return bool(self.tg_token and self.tg_chat_id)

    # ─── Public Notification Triggers ─────────────────────────────────────────

    def notify_trade_executed(
        self,
        symbol: str,
        direction: int,
        lots: float,
        price: float,
        sl: Optional[float],
        tp: Optional[float],
        ticket: int,
        decision: Any,
    ) -> None:
        """
        Triggered when MT5Executor successfully fills a Council decision.
        Runs asynchronously in a background thread.
        """
        if not self.enabled or not self.cfg.notify_on_trade:
            return

        if not self.has_discord and not self.has_telegram:
            return

        threading.Thread(
            target=self._send_trade_signals_worker,
            args=(symbol, direction, lots, price, sl, tp, ticket, decision),
            daemon=True,
        ).start()

    def notify_position_closed(
        self,
        symbol: str,
        ticket: int,
        reason: str = "Council opposite signal",
    ) -> None:
        """
        Triggered when an opposite position is closed by the executor.
        """
        if not self.enabled or not self.cfg.notify_on_close:
            return

        if not self.has_discord and not self.has_telegram:
            return

        threading.Thread(
            target=self._send_close_worker,
            args=(symbol, ticket, reason),
            daemon=True,
        ).start()

    # ─── Workers ─────────────────────────────────────────────────────────────

    def _send_trade_signals_worker(
        self,
        symbol: str,
        direction: int,
        lots: float,
        price: float,
        sl: Optional[float],
        tp: Optional[float],
        ticket: int,
        decision: Any,
    ) -> None:
        side_str = "BUY" if direction > 0 else "SELL"
        side_emoji = "🟢" if direction > 0 else "🔴"
        regime = getattr(decision, "regime", "UNKNOWN")
        confidence = getattr(decision, "council_confidence", 0.0)
        expected_rr = getattr(decision, "expected_rr", 0.0)
        signal_val = getattr(decision, "consensus_signal", 0.0)
        risk_pct = settings.mt5.default_risk_pct * 100.0

        sl_pts = abs(price - sl) if sl else 0.0
        tp_pts = abs(tp - price) if tp else 0.0

        # 1. Send Discord Webhook Embed
        if self.has_discord:
            try:
                embed_color = 0x00FF88 if direction > 0 else 0xFF3366  # Neon green / crimson red
                discord_payload = {
                    "username": "FinRL-X Trading Council",
                    "avatar_url": "https://raw.githubusercontent.com/ElMoorish/FinRL-X-MT5/main/docs/assets/finrl_x_icon.png",
                    "embeds": [
                        {
                            "title": f"🏛️ Trade Executed: {side_emoji} {side_str} {symbol}",
                            "description": f"**K-Dense Council Consensus:** `{signal_val:+.3f}` (Conviction: `{confidence:.1%}`)",
                            "color": embed_color,
                            "fields": [
                                {"name": "Symbol", "value": f"**{symbol}**", "inline": True},
                                {"name": "Action", "value": f"{side_emoji} **{side_str}**", "inline": True},
                                {"name": "Volume", "value": f"`{lots:.2f}` Lots", "inline": True},
                                {"name": "Entry Price", "value": f"`{price:.2f}`", "inline": True},
                                {"name": "Stop Loss", "value": f"`{sl:.2f}` (-{sl_pts:.1f} pts)" if sl else "`None`", "inline": True},
                                {"name": "Take Profit", "value": f"`{tp:.2f}` (+{tp_pts:.1f} pts)" if tp else "`None`", "inline": True},
                                {"name": "Risk / Reward", "value": f"**1 : {expected_rr:.2f}**", "inline": True},
                                {"name": "Market Regime", "value": f"`{regime}`", "inline": True},
                                {"name": "Risk Profile", "value": f"`{risk_pct:.2f}%` Base Risk", "inline": True},
                                {"name": "MT5 Order Ticket", "value": f"`#{ticket}`", "inline": True},
                            ],
                            "footer": {
                                "text": "FinRL-X Multi-Agent MoE Infrastructure • Ultra-Conservative Profile"
                            },
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                }
                self._http_post(self.discord_url, discord_payload)
                logger.info(f"📢 Discord trade alert sent for #{ticket} ({symbol})")
            except Exception as e:
                logger.warning(f"Failed sending Discord trade notification: {e}")

        # 2. Send Telegram Bot Message
        if self.has_telegram:
            try:
                tg_text = (
                    f"🏛️ <b>FinRL-X Council Order Fired!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{side_emoji} <b>Action:</b> <b>{side_str} {symbol}</b>\n"
                    f"📦 <b>Volume:</b> <code>{lots:.2f} Lots</code> ({risk_pct:.2f}% Risk)\n"
                    f"💵 <b>Entry Price:</b> <code>{price:.2f}</code>\n"
                    f"🛡️ <b>Stop Loss:</b> <code>{sl:.2f}</code> (-{sl_pts:.1f} pts)\n"
                    f"🎯 <b>Take Profit:</b> <code>{tp:.2f}</code> (+{tp_pts:.1f} pts)\n"
                    f"⚖️ <b>Expected R:R:</b> <code>1 : {expected_rr:.2f}</code>\n"
                    f"🌡️ <b>Regime:</b> <code>{regime}</code> (Conf: {confidence:.1%})\n"
                    f"🎫 <b>MT5 Ticket:</b> <code>#{ticket}</code>\n"
                    f"⏰ <b>UTC Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"<i>Sent from FinRL-X Model Infrastructure</i>"
                )
                tg_url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
                tg_payload = {
                    "chat_id": self.tg_chat_id,
                    "text": tg_text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                }
                self._http_post(tg_url, tg_payload)
                logger.info(f"📢 Telegram trade alert sent for #{ticket} ({symbol})")
            except Exception as e:
                logger.warning(f"Failed sending Telegram trade notification: {e}")

    def _send_close_worker(self, symbol: str, ticket: int, reason: str) -> None:
        msg = f"🔄 <b>FinRL-X Position Closed</b>\nTicket: <code>#{ticket}</code> on <b>{symbol}</b>\nReason: <i>{reason}</i>"
        if self.has_telegram:
            try:
                tg_url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
                self._http_post(tg_url, {"chat_id": self.tg_chat_id, "text": msg, "parse_mode": "HTML"})
            except Exception as e:
                logger.warning(f"Telegram close alert failed: {e}")

        if self.has_discord:
            try:
                discord_payload = {
                    "username": "FinRL-X Trading Council",
                    "embeds": [
                        {
                            "title": f"🔄 Position Closed: #{ticket} ({symbol})",
                            "description": f"**Reason:** {reason}",
                            "color": 0x99AAB5,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                }
                self._http_post(self.discord_url, discord_payload)
            except Exception as e:
                logger.warning(f"Discord close alert failed: {e}")

    # ─── HTTP Utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _http_post(url: str, payload: dict, timeout: int = 10) -> None:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "FinRL-X-Council/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status not in (200, 204):
                raise RuntimeError(f"HTTP status {resp.status}: {resp.read().decode('utf-8', errors='ignore')}")


if __name__ == "__main__":
    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description="Test FinRL-X Telegram/Discord Trade Notifier")
    parser.add_argument("--test", action="store_true", help="Send a test trade alert")
    args = parser.parse_args()

    notifier = TradeNotifier()
    print("=== FinRL-X Notifier Status ===")
    print(f"Discord Configured:  {notifier.has_discord} ({notifier.discord_url[:30]}...)" if notifier.has_discord else "Discord Configured:  False")
    print(f"Telegram Configured: {notifier.has_telegram} (Chat ID: {notifier.tg_chat_id})" if notifier.has_telegram else "Telegram Configured: False")

    if args.test:
        print("\nSending simulated trade signal...")
        mock_decision = SimpleNamespace(
            regime="BULL",
            council_confidence=0.85,
            expected_rr=2.75,
            consensus_signal=0.42,
        )
        notifier._send_trade_signals_worker(
            symbol="NAS100.x",
            direction=1,
            lots=0.01,
            price=29450.00,
            sl=29390.00,
            tp=29615.00,
            ticket=99999999,
            decision=mock_decision,
        )
        print("Dispatch test complete. Check your Telegram/Discord channel!")
