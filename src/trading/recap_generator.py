"""
FinRL-X Automated Trade Recap Generator
========================================
Scans MT5 deal history and generates comprehensive Daily and Weekly
Performance Recaps with Win Rate, Net PnL, Profit Factor, Best/Worst Trade,
and Session Breakdown.

Dispatches formatted reports directly to Telegram and Discord Webhook,
and saves markdown audit summaries in reports/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Any

import numpy as np
import MetaTrader5 as mt5
from loguru import logger

from src.config.settings import settings
from src.trading.trade_notifier import TradeNotifier


class RecapGenerator:
    """
    Scans live MT5 history deals and generates Daily & Weekly Recaps.
    """

    def __init__(self, notifier: Optional[TradeNotifier] = None):
        self.notifier = notifier or TradeNotifier()
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ─── Public API ───────────────────────────────────────────────────────────

    def generate_daily_recap(
        self,
        date: Optional[datetime] = None,
        symbol: Optional[str] = None,
        magic: Optional[int] = settings.mt5.magic_number,
        dispatch: bool = True,
    ) -> dict:
        """
        Generate recap for a specific day (default: today UTC, Council trades only).
        """
        now = datetime.now(timezone.utc)
        target_date = date or now
        start_dt = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        # Don't query into the future
        if end_dt > now:
            end_dt = now

        title = f"Daily Recap — {start_dt.strftime('%A, %B %d, %Y')}"
        recap_data = self._compute_recap_metrics(start_dt, end_dt, symbol, magic_filter=magic, period_type="daily")

        if dispatch:
            self._dispatch_recap(recap_data, title, period_label=start_dt.strftime("%Y-%m-%d"))

        return recap_data

    def generate_weekly_recap(
        self,
        weeks_back: int = 0,
        symbol: Optional[str] = None,
        magic: Optional[int] = settings.mt5.magic_number,
        dispatch: bool = True,
    ) -> dict:
        """
        Generate recap for the current or past week (Monday to Sunday UTC, Council trades only).
        """
        now = datetime.now(timezone.utc)
        # Monday of target week
        monday = now - timedelta(days=now.weekday() + (7 * weeks_back))
        start_dt = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = start_dt + timedelta(days=6, hours=23, minutes=59, seconds=59)
        if end_dt > now:
            end_dt = now

        title = f"Weekly Recap — Week {start_dt.strftime('%W')} ({start_dt.strftime('%b %d')} → {end_dt.strftime('%b %d, %Y')})"
        recap_data = self._compute_recap_metrics(start_dt, end_dt, symbol, magic_filter=magic, period_type="weekly")

        if dispatch:
            self._dispatch_recap(recap_data, title, period_label=f"Week_{start_dt.strftime('%Y_W%W')}")

        return recap_data

    # ─── Metrics Computation ──────────────────────────────────────────────────

    def _compute_recap_metrics(
        self,
        start_dt: datetime,
        end_dt: datetime,
        symbol_filter: Optional[str] = None,
        magic_filter: Optional[int] = settings.mt5.magic_number,
        period_type: str = "daily",
    ) -> dict:
        """Scan MT5 deals between start_dt and end_dt filtered by Council magic number."""
        if not mt5.initialize():
            logger.error("Failed to connect to MT5 for recap generation")
            return {"error": "MT5 not connected"}

        deals = mt5.history_deals_get(start_dt, end_dt)
        closed_deals = []

        if deals:
            for d in deals:
                # DEAL_ENTRY_OUT (1) = trade close deal
                if d.entry == mt5.DEAL_ENTRY_OUT:
                    if symbol_filter and d.symbol != symbol_filter:
                        continue
                    if magic_filter is not None and d.magic != magic_filter:
                        continue
                    closed_deals.append(d)

        acc = mt5.account_info()
        current_equity = acc.equity if acc else 10000.0
        current_balance = acc.balance if acc else 10000.0

        if not closed_deals:
            return {
                "period_type": period_type,
                "start_dt": start_dt,
                "end_dt": end_dt,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "net_pnl": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "current_equity": current_equity,
                "current_balance": current_balance,
                "trades": [],
                "session_breakdown": {},
            }

        pnls = [d.profit + d.swap + d.commission for d in closed_deals]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_trades = len(pnls)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        net_pnl = sum(pnls)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        avg_win = (gross_profit / win_count) if win_count > 0 else 0.0
        avg_loss = (gross_loss / loss_count) if loss_count > 0 else 0.0
        best_trade = max(pnls) if pnls else 0.0
        worst_trade = min(pnls) if pnls else 0.0

        # Session Breakdown
        session_stats = {"Asian": {"trades": 0, "pnl": 0.0, "wins": 0},
                         "London": {"trades": 0, "pnl": 0.0, "wins": 0},
                         "New York": {"trades": 0, "pnl": 0.0, "wins": 0},
                         "Off-Hours": {"trades": 0, "pnl": 0.0, "wins": 0}}

        for d in closed_deals:
            dt = datetime.fromtimestamp(d.time, tz=timezone.utc)
            h = dt.hour + dt.minute / 60.0
            sess = "Asian" if 0.0 <= h < 7.0 else "London" if 7.0 <= h < 13.5 else "New York" if 13.5 <= h < 21.0 else "Off-Hours"
            p = d.profit + d.swap + d.commission
            session_stats[sess]["trades"] += 1
            session_stats[sess]["pnl"] += p
            if p > 0:
                session_stats[sess]["wins"] += 1

        trade_details = []
        for d in closed_deals:
            dt = datetime.fromtimestamp(d.time, tz=timezone.utc)
            pnl = d.profit + d.swap + d.commission
            trade_details.append({
                "ticket": d.ticket,
                "symbol": d.symbol,
                "time": dt.strftime("%H:%M:%S"),
                "volume": d.volume,
                "pnl": pnl,
                "magic": d.magic,
            })

        return {
            "period_type": period_type,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "total_trades": total_trades,
            "wins": win_count,
            "losses": loss_count,
            "win_rate": win_rate,
            "net_pnl": net_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "current_equity": current_equity,
            "current_balance": current_balance,
            "trades": trade_details,
            "session_breakdown": session_stats,
        }

    # ─── Formatting & Dispatch ────────────────────────────────────────────────

    def _dispatch_recap(self, data: dict, title: str, period_label: str) -> None:
        """Send formatted reports to Telegram, Discord, and save to markdown file."""
        if "error" in data:
            logger.warning(f"Skipping dispatch due to error: {data['error']}")
            return

        # 1. Save Markdown report
        filename = f"recap_{data['period_type']}_{period_label}.md"
        report_path = self.reports_dir / filename
        md_content = self._build_markdown_report(data, title)
        report_path.write_text(md_content, encoding="utf-8")
        logger.info(f"💾 Saved {data['period_type']} recap report → {report_path}")

        # 2. Dispatch to Discord
        if self.notifier.has_discord:
            self._send_discord_recap(data, title)

        # 3. Dispatch to Telegram
        if self.notifier.has_telegram:
            self._send_telegram_recap(data, title)

    def _send_discord_recap(self, data: dict, title: str) -> None:
        net_pnl = data["net_pnl"]
        is_profit = net_pnl >= 0
        embed_color = 0x00FF88 if is_profit else 0xFF3366  # Neon green / Red
        pnl_sign = "+" if net_pnl >= 0 else ""

        fields = [
            {"name": "💰 Net PnL", "value": f"**`{pnl_sign}${net_pnl:,.2f}`**", "inline": True},
            {"name": "🎯 Win Rate", "value": f"**`{data['win_rate']:.1f}%`** ({data['wins']}W / {data['losses']}L)", "inline": True},
            {"name": "⚖️ Profit Factor", "value": f"**`{data['profit_factor']:.2f}`**", "inline": True},
            {"name": "📈 Gross Profit", "value": f"`+${data['gross_profit']:,.2f}`", "inline": True},
            {"name": "📉 Gross Loss", "value": f"`-${data['gross_loss']:,.2f}`", "inline": True},
            {"name": "📦 Total Trades", "value": f"`{data['total_trades']}`", "inline": True},
            {"name": "🏆 Best Trade", "value": f"`+${data['best_trade']:,.2f}`", "inline": True},
            {"name": "⚠️ Worst Trade", "value": f"`${data['worst_trade']:,.2f}`", "inline": True},
            {"name": "📊 Avg Win / Loss", "value": f"`+${data['avg_win']:.2f} / -${data['avg_loss']:.2f}`", "inline": True},
            {"name": "🏛️ Current Equity", "value": f"`${data['current_equity']:,.2f}`", "inline": True},
            {"name": "💵 Current Balance", "value": f"`${data['current_balance']:,.2f}`", "inline": True},
            {"name": "🛡️ Risk Mode", "value": f"`{settings.mt5.default_risk_pct*100:.2f}% Base Risk`", "inline": True},
        ]

        # Add session breakdown field
        sess_lines = []
        for sess_name, s in data["session_breakdown"].items():
            if s["trades"] > 0:
                s_wr = (s["wins"] / s["trades"] * 100) if s["trades"] > 0 else 0.0
                sess_lines.append(f"• **{sess_name}:** {s['trades']} trds | `{s_wr:.0f}% WR` | `${s['pnl']:+,.2f}`")
        if sess_lines:
            fields.append({"name": "🌍 Session Performance", "value": "\n".join(sess_lines), "inline": False})

        discord_payload = {
            "username": "FinRL-X Performance Journal",
            "avatar_url": "https://raw.githubusercontent.com/ElMoorish/FinRL-X-MT5/main/docs/assets/finrl_x_icon.png",
            "embeds": [
                {
                    "title": f"📊 {title}",
                    "description": f"Automated performance audit from **FinRL-X Model Infrastructure**.",
                    "color": embed_color,
                    "fields": fields,
                    "footer": {
                        "text": "FinRL-X Institutional MoE • Automated Performance Journal"
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        }

        try:
            self.notifier._http_post(self.notifier.discord_url, discord_payload)
            logger.info(f"📢 Discord {data['period_type']} recap sent successfully")
        except Exception as e:
            logger.warning(f"Failed sending Discord recap: {e}")

    def _send_telegram_recap(self, data: dict, title: str) -> None:
        net_pnl = data["net_pnl"]
        pnl_sign = "+" if net_pnl >= 0 else ""
        pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"

        sess_lines = []
        for sess_name, s in data["session_breakdown"].items():
            if s["trades"] > 0:
                s_wr = (s["wins"] / s["trades"] * 100) if s["trades"] > 0 else 0.0
                sess_lines.append(f"  • <b>{sess_name}:</b> <code>{s['trades']} trds</code> | <code>{s_wr:.0f}% WR</code> | <code>${s['pnl']:+,.2f}</code>")
        session_text = "\n".join(sess_lines) if sess_lines else "  • <i>No session trades</i>"

        tg_text = (
            f"📊 <b>{title}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{pnl_emoji} <b>Net PnL:</b> <b>{pnl_sign}${net_pnl:,.2f}</b>\n"
            f"🎯 <b>Win Rate:</b> <b>{data['win_rate']:.1f}%</b> ({data['wins']}W / {data['losses']}L)\n"
            f"⚖️ <b>Profit Factor:</b> <code>{data['profit_factor']:.2f}</code>\n"
            f"📦 <b>Total Trades:</b> <code>{data['total_trades']}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 <b>Gross Profit:</b> <code>+${data['gross_profit']:,.2f}</code>\n"
            f"📉 <b>Gross Loss:</b> <code>-${data['gross_loss']:,.2f}</code>\n"
            f"🏆 <b>Best Trade:</b> <code>+${data['best_trade']:,.2f}</code>\n"
            f"⚠️ <b>Worst Trade:</b> <code>${data['worst_trade']:,.2f}</code>\n"
            f"📊 <b>Avg Win / Loss:</b> <code>+${data['avg_win']:.2f} / -${data['avg_loss']:.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌍 <b>Session Breakdown:</b>\n"
            f"{session_text}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏛️ <b>Account Equity:</b> <code>${data['current_equity']:,.2f}</code>\n"
            f"🛡️ <b>Risk Profile:</b> <code>{settings.mt5.default_risk_pct*100:.2f}% Base Risk</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>FinRL-X Institutional Model Infrastructure</i>"
        )

        tg_url = f"https://api.telegram.org/bot{self.notifier.tg_token}/sendMessage"
        tg_payload = {
            "chat_id": self.notifier.tg_chat_id,
            "text": tg_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            self.notifier._http_post(tg_url, tg_payload)
            logger.info(f"📢 Telegram {data['period_type']} recap sent to group successfully")
        except Exception as e:
            logger.warning(f"Failed sending Telegram recap: {e}")

    def _build_markdown_report(self, data: dict, title: str) -> str:
        pnl_sign = "+" if data['net_pnl'] >= 0 else ""
        lines = [
            f"# {title}",
            "",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Period:** {data['start_dt'].strftime('%Y-%m-%d %H:%M')} → {data['end_dt'].strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Executive Summary",
            "",
            f"| Metric | Result |",
            f"| :--- | :--- |",
            f"| **Net PnL** | **{pnl_sign}${data['net_pnl']:,.2f}** |",
            f"| **Win Rate** | **{data['win_rate']:.1f}%** ({data['wins']} Wins / {data['losses']} Losses) |",
            f"| **Profit Factor** | **{data['profit_factor']:.2f}** |",
            f"| **Gross Profit** | +${data['gross_profit']:,.2f} |",
            f"| **Gross Loss** | -${data['gross_loss']:,.2f} |",
            f"| **Total Closed Trades** | {data['total_trades']} |",
            f"| **Average Win** | +${data['avg_win']:.2f} |",
            f"| **Average Loss** | -${data['avg_loss']:.2f} |",
            f"| **Best Trade** | +${data['best_trade']:,.2f} |",
            f"| **Worst Trade** | ${data['worst_trade']:,.2f} |",
            f"| **Account Equity** | ${data['current_equity']:,.2f} |",
            f"| **Account Balance** | ${data['current_balance']:,.2f} |",
            "",
            "## Session Performance Breakdown",
            "",
            "| Session | Trades | Wins | Losses | Win Rate | Net PnL |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        for s_name, s in data["session_breakdown"].items():
            wr = (s['wins'] / s['trades'] * 100.0) if s['trades'] > 0 else 0.0
            l = s['trades'] - s['wins']
            lines.append(f"| **{s_name}** | {s['trades']} | {s['wins']} | {l} | {wr:.1f}% | ${s['pnl']:+,.2f} |")

        lines.extend([
            "",
            "## Closed Deals Log",
            "",
            "| Ticket | Symbol | Time (UTC) | Volume | PnL ($) |",
            "| :--- | :--- | :---: | :---: | :--- |",
        ])

        for t in data["trades"]:
            lines.append(f"| #{t['ticket']} | {t['symbol']} | {t['time']} | {t['volume']} | ${t['pnl']:+,.2f} |")

        return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinRL-X Automated Trade Recap Generator")
    parser.add_argument("--daily", action="store_true", help="Generate today's Daily Recap")
    parser.add_argument("--weekly", action="store_true", help="Generate current Weekly Recap")
    parser.add_argument("--days", type=int, default=0, help="Generate recap for N days back (0=today)")
    parser.add_argument("--weeks", type=int, default=0, help="Generate recap for N weeks back (0=this week)")
    parser.add_argument("--symbol", type=str, default=None, help="Filter by specific symbol (e.g. NAS100.x)")
    parser.add_argument("--no-dispatch", action="store_true", help="Do not send to Discord/Telegram (preview only)")

    args = parser.parse_args()
    gen = RecapGenerator()

    if args.weekly:
        print(f"Generating Weekly Recap (weeks_back={args.weeks})...")
        res = gen.generate_weekly_recap(weeks_back=args.weeks, symbol=args.symbol, dispatch=not args.no_dispatch)
        print(f"Weekly Recap complete: {res['total_trades']} trades | Net PnL: ${res['net_pnl']:,.2f} | Win Rate: {res['win_rate']:.1f}%")
    else:
        target_date = datetime.now(timezone.utc) - timedelta(days=args.days) if args.days > 0 else None
        print(f"Generating Daily Recap for {'today' if not target_date else target_date.date()}...")
        res = gen.generate_daily_recap(date=target_date, symbol=args.symbol, dispatch=not args.no_dispatch)
        print(f"Daily Recap complete: {res['total_trades']} trades | Net PnL: ${res['net_pnl']:,.2f} | Win Rate: {res['win_rate']:.1f}%")
