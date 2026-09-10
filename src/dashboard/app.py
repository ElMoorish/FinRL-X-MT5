"""
FinRL-X-MT5 Real-Time Dashboard Server
======================================
FastAPI application providing real-time MT5 account telemetry,
60fps TradingView candlestick charting feeds, live Council
Chain-of-Thought deliberation streaming, and Prop Firm Guardian meters.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Any
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from src.config.settings import settings
from src.data.mt5_tick_fetcher import MT5TickFetcher
from src.data.correlation_fuser import CorrelationFuser
from src.council.council import Council, TradingDecision
from src.dashboard.cot_generator import ChainOfThoughtGenerator

app = FastAPI(title="FinRL-X-MT5 Council Terminal", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ─── In-Memory State Cache ───────────────────────────────────────────────────
class TerminalState:
    def __init__(self):
        self.active_symbol = "NAS100.x"
        self.latest_decision: Optional[TradingDecision] = None
        self.latest_cot: Optional[dict[str, Any]] = None
        self.councils: dict[str, Council] = {}
        self.day_start_equity: Optional[float] = None
        self.current_day: Optional[int] = None
        self.peak_equity: float = 10_000.0

state = TerminalState()


def get_active_council(symbol: str, reload: bool = False) -> Optional[Council]:
    if reload or symbol not in state.councils:
        try:
            state.councils[symbol] = Council().load(symbol)
            logger.info(f"Loaded Council for {symbol} into dashboard cache (reload={reload})")
        except Exception as e:
            logger.error(f"Failed to load Council for {symbol}: {e}")
            return None
    return state.councils.get(symbol)


# ─── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
async def get_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"status": "Dashboard frontend initializing..."})
    return FileResponse(index_file)


@app.get("/api/account")
async def get_account_state():
    """Retrieve live MT5 account telemetry."""
    if not mt5.initialize():
        return JSONResponse({"connected": False, "error": "MT5 terminal not running"}, status_code=503)

    acc = mt5.account_info()
    if acc is None:
        return JSONResponse({"connected": False, "error": "No account login detected"}, status_code=503)

    today = datetime.now(timezone.utc).day
    if state.current_day != today or state.day_start_equity is None:
        state.day_start_equity = acc.equity
        state.current_day = today

    state.peak_equity = max(state.peak_equity, acc.equity)

    daily_loss_usd = max(0.0, state.day_start_equity - acc.equity) if state.day_start_equity else 0.0
    daily_loss_pct = (daily_loss_usd / state.day_start_equity) * 100.0 if state.day_start_equity else 0.0
    total_dd_usd = max(0.0, state.peak_equity - acc.equity)
    total_dd_pct = (total_dd_usd / state.peak_equity) * 100.0 if state.peak_equity else 0.0

    return {
        "connected": True,
        "login": acc.login,
        "server": acc.server,
        "currency": acc.currency,
        "leverage": acc.leverage,
        "balance": round(acc.balance, 2),
        "equity": round(acc.equity, 2),
        "profit": round(acc.profit, 2),
        "free_margin": round(acc.margin_free, 2),
        "margin_level": round(acc.margin_level, 2) if acc.margin_level else 1000.0,
        "trade_allowed": acc.trade_allowed,
        "prop_firm": {
            "mode": "Ultra-Safe Prop Firm",
            "base_risk_pct": settings.mt5.default_risk_pct * 100.0,
            "daily_drawdown_usd": round(daily_loss_usd, 2),
            "daily_drawdown_pct": round(daily_loss_pct, 2),
            "daily_limit_pct": settings.mt5.max_daily_loss_pct * 100.0,
            "total_drawdown_usd": round(total_dd_usd, 2),
            "total_drawdown_pct": round(total_dd_pct, 2),
            "total_limit_pct": settings.mt5.max_drawdown_halt_pct * 100.0,
            "target_profit_usd": 800.0,  # 8% standard Phase 1 on $10k
            "profit_progress_pct": min(100.0, max(0.0, (acc.profit / 800.0) * 100.0)),
        }
    }


TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "D1": mt5.TIMEFRAME_D1,
}


@app.get("/api/bars")
async def get_chart_bars(
    symbol: str = Query("NAS100.x"),
    tf: str = Query("M5"),
    count: int = Query(160, ge=30, le=500),
):
    """Retrieve historical bars and H1 Macro Trend Governor line."""
    if not mt5.initialize():
        return JSONResponse({"error": "MT5 not connected"}, status_code=503)

    mt5_tf = TIMEFRAME_MAP.get(tf.upper(), mt5.TIMEFRAME_M5)
    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
    if rates is None or len(rates) == 0:
        return JSONResponse({"error": f"No rates found for {symbol}"}, status_code=404)

    bars = []
    for r in rates:
        bars.append({
            "time": int(r["time"]),  # Unix timestamp in seconds
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": int(r["tick_volume"]),
        })

    # Compute H1 Trend Governor (EMA 50 on H1)
    current_h1_ema = 0.0
    try:
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 70)
        if h1_rates is not None and len(h1_rates) >= 50:
            h1_df = pd.DataFrame(h1_rates)
            h1_df["ema50"] = h1_df["close"].ewm(span=50, adjust=False).mean()
            current_h1_ema = float(h1_df["ema50"].iloc[-1])
    except Exception as e:
        logger.debug(f"H1 EMA calculation error: {e}")

    tick = mt5.symbol_info_tick(symbol)
    quote = {
        "bid": tick.bid if tick else 0.0,
        "ask": tick.ask if tick else 0.0,
        "spread": (tick.ask - tick.bid) if tick else 0.0,
    }

    return {
        "symbol": symbol,
        "timeframe": tf.upper(),
        "bars": bars,
        "quote": quote,
        "h1_ema": round(current_h1_ema, 2),
    }


@app.get("/api/council/decision")
async def get_council_decision(
    symbol: str = Query("NAS100.x"),
    force: bool = Query(False)
):
    """Retrieve synchronized live Council decision or run real-time deliberation."""
    state_file = Path("data") / f"live_council_state_{symbol}.json"

    # If not forcing recalculation and live trader state exists, serve synchronized live state
    if not force and state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                cached_cot = json.load(f)
            ts = cached_cot.get("timestamp_utc")
            if ts:
                age_sec = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()
                if age_sec < 900:  # Valid within last 15 minutes (3 M5 bars)
                    state.active_symbol = symbol
                    state.latest_cot = cached_cot
                    return cached_cot
        except Exception as e:
            logger.warning(f"Could not read cached live council state: {e}")

    if not mt5.initialize():
        return JSONResponse({"error": "MT5 not connected"}, status_code=503)

    council = get_active_council(symbol, reload=force)
    if not council:
        return JSONResponse({"error": f"Council not found for {symbol}"}, status_code=404)

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=48)

    with MT5TickFetcher() as fetcher:
        fuser = CorrelationFuser(tick_fetcher=fetcher)
        features = fuser.build_feature_store(symbol, start_dt, end_dt)

    if features.is_empty():
        return JSONResponse({"error": "Could not extract features"}, status_code=500)

    decision = council.decide(features, symbol)
    cot = ChainOfThoughtGenerator.generate(decision)
    cot["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    cot["source"] = "DASHBOARD_ON_DEMAND"

    # Save to state file so all clients stay in sync
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(cot, f, indent=2)
    except Exception:
        pass

    state.latest_decision = decision
    state.latest_cot = cot
    state.active_symbol = symbol

    return cot


@app.get("/api/positions")
async def get_positions():
    """Retrieve open positions, recent closed deals, and daily session performance."""
    if not mt5.initialize():
        return JSONResponse({"error": "MT5 not connected"}, status_code=503)

    positions_raw = mt5.positions_get() or []
    positions = []
    for p in positions_raw:
        positions.append({
            "ticket": p.ticket,
            "symbol": p.symbol,
            "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
            "volume": p.volume,
            "price_open": p.price_open,
            "price_current": p.price_current,
            "sl": p.sl,
            "tp": p.tp,
            "profit": round(p.profit, 2),
            "time": datetime.fromtimestamp(p.time).strftime("%Y-%m-%d %H:%M:%S"),
            "comment": p.comment,
        })

    # Recent deals & session statistics
    now = datetime.now(timezone.utc)
    today_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    deals_raw = mt5.history_deals_get(now - timedelta(days=7), now) or []
    deals = []
    today_pnl = 0.0
    today_wins = 0
    today_losses = 0
    today_deals_count = 0

    for d in deals_raw:
        if d.entry == mt5.DEAL_ENTRY_OUT:
            d_time = datetime.fromtimestamp(d.time, tz=timezone.utc)
            deal_data = {
                "ticket": d.ticket,
                "order": d.order,
                "symbol": d.symbol,
                "type": "SELL" if d.type == mt5.DEAL_TYPE_SELL else "BUY",
                "volume": d.volume,
                "price": d.price,
                "profit": round(d.profit, 2),
                "commission": round(d.commission, 2),
                "time": d_time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            deals.append(deal_data)

            if d_time >= today_start:
                today_pnl += d.profit
                today_deals_count += 1
                if d.profit > 0:
                    today_wins += 1
                elif d.profit < 0:
                    today_losses += 1

    today_win_rate = (today_wins / (today_wins + today_losses) * 100.0) if (today_wins + today_losses) > 0 else 0.0

    session_stats = {
        "today_pnl": round(today_pnl, 2),
        "today_trades": today_deals_count,
        "today_wins": today_wins,
        "today_losses": today_losses,
        "today_win_rate": round(today_win_rate, 1),
    }

    return {
        "open_positions": positions,
        "recent_deals": list(reversed(deals))[:30],
        "session_stats": session_stats,
    }


# ─── WebSocket Live Stream ───────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    last_cot_ts = ""
    try:
        while True:
            symbol = state.active_symbol
            # Poll MT5 for current quote and account
            if mt5.initialize():
                tick = mt5.symbol_info_tick(symbol)
                acc = mt5.account_info()
                if tick and acc:
                    await websocket.send_json({
                        "type": "HEARTBEAT",
                        "timestamp": int(datetime.now().timestamp()),
                        "symbol": symbol,
                        "bid": tick.bid,
                        "ask": tick.ask,
                        "spread": round((tick.ask - tick.bid) / 0.01, 1),
                        "equity": round(acc.equity, 2),
                        "balance": round(acc.balance, 2),
                        "profit": round(acc.profit, 2),
                    })

                # Broadcast live Council deliberation updates when updated
                state_file = Path("data") / f"live_council_state_{symbol}.json"
                if state_file.exists():
                    try:
                        with open(state_file, "r", encoding="utf-8") as f:
                            live_cot = json.load(f)
                        ts = live_cot.get("timestamp_utc", "")
                        if ts and ts != last_cot_ts:
                            last_cot_ts = ts
                            await websocket.send_json({
                                "type": "COUNCIL_DECISION",
                                "symbol": symbol,
                                "cot": live_cot,
                            })
                    except Exception:
                        pass
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)


# Mount Static Files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
