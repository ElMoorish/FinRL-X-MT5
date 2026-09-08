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


def get_active_council(symbol: str) -> Optional[Council]:
    if symbol not in state.councils:
        try:
            state.councils[symbol] = Council().load(symbol)
            logger.info(f"Loaded Council for {symbol} into dashboard cache")
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


@app.get("/api/bars")
async def get_chart_bars(
    symbol: str = Query("NAS100.x"),
    count: int = Query(150, ge=30, le=500),
):
    """Retrieve M5 historical bars formatted for TradingView Lightweight-Charts."""
    if not mt5.initialize():
        return JSONResponse({"error": "MT5 not connected"}, status_code=503)

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count)
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

    tick = mt5.symbol_info_tick(symbol)
    quote = {
        "bid": tick.bid if tick else 0.0,
        "ask": tick.ask if tick else 0.0,
        "spread": (tick.ask - tick.bid) if tick else 0.0,
    }

    return {"symbol": symbol, "bars": bars, "quote": quote}


@app.get("/api/council/decision")
async def get_council_decision(symbol: str = Query("NAS100.x")):
    """Run real-time Council deliberation and generate human-readable Chain of Thought."""
    if not mt5.initialize():
        return JSONResponse({"error": "MT5 not connected"}, status_code=503)

    council = get_active_council(symbol)
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

    state.latest_decision = decision
    state.latest_cot = cot
    state.active_symbol = symbol

    return cot


@app.get("/api/positions")
async def get_positions():
    """Retrieve open positions and recent closed deals."""
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

    # Recent deals
    now = datetime.now(timezone.utc)
    deals_raw = mt5.history_deals_get(now - timedelta(days=2), now) or []
    deals = []
    for d in deals_raw[-15:]:
        if d.entry == mt5.DEAL_ENTRY_OUT:
            deals.append({
                "ticket": d.ticket,
                "order": d.order,
                "symbol": d.symbol,
                "type": "SELL" if d.type == mt5.DEAL_TYPE_SELL else "BUY",
                "volume": d.volume,
                "price": d.price,
                "profit": round(d.profit, 2),
                "commission": round(d.commission, 2),
                "time": datetime.fromtimestamp(d.time).strftime("%Y-%m-%d %H:%M:%S"),
            })

    return {"open_positions": positions, "recent_deals": list(reversed(deals))}


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
    symbol = state.active_symbol
    try:
        while True:
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
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)


# Mount Static Files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
