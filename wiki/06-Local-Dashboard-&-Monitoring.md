# 06. Local Monitoring Dashboard & Telemetry Server

The **FinRL-X Institutional Dashboard** provides local, real-time observability over open positions, Council voting deliberations, account equity curves, and Commitment of Traders (COT) institutional sentiment.

It runs locally on your machine via an asynchronous FastAPI backend and a zero-dependency lightweight web interface.

---

## 🖥️ Dashboard Architecture & Endpoints

```text
┌─────────────────────────────────────────────────────────────┐
│                 FASTAPI SERVER (PORT 8000)                  │
├───────────────────┬─────────────────────────────────────────┤
│ Endpoint          │ Payload Description                     │
├───────────────────┼─────────────────────────────────────────┤
│ GET /api/account  │ Equity, balance, margin, floating P&L   │
│ GET /api/positions│ Active open orders, tickets, SL/TP levels│
│ GET /api/council  │ Live 5-specialist voting consensus      │
│ GET /api/cot      │ Institutional CFTC smart-money sentiment│
│ WS  /ws/stream    │ Real-time tick and order-flow streaming │
└───────────────────┴─────────────────────────────────────────┘
```

---

## 🚀 Launching the Dashboard Server

To start the dashboard locally alongside your trading engine:

```powershell
# In a separate PowerShell terminal:
cd c:\Users\aitsi\Desktop\FinRL-X-MT5
.\.venv\Scripts\Activate.ps1

# Launch FastAPI server on port 8000
python -m src.dashboard.app --port 8000
```

Once running, navigate to **`http://127.0.0.1:8000`** in any web browser.

---

## 📊 Core Dashboard Features

### 1. Real-Time Account Telemetry
- **Live Equity Tracking:** Instant updates of balance, equity, margin level, and floating P&L directly from MetaTrader 5 IPC pipes.
- **Daily Drawdown Gauge:** A dynamic progress indicator showing distance to the 5.0% daily circuit breaker threshold.

### 2. Council Deliberation Feed
- Displays real-time confidence scores and voting states for each of the 5 Specialists on the active M5 candle:
  - **Expert 1 (SAC):** Directional action confidence ($[-1.0, +1.0]$).
  - **Expert 2 (HMM):** Active market regime (Bull, Bear, or Sideways).
  - **Expert 3 (TimesFM):** Multi-horizon volatility corridor projection.
  - **Expert 4 (Analyst):** Order-flow imbalance ratio & SHAP contribution.
  - **Expert 5 (Actuary):** Sized lot volume and Bayesian stop distances.

### 3. Live Ticket Audit & Breakeven Monitor
- Inspects active orders placed by Magic Number (`20260908`).
- Displays ticket numbers, exact fill prices, broker slippage, and dynamic breakeven indicators (showing whether +1.0R has been achieved).

### 4. Commitment of Traders (COT) Generator
- Automatically aggregates weekly CFTC Commitments of Traders data for index futures (E-mini Nasdaq 100, S&P 500).
- Calculates the **Commercial Hedger Index** and **Asset Manager Net Positioning** to establish macro bias.

---

## 🛠️ Telemetry Troubleshooting

### Issue: "MT5 Terminal Not Connected"
- **Cause:** MetaTrader 5 desktop client is closed or not logged into an active account.
- **Resolution:** Open the MT5 desktop client, log in, and verify the network connection status icon in the bottom right corner of the terminal shows green bars.

### Issue: "Port 8000 Already in Use"
- **Cause:** A previous dashboard instance is running in the background.
- **Resolution:** Launch on an alternative port:
  ```powershell
  python -m src.dashboard.app --port 8080
  ```
  Then access at `http://127.0.0.1:8080`.
