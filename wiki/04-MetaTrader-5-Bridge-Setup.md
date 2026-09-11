# 04. MetaTrader 5 Bridge Setup & Live Deployment Guide

This guide details the complete configuration required to bridge the **FinRL-X Python Engine** to your **MetaTrader 5 terminal** for automated live order execution, risk monitoring, and webhook alerts.

---

## 💻 System Prerequisites

- **Operating System:** Windows 10 / 11 or Windows Server 2022 (x64)
- **MetaTrader 5:** Terminal Build 6000 or higher (Official MetaQuotes or Broker-Branded)
- **Python:** Version 3.10 or 3.11 (64-bit required for PyTorch & MT5 wheels)
- **Hardware Minimum:** 4 vCPUs, 8 GB RAM (16 GB recommended for TimesFM inference)

---

## 🛠️ Step 1: MT5 Terminal Configuration

Before launching the Python scripts, your MetaTrader 5 desktop client must be authorized to permit external IPC (Inter-Process Communication) and Expert Advisor automated routing:

1. Open MetaTrader 5.
2. In the top navigation bar, click **Tools** $\rightarrow$ **Options** (or press `Ctrl + O`).
3. Switch to the **Expert Advisors** tab.
4. Configure the settings exactly as follows:
   - ✅ **Allow automated trading**
   - ⬜ *Disable automated trading when the account has been changed* (uncheck to prevent disconnections)
   - ⬜ *Disable automated trading when the profile has been changed* (uncheck)
   - ✅ **Allow WebRequest for listed URL**
5. Ensure the **Algo Trading** master toggle on the main toolbar is switched **ON** (Green icon).

---

## 📦 Step 2: Environment Setup & Dependencies

Open PowerShell in administrator mode and navigate to the project directory:

```powershell
# 1. Navigate to repo
cd c:\Users\aitsi\Desktop\FinRL-X-MT5

# 2. Create isolated Python 3.10+ virtual environment
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔐 Step 3: Environment Credentials (`.env`)

Copy the template configuration file into your active `.env`:

```powershell
cp .env.example .env
```

Edit `.env` with your broker credentials and risk preferences:

```ini
# MetaTrader 5 Connection
MT5_LOGIN=10557430
MT5_PASSWORD=YourInstitutionalPasswordHere
MT5_SERVER=GoatFunded-Server3
MT5_PATH="C:\Program Files\MetaTrader 5\terminal64.exe"

# Quantitative Risk Ceiling
DEFAULT_RISK_PCT=0.0050      # 0.50% Capital Risk Ceiling per trade
MAX_CONSECUTIVE_LOSSES=3     # Auto-halt trading if 3 consecutive losses occur
MIN_INDEX_STOP_PTS=60.0      # 60.0-point minimum breathing buffer on NAS100/US30

# High-Conviction Consensus Threshold
MIN_CONSENSUS_THRESHOLD=0.70 # Minimum 70% agreement across Council specialists

# Webhook Alert Dispatch
TELEGRAM_BOT_TOKEN="your_bot_token_here"
TELEGRAM_CHAT_ID="your_telegram_chat_id"
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/your_webhook_here"
```

---

## 🚀 Step 4: Launching the Live Trading Engine

Launch the engine using the dedicated PowerShell startup script:

```powershell
# Live execution on NAS100.x with 0.50% risk ceiling
.\run_live_trader.ps1 -Symbols "NAS100.x" -RiskPct 0.0050
```

### Script Execution Parameters:
- `-Symbols`: Comma-separated symbol ticker list (e.g. `"NAS100.x,US30.x"`).
- `-RiskPct`: Capital risk allocation fraction (default: `0.0050` for 0.50%).
- `-DryRun`: Run the council and log decisions without placing orders in MT5.

---

## 🩺 Step 5: Verification & Health Check

When launched, verify the console output demonstrates clean initialization:

```text
2026-09-11 16:15:00.120 | INFO | MT5Executor: Terminal initialized successfully on GoatFunded-Server3
2026-09-11 16:15:00.245 | INFO | Account Info: Equity=$9,805.56 | Balance=$9,805.56 | Leverage=1:100
2026-09-11 16:15:00.312 | INFO | Symbol Verified: NAS100.x | Digits=2 | Point=0.01 | ContractSize=10.0
2026-09-11 16:15:00.380 | INFO | Council Engine: 5 Specialists loaded and synchronized to M5 timeframe
2026-09-11 16:15:00.410 | INFO | Live Engine listening for M5 bar close events...
```

If `ContractSize` outputs anything other than your broker's actual index multiplier, the engine will safely halt order generation to protect your balance.
