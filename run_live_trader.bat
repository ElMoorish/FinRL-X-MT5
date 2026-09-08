@echo off
TITLE FinRL-X-MT5 Ultra-Safe Prop Firm Live Trader
COLOR 0B
echo ======================================================================
echo    FinRL-X-MT5: Ultra-Safe Prop Firm Live Trader (NAS100.x)
echo ======================================================================
echo Attaching to active MetaTrader 5 terminal...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_live_trader.ps1"
pause
