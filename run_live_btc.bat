@echo off
TITLE FinRL-X-MT5 Live Trader - BTCUSD.x
COLOR 0B
echo Launching Live Trader for BTCUSD.x (0.25%% Fixed Risk)...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_live_trader.ps1" -Symbol "BTCUSD.x"
pause
