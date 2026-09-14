@echo off
TITLE FinRL-X-MT5 All-in-One Launcher
echo ======================================================================
echo    FinRL-X-MT5: Launching Live Trader and Terminal Dashboard
echo ======================================================================
echo Starting Dashboard on http://localhost:8000 ...
start "FinRL Dashboard" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dashboard.ps1"
timeout /t 2 /nobreak >nul
echo Starting Live Trader for BTCUSD.x ...
start "FinRL Live Trader" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_live_trader.ps1" -Symbol "BTCUSD.x"
echo.
echo Both services are now running in their own windows.
echo You can close this launcher window.
timeout /t 5
