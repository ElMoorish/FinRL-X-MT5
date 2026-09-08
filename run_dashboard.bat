@echo off
TITLE FinRL-X-MT5 Council Terminal Dashboard
COLOR 0A
echo ======================================================================
echo    FinRL-X-MT5: Council Terminal Dashboard
echo    URL: http://localhost:8000
echo ======================================================================
echo Launching dashboard server and opening browser...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dashboard.ps1"
pause
