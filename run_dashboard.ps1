# ==============================================================================
# FinRL-X-MT5: Interactive Council Terminal Dashboard Launcher
# ==============================================================================
$PythonPath = "C:\Users\aitsi\AppData\Local\Programs\Python\Python311\python.exe"
$WorkingDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location $WorkingDir
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "FinRL-X-MT5: Council Terminal Dashboard" -ForegroundColor Cyan
Write-Host "URL: http://localhost:8000" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan

# Open browser automatically after 1 second
Start-Process "http://localhost:8000"

& $PythonPath -m src.main_mt5 dashboard --port 8000
