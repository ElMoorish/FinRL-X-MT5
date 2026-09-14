# ==============================================================================
# FinRL-X-MT5: Ultra-Safe Live Prop Firm Trading Runner
# ==============================================================================
# Automatically attaches to active MT5 terminal, monitors connection,
# runs the K-Dense Council for NAS100.x on M5 candle closes, and auto-restarts
# if broker connection temporarily drops.
# ==============================================================================

param(
    [string]$Symbol = "BTCUSD.x"
)

$PythonPath = "C:\Users\aitsi\AppData\Local\Programs\Python\Python311\python.exe"
$WorkingDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location $WorkingDir
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "FinRL-X-MT5: High-Conviction Prop Firm Live Trader" -ForegroundColor Cyan
Write-Host "Symbol: $Symbol | Risk: 0.25% Fixed | Phase 3 Pure Regime Gating" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

while ($true) {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting Council Live Trading Loop for $Symbol..." -ForegroundColor Green
    
    & $PythonPath -m src.main_mt5 live --symbols $Symbol
    
    $ExitCode = $LASTEXITCODE
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Process exited with code: $ExitCode" -ForegroundColor Red
    Write-Host "Reconnecting and restarting live trading in 10 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
}
