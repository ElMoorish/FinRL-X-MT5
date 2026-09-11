# ==============================================================================
# FinRL-X Prime Quant • GitHub Wiki Automated Synchronization Script
# ==============================================================================
# Usage:
#   .\sync_wiki.ps1
#
# Prerequisite:
#   Make sure the Wiki feature is enabled in your repository settings:
#   GitHub -> Settings -> Features -> Check "Wikis"
#   Then create the first page on GitHub to initialize the wiki repository.
# ==============================================================================

[CmdletBinding()]
param (
    [string]$WikiRepoUrl = "https://github.com/ElMoorish/FinRL-X-MT5.wiki.git",
    [string]$SourceDir = "",
    [string]$TempDir = ""
)

$ScriptBase = if (![string]::IsNullOrWhiteSpace($PSScriptRoot)) { $PSScriptRoot } else { (Get-Location).Path }
if ([string]::IsNullOrWhiteSpace($SourceDir)) { $SourceDir = Join-Path $ScriptBase "wiki" }
if ([string]::IsNullOrWhiteSpace($TempDir)) { $TempDir = Join-Path $ScriptBase ".wiki_repo" }

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "   FINRL-X GITHUB WIKI SYNCHRONIZATION ENGINE           " -ForegroundColor Cyan
Write-Host "========================================================`n" -ForegroundColor Cyan

if (-not (Test-Path $SourceDir)) {
    Write-Error "Source directory '$SourceDir' not found! Please run from the project root."
    exit 1
}

# 1. Clean or Clone Wiki Repository
if (Test-Path $TempDir) {
    Write-Host "[1/4] Updating existing local wiki clone..." -ForegroundColor Yellow
    Push-Location $TempDir
    try {
        git pull --quiet
    } catch {
        Write-Warning "Failed to pull existing wiki clone. Re-cloning..."
        Pop-Location
        Remove-Item -Recurse -Force $TempDir
        git clone $WikiRepoUrl $TempDir
        Push-Location $TempDir
    }
} else {
    Write-Host "[1/4] Cloning GitHub Wiki repository ($WikiRepoUrl)..." -ForegroundColor Yellow
    git clone $WikiRepoUrl $TempDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "`n[ERROR] Failed to clone wiki repository!"
        Write-Host "Make sure you have enabled Wikis on GitHub and created your first page:" -ForegroundColor Red
        Write-Host "1. Go to https://github.com/ElMoorish/FinRL-X-MT5/wiki" -ForegroundColor Gray
        Write-Host "2. Click 'Create the first page' and click 'Save'" -ForegroundColor Gray
        Write-Host "3. Run this script again!`n" -ForegroundColor Gray
        exit 1
    }
    Push-Location $TempDir
}

# 2. Copy source Markdown files into the wiki clone
Write-Host "[2/4] Copying Wiki pages from '$SourceDir'..." -ForegroundColor Yellow
Copy-Item -Path "$SourceDir\*.md" -Destination $TempDir -Force

# 3. Stage and Commit
Write-Host "[3/4] Staging and committing updates..." -ForegroundColor Yellow
git add .
$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "[OK] Wiki is already up-to-date. No changes detected." -ForegroundColor Green
    Pop-Location
    exit 0
}

git commit -m "docs(wiki): sync 5-agent council, 0.5% risk ceiling, and MT5 deployment guides"

# 4. Push to GitHub
Write-Host "[4/4] Pushing updates to GitHub Wiki..." -ForegroundColor Yellow
git push origin HEAD

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================================" -ForegroundColor Green
    Write-Host "  SUCCESS! GitHub Wiki is live and synchronized!        " -ForegroundColor Green
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host "View your published Wiki at:" -ForegroundColor Cyan
    Write-Host "https://github.com/ElMoorish/FinRL-X-MT5/wiki`n" -ForegroundColor White
} else {
    Write-Error "`nFailed to push wiki changes. Please verify GitHub write permissions."
}

Pop-Location
