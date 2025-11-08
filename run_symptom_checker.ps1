# Symptom Checker - PowerShell Launcher
# This script automatically uses the correct Python environment

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Symptom Checker - Starting" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if virtual environment exists
$pythonPath = Join-Path $scriptDir "symp_env\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Expected path: $pythonPath" -ForegroundColor Yellow
    Write-Host "Please ensure symp_env folder exists" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[INFO] Using Python from virtual environment..." -ForegroundColor Cyan
Write-Host "[INFO] Location: $pythonPath" -ForegroundColor Cyan
Write-Host ""

# Check if gradio_app.py exists
if (-not (Test-Path "gradio_app.py")) {
    Write-Host "ERROR: gradio_app.py not found!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[OK] Starting Symptom Checker on port 7860..." -ForegroundColor Green
Write-Host "[OK] Open browser to: http://localhost:7860" -ForegroundColor Green
Write-Host "[OK] Press Ctrl+C to stop the server" -ForegroundColor Green
Write-Host ""

# Start the app with the correct Python
& $pythonPath gradio_app.py

# Check for errors
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: App failed to start (Exit code: $LASTEXITCODE)" -ForegroundColor Red
    Write-Host "Please check the error messages above" -ForegroundColor Yellow
}

Read-Host "Press Enter to exit"
