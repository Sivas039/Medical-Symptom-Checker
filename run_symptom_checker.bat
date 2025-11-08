@echo off
REM Symptom Checker - Launcher Script
REM This script automatically uses the correct Python environment

echo.
echo ========================================
echo  Symptom Checker - Starting
echo ========================================
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "symp_env\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please ensure symp_env folder exists
    pause
    exit /b 1
)

echo [INFO] Using Python from virtual environment...
echo [INFO] Location: %cd%\symp_env\Scripts\python.exe
echo.

REM Start the app with the correct Python
echo [OK] Starting Symptom Checker on port 7860...
echo [OK] Open browser to: http://localhost:7860
echo [OK] Press Ctrl+C to stop the server
echo.

"%cd%\symp_env\Scripts\python.exe" gradio_app.py

REM If the app exits with error
if errorlevel 1 (
    echo.
    echo ERROR: App failed to start
    echo Please check the error messages above
    pause
)

pause
