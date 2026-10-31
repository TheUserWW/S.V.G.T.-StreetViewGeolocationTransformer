@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Project Python environment was not found. See README.md for setup.
    pause
    exit /b 1
)
if not exist ".gui" mkdir ".gui"
start "" ".venv\Scripts\pythonw.exe" -m streetgeo map 1>>".gui\prediction_startup.log" 2>&1
