@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
py -3 -c "import sys" >nul 2>&1
if errorlevel 1 goto use_python
py -3 "%~dp0bootstrap.py" %*
set "client_exit=%errorlevel%"
goto finish
:use_python
python -c "import sys" >nul 2>&1
if errorlevel 1 goto missing_python
python "%~dp0bootstrap.py" %*
set "client_exit=%errorlevel%"
goto finish
:missing_python
echo Python was not found. Install 64-bit Python 3.10-3.15 with Tcl/Tk, then retry.
set "client_exit=1"
:finish
if not "%client_exit%"=="0" pause
exit /b %client_exit%
