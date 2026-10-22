@echo off
cd /d "%~dp0"
if not exist "%~dp0venv\Scripts\pythonw.exe" (
    echo Please create venv and install requirements-motor-bo.txt first.
    pause
    exit /b 1
)
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0motor_bo_gui.py"
