@echo off
cd /d "%~dp0"
set "LOCALAPPDATA=%~dp0userdata"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup GUI.cmd once before starting.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" src\ui\app.py
if errorlevel 1 pause
