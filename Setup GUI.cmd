@echo off
cd /d "%~dp0"
python -m venv --system-site-packages .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --ignore-installed --no-deps onnxruntime-directml==1.24.4
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --ignore-installed --no-deps opencv-contrib-python==4.13.0.92
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install PySide6 numpy opencv-contrib-python pywin32 mss dxcam psutil comtypes
if errorlevel 1 goto failed
echo Setup complete. Double-click Start GUI.cmd.
pause
exit /b 0
:failed
echo Setup failed. Copy the error above.
pause
exit /b 1
