@echo off
setlocal
cd /d "%~dp0.."
if not exist "ml\.venv\Scripts\python.exe" (
    py -3.12 -m venv ml\.venv
    if errorlevel 1 goto failed
)
"ml\.venv\Scripts\python.exe" -m pip install -r ml\requirements.lock
if errorlevel 1 goto failed
"ml\.venv\Scripts\python.exe" -m streamlit run ml\label.py --server.address 127.0.0.1 --browser.gatherUsageStats false
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Setup or startup failed. Install Python 3.12 with the Python launcher and check the error above.
pause
exit /b 1
