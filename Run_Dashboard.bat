@echo off
cd /d "%~dp0"
chcp 65001 >nul 2>&1

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH!
    pause
    exit /b 1
)

where streamlit >nul 2>&1
if errorlevel 1 (
    echo [INFO] Checking streamlit via python...
    python -c "import streamlit" 2>nul
    if errorlevel 1 (
        echo [ERROR] Streamlit not found. Install with: pip install streamlit
        pause
        exit /b 1
    )
)

echo ===================================================
echo   GSA Dashboard - Streamlit
echo ===================================================
echo.
echo   Starting dashboard...
echo   Browser will open automatically.
echo   Press Ctrl+C to stop.
echo ===================================================

python -Xutf8 -m streamlit run "deshbord giboi.py"

if not "%~1"=="--no-pause" pause
