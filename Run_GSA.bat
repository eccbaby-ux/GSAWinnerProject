@echo off
cd /d "%~dp0"
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH!
    echo Please install Python or add it to PATH.
    if /i "%~1"=="--no-pause" exit /b 1
    pause
    exit /b 1
)

echo ===================================================
echo  GSA Master Pro - PURE MATH PIPELINE (NO OVERFITTING)
echo ===================================================
echo.

echo [Step 1/3] Fetching Winner odds (winner_auto_fetcher.py)...
python -Xutf8 winner_auto_fetcher.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Step 1 failed - aborting to avoid stale data.
    if /i "%~1"=="--no-pause" exit /b 1
    pause
    exit /b
)

echo.
echo [Step 2/3] Running main analysis engine (v76_Master_Nachshon.py)...
python -Xutf8 v76_Master_Nachshon.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] V76 engine crashed - aborting.
    if /i "%~1"=="--no-pause" exit /b 1
    pause
    exit /b
)

echo.
echo [Step 3/3] Running Reality Check and Risk Management (dadima_correction.py)...
python -Xutf8 dadima_correction.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Dadima correction failed.
    if /i "%~1"=="--no-pause" exit /b 1
    pause
    exit /b
)

echo.
echo ===================================================
echo [SUCCESS] Run Complete! Clean data generated.
echo ===================================================
echo To view results, open a terminal and run:
echo streamlit run "deshbord giboi.py"
echo ===================================================
if /i "%~1"=="--no-pause" exit /b 0
pause