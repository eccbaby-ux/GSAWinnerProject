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
echo  GSA Training Pipeline - Learning Process
echo ===================================================
echo.

echo [Step 1/5] Updating match results from API (result_updater.py)...
python -Xutf8 result_updater.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Step 1 failed - result updater crashed.
    if /i "%~1"=="--no-pause" exit /b 1
    pause
    exit /b 1
)

echo.
echo [Step 2/5] Updating ELO ratings (elo_updater.py)...
python -Xutf8 elo_updater.py
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] ELO updater failed - continuing without ELO ratings.
)

echo.
echo [Step 3/5] Running Auto-Learner (v79_Auto_Learner.py)...
python -Xutf8 v79_Auto_Learner.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Step 3 failed - Auto-Learner crashed.
    if /i "%~1"=="--no-pause" exit /b 1
    pause
    exit /b 1
)

echo.
echo [Step 4/5] Running Shishka (shishka_run_and_save.py)...
python -Xutf8 shishka_run_and_save.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Step 4 failed - Shishka crashed.
    if /i "%~1"=="--no-pause" exit /b 1
    pause
    exit /b 1
)

echo.
echo [Step 5/5] Running Ticha System (ticha_system.py)...
python -Xutf8 ticha_system.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Step 5 failed - Ticha system crashed.
    if /i "%~1"=="--no-pause" exit /b 1
    pause
    exit /b 1
)

echo.
echo ===================================================
echo [SUCCESS] Training pipeline complete!
echo ===================================================
if /i "%~1"=="--no-pause" exit /b 0
pause
