@echo off
cd /d "%~dp0"
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH!
    pause
    exit /b 1
)

echo ===================================================
echo  GSA Daily Full Pipeline - Train + Run + Push
echo ===================================================
echo.

:: ============ TRAIN PHASE ============
echo === TRAIN: Step 1/5 - Updating match results ===
python -Xutf8 result_updater.py
if %errorlevel% neq 0 (
    echo [ERROR] result_updater.py failed - aborting.
    pause
    exit /b 1
)

echo.
echo === TRAIN: Step 2/5 - Updating ELO ratings ===
python -Xutf8 elo_updater.py
if %errorlevel% neq 0 (
    echo [WARNING] ELO updater failed - continuing.
)

echo.
echo === TRAIN: Step 3/5 - Auto Learner ===
python -Xutf8 v79_Auto_Learner.py
if %errorlevel% neq 0 (
    echo [ERROR] v79_Auto_Learner.py failed - aborting.
    pause
    exit /b 1
)

echo.
echo === TRAIN: Step 4/5 - Shishka ===
python -Xutf8 shishka_run_and_save.py
if %errorlevel% neq 0 (
    echo [ERROR] shishka_run_and_save.py failed - aborting.
    pause
    exit /b 1
)

echo.
echo === TRAIN: Step 5/5 - Ticha ===
python -Xutf8 ticha_system.py
if %errorlevel% neq 0 (
    echo [ERROR] ticha_system.py failed - aborting.
    pause
    exit /b 1
)

:: ============ RUN PHASE ============
echo.
echo === RUN: Step 1/3 - Fetching Winner odds ===
python -Xutf8 winner_auto_fetcher.py
if %errorlevel% neq 0 (
    echo [ERROR] winner_auto_fetcher.py failed - aborting.
    pause
    exit /b 1
)

echo.
echo === RUN: Step 2/3 - Main analysis engine ===
python -Xutf8 v76_Master_Nachshon.py
if %errorlevel% neq 0 (
    echo [ERROR] v76_Master_Nachshon.py failed - aborting.
    pause
    exit /b 1
)

echo.
echo === RUN: Step 3/3 - Risk management ===
python -Xutf8 dadima_correction.py
if %errorlevel% neq 0 (
    echo [ERROR] dadima_correction.py failed - aborting.
    pause
    exit /b 1
)

:: ============ GIT PUSH ============
echo.
echo === GIT: Pushing results to GitHub ===
git add gsa_history.db analysis_results_v76.json winner_odds_cache.json ^
        winner_odds_previous.json ticha_params.json calibration_params.json ^
        translation_cache.json matches.txt
git diff --cached --quiet
if !errorlevel! neq 0 (
    git commit -m "auto: daily pipeline %date% %time:~0,5%"
    git push
    if !errorlevel! neq 0 (
        echo [ERROR] git push failed - check connection/credentials.
        pause
        exit /b 1
    )
    echo [SUCCESS] Pushed to GitHub - dashboard will refresh automatically.
) else (
    echo [INFO] No changes to commit.
)

echo.
echo ===================================================
echo [SUCCESS] Full pipeline complete! Dashboard updated.
echo ===================================================
pause
