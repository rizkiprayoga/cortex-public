@echo off
REM ===========================================================================
REM  start_trading_bot.bat — Windows logon launcher (headless)
REM
REM  Invoked by Task Scheduler at user logon (see install_autostart.ps1).
REM  Delegates everything to launch.ps1 so the post-restart verify flow
REM  runs automatically when the PC comes back up.
REM
REM  Steps (handled inside launch.ps1):
REM    1. Launch MT5 terminal if not already running
REM    2. Run post-restart verify or preflight (auto-detected from heartbeat)
REM    3. Start main.py — stdout/stderr appended to data\logs\autostart.log
REM ===========================================================================

setlocal enabledelayedexpansion

REM Force UTF-8 for all Python I/O so stdlib logging can emit non-ASCII
REM characters (e.g. arrow in feature_engineering messages) without blowing
REM up when stdout is redirected to a file via >>. Without this, Python 3.13
REM on Windows uses cp1252 for pipes and crashes on the first non-ASCII log.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PROJECT_ROOT=g:\AI_Trading_Bot\Cortex"
set "LOG_DIR=%PROJECT_ROOT%\data\logs"
set "LOG_FILE=%LOG_DIR%\autostart.log"

REM Default MT5 terminal path — override by setting MT5_TERMINAL_PATH in .env
set "MT5_TERMINAL=C:\Program Files\MetaTrader 5\terminal64.exe"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PROJECT_ROOT%"

echo. >> "%LOG_FILE%"
echo [%date% %time%] ============================================ >> "%LOG_FILE%"
echo [%date% %time%] === Cortex autostart === >> "%LOG_FILE%"

REM --- Load MT5_TERMINAL_PATH from .env if present -----------------------------
if exist "%PROJECT_ROOT%\.env" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%PROJECT_ROOT%\.env") do (
        if /i "%%a"=="MT5_TERMINAL_PATH" set "MT5_TERMINAL=%%b"
    )
)

REM --- Launch MT5 if not already running ---------------------------------------
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I "terminal64.exe" >NUL
if errorlevel 1 (
    if exist "%MT5_TERMINAL%" (
        echo [%date% %time%] Launching MT5 from "%MT5_TERMINAL%" >> "%LOG_FILE%"
        start "" "%MT5_TERMINAL%"
        REM Give MT5 time to connect to the broker and load symbols
        timeout /t 25 /nobreak >NUL
    ) else (
        echo [%date% %time%] ERROR: MT5 terminal not found at "%MT5_TERMINAL%" >> "%LOG_FILE%"
        exit /b 1
    )
) else (
    echo [%date% %time%] MT5 already running >> "%LOG_FILE%"
)

REM --- Delegate to launch.ps1 in Autostart mode --------------------------------
REM -Autostart suppresses browser auto-open and interactive prompts.
echo [%date% %time%] Invoking launch.ps1 -Autostart >> "%LOG_FILE%"
powershell.exe -NoLogo -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\launch.ps1" -Autostart >> "%LOG_FILE%" 2>&1

set BOT_EXIT=%errorlevel%
echo [%date% %time%] launch.ps1 exited with code %BOT_EXIT% >> "%LOG_FILE%"
exit /b %BOT_EXIT%
