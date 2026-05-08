# ============================================================================
#  restart_bot.ps1 -- one-click bot restart.
#
#  Stops any running bot processes (main.py, wscript autostart, start_trading_bot),
#  clears stale PID lock, then fires the CortexTradingBot scheduled task so the
#  bot comes back detached (no console, survives terminal closes).
#
#  Shortcut target launches this with -WindowStyle Normal so the user sees
#  confirmation lines, then the window auto-closes after 4 seconds.
# ============================================================================

$ErrorActionPreference = "Continue"

$ProjectRoot = "g:\AI_Trading_Bot\Cortex"
$PidFile     = Join-Path $ProjectRoot "data\state\bot.pid"
$TaskName    = "CortexTradingBot"

Write-Host ""
Write-Host "=== Cortex Bot Restart ===" -ForegroundColor Cyan
Write-Host ""

# ---- 1. Stop everything bot-related ---------------------------------------
$patterns = 'main\.py|autostart_hidden|start_trading_bot|launch\.ps1'
$running = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
           Where-Object { $_.CommandLine -match $patterns }

if ($running) {
    Write-Host "Stopping $($running.Count) running process(es)..." -ForegroundColor Yellow
    foreach ($p in $running) {
        Write-Host ("  PID {0}: {1}" -f $p.ProcessId, ($p.CommandLine -replace '.*\\', '')) -ForegroundColor DarkGray
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
} else {
    Write-Host "No running bot processes found." -ForegroundColor DarkGray
}

# ---- 2. Clean stale PID file ----------------------------------------------
if (Test-Path $PidFile) {
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Write-Host "Removed stale PID lock." -ForegroundColor DarkGray
}

# ---- 3. Fire the scheduled task (detached, hidden) ------------------------
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host ""
    Write-Host "ERROR: Scheduled task '$TaskName' not installed." -ForegroundColor Red
    Write-Host "Run scripts\install_autostart.ps1 first." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to close"
    exit 1
}

Start-ScheduledTask -TaskName $TaskName
Write-Host ""
Write-Host "Fired scheduled task '$TaskName'." -ForegroundColor Green

# ---- 4. Verify a python.exe main.py came up -------------------------------
Start-Sleep -Seconds 5
$proc = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'main\.py' }

if ($proc) {
    Write-Host ""
    Write-Host "Bot is running (PID $($proc.ProcessId))." -ForegroundColor Green
    Write-Host "Dashboard: http://localhost:8787" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "WARNING: No main.py process detected after 5s." -ForegroundColor Yellow
    Write-Host "Check data\logs\autostart.log for errors." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "(This window closes in 4s)" -ForegroundColor DarkGray
Start-Sleep -Seconds 4
