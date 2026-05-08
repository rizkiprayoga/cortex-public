# ============================================================================
#  launch.ps1 --one-click Cortex launcher with auto-detection
#
#  What it does:
#    * Detects if the bot is already running -> just opens dashboard.
#    * Detects if the PC just rebooted AND the bot was running before the
#      reboot -> runs POST-RESTART verify (compares broker state vs the
#      heartbeat taken before shutdown).
#    * Otherwise -> runs full pre-flight validation before starting.
#    * Starts main.py in this window (live logs, Ctrl-C stops cleanly).
#    * Auto-opens http://localhost:8787 once the API is ready.
#
#  Flags:
#    -WithFrontend   Also start the Vite dev server in a new window.
#    -SkipChecks     Bypass preflight/verify (not recommended).
#    -Autostart      Headless mode for Task Scheduler (no prompts, no
#                    browser auto-open, quieter output).
# ============================================================================

param(
    [switch]$WithFrontend,
    [switch]$SkipChecks,
    [switch]$Autostart
)

$ErrorActionPreference = "Stop"
if (-not $Autostart) {
    $Host.UI.RawUI.WindowTitle = "Cortex Trading Bot"
}

$ProjectRoot = "g:\AI_Trading_Bot\Cortex"
$VenvPython  = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$Heartbeat   = Join-Path $ProjectRoot "data\logs\bot_heartbeat.json"
$TradingLog  = Join-Path $ProjectRoot "data\logs\trading_bot.log"

function Write-Step($msg, $color = "Cyan") {
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor $color
    Write-Host "  $msg" -ForegroundColor $color
    Write-Host "=================================================================" -ForegroundColor $color
}

function Prompt-ToExit($code) {
    if (-not $Autostart) {
        Read-Host "Press Enter to close"
    }
    exit $code
}

Set-Location $ProjectRoot

if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: venv Python not found at $VenvPython" -ForegroundColor Red
    Write-Host "  Create it first:  python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt" -ForegroundColor Yellow
    Prompt-ToExit 1
}

# ----------------------------------------------------------------------------
# Detection: is the bot already running?
# ----------------------------------------------------------------------------
$BotAlreadyRunning = $false
try {
    $procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue
    foreach ($p in $procs) {
        if ($p.CommandLine -and $p.CommandLine -match "main\.py") {
            $BotAlreadyRunning = $true
            break
        }
    }
} catch {}

if ($BotAlreadyRunning) {
    Write-Step "Bot is already running --opening dashboard" "Green"
    Start-Process "http://localhost:8787"
    if (-not $Autostart) {
        Write-Host ""
        Write-Host "  Tail the log with:  Get-Content data\logs\trading_bot.log -Wait -Tail 20" -ForegroundColor Cyan
        Write-Host ""
        Read-Host "Press Enter to close this window (the bot keeps running)"
    }
    exit 0
}

# ----------------------------------------------------------------------------
# Detection: fresh start vs post-restart
# ----------------------------------------------------------------------------
$bootTime  = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
$uptimeMin = ((Get-Date) - $bootTime).TotalMinutes

$Mode = "fresh"  # default: full preflight
if ((Test-Path $Heartbeat) -and $uptimeMin -lt 30) {
    try {
        $hb = Get-Content $Heartbeat -Raw | ConvertFrom-Json
        # Heartbeat timestamp prior to boot -> bot was running in previous session
        $hbTime = [datetime]::Parse($hb.timestamp_utc).ToUniversalTime()
        if ($hbTime -lt $bootTime.ToUniversalTime()) {
            $Mode = "post-restart"
        }
    } catch {}
}

Write-Host ""
Write-Host "  Mode: $Mode   (PC uptime $([int]$uptimeMin) min)" -ForegroundColor DarkCyan

# ----------------------------------------------------------------------------
# Pre-flight or post-restart verification
# ----------------------------------------------------------------------------
if (-not $SkipChecks) {
    if ($Mode -eq "post-restart") {
        Write-Step "POST-RESTART VERIFY --comparing broker state vs heartbeat"
        & $VenvPython "scripts\paper_trading_ops.py" verify
    } else {
        Write-Step "PRE-FLIGHT --validating readiness"
        & $VenvPython "scripts\paper_trading_ops.py" preflight --skip-smoke
    }
    if ($LASTEXITCODE -ne 0) {
        $cmdName = if ($Mode -eq "post-restart") { "verify" } else { "preflight" }
        Write-Host ""
        Write-Host "Checks FAILED -- fix the issues above before launching." -ForegroundColor Red
        Write-Host "Tip: rerun with  python scripts\paper_trading_ops.py $cmdName" -ForegroundColor Yellow
        Prompt-ToExit 1
    }
}

# ----------------------------------------------------------------------------
# Optional: frontend dev server
# ----------------------------------------------------------------------------
if ($WithFrontend) {
    Write-Step "Launching frontend dev server (new window)"
    $frontendCmd = "cd /d `"$ProjectRoot\frontend`" && npm run dev"
    Start-Process "cmd.exe" -ArgumentList "/k", $frontendCmd
}

# ----------------------------------------------------------------------------
# Schedule auto-open of the dashboard (unless headless autostart)
# ----------------------------------------------------------------------------
if (-not $Autostart) {
    Start-Job -ScriptBlock {
        $deadline = (Get-Date).AddSeconds(90)
        while ((Get-Date) -lt $deadline) {
            try {
                $sock = New-Object System.Net.Sockets.TcpClient
                $sock.Connect("127.0.0.1", 8787)
                $sock.Close()
                Start-Process "http://localhost:8787"
                return
            } catch {
                Start-Sleep -Milliseconds 1000
            }
        }
    } | Out-Null
}

# ----------------------------------------------------------------------------
# Run main.py in this window
# ----------------------------------------------------------------------------
Write-Step "Starting Cortex bot  (Ctrl-C to stop)"
if (-not $Autostart) {
    Write-Host "  Dashboard will open in your browser as soon as the API is ready." -ForegroundColor Cyan
    Write-Host "  Live logs stream below. Full log: data\logs\trading_bot.log" -ForegroundColor Cyan
}
Write-Host ""

& $VenvPython "main.py"

Write-Host ""
Write-Host "Bot exited (code=$LASTEXITCODE)." -ForegroundColor Yellow
Prompt-ToExit $LASTEXITCODE
