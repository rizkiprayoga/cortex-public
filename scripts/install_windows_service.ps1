# ============================================================================
#  install_windows_service.ps1 -- Register Cortex as a Windows service via NSSM
#
#  Why: running main.py as a foreground PowerShell process means any closed
#  terminal kills the bot. A service runs in the background, survives logout,
#  auto-restarts on crash, and exposes standard Start/Stop/Restart-Service
#  commands. NSSM is the de-facto "service wrapper" for Windows and is the
#  cleanest way to daemonize a Python script without rewriting anything.
#
#  Run once (auto-elevates with UAC if you are not admin):
#      powershell -ExecutionPolicy Bypass -File scripts\install_windows_service.ps1
#
#  After install:
#      Start-Service  CortexBot       # start the bot
#      Stop-Service   CortexBot       # stop it
#      Restart-Service CortexBot      # apply config changes
#      Get-Service    CortexBot       # current state
#      services.msc                   # GUI view
#
#  Logs (service stdout/stderr):
#      data\logs\service-stdout.log
#      data\logs\service-stderr.log
#  Bot's own rotating log still at:
#      data\logs\trading_bot.log
#
#  The service runs under the CURRENT USER account (not SYSTEM) because MT5
#  requires an interactive desktop session. You need to be logged in at
#  least once for MT5 to attach. The bot retries MT5 connection on failure
#  so even if it starts before your first login of the day, it recovers
#  automatically when MT5 becomes available.
#
#  This script ALSO disables the CortexTradingBot Task Scheduler task if
#  it exists, to prevent dual-start (service + task both launching bots).
# ============================================================================

param([switch]$Elevated)

$ServiceName  = "CortexBot"
$ProjectRoot  = "g:\AI_Trading_Bot\Cortex"
$VenvPython   = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$MainPy       = Join-Path $ProjectRoot "main.py"
$NssmDir      = Join-Path $ProjectRoot "scripts\nssm"
$NssmExe      = Join-Path $NssmDir "nssm.exe"
$LogDir       = Join-Path $ProjectRoot "data\logs"

# Sanity --------------------------------------------------------------------
if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: venv python not found at $VenvPython" -ForegroundColor Red
    Write-Host "  Create it:  python -m venv venv; venv\Scripts\activate; pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"; exit 1
}
if (-not (Test-Path $MainPy)) {
    Write-Host "ERROR: main.py not found at $MainPy" -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}

# Admin check ---------------------------------------------------------------
function Is-Admin {
    $id = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $p  = New-Object System.Security.Principal.WindowsPrincipal($id)
    return $p.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Is-Admin) -and -not $Elevated) {
    Write-Host "Admin privileges required. Relaunching with UAC prompt..." -ForegroundColor Yellow
    try {
        Start-Process -FilePath "powershell.exe" `
            -ArgumentList "-NoLogo","-ExecutionPolicy","Bypass",
                          "-File","`"$($MyInvocation.MyCommand.Path)`"","-Elevated" `
            -Verb RunAs -ErrorAction Stop
        Write-Host "Elevated instance launched. This window can close." -ForegroundColor Green
        Read-Host "Press Enter"; exit 0
    } catch {
        Write-Host "User declined UAC elevation. Service NOT installed." -ForegroundColor Red
        Read-Host "Press Enter"; exit 1
    }
}

# Fetch NSSM if missing -----------------------------------------------------
if (-not (Test-Path $NssmExe)) {
    Write-Host "NSSM binary not found at $NssmExe"
    Write-Host "Downloading NSSM 2.24 from https://nssm.cc..." -ForegroundColor Cyan
    $tmpZip = Join-Path $env:TEMP "nssm-2.24.zip"
    $tmpExt = Join-Path $env:TEMP "nssm-extract"
    try {
        Invoke-WebRequest -Uri "https://nssm.cc/release/nssm-2.24.zip" `
            -OutFile $tmpZip -UseBasicParsing -ErrorAction Stop
        if (Test-Path $tmpExt) { Remove-Item $tmpExt -Recurse -Force }
        Expand-Archive -Path $tmpZip -DestinationPath $tmpExt -Force
        New-Item -ItemType Directory -Path $NssmDir -Force | Out-Null
        # Pick 64-bit binary from the extracted tree
        $src = Get-ChildItem -Path $tmpExt -Recurse -Filter "nssm.exe" |
               Where-Object { $_.FullName -match "win64" } |
               Select-Object -First 1
        if (-not $src) { throw "nssm.exe (win64) not found inside the downloaded archive" }
        Copy-Item $src.FullName $NssmExe -Force
        Remove-Item $tmpZip, $tmpExt -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "NSSM installed to $NssmExe" -ForegroundColor Green
    } catch {
        Write-Host "NSSM download failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Manual steps:" -ForegroundColor Yellow
        Write-Host "  1. Download https://nssm.cc/release/nssm-2.24.zip"
        Write-Host "  2. Extract win64\nssm.exe to $NssmExe"
        Write-Host "  3. Re-run this script"
        Read-Host "Press Enter"; exit 1
    }
}

# Disable old Task Scheduler autostart so we don't dual-start ---------------
$oldTask = Get-ScheduledTask -TaskName "CortexTradingBot" -ErrorAction SilentlyContinue
if ($oldTask) {
    Write-Host "Disabling old Task Scheduler autostart 'CortexTradingBot' to avoid dual-start..." -ForegroundColor Yellow
    try {
        Disable-ScheduledTask -TaskName "CortexTradingBot" -ErrorAction Stop | Out-Null
        Write-Host "  Task disabled (kept in place -- run uninstall_autostart.ps1 to delete it entirely)" -ForegroundColor DarkGray
    } catch {
        Write-Host "  Could not disable task: $($_.Exception.Message)" -ForegroundColor DarkYellow
    }
}

# Stop + remove existing service if present (idempotent install) ------------
$existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Existing service '$ServiceName' found -- stopping + removing for fresh install..."
    & $NssmExe stop    $ServiceName confirm 2>&1 | Out-Null
    & $NssmExe remove  $ServiceName confirm 2>&1 | Out-Null
    Start-Sleep -Seconds 1
}

# Install ------------------------------------------------------------------
Write-Host ""
Write-Host "Installing Windows service '$ServiceName'..." -ForegroundColor Cyan
& $NssmExe install $ServiceName $VenvPython $MainPy | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "nssm install returned $LASTEXITCODE" -ForegroundColor Red
    Read-Host "Press Enter"; exit 1
}

# Working directory (so relative paths in main.py resolve correctly)
& $NssmExe set $ServiceName AppDirectory $ProjectRoot | Out-Null

# Stdout / stderr log files (NSSM handles rotation via AppRotate* later)
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
& $NssmExe set $ServiceName AppStdout (Join-Path $LogDir "service-stdout.log") | Out-Null
& $NssmExe set $ServiceName AppStderr (Join-Path $LogDir "service-stderr.log") | Out-Null

# Rotate logs at 10 MB to prevent unbounded growth
& $NssmExe set $ServiceName AppRotateFiles 1    | Out-Null
& $NssmExe set $ServiceName AppRotateOnline 1   | Out-Null
& $NssmExe set $ServiceName AppRotateBytes 10485760 | Out-Null

# Run as the current user (MT5 needs interactive desktop)
$currentUser = "$env:USERDOMAIN\$env:USERNAME"
Write-Host "  Account: $currentUser" -ForegroundColor DarkGray
$sec = Read-Host "Enter your Windows password (visible only once, for service account setup)" -AsSecureString
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
$pwd  = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
[System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($BSTR)
& $NssmExe set $ServiceName ObjectName $currentUser $pwd | Out-Null
$pwd = $null  # clear from memory

# Startup Automatic (starts on boot)
& $NssmExe set $ServiceName Start SERVICE_AUTO_START | Out-Null

# Restart policy: quick 3x, then back off to every 60s
& $NssmExe set $ServiceName AppExit Default Restart | Out-Null
& $NssmExe set $ServiceName AppRestartDelay 2000   | Out-Null
& $NssmExe set $ServiceName AppThrottle 10000      | Out-Null

# Description
& $NssmExe set $ServiceName Description "Cortex Trading Bot -- automated MT5 trading system with HMM+LSTM signal fusion" | Out-Null

Write-Host ""
Write-Host "[OK] Service '$ServiceName' installed." -ForegroundColor Green
Write-Host ""
Write-Host "Starting the service now..." -ForegroundColor Cyan
Start-Service -Name $ServiceName
Start-Sleep -Seconds 3
$state = (Get-Service -Name $ServiceName).Status
Write-Host "Service state: $state" -ForegroundColor (if ($state -eq 'Running') {'Green'} else {'Yellow'})

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  - Verify:    Get-Service CortexBot"
Write-Host "  - Tail log:  Get-Content $LogDir\trading_bot.log -Tail 20 -Wait"
Write-Host "  - Stop:      Stop-Service CortexBot"
Write-Host "  - Uninstall: scripts\uninstall_windows_service.ps1"
Write-Host ""
Read-Host "Press Enter to close"
