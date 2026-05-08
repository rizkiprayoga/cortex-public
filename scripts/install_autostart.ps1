# ============================================================================
#  install_autostart.ps1 -- Register CortexTradingBot with Windows Task Scheduler
#
#  Run once. No admin required (uses -RunLevel Limited, user scope).
#
#  What it does:
#    - Creates/updates a scheduled task named "CortexTradingBot"
#    - Trigger: at logon of the current user
#    - Action:  cmd.exe /c "<project>\scripts\start_trading_bot.bat"
#    - Settings: unlimited runtime, restart up to 3x on crash, no battery limits
#    - Principal: current user, RunLevel Limited (no UAC prompt at launch)
#
#  If you hit 'Access is denied', the script will auto-relaunch itself with
#  UAC elevation. That's only needed if your PC has restrictive group policies.
#
#  After installation:
#      Start-ScheduledTask -TaskName "CortexTradingBot"   # test-fire now
#      Get-Content data\logs\autostart.log -Tail 20 -Wait # follow the log
# ============================================================================

param([switch]$Elevated)

$TaskName = "CortexTradingBot"
$VbsPath  = "g:\AI_Trading_Bot\Cortex\scripts\autostart_hidden.vbs"
$BatPath  = "g:\AI_Trading_Bot\Cortex\scripts\start_trading_bot.bat"

if (-not (Test-Path $VbsPath)) {
    Write-Host "ERROR: Hidden launcher not found at $VbsPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
if (-not (Test-Path $BatPath)) {
    Write-Host "ERROR: Batch launcher not found at $BatPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Guard against dual-start: if the NSSM 'CortexBot' service is installed
# AND running, installing the Task Scheduler autostart on top would launch
# TWO bot instances at every logon — the PID lock would immediately kill
# the second one but it would still be confusing. Ask the user.
$svc = Get-Service -Name "CortexBot" -ErrorAction SilentlyContinue
if ($svc) {
    Write-Host ""
    Write-Host "WARNING: NSSM service 'CortexBot' is currently installed (state: $($svc.Status))." -ForegroundColor Yellow
    Write-Host "         The service already auto-starts the bot at boot." -ForegroundColor Yellow
    Write-Host "         Installing this Task Scheduler autostart too would mean TWO bot instances" -ForegroundColor Yellow
    Write-Host "         trying to launch at every logon. The PID lock would kill one, but you" -ForegroundColor Yellow
    Write-Host "         should not install both." -ForegroundColor Yellow
    Write-Host ""
    $ans = Read-Host "Continue installing Task Scheduler autostart anyway? [y/N]"
    if ($ans -notmatch '^[yY]') {
        Write-Host "Aborted. Use scripts\uninstall_windows_service.ps1 to remove the service first" -ForegroundColor Cyan
        Write-Host "if you want to use the Task Scheduler path instead." -ForegroundColor Cyan
        Read-Host "Press Enter to exit"
        exit 0
    }
}

function Is-Admin {
    $id = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object System.Security.Principal.WindowsPrincipal($id)
    return $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
}

# wscript.exe runs the .vbs with no console window at all — no cmd flash, no
# PowerShell window either (start_trading_bot.bat now passes -WindowStyle Hidden
# to launch.ps1). The bot still writes to data\logs\autostart.log and
# trading_bot.log exactly as before.
$Action = New-ScheduledTaskAction `
    -Execute "wscript.exe" `
    -Argument "`"$VbsPath`""

$Trigger = New-ScheduledTaskTrigger `
    -AtLogOn `
    -User $env:USERNAME

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 2) `
    -MultipleInstances IgnoreNew

# -RunLevel Limited means the task does NOT require admin to execute. Admin
# is only needed to REGISTER the task if the host has tight group policies.
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "Auto-starts the Cortex trading bot at Windows logon" `
        -Force -ErrorAction Stop | Out-Null

    Write-Host ""
    Write-Host "[OK] Task '$TaskName' installed." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  - Test-fire it now:           Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  - Follow the log:             Get-Content data\logs\autostart.log -Tail 30 -Wait"
    Write-Host "  - View in Task Scheduler UI:  taskschd.msc"
    Write-Host "  - Remove it later:            scripts\uninstall_autostart.ps1"
    Write-Host ""
    Read-Host "Press Enter to close"
}
catch {
    $msg = $_.Exception.Message
    Write-Host ""
    Write-Host "Register-ScheduledTask failed: $msg" -ForegroundColor Yellow

    # Auto-relaunch with admin if we haven't already, and the error looks like a privilege issue.
    if ((-not $Elevated) -and ($msg -match "Access is denied|PermissionDenied|0x80070005")) {
        Write-Host "Relaunching with admin privileges (UAC prompt incoming)..." -ForegroundColor Cyan
        $scriptPath = $MyInvocation.MyCommand.Path
        try {
            Start-Process -FilePath "powershell.exe" `
                -ArgumentList "-NoLogo","-ExecutionPolicy","Bypass",
                              "-File","`"$scriptPath`"","-Elevated" `
                -Verb RunAs -ErrorAction Stop
            Write-Host "Elevated instance started. This window can be closed." -ForegroundColor Green
            Read-Host "Press Enter to close"
            exit 0
        } catch {
            Write-Host "User declined UAC elevation. Task NOT installed." -ForegroundColor Red
        }
    }

    Write-Host ""
    Write-Host "Manual workaround:" -ForegroundColor Yellow
    Write-Host "  Right-click PowerShell -> Run as Administrator, then:" -ForegroundColor Yellow
    Write-Host "    powershell -ExecutionPolicy Bypass -File $($MyInvocation.MyCommand.Path)" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to close"
    exit 1
}
