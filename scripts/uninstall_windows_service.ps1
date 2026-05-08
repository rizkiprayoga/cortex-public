# ============================================================================
#  uninstall_windows_service.ps1 -- Stop + unregister the CortexBot service
#
#  Safe to run whether the service exists or not (idempotent).
#  Auto-elevates via UAC if you are not running as admin.
#
#  Does NOT delete the bundled NSSM binary (scripts/nssm/nssm.exe) or
#  the service log files (data/logs/service-*.log) -- remove those
#  manually if you want a clean slate.
# ============================================================================

param([switch]$Elevated)

$ServiceName = "CortexBot"
$ProjectRoot = "g:\AI_Trading_Bot\Cortex"
$NssmExe     = Join-Path $ProjectRoot "scripts\nssm\nssm.exe"

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
        Read-Host "Press Enter"; exit 0
    } catch {
        Write-Host "User declined UAC. Service not removed." -ForegroundColor Red
        Read-Host "Press Enter"; exit 1
    }
}

$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if (-not $svc) {
    Write-Host "Service '$ServiceName' is not installed -- nothing to do." -ForegroundColor DarkGray
    Read-Host "Press Enter"; exit 0
}

Write-Host "Stopping service '$ServiceName'..." -ForegroundColor Cyan
try {
    Stop-Service -Name $ServiceName -Force -ErrorAction Stop
} catch {
    Write-Host "  Stop-Service raised: $($_.Exception.Message) -- continuing anyway." -ForegroundColor Yellow
}

if (-not (Test-Path $NssmExe)) {
    Write-Host "NSSM binary missing at $NssmExe." -ForegroundColor Red
    Write-Host "Falling back to 'sc.exe delete' which leaves NSSM wrapper state behind."
    sc.exe delete $ServiceName | Out-Null
} else {
    Write-Host "Unregistering service via NSSM..." -ForegroundColor Cyan
    & $NssmExe remove $ServiceName confirm | Out-Null
}

Start-Sleep -Seconds 1
$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($svc) {
    Write-Host "[WARN] service '$ServiceName' still exists -- reboot may be required." -ForegroundColor Yellow
} else {
    Write-Host "[OK] service '$ServiceName' removed." -ForegroundColor Green
}

# Offer to re-enable the Task Scheduler autostart since the user no
# longer has a service running the bot automatically.
$task = Get-ScheduledTask -TaskName "CortexTradingBot" -ErrorAction SilentlyContinue
if ($task -and $task.State -eq "Disabled") {
    Write-Host ""
    Write-Host "Found disabled Task Scheduler autostart 'CortexTradingBot'." -ForegroundColor Cyan
    $ans = Read-Host "Re-enable it so the bot still auto-starts on logon? [y/N]"
    if ($ans -match '^[yY]') {
        Enable-ScheduledTask -TaskName "CortexTradingBot" | Out-Null
        Write-Host "Task re-enabled. The bot will now auto-start in a PowerShell window on your next logon." -ForegroundColor Green
    }
}

Read-Host "Press Enter to close"
