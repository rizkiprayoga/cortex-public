<#
.SYNOPSIS
    Register a daily Windows Scheduled Task that runs db_backup.ps1 at
    22:00 UTC (05:00 Jakarta — post-NY-rollover, bot between H4 bars).

.DESCRIPTION
    Creates Scheduled Task "CortexDBBackup". Already-existing task is
    unregistered first, so this script is safe to re-run after editing
    db_backup.ps1.

    Mirrors the pattern used in install_autostart.ps1 — task runs as
    the current user, uses powershell.exe -NoProfile -File, allows
    multiple instances only to not pile up.

    Timing note: the task is scheduled in LOCAL time with the computer's
    current timezone offset applied. On a machine set to Asia/Jakarta
    (GMT+7), specifying 05:00 local = 22:00 UTC. The script detects the
    machine's offset from UTC and computes the local time that matches
    the desired 22:00 UTC window.

.EXAMPLE
    powershell.exe -ExecutionPolicy Bypass -File scripts\install_db_backup_task.ps1
#>
param(
    [string]$RepoRoot = "g:\AI_Trading_Bot\Cortex",
    [string]$TaskName = "CortexDBBackup",
    [string]$TargetUtcHour = "22:00"
)

$ErrorActionPreference = "Stop"

$ScriptPath = Join-Path $RepoRoot "scripts\db_backup.ps1"
if (-not (Test-Path $ScriptPath)) {
    Write-Host "ERROR: $ScriptPath not found" -ForegroundColor Red
    exit 1
}

# Compute local time matching 22:00 UTC.
#   LocalTime = UTC + offset
#   Offset is taken from the current system (e.g. +07:00 Jakarta -> 05:00 local)
$utcHour = [int]($TargetUtcHour.Split(":")[0])
$utcMin  = [int]($TargetUtcHour.Split(":")[1])
$utcTime = (Get-Date).ToUniversalTime().Date.AddHours($utcHour).AddMinutes($utcMin)
$localTime = $utcTime.ToLocalTime()
$localHour = $localTime.Hour
$localMin  = $localTime.Minute
$localHHMM = "{0:00}:{1:00}" -f $localHour, $localMin
Write-Host "Target: $TargetUtcHour UTC -> $localHHMM local (offset applied automatically)"

# Unregister any existing task with the same name first
try {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Unregistering existing Scheduled Task: $TaskName"
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
} catch {
    # Nothing to remove — fine
}

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$ScriptPath`"" `
    -WorkingDirectory $RepoRoot

$Trigger = New-ScheduledTaskTrigger -Daily -At $localHHMM

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

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
        -Description "Cortex Postgres DB nightly backup (22:00 UTC = 05:00 Jakarta)" `
        | Out-Null
    Write-Host "Scheduled Task '$TaskName' registered at $localHHMM daily." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next run info:"
    (Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo) | Format-List LastRunTime, LastTaskResult, NextRunTime
    Write-Host "To run manually now:  Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "To unregister:        Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
} catch {
    Write-Host "Register-ScheduledTask failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
