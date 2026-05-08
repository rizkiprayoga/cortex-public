# ============================================================================
#  uninstall_autostart.ps1 — Remove CortexTradingBot from Windows Task Scheduler
#
#  Run as Administrator:
#      powershell -ExecutionPolicy Bypass -File scripts\uninstall_autostart.ps1
# ============================================================================

$TaskName = "CortexTradingBot"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $existing) {
    Write-Host "Task '$TaskName' is not installed — nothing to do." -ForegroundColor Yellow
    exit 0
}

try {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "[OK] Task '$TaskName' removed." -ForegroundColor Green
}
catch {
    Write-Host "ERROR: failed to remove task." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Tip: run this script as Administrator." -ForegroundColor Yellow
    exit 1
}
