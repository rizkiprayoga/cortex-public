# ============================================================================
#  unlock_dashboard.ps1 — one-click unlock + open dashboard
#
#  Triggered by the "Cortex Unlock" desktop shortcut. Does three things,
#  silently, and exits:
#
#    1. Deletes data\state\DASHBOARD_LOCKED.flag if present (the file-
#       based unlock — works even when the bot is offline).
#    2. POSTs /api/system/unlock as a fallback (handles the legacy
#       in-memory lock if you have an older bot version still running).
#    3. Opens http://localhost:8787 in your default browser.
#
#  No console window. No prompts. Just unlocked.
# ============================================================================

$ProjectRoot = "g:\AI_Trading_Bot\Cortex"
$LockFile    = Join-Path $ProjectRoot "data\state\DASHBOARD_LOCKED.flag"

# 1. Delete the file-based lock flag (works even if bot is stopped)
if (Test-Path $LockFile) {
    try {
        Remove-Item $LockFile -Force -ErrorAction Stop
    } catch {
        # File might be re-created instantly by the bot — ignore
    }
}

# 2. Hit the API unlock endpoint as a belt-and-braces measure (handles
#    the case where an older bot version is running with the in-memory
#    lock that doesn't read from the file).
try {
    Invoke-WebRequest `
        -Method POST `
        -Uri "http://127.0.0.1:8787/api/system/unlock" `
        -TimeoutSec 3 `
        -UseBasicParsing `
        -ErrorAction SilentlyContinue | Out-Null
} catch {
    # Bot might not be running yet — that's fine, the file delete handles
    # it once the bot reads the absent file on its next is_locked check.
}

# 3. Open the dashboard
Start-Process "http://localhost:8787"
