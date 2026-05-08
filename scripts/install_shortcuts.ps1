# ============================================================================
#  install_shortcuts.ps1 — one desktop shortcut that launches everything
#
#  Run once (no admin needed):
#      powershell -ExecutionPolicy Bypass -File scripts\install_shortcuts.ps1
#
#  Installs a single "Cortex" shortcut on your Desktop and in the Start
#  Menu. Double-clicking it runs scripts/launch.ps1, which:
#    1. Runs preflight checks (aborts if anything critical is wrong).
#    2. Starts the bot in this window (logs visible, Ctrl-C to stop).
#    3. Opens http://localhost:8787 in your browser once the API is ready.
#
#  No separate shortcuts for frontend / dashboard / preflight — one button
#  does it all. (The frontend dev server is optional; add -WithFrontend
#  to the shortcut's Arguments field in Properties if you want it.)
#
#  Old multi-shortcut install? Delete these .lnk files manually:
#    Desktop\Cortex Bot.lnk
#    Desktop\Cortex Frontend.lnk
#    Desktop\Cortex Dashboard.lnk
#    Desktop\Cortex Preflight.lnk
#  The new installer also auto-deletes them (see below).
# ============================================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = "g:\AI_Trading_Bot\Cortex"
$LaunchPs1   = Join-Path $ProjectRoot "scripts\launch.ps1"
$Icon        = "$env:SystemRoot\System32\SHELL32.dll"

if (-not (Test-Path $LaunchPs1)) {
    Write-Host "ERROR: launch.ps1 not found at $LaunchPs1" -ForegroundColor Red
    exit 1
}

$Desktop   = [Environment]::GetFolderPath("Desktop")
$StartMenu = Join-Path ([Environment]::GetFolderPath("Programs")) "Cortex Trading Bot"

if (-not (Test-Path $StartMenu)) {
    New-Item -ItemType Directory -Path $StartMenu | Out-Null
}

# Clean up old multi-shortcut install if present (incl. the deprecated
# 'Cortex Health' which has been folded into the dashboard System tab).
$oldNames = @("Cortex Bot", "Cortex Frontend", "Cortex Dashboard",
              "Cortex Preflight", "Cortex Health")
foreach ($name in $oldNames) {
    foreach ($dir in @($Desktop, $StartMenu)) {
        $old = Join-Path $dir "$name.lnk"
        if (Test-Path $old) {
            Remove-Item $old -Force
            Write-Host "  Removed old shortcut: $old" -ForegroundColor DarkGray
        }
    }
}

function New-Shortcut {
    param(
        [string]$Name,
        [string]$Target,
        [string]$Arguments,
        [string]$WorkingDir,
        [string]$IconPath,
        [int]$IconIndex,
        [string]$Description
    )
    $wsh = New-Object -ComObject WScript.Shell
    foreach ($dir in @($Desktop, $StartMenu)) {
        $path = Join-Path $dir "$Name.lnk"
        $sc = $wsh.CreateShortcut($path)
        $sc.TargetPath       = $Target
        $sc.Arguments        = $Arguments
        $sc.WorkingDirectory = $WorkingDir
        $sc.IconLocation     = "$IconPath,$IconIndex"
        $sc.Description      = $Description
        $sc.WindowStyle      = 1  # normal window
        $sc.Save()
        Write-Host "  Created: $path" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Installing the 'Cortex' one-click launcher..." -ForegroundColor Cyan
Write-Host ""

# The single Cortex launcher shortcut. PowerShell is invoked with
# -ExecutionPolicy Bypass so the user never needs to mess with
# Set-ExecutionPolicy.
New-Shortcut `
    -Name        "Cortex" `
    -Target      "powershell.exe" `
    -Arguments   "-NoLogo -ExecutionPolicy Bypass -File `"$LaunchPs1`"" `
    -WorkingDir  $ProjectRoot `
    -IconPath    $Icon `
    -IconIndex   137 `
    -Description "Start Cortex: preflight + bot + auto-open dashboard."

# Unlock-dashboard helper: deletes the lock-flag file and opens the
# dashboard in the browser. Closes itself in 2s — no clutter.
$UnlockPs1 = Join-Path $ProjectRoot "scripts\unlock_dashboard.ps1"
New-Shortcut `
    -Name        "Cortex Unlock" `
    -Target      "powershell.exe" `
    -Arguments   "-NoLogo -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$UnlockPs1`"" `
    -WorkingDir  $ProjectRoot `
    -IconPath    $Icon `
    -IconIndex   47 `
    -Description "Unlock the Cortex dashboard and open it in the browser."

# Health monitor lives inside the dashboard System tab now (no separate
# shortcut). Old 'Cortex Health' shortcut auto-removed below if present.

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host ""
Write-Host "Double-click the 'Cortex' icon on your Desktop to launch." -ForegroundColor Cyan
Write-Host "  It runs:  preflight  ->  python main.py  ->  opens http://localhost:8787"
Write-Host ""
Write-Host "If you also want the frontend dev server to start automatically," -ForegroundColor Yellow
Write-Host "right-click the Cortex shortcut -> Properties -> Target and append" -ForegroundColor Yellow
Write-Host "'-WithFrontend' to the arguments." -ForegroundColor Yellow
Write-Host ""
