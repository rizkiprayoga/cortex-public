<#
.SYNOPSIS
    Nightly Postgres backup for the Cortex trading bot.

.DESCRIPTION
    Dumps the trading_bot database via pg_dump in compressed custom format,
    skipping the two large derived tables (engineered_features +
    feature_vectors) that monthly retrain regenerates automatically.
    Prunes old backups using a grandfather-father-son retention policy
    (7 daily + 4 weekly + 3 monthly). Logs to data/logs/db_backup.log.

.NOTES
    Runs unattended via a Windows Scheduled Task (see
    install_db_backup_task.ps1). Designed to exit non-zero on any
    failure so the Task Scheduler marks the run as failed — catches
    silent breakage via the dashboard staleness indicator (future work).

    Scheduled daily at 22:00 UTC = 05:00 Jakarta — the quiet post-NY
    rollover window when forex is between daily sessions and the bot
    is between H4 bars (20:00 UTC done, 00:00 UTC next).

    DB password is read from the .env file using a simple regex parse.
    No dependency on dotenv tooling so this runs even from a restricted
    Scheduled-Task context.

.PARAMETER DestDir
    Where dumps land. Defaults to g:\AI_Trading_Bot\db_backups\ per the
    2026-04-19 decision (same disk, different folder).

.PARAMETER PgDumpPath
    Full path to pg_dump.exe. Defaults to the PostgreSQL 18 install.

.EXAMPLE
    powershell.exe -File scripts\db_backup.ps1
#>
param(
    [string]$DestDir    = "g:\AI_Trading_Bot\db_backups",
    [string]$PgDumpPath = "C:\Program Files\PostgreSQL\18\bin\pg_dump.exe",
    [string]$RepoRoot   = "g:\AI_Trading_Bot\Cortex",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Logging — tee to log file + stdout
# ---------------------------------------------------------------------------
$LogFile = Join-Path $RepoRoot "data\logs\db_backup.log"
$LogDir  = Split-Path $LogFile -Parent
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

function Log($msg) {
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss 'UTC'")
    $line = "[$ts] $msg"
    Add-Content -Path $LogFile -Value $line -Encoding utf8
    Write-Host $line
}

Log "db_backup.ps1 START (dryrun=$($DryRun.IsPresent))"

# ---------------------------------------------------------------------------
# Parse DSN from .env
# ---------------------------------------------------------------------------
$EnvFile = Join-Path $RepoRoot ".env"
if (-not (Test-Path $EnvFile)) {
    Log "FATAL: .env not found at $EnvFile"
    exit 1
}

# Match POSTGRES_DSN=postgresql[+driver]://user:pass@host:port/dbname
$dsnLine = Select-String -Path $EnvFile -Pattern "^POSTGRES_DSN=" | Select-Object -First 1
if (-not $dsnLine) {
    Log "FATAL: POSTGRES_DSN not set in .env"
    exit 1
}
$dsn = $dsnLine.Line -replace "^POSTGRES_DSN=", ""
if ($dsn -notmatch "^postgres(?:ql)?(?:\+[a-z]+)?://([^:]+):([^@]+)@([^:/]+):(\d+)/(.+?)(\?.*)?$") {
    Log "FATAL: POSTGRES_DSN format not recognized. Expected postgresql[+driver]://user:pass@host:port/dbname"
    exit 1
}
$pgUser = $Matches[1]
$pgPass = $Matches[2]
$pgHost = $Matches[3]
$pgPort = $Matches[4]
$pgDb   = $Matches[5]
Log "Parsed DSN: user=$pgUser host=$pgHost port=$pgPort db=$pgDb"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if (-not (Test-Path $PgDumpPath)) {
    Log "FATAL: pg_dump.exe not found at $PgDumpPath"
    exit 1
}
if (-not (Test-Path $DestDir)) {
    Log "Creating backup directory: $DestDir"
    if (-not $DryRun) { New-Item -ItemType Directory -Path $DestDir -Force | Out-Null }
}

# ---------------------------------------------------------------------------
# Dump
# ---------------------------------------------------------------------------
$dateStr = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd")
$fileName = "cortex-$dateStr.dump"
$outFile = Join-Path $DestDir $fileName
Log "Dumping $pgDb -> $outFile"
Log "Excluding (table data only, schema kept): engineered_features, feature_vectors (derived, regenerable via monthly retrain), feature_store + 432 monthly partitions (regenerable via scripts/backfill_feature_store.py)"

# Use PGPASSWORD env var rather than passing on command line (visible in
# process list otherwise). -Fc = custom format (compressed, indexable on
# restore). --no-owner strips owner metadata so a restore doesn't need
# the same OS-level role. --no-acl drops grants for the same reason.
$env:PGPASSWORD = $pgPass
try {
    $dumpArgs = @(
        "-h", $pgHost,
        "-p", $pgPort,
        "-U", $pgUser,
        "-d", $pgDb,
        "-Fc",
        "--no-owner",
        "--no-acl",
        "--exclude-table=engineered_features",
        "--exclude-table=feature_vectors",
        # feature_store + its 432 monthly partitions are regenerable via
        # `python -m scripts.backfill_feature_store`. Excluding the table
        # data (but keeping the schema) avoids inflating dumps with cached
        # external-source data we can always re-fetch.
        "--exclude-table-data=feature_store",
        "--exclude-table-data=feature_store_*",
        "-f", $outFile
    )
    if ($DryRun) {
        Log "DRYRUN: would run pg_dump $($dumpArgs -join ' ')"
    } else {
        $proc = Start-Process -FilePath $PgDumpPath -ArgumentList $dumpArgs `
                              -NoNewWindow -Wait -PassThru `
                              -RedirectStandardError (Join-Path $DestDir ".pg_dump.err")
        $errText = ""
        $errFile = Join-Path $DestDir ".pg_dump.err"
        if (Test-Path $errFile) {
            $errText = Get-Content $errFile -Raw
            Remove-Item $errFile -ErrorAction SilentlyContinue
        }
        if ($proc.ExitCode -ne 0) {
            Log "FATAL: pg_dump exited $($proc.ExitCode). stderr:"
            Log $errText
            exit $proc.ExitCode
        }
        if (Test-Path $outFile) {
            $sizeMB = [math]::Round((Get-Item $outFile).Length / 1MB, 2)
            Log "Dump OK: $fileName ($sizeMB MB)"
        } else {
            Log "FATAL: dump completed exit=0 but output file not found"
            exit 1
        }
    }
} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------------------
# Retention — grandfather-father-son (GFS)
#   Keep:
#     - last 7 daily dumps
#     - last 4 weekly dumps (Sundays)
#     - last 3 monthly dumps (1st of the month)
#   Prune everything else.
# ---------------------------------------------------------------------------
Log "Pruning old dumps (retention: 7 daily + 4 weekly + 3 monthly)"
$allDumps = Get-ChildItem -Path $DestDir -Filter "cortex-*.dump" |
    Where-Object { $_.Name -match "^cortex-(\d{4})-(\d{2})-(\d{2})\.dump$" } |
    ForEach-Object {
        [PSCustomObject]@{
            File = $_
            Date = [datetime]::ParseExact(
                $_.BaseName.Substring(7), "yyyy-MM-dd",
                [System.Globalization.CultureInfo]::InvariantCulture
            )
        }
    } |
    Sort-Object -Property Date -Descending

$keep = New-Object System.Collections.Generic.HashSet[string]

# Daily: 7 most recent
$allDumps | Select-Object -First 7 | ForEach-Object { $keep.Add($_.File.FullName) | Out-Null }

# Weekly: 4 most recent Sundays
$allDumps | Where-Object { $_.Date.DayOfWeek -eq "Sunday" } |
    Select-Object -First 4 | ForEach-Object { $keep.Add($_.File.FullName) | Out-Null }

# Monthly: 3 most recent 1st-of-month dumps
$allDumps | Where-Object { $_.Date.Day -eq 1 } |
    Select-Object -First 3 | ForEach-Object { $keep.Add($_.File.FullName) | Out-Null }

$toDelete = $allDumps | Where-Object { -not $keep.Contains($_.File.FullName) }
if ($toDelete.Count -eq 0) {
    Log "Retention: nothing to prune (kept $($keep.Count) of $($allDumps.Count))"
} else {
    foreach ($d in $toDelete) {
        Log "Pruning: $($d.File.Name)"
        if (-not $DryRun) {
            Remove-Item -Path $d.File.FullName -Force
        }
    }
    Log "Retention: kept $($keep.Count), pruned $($toDelete.Count)"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
$remaining = Get-ChildItem -Path $DestDir -Filter "cortex-*.dump"
$totalMB = [math]::Round(($remaining | Measure-Object Length -Sum).Sum / 1MB, 2)
Log "Final inventory: $($remaining.Count) dumps, $totalMB MB total"
Log "db_backup.ps1 DONE"
exit 0
