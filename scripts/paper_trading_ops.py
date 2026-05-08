"""
paper_trading_ops.py — one-stop operator helper for paper trading.

Three subcommands covering the paper-trading lifecycle:

    preflight     Validate everything is ready BEFORE you start the bot.
                  Env vars, models, DB, MT5, no stale halt flag.

    snapshot      Take a labeled state snapshot BEFORE a PC restart so you
                  can verify continuity after rebooting. Writes JSON to
                  data/logs/restart_snapshot.json.

    verify        Run AFTER the PC reboots to confirm the whole stack came
                  back up: PostgreSQL service, MT5 terminal, bot process,
                  dashboard API, log continuity, broker vs DB consistency.
                  Compares against the last snapshot if one exists.

Usage:
    python scripts/paper_trading_ops.py preflight
    python scripts/paper_trading_ops.py snapshot
    python scripts/paper_trading_ops.py verify
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

SNAPSHOT_PATH = ROOT / "data" / "logs" / "restart_snapshot.json"
TRADING_LOG = ROOT / "data" / "logs" / "trading_bot.log"
ERROR_LOG = ROOT / "data" / "logs" / "errors.log"
HALT_FLAG = ROOT / "data" / "logs" / "TRADING_HALTED.flag"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

_passes: list[str] = []
_failures: list[tuple[str, str]] = []
_warnings: list[tuple[str, str]] = []


def ok(label: str) -> None:
    _passes.append(label)
    print(f"  {GREEN}PASS{RESET}  {label}")


def fail(label: str, detail: str = "") -> None:
    _failures.append((label, detail))
    print(f"  {RED}FAIL{RESET}  {label}")
    if detail:
        print(f"        {detail}")


def warn(label: str, detail: str = "") -> None:
    _warnings.append((label, detail))
    print(f"  {YELLOW}WARN{RESET}  {label}")
    if detail:
        print(f"        {detail}")


def header(text: str) -> None:
    print()
    print(BOLD + "=" * 72 + RESET)
    print(BOLD + f" {text}" + RESET)
    print(BOLD + "=" * 72 + RESET)


def _reset_results() -> None:
    _passes.clear()
    _failures.clear()
    _warnings.clear()


def _print_summary(mode: str) -> int:
    print()
    print("-" * 72)
    print(f"  {GREEN}{len(_passes)} passed{RESET}   "
          f"{YELLOW}{len(_warnings)} warnings{RESET}   "
          f"{RED}{len(_failures)} failed{RESET}")
    if _warnings:
        print("\n  Warnings (non-blocking):")
        for label, detail in _warnings:
            print(f"    - {label}{': ' + detail if detail else ''}")
    if _failures:
        print("\n  Failed — address these before proceeding:")
        for label, detail in _failures:
            print(f"    - {label}{': ' + detail if detail else ''}")
        print(f"\n  {RED}{mode.upper()} CHECK FAILED{RESET}")
        return 1
    if _warnings:
        print(f"\n  {YELLOW}{mode.upper()} CHECK PASSED WITH WARNINGS{RESET}")
    else:
        print(f"\n  {GREEN}{mode.upper()} CHECK PASSED — you're good to go{RESET}")
    return 0


# =========================================================================
# Individual probes (shared across subcommands)
# =========================================================================

def probe_env_vars() -> None:
    required = [
        "POSTGRES_DSN",
        "POSTGRES_PASSWORD",
        "MT5_LOGIN",
        "MT5_PASSWORD",
        "MT5_SERVER",
        "DASHBOARD_PW_HASH",
        "DASHBOARD_JWT_SECRET",
    ]
    optional = [
        "DASHBOARD_USERNAME",
        "MT5_TERMINAL_PATH",
        "FRED_API_KEY",
        "NEWS_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "ALERT_EMAIL_FROM",
    ]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        fail("env: required vars set", f"missing: {', '.join(missing)}")
    else:
        ok("env: required vars set")

    # Sanity on specific formats
    pw = os.environ.get("DASHBOARD_PW_HASH", "")
    if pw.startswith("$2") and len(pw) >= 50:
        ok("env: DASHBOARD_PW_HASH looks like bcrypt")
    elif pw and not pw.startswith("$2"):
        fail("env: DASHBOARD_PW_HASH", f"not bcrypt — got {pw[:15]}...")

    secret = os.environ.get("DASHBOARD_JWT_SECRET", "")
    if len(secret) >= 32:
        ok("env: DASHBOARD_JWT_SECRET strong (>=32 chars)")
    elif secret:
        fail("env: DASHBOARD_JWT_SECRET", f"too short ({len(secret)} chars, need >=32)")

    # Optional (info only)
    absent_optional = [v for v in optional if not os.environ.get(v)]
    if absent_optional:
        warn("env: optional vars unset (features degraded)",
             ", ".join(absent_optional))


def probe_models() -> None:
    missing = []
    for sym in ("XAUUSD", "USDJPY", "EURUSD", "USDCAD"):
        for path in [
            f"data/models/hmm_{sym}.pkl",
            f"data/models/lstm_{sym}.pt",
            f"data/models/lstm_scaler_{sym}.pkl",
            f"data/models/lstm_{sym}.pca.pkl",
        ]:
            if not (ROOT / path).exists():
                missing.append(path)
    if missing:
        fail("models: all 4 symbols have HMM + LSTM + scaler + PCA",
             f"missing: {', '.join(missing[:3])}" + ("..." if len(missing) > 3 else ""))
    else:
        ok("models: all 4 symbols have HMM + LSTM + scaler + PCA")


def probe_postgres() -> dict:
    """Returns DB snapshot dict if reachable, else empty dict."""
    # Service state (Windows)
    if platform.system() == "Windows":
        try:
            r = subprocess.run(
                ["sc", "query", "postgresql-x64-18"],
                capture_output=True, text=True, timeout=10,
            )
            if "RUNNING" in r.stdout:
                ok("postgres: Windows service RUNNING")
            else:
                fail("postgres: Windows service",
                     "not RUNNING — try: sc start postgresql-x64-18")
        except Exception as exc:
            warn("postgres: service check skipped", str(exc))

    # Actual connect + row counts
    try:
        import asyncio
        from src.data_pipeline.data_store import DataStore
        async def _probe():
            store = DataStore()
            await store.connect()
            try:
                from sqlalchemy import text
                async with store._session_factory() as session:
                    tables = {}
                    for t in ("ohlcv_bars", "trades", "signals",
                              "equity_history", "model_predictions"):
                        r = await session.execute(text(f"SELECT COUNT(*) FROM {t}"))
                        tables[t] = int(r.scalar() or 0)
                    size = await session.execute(
                        text("SELECT pg_database_size(current_database())")
                    )
                    tables["_db_bytes"] = int(size.scalar() or 0)
                return tables
            finally:
                await store.close()
        info = asyncio.run(_probe())
        ok(f"postgres: reachable ({info['_db_bytes'] / 1e6:.1f} MB)")
        return info
    except Exception as exc:
        fail("postgres: connect", str(exc))
        return {}


def probe_mt5_process() -> bool:
    """Is MT5 terminal running? (Windows only)"""
    if platform.system() != "Windows":
        warn("mt5: process check skipped (non-Windows)")
        return False
    try:
        r = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq terminal64.exe"],
            capture_output=True, text=True, timeout=10,
        )
        if "terminal64.exe" in r.stdout:
            ok("mt5: terminal64.exe process running")
            return True
        else:
            warn("mt5: terminal64.exe not running",
                 "launch MetaTrader 5 before starting the bot")
            return False
    except Exception as exc:
        warn("mt5: process check failed", str(exc))
        return False


def probe_mt5_terminal_path() -> None:
    path = os.environ.get(
        "MT5_TERMINAL_PATH",
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
    )
    if Path(path).exists():
        ok(f"mt5: terminal binary exists")
    else:
        fail("mt5: terminal binary exists",
             f"MT5_TERMINAL_PATH={path} does not exist")


def probe_mt5_connect() -> dict:
    """Try to connect to the broker. Returns account info dict if OK."""
    try:
        from src.broker.mt5_connector import MT5Connector
        conn = MT5Connector(max_retries=2, retry_delay=3)
        if conn.connect():
            import MetaTrader5 as mt5
            info = mt5.account_info()
            positions = mt5.positions_get() or []
            snapshot = {
                "login":    int(info.login) if info else 0,
                "server":   str(info.server) if info else "",
                "balance":  float(info.balance) if info else 0.0,
                "equity":   float(info.equity) if info else 0.0,
                "margin":   float(info.margin) if info else 0.0,
                "free_margin": float(info.margin_free) if info else 0.0,
                "positions_count": len(positions),
                "positions": [{
                    "ticket": int(p.ticket),
                    "symbol": str(p.symbol),
                    "type":   "buy" if p.type == 0 else "sell",
                    "volume": float(p.volume),
                    "price_open": float(p.price_open),
                    "profit": float(p.profit),
                } for p in positions],
            }
            ok(f"mt5: connected to {snapshot['server']} account {snapshot['login']}")
            ok(f"mt5: equity=${snapshot['equity']:.2f} "
               f"open_positions={snapshot['positions_count']}")
            conn.disconnect()
            return snapshot
        else:
            fail("mt5: connect", "broker login failed")
            return {}
    except Exception as exc:
        fail("mt5: connect", str(exc))
        return {}


def probe_halt_flag() -> None:
    if not HALT_FLAG.exists():
        ok("safety: TRADING_HALTED.flag absent (not halted)")
        return
    # Flag exists — verify its HMAC
    try:
        from src.safety.circuit_breaker import CircuitBreaker
        valid = CircuitBreaker._verify_halt_flag(HALT_FLAG)
        if valid:
            warn("safety: TRADING_HALTED.flag present (HMAC-valid)",
                 "the bot will refuse new entries. Resolve the cause, "
                 "then delete: rm data/logs/TRADING_HALTED.flag")
        else:
            fail("safety: TRADING_HALTED.flag HMAC invalid",
                 "file looks tampered — delete it manually if you trust the source")
    except Exception as exc:
        warn("safety: halt flag verify failed", str(exc))


def probe_dashboard(timeout_s: float = 2.0) -> bool:
    """Is the dashboard API listening on :8787?"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout_s)
            s.connect(("127.0.0.1", 8787))
        ok("dashboard: API port 8787 listening")
        return True
    except (ConnectionRefusedError, OSError, socket.timeout):
        return False


def probe_bot_process() -> bool:
    """Is there a python.exe running main.py?"""
    if platform.system() != "Windows":
        return False
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'",
             "get", "CommandLine", "/FORMAT:LIST"],
            capture_output=True, text=True, timeout=15,
        )
        return "main.py" in r.stdout
    except Exception:
        return False


def probe_recent_log_activity(max_age_seconds: int = 600) -> None:
    if not TRADING_LOG.exists():
        warn("log: trading_bot.log missing",
             "bot may never have written a log here — will appear on first tick")
        return
    age = time.time() - TRADING_LOG.stat().st_mtime
    if age < max_age_seconds:
        ok(f"log: trading_bot.log fresh ({int(age)}s old)")
    else:
        warn("log: trading_bot.log stale",
             f"last write {int(age)}s ago — bot may be stopped")


def probe_smoke_tests() -> None:
    try:
        r = subprocess.run(
            [sys.executable, "scripts/smoke_test.py", "--skip-tests"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            ok("smoke: all integration checks pass")
        else:
            fail("smoke: integration checks",
                 r.stdout.strip().splitlines()[-1] if r.stdout else "see output")
    except Exception as exc:
        warn("smoke: run skipped", str(exc))


# =========================================================================
# Subcommand: preflight
# =========================================================================

def cmd_preflight(args) -> int:
    _reset_results()
    header("PRE-FLIGHT — validate readiness before starting the bot")

    probe_env_vars()
    probe_models()
    probe_postgres()
    probe_mt5_terminal_path()
    mt5_running = probe_mt5_process()
    # Only attempt a broker login if MT5 is already running
    if mt5_running:
        probe_mt5_connect()
    probe_halt_flag()
    if not args.skip_smoke:
        probe_smoke_tests()

    print()
    print(f"  Next step: start the bot with  {CYAN}python main.py{RESET}")
    print(f"  Dashboard: {CYAN}http://localhost:8787{RESET}")
    print(f"  When ready for unattended operation, run:")
    print(f"    {CYAN}powershell -ExecutionPolicy Bypass -File scripts\\install_autostart.ps1{RESET}")
    return _print_summary("preflight")


# =========================================================================
# Subcommand: snapshot (pre-restart)
# =========================================================================

def cmd_snapshot(args) -> int:
    _reset_results()
    header("SNAPSHOT — capture state before PC restart")

    snap: dict = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "hostname": platform.node(),
    }

    # git HEAD
    try:
        snap["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
        ok(f"git: HEAD = {snap['git_commit'][:10]}")
    except Exception as exc:
        warn("git: HEAD read failed", str(exc))
        snap["git_commit"] = ""

    # Postgres snapshot
    db = probe_postgres()
    snap["postgres"] = db

    # MT5 + broker snapshot (without touching the running bot if any)
    mt5_running = probe_mt5_process()
    if mt5_running:
        snap["mt5"] = probe_mt5_connect()
    else:
        warn("snapshot: MT5 not running — broker snapshot skipped",
             "launch MT5 terminal before snapshotting for full fidelity")
        snap["mt5"] = {}

    # Dashboard / bot process (if running)
    snap["dashboard_up"] = probe_dashboard()
    snap["bot_process_running"] = probe_bot_process()
    if snap["bot_process_running"]:
        ok("bot: python main.py currently running")
    else:
        warn("bot: not running",
             "snapshot will lack live-state context; post-restart verify "
             "won't be able to confirm a restart happened")

    # Last N lines of trading log
    if TRADING_LOG.exists():
        tail_lines = TRADING_LOG.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-50:]
        snap["trading_log_tail"] = tail_lines
        ok(f"log: captured last {len(tail_lines)} lines of trading_bot.log")
    else:
        snap["trading_log_tail"] = []

    # Halt flag state
    snap["halt_flag_present"] = HALT_FLAG.exists()

    # Write snapshot
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(
        json.dumps(snap, indent=2, default=str), encoding="utf-8",
    )
    ok(f"snapshot saved to {SNAPSHOT_PATH}")

    print()
    print(f"  Snapshot captured at {CYAN}{snap['timestamp_utc']}{RESET}")
    if snap.get("mt5"):
        print(f"  Broker equity:  {CYAN}${snap['mt5'].get('equity', 0):,.2f}{RESET}")
        print(f"  Open positions: {CYAN}{snap['mt5'].get('positions_count', 0)}{RESET}")
    print()
    print(f"  You can now safely restart the PC.")
    print(f"  After reboot, run: {CYAN}python scripts/paper_trading_ops.py verify{RESET}")
    return _print_summary("snapshot")


# =========================================================================
# Subcommand: verify (post-restart)
# =========================================================================

def cmd_verify(args) -> int:
    _reset_results()
    header("POST-RESTART VERIFY — confirm the stack is back up")

    # Load the most recent snapshot for comparison
    snap: dict = {}
    if SNAPSHOT_PATH.exists():
        try:
            snap = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
            age_s = (datetime.now(tz=timezone.utc)
                     - datetime.fromisoformat(snap["timestamp_utc"])).total_seconds()
            ok(f"snapshot: loaded (age {int(age_s / 60)} min)")
        except Exception as exc:
            warn("snapshot: parse failed", str(exc))
    else:
        warn("snapshot: no restart_snapshot.json found",
             "run 'snapshot' before restart to enable diff comparisons")

    # Service + process states
    db = probe_postgres()
    mt5_up = probe_mt5_process()
    probe_mt5_terminal_path()

    # If autostart is configured, the bot should be running
    bot_up = probe_bot_process()
    dash_up = probe_dashboard()
    if bot_up:
        ok("bot: python main.py is running (autostart working)")
    else:
        warn("bot: not running yet",
             f"start it manually: {CYAN}python main.py{RESET}  or enable autostart "
             "via install_autostart.ps1")
    if dash_up:
        ok("dashboard: responding on port 8787")
    elif bot_up:
        warn("dashboard: port 8787 not yet listening (may still be initializing)")

    # Halt flag check
    probe_halt_flag()

    # Broker comparison (if MT5 + snapshot available)
    if mt5_up and snap.get("mt5"):
        now = probe_mt5_connect()
        if now:
            old = snap["mt5"]
            delta_eq = now.get("equity", 0) - old.get("equity", 0)
            # Some drift is expected (floating PnL, swap)
            if abs(delta_eq) < 100:
                ok(f"broker: equity stable (delta=${delta_eq:+.2f} vs snapshot)")
            else:
                warn(f"broker: equity drifted delta=${delta_eq:+.2f}",
                     "check if positions changed / stops hit during downtime")
            if now.get("positions_count") == old.get("positions_count"):
                ok(f"broker: position count matches "
                   f"({now.get('positions_count')} open)")
            else:
                warn(f"broker: position count changed "
                     f"{old.get('positions_count')} -> "
                     f"{now.get('positions_count')}",
                     "some positions may have hit SL/TP during downtime — "
                     "verify in data/logs/trade_events.csv")

    # Log continuity check
    probe_recent_log_activity(max_age_seconds=1200)

    # DB growth check (signals/equity history should accumulate if bot ran)
    if snap.get("postgres") and db:
        for tbl in ("signals", "equity_history"):
            before = snap["postgres"].get(tbl, 0)
            after = db.get(tbl, 0)
            if after > before:
                ok(f"db: {tbl} grew +{after - before} rows since snapshot")
            elif after == before and bot_up:
                warn(f"db: {tbl} unchanged since snapshot",
                     "if the bot has been running >5 min, expect some rows")

    # Reconcile log hint
    if TRADING_LOG.exists():
        recent = "\n".join(
            TRADING_LOG.read_text(encoding="utf-8", errors="replace")
            .splitlines()[-200:]
        )
        if "reconciled" in recent.lower():
            ok("log: 'reconciled' marker present (positions re-attached)")
        elif bot_up:
            warn("log: no 'reconciled' marker in last 200 lines",
                 "check data/logs/trading_bot.log for startup messages")

    print()
    print(f"  Recommended next step: run the paper-trading analyzer")
    print(f"    {CYAN}python scripts/analyze_paper_trading.py --since 1{RESET}")
    return _print_summary("verify")


# =========================================================================
# Entry
# =========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="paper_trading_ops",
        description="Operator helper for paper trading lifecycle",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preflight", help="Validate readiness before start")
    p_pre.add_argument("--skip-smoke", action="store_true",
                       help="Skip the integration smoke tests (faster)")
    p_pre.set_defaults(func=cmd_preflight)

    p_snap = sub.add_parser("snapshot", help="Capture state before PC restart")
    p_snap.set_defaults(func=cmd_snapshot)

    p_ver = sub.add_parser("verify", help="Verify stack came back after reboot")
    p_ver.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
