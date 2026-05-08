"""
routes/system.py — System status, bot control, and dashboard lock endpoints.

GET   /api/system/health       → liveness check (always 200, no auth)
GET   /api/system/status       → system status (auth required)
GET   /api/bot/status          → bot running state
POST  /api/bot/control         → start / pause / stop
POST  /api/system/unlock       → unlock dashboard (localhost only)
POST  /api/system/lock         → lock dashboard
GET   /api/system/lock-status  → lock state (no auth, minimal info)
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.auth import get_current_user
from src.api.schemas import (
    BotControlRequest,
    BotStatusResponse,
    HealthResponse,
    LockStatusResponse,
    RestartResponse,
    SystemStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


def _get_live_state(request: Request):
    return request.app.state.live_state


def _is_localhost(request: Request) -> bool:
    """
    Check if the request originates from localhost.

    Security Audit C4: reject any request with proxy headers —
    if X-Forwarded-For is present, a reverse proxy is in the chain
    and request.client.host is unreliable for locality detection.
    """
    if request.client is None:
        return False
    # Reject if any proxy headers indicate upstream forwarding
    proxy_headers = ("x-forwarded-for", "x-real-ip", "forwarded",
                      "x-forwarded-host", "cf-connecting-ip")
    for h in proxy_headers:
        if request.headers.get(h):
            return False
    host = request.client.host
    return host in ("127.0.0.1", "::1", "localhost")


# ---------------------------------------------------------------------------
# Health (no auth — used by monitoring / Cloudflare health checks)
# ---------------------------------------------------------------------------

@router.get("/api/system/health", response_model=HealthResponse)
async def health():
    """Liveness probe — always returns 200."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Lock / unlock (dashboard security gate)
# ---------------------------------------------------------------------------

@router.get("/api/system/lock-status", response_model=LockStatusResponse)
async def lock_status(request: Request):
    """
    Public endpoint — returns only lock state and whether the client
    is on localhost. No account info, no version, no fingerprint.
    """
    ls = _get_live_state(request)
    return LockStatusResponse(
        locked=ls.dashboard_lock.is_locked,
        is_local=_is_localhost(request),
    )


@router.post("/api/system/unlock")
async def unlock_dashboard(
    request: Request,
    _user: str = Depends(get_current_user),
):
    """
    Unlock the dashboard.

    Requires BOTH:
    1. Valid JWT token (get_current_user dependency)
    2. Request originating from localhost (no proxy headers)

    Security Audit C4 + M2: JWT auth required in addition to localhost
    check, preventing any local process from unlocking without
    knowing the dashboard password.
    """
    if not _is_localhost(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Dashboard can only be unlocked from localhost",
        )
    ls = _get_live_state(request)
    ls.dashboard_lock.unlock()
    return {"status": "unlocked"}


@router.post("/api/system/lock")
async def lock_dashboard(
    request: Request,
    _user: str = Depends(get_current_user),
):
    """Lock the dashboard. Available from any authenticated session."""
    ls = _get_live_state(request)
    ls.dashboard_lock.lock()
    return {"status": "locked"}


# ---------------------------------------------------------------------------
# System status (auth required)
# ---------------------------------------------------------------------------

@router.get("/api/system/status", response_model=SystemStatusResponse)
async def system_status(
    request: Request,
    _user: str = Depends(get_current_user),
):
    """System overview: bot state, uptime, positions, breaker, health."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    uptime = (
        datetime.now(tz=timezone.utc) - ls.started_at
    ).total_seconds()

    # --- Health monitor fields (Phase D dashboard health card) ---
    # Heartbeat freshness
    hb_age: float | None = None
    hb_equity: float | None = None
    hb_open: int | None = None
    try:
        from pathlib import Path
        import json as _json
        hb_path = Path("data/logs/bot_heartbeat.json")
        if hb_path.exists():
            hb = _json.loads(hb_path.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(hb["timestamp_utc"])
            hb_age = (datetime.now(tz=timezone.utc) - ts).total_seconds()
            hb_equity = float(hb.get("equity") or 0.0)
            hb_open = int(hb.get("open_positions") or 0)
    except Exception:
        pass

    # Recent errors — last 10 minutes, with known-noise filter
    NOISE_PATTERNS = (
        "google_trends", "cot_data",
        # All FRED variants — provider often 500s briefly, falls back to neutral
        "FRED returned no data", "FRED fetch failed",
        "news_sentiment", "No keywords",
    )
    recent_errors: list[str] = []
    log_tail: list[str] = []
    try:
        from pathlib import Path
        from datetime import timedelta as _td
        cutoff = datetime.now() - _td(minutes=10)
        err_path = Path("data/logs/errors.log")
        if err_path.exists():
            for line in err_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                try:
                    line_ts = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                except (ValueError, IndexError):
                    continue
                if line_ts < cutoff:
                    continue
                if any(p in line for p in NOISE_PATTERNS):
                    continue
                recent_errors.append(line)
            recent_errors = recent_errors[-10:]  # cap

        log_path = Path("data/logs/trading_bot.log")
        if log_path.exists():
            log_tail = log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[-15:]
    except Exception:
        pass

    return SystemStatusResponse(
        bot_status=ls.bot_control.status.value,
        uptime_seconds=uptime,
        positions_count=len(ls.tracked_positions),
        breaker_active=ls.circuit_breaker.is_halted(),
        heartbeat_age_seconds=hb_age,
        heartbeat_equity=hb_equity,
        heartbeat_open_positions=hb_open,
        dashboard_locked=ls.dashboard_lock.is_locked,
        recent_errors=recent_errors,
        log_tail=log_tail,
    )


# ---------------------------------------------------------------------------
# Bot control (auth required)
# ---------------------------------------------------------------------------

@router.get("/api/bot/status", response_model=BotStatusResponse)
async def bot_status(
    request: Request,
    _user: str = Depends(get_current_user),
):
    """Current bot running state."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()
    bc = ls.bot_control

    return BotStatusResponse(
        status=bc.status.value,
        last_change=bc.last_change,
        changed_by=bc.changed_by,
    )


@router.post("/api/bot/control", response_model=BotStatusResponse)
async def bot_control(
    body: BotControlRequest,
    request: Request,
    _user: str = Depends(get_current_user),
):
    """
    Control the bot: start, pause, or stop.

    - ``start``: resume from paused state
    - ``pause``: suspend trading loop (RiskMonitor stays alive)
    - ``stop``: halt trading + flatten all positions (requires
      ``confirmation: "STOP"`` in the request body)
    """
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()
    bc = ls.bot_control
    action = body.action.lower()

    if action == "start":
        bc.start(by=_user)
    elif action == "pause":
        bc.pause(by=_user)
    elif action == "stop":
        if body.confirmation != "STOP":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Stop requires confirmation: "STOP"',
            )
        bc.stop(by=_user)
        # EmergencyClose is triggered by the trading loop when it reads
        # STOPPED status — not from this handler. This keeps the broker
        # interaction on the trading thread, not the HTTP thread.
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown action: {action}. Use start, pause, or stop.",
        )

    return BotStatusResponse(
        status=bc.status.value,
        last_change=bc.last_change,
        changed_by=bc.changed_by,
    )


# ---------------------------------------------------------------------------
# Restart (auth required) — dashboard restart button
# ---------------------------------------------------------------------------

@router.post("/api/system/restart", response_model=RestartResponse)
async def restart_bot(
    request: Request,
    _user: str = Depends(get_current_user),
):
    """
    Restart the bot process.

    Spawns a detached PowerShell helper that (a) waits a moment so the
    HTTP response can return, (b) kills the current main.py + any stale
    autostart wrappers, (c) fires the CortexTradingBot scheduled task so
    the bot comes back detached with no console window.

    The helper runs with CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS so
    it survives after the current python.exe is killed.

    Windows-only. Requires the 'CortexTradingBot' scheduled task to be
    installed (scripts\\install_autostart.ps1).
    """
    import subprocess
    import sys

    if sys.platform != "win32":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Dashboard restart is only supported on Windows.",
        )

    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    ps_cmd = (
        "Start-Sleep -Seconds 2; "
        "Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | "
        "Where-Object { $_.CommandLine -match 'main\\.py|autostart_hidden|start_trading_bot|launch\\.ps1' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; "
        "Start-Sleep -Seconds 1; "
        "Remove-Item 'G:\\AI_Trading_Bot\\Cortex\\data\\state\\bot.pid' -Force -ErrorAction SilentlyContinue; "
        "Start-ScheduledTask -TaskName 'CortexTradingBot'"
    )

    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP — survives parent death
    DETACHED = 0x00000008
    NEW_GROUP = 0x00000200

    try:
        proc = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-WindowStyle", "Hidden",
                "-Command", ps_cmd,
            ],
            creationflags=DETACHED | NEW_GROUP,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="powershell.exe not found",
        )
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to spawn restart helper: {e}",
        )

    logger.warning(
        "Bot restart requested by '%s' via /api/system/restart (helper PID %d)",
        _user, proc.pid,
    )

    return RestartResponse(
        status="scheduled",
        message=(
            "Restart helper spawned. Bot will stop in ~2s and come back "
            "detached in ~4-6s. Refresh the dashboard after 10s."
        ),
        pid=proc.pid,
    )
