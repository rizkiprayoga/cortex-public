"""
live_state.py — Shared state container injected into FastAPI.

LiveState holds references to all in-memory objects that API handlers
need to read. Constructed in main.py after the trading loop is wired
up, then passed into ``build_app()``.

BotControl is a thin helper that lets the dashboard start / pause /
stop the trading loop. The trading loop reads ``bot_control.status``
each tick and skips signal processing when paused.
"""

import enum
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BotStatus(str, enum.Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class BotControl:
    """
    Thread-safe bot lifecycle controller.

    Semantics
    ---------
    RUNNING  — normal trading loop, all signals processed.
    PAUSED   — trading loop sleeps (no new signals/orders) but
               RiskMonitor stays alive guarding open positions.
    STOPPED  — loop halts; caller should fire EmergencyClose to
               flatten all positions.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = BotStatus.RUNNING
        self._last_change = datetime.now(tz=timezone.utc)
        self._changed_by: str = "system"

    @property
    def status(self) -> BotStatus:
        with self._lock:
            return self._status

    @property
    def last_change(self) -> datetime:
        with self._lock:
            return self._last_change

    @property
    def changed_by(self) -> str:
        with self._lock:
            return self._changed_by

    def start(self, by: str = "dashboard") -> None:
        with self._lock:
            if self._status == BotStatus.STOPPED:
                logger.warning(
                    "Cannot start from STOPPED state — restart the process"
                )
                return
            self._status = BotStatus.RUNNING
            self._last_change = datetime.now(tz=timezone.utc)
            self._changed_by = by
        logger.info("Bot control: RUNNING (by %s)", by)

    def pause(self, by: str = "dashboard") -> None:
        with self._lock:
            self._status = BotStatus.PAUSED
            self._last_change = datetime.now(tz=timezone.utc)
            self._changed_by = by
        logger.info("Bot control: PAUSED (by %s)", by)

    def stop(self, by: str = "dashboard") -> None:
        with self._lock:
            self._status = BotStatus.STOPPED
            self._last_change = datetime.now(tz=timezone.utc)
            self._changed_by = by
        logger.info("Bot control: STOPPED (by %s) — EmergencyClose expected", by)


DASHBOARD_LOCK_FILE = Path("data/state/DASHBOARD_LOCKED.flag")


class DashboardLock:
    """
    File-based lock gate for the public dashboard.

    The presence of ``data/state/DASHBOARD_LOCKED.flag`` is the source of
    truth: file exists -> dashboard locked; file absent -> dashboard
    unlocked. The bot writes the file at startup (locked by default) and
    re-creates it after ``idle_timeout_seconds`` of no authenticated
    requests.

    To unlock the dashboard, simply delete the file:

        rm data/state/DASHBOARD_LOCKED.flag

    Or via PowerShell:

        Remove-Item data\\state\\DASHBOARD_LOCKED.flag

    The legacy POST /api/system/unlock endpoint also deletes the file so
    existing automation keeps working.
    """

    def __init__(self, idle_timeout_seconds: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._idle_timeout = idle_timeout_seconds
        self._last_authed_request: Optional[datetime] = None
        DASHBOARD_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Default: unlocked. Opt-in locking via env var for remote access.
        if os.getenv("DASHBOARD_LOCK_ON_STARTUP", "").lower() in ("1", "true", "yes"):
            self._write_flag()
            logger.info("Dashboard LOCKED (DASHBOARD_LOCK_ON_STARTUP=true)")
        else:
            self._delete_flag()
            logger.info("Dashboard UNLOCKED (default — local access)")

    @property
    def is_locked(self) -> bool:
        with self._lock:
            self._check_idle_timeout()
            return DASHBOARD_LOCK_FILE.exists()

    def unlock(self) -> None:
        with self._lock:
            self._delete_flag()
            self._last_authed_request = datetime.now(tz=timezone.utc)
        logger.info("Dashboard UNLOCKED")

    def lock(self) -> None:
        with self._lock:
            self._write_flag()
        logger.info("Dashboard LOCKED")

    def touch(self) -> None:
        """Record an authenticated request to reset the idle timer."""
        with self._lock:
            self._last_authed_request = datetime.now(tz=timezone.utc)

    # -- internals --------------------------------------------------------

    def _write_flag(self) -> None:
        try:
            ts = datetime.now(tz=timezone.utc).isoformat()
            DASHBOARD_LOCK_FILE.write_text(
                f"Dashboard locked at {ts}\n"
                f"\n"
                f"Delete this file to unlock the dashboard.\n"
                f"  PowerShell:  Remove-Item data\\state\\DASHBOARD_LOCKED.flag\n"
                f"  Bash:        rm data/state/DASHBOARD_LOCKED.flag\n"
                f"\n"
                f"The bot will re-create this file:\n"
                f"  - At startup\n"
                f"  - After {int(self._idle_timeout / 60)} minutes of no authed activity\n",
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("DashboardLock could not write flag file: %s", exc)

    def _delete_flag(self) -> None:
        try:
            DASHBOARD_LOCK_FILE.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("DashboardLock could not delete flag file: %s", exc)

    def _check_idle_timeout(self) -> None:
        """Auto-lock (re-create flag) if no authed request for idle_timeout."""
        if self._idle_timeout <= 0:
            return  # idle re-lock disabled
        if self._last_authed_request is None:
            return
        if DASHBOARD_LOCK_FILE.exists():
            return  # already locked
        elapsed = (
            datetime.now(tz=timezone.utc) - self._last_authed_request
        ).total_seconds()
        if elapsed >= self._idle_timeout:
            # Flag absent + stale timestamp = external unlock (manual rm or
            # unlock shortcut, which bypass unlock()). Treat as a fresh
            # session so the operator gets a full idle window to sign in,
            # instead of being re-locked on the next poll.
            self._last_authed_request = datetime.now(tz=timezone.utc)
            logger.info(
                "Dashboard flag deleted externally after %.0f s idle — "
                "granting fresh unlock window",
                elapsed,
            )


@dataclass
class LiveState:
    """
    Read-only view of all shared trading state.

    Constructed in main.py and passed into ``build_app(live_state)``.
    API handlers read from these references — never import from main.
    """

    tracked_positions: dict[int, Any]
    combiner: Any  # SignalCombiner
    circuit_breaker: Any  # CircuitBreaker
    account_monitor: Any  # AccountMonitor
    risk_monitor: Any  # RiskMonitor
    order_manager: Any  # OrderManager
    orchestrator: Any  # StrategyOrchestrator
    portfolio: Any  # PortfolioManager
    data_store: Any  # DataStore
    config_store: Any = None  # ConfigStore (optional, for hot-reload persistence)
    bot_control: BotControl = field(default_factory=BotControl)
    dashboard_lock: DashboardLock = field(default_factory=DashboardLock)
    # Startup timestamp for uptime display
    started_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    # Current MT5 account login number — used to tag and filter data
    current_account_id: Optional[int] = None
    # Audit C5: threading lock protecting tracked_positions across
    # the async main loop, the RiskMonitor OS thread, and API handlers.
    # Required because dict mutations are not atomic across threads.
    positions_lock: threading.Lock = field(default_factory=threading.Lock)
    # Audit H8: lock protecting current_account_id reads/writes
    account_lock: threading.Lock = field(default_factory=threading.Lock)

    def get_account_id(self) -> Optional[int]:
        """
        Thread-safe read of current_account_id.

        Audit H8: all readers should use this method instead of direct
        attribute access to avoid TOCTOU races with the account switch
        API endpoint.
        """
        with self.account_lock:
            return self.current_account_id

    def set_account_id(self, account_id: Optional[int]) -> None:
        """Thread-safe write of current_account_id (Audit H8)."""
        with self.account_lock:
            self.current_account_id = account_id
