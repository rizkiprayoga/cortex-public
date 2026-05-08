"""
circuit_breaker.py — Multi-Level Trading Halt Mechanism

Monitors drawdown and halts / throttles trading when predefined loss
thresholds are breached. Operates INDEPENDENTLY of the AI Brain — does
NOT consult the HMM or LSTM before acting.

Five threshold levels
---------------------
    Daily soft   (default 2%)  → halve all new position sizes
    Daily hard   (default 3%)  → flat + halt for the rest of the UTC day
    Weekly soft  (default 5%)  → halve all new position sizes
    Weekly hard  (default 7%)  → flat + halt for the rest of the ISO week
    Peak sticky  (default 10%) → flat + halt until manual reset

The "flat" levels require the caller (RiskMonitor) to drive
EmergencyClose.close_all() when ``requires_flat()`` returns True.

Size multiplier semantics
-------------------------
``current_size_multiplier()`` returns the multiplier the next trade
should apply on top of whatever sizing the allocation layer computes:

    1.0  — no active breaker (or only informational level)
    0.5  — at least one soft breaker is active (halve)
    0.0  — at least one hard breaker is active (no new trades)

The worst-of rule means a daily-soft + weekly-hard combination gives
0.0. Soft breakers persist until their own reset point; hard breakers
hold any new entries until the same reset.

Reset semantics
---------------
    daily  breakers: reset at 00:00 UTC each day
    weekly breakers: reset at Monday 00:00 UTC each week
    peak   breaker:  sticky — only ``manual_reset()`` clears it

``check_and_update()`` is the canonical entry point called every
RiskMonitor tick. It takes the current equity, daily and weekly
starting equities, and the peak equity; it updates every breaker's
state in one pass and returns a snapshot.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HALT_FLAG_FILE = Path("data/logs/TRADING_HALTED.flag")


@dataclass
class BreakerSnapshot:
    """
    Immutable view of the circuit breaker state at one tick.

    The RiskMonitor uses this to decide whether to call
    EmergencyClose.close_all() and to write the audit log row.
    """

    multiplier: float                  # 1.0 / 0.5 / 0.0
    requires_flat: bool                # must close all positions now
    active_breakers: list[str]         # ["daily_hard", "peak_sticky", ...]
    daily_dd_pct: float
    weekly_dd_pct: float
    peak_dd_pct: float
    reason: str                        # single human-readable summary


class CircuitBreaker:
    """
    Thread-safe multi-level circuit breaker.

    Usage
    -----
        cb = CircuitBreaker()  # defaults from settings.yaml
        snap = cb.check_and_update(
            current_equity=9800.0,
            daily_start_equity=10000.0,
            weekly_start_equity=10000.0,
            peak_equity=10000.0,
        )
        if snap.requires_flat:
            emergency_close.close_all()
        lot_size *= snap.multiplier
    """

    def __init__(
        self,
        max_daily_loss_soft_pct: float = 3.0,
        max_daily_loss_hard_pct: float = 5.0,
        max_weekly_loss_soft_pct: float = 5.0,
        max_weekly_loss_hard_pct: float = 7.0,
        max_peak_drawdown_pct: float = 10.0,
        consecutive_loss_limit: int = 4,
        consecutive_halt_hours: int = 4,
    ):
        self.max_daily_loss_soft_pct = max_daily_loss_soft_pct
        self.max_daily_loss_hard_pct = max_daily_loss_hard_pct
        self.max_weekly_loss_soft_pct = max_weekly_loss_soft_pct
        self.max_weekly_loss_hard_pct = max_weekly_loss_hard_pct
        self.max_peak_drawdown_pct = max_peak_drawdown_pct
        self.consecutive_loss_limit = consecutive_loss_limit
        self.consecutive_halt_hours = consecutive_halt_hours

        self._lock = threading.Lock()

        # Active breaker state (flipped by check_and_update / reset)
        self._daily_soft = False
        self._daily_hard = False
        self._weekly_soft = False
        self._weekly_hard = False
        self._peak_sticky = False

        # Consecutive loss tracking
        self._consecutive_losses: int = 0
        self._consecutive_halt_until: Optional[datetime] = None

        # Reset anchors — updated whenever the corresponding period rolls
        self._daily_anchor: Optional[datetime] = None
        self._weekly_anchor: Optional[datetime] = None

        # Wave 6 fix #7: detect a persisted halt flag from a prior run.
        # If the file exists, a previous bot run tripped the peak breaker
        # and was halted until manual reset. We must resume in the halted
        # state so an operator restart doesn't silently re-enable trading.
        # Only peak persists — daily/weekly breakers reset on their own
        # cadence and are not sticky across restart by design.
        if HALT_FLAG_FILE.exists() and self._verify_halt_flag(HALT_FLAG_FILE):
            self._peak_sticky = True
            logger.warning(
                "CircuitBreaker: resuming in HALTED state — peak breaker "
                "persisted via %s. Call manual_reset() after investigation "
                "to clear and re-enable trading.",
                HALT_FLAG_FILE,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_update(
        self,
        current_equity: float,
        daily_start_equity: float,
        weekly_start_equity: float,
        peak_equity: float,
        now: Optional[datetime] = None,
    ) -> BreakerSnapshot:
        """
        Evaluate every breaker against the current equity and return a
        snapshot describing the worst-active state.

        Args:
            current_equity:      Current account equity
            daily_start_equity:  Equity at 00:00 UTC today
            weekly_start_equity: Equity at Monday 00:00 UTC
            peak_equity:         All-time peak equity since launch
            now:                 Override for tests (defaults to utcnow)
        """
        now = now or datetime.now(tz=timezone.utc)

        with self._lock:
            self._maybe_reset_periods(now)

            daily_dd = self._drawdown_pct(daily_start_equity, current_equity)
            weekly_dd = self._drawdown_pct(weekly_start_equity, current_equity)
            peak_dd = self._drawdown_pct(peak_equity, current_equity)

            # Breakers latch on — they stay set until the corresponding
            # reset point, even if price recovers intra-period.
            if daily_dd >= self.max_daily_loss_soft_pct:
                self._daily_soft = True
            if daily_dd >= self.max_daily_loss_hard_pct:
                self._daily_hard = True

            if weekly_dd >= self.max_weekly_loss_soft_pct:
                self._weekly_soft = True
            if weekly_dd >= self.max_weekly_loss_hard_pct:
                self._weekly_hard = True

            if peak_dd >= self.max_peak_drawdown_pct:
                # Wave 6 fix #7: first-time transition writes the halt
                # flag file so a restart detects the peak state. The flag
                # lives under data/logs/ alongside the audit trail and
                # survives crashes / Ctrl+C / reboots. Only manual_reset()
                # clears it.
                if not self._peak_sticky:
                    self._peak_sticky = True
                    self._write_halt_flag()

            # Audit H10: expire consecutive-loss halt explicitly here
            # rather than as a side-effect inside _multiplier().
            self._expire_consecutive_halt()

            return self._snapshot(daily_dd, weekly_dd, peak_dd)

    def _expire_consecutive_halt(self) -> None:
        """
        Clear consecutive-loss halt if the expiry time has passed.

        Audit H10: extracted from _multiplier() so expiry logic is
        explicit and _multiplier() is side-effect-free.
        """
        if self._consecutive_halt_until is None:
            return
        now = datetime.now(timezone.utc)
        if now >= self._consecutive_halt_until:
            self._consecutive_halt_until = None
            self._consecutive_losses = 0

    def current_size_multiplier(self) -> float:
        """
        Return the size multiplier without updating any state.

        Worst-of rule:
            any hard breaker → 0.0
            any soft breaker → 0.5
            otherwise         → 1.0
        """
        with self._lock:
            return self._multiplier()

    def is_halted(self) -> bool:
        """
        True if any hard breaker is active (no new trades allowed).

        Audit H10: includes consecutive-loss halt if within window.
        Callers may need to call _expire_consecutive_halt() first if
        they want an up-to-date result.
        """
        with self._lock:
            if self._daily_hard or self._weekly_hard or self._peak_sticky:
                return True
            if self._consecutive_halt_until is not None:
                now = datetime.now(timezone.utc)
                if now < self._consecutive_halt_until:
                    return True
            return False

    def requires_flat(self) -> bool:
        """
        True if RiskMonitor should call EmergencyClose now.

        Same condition as is_halted — hard breakers demand flat state.
        """
        return self.is_halted()

    def active_breakers(self) -> list[str]:
        """Return the list of active breaker names (for audit logging)."""
        with self._lock:
            return self._active_breaker_names()

    def consecutive_losses(self) -> int:
        """Snapshot the current consecutive-SL count (for UI rail gauge)."""
        with self._lock:
            return self._consecutive_losses

    def manual_reset(self) -> None:
        """
        Operator-initiated clear of ALL active breakers, including peak.

        Used after investigation of a major drawdown event. Does not
        affect the reset anchors — those roll forward normally.
        """
        with self._lock:
            self._daily_soft = False
            self._daily_hard = False
            self._weekly_soft = False
            self._weekly_hard = False
            self._peak_sticky = False
            try:
                # Audit M7: atomic delete — no TOCTOU check
                HALT_FLAG_FILE.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not delete halt flag file: %s", exc)
            self._consecutive_losses = 0
            self._consecutive_halt_until = None
            logger.warning("CircuitBreaker manually reset by operator")

    def record_trade_result(self, is_loss: bool) -> None:
        """
        Track consecutive losses for the consecutive-loss breaker.

        Call after each trade closes. When ``consecutive_loss_limit``
        consecutive SL hits occur, trading pauses for
        ``consecutive_halt_hours`` hours.
        """
        with self._lock:
            if is_loss:
                self._consecutive_losses += 1
                if self._consecutive_losses >= self.consecutive_loss_limit:
                    from datetime import timedelta
                    now = datetime.now(timezone.utc)
                    self._consecutive_halt_until = now + timedelta(
                        hours=self.consecutive_halt_hours,
                    )
                    logger.warning(
                        "CircuitBreaker: %d consecutive losses — "
                        "halting new trades until %s",
                        self._consecutive_losses,
                        self._consecutive_halt_until.isoformat(),
                    )
            else:
                self._consecutive_losses = 0

    @staticmethod
    def get_dd_scaler(peak_dd_pct: float) -> float:
        """
        Continuous drawdown-aware size scaler.

        Returns a multiplier [0.0, 1.0] that gradually reduces
        position sizes as drawdown deepens, instead of the binary
        soft/hard threshold jumps.

            DD < 5%  → 1.0 (full size)
            DD < 8%  → 0.75
            DD < 10% → 0.50
            DD >= 10% → 0.0 (peak breaker)
        """
        if peak_dd_pct < 5.0:
            return 1.0
        if peak_dd_pct < 8.0:
            return 0.75
        if peak_dd_pct < 10.0:
            return 0.50
        return 0.0

    # ------------------------------------------------------------------
    # Hot-reload setters (Phase 10.2)
    # ------------------------------------------------------------------

    def set_daily_soft(self, pct: float) -> None:
        """Update daily soft breaker threshold. Thread-safe."""
        if not (0.0 < pct < 100.0):
            raise ValueError(f"daily_soft must be 0 < pct < 100, got {pct}")
        with self._lock:
            old = self.max_daily_loss_soft_pct
            self.max_daily_loss_soft_pct = pct
        logger.info("CircuitBreaker: daily_soft %.2f%% -> %.2f%%", old, pct)

    def set_daily_hard(self, pct: float) -> None:
        """Update daily hard breaker threshold. Thread-safe."""
        if not (0.0 < pct < 100.0):
            raise ValueError(f"daily_hard must be 0 < pct < 100, got {pct}")
        with self._lock:
            old = self.max_daily_loss_hard_pct
            self.max_daily_loss_hard_pct = pct
        logger.info("CircuitBreaker: daily_hard %.2f%% -> %.2f%%", old, pct)

    def set_weekly_soft(self, pct: float) -> None:
        """Update weekly soft breaker threshold. Thread-safe."""
        if not (0.0 < pct < 100.0):
            raise ValueError(f"weekly_soft must be 0 < pct < 100, got {pct}")
        with self._lock:
            old = self.max_weekly_loss_soft_pct
            self.max_weekly_loss_soft_pct = pct
        logger.info("CircuitBreaker: weekly_soft %.2f%% -> %.2f%%", old, pct)

    def set_weekly_hard(self, pct: float) -> None:
        """Update weekly hard breaker threshold. Thread-safe."""
        if not (0.0 < pct < 100.0):
            raise ValueError(f"weekly_hard must be 0 < pct < 100, got {pct}")
        with self._lock:
            old = self.max_weekly_loss_hard_pct
            self.max_weekly_loss_hard_pct = pct
        logger.info("CircuitBreaker: weekly_hard %.2f%% -> %.2f%%", old, pct)

    def set_peak(self, pct: float) -> None:
        """Update peak drawdown threshold. Thread-safe."""
        if not (0.0 < pct < 100.0):
            raise ValueError(f"peak must be 0 < pct < 100, got {pct}")
        with self._lock:
            old = self.max_peak_drawdown_pct
            self.max_peak_drawdown_pct = pct
        logger.info("CircuitBreaker: peak %.2f%% -> %.2f%%", old, pct)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _drawdown_pct(reference: float, current: float) -> float:
        """Return drawdown as a positive percentage (0 if in profit)."""
        if reference <= 0.0:
            return 0.0
        dd = (reference - current) / reference * 100.0
        return max(0.0, dd)

    @staticmethod
    def _get_halt_hmac_key() -> bytes:
        """
        Return the HMAC key used to sign the halt flag file.

        Audit H4: reuses DASHBOARD_JWT_SECRET as the key. If not set,
        falls back to a weak default (signature verification will still
        prevent casual tampering but is not cryptographically strong).
        """
        import os
        key = os.environ.get("DASHBOARD_JWT_SECRET", "").encode("utf-8")
        if not key:
            # Weak fallback — warn but don't crash. Flag file still
            # provides some protection against accidental tampering.
            key = b"cortex-halt-flag-fallback-key-change-me"
        return key

    @classmethod
    def _halt_flag_payload(cls, timestamp_iso: str) -> str:
        """Produce a signed halt-flag payload: "<iso>|<hmac_hex>"."""
        import hmac as _hmac
        import hashlib
        key = cls._get_halt_hmac_key()
        sig = _hmac.new(
            key, timestamp_iso.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return f"{timestamp_iso}|{sig}"

    @classmethod
    def _verify_halt_flag(cls, path: Path) -> bool:
        """
        Verify the halt flag's HMAC signature. Returns True if valid
        (trading should remain halted) or the file is legacy/empty
        (treat as valid for backward compatibility with existing flags).
        """
        import hmac as _hmac
        import hashlib
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            return True  # Can't read — safe default: stay halted

        # Backward compat: empty files are legacy (pre-HMAC) halt flags
        if not content:
            logger.warning(
                "CircuitBreaker: halt flag is unsigned (legacy format). "
                "Honoring it anyway — safe default."
            )
            return True

        if "|" not in content:
            logger.critical(
                "CircuitBreaker: halt flag file %s has invalid format — "
                "treating as HALTED (fail-safe)",
                path,
            )
            return True

        timestamp_iso, sig = content.rsplit("|", 1)
        key = cls._get_halt_hmac_key()
        expected_sig = _hmac.new(
            key, timestamp_iso.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if _hmac.compare_digest(sig, expected_sig):
            return True
        logger.critical(
            "CircuitBreaker: halt flag HMAC mismatch — possible tampering! "
            "Treating as HALTED anyway (fail-safe)."
        )
        return True  # Fail safe: always honor the flag

    @classmethod
    def _write_halt_flag(cls) -> None:
        """
        Create the sentinel halt flag file so a future instance starts
        up in HALTED state even if an operator restarts the process
        without calling manual_reset().

        Audit H4: flag is HMAC-signed using DASHBOARD_JWT_SECRET so
        external tampering is detectable (though we fail safe — any
        anomaly keeps trading halted).
        """
        try:
            HALT_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
            from datetime import datetime, timezone
            ts = datetime.now(tz=timezone.utc).isoformat()
            payload = cls._halt_flag_payload(ts)
            HALT_FLAG_FILE.write_text(payload, encoding="utf-8")
            logger.critical(
                "CircuitBreaker: PEAK DRAWDOWN BREACHED — halt flag "
                "written to %s. Trading halted until manual_reset().",
                HALT_FLAG_FILE,
            )
        except OSError as exc:
            logger.error(
                "CircuitBreaker: failed to write halt flag %s: %s",
                HALT_FLAG_FILE,
                exc,
            )

    def _multiplier(self) -> float:
        """
        Pure function — no state mutation. Use _expire_consecutive_halt()
        separately (called from check_and_update) to clear expired halts.
        """
        if self._daily_hard or self._weekly_hard or self._peak_sticky:
            return 0.0
        # Consecutive loss halt — read-only check
        if self._consecutive_halt_until is not None:
            now = datetime.now(timezone.utc)
            if now < self._consecutive_halt_until:
                return 0.0
        if self._daily_soft or self._weekly_soft:
            return 0.5
        return 1.0

    def _active_breaker_names(self) -> list[str]:
        names: list[str] = []
        if self._daily_soft:
            names.append("daily_soft")
        if self._daily_hard:
            names.append("daily_hard")
        if self._weekly_soft:
            names.append("weekly_soft")
        if self._weekly_hard:
            names.append("weekly_hard")
        if self._peak_sticky:
            names.append("peak_sticky")
        return names

    def _snapshot(
        self,
        daily_dd: float,
        weekly_dd: float,
        peak_dd: float,
    ) -> BreakerSnapshot:
        multiplier = self._multiplier()
        active = self._active_breaker_names()
        flat = multiplier == 0.0
        if not active:
            reason = "no active breakers"
        else:
            reason = (
                f"active={','.join(active)} "
                f"daily_dd={daily_dd:.2f}% weekly_dd={weekly_dd:.2f}% "
                f"peak_dd={peak_dd:.2f}%"
            )
        snap = BreakerSnapshot(
            multiplier=multiplier,
            requires_flat=flat,
            active_breakers=list(active),
            daily_dd_pct=daily_dd,
            weekly_dd_pct=weekly_dd,
            peak_dd_pct=peak_dd,
            reason=reason,
        )
        # Cache for dashboard read-only access via get_last_snapshot()
        self._last_snapshot = snap
        return snap

    def get_last_snapshot(self) -> Optional[BreakerSnapshot]:
        """
        Return the most recent BreakerSnapshot computed by
        check_and_update (called by RiskMonitor every 30s), or None if
        the monitor hasn't run a cycle yet. Read-only — intended for
        the dashboard API so it doesn't have to recompute DD itself.
        """
        return getattr(self, "_last_snapshot", None)

    def _maybe_reset_periods(self, now: datetime) -> None:
        """
        Clear daily breakers at UTC midnight and weekly breakers at
        Monday 00:00 UTC. Called inside the lock from check_and_update.
        """
        # Daily anchor — reset whenever the UTC date changes
        today = now.date()
        if self._daily_anchor is None or self._daily_anchor.date() < today:
            if self._daily_anchor is not None and (self._daily_soft or self._daily_hard):
                logger.info(
                    "Daily breakers resetting at UTC midnight (was soft=%s hard=%s)",
                    self._daily_soft,
                    self._daily_hard,
                )
            self._daily_soft = False
            self._daily_hard = False
            self._daily_anchor = datetime(
                year=now.year, month=now.month, day=now.day, tzinfo=timezone.utc
            )

        # Weekly anchor — reset at Monday 00:00 UTC
        # ISO weekday: Monday = 1 .. Sunday = 7
        # Compute the Monday of *this* ISO week
        days_since_monday = now.isoweekday() - 1
        monday_date = today.fromordinal(today.toordinal() - days_since_monday)
        monday_anchor = datetime(
            year=monday_date.year,
            month=monday_date.month,
            day=monday_date.day,
            tzinfo=timezone.utc,
        )
        if self._weekly_anchor is None or self._weekly_anchor < monday_anchor:
            if self._weekly_anchor is not None and (
                self._weekly_soft or self._weekly_hard
            ):
                logger.info(
                    "Weekly breakers resetting at Monday UTC (was soft=%s hard=%s)",
                    self._weekly_soft,
                    self._weekly_hard,
                )
            self._weekly_soft = False
            self._weekly_hard = False
            self._weekly_anchor = monday_anchor
