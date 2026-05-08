"""
risk_monitor.py — Real-Time Risk Monitoring Loop

Runs as an independent background thread that polls the account every
``check_interval_seconds`` (default 30 s). On each tick it:

    1. Reads the current account snapshot via AccountMonitor
    2. Updates the all-time peak equity and the daily/weekly anchors
    3. Passes the snapshot through ``CircuitBreaker.check_and_update()``
    4. If the resulting BreakerSnapshot ``requires_flat``, calls
       ``EmergencyClose.close_all()`` and writes an audit row that
       includes the active HMM regime and LSTM prediction at the time
       of the trip (read from a shared last-signal reference)

The monitor is INDEPENDENT of the Brain — it runs regardless of whether
HMM/LSTM are producing signals. The Brain's latest SignalResult is
passed in through ``attach_signal_ref()`` and read (not written) each
tick for audit enrichment only — the monitor never asks the Brain for
permission to act.

Daily and weekly anchors are set at startup from the current equity
and rolled forward at the UTC midnight / Monday-midnight boundaries
so drawdown numbers reset with the breakers.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

from src.broker.account_monitor import AccountMonitor, AccountSnapshot
from src.broker.mt5_connector import MT5Connector
from src.safety.circuit_breaker import BreakerSnapshot, CircuitBreaker

if TYPE_CHECKING:
    from src.brain.signal_combiner import SignalCombiner

logger = logging.getLogger(__name__)


class RiskMonitor:
    """
    Background thread that polls account state and drives the breaker.

    Usage
    -----
        monitor = RiskMonitor(
            connector,
            circuit_breaker,
            check_interval_seconds=30,
        )
        monitor.attach_signal_ref(lambda: combiner.last_signal)
        monitor.start()
        ...
        monitor.stop()
    """

    def __init__(
        self,
        connector: MT5Connector,
        circuit_breaker: CircuitBreaker,
        check_interval_seconds: float = 30.0,
        min_margin_level: float = 200.0,
        emergency_close=None,
        max_consecutive_read_failures: int = 5,
        alert_manager=None,
    ):
        self.connector = connector
        self.circuit_breaker = circuit_breaker
        self.check_interval = check_interval_seconds
        self.min_margin_level = min_margin_level
        self.max_consecutive_read_failures = max_consecutive_read_failures
        self._account_monitor = AccountMonitor(connector)
        self._alert_manager = alert_manager

        # Lazy: defer construction so we don't hit MT5 at import time.
        self._emergency_close = emergency_close

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._peak_equity: float = 0.0
        self._daily_start_equity: float = 0.0
        self._weekly_start_equity: float = 0.0
        self._daily_anchor_date = None
        self._weekly_anchor_date = None

        # Tracks consecutive account-read failures. If we can't read the
        # account, we can't know whether breakers *should* have tripped.
        # After max_consecutive_read_failures we reconnect, and if that
        # fails we fail *closed* (flatten everything) as a safety measure.
        self._consecutive_read_failures: int = 0

        # Optional signal reference getter — RiskMonitor reads the
        # latest SignalResult for audit enrichment only.
        self._signal_ref: Optional[Callable[[], object]] = None

        # Wave 6 fix #24: RiskMonitor holds a reference to the main
        # loop's tracked-positions dict so that after EmergencyClose
        # fires we can atomically clear the main loop's view of "still
        # open" positions. Without this the exit ladder on the next
        # tick would try to process ghost tickets whose broker
        # counterparts just got flattened. Main.py calls
        # ``set_position_tracker`` at startup — the ref stays None
        # until then, and ``get_positions_snapshot`` is still safe.
        self._tracked_positions_ref: Optional[dict[int, Any]] = None
        # Audit C5: threading lock wrapping tracked_positions mutations.
        # The RiskMonitor runs in a daemon OS thread while the main loop
        # mutates tracked_positions from async context — without this
        # lock, dict iteration/mutation can race across threads.
        self._positions_lock: Optional[Any] = None
        # Wave 6 fix #24: SignalCombiner reference so that
        # ``_handle_flat_required`` can fire ``combiner.reset_state()``
        # after EmergencyClose. Flushes the 4-bar flickering ring so
        # the first post-halt bar cannot inherit a pre-halt direction
        # whose underlying regime changed during the halt.
        self._signal_combiner: Optional["SignalCombiner"] = None
        # Audit C7: reference to the main event loop so we can schedule
        # combiner.reset_state() via call_soon_threadsafe instead of
        # calling it directly from this OS thread.
        self._main_event_loop: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach_signal_ref(self, getter: Callable[[], object]) -> None:
        """
        Hand the monitor a zero-arg getter that returns the latest
        SignalResult (or None). Used to enrich halt audit logs.
        """
        self._signal_ref = getter

    def set_signal_combiner(self, combiner: "SignalCombiner") -> None:
        """
        Wave 6 fix #24: wire the SignalCombiner so RiskMonitor can call
        ``combiner.reset_state()`` after EmergencyClose fires. Flushes
        the 4-bar flickering ring so the first post-halt bar cannot
        inherit a pre-halt direction whose underlying regime changed
        during the halt window.
        """
        self._signal_combiner = combiner

    def set_position_tracker(
        self, tracked_positions: dict[int, Any],
        positions_lock: Optional[Any] = None,
    ) -> None:
        """
        Hand RiskMonitor a reference to the main loop's tracked-positions
        dict plus an optional threading lock (Audit C5) that guards all
        mutations. Main loop must use the same lock when iterating the
        dict to prevent cross-thread races.
        """
        self._tracked_positions_ref = tracked_positions
        self._positions_lock = positions_lock

    def set_main_event_loop(self, loop: Any) -> None:
        """
        Audit C7: register the main asyncio event loop so RiskMonitor
        (running in an OS thread) can schedule combiner.reset_state()
        via loop.call_soon_threadsafe() — keeps combiner state
        mutations on a single thread.
        """
        self._main_event_loop = loop

    def get_positions_snapshot(self) -> dict[int, Any]:
        """
        Wave 6 fix #24: return a defensive shallow copy of the main
        loop's tracked-positions dict. Callers can iterate or mutate
        the returned dict freely without affecting RiskMonitor's
        internal reference. Returns an empty dict if
        ``set_position_tracker`` has not been called yet.
        """
        if self._tracked_positions_ref is None:
            return {}
        return dict(self._tracked_positions_ref)

    def get_peak_equity(self) -> float:
        """
        Return the peak equity observed by the monitor loop since start.

        Wave 6 fix #20: the main trading loop reads this to feed the
        drawdown-aware allocation clamp in ``StrategyOrchestrator``.
        Read-only; thread-safe because float reads in CPython are atomic
        and the worst case on a concurrent write is that we see last
        tick's value, which only affects whether the clamp triggers one
        tick earlier or later — not a safety-relevant difference.
        """
        return self._peak_equity

    def start(self) -> None:
        """Start the background monitoring loop in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop, name="RiskMonitor", daemon=True
        )
        self._thread.start()
        logger.info(
            "RiskMonitor started (interval=%.1fs, min_margin_level=%.1f%%)",
            self.check_interval,
            self.min_margin_level,
        )

    def stop(self) -> None:
        """Signal the monitoring thread to stop cleanly."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.check_interval, 5.0))
            self._thread = None
        logger.info("RiskMonitor stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """
        Poll loop — one iteration every ``check_interval`` seconds.
        """
        try:
            first = self._account_monitor.get_info()
            self._initialize_anchors(first)
        except Exception as exc:
            logger.error("RiskMonitor could not read initial account info: %s", exc)
            return

        while not self._stop_event.is_set():
            try:
                snapshot = self._account_monitor.get_info()
                # Successful read → reset failure counter.
                self._consecutive_read_failures = 0
                self._update_anchors(snapshot)

                breaker_snap = self.circuit_breaker.check_and_update(
                    current_equity=snapshot.equity,
                    daily_start_equity=self._daily_start_equity,
                    weekly_start_equity=self._weekly_start_equity,
                    peak_equity=self._peak_equity,
                )

                self._warn_low_margin_level(snapshot)

                if breaker_snap.requires_flat:
                    self._handle_flat_required(snapshot, breaker_snap)
            except Exception as exc:
                self._consecutive_read_failures += 1
                logger.error(
                    "RiskMonitor tick failed (%d/%d consecutive): %s",
                    self._consecutive_read_failures,
                    self.max_consecutive_read_failures,
                    exc,
                )
                if self._consecutive_read_failures >= self.max_consecutive_read_failures:
                    self._handle_read_failure_escalation()

            self._stop_event.wait(self.check_interval)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _initialize_anchors(self, snapshot: AccountSnapshot) -> None:
        now = datetime.now(tz=timezone.utc)
        self._peak_equity = max(self._peak_equity, snapshot.equity)
        self._daily_start_equity = snapshot.equity
        self._weekly_start_equity = snapshot.equity
        self._daily_anchor_date = now.date()
        self._weekly_anchor_date = self._iso_monday(now)

    def _update_anchors(self, snapshot: AccountSnapshot) -> None:
        now = datetime.now(tz=timezone.utc)
        self._peak_equity = max(self._peak_equity, snapshot.equity)
        if self._daily_anchor_date is None or now.date() > self._daily_anchor_date:
            self._daily_start_equity = snapshot.equity
            self._daily_anchor_date = now.date()
        monday = self._iso_monday(now)
        if self._weekly_anchor_date is None or monday > self._weekly_anchor_date:
            self._weekly_start_equity = snapshot.equity
            self._weekly_anchor_date = monday

    @staticmethod
    def _iso_monday(now: datetime):
        days_since_monday = now.isoweekday() - 1
        return now.date().fromordinal(now.date().toordinal() - days_since_monday)

    def _warn_low_margin_level(self, snapshot: AccountSnapshot) -> None:
        if snapshot.margin <= 0:
            return
        if snapshot.margin_level < self.min_margin_level:
            logger.warning(
                "Margin level %.1f%% below threshold %.1f%% "
                "(equity=%.2f margin=%.2f)",
                snapshot.margin_level,
                self.min_margin_level,
                snapshot.equity,
                snapshot.margin,
            )

    def _handle_flat_required(
        self,
        snapshot: AccountSnapshot,
        breaker_snap: BreakerSnapshot,
    ) -> None:
        """
        Breakers demand we close everything. Fetch the current signal
        context (best-effort) and write the full audit row before
        firing EmergencyClose.
        """
        signal = None
        if self._signal_ref is not None:
            try:
                signal = self._signal_ref()
            except Exception as exc:  # pragma: no cover
                logger.warning("signal_ref() raised: %s", exc)

        regime_label = getattr(
            getattr(signal, "regime", None), "regime_label", "unknown"
        )
        lstm_pred = getattr(signal, "lstm_prediction", None)

        logger.critical(
            "CIRCUIT BREAKER TRIPPED active=%s daily_dd=%.2f%% "
            "weekly_dd=%.2f%% peak_dd=%.2f%% equity=%.2f "
            "regime=%s lstm_pred=%s — firing EmergencyClose",
            ",".join(breaker_snap.active_breakers) or "none",
            breaker_snap.daily_dd_pct,
            breaker_snap.weekly_dd_pct,
            breaker_snap.peak_dd_pct,
            snapshot.equity,
            regime_label,
            lstm_pred,
        )

        # Alert: circuit breaker trip
        if self._alert_manager is not None:
            try:
                self._alert_manager.notify_breaker_trip(
                    active_breakers=breaker_snap.active_breakers,
                    daily_dd_pct=breaker_snap.daily_dd_pct,
                    weekly_dd_pct=breaker_snap.weekly_dd_pct,
                    peak_dd_pct=breaker_snap.peak_dd_pct,
                    equity=snapshot.equity,
                    requires_flat=breaker_snap.requires_flat,
                )
            except Exception as exc:
                logger.warning("Alert dispatch failed: %s", exc)

        ec = self._resolve_emergency_close()
        if ec is None:
            return
        result = ec.close_all()
        logger.critical(
            "EmergencyClose result closed=%s failed=%s",
            result.get("closed", []),
            result.get("failed", []),
        )

        # Alert: emergency close result
        if self._alert_manager is not None:
            try:
                self._alert_manager.notify_emergency_close(
                    closed_tickets=result.get("closed", []),
                    failed_tickets=result.get("failed", []),
                )
            except Exception as exc:
                logger.warning("Alert dispatch failed: %s", exc)

        self._post_halt_cleanup()

    def _post_halt_cleanup(self) -> None:
        """
        Wave 6 fix #24: after EmergencyClose flattens everything, the
        main loop's tracked-positions view is stale — every ticket it
        knows about was just closed at the broker. Clear the shared
        dict so the next tick's exit ladder does not try to work on
        ghost tickets. Also flush the SignalCombiner's 4-bar flickering
        ring so the first post-halt bar cannot inherit a pre-halt
        direction whose underlying regime may have changed during the
        halt window.
        """
        # Audit C5: lock-protected dict mutation — prevents race with
        # main loop iterating tracked_positions.values() in exit manager.
        if self._tracked_positions_ref is not None:
            if self._positions_lock is not None:
                with self._positions_lock:
                    cleared = len(self._tracked_positions_ref)
                    self._tracked_positions_ref.clear()
            else:
                cleared = len(self._tracked_positions_ref)
                self._tracked_positions_ref.clear()
            logger.critical(
                "Post-halt cleanup: cleared %d tracked positions", cleared
            )
        # Audit C7: schedule combiner.reset_state() on the main event
        # loop instead of calling directly from this OS thread. The
        # combiner's internal dicts (_recent_dirs, _last_signal_bar)
        # are mutated from async context; cross-thread clears can
        # leave the combiner in a partial state.
        if self._signal_combiner is not None:
            try:
                if self._main_event_loop is not None:
                    self._main_event_loop.call_soon_threadsafe(
                        self._signal_combiner.reset_state
                    )
                    logger.critical(
                        "Post-halt cleanup: scheduled SignalCombiner.reset_state()"
                    )
                else:
                    # Fallback — risky but better than skipping the reset
                    self._signal_combiner.reset_state()
                    logger.critical(
                        "Post-halt cleanup: SignalCombiner.reset_state() fired "
                        "directly (no event loop ref)"
                    )
            except Exception as exc:  # pragma: no cover
                logger.warning("combiner.reset_state() raised: %s", exc)

    def _handle_read_failure_escalation(self) -> None:
        """
        After N consecutive account-read failures: attempt to reconnect
        the MT5 terminal. If that also fails, fire EmergencyClose with a
        synthetic ``risk_monitor_offline`` breaker so positions are flat
        rather than running blind under a potentially drawn-down account.

        The counter is reset when escalation completes, regardless of
        outcome, so subsequent ticks re-evaluate from a clean slate.
        """
        n = self._consecutive_read_failures
        logger.critical(
            "RiskMonitor cannot read account for %d consecutive ticks — "
            "attempting MT5 reconnect",
            n,
        )
        reconnect_ok = False
        try:
            reconnect_ok = bool(self.connector.reconnect())
        except Exception as exc:
            logger.error("RiskMonitor reconnect raised: %s", exc)
            reconnect_ok = False

        if reconnect_ok:
            logger.warning(
                "RiskMonitor reconnected after %d failed reads — resuming normal polling",
                n,
            )
            self._consecutive_read_failures = 0
            return

        logger.critical(
            "RiskMonitor giving up after %d failed reads + reconnect failure — "
            "firing EmergencyClose as a safety measure",
            n,
        )
        # Synthesize a BreakerSnapshot so _handle_flat_required's logging and
        # close_all() path is reused verbatim. Equity/DD fields are unknown
        # because we couldn't read the account — write NaN-safe sentinels.
        synthetic_snap = BreakerSnapshot(
            multiplier=0.0,
            requires_flat=True,
            active_breakers=["risk_monitor_offline"],
            daily_dd_pct=float("nan"),
            weekly_dd_pct=float("nan"),
            peak_dd_pct=float("nan"),
            reason="risk_monitor_offline: account read failed repeatedly",
        )
        # We don't have a real AccountSnapshot, so call EmergencyClose
        # directly and log our own row. _handle_flat_required would crash
        # trying to read snapshot.equity.
        logger.critical(
            "CIRCUIT BREAKER SYNTHETIC TRIP active=risk_monitor_offline "
            "reason=%s — firing EmergencyClose",
            synthetic_snap.reason,
        )
        ec = self._resolve_emergency_close()
        if ec is not None:
            try:
                result = ec.close_all()
                logger.critical(
                    "EmergencyClose result closed=%s failed=%s",
                    result.get("closed", []),
                    result.get("failed", []),
                )
                self._post_halt_cleanup()
            except Exception as exc:
                logger.error("EmergencyClose.close_all() raised: %s", exc)
        # Reset the counter so we don't re-escalate every tick.
        self._consecutive_read_failures = 0

    def _resolve_emergency_close(self):
        if self._emergency_close is not None:
            return self._emergency_close
        try:
            from src.safety.emergency_close import EmergencyClose
            self._emergency_close = EmergencyClose(self.connector)
            return self._emergency_close
        except Exception as exc:  # pragma: no cover
            logger.error("Could not construct EmergencyClose: %s", exc)
            return None
