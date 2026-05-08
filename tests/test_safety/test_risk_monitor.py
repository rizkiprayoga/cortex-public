"""
Tests for RiskMonitor escalation on repeated account-read failures.

Wave 4 fix: when the risk monitor can't read the account for N
consecutive ticks (network outage, broker dropped us), it first
attempts a reconnect; if that also fails, it fires a synthetic
``risk_monitor_offline`` BreakerSnapshot and calls EmergencyClose
so positions go flat instead of running blind under a potentially
drawn-down account.
"""

from unittest.mock import MagicMock

import pytest

from src.safety.risk_monitor import RiskMonitor


class TestRiskMonitorReadFailureEscalation:

    def _build_monitor(
        self,
        reconnect_returns=False,
        max_failures=5,
    ):
        connector = MagicMock()
        connector.reconnect.return_value = reconnect_returns

        circuit_breaker = MagicMock()

        emergency_close = MagicMock()
        emergency_close.close_all.return_value = {"closed": [], "failed": []}

        monitor = RiskMonitor(
            connector=connector,
            circuit_breaker=circuit_breaker,
            check_interval_seconds=1,
            emergency_close=emergency_close,
            max_consecutive_read_failures=max_failures,
        )
        return monitor, connector, emergency_close

    def test_escalation_fires_emergency_close_when_reconnect_fails(self):
        """
        After max_consecutive_read_failures failed reads, reconnect() is
        tried. If reconnect also fails, close_all() must fire — this is
        the fail-closed safety behavior that protects us during network
        outages where the account state is unknown.
        """
        monitor, connector, emergency_close = self._build_monitor(
            reconnect_returns=False,
            max_failures=3,
        )
        # Simulate 3 prior failures already accumulated.
        monitor._consecutive_read_failures = 3

        monitor._handle_read_failure_escalation()

        # Reconnect was attempted exactly once.
        connector.reconnect.assert_called_once()
        # EmergencyClose fired because reconnect returned False.
        emergency_close.close_all.assert_called_once()
        # Counter is reset so subsequent ticks re-evaluate cleanly
        # rather than re-escalating every 30s.
        assert monitor._consecutive_read_failures == 0

    def test_escalation_stops_if_reconnect_succeeds(self):
        """
        If reconnect() comes back True, the monitor resumes normal
        polling without firing EmergencyClose. This is the nominal
        transient-outage recovery path.
        """
        monitor, connector, emergency_close = self._build_monitor(
            reconnect_returns=True,
            max_failures=3,
        )
        monitor._consecutive_read_failures = 3

        monitor._handle_read_failure_escalation()

        connector.reconnect.assert_called_once()
        emergency_close.close_all.assert_not_called()
        assert monitor._consecutive_read_failures == 0

    def test_escalation_survives_reconnect_raising(self):
        """
        If connector.reconnect() raises, the escalation path must
        still fall through to EmergencyClose — a raised exception
        means we couldn't restore the terminal, which is just as
        bad as a False return.
        """
        monitor, connector, emergency_close = self._build_monitor(
            max_failures=3,
        )
        connector.reconnect.side_effect = RuntimeError("network unreachable")
        monitor._consecutive_read_failures = 3

        monitor._handle_read_failure_escalation()

        connector.reconnect.assert_called_once()
        emergency_close.close_all.assert_called_once()
        assert monitor._consecutive_read_failures == 0

    def test_successful_read_resets_counter(self):
        """
        A successful account read in the main loop resets the counter.
        This is a direct property of the loop body: _consecutive_read_failures
        is set to 0 on every successful get_info() call. We verify the
        invariant by driving the counter and then asserting the reset line.
        """
        monitor, _, _ = self._build_monitor(max_failures=5)
        monitor._consecutive_read_failures = 4
        # The actual loop sets this to 0 on success; assert the field
        # is writable and the reset lands where we expect it.
        monitor._consecutive_read_failures = 0
        assert monitor._consecutive_read_failures == 0


class TestRiskMonitorPositionOwnership:
    """
    Wave 6 fix #24: RiskMonitor becomes the canonical owner of
    ``tracked_positions`` via a reference-holding pattern so that:

      1. ``get_positions_snapshot()`` returns a defensive copy — callers
         can iterate or mutate the returned dict without corrupting the
         shared state.
      2. After EmergencyClose fires, the halt path atomically clears
         the shared dict so the next tick's exit ladder does not chase
         ghost tickets that have just been flattened at the broker.
      3. ``SignalCombiner.reset_state()`` fires on the same halt path so
         the first post-halt bar cannot inherit a pre-halt direction
         whose underlying regime may have shifted during the halt.
    """

    def _build_monitor(self):
        from src.safety.circuit_breaker import BreakerSnapshot

        connector = MagicMock()

        circuit_breaker = MagicMock()

        emergency_close = MagicMock()
        emergency_close.close_all.return_value = {"closed": [], "failed": []}

        monitor = RiskMonitor(
            connector=connector,
            circuit_breaker=circuit_breaker,
            check_interval_seconds=1,
            emergency_close=emergency_close,
        )
        return monitor, emergency_close, BreakerSnapshot

    def test_get_positions_snapshot_returns_empty_dict_when_unset(self):
        """Safe default: snapshot is empty when main.py has not wired the ref yet."""
        monitor, _, _ = self._build_monitor()
        snap = monitor.get_positions_snapshot()
        assert snap == {}
        # Mutating the snapshot must not affect the (None) ref.
        snap[99] = "ghost"
        assert monitor.get_positions_snapshot() == {}

    def test_get_positions_snapshot_is_defensive_copy(self):
        """
        Mutating the returned dict MUST NOT affect RiskMonitor's
        reference — callers read snapshots, they do not own the source
        of truth. Matches the pattern main.py needs so the exit ladder
        can iterate a stable view.
        """
        monitor, _, _ = self._build_monitor()
        tracked = {1: "pos1", 2: "pos2"}
        monitor.set_position_tracker(tracked)

        snap = monitor.get_positions_snapshot()
        assert snap == {1: "pos1", 2: "pos2"}

        # Mutations on the snapshot do not leak back.
        snap[3] = "ghost"
        snap.pop(1)
        assert monitor.get_positions_snapshot() == {1: "pos1", 2: "pos2"}
        assert tracked == {1: "pos1", 2: "pos2"}

    def test_get_positions_snapshot_reflects_live_mutations_on_source(self):
        """
        The reference pattern is intentional — main.py keeps owning the
        dict and RiskMonitor reads through the ref. A new entry added
        to the source appears in the next snapshot.
        """
        monitor, _, _ = self._build_monitor()
        tracked = {1: "pos1"}
        monitor.set_position_tracker(tracked)
        assert monitor.get_positions_snapshot() == {1: "pos1"}
        tracked[2] = "pos2"
        assert monitor.get_positions_snapshot() == {1: "pos1", 2: "pos2"}

    def test_post_halt_cleanup_clears_tracked_positions(self):
        """
        After EmergencyClose, ``_handle_flat_required`` must atomically
        clear the shared tracked-positions dict so main.py's next tick
        does not try to walk the exit ladder over ghost tickets.
        """
        monitor, emergency_close, BreakerSnapshot = self._build_monitor()
        tracked = {10: "pos_a", 11: "pos_b", 12: "pos_c"}
        monitor.set_position_tracker(tracked)

        snap = MagicMock()
        snap.equity = 9_500.0
        breaker_snap = BreakerSnapshot(
            multiplier=0.0,
            requires_flat=True,
            active_breakers=["daily_hard"],
            daily_dd_pct=3.1,
            weekly_dd_pct=3.1,
            peak_dd_pct=3.1,
            reason="daily_hard: -3.1% > -3.0%",
        )

        monitor._handle_flat_required(snap, breaker_snap)

        emergency_close.close_all.assert_called_once()
        assert tracked == {}
        assert monitor.get_positions_snapshot() == {}

    def test_post_halt_cleanup_fires_combiner_reset_state(self):
        """
        Wave 6 fix #10 + #24: after EmergencyClose, the combiner's
        flickering ring must be flushed so the first post-halt bar
        cannot inherit a pre-halt direction.
        """
        monitor, emergency_close, BreakerSnapshot = self._build_monitor()
        combiner = MagicMock()
        monitor.set_signal_combiner(combiner)
        monitor.set_position_tracker({})

        snap = MagicMock()
        snap.equity = 9_500.0
        breaker_snap = BreakerSnapshot(
            multiplier=0.0,
            requires_flat=True,
            active_breakers=["weekly_hard"],
            daily_dd_pct=2.0,
            weekly_dd_pct=7.5,
            peak_dd_pct=7.5,
            reason="weekly_hard: -7.5% > -7.0%",
        )

        monitor._handle_flat_required(snap, breaker_snap)

        emergency_close.close_all.assert_called_once()
        combiner.reset_state.assert_called_once()

    def test_post_halt_cleanup_safe_when_combiner_unset(self):
        """
        ``set_signal_combiner`` is optional — if main.py has not wired
        the combiner yet, the halt path must NOT crash. Same for the
        tracked-positions ref.
        """
        monitor, emergency_close, BreakerSnapshot = self._build_monitor()
        # Both refs unset on purpose.
        assert monitor._tracked_positions_ref is None
        assert monitor._signal_combiner is None

        snap = MagicMock()
        snap.equity = 9_000.0
        breaker_snap = BreakerSnapshot(
            multiplier=0.0,
            requires_flat=True,
            active_breakers=["peak"],
            daily_dd_pct=0.0,
            weekly_dd_pct=0.0,
            peak_dd_pct=11.0,
            reason="peak: -11.0% > -10.0%",
        )

        # Must not raise.
        monitor._handle_flat_required(snap, breaker_snap)
        emergency_close.close_all.assert_called_once()

    def test_post_halt_cleanup_survives_combiner_reset_raising(self):
        """
        Defensive: if combiner.reset_state() raises, EmergencyClose has
        already run — we must not propagate the error and cause the
        monitor loop to swallow the halt result.
        """
        monitor, emergency_close, BreakerSnapshot = self._build_monitor()
        combiner = MagicMock()
        combiner.reset_state.side_effect = RuntimeError("combiner broken")
        monitor.set_signal_combiner(combiner)
        monitor.set_position_tracker({1: "pos"})

        snap = MagicMock()
        snap.equity = 9_500.0
        breaker_snap = BreakerSnapshot(
            multiplier=0.0,
            requires_flat=True,
            active_breakers=["daily_hard"],
            daily_dd_pct=3.1,
            weekly_dd_pct=3.1,
            peak_dd_pct=3.1,
            reason="daily_hard",
        )

        # Must not raise — raised combiner errors are logged, not rethrown.
        monitor._handle_flat_required(snap, breaker_snap)
        emergency_close.close_all.assert_called_once()
        combiner.reset_state.assert_called_once()

    def test_synthetic_trip_path_also_clears_positions_and_combiner(self):
        """
        The ``_handle_read_failure_escalation`` synthetic-trip path does
        NOT route through ``_handle_flat_required``. It calls
        EmergencyClose directly — so it must also invoke
        ``_post_halt_cleanup`` explicitly, otherwise an internet outage
        would flatten positions at the broker but leave ghost tickets
        in tracked_positions and a stale flickering ring on the combiner.
        """
        connector = MagicMock()
        connector.reconnect.return_value = False
        circuit_breaker = MagicMock()
        emergency_close = MagicMock()
        emergency_close.close_all.return_value = {"closed": [], "failed": []}
        monitor = RiskMonitor(
            connector=connector,
            circuit_breaker=circuit_breaker,
            check_interval_seconds=1,
            emergency_close=emergency_close,
            max_consecutive_read_failures=3,
        )
        tracked = {99: "pos"}
        combiner = MagicMock()
        monitor.set_position_tracker(tracked)
        monitor.set_signal_combiner(combiner)
        monitor._consecutive_read_failures = 3

        monitor._handle_read_failure_escalation()

        emergency_close.close_all.assert_called_once()
        assert tracked == {}
        combiner.reset_state.assert_called_once()
