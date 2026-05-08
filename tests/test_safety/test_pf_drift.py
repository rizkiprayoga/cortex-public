"""
Tests for ``src/safety/pf_drift.py`` — live-vs-backtest drift monitor (A-4).

Pins the invariant contract: ``strategy.live_pf_drift`` fires at WARN
when live PF is 70-80% of baseline, ALERT below 70%, nothing above 80%
or with insufficient data.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.safety import invariants, pf_drift
from src.safety.invariants import Severity
from src.safety.pf_drift import (
    ALERT_RATIO,
    MIN_TRADES_TO_CHECK,
    WARN_RATIO,
    check_pf_drift,
    compute_pf_from_pnls,
)


class TestComputePfFromPnls:
    def test_empty_returns_none(self):
        assert compute_pf_from_pnls([]) is None

    def test_only_wins_returns_none(self):
        # Undefined — gross_loss == 0. Not a "drift event" — see v1 design.
        assert compute_pf_from_pnls([10.0, 5.0, 20.0]) is None

    def test_only_losses_returns_zero(self):
        # gross_profit=0 / gross_loss>0 → PF=0.0. Meaningful signal.
        assert compute_pf_from_pnls([-10.0, -5.0]) == 0.0

    def test_mixed_trades(self):
        # Gross profit = 100, gross loss = 40 → PF = 2.5
        assert compute_pf_from_pnls([100.0, -40.0]) == pytest.approx(2.5)

    def test_zeros_ignored(self):
        # Zero-pnl trades (e.g. scratches) don't contribute to PF math.
        assert compute_pf_from_pnls([100.0, 0.0, -40.0, 0.0]) == pytest.approx(2.5)

    def test_none_values_skipped(self):
        assert compute_pf_from_pnls([100.0, None, -40.0]) == pytest.approx(2.5)


def _make_ds_stub():
    """Minimal stand-in — the functions we call are monkey-patched."""
    class _DS:
        pass
    return _DS()


def _run_check(baseline, live_pf_n, monkeypatch):
    """Helper: install mocks for the two IO helpers, run check, return result."""
    ds = _make_ds_stub()

    async def _baseline(_ds, _sym):
        return baseline

    async def _live(_ds, _sym, account_id=None, window_days=30):
        return live_pf_n

    monkeypatch.setattr(pf_drift, "get_baseline_pf", _baseline)
    monkeypatch.setattr(pf_drift, "compute_live_pf", _live)
    return asyncio.run(check_pf_drift(ds, "XAUUSD"))


class TestCheckPfDrift:
    def test_no_baseline_skips_silently(self, monkeypatch, tmp_path):
        reg = invariants.InvariantRegistry(
            jsonl_path=tmp_path / "inv.jsonl", halt_flag=tmp_path / "HALT",
        )
        monkeypatch.setattr(invariants, "_REGISTRY", reg)
        result = _run_check(baseline=None, live_pf_n=(2.0, 20), monkeypatch=monkeypatch)
        assert result.severity is None
        assert result.reason == "no_baseline"
        # Invariant must NOT have fired.
        assert not any(
            f.invariant == "strategy.live_pf_drift" for f in reg.recent()
        )

    def test_insufficient_trades_skips_silently(self, monkeypatch, tmp_path):
        reg = invariants.InvariantRegistry(
            jsonl_path=tmp_path / "inv.jsonl", halt_flag=tmp_path / "HALT",
        )
        monkeypatch.setattr(invariants, "_REGISTRY", reg)
        # baseline exists, live has 5 trades — below MIN_TRADES_TO_CHECK floor.
        result = _run_check(
            baseline=3.0, live_pf_n=(1.5, MIN_TRADES_TO_CHECK - 1),
            monkeypatch=monkeypatch,
        )
        assert result.severity is None
        assert "insufficient_trades" in result.reason
        assert not any(
            f.invariant == "strategy.live_pf_drift" for f in reg.recent()
        )

    def test_live_outperforming_is_ok(self, monkeypatch, tmp_path):
        reg = invariants.InvariantRegistry(
            jsonl_path=tmp_path / "inv.jsonl", halt_flag=tmp_path / "HALT",
        )
        monkeypatch.setattr(invariants, "_REGISTRY", reg)
        # live 4.0 vs baseline 3.0 → ratio 1.33 → no alarm.
        result = _run_check(
            baseline=3.0, live_pf_n=(4.0, 20), monkeypatch=monkeypatch,
        )
        assert result.severity is None
        assert "ok" in result.reason

    def test_live_near_baseline_is_ok(self, monkeypatch, tmp_path):
        reg = invariants.InvariantRegistry(
            jsonl_path=tmp_path / "inv.jsonl", halt_flag=tmp_path / "HALT",
        )
        monkeypatch.setattr(invariants, "_REGISTRY", reg)
        # ratio exactly at WARN threshold → ok (threshold is >=).
        live = 3.0 * WARN_RATIO
        result = _run_check(
            baseline=3.0, live_pf_n=(live, 20), monkeypatch=monkeypatch,
        )
        assert result.severity is None

    def test_warn_threshold_fires_warn(self, monkeypatch, tmp_path):
        reg = invariants.InvariantRegistry(
            jsonl_path=tmp_path / "inv.jsonl", halt_flag=tmp_path / "HALT",
        )
        monkeypatch.setattr(invariants, "_REGISTRY", reg)
        # ratio 0.75 → between ALERT (0.7) and WARN (0.8) → WARN.
        live = 3.0 * 0.75
        result = _run_check(
            baseline=3.0, live_pf_n=(live, 20), monkeypatch=monkeypatch,
        )
        assert result.severity is Severity.WARN
        recent = list(reg.recent())
        assert len(recent) == 1
        assert recent[0].invariant == "strategy.live_pf_drift"
        assert recent[0].severity == Severity.WARN.value
        assert recent[0].symbol == "XAUUSD"

    def test_alert_threshold_fires_alert(self, monkeypatch, tmp_path):
        reg = invariants.InvariantRegistry(
            jsonl_path=tmp_path / "inv.jsonl", halt_flag=tmp_path / "HALT",
        )
        monkeypatch.setattr(invariants, "_REGISTRY", reg)
        # ratio 0.5 → below ALERT (0.7) → ALERT severity.
        live = 3.0 * 0.5
        result = _run_check(
            baseline=3.0, live_pf_n=(live, 20), monkeypatch=monkeypatch,
        )
        assert result.severity is Severity.ALERT
        recent = list(reg.recent())
        assert recent[0].severity == Severity.ALERT.value
        assert recent[0].context["ratio"] == pytest.approx(0.5)
        assert recent[0].context["n_trades"] == 20

    def test_live_no_losses_skips_silently(self, monkeypatch, tmp_path):
        reg = invariants.InvariantRegistry(
            jsonl_path=tmp_path / "inv.jsonl", halt_flag=tmp_path / "HALT",
        )
        monkeypatch.setattr(invariants, "_REGISTRY", reg)
        # compute_live_pf returned (None, n) → gross_loss==0 → undefined PF.
        # Not a drift event — possibly lucky streak or too-few-losses.
        result = _run_check(
            baseline=3.0, live_pf_n=(None, 20), monkeypatch=monkeypatch,
        )
        assert result.severity is None
        assert result.reason == "no_losses_in_window"
        assert not any(
            f.invariant == "strategy.live_pf_drift" for f in reg.recent()
        )

    def test_zero_pf_fires_alert(self, monkeypatch, tmp_path):
        """All losses in the window → PF = 0 → ratio = 0 → ALERT.
        This is the clearest regression signal possible."""
        reg = invariants.InvariantRegistry(
            jsonl_path=tmp_path / "inv.jsonl", halt_flag=tmp_path / "HALT",
        )
        monkeypatch.setattr(invariants, "_REGISTRY", reg)
        result = _run_check(
            baseline=3.0, live_pf_n=(0.0, 20), monkeypatch=monkeypatch,
        )
        assert result.severity is Severity.ALERT
        assert result.ratio == 0.0


class TestRunDriftChecks:
    def test_one_bad_symbol_does_not_halt_others(self, monkeypatch, tmp_path):
        """If one symbol's check raises, the others still complete."""
        reg = invariants.InvariantRegistry(
            jsonl_path=tmp_path / "inv.jsonl", halt_flag=tmp_path / "HALT",
        )
        monkeypatch.setattr(invariants, "_REGISTRY", reg)

        call_count = {"n": 0}

        async def _mock_check(_ds, symbol, account_id=None, window_days=30):
            call_count["n"] += 1
            if symbol == "EURUSD":
                raise RuntimeError("boom")
            return pf_drift.DriftCheckResult(
                symbol, 3.0, 2.8, 0.93, 15, None, "ok",
            )

        monkeypatch.setattr(pf_drift, "check_pf_drift", _mock_check)
        ds = _make_ds_stub()
        results = asyncio.run(
            pf_drift.run_drift_checks(ds, ["XAUUSD", "EURUSD", "USDJPY"])
        )
        assert call_count["n"] == 3
        assert len(results) == 3
        bad = next(r for r in results if r.symbol == "EURUSD")
        assert bad.severity is None
        assert "error" in bad.reason
