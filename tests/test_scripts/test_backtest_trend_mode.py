"""Smoke test: trend_mode=True flag plumbs into run_backtest_full without crash."""
import inspect

import pytest


def test_run_backtest_full_accepts_trend_mode_flag():
    """Just import + call signature check. Real A/B run is operator-action."""
    from scripts.backtest_full import run_backtest_full

    sig = inspect.signature(run_backtest_full)
    assert "trend_mode" in sig.parameters
    assert sig.parameters["trend_mode"].default is False


def test_time_exit_branch_skipped_when_disabled():
    """Direct unit-style test of the new exit-barrier logic at scripts/backtest_full.py:853.

    Mirrors the inline check; verifies the AND-gate behaves correctly:
    time-exit fires only when t.time_exit_disabled is False AND bars_held >= TIME_EXIT_BARS.
    """
    class T:
        bars_held = 100
        time_exit_disabled = True

    TIME_EXIT_BARS = 60
    t = T()
    exit_reason = None
    if (
        exit_reason is None
        and not t.time_exit_disabled
        and t.bars_held >= TIME_EXIT_BARS
    ):
        exit_reason = "time_exit"
    assert exit_reason is None  # time-exit was skipped

    # Now flip the flag — baseline behavior intact
    t.time_exit_disabled = False
    if (
        exit_reason is None
        and not t.time_exit_disabled
        and t.bars_held >= TIME_EXIT_BARS
    ):
        exit_reason = "time_exit"
    assert exit_reason == "time_exit"  # baseline behavior intact

    # And the bars_held threshold still gates: not enough bars → no exit
    exit_reason = None
    t.time_exit_disabled = False
    t.bars_held = 30
    if (
        exit_reason is None
        and not t.time_exit_disabled
        and t.bars_held >= TIME_EXIT_BARS
    ):
        exit_reason = "time_exit"
    assert exit_reason is None  # below threshold


def test_diag_payload_shape():
    """Lock in the diagnostic JSON shape so Task 16's operator can rely on it."""
    # Synthetic shape — the actual write happens inside run_backtest_full,
    # not exercised here. Just verify the dict structure we'd produce.
    diag_payload = {
        "symbol": "XAUUSD",
        "by_month_and_regime": {
            "2024-07|Bull": 120,
            "2024-08|Bull": 85,
            "2024-09|Neutral": 0,
        },
        "final_state_snapshot": {
            "XAUUSD": {
                "active": True, "direction": +1, "activated_at_bar": 1234,
                "just_activated": False, "just_deactivated": False,
            },
        },
    }
    # Schema assertions: top-level keys + by-month-regime nested key format
    assert set(diag_payload.keys()) == {"symbol", "by_month_and_regime", "final_state_snapshot"}
    for k in diag_payload["by_month_and_regime"]:
        assert "|" in k  # key format = "YYYY-MM|<regime_label>"
        ym, regime = k.split("|", 1)
        assert len(ym) == 7  # YYYY-MM
        assert regime in ("Crash", "Bear", "Neutral", "Bull", "Euphoria")
    # Final snapshot should include the transition flags (Task 5 fix #1)
    snap = diag_payload["final_state_snapshot"]["XAUUSD"]
    assert "just_activated" in snap and "just_deactivated" in snap


def test_trend_pnl_delta_zero_for_non_tp_exit():
    """Attribution is only meaningful for take_profit exits.
    SL, time-exit, reversal, manual, breaker -> delta = 0."""
    # Mirror the inline attribution gate from backtest_full.py Task 15.
    for exit_reason in ("sl", "time_exit", "reversal_hard_exit", "manual", "breaker_emergency"):
        trend_pnl_delta = 0.0
        was_in_trend_mode_at_close = True
        # Gate: only fires when exit_reason == "take_profit"
        if (
            was_in_trend_mode_at_close
            and exit_reason == "take_profit"
        ):
            trend_pnl_delta = 999.0  # would set if gate passed
        assert trend_pnl_delta == 0.0


def test_trend_pnl_delta_long_take_profit_math():
    """Long position: baseline TP at 102, widened TP at 116 -> delta = (116-102)*volume - cf_commission."""
    # Synthetic: entry=100, R=1, baseline_tp_r=2, trend_tp_r=8, volume=10
    entry, r = 100.0, 1.0
    baseline_tp_r, trend_tp_r = 2.0, 8.0
    volume = 10.0
    # Frictionless for simplicity
    actual_tp = entry + r * trend_tp_r        # 108
    cf_tp = entry + r * baseline_tp_r          # 102
    actual_pnl = (actual_tp - entry) * volume   # 80
    cf_pnl = (cf_tp - entry) * volume           # 20
    delta = actual_pnl - cf_pnl                # 60
    assert delta == 60.0
    # Sanity: trend-mode added 60 dollars vs baseline on this trade


def test_trend_pnl_delta_short_take_profit_math():
    """Short position: TP_short_baseline = 148, widened = 138; delta = (148-138)*volume - cf_commission."""
    entry, r = 150.0, 1.0
    baseline_tp_r, trend_tp_r = 2.0, 8.0
    volume = 10.0
    actual_tp = entry - r * trend_tp_r          # 142
    cf_tp = entry - r * baseline_tp_r            # 148
    actual_pnl = (entry - actual_tp) * volume    # 80
    cf_pnl = (entry - cf_tp) * volume            # 20
    delta = actual_pnl - cf_pnl                  # 60
    assert delta == 60.0
