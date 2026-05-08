"""
Tests for ExitManager — Triple Barrier exit system + reversal.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.strategy.exit_manager import (
    ExitAction,
    ExitManager,
    OpenPosition,
    TierStateStore,
)


def make_long(
    symbol: str = "XAUUSD",
    entry: float = 2000.0,
    stop: float = 1990.0,      # R = 10
    volume: float = 1.0,
    atr_trail_mult: float = 2.0,
) -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        ticket=101,
        direction="buy",
        entry_price=entry,
        initial_stop=stop,
        current_stop=stop,
        volume=volume,
        initial_volume=volume,
        atr_trail_mult=atr_trail_mult,
        strategy_name="test_strategy",
    )


def make_short(
    symbol: str = "XAUUSD",
    entry: float = 2000.0,
    stop: float = 2010.0,      # R = 10
    volume: float = 1.0,
    atr_trail_mult: float = 2.0,
) -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        ticket=201,
        direction="sell",
        entry_price=entry,
        initial_stop=stop,
        current_stop=stop,
        volume=volume,
        initial_volume=volume,
        atr_trail_mult=atr_trail_mult,
        strategy_name="test_strategy",
    )


class TestBELockLong:
    """Test breakeven lock at +1R for long positions."""

    def test_below_be_trigger_no_action(self):
        em = ExitManager(tp_r_multiple=2.5, be_trigger_r=1.0)
        pos = make_long()
        actions = em.check_exits([pos], {"XAUUSD": 2005.0}, {"XAUUSD": 5.0})
        # Price at +0.5R — no BE lock yet
        assert not any(a.action == "modify_stop" for a in actions)
        assert not pos.be_locked

    def test_at_be_trigger_locks_stop(self):
        em = ExitManager(tp_r_multiple=2.5, be_trigger_r=1.0)
        pos = make_long()
        # Price at +1R = entry + R = 2000 + 10 = 2010
        actions = em.check_exits([pos], {"XAUUSD": 2010.0}, {"XAUUSD": 5.0})
        assert len(actions) == 1
        assert actions[0].action == "modify_stop"
        assert actions[0].new_stop == 2000.0  # breakeven
        assert pos.be_locked


class TestTakeProfitLong:
    """Test take-profit barrier at +2.5R."""

    def test_below_tp_no_exit(self):
        em = ExitManager(tp_r_multiple=2.5, be_trigger_r=1.0)
        pos = make_long()
        pos.be_locked = True  # Already locked
        # Price at +2R — not enough for TP
        actions = em.check_exits([pos], {"XAUUSD": 2020.0}, {"XAUUSD": 5.0})
        assert not any(a.action == "full_close" and "take_profit" in a.reason for a in actions)

    def test_at_tp_full_close(self):
        em = ExitManager(tp_r_multiple=2.5, be_trigger_r=1.0)
        pos = make_long()
        pos.be_locked = True
        # Price at +2.5R = 2000 + 25 = 2025
        actions = em.check_exits([pos], {"XAUUSD": 2025.0}, {"XAUUSD": 5.0})
        tp_actions = [a for a in actions if a.action == "full_close" and "take_profit" in a.reason]
        assert len(tp_actions) == 1
        assert tp_actions[0].close_volume == 1.0


class TestTimeExit:
    """Test time-based vertical barrier."""

    def test_before_time_exit_no_close(self):
        # bars_held is now wall-clock-derived from opened_at.
        em = ExitManager(tp_r_multiple=2.5, time_exit_bars=20)
        pos = make_long()
        pos.opened_at = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        # 19 H1 bars after entry — under the 20-bar limit
        now = datetime(2026, 1, 1, 19, 30, tzinfo=timezone.utc)
        actions = em.check_exits(
            [pos], {"XAUUSD": 2005.0}, {"XAUUSD": 5.0}, now=now,
        )
        assert not any(a.action == "full_close" and "time_exit" in a.reason for a in actions)
        assert pos.bars_held == 19

    def test_at_time_exit_full_close(self):
        em = ExitManager(tp_r_multiple=2.5, time_exit_bars=20)
        pos = make_long()
        pos.opened_at = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        # 20 full H1 bars elapsed
        now = datetime(2026, 1, 1, 20, 0, tzinfo=timezone.utc)
        actions = em.check_exits(
            [pos], {"XAUUSD": 2005.0}, {"XAUUSD": 5.0}, now=now,
        )
        time_actions = [a for a in actions if a.action == "full_close" and "time_exit" in a.reason]
        assert len(time_actions) == 1

    def test_per_position_threshold_wins_over_class_default(self):
        # Class default says 20 bars; per-position override to 100.
        em = ExitManager(tp_r_multiple=2.5, time_exit_bars=20)
        pos = make_long()
        # Use a Monday → Wednesday window so no weekend hours are excluded.
        pos.opened_at = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)  # Mon
        pos.time_exit_bars = 100  # ETHUSD-style long leash
        # 50 H1 bars — would have tripped the class default, but per-position wins
        now = datetime(2026, 1, 7, 2, 0, tzinfo=timezone.utc)  # Wed
        actions = em.check_exits(
            [pos], {"XAUUSD": 2005.0}, {"XAUUSD": 5.0}, now=now,
        )
        assert not any("time_exit" in a.reason for a in actions)
        assert pos.bars_held == 50

    def test_no_opened_at_means_no_time_exit(self):
        # Legacy/reconciled positions may lack opened_at — must not
        # spuriously close.
        em = ExitManager(tp_r_multiple=2.5, time_exit_bars=20)
        pos = make_long()
        pos.opened_at = None
        now = datetime(2030, 1, 1, tzinfo=timezone.utc)
        actions = em.check_exits(
            [pos], {"XAUUSD": 2005.0}, {"XAUUSD": 5.0}, now=now,
        )
        assert not any("time_exit" in a.reason for a in actions)
        assert pos.bars_held == 0


class TestWeekendExclusion:
    """
    Weekend hours (Sat 00:00 → Mon 00:00 UTC) must be excluded from the
    H1 bar counter for forex/metals so time_exit fires after N trading
    hours, matching backtest bar counts.
    """

    def test_weekday_window_unchanged(self):
        # Mon 00:00 → Wed 02:00 UTC = 50 wall-clock hours, no weekend.
        em = ExitManager(tp_r_multiple=2.5, time_exit_bars=60)
        pos = make_long(symbol="EURUSD")
        pos.opened_at = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
        now = datetime(2026, 1, 7, 2, 0, tzinfo=timezone.utc)
        em.check_exits([pos], {"EURUSD": 1.10}, {"EURUSD": 0.001}, now=now)
        assert pos.bars_held == 50

    def test_spans_full_weekend_subtracts_48h(self):
        # Fri 22:00 → Mon 10:00 UTC = 60 wall-clock, 48 of which are weekend.
        em = ExitManager(tp_r_multiple=2.5, time_exit_bars=60)
        pos = make_long(symbol="USDJPY")
        pos.opened_at = datetime(2026, 1, 2, 22, 0, tzinfo=timezone.utc)  # Fri
        now = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)            # Mon
        em.check_exits([pos], {"USDJPY": 158.0}, {"USDJPY": 0.5}, now=now)
        assert pos.bars_held == 12

    def test_spans_partial_saturday(self):
        # Thu 00:00 → Sat 02:00 UTC = 50 wall-clock, 2 of which are Saturday.
        em = ExitManager(tp_r_multiple=2.5, time_exit_bars=100)
        pos = make_long(symbol="XAUUSD")
        pos.opened_at = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)   # Thu
        now = datetime(2026, 1, 3, 2, 0, tzinfo=timezone.utc)             # Sat
        em.check_exits([pos], {"XAUUSD": 2005.0}, {"XAUUSD": 5.0}, now=now)
        assert pos.bars_held == 48

    def test_weekend_fix_prevents_premature_exit(self):
        # Friday-evening entry, 60-bar limit. Before the fix this would
        # time-exit Monday morning after ~48 weekend + 12 trading hours.
        # Now it correctly holds until ~60 actual trading hours elapse.
        em = ExitManager(tp_r_multiple=2.5, time_exit_bars=60)
        pos = make_long(symbol="USDCAD")
        pos.opened_at = datetime(2026, 1, 2, 22, 0, tzinfo=timezone.utc)  # Fri 22
        now = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)            # Mon 10
        actions = em.check_exits(
            [pos], {"USDCAD": 1.37}, {"USDCAD": 0.001}, now=now,
        )
        assert not any("time_exit" in a.reason for a in actions)
        assert pos.bars_held == 12

    def test_ethusd_keeps_wall_clock(self):
        # Crypto is 24/7; no weekend subtraction.
        em = ExitManager(tp_r_multiple=2.5, time_exit_bars=100)
        pos = make_long(symbol="ETHUSD")
        pos.opened_at = datetime(2026, 1, 2, 22, 0, tzinfo=timezone.utc)  # Fri
        now = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)            # Mon
        em.check_exits([pos], {"ETHUSD": 2300.0}, {"ETHUSD": 50.0}, now=now)
        assert pos.bars_held == 60  # full wall-clock hours

    def test_multiple_weekends(self):
        # Two weekends across ~14 days — 96 hours of weekend.
        em = ExitManager(tp_r_multiple=2.5, time_exit_bars=500)
        pos = make_long(symbol="EURUSD")
        pos.opened_at = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)   # Thu
        now = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)            # Thu
        em.check_exits([pos], {"EURUSD": 1.10}, {"EURUSD": 0.001}, now=now)
        # 14 days × 24h = 336h wall-clock − 96h weekend = 240h trading.
        assert pos.bars_held == 240


class TestShortPositions:
    """Test Triple Barrier for short positions."""

    def test_short_be_lock_at_1r(self):
        em = ExitManager(tp_r_multiple=2.5, be_trigger_r=1.0)
        pos = make_short()
        # Price at -1R from entry = 2000 - 10 = 1990 for short
        actions = em.check_exits([pos], {"XAUUSD": 1990.0}, {"XAUUSD": 5.0})
        assert len(actions) == 1
        assert actions[0].action == "modify_stop"
        assert actions[0].new_stop == 2000.0  # breakeven
        assert pos.be_locked

    def test_short_tp_at_2_5r(self):
        em = ExitManager(tp_r_multiple=2.5, be_trigger_r=1.0)
        pos = make_short()
        pos.be_locked = True
        # Price at -2.5R = 2000 - 25 = 1975
        actions = em.check_exits([pos], {"XAUUSD": 1975.0}, {"XAUUSD": 5.0})
        tp_actions = [a for a in actions if a.action == "full_close"]
        assert len(tp_actions) == 1


class TestReversalHardExit:
    """Test reversal hard-exit (4 opposite signals)."""

    def test_four_opposite_signals_forces_full_close(self):
        em = ExitManager(reversal_bars_required=4)
        pos = make_long()
        pos.opened_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        # Pin now so bars_held stays under the 20-bar time-exit limit —
        # otherwise wall-clock drift fires time_exit before reversal.
        now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        signals = {"XAUUSD": ["sell", "sell", "sell", "sell"]}
        actions = em.check_exits([pos], {"XAUUSD": 2005.0}, {"XAUUSD": 5.0},
                                  recent_signals=signals, now=now)
        assert len(actions) == 1
        assert actions[0].action == "full_close"
        assert "reversal" in actions[0].reason

    def test_three_opposite_plus_one_matching_does_not_trigger(self):
        em = ExitManager(reversal_bars_required=4)
        pos = make_long()
        pos.opened_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        signals = {"XAUUSD": ["sell", "sell", "sell", "buy"]}
        actions = em.check_exits([pos], {"XAUUSD": 2005.0}, {"XAUUSD": 5.0},
                                  recent_signals=signals, now=now)
        reversal_actions = [a for a in actions if "reversal" in a.reason]
        assert len(reversal_actions) == 0


class TestReversalNewestLegOnly:
    """Test that only newest leg closes on reversal."""

    def test_only_newest_leg_is_closed_on_reversal(self):
        em = ExitManager(reversal_bars_required=4)
        older = make_long()
        older.ticket = 100
        older.be_locked = True
        older.opened_at = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

        newer = make_long()
        newer.ticket = 101
        newer.opened_at = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)

        signals = {"XAUUSD": ["sell", "sell", "sell", "sell"]}
        # Pin now so neither leg trips the time-exit barrier (older=10, newer=4).
        now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        actions = em.check_exits(
            [older, newer], {"XAUUSD": 2005.0}, {"XAUUSD": 5.0},
            recent_signals=signals, now=now,
        )
        reversal_actions = [a for a in actions if "reversal" in a.reason]
        assert len(reversal_actions) == 1
        assert reversal_actions[0].ticket == 101  # newest

    def test_older_legs_still_receive_barrier_logic(self):
        em = ExitManager(tp_r_multiple=2.5, be_trigger_r=1.0, reversal_bars_required=4)
        older = make_long()
        older.ticket = 100
        older.opened_at = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

        newer = make_long()
        newer.ticket = 101
        newer.opened_at = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)

        signals = {"XAUUSD": ["sell", "sell", "sell", "sell"]}
        # Price at +1R — older leg should get BE lock, newer gets reversal.
        # Pin now so neither leg trips the time-exit barrier.
        now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        actions = em.check_exits(
            [older, newer], {"XAUUSD": 2010.0}, {"XAUUSD": 5.0},
            recent_signals=signals, now=now,
        )
        # Newer closed by reversal
        assert any(a.ticket == 101 and "reversal" in a.reason for a in actions)
        # Older gets BE lock
        assert any(a.ticket == 100 and a.action == "modify_stop" for a in actions)


class TestSimulatedTrajectoryLong:
    """Simulate a full Triple Barrier trade lifecycle."""

    def test_full_barrier_sequence(self):
        em = ExitManager(tp_r_multiple=2.5, be_trigger_r=1.0, time_exit_bars=20)
        pos = make_long()

        # Bar 1-5: price drifts up slowly, below +1R
        for price in [2002, 2004, 2006, 2008, 2009]:
            actions = em.check_exits([pos], {"XAUUSD": price}, {"XAUUSD": 5.0})
            assert not any(a.action in ("full_close", "modify_stop") for a in actions)

        # Bar 6: price hits +1R → BE lock
        actions = em.check_exits([pos], {"XAUUSD": 2010.0}, {"XAUUSD": 5.0})
        assert pos.be_locked
        assert pos.current_stop == 2000.0

        # Bar 7-12: price continues up
        for price in [2012, 2015, 2018, 2020, 2022, 2023]:
            actions = em.check_exits([pos], {"XAUUSD": price}, {"XAUUSD": 5.0})

        # Bar 13: price hits +2.5R → TP exit
        actions = em.check_exits([pos], {"XAUUSD": 2025.0}, {"XAUUSD": 5.0})
        assert any(a.action == "full_close" and "take_profit" in a.reason for a in actions)


class TestTierStateStore:
    """Test persistence of exit state across restarts."""

    def test_roundtrip_via_store(self, tmp_path):
        path = tmp_path / "tier_state.json"
        store = TierStateStore(path)
        store.upsert(42, be_locked=True, bars_held=5)
        store.upsert(43, be_locked=False, bars_held=0)

        loaded = TierStateStore(path)
        assert loaded.get(42)["be_locked"] is True
        assert loaded.get(42)["bars_held"] == 5
        assert loaded.get(43)["be_locked"] is False

    def test_corrupt_file_starts_empty_with_warning(self, tmp_path):
        path = tmp_path / "tier_state.json"
        path.write_text("NOT_JSON!!!")
        store = TierStateStore(path)
        assert len(store) == 0

    def test_missing_file_is_silently_empty(self, tmp_path):
        store = TierStateStore(tmp_path / "nonexistent.json")
        assert len(store) == 0

    def test_upsert_merges_without_overwriting(self, tmp_path):
        store = TierStateStore(tmp_path / "tier.json")
        store.upsert(1, be_locked=False, bars_held=0)
        store.upsert(1, be_locked=True)
        rec = store.get(1)
        assert rec["be_locked"] is True
        assert rec["bars_held"] == 0

    def test_delete_removes_record_and_flushes(self, tmp_path):
        store = TierStateStore(tmp_path / "tier.json")
        store.upsert(99, be_locked=True)
        store.delete(99)
        assert store.get(99) is None

    def test_exit_manager_writes_to_store_on_be_lock(self, tmp_path):
        store = TierStateStore(tmp_path / "tier.json")
        em = ExitManager(tp_r_multiple=2.5, be_trigger_r=1.0, tier_state_store=store)
        pos = make_long()
        # Price at +1R → triggers BE lock and store write
        em.check_exits([pos], {"XAUUSD": 2010.0}, {"XAUUSD": 5.0})
        rec = store.get(101)
        assert rec is not None
        assert rec["be_locked"] is True


class TestTier1DoneProperty:
    """Verify tier_1_done property maps to be_locked for pyramiding gate."""

    def test_tier_1_done_reflects_be_locked(self):
        pos = make_long()
        assert pos.tier_1_done is False
        pos.be_locked = True
        assert pos.tier_1_done is True
