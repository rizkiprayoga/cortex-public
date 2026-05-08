"""
Tests for the multi-level CircuitBreaker.

Thresholds exercised:
    daily  soft 2%  / hard 3%
    weekly soft 5%  / hard 7%
    peak   sticky 10%
"""

from datetime import datetime, timezone

import pytest

from src.safety.circuit_breaker import BreakerSnapshot, CircuitBreaker


def fixed_now(day: int = 15) -> datetime:
    # Mid-week, mid-month — avoids anchor-roll interactions by default.
    return datetime(2026, 4, day, 12, 0, 0, tzinfo=timezone.utc)


class TestMultipliers:

    def setup_method(self):
        self.cb = CircuitBreaker()

    def test_clean_state_multiplier_is_one(self):
        snap = self.cb.check_and_update(
            current_equity=10_000.0,
            daily_start_equity=10_000.0,
            weekly_start_equity=10_000.0,
            peak_equity=10_000.0,
            now=fixed_now(),
        )
        assert snap.multiplier == 1.0
        assert not snap.requires_flat
        assert snap.active_breakers == []

    def test_daily_soft_halves(self):
        # −3.5% of daily start → crosses daily_soft (3%) but not daily_hard (5%)
        snap = self.cb.check_and_update(
            current_equity=9_650.0,
            daily_start_equity=10_000.0,
            weekly_start_equity=10_000.0,
            peak_equity=10_000.0,
            now=fixed_now(),
        )
        assert snap.multiplier == 0.5
        assert "daily_soft" in snap.active_breakers
        assert "daily_hard" not in snap.active_breakers

    def test_daily_hard_flats(self):
        # −5.5% of daily start → crosses daily_hard (5%)
        snap = self.cb.check_and_update(
            current_equity=9_450.0,
            daily_start_equity=10_000.0,
            weekly_start_equity=10_000.0,
            peak_equity=10_000.0,
            now=fixed_now(),
        )
        assert snap.multiplier == 0.0
        assert snap.requires_flat
        assert "daily_hard" in snap.active_breakers

    def test_weekly_soft_halves(self):
        # −5.5% of weekly start → weekly_soft but not weekly_hard
        snap = self.cb.check_and_update(
            current_equity=9_450.0,
            daily_start_equity=9_450.0,  # daily reset
            weekly_start_equity=10_000.0,
            peak_equity=10_000.0,
            now=fixed_now(),
        )
        assert snap.multiplier == 0.5
        assert "weekly_soft" in snap.active_breakers

    def test_weekly_hard_flats(self):
        snap = self.cb.check_and_update(
            current_equity=9_290.0,
            daily_start_equity=9_290.0,
            weekly_start_equity=10_000.0,
            peak_equity=10_000.0,
            now=fixed_now(),
        )
        assert snap.multiplier == 0.0
        assert snap.requires_flat
        assert "weekly_hard" in snap.active_breakers

    def test_peak_sticky_flats(self):
        snap = self.cb.check_and_update(
            current_equity=8_999.0,
            daily_start_equity=8_999.0,
            weekly_start_equity=8_999.0,
            peak_equity=10_000.0,
            now=fixed_now(),
        )
        assert snap.multiplier == 0.0
        assert "peak_sticky" in snap.active_breakers

    def test_peak_is_sticky_through_normal_check(self):
        # Trip the peak breaker …
        self.cb.check_and_update(
            current_equity=8_999.0,
            daily_start_equity=8_999.0,
            weekly_start_equity=8_999.0,
            peak_equity=10_000.0,
            now=fixed_now(),
        )
        # … then re-run with a clean equity. The peak flag should
        # persist until manual_reset.
        snap = self.cb.check_and_update(
            current_equity=10_500.0,
            daily_start_equity=10_500.0,
            weekly_start_equity=10_500.0,
            peak_equity=10_500.0,
            now=fixed_now(),
        )
        assert "peak_sticky" in snap.active_breakers
        assert snap.multiplier == 0.0

    def test_manual_reset_clears_everything(self):
        self.cb.check_and_update(
            current_equity=8_000.0,
            daily_start_equity=8_000.0,
            weekly_start_equity=8_000.0,
            peak_equity=10_000.0,
            now=fixed_now(),
        )
        assert self.cb.is_halted()
        self.cb.manual_reset()
        assert not self.cb.is_halted()
        assert self.cb.active_breakers() == []


class TestLatchingAndResets:

    def test_daily_breakers_reset_at_midnight(self):
        cb = CircuitBreaker()
        day1 = datetime(2026, 4, 15, 23, 0, 0, tzinfo=timezone.utc)
        # Trip daily_hard on day 1 (need −5%+ to cross new 5% hard threshold)
        cb.check_and_update(
            current_equity=9_450.0,
            daily_start_equity=10_000.0,
            weekly_start_equity=10_000.0,
            peak_equity=10_000.0,
            now=day1,
        )
        assert cb.is_halted()
        # Roll into NEXT WEEK Monday so weekly breakers also reset.
        # This cleanly verifies daily resets without weekly interference.
        next_monday = datetime(2026, 4, 20, 0, 5, 0, tzinfo=timezone.utc)
        snap = cb.check_and_update(
            current_equity=9_800.0,
            daily_start_equity=9_800.0,
            weekly_start_equity=9_800.0,
            peak_equity=10_000.0,
            now=next_monday,
        )
        assert snap.multiplier == 1.0
        assert "daily_hard" not in snap.active_breakers

    def test_weekly_breakers_reset_on_monday(self):
        cb = CircuitBreaker()
        # Friday — trip weekly_hard at −8%
        friday = datetime(2026, 4, 17, 10, 0, 0, tzinfo=timezone.utc)
        cb.check_and_update(
            current_equity=9_200.0,
            daily_start_equity=9_200.0,
            weekly_start_equity=10_000.0,
            peak_equity=10_000.0,
            now=friday,
        )
        assert cb.is_halted()
        # Following Monday 00:05 UTC → weekly breakers reset
        monday = datetime(2026, 4, 20, 0, 5, 0, tzinfo=timezone.utc)
        snap = cb.check_and_update(
            current_equity=9_200.0,
            daily_start_equity=9_200.0,
            weekly_start_equity=9_200.0,
            peak_equity=10_000.0,
            now=monday,
        )
        assert "weekly_hard" not in snap.active_breakers
        assert "weekly_soft" not in snap.active_breakers
        assert snap.multiplier == 1.0

    def test_worst_of_rule_picks_flat_over_halve(self):
        cb = CircuitBreaker()
        # daily_soft (−3.5%) AND weekly_hard (−8%) at once → flat wins
        snap = cb.check_and_update(
            current_equity=9_200.0,
            daily_start_equity=9_533.0,  # 3.5% daily DD vs this anchor
            weekly_start_equity=10_000.0,
            peak_equity=10_000.0,
            now=fixed_now(),
        )
        assert "daily_soft" in snap.active_breakers
        assert "weekly_hard" in snap.active_breakers
        assert snap.multiplier == 0.0


class TestHaltFlagFilePersistence:
    """
    Wave 6 fix #7: tripping the peak breaker writes a sentinel halt
    flag file so that a future bot instance starts in HALTED state
    after a crash / Ctrl+C / reboot. Only ``manual_reset()`` clears
    the file — daily/weekly breakers never write the flag because
    they reset on their own cadence.
    """

    def test_peak_trip_writes_halt_flag(self, tmp_path, monkeypatch):
        from src.safety import circuit_breaker as cb_mod

        flag_path = tmp_path / "TRADING_HALTED.flag"
        monkeypatch.setattr(cb_mod, "HALT_FLAG_FILE", flag_path)

        cb = cb_mod.CircuitBreaker()
        assert not flag_path.exists()

        cb.check_and_update(
            current_equity=8_999.0,
            daily_start_equity=8_999.0,
            weekly_start_equity=8_999.0,
            peak_equity=10_000.0,
            now=fixed_now(),
        )
        assert flag_path.exists()

    def test_restart_with_flag_resumes_halted(self, tmp_path, monkeypatch):
        from src.safety import circuit_breaker as cb_mod

        flag_path = tmp_path / "TRADING_HALTED.flag"
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        flag_path.touch()
        monkeypatch.setattr(cb_mod, "HALT_FLAG_FILE", flag_path)

        cb = cb_mod.CircuitBreaker()
        # Fresh instance — no explicit trip — but the persisted flag
        # must be picked up on construction.
        assert cb.is_halted()
        assert "peak_sticky" in cb.active_breakers()

    def test_manual_reset_deletes_flag(self, tmp_path, monkeypatch):
        from src.safety import circuit_breaker as cb_mod

        flag_path = tmp_path / "TRADING_HALTED.flag"
        monkeypatch.setattr(cb_mod, "HALT_FLAG_FILE", flag_path)

        cb = cb_mod.CircuitBreaker()
        cb.check_and_update(
            current_equity=8_999.0,
            daily_start_equity=8_999.0,
            weekly_start_equity=8_999.0,
            peak_equity=10_000.0,
            now=fixed_now(),
        )
        assert flag_path.exists()

        cb.manual_reset()
        assert not cb.is_halted()
        assert not flag_path.exists()

    def test_daily_hard_does_not_write_flag(self, tmp_path, monkeypatch):
        from src.safety import circuit_breaker as cb_mod

        flag_path = tmp_path / "TRADING_HALTED.flag"
        monkeypatch.setattr(cb_mod, "HALT_FLAG_FILE", flag_path)

        cb = cb_mod.CircuitBreaker()
        cb.check_and_update(
            current_equity=9_690.0,
            daily_start_equity=10_000.0,
            weekly_start_equity=10_000.0,
            peak_equity=10_000.0,
            now=fixed_now(),
        )
        # Daily-hard trips ≠ peak trip — flag must NOT be written.
        assert not flag_path.exists()
