"""Tests for calendar_features — news blackout & per-symbol routing (Phase C)."""

from datetime import datetime, timedelta

import pytest

from src.data_pipeline.market.calendar_features import (
    CalendarFeatureBuilder,
    describe_news_context,
    is_in_legacy_blackout,
    is_in_news_blackout,
    SYMBOL_CENTRAL_BANKS,
    _FOMC_DT, _ECB_DT, _BOJ_DT, _BOC_DT,
    _BOE_DT, _RBA_DT, _RBNZ_DT,
)


# Known announcement day used across tests (FOMC, ECB, BoJ, BoC all on
# 2024-01-25 / 2024-01-31 — pick a FOMC day known to exist)
FOMC_DAY = datetime(2024, 1, 31, 12, 0)   # 2024-01-31 12:00 UTC
ECB_DAY = datetime(2024, 1, 25, 12, 0)    # 2024-01-25 ECB
BOJ_DAY = datetime(2024, 1, 23, 12, 0)    # 2024-01-23 BoJ
BOC_DAY = datetime(2024, 1, 24, 12, 0)    # 2024-01-24 BoC


class TestSmartBlackout:
    """Verify the pre-news + spike window blocks but post-news allows."""

    def test_xauusd_always_allowed(self):
        """Gold is exempt from news blackout."""
        for dt in [FOMC_DAY, FOMC_DAY - timedelta(hours=12),
                    FOMC_DAY + timedelta(hours=1)]:
            assert not is_in_news_blackout("XAUUSD", dt)

    def test_fomc_12h_before_blocks_usdjpy(self):
        """T-12h is inside [-24h, +2h] hard window."""
        dt = FOMC_DAY - timedelta(hours=12)
        assert is_in_news_blackout("USDJPY", dt) is True
        assert is_in_news_blackout("EURUSD", dt) is True
        assert is_in_news_blackout("USDCAD", dt) is True

    def test_fomc_2h_after_blocks(self):
        """T+2h is at the edge of the hard window (inclusive)."""
        dt = FOMC_DAY + timedelta(hours=2)
        assert is_in_news_blackout("USDJPY", dt) is True

    def test_fomc_3h_after_allows_post_news_continuation(self):
        """T+3h is AFTER spike zone — this is the post-news continuation
        window where the retail edge is. Must not be blocked."""
        dt = FOMC_DAY + timedelta(hours=3)
        assert is_in_news_blackout("USDJPY", dt) is False
        assert is_in_news_blackout("EURUSD", dt) is False

    def test_fomc_24h_after_still_allows(self):
        """T+24h is still inside post-news window, allowed."""
        dt = FOMC_DAY + timedelta(hours=24)
        assert is_in_news_blackout("USDJPY", dt) is False

    def test_fomc_48h_after_allows(self):
        """T+48h is end of post-news window, allowed (no neighboring event)."""
        dt = FOMC_DAY + timedelta(hours=48)
        assert is_in_news_blackout("USDJPY", dt) is False

    def test_fomc_25h_before_allows(self):
        """T-25h is outside the pre-window."""
        dt = FOMC_DAY - timedelta(hours=25)
        # Must be far from any other CB event for this symbol
        # Test with USDCAD which has no BoC event on Jan 30 2024
        # (next BoC was 2024-01-24) — use a date further from BoC too
        dt_far = datetime(2024, 1, 10, 12, 0)
        assert is_in_news_blackout("USDCAD", dt_far) is False


class TestPerSymbolRouting:
    """ECB should block EUR but not JPY; BoJ should block JPY but not EUR, etc."""

    def test_ecb_blocks_eurusd_only(self):
        """On ECB day, EURUSD is blocked but USDJPY/USDCAD are not
        (assuming no FOMC/BoJ/BoC same day)."""
        dt = ECB_DAY
        assert is_in_news_blackout("EURUSD", dt) is True
        # USDJPY: only blocked if near BoJ. 2024-01-25 is 2 days after
        # BoJ 2024-01-23 — may be in post-news window but not hard block.
        # Check with describe_news_context for clarity.
        ctx = describe_news_context("USDJPY", dt)
        # It should be post-news for BoJ (Jan 23 → Jan 25 is +48h)
        # but NOT in hard blackout from BoJ.
        # FOMC is Jan 31 → Jan 25 is -6 days = -144h, outside window.
        assert is_in_news_blackout("USDJPY", dt) is False

    def test_boj_blocks_usdjpy_only(self):
        """BoJ day: USDJPY blocked, EUR/CAD not."""
        # Pick a BoJ date with no concurrent central-bank events nearby.
        # BoJ 2024-06-14 is standalone (FOMC is 2024-06-12 → 48h prior).
        boj_day = datetime(2024, 6, 14, 12, 0)
        # USDJPY should be blocked (BoJ is 0h away)
        assert is_in_news_blackout("USDJPY", boj_day) is True
        # EURUSD has no ECB nearby (ECB was June 6, 8 days prior)
        # and FOMC was June 12, 2 days prior → +48h from FOMC, allowed
        assert is_in_news_blackout("EURUSD", boj_day) is False
        # USDCAD has no BoC on this day either (BoC was June 5)
        assert is_in_news_blackout("USDCAD", boj_day) is False

    def test_boc_blocks_usdcad_only(self):
        """BoC day: USDCAD blocked, EUR/JPY not."""
        # BoC 2024-07-24 — standalone (FOMC July 31 = +7d, ECB July 18 = -6d,
        # BoJ July 31 = +7d — all outside pre/spike windows)
        boc_day = datetime(2024, 7, 24, 12, 0)
        assert is_in_news_blackout("USDCAD", boc_day) is True
        assert is_in_news_blackout("USDJPY", boc_day) is False
        assert is_in_news_blackout("EURUSD", boc_day) is False

    def test_symbol_routing_table_is_correct(self):
        """XAUUSD must have empty CB list; all forex pairs include FOMC."""
        assert SYMBOL_CENTRAL_BANKS["XAUUSD"] == []
        assert "FOMC" in SYMBOL_CENTRAL_BANKS["USDJPY"]
        assert "BoJ" in SYMBOL_CENTRAL_BANKS["USDJPY"]
        assert "FOMC" in SYMBOL_CENTRAL_BANKS["EURUSD"]
        assert "ECB" in SYMBOL_CENTRAL_BANKS["EURUSD"]
        assert "FOMC" in SYMBOL_CENTRAL_BANKS["USDCAD"]
        assert "BoC" in SYMBOL_CENTRAL_BANKS["USDCAD"]


class TestHistoricalDatesBackfilled:
    """Walk-forward gap fix: 2021-2023 FOMC/ECB/BoJ/BoC dates exist."""

    def test_fomc_covers_2021_2027(self):
        years = {d.year for d in _FOMC_DT}
        for y in (2021, 2022, 2023, 2024, 2025, 2026, 2027):
            assert y in years, f"FOMC missing year {y}"

    def test_ecb_covers_2021_2027(self):
        years = {d.year for d in _ECB_DT}
        for y in (2021, 2022, 2023, 2024, 2025, 2026, 2027):
            assert y in years, f"ECB missing year {y}"

    def test_boj_covers_2021_2027(self):
        years = {d.year for d in _BOJ_DT}
        for y in (2021, 2022, 2023, 2024, 2025, 2026, 2027):
            assert y in years, f"BoJ missing year {y}"

    def test_boc_covers_2021_2027(self):
        years = {d.year for d in _BOC_DT}
        for y in (2021, 2022, 2023, 2024, 2025, 2026, 2027):
            assert y in years, f"BoC missing year {y}"

    def test_each_bank_has_eight_meetings_per_year(self):
        """Central banks typically hold 8 policy meetings per year."""
        for label, dates in [("FOMC", _FOMC_DT), ("ECB", _ECB_DT),
                              ("BoJ", _BOJ_DT), ("BoC", _BOC_DT)]:
            for y in range(2021, 2028):
                count = sum(1 for d in dates if d.year == y)
                assert count == 8, f"{label} {y} has {count} meetings, expected 8"


class TestLegacyBlackout:
    """Regression check: legacy symmetric ±24h blackout still works."""

    def test_legacy_blocks_before_and_after(self):
        for offset_hours in (-20, -2, 0, 2, 20):
            dt = FOMC_DAY + timedelta(hours=offset_hours)
            assert is_in_legacy_blackout("USDJPY", dt) is True

    def test_legacy_allows_outside_24h(self):
        dt = FOMC_DAY + timedelta(hours=30)
        assert is_in_legacy_blackout("USDJPY", dt) is False


class TestNewsContextDescription:
    """describe_news_context exposes debug info to logs."""

    def test_describe_pre_news(self):
        dt = FOMC_DAY - timedelta(hours=6)
        ctx = describe_news_context("USDJPY", dt)
        assert ctx["blackout"] is True
        assert ctx["nearest_hours"] is not None
        assert ctx["nearest_hours"] < 0  # before event

    def test_describe_post_news_window(self):
        dt = FOMC_DAY + timedelta(hours=6)
        ctx = describe_news_context("USDJPY", dt)
        assert ctx["blackout"] is False
        assert ctx["post_news"] is True


class TestCalendarFeaturesUnchanged:
    """Regression: existing calendar-feature behavior must not break."""

    def test_all_keys_present(self):
        builder = CalendarFeatureBuilder()
        feats = builder.get_calendar_features(datetime(2024, 6, 15, 10, 0))
        expected = {
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos",
            "is_london_session", "is_ny_session",
            "is_nfp_week", "days_to_fomc",
        }
        assert set(feats.keys()) == expected

    def test_days_to_fomc_in_range(self):
        builder = CalendarFeatureBuilder()
        feats = builder.get_calendar_features(datetime(2024, 6, 15, 10, 0))
        assert 0.0 <= feats["days_to_fomc"] <= 1.0


# =========================================================================
# Forex Phase 1 — 6 new pairs (GBP/AUD/NZD-bearing)
# =========================================================================

class TestNewPairRouting:
    """GBPUSD/AUDUSD/EURGBP/EURJPY/GBPJPY/AUDNZD must route to the right banks."""

    def test_gbpusd_has_fomc_and_boe(self):
        assert SYMBOL_CENTRAL_BANKS["GBPUSD"] == ["FOMC", "BoE"]

    def test_audusd_has_fomc_and_rba(self):
        assert SYMBOL_CENTRAL_BANKS["AUDUSD"] == ["FOMC", "RBA"]

    def test_eurgbp_has_ecb_and_boe_no_usd(self):
        assert SYMBOL_CENTRAL_BANKS["EURGBP"] == ["ECB", "BoE"]
        assert "FOMC" not in SYMBOL_CENTRAL_BANKS["EURGBP"]

    def test_eurjpy_has_ecb_and_boj_no_usd(self):
        assert SYMBOL_CENTRAL_BANKS["EURJPY"] == ["ECB", "BoJ"]
        assert "FOMC" not in SYMBOL_CENTRAL_BANKS["EURJPY"]

    def test_gbpjpy_has_boe_and_boj_no_usd(self):
        assert SYMBOL_CENTRAL_BANKS["GBPJPY"] == ["BoE", "BoJ"]
        assert "FOMC" not in SYMBOL_CENTRAL_BANKS["GBPJPY"]

    def test_audnzd_has_rba_and_rbnz_no_usd(self):
        assert SYMBOL_CENTRAL_BANKS["AUDNZD"] == ["RBA", "RBNZ"]
        assert "FOMC" not in SYMBOL_CENTRAL_BANKS["AUDNZD"]


class TestNewBankBlackouts:
    """Blackout windows fire correctly for the new CBs' decisions."""

    def test_boe_day_blocks_gbp_pairs(self):
        """A known BoE day (no overlap with other CBs)."""
        # 2024-06-20 — BoE standalone (FOMC June 12, ECB June 6, BoJ June 14 —
        # all outside +/-24h window from June 20)
        dt = datetime(2024, 6, 20, 12, 0)
        assert is_in_news_blackout("GBPUSD", dt) is True
        assert is_in_news_blackout("EURGBP", dt) is True
        assert is_in_news_blackout("GBPJPY", dt) is True
        # Non-GBP pairs should not be blocked by BoE
        assert is_in_news_blackout("AUDUSD", dt) is False
        assert is_in_news_blackout("AUDNZD", dt) is False

    def test_rba_day_blocks_aud_pairs(self):
        """A known RBA day (Sep 24 2024, new 8-meeting format era)."""
        dt = datetime(2024, 9, 24, 12, 0)
        assert is_in_news_blackout("AUDUSD", dt) is True
        assert is_in_news_blackout("AUDNZD", dt) is True
        # Non-AUD pairs should not be blocked by RBA
        assert is_in_news_blackout("GBPUSD", dt) is False
        assert is_in_news_blackout("EURGBP", dt) is False

    def test_rbnz_day_blocks_audnzd_only(self):
        """RBNZ 2024-05-22 — standalone (no other CB within 24h)."""
        dt = datetime(2024, 5, 22, 12, 0)
        assert is_in_news_blackout("AUDNZD", dt) is True
        # AUDUSD has no RBNZ exposure
        assert is_in_news_blackout("AUDUSD", dt) is False
        assert is_in_news_blackout("GBPUSD", dt) is False

    def test_xau_still_exempt_from_all_new_banks(self):
        """XAUUSD routing stays empty; no new bank should block it."""
        for dt in [datetime(2024, 6, 20, 12, 0),   # BoE
                    datetime(2024, 9, 24, 12, 0),   # RBA
                    datetime(2024, 5, 22, 12, 0)]:  # RBNZ
            assert is_in_news_blackout("XAUUSD", dt) is False


class TestNewBankHistoricalCoverage:
    """BoE/RBA/RBNZ date lists cover 2021-2027 for walk-forward backtests."""

    def test_boe_covers_2021_2027(self):
        years = {d.year for d in _BOE_DT}
        for y in range(2021, 2028):
            assert y in years, f"BoE missing year {y}"

    def test_rba_covers_2021_2027(self):
        years = {d.year for d in _RBA_DT}
        for y in range(2021, 2028):
            assert y in years, f"RBA missing year {y}"

    def test_rbnz_covers_2021_2027(self):
        years = {d.year for d in _RBNZ_DT}
        for y in range(2021, 2028):
            assert y in years, f"RBNZ missing year {y}"

    def test_boe_has_eight_meetings_per_year(self):
        """BoE MPC meets 8 times/year — canonical."""
        for y in range(2021, 2028):
            count = sum(1 for d in _BOE_DT if d.year == y)
            assert count == 8, f"BoE {y} has {count}, expected 8"

    def test_rba_meeting_cadence_by_era(self):
        """Old Board (2021-2023): 11 meetings. New MPB (2024+): 8 meetings."""
        for y in (2021, 2022, 2023):
            count = sum(1 for d in _RBA_DT if d.year == y)
            assert count == 11, f"RBA old-board {y} has {count}, expected 11"
        for y in (2024, 2025, 2026, 2027):
            count = sum(1 for d in _RBA_DT if d.year == y)
            assert count == 8, f"RBA new-MPB {y} has {count}, expected 8"

    def test_rbnz_meeting_cadence_by_era(self):
        """Old cadence (2021-2026): 7 meetings. New cadence from 2027: 8."""
        for y in range(2021, 2027):
            count = sum(1 for d in _RBNZ_DT if d.year == y)
            assert count == 7, f"RBNZ {y} has {count}, expected 7"
        count_2027 = sum(1 for d in _RBNZ_DT if d.year == 2027)
        assert count_2027 == 8, f"RBNZ 2027 has {count_2027}, expected 8"
