"""
calendar_features.py — Temporal & Calendar Feature Builder + News Blackout

Computes calendar/temporal features from bar timestamps AND provides a
multi-central-bank news-blackout checker for the trading engine.

Two responsibilities:

1. **Calendar features** (~10 per bar) for LSTM/HMM training:
   - hour_sin, hour_cos       — time-of-day cyclical encoding
   - dow_sin, dow_cos         — day-of-week cyclical encoding
   - month_sin, month_cos     — month-of-year cyclical encoding
   - is_london_session        — 1 during London hours (07:00–16:00 UTC)
   - is_ny_session            — 1 during New York hours (13:00–22:00 UTC)
   - is_nfp_week              — 1 during US NFP week (1st Friday)
   - days_to_fomc             — normalized days until next FOMC meeting

2. **Smart news blackout** for trade-entry gating:
   - Per-symbol routing: FOMC affects all USD pairs; ECB → EURUSD;
     BoJ → USDJPY; BoC → USDCAD. XAUUSD is exempt (gold often
     benefits from either direction of rate decisions).
   - Smart windowing: blocks only the pre-news + spike zone
     (T-24h to T+2h). The post-news continuation window
     (T+2h to T+48h) is INTENTIONALLY NOT blocked — riding the
     confirmed trend after the spike settles is where retail edge is.
   - Binary old behavior (block ±24h) is available via
     ``is_in_legacy_blackout()`` for regression comparison.

Dates are curated from public central-bank calendars. Historical dates
2021-2024 are included so walk-forward backtests reflect proper news
filtering. Future dates 2026-2027 are scheduled meetings.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================================
# Central bank meeting dates (announcement days, UTC)
# =========================================================================

# FOMC — US Federal Reserve
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_DATES = [
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # 2026 — verified against federalreserve.gov/monetarypolicy/fomccalendars.htm (2026-04-15).
    # Note: no November meeting in 2026; last two are Oct 28 + Dec 9.
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    # 2027 (Fed publishes ~mid-year; verify before year-end)
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-16",
    "2027-07-28", "2027-09-22", "2027-11-03", "2027-12-15",
]

# ECB — European Central Bank monetary policy meetings
# Source: https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html
ECB_DATES = [
    # 2021
    "2021-01-21", "2021-03-11", "2021-04-22", "2021-06-10",
    "2021-07-22", "2021-09-09", "2021-10-28", "2021-12-16",
    # 2022
    "2022-02-03", "2022-03-10", "2022-04-14", "2022-06-09",
    "2022-07-21", "2022-09-08", "2022-10-27", "2022-12-15",
    # 2023
    "2023-02-02", "2023-03-16", "2023-05-04", "2023-06-15",
    "2023-07-27", "2023-09-14", "2023-10-26", "2023-12-14",
    # 2024
    "2024-01-25", "2024-03-07", "2024-04-11", "2024-06-06",
    "2024-07-18", "2024-09-12", "2024-10-17", "2024-12-12",
    # 2025
    "2025-01-30", "2025-03-06", "2025-04-17", "2025-06-05",
    "2025-07-24", "2025-09-11", "2025-10-30", "2025-12-18",
    # 2026 — future dates verified against ecb.europa.eu 2026 calendar
    # (2026-04-15). Past dates (Jan/Mar) not re-verified, left as-is.
    "2026-01-29", "2026-03-05", "2026-04-30", "2026-06-11",
    "2026-07-23", "2026-09-10", "2026-10-29", "2026-12-17",
    # 2027
    "2027-01-28", "2027-03-04", "2027-04-22", "2027-06-03",
    "2027-07-22", "2027-09-16", "2027-10-28", "2027-12-16",
]

# BoJ — Bank of Japan monetary policy meetings
# Source: https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm
BOJ_DATES = [
    # 2021
    "2021-01-21", "2021-03-19", "2021-04-27", "2021-06-18",
    "2021-07-16", "2021-09-22", "2021-10-28", "2021-12-17",
    # 2022
    "2022-01-18", "2022-03-18", "2022-04-28", "2022-06-17",
    "2022-07-21", "2022-09-22", "2022-10-28", "2022-12-20",
    # 2023
    "2023-01-18", "2023-03-10", "2023-04-28", "2023-06-16",
    "2023-07-28", "2023-09-22", "2023-10-31", "2023-12-19",
    # 2024
    "2024-01-23", "2024-03-19", "2024-04-26", "2024-06-14",
    "2024-07-31", "2024-09-20", "2024-10-31", "2024-12-19",
    # 2025
    "2025-01-24", "2025-03-19", "2025-05-01", "2025-06-17",
    "2025-07-31", "2025-09-19", "2025-10-30", "2025-12-19",
    # 2026 — verified against boj.or.jp (2026-04-15). Jun/Sep still "TBA"
    # on the BoJ site; kept as placeholders, verify before those months.
    "2026-01-23", "2026-03-19", "2026-04-28", "2026-06-17",
    "2026-07-31", "2026-09-18", "2026-10-30", "2026-12-18",
    # 2027 (approximate — BoJ publishes year-by-year)
    "2027-01-22", "2027-03-18", "2027-04-28", "2027-06-17",
    "2027-07-29", "2027-09-17", "2027-10-28", "2027-12-17",
]

# BoC — Bank of Canada monetary policy announcements
# Source: https://www.bankofcanada.ca/core-functions/monetary-policy/key-interest-rate/
BOC_DATES = [
    # 2021
    "2021-01-20", "2021-03-10", "2021-04-21", "2021-06-09",
    "2021-07-14", "2021-09-08", "2021-10-27", "2021-12-08",
    # 2022
    "2022-01-26", "2022-03-02", "2022-04-13", "2022-06-01",
    "2022-07-13", "2022-09-07", "2022-10-26", "2022-12-07",
    # 2023
    "2023-01-25", "2023-03-08", "2023-04-12", "2023-06-07",
    "2023-07-12", "2023-09-06", "2023-10-25", "2023-12-06",
    # 2024
    "2024-01-24", "2024-03-06", "2024-04-10", "2024-06-05",
    "2024-07-24", "2024-09-04", "2024-10-23", "2024-12-11",
    # 2025
    "2025-01-29", "2025-03-12", "2025-04-16", "2025-06-04",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 — verified against bankofcanada.ca 2026 schedule (2026-04-15).
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-10",
    "2026-07-15", "2026-09-02", "2026-10-28", "2026-12-09",
    # 2027 (BoC not yet published — heuristic, verify before year-end)
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-02",
    "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
]

# BoE — Bank of England Monetary Policy Committee announcements
# Source: https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates
# MPC meets 8 times/year, announcements at 12:00 noon UK time on Thursdays.
# Historical + 2026 confirmed, 2027 provisional. Verified 2026-04-24.
BOE_DATES = [
    # 2021
    "2021-02-04", "2021-03-18", "2021-05-06", "2021-06-24",
    "2021-08-05", "2021-09-23", "2021-11-04", "2021-12-16",
    # 2022
    "2022-02-03", "2022-03-17", "2022-05-05", "2022-06-16",
    "2022-08-04", "2022-09-15", "2022-11-03", "2022-12-15",
    # 2023
    "2023-02-02", "2023-03-23", "2023-05-11", "2023-06-22",
    "2023-08-03", "2023-09-21", "2023-11-02", "2023-12-14",
    # 2024
    "2024-02-01", "2024-03-21", "2024-05-09", "2024-06-20",
    "2024-08-01", "2024-09-19", "2024-11-07", "2024-12-19",
    # 2025
    "2025-02-06", "2025-03-20", "2025-05-08", "2025-06-19",
    "2025-08-07", "2025-09-18", "2025-11-06", "2025-12-18",
    # 2026 — confirmed
    "2026-02-05", "2026-03-19", "2026-04-30", "2026-06-18",
    "2026-07-30", "2026-09-17", "2026-11-05", "2026-12-17",
    # 2027 — provisional
    "2027-02-04", "2027-03-18", "2027-04-29", "2027-06-17",
    "2027-07-29", "2027-09-16", "2027-11-04", "2027-12-16",
]

# RBA — Reserve Bank of Australia monetary policy decisions
# Source: https://www.rba.gov.au/monetary-policy/int-rate-decisions/
#         https://www.rba.gov.au/schedules-events/board-meeting-schedules.html
# 2021-2023: old RBA Board, 11 meetings/year (no January). First Tuesday of month.
# 2024: switched to 8 meetings/year (new format announced 2023).
# 2025+: old RBA Board finalized 2025-02-18; new Monetary Policy Board from
#        2025-03-01, meetings are 2-day; decision announced at 14:30 AEST
#        on the SECOND day. Values below are the announcement day.
# Verified 2026-04-24.
RBA_DATES = [
    # 2021 (old Board, 11 meetings)
    "2021-02-02", "2021-03-02", "2021-04-06", "2021-05-04",
    "2021-06-01", "2021-07-06", "2021-08-03", "2021-09-07",
    "2021-10-05", "2021-11-02", "2021-12-07",
    # 2022 (old Board, 11 meetings)
    "2022-02-01", "2022-03-01", "2022-04-05", "2022-05-03",
    "2022-06-07", "2022-07-05", "2022-08-02", "2022-09-06",
    "2022-10-04", "2022-11-01", "2022-12-06",
    # 2023 (old Board, 11 meetings)
    "2023-02-07", "2023-03-07", "2023-04-04", "2023-05-02",
    "2023-06-06", "2023-07-04", "2023-08-01", "2023-09-05",
    "2023-10-03", "2023-11-07", "2023-12-05",
    # 2024 (new 8-meetings format)
    "2024-02-06", "2024-03-19", "2024-05-07", "2024-06-18",
    "2024-08-06", "2024-09-24", "2024-11-05", "2024-12-10",
    # 2025 (final old Board meeting Feb 18, then new MPB from March)
    "2025-02-18", "2025-04-01", "2025-05-20", "2025-07-08",
    "2025-08-12", "2025-09-30", "2025-11-04", "2025-12-09",
    # 2026 (MPB confirmed — announcement is SECOND day of 2-day meeting)
    "2026-02-03", "2026-03-17", "2026-05-05", "2026-06-16",
    "2026-08-11", "2026-09-29", "2026-11-03", "2026-12-08",
    # 2027 (MPB preliminary)
    "2027-02-09", "2027-03-23", "2027-05-04", "2027-06-22",
    "2027-08-10", "2027-09-21", "2027-11-02", "2027-12-14",
]

# RBNZ — Reserve Bank of New Zealand OCR decisions
# Source: https://www.rbnz.govt.nz/monetary-policy/monetary-policy-decisions
#         https://www.rbnz.govt.nz/news-and-events/how-we-release-information
# 2021-2026: 7 decisions/year (4 Monetary Policy Statements + 3 Reviews).
# 2027+: 8 decisions/year (aligned with new monthly CPI releases from Stats NZ).
# Announcements at 14:00 NZST/NZDT. Verified 2026-04-24.
RBNZ_DATES = [
    # 2021 (7 meetings)
    "2021-02-24", "2021-04-14", "2021-05-26", "2021-07-14",
    "2021-08-18", "2021-10-06", "2021-11-24",
    # 2022 (7 meetings)
    "2022-02-23", "2022-04-13", "2022-05-25", "2022-07-13",
    "2022-08-17", "2022-10-05", "2022-11-23",
    # 2023 (7 meetings)
    "2023-02-22", "2023-04-05", "2023-05-24", "2023-07-12",
    "2023-08-16", "2023-10-04", "2023-11-29",
    # 2024 (7 meetings)
    "2024-02-28", "2024-04-10", "2024-05-22", "2024-07-10",
    "2024-08-14", "2024-10-09", "2024-11-27",
    # 2025 (7 meetings)
    "2025-02-19", "2025-04-09", "2025-05-28", "2025-07-09",
    "2025-08-20", "2025-10-08", "2025-11-26",
    # 2026 (7 meetings — last 7-meetings year)
    "2026-02-18", "2026-04-08", "2026-05-27", "2026-07-08",
    "2026-09-02", "2026-10-28", "2026-12-09",
    # 2027 (8 meetings — new cadence)
    "2027-02-10", "2027-03-17", "2027-05-05", "2027-06-16",
    "2027-08-04", "2027-09-15", "2027-10-27", "2027-12-08",
]


def _parse_dates(date_list: list[str]) -> list[datetime]:
    return [datetime.strptime(d, "%Y-%m-%d") for d in date_list]


_FOMC_DT = _parse_dates(FOMC_DATES)
_ECB_DT = _parse_dates(ECB_DATES)
_BOJ_DT = _parse_dates(BOJ_DATES)
_BOC_DT = _parse_dates(BOC_DATES)
_BOE_DT = _parse_dates(BOE_DATES)
_RBA_DT = _parse_dates(RBA_DATES)
_RBNZ_DT = _parse_dates(RBNZ_DATES)


def calendar_freshness_warning(min_lookahead_days: int = 60) -> Optional[str]:
    """
    Return a warning string if any central-bank calendar is nearing its
    last known date (audit HIGH-1 — static calendars silently rot).

    Returns None when all calendars extend at least ``min_lookahead_days``
    into the future from now. Log the returned string at startup so
    operators know when to refresh the hardcoded dates.
    """
    now = datetime.utcnow()
    stale: list[str] = []
    for label, dates in (("FOMC", _FOMC_DT), ("ECB", _ECB_DT),
                          ("BoJ", _BOJ_DT), ("BoC", _BOC_DT),
                          ("BoE", _BOE_DT), ("RBA", _RBA_DT),
                          ("RBNZ", _RBNZ_DT)):
        if not dates:
            stale.append(f"{label}: empty")
            continue
        last = max(dates)
        days_left = (last - now).days
        if days_left < min_lookahead_days:
            stale.append(
                f"{label}: last known {last.date()} ({days_left} days away)"
            )
    if not stale:
        return None
    return (
        "News-blackout calendar is nearing end of coverage — refresh "
        "FOMC/ECB/BoJ/BoC/BoE/RBA/RBNZ dates in calendar_features.py: " + "; ".join(stale)
    )


# =========================================================================
# Per-symbol central-bank routing
# =========================================================================
# XAUUSD is exempt — gold tends to move on any rate decision and often
# benefits from both hawkish (inflation hedge demand) and dovish (lower
# opportunity cost) surprises.

SYMBOL_CENTRAL_BANKS: dict[str, list[str]] = {
    "XAUUSD":  [],                    # exempt
    "XAU/USD": [],
    "USDJPY":  ["FOMC", "BoJ"],
    "USD/JPY": ["FOMC", "BoJ"],
    "EURUSD":  ["FOMC", "ECB"],
    "EUR/USD": ["FOMC", "ECB"],
    "USDCAD":  ["FOMC", "BoC"],
    "USD/CAD": ["FOMC", "BoC"],
    "BTCUSD":  ["FOMC"],              # retained for completeness
    "BTC/USD": ["FOMC"],
    # Forex Phase 1 — 6 new pairs (2026-04-24). Crosses (EURGBP, EURJPY,
    # GBPJPY, AUDNZD) route to BOTH of their component banks; USD is not
    # in the cross's quote either side, but an FOMC decision still moves
    # USD-adjacent flows, so USD pairs retain FOMC even when USD is quote.
    "GBPUSD":  ["FOMC", "BoE"],
    "GBP/USD": ["FOMC", "BoE"],
    "AUDUSD":  ["FOMC", "RBA"],
    "AUD/USD": ["FOMC", "RBA"],
    "EURGBP":  ["ECB", "BoE"],
    "EUR/GBP": ["ECB", "BoE"],
    "EURJPY":  ["ECB", "BoJ"],
    "EUR/JPY": ["ECB", "BoJ"],
    "GBPJPY":  ["BoE", "BoJ"],
    "GBP/JPY": ["BoE", "BoJ"],
    "AUDNZD":  ["RBA", "RBNZ"],
    "AUD/NZD": ["RBA", "RBNZ"],
}

_CB_DATE_MAP = {
    "FOMC": _FOMC_DT,
    "ECB":  _ECB_DT,
    "BoJ":  _BOJ_DT,
    "BoC":  _BOC_DT,
    "BoE":  _BOE_DT,
    "RBA":  _RBA_DT,
    "RBNZ": _RBNZ_DT,
}


# =========================================================================
# Actual announcement times (local, DST-aware via zoneinfo)
# =========================================================================
# Used by the dashboard news-blackout route so the T+2h window ends at the
# correct wall-clock time. Training/backtest math still anchors on 12:00
# UTC (see `_min_hours_to_event`) for calibration consistency — do not
# route training through these values without a full revalidation.

CB_ANNOUNCE_LOCAL: dict[str, tuple[str, int, int]] = {
    "FOMC": ("America/New_York",  14, 0),    # 2:00 PM ET   (rate decision)
    "ECB":  ("Europe/Berlin",     14, 15),   # 2:15 PM CET  (same zone as Frankfurt; Berlin ships in tzdata)
    "BoJ":  ("Asia/Tokyo",        12, 0),    # ~noon Tokyo  (varies; approximate)
    "BoC":  ("America/Toronto",    9, 45),   # 9:45 AM ET   (rate announcement)
    "BoE":  ("Europe/London",     12, 0),    # 12:00 noon UK time
    "RBA":  ("Australia/Sydney",  14, 30),   # 2:30 PM AEST (second day of 2-day meeting)
    "RBNZ": ("Pacific/Auckland",  14, 0),    # 2:00 PM NZT
}


def cb_announce_utc(cb: str, date: datetime) -> datetime:
    """
    Return the actual announcement time in UTC for a given central bank
    and meeting date, honoring each CB's local timezone and DST.

    ``date`` supplies the calendar day; the time-of-day fields are
    overwritten with the CB's scheduled local announcement time.
    """
    tzname, hour, minute = CB_ANNOUNCE_LOCAL[cb]
    local = date.replace(
        hour=hour, minute=minute, second=0, microsecond=0,
        tzinfo=ZoneInfo(tzname),
    )
    return local.astimezone(timezone.utc)


# =========================================================================
# Smart news blackout windows
# =========================================================================
# Window convention (hours relative to announcement time T):
#
#    [T-24h ... T-2h]   HARD BLOCK — pre-positioning risk, avoid new trades
#    [T-2h  ... T+2h]   HARD BLOCK — spike zone, spreads widen, slippage
#    [T+2h  ... T+48h]  ALLOW       — continuation window (the edge)
#
# Rationale: retail execution during the spike is dominated by widened
# spreads and stop slippage. The real edge is in the post-spike
# continuation trend, which our LSTM can detect just like any other move.

BLACKOUT_PRE_HOURS = 24.0
BLACKOUT_POST_HOURS = 2.0


def _min_hours_to_event(
    dt: datetime, events: list[datetime],
) -> tuple[Optional[float], Optional[datetime]]:
    """
    Return (hours, event_dt) for the event closest to ``dt`` in absolute
    time. ``hours`` is signed — negative before the event, positive after.

    Returns (None, None) if no events in the list.
    """
    if not events:
        return None, None
    best_abs = float("inf")
    best_signed = 0.0
    best_event = events[0]
    dt_naive = dt.replace(tzinfo=None) if dt.tzinfo else dt
    for event in events:
        # Treat central-bank announcement time as midday (12:00 UTC) for
        # the purpose of window math. Exact announcement time varies
        # slightly per bank (FOMC 18:00 UTC, ECB 12:15 UTC, BoJ ~03:00 UTC,
        # BoC 14:00 UTC) but ±a few hours is noise vs a 24h pre-window.
        event_mid = event.replace(hour=12, minute=0, second=0, microsecond=0)
        signed_hours = (dt_naive - event_mid).total_seconds() / 3600.0
        if abs(signed_hours) < best_abs:
            best_abs = abs(signed_hours)
            best_signed = signed_hours
            best_event = event_mid
    return best_signed, best_event


def is_in_news_blackout(symbol: str, dt: datetime) -> bool:
    """
    Return True iff ``dt`` falls inside the hard-blackout window for any
    central bank that affects ``symbol``.

    Hard window: T-24h to T+2h (pre-positioning + spike zone).
    Post-news T+2h to T+48h is NOT blocked — the continuation trend is
    where the edge lives.

    XAUUSD is always allowed (no central-bank filter).

    Args:
        symbol: Trading symbol (e.g., "USDJPY", "EURUSD")
        dt:     Bar timestamp (naive UTC; tz-aware OK — normalized internally)

    Returns:
        True if new entries should be blocked for ``symbol`` at ``dt``.
    """
    cbs = SYMBOL_CENTRAL_BANKS.get(symbol.upper(), [])
    if not cbs:
        return False

    for cb in cbs:
        events = _CB_DATE_MAP.get(cb, [])
        signed_hours, _ = _min_hours_to_event(dt, events)
        if signed_hours is None:
            continue
        # Hard window: [-24h, +2h]
        if -BLACKOUT_PRE_HOURS <= signed_hours <= BLACKOUT_POST_HOURS:
            return True
    return False


def is_in_legacy_blackout(symbol: str, dt: datetime,
                           window_hours: float = 24.0) -> bool:
    """
    Legacy symmetric blackout window (±N hours). Preserved for comparison
    with earlier Phase A.6 backtests that used ``delta_hours < 24``.
    """
    cbs = SYMBOL_CENTRAL_BANKS.get(symbol.upper(), [])
    if not cbs:
        return False
    for cb in cbs:
        events = _CB_DATE_MAP.get(cb, [])
        signed_hours, _ = _min_hours_to_event(dt, events)
        if signed_hours is None:
            continue
        if abs(signed_hours) < window_hours:
            return True
    return False


def describe_news_context(symbol: str, dt: datetime) -> dict:
    """
    Return a dict summarizing news state for a given symbol/time.

    Returns:
        {
            "blackout":       bool,
            "post_news":      bool,   # True if in [+2h, +48h] window
            "nearest_cb":     str | None,
            "nearest_hours":  float | None,  # signed hours (neg = before)
        }

    Useful for logging / alerts / debugging.
    """
    cbs = SYMBOL_CENTRAL_BANKS.get(symbol.upper(), [])
    result = {
        "blackout": False, "post_news": False,
        "nearest_cb": None, "nearest_hours": None,
    }
    best_abs = float("inf")
    for cb in cbs:
        events = _CB_DATE_MAP.get(cb, [])
        signed_hours, _ = _min_hours_to_event(dt, events)
        if signed_hours is None:
            continue
        if abs(signed_hours) < best_abs:
            best_abs = abs(signed_hours)
            result["nearest_cb"] = cb
            result["nearest_hours"] = signed_hours
        if -BLACKOUT_PRE_HOURS <= signed_hours <= BLACKOUT_POST_HOURS:
            result["blackout"] = True
        elif BLACKOUT_POST_HOURS < signed_hours <= 48.0:
            result["post_news"] = True
    return result


# =========================================================================
# Calendar features (unchanged — used by training pipeline)
# =========================================================================

class CalendarFeatureBuilder:
    """
    Computes temporal/calendar features from bar timestamps.

    Usage:
        builder = CalendarFeatureBuilder()
        features = builder.get_calendar_features(bar_timestamp)
    """

    def get_calendar_features(self, bar_timestamp: datetime) -> dict[str, float]:
        """Compute all calendar features for a single bar timestamp."""
        features: dict[str, float] = {}

        # Time-of-day cyclical encoding (24h period)
        hour = bar_timestamp.hour + bar_timestamp.minute / 60
        features["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
        features["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))

        # Day-of-week cyclical encoding (7-day period, Mon=0)
        dow = bar_timestamp.weekday()
        features["dow_sin"] = float(np.sin(2 * np.pi * dow / 7))
        features["dow_cos"] = float(np.cos(2 * np.pi * dow / 7))

        # Month-of-year cyclical encoding (12-month period)
        month = bar_timestamp.month
        features["month_sin"] = float(np.sin(2 * np.pi * month / 12))
        features["month_cos"] = float(np.cos(2 * np.pi * month / 12))

        # Trading session flags (UTC)
        features["is_london_session"] = 1.0 if 7 <= bar_timestamp.hour < 16 else 0.0
        features["is_ny_session"] = 1.0 if 13 <= bar_timestamp.hour < 22 else 0.0

        # NFP week flag (Non-Farm Payrolls — first Friday of each month)
        features["is_nfp_week"] = float(self._is_nfp_week(bar_timestamp))

        # Days to next FOMC meeting (normalized to [0, 1] over 8-week cycle)
        features["days_to_fomc"] = self._days_to_fomc(bar_timestamp)

        return features

    @staticmethod
    def _is_nfp_week(dt: datetime) -> bool:
        """Return True if date falls in the same ISO week as the first Friday of the month."""
        first_day = dt.replace(day=1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        return dt.isocalendar()[1] == first_friday.isocalendar()[1]

    @staticmethod
    def _days_to_fomc(dt: datetime) -> float:
        """Return normalized days to next FOMC meeting (0=day-of, 1=~56 days away)."""
        for fomc_dt in _FOMC_DT:
            if fomc_dt >= dt:
                delta = (fomc_dt - dt).days
                return float(min(delta / 56, 1.0))
        return 0.5

    def get_historical_calendar_features(
        self, timestamps: "pd.DatetimeIndex"
    ) -> "pd.DataFrame":
        """
        Compute calendar features for every bar timestamp. Pure computation,
        no API calls — works for any date.
        """
        import pandas as pd

        rows = []
        for ts in timestamps:
            dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            rows.append(self.get_calendar_features(dt))
        return pd.DataFrame(rows, index=timestamps)
