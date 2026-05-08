"""
build_economic_calendar.py — Generate config/economic_calendar.yaml from rules.

Emits one row per event for 2026 with:
  - name          (e.g. "FOMC Rate Decision", "US NFP")
  - event_utc     (ISO-8601 UTC, DST-correct via zoneinfo)
  - tier          (1 | 2 | 3 — only tier 1 triggers news blackout)
  - affects       (list of symbols — [] means no symbol routing)

Rerun this whenever central-bank calendars roll forward or a release
date correction lands. Verify against BLS / BoC / Stats Canada / ECB
official schedules before publishing — some 2026 non-CB dates below are
heuristics (1st Friday for NFP, 2nd week for US CPI) rather than
confirmed government schedules.

Tiering rationale: see docs/INVARIANTS.md + the session where we decided
to block only Tier 1 and passively track Tier 2/3 for impact analysis.
"""

from __future__ import annotations

import calendar as _cal
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

# Reuse the canonical CB rate-decision date lists so this file stays
# single-sourced with the training-pipeline blackout math.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_pipeline.market.calendar_features import (  # noqa: E402
    FOMC_DATES, ECB_DATES, BOJ_DATES, BOC_DATES,
    BOE_DATES, RBA_DATES, RBNZ_DATES,
)

OUT_PATH = Path(__file__).resolve().parents[1] / "config" / "economic_calendar.yaml"

UTC = ZoneInfo("UTC")

# Hand-verified overrides for events whose heuristic date is wrong for
# a specific month. Key = (name, year, month) -> date(y, m, d).
# Add rows here as you confirm official release dates from BLS / Stats
# Canada / Eurostat / BoJ / MoF JP. Regenerate the YAML afterward:
#     python scripts/build_economic_calendar.py
MANUAL_DATE_OVERRIDES: dict[tuple[str, int, int], date] = {
    # Stats Canada Apr 2026 CPI release — TradingView confirmed Apr 20 @ 12:30 UTC.
    ("Canada CPI YoY", 2026, 4): date(2026, 4, 20),
}

# ---------------------------------------------------------------------------
# Symbol groups — keep aligned with calendar_features.SYMBOL_CENTRAL_BANKS.
# Each pair appears in every currency's group it exposes.
#
# Updated 2026-04-29 for the universe sweep sprint winner universe (10 pairs) PLUS the
# remaining Phase 1 / an earlier sprint pairs that may re-enter selection later.
# A pair is included whether or not it's currently enabled — the news API
# filter (routes/news.py::LIVE_SYMBOLS) decides what to surface.
# ---------------------------------------------------------------------------
USD_PAIRS = [
    "EURUSD", "USDJPY", "USDCAD", "GBPUSD", "AUDUSD",
    "USDCHF", "NZDUSD",  # added 2026-04-29
]
USD_PAIRS_PLUS_XAU = USD_PAIRS + ["XAUUSD"]

EUR_PAIRS = [
    "EURUSD", "EURGBP", "EURJPY",
    "EURAUD", "EURCHF",  # added 2026-04-29
]
JPY_PAIRS = [
    "USDJPY", "EURJPY", "GBPJPY",
    "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",  # added 2026-04-29 (4 cross-JPY)
]
CAD_PAIRS = [
    "USDCAD",
    "CADJPY",  # added 2026-04-29
]
GBP_PAIRS = [
    "GBPUSD", "EURGBP", "GBPJPY",
    "GBPCHF", "GBPAUD",  # added 2026-04-29
]
AUD_PAIRS = [
    "AUDUSD", "AUDNZD",
    "AUDJPY", "EURAUD", "GBPAUD",  # added 2026-04-29
]
NZD_PAIRS = [
    "AUDNZD",
    "NZDUSD", "NZDJPY",  # added 2026-04-29
]
# CHF group is NEW — was missing entirely (no SNB events fired before).
CHF_PAIRS = [
    "USDCHF", "EURCHF", "GBPCHF", "CHFJPY",
]

# Legacy aliases retained for backward compat with inline callers below.
USD_ALL = USD_PAIRS
USD_ALL_PLUS_XAU = USD_PAIRS_PLUS_XAU


@dataclass
class EventRow:
    name: str
    event_utc: datetime
    tier: int
    affects: list[str]


def _local_to_utc(d: date, local_tz: str, hour: int, minute: int) -> datetime:
    """Convert a local wall-clock datetime to UTC, DST-aware."""
    local = datetime(d.year, d.month, d.day, hour, minute, tzinfo=ZoneInfo(local_tz))
    return local.astimezone(UTC)


def _parse_dates(date_strs: list[str]) -> list[date]:
    return [datetime.strptime(s, "%Y-%m-%d").date() for s in date_strs]


def _first_friday(year: int, month: int) -> date:
    _, first_weekday, days_in_month = (None, *_cal.monthrange(year, month))
    # monthrange returns (weekday_of_day_1, days_in_month)
    first_dow, _ = _cal.monthrange(year, month)
    # weekday(): Mon=0 .. Sun=6; Friday=4
    offset = (4 - first_dow) % 7
    return date(year, month, 1 + offset)


def _cb_events(
    name: str,
    dates: list[str],
    local_tz: str,
    hour: int,
    minute: int,
    tier: int,
    affects: list[str],
) -> list[EventRow]:
    out: list[EventRow] = []
    for d in _parse_dates(dates):
        if d.year != 2026:
            continue
        out.append(EventRow(
            name=name,
            event_utc=_local_to_utc(d, local_tz, hour, minute),
            tier=tier,
            affects=affects,
        ))
    return out


def _monthly_events(
    name: str,
    year: int,
    day_of_month_fn: Callable[[int, int], date],
    local_tz: str,
    hour: int,
    minute: int,
    tier: int,
    affects: list[str],
    months: Optional[list[int]] = None,
) -> list[EventRow]:
    out: list[EventRow] = []
    for m in (months or range(1, 13)):
        d = MANUAL_DATE_OVERRIDES.get((name, year, m)) or day_of_month_fn(year, m)
        out.append(EventRow(
            name=name,
            event_utc=_local_to_utc(d, local_tz, hour, minute),
            tier=tier,
            affects=affects,
        ))
    return out


def build_2026() -> list[EventRow]:
    rows: list[EventRow] = []

    # -----------------------------------------------------------------
    # TIER 1 — blackout-eligible. Historically the biggest FX movers.
    # -----------------------------------------------------------------

    # Central-bank rate decisions (dates pulled from calendar_features.py;
    # official announcement times per bank, DST-aware). Each CB affects
    # every pair that exposes its currency, including crosses.
    rows += _cb_events("FOMC Rate Decision", FOMC_DATES,
                        "America/New_York", 14,  0, 1, USD_PAIRS_PLUS_XAU)
    rows += _cb_events("ECB Rate Decision",  ECB_DATES,
                        "Europe/Berlin",    14, 15, 1, EUR_PAIRS)
    rows += _cb_events("BoJ Rate Decision",  BOJ_DATES,
                        "Asia/Tokyo",       12,  0, 1, JPY_PAIRS)
    rows += _cb_events("BoC Rate Decision",  BOC_DATES,
                        "America/Toronto",   9, 45, 1, CAD_PAIRS)
    rows += _cb_events("BoE Rate Decision",  BOE_DATES,
                        "Europe/London",    12,  0, 1, GBP_PAIRS)
    rows += _cb_events("RBA Rate Decision",  RBA_DATES,
                        "Australia/Sydney", 14, 30, 1, AUD_PAIRS)
    rows += _cb_events("RBNZ OCR Decision",  RBNZ_DATES,
                        "Pacific/Auckland", 14,  0, 1, NZD_PAIRS)

    # SNB Monetary Policy Assessment — quarterly (Mar / Jun / Sep / Dec),
    # Thursday at 09:30 CET. 2026 dates per SNB published schedule (verify
    # at https://www.snb.ch/en/iabout/monpol/id/monpol_current).
    # Tier 1 — SNB surprise moves (e.g. 2015 floor removal) move CHF 5%+.
    SNB_DATES_2026 = [
        "2026-03-19",  # Q1 assessment
        "2026-06-18",  # Q2 assessment
        "2026-09-24",  # Q3 assessment
        "2026-12-17",  # Q4 assessment
    ]
    rows += _cb_events("SNB Rate Decision", SNB_DATES_2026,
                        "Europe/Zurich", 9, 30, 1, CHF_PAIRS)

    # Switzerland CPI YoY — released by FSO ~3rd of each month at 8:30 CET.
    # Tier 1 — informs SNB outlook, often moves CHF intraday.
    def _third_weekday_of_month(y: int, m: int) -> date:
        """3rd of the month, rolled to next weekday if it falls on weekend."""
        d = date(y, m, 3)
        while d.weekday() >= 5:
            d = d + timedelta(days=1)
        return d
    rows += _monthly_events(
        "Switzerland CPI YoY", 2026, _third_weekday_of_month,
        "Europe/Zurich", 8, 30, 1, CHF_PAIRS,
    )

    # US NFP — 1st Friday, 8:30 AM ET. Affects all USD pairs + XAU.
    rows += _monthly_events(
        "US Non-Farm Payrolls", 2026, _first_friday,
        "America/New_York", 8, 30, 1, USD_PAIRS_PLUS_XAU,
    )

    # Canada Employment — typically same 1st Friday as NFP, 8:30 AM ET.
    rows += _monthly_events(
        "Canada Employment", 2026, _first_friday,
        "America/Toronto", 8, 30, 1, CAD_PAIRS,
    )

    # US CPI — mid-month, ~8:30 AM ET. Heuristic: 2nd Wednesday.
    # BLS publishes the true schedule; verify before each month.
    def _second_wednesday(y: int, m: int) -> date:
        first_dow, _ = _cal.monthrange(y, m)
        offset = (2 - first_dow) % 7  # Wednesday = 2
        first_wed = 1 + offset
        return date(y, m, first_wed + 7)
    rows += _monthly_events(
        "US CPI YoY", 2026, _second_wednesday,
        "America/New_York", 8, 30, 1, USD_PAIRS_PLUS_XAU,
    )

    # Canada CPI — mid-month, 8:30 AM ET. Heuristic: 3rd Tuesday.
    # Verify against Stats Canada schedule.
    def _third_tuesday(y: int, m: int) -> date:
        first_dow, _ = _cal.monthrange(y, m)
        offset = (1 - first_dow) % 7  # Tuesday = 1
        first_tue = 1 + offset
        return date(y, m, first_tue + 14)
    rows += _monthly_events(
        "Canada CPI YoY", 2026, _third_tuesday,
        "America/Toronto", 8, 30, 1, CAD_PAIRS,
    )

    # UK CPI — ONS publishes ~3rd Wednesday of the month at 7:00 UK time.
    # Tier 1 — sterling often moves 0.5-1% on a surprise print.
    def _third_wednesday(y: int, m: int) -> date:
        first_dow, _ = _cal.monthrange(y, m)
        offset = (2 - first_dow) % 7  # Wednesday = 2
        first_wed = 1 + offset
        return date(y, m, first_wed + 14)
    rows += _monthly_events(
        "UK CPI YoY", 2026, _third_wednesday,
        "Europe/London", 7, 0, 1, GBP_PAIRS,
    )

    # UK Employment (Claimant Count / Unemployment) — ONS releases
    # ~3rd Tuesday, 7:00 UK time. Tier 1 for sterling.
    rows += _monthly_events(
        "UK Employment", 2026, _third_tuesday,
        "Europe/London", 7, 0, 1, GBP_PAIRS,
    )

    # Australia CPI — ABS, quarterly (Jan/Apr/Jul/Oct), 11:30 AEST.
    # Heuristic: 4th Wednesday of the release month.
    def _fourth_wednesday(y: int, m: int) -> date:
        first_dow, _ = _cal.monthrange(y, m)
        offset = (2 - first_dow) % 7
        first_wed = 1 + offset
        return date(y, m, first_wed + 21)
    rows += _monthly_events(
        "Australia CPI YoY", 2026, _fourth_wednesday,
        "Australia/Sydney", 11, 30, 1, AUD_PAIRS,
        months=[1, 4, 7, 10],
    )

    # Australia Employment — ABS monthly, ~3rd Thursday, 11:30 AEST.
    # Tier 1 — jobs print is the single biggest AUD mover outside RBA.
    def _third_thursday_local(y: int, m: int) -> date:
        first_dow, _ = _cal.monthrange(y, m)
        offset = (3 - first_dow) % 7  # Thursday = 3
        first_thu = 1 + offset
        return date(y, m, first_thu + 14)
    rows += _monthly_events(
        "Australia Employment", 2026, _third_thursday_local,
        "Australia/Sydney", 11, 30, 1, AUD_PAIRS,
    )

    # New Zealand CPI — Stats NZ, quarterly in 2026 (moves to monthly in
    # 2027 — see RBNZ schedule notes). Heuristic: ~3rd week of month after
    # quarter-end at 10:45 NZT.
    rows += _monthly_events(
        "NZ CPI YoY", 2026, _third_wednesday,
        "Pacific/Auckland", 10, 45, 1, NZD_PAIRS,
        months=[1, 4, 7, 10],
    )

    # FOMC Minutes — 3 weeks after each meeting, 14:00 ET.
    for d in _parse_dates(FOMC_DATES):
        if d.year != 2026:
            continue
        minutes_date = d + timedelta(days=21)
        rows.append(EventRow(
            name="FOMC Minutes",
            event_utc=_local_to_utc(minutes_date, "America/New_York", 14, 0),
            tier=1,
            affects=USD_ALL_PLUS_XAU,
        ))

    # -----------------------------------------------------------------
    # TIER 2 — tracked for display + post-hoc impact analysis. No blackout.
    # -----------------------------------------------------------------

    # Eurozone flash CPI — monthly, ~11:00 CET, typically end of prior month
    # or first business day. Heuristic: last business day of month.
    def _last_business_day(y: int, m: int) -> date:
        _, last_day = _cal.monthrange(y, m)
        d = date(y, m, last_day)
        while d.weekday() >= 5:
            d = d - timedelta(days=1)
        return d
    rows += _monthly_events(
        "Eurozone Flash CPI", 2026, _last_business_day,
        "Europe/Berlin", 11, 0, 2, EUR_PAIRS,
    )

    # Japan National CPI — monthly, 08:30 JST, usually 3rd Friday.
    def _third_friday(y: int, m: int) -> date:
        first_dow, _ = _cal.monthrange(y, m)
        offset = (4 - first_dow) % 7
        first_fri = 1 + offset
        return date(y, m, first_fri + 14)
    rows += _monthly_events(
        "Japan National CPI", 2026, _third_friday,
        "Asia/Tokyo", 8, 30, 2, JPY_PAIRS,
    )

    # US Core PCE — monthly, 8:30 AM ET, end-of-month (~last Friday).
    def _last_friday(y: int, m: int) -> date:
        _, last_day = _cal.monthrange(y, m)
        d = date(y, m, last_day)
        while d.weekday() != 4:
            d = d - timedelta(days=1)
        return d
    rows += _monthly_events(
        "US Core PCE", 2026, _last_friday,
        "America/New_York", 8, 30, 2, USD_PAIRS_PLUS_XAU,
    )

    # US Retail Sales — monthly, 8:30 AM ET, mid-month Thursday heuristic.
    def _third_thursday(y: int, m: int) -> date:
        first_dow, _ = _cal.monthrange(y, m)
        offset = (3 - first_dow) % 7  # Thursday = 3
        first_thu = 1 + offset
        return date(y, m, first_thu + 14)
    rows += _monthly_events(
        "US Retail Sales", 2026, _third_thursday,
        "America/New_York", 8, 30, 2, USD_PAIRS,
    )

    # ECB Minutes — 4 weeks after each ECB decision, 13:30 CET.
    for d in _parse_dates(ECB_DATES):
        if d.year != 2026:
            continue
        minutes_date = d + timedelta(days=28)
        rows.append(EventRow(
            name="ECB Minutes",
            event_utc=_local_to_utc(minutes_date, "Europe/Berlin", 13, 30),
            tier=2,
            affects=EUR_PAIRS,
        ))

    # BoJ Summary of Opinions — ~8 days after each BoJ meeting, 8:50 JST.
    for d in _parse_dates(BOJ_DATES):
        if d.year != 2026:
            continue
        summary_date = d + timedelta(days=8)
        rows.append(EventRow(
            name="BoJ Summary of Opinions",
            event_utc=_local_to_utc(summary_date, "Asia/Tokyo", 8, 50),
            tier=2,
            affects=JPY_PAIRS,
        ))

    # US GDP advance (quarterly) — ~last Thursday of Jan/Apr/Jul/Oct, 8:30 ET.
    def _last_thursday(y: int, m: int) -> date:
        _, last_day = _cal.monthrange(y, m)
        d = date(y, m, last_day)
        while d.weekday() != 3:
            d = d - timedelta(days=1)
        return d
    rows += _monthly_events(
        "US GDP (advance)", 2026, _last_thursday,
        "America/New_York", 8, 30, 2, USD_PAIRS_PLUS_XAU,
        months=[1, 4, 7, 10],
    )

    # BoE Minutes — released alongside each MPC announcement at 12:00 UK.
    # Already captured by the BoE Rate Decision event, but add a Tier 2
    # stub 3 days before (when any pre-briefing leaks surface historically).
    # Keep light-touch: just one Tier 2 marker per BoE meeting.
    # (No separate release — skip to keep YAML concise.)

    # UK Retail Sales — ONS ~3rd Friday, 7:00 UK. Tier 2.
    rows += _monthly_events(
        "UK Retail Sales", 2026, _third_friday,
        "Europe/London", 7, 0, 2, GBP_PAIRS,
    )

    # UK GDP monthly — ~6 weeks after month-end. Too heuristic to model
    # reliably; skip for now (Tier 2/3 only, add MANUAL_DATE_OVERRIDES if
    # needed).

    # -----------------------------------------------------------------
    # TIER 3 — not populated here. Jobless Claims (weekly), ISM, etc.
    # Add manually if a live trade is hurt by one.
    # -----------------------------------------------------------------

    rows.sort(key=lambda r: r.event_utc)
    return rows


def render_yaml(rows: list[EventRow]) -> str:
    out: list[str] = []
    out.append("# Auto-generated by scripts/build_economic_calendar.py")
    out.append("# Regenerate after central-bank calendars update or when verifying heuristic-dated events.")
    out.append("# Tier 1 events trigger news_blackout. Tier 2/3 are tracked but do not block trades.")
    out.append("")
    out.append("events:")
    for r in rows:
        iso = r.event_utc.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        affects = "[" + ", ".join(r.affects) + "]"
        out.append(f"  - name: \"{r.name}\"")
        out.append(f"    event_utc: \"{iso}\"")
        out.append(f"    tier: {r.tier}")
        out.append(f"    affects: {affects}")
    out.append("")
    return "\n".join(out)


def main() -> None:
    rows = build_2026()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(render_yaml(rows), encoding="utf-8")
    by_tier: dict[int, int] = {}
    for r in rows:
        by_tier[r.tier] = by_tier.get(r.tier, 0) + 1
    print(f"Wrote {len(rows)} events to {OUT_PATH}")
    for t in sorted(by_tier):
        print(f"  tier {t}: {by_tier[t]}")


if __name__ == "__main__":
    main()
