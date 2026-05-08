"""
economic_calendar.py — Reads ``config/economic_calendar.yaml`` and exposes
structured event queries for the live dashboard, news-blackout check, and
post-hoc impact analysis.

YAML schema (see ``scripts/build_economic_calendar.py`` for the generator):

    events:
      - name: "BoC Rate Decision"
        event_utc: "2026-04-15T13:45:00Z"
        tier: 1                      # 1 blocks trades; 2/3 tracked only
        affects: [USDCAD]            # symbol-routing for blackout + display

Tiering policy (2026-04-15 decision):
  Tier 1 → triggers news_blackout window [T-24h, T+2h]. High confidence,
           observable spread-widening risk (CB rate decisions, NFP, US CPI,
           Canada CPI, Canada Employment, FOMC Minutes).
  Tier 2 → tracked for display + impact analysis. No trade block.
           (EZ flash CPI, JP CPI, US Core PCE, Retail Sales, ECB minutes,
           BoJ summary, US GDP advance.)
  Tier 3 → not auto-populated. Add manually if a live trade is impacted.

Public API:

    load_events()                              -> list[EconomicEvent]
    events_for_symbol(symbol, start, end)      -> list[EconomicEvent]
    is_in_blackout(symbol, dt)                 -> bool
    active_blackout(symbol, dt)                -> Optional[EconomicEvent]
    next_blackout(symbol, dt)                  -> Optional[EconomicEvent]
    nearest_event(symbol, dt)                  -> Optional[(event, signed_hours)]
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import yaml

# Blackout window (hours relative to event).
BLACKOUT_PRE_HOURS = 24.0
BLACKOUT_POST_HOURS = 2.0

# Only this tier triggers news_blackout.
BLACKOUT_TIER = 1

# Default location; overridable for tests.
_DEFAULT_YAML_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "economic_calendar.yaml"
)

_CACHE_LOCK = threading.Lock()
_CACHED_PATH: Optional[Path] = None
_CACHED_MTIME: Optional[float] = None
_CACHED_EVENTS: list["EconomicEvent"] = []


@dataclass(frozen=True)
class EconomicEvent:
    name: str
    event_utc: datetime
    tier: int
    affects: tuple[str, ...] = field(default_factory=tuple)

    @property
    def blackout_start(self) -> datetime:
        return self.event_utc - timedelta(hours=BLACKOUT_PRE_HOURS)

    @property
    def blackout_end(self) -> datetime:
        return self.event_utc + timedelta(hours=BLACKOUT_POST_HOURS)

    @property
    def is_blackout_tier(self) -> bool:
        return self.tier == BLACKOUT_TIER


def _parse_event(row: dict) -> EconomicEvent:
    raw_ts = row["event_utc"]
    if isinstance(raw_ts, datetime):
        dt = raw_ts
    else:
        # PyYAML may return a str for "2026-04-15T13:45:00Z"; fromisoformat
        # doesn't accept trailing Z before 3.11 — normalize defensively.
        s = str(raw_ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    affects_raw = row.get("affects") or []
    return EconomicEvent(
        name=str(row["name"]),
        event_utc=dt,
        tier=int(row["tier"]),
        affects=tuple(str(s).upper() for s in affects_raw),
    )


def load_events(path: Optional[Path] = None) -> list[EconomicEvent]:
    """Load events from YAML. Cached by (path, mtime) so in-process callers
    pay the parse cost once per file edit."""
    global _CACHED_PATH, _CACHED_MTIME, _CACHED_EVENTS
    target = path or _DEFAULT_YAML_PATH
    try:
        mtime = target.stat().st_mtime
    except FileNotFoundError:
        return []
    with _CACHE_LOCK:
        if (
            _CACHED_PATH == target
            and _CACHED_MTIME == mtime
            and _CACHED_EVENTS
        ):
            return list(_CACHED_EVENTS)
        with target.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        rows = doc.get("events") or []
        events = [_parse_event(r) for r in rows]
        events.sort(key=lambda e: e.event_utc)
        _CACHED_PATH = target
        _CACHED_MTIME = mtime
        _CACHED_EVENTS = events
        return list(events)


def events_for_symbol(
    symbol: str,
    start: datetime,
    end: datetime,
    tiers: Optional[tuple[int, ...]] = None,
) -> list[EconomicEvent]:
    """Events affecting ``symbol`` in [start, end] (inclusive)."""
    sym = symbol.upper()
    evs = load_events()
    out: list[EconomicEvent] = []
    for e in evs:
        if e.event_utc < start or e.event_utc > end:
            continue
        if sym not in e.affects:
            continue
        if tiers is not None and e.tier not in tiers:
            continue
        out.append(e)
    return out


def active_blackout(symbol: str, dt: datetime) -> Optional[EconomicEvent]:
    """Return the currently-active Tier 1 event for ``symbol`` at ``dt``,
    or None."""
    sym = symbol.upper()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    for e in load_events():
        if e.tier != BLACKOUT_TIER or sym not in e.affects:
            continue
        if e.blackout_start <= dt <= e.blackout_end:
            return e
    return None


def next_blackout(symbol: str, dt: datetime) -> Optional[EconomicEvent]:
    """Return the next upcoming Tier 1 event for ``symbol`` after ``dt``."""
    sym = symbol.upper()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    for e in load_events():
        if e.tier != BLACKOUT_TIER or sym not in e.affects:
            continue
        if dt < e.blackout_start:
            return e
    return None


def is_in_blackout(symbol: str, dt: datetime) -> bool:
    return active_blackout(symbol, dt) is not None


def nearest_event(
    symbol: str,
    dt: datetime,
    tiers: Optional[tuple[int, ...]] = None,
) -> Optional[tuple[EconomicEvent, float]]:
    """Return (event, signed_hours_to_event) for the event nearest ``dt``
    that affects ``symbol``.

    Sign convention: ``signed_hours = (dt - event_utc) / 3600``, so
    **positive** = event is in the past, **negative** = event is in the
    future. Live dashboards display "-121h" for "CPI in 5 days."
    """
    sym = symbol.upper()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    best: Optional[tuple[EconomicEvent, float]] = None
    for e in load_events():
        if sym not in e.affects:
            continue
        if tiers is not None and e.tier not in tiers:
            continue
        delta_h = (dt - e.event_utc).total_seconds() / 3600.0
        if best is None or abs(delta_h) < abs(best[1]):
            best = (e, delta_h)
    return best


def describe_blackout_context(symbol: str, dt: datetime) -> dict:
    """Summary dict used by main.py signal-audit log (drop-in for the
    legacy ``describe_news_context``)."""
    active = active_blackout(symbol, dt)
    near = nearest_event(symbol, dt, tiers=(BLACKOUT_TIER,))
    return {
        "blackout": active is not None,
        "active_event": active.name if active else None,
        "nearest_event": near[0].name if near else None,
        "nearest_hours": near[1] if near else None,
    }
