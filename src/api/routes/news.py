"""
routes/news.py — Economic-calendar endpoints for the dashboard.

GET /api/news/blackouts → per-symbol current blackout state + next event.
GET /api/news/events    → all events in a window (all tiers), for the
                           full calendar view on the dashboard.

Delegates to src.data_pipeline.market.economic_calendar, which reads
from config/economic_calendar.yaml. Only Tier 1 events trigger blackouts;
Tier 2/3 are surfaced for display + post-hoc impact analysis.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from src.api.auth import get_current_user
from src.data_pipeline.market import economic_calendar as _ec
from src.data_pipeline.market.economic_calendar import BLACKOUT_TIER, EconomicEvent

router = APIRouter(prefix="/api/news", tags=["news"])

# the universe sweep sprint the trading universe — DEV PREVIEW. Keep in sync with main.py::SYMBOLS and
# config/settings.yaml::trading.symbols.
LIVE_SYMBOLS: tuple[str, ...] = (
    "XAUUSD", "GBPUSD", "USDJPY", "USDCAD", "NZDUSD",
    "USDCHF", "GBPCHF", "EURAUD", "GBPAUD", "EURJPY",
)


class NewsEvent(BaseModel):
    # ``cb`` kept for frontend back-compat; now carries the full event name
    # (e.g. "BoC Rate Decision", "US Non-Farm Payrolls", "Canada CPI YoY").
    cb: str
    event_utc: datetime
    blackout_start_utc: datetime
    blackout_end_utc: datetime
    tier: int = BLACKOUT_TIER


class NewsSymbolEntry(BaseModel):
    symbol: str
    central_banks: list[str]
    state: str
    active_event: Optional[NewsEvent] = None
    next_event: Optional[NewsEvent] = None
    exempt: bool = False


class NewsBlackoutResponse(BaseModel):
    generated_at: datetime
    symbols: list[NewsSymbolEntry]


class CalendarEventDTO(BaseModel):
    name: str
    event_utc: datetime
    tier: int
    affects: list[str]


class EconomicCalendarResponse(BaseModel):
    start: datetime
    end: datetime
    events: list[CalendarEventDTO]
    count: int


def _to_news_event(e: EconomicEvent) -> NewsEvent:
    return NewsEvent(
        cb=e.name,
        event_utc=e.event_utc,
        blackout_start_utc=e.blackout_start,
        blackout_end_utc=e.blackout_end,
        tier=e.tier,
    )


def _symbol_cbs(symbol: str) -> list[str]:
    """Distinct list of affecting events (for display in the symbol card)."""
    s = symbol.upper()
    cbs: list[str] = []
    seen: set[str] = set()
    for ev in _ec.load_events():
        if s in ev.affects and ev.name not in seen:
            cbs.append(ev.name)
            seen.add(ev.name)
    return cbs


def _symbol_entry(symbol: str, now: datetime) -> NewsSymbolEntry:
    s = symbol.upper()
    affecting_any = any(s in ev.affects for ev in _ec.load_events())
    if not affecting_any:
        return NewsSymbolEntry(
            symbol=symbol, central_banks=[], state="clear", exempt=True,
        )

    active = _ec.active_blackout(s, now)
    nxt = _ec.next_blackout(s, now)

    # Post-news window: inside [T+2h, T+48h] of the last Tier 1 event.
    post_news = False
    if active is None:
        for e in _ec.load_events():
            if e.tier != BLACKOUT_TIER or s not in e.affects:
                continue
            post_end = e.event_utc + timedelta(hours=48)
            if e.blackout_end < now <= post_end:
                post_news = True
                break

    state = "blackout" if active else ("post_news" if post_news else "clear")
    return NewsSymbolEntry(
        symbol=symbol,
        central_banks=_symbol_cbs(symbol),
        state=state,
        active_event=_to_news_event(active) if active else None,
        next_event=_to_news_event(nxt) if nxt else None,
    )


@router.get("/blackouts", response_model=NewsBlackoutResponse)
def get_blackouts(_user=Depends(get_current_user)) -> NewsBlackoutResponse:
    """Current blackout state + next upcoming Tier 1 event per symbol."""
    now = datetime.now(tz=timezone.utc)
    return NewsBlackoutResponse(
        generated_at=now,
        symbols=[_symbol_entry(sym, now) for sym in LIVE_SYMBOLS],
    )


@router.get("/events", response_model=EconomicCalendarResponse)
def get_events(
    days: int = Query(30, ge=1, le=180),
    symbol: Optional[str] = Query(None),
    tiers: Optional[str] = Query(None, description="comma-separated tiers e.g. '1,2'"),
    _user: str = Depends(get_current_user),
) -> EconomicCalendarResponse:
    """Events in the next ``days`` window. Optional symbol / tier filter."""
    start = datetime.now(tz=timezone.utc)
    end = start + timedelta(days=days)
    tier_filter = None
    if tiers:
        try:
            tier_filter = tuple(int(t) for t in tiers.split(",") if t.strip())
        except ValueError:
            tier_filter = None

    if symbol:
        evs = _ec.events_for_symbol(symbol, start, end, tiers=tier_filter)
    else:
        evs = [
            e for e in _ec.load_events()
            if start <= e.event_utc <= end
            and (tier_filter is None or e.tier in tier_filter)
        ]

    return EconomicCalendarResponse(
        start=start,
        end=end,
        events=[
            CalendarEventDTO(
                name=e.name, event_utc=e.event_utc,
                tier=e.tier, affects=list(e.affects),
            )
            for e in evs
        ],
        count=len(evs),
    )
