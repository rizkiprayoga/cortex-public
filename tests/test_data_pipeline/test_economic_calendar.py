"""Tests for src/data_pipeline/market/economic_calendar.py."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent

import pytest

from src.data_pipeline.market import economic_calendar as _ec


@pytest.fixture
def tmp_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "calendar.yaml"
    path.write_text(dedent("""
        events:
          - name: "FOMC Rate Decision"
            event_utc: "2026-04-29T18:00:00Z"
            tier: 1
            affects: [EURUSD, USDJPY, USDCAD, XAUUSD]
          - name: "Canada CPI YoY"
            event_utc: "2026-04-20T12:30:00Z"
            tier: 1
            affects: [USDCAD]
          - name: "US Retail Sales"
            event_utc: "2026-04-16T12:30:00Z"
            tier: 2
            affects: [EURUSD, USDJPY, USDCAD]
    """).strip(), encoding="utf-8")
    # Reset cache so each test re-parses its fixture.
    _ec._CACHED_PATH = None
    _ec._CACHED_MTIME = None
    _ec._CACHED_EVENTS = []
    return path


def test_load_events_parses_yaml(tmp_yaml):
    evs = _ec.load_events(tmp_yaml)
    assert len(evs) == 3
    names = [e.name for e in evs]
    # Sorted chronologically
    assert names == ["US Retail Sales", "Canada CPI YoY", "FOMC Rate Decision"]
    assert evs[0].tier == 2
    assert evs[1].affects == ("USDCAD",)


def test_active_blackout_hits_tier_1(tmp_yaml, monkeypatch):
    monkeypatch.setattr(_ec, "_DEFAULT_YAML_PATH", tmp_yaml)
    _ec._CACHED_PATH = None
    # Inside the CA CPI window (T-24h to T+2h)
    dt = datetime(2026, 4, 20, 13, 0, tzinfo=timezone.utc)
    active = _ec.active_blackout("USDCAD", dt)
    assert active is not None
    assert active.name == "Canada CPI YoY"


def test_tier_2_event_does_not_trigger_blackout(tmp_yaml, monkeypatch):
    monkeypatch.setattr(_ec, "_DEFAULT_YAML_PATH", tmp_yaml)
    _ec._CACHED_PATH = None
    # Right on top of US Retail Sales (Tier 2)
    dt = datetime(2026, 4, 16, 12, 30, tzinfo=timezone.utc)
    assert _ec.active_blackout("EURUSD", dt) is None
    assert _ec.is_in_blackout("EURUSD", dt) is False


def test_next_blackout_returns_upcoming_tier_1(tmp_yaml, monkeypatch):
    monkeypatch.setattr(_ec, "_DEFAULT_YAML_PATH", tmp_yaml)
    _ec._CACHED_PATH = None
    dt = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    nxt = _ec.next_blackout("USDCAD", dt)
    assert nxt is not None
    assert nxt.name == "Canada CPI YoY"


def test_events_for_symbol_filters_and_tiers(tmp_yaml, monkeypatch):
    monkeypatch.setattr(_ec, "_DEFAULT_YAML_PATH", tmp_yaml)
    _ec._CACHED_PATH = None
    start = datetime(2026, 4, 1, tzinfo=timezone.utc)
    end = datetime(2026, 5, 1, tzinfo=timezone.utc)
    all_usdcad = _ec.events_for_symbol("USDCAD", start, end)
    assert len(all_usdcad) == 3
    tier1_only = _ec.events_for_symbol("USDCAD", start, end, tiers=(1,))
    assert len(tier1_only) == 2
    assert all(e.tier == 1 for e in tier1_only)


def test_nearest_event_signed_hours(tmp_yaml, monkeypatch):
    monkeypatch.setattr(_ec, "_DEFAULT_YAML_PATH", tmp_yaml)
    _ec._CACHED_PATH = None
    dt = datetime(2026, 4, 20, 14, 0, tzinfo=timezone.utc)  # +1.5h after CPI
    result = _ec.nearest_event("USDCAD", dt, tiers=(1,))
    assert result is not None
    e, hours = result
    assert e.name == "Canada CPI YoY"
    assert 1.4 < hours < 1.6


def test_describe_blackout_context_shape(tmp_yaml, monkeypatch):
    monkeypatch.setattr(_ec, "_DEFAULT_YAML_PATH", tmp_yaml)
    _ec._CACHED_PATH = None
    dt = datetime(2026, 4, 20, 13, 0, tzinfo=timezone.utc)
    ctx = _ec.describe_blackout_context("USDCAD", dt)
    assert ctx["blackout"] is True
    assert ctx["active_event"] == "Canada CPI YoY"
    assert ctx["nearest_event"] == "Canada CPI YoY"
    assert isinstance(ctx["nearest_hours"], float)


def test_naive_datetime_treated_as_utc(tmp_yaml, monkeypatch):
    monkeypatch.setattr(_ec, "_DEFAULT_YAML_PATH", tmp_yaml)
    _ec._CACHED_PATH = None
    naive = datetime(2026, 4, 20, 13, 0)
    assert _ec.is_in_blackout("USDCAD", naive) is True


def test_missing_yaml_returns_empty_list(tmp_path):
    _ec._CACHED_PATH = None
    assert _ec.load_events(tmp_path / "missing.yaml") == []
