"""Tests for src/safety/invariants.py registry and dedup behavior."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.safety.invariants import InvariantRegistry, Severity


@pytest.fixture
def reg(tmp_path):
    jsonl = tmp_path / "invariants.jsonl"
    halt = tmp_path / "HALT.flag"
    return InvariantRegistry(
        telegram_send=None,
        jsonl_path=jsonl,
        halt_flag=halt,
    )


def test_passing_check_returns_true_and_does_not_record(reg):
    assert reg.check("x.ok", True) is True
    assert reg.recent() == []


def test_failing_check_records_and_returns_false(reg):
    assert reg.check("x.bad", False, message="boom") is False
    recent = reg.recent()
    assert len(recent) == 1
    assert recent[0].invariant == "x.bad"
    assert recent[0].passed is False
    assert recent[0].message == "boom"


def test_jsonl_is_appended(reg, tmp_path):
    reg.check("x.bad", False)
    reg.check("x.bad", False)
    path: Path = reg._jsonl_path
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def _isolated_registry(tmp_path, telegram_send=None) -> InvariantRegistry:
    """Factory that always pins jsonl + halt_flag to tmp_path so tests
    never pollute data/logs/invariants.jsonl."""
    return InvariantRegistry(
        telegram_send=telegram_send,
        jsonl_path=tmp_path / "invariants.jsonl",
        halt_flag=tmp_path / "HALT.flag",
    )


def test_telegram_fires_once_per_dedup_window(tmp_path):
    sent: list[str] = []
    reg = _isolated_registry(tmp_path, telegram_send=sent.append)
    reg.check("x.bad", False, severity=Severity.ALERT, symbol="XAUUSD")
    reg.check("x.bad", False, severity=Severity.ALERT, symbol="XAUUSD")
    reg.check("x.bad", False, severity=Severity.ALERT, symbol="XAUUSD")
    assert len(sent) == 1, "dedup should suppress duplicates"


def test_different_symbols_fire_separately(tmp_path):
    sent: list[str] = []
    reg = _isolated_registry(tmp_path, telegram_send=sent.append)
    reg.check("x.bad", False, severity=Severity.ALERT, symbol="XAUUSD")
    reg.check("x.bad", False, severity=Severity.ALERT, symbol="EURUSD")
    assert len(sent) == 2


def test_warn_never_fires_telegram(tmp_path):
    sent: list[str] = []
    reg = _isolated_registry(tmp_path, telegram_send=sent.append)
    for _ in range(5):
        reg.check("x.noisy", False, severity=Severity.WARN, symbol="X")
    assert sent == []


def test_critical_writes_halt_flag(tmp_path):
    halt = tmp_path / "HALT.flag"
    reg = InvariantRegistry(
        telegram_send=lambda _: None,
        jsonl_path=tmp_path / "i.jsonl",
        halt_flag=halt,
    )
    reg.check("x.critical", False, severity=Severity.CRITICAL)
    assert halt.exists()
    assert "x.critical" in halt.read_text(encoding="utf-8")


def test_recent_respects_limit_and_severity_filter(reg):
    reg.check("a", False, severity=Severity.WARN)
    reg.check("b", False, severity=Severity.ALERT)
    reg.check("c", False, severity=Severity.WARN)
    assert len(reg.recent(limit=2)) == 2
    alerts = reg.recent(severity=Severity.ALERT)
    assert len(alerts) == 1 and alerts[0].invariant == "b"


def test_jsonl_rotates_when_oversized(tmp_path):
    reg = _isolated_registry(tmp_path)
    reg.ROTATE_AT_BYTES = 256  # tiny threshold for test
    for i in range(50):
        reg.check(f"x.{i}", False, severity=Severity.WARN)
    archive = reg._jsonl_path.with_suffix(reg._jsonl_path.suffix + ".1")
    assert archive.exists(), "archive file should be created on rotation"
    assert reg._jsonl_path.stat().st_size < reg.ROTATE_AT_BYTES + 200


def test_registry_never_raises_on_bad_telegram(tmp_path):
    def boom(_):
        raise RuntimeError("telegram down")
    reg = _isolated_registry(tmp_path, telegram_send=boom)
    # Must not propagate — trading loop depends on this.
    reg.check("x", False, severity=Severity.ALERT)
    assert len(reg.recent()) == 1
