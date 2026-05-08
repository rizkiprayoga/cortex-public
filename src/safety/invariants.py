"""
invariants.py — Live runtime invariant registry.

Cheap assertions sprinkled through the hot paths (OrderManager, ExitManager,
AccountMonitor, news route, data pipeline) that catch silent divergence
between live behavior and the documented/configured rules. Inspired by a
streak of subtle bugs (wrong time-exit bar count, missing commission/swap
on closed deals, news-blackout end anchored on 12:00 UTC instead of the
real announcement time) that unit tests did not catch because they
assert units, not cross-system invariants.

Design
------
    check(name, condition, severity=WARN, symbol=..., context={...})

- Every call writes one JSON line to ``data/logs/invariants.jsonl``.
- WARN     -> log only; surfaces on the dashboard Health card.
- ALERT    -> log + Telegram ONCE per (name, dedup_key) per 24h.
- CRITICAL -> log + Telegram + touches the TRADING_HALTED.flag file so the
              main loop halts on the next tick. Reserved for invariants
              with zero false-positive tolerance. None in v1.

Never raises. Failures in the registry itself are swallowed so a buggy
guard can never crash the trading loop.
"""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger


class Severity(str, Enum):
    WARN = "WARN"
    ALERT = "ALERT"
    CRITICAL = "CRITICAL"


@dataclass
class InvariantFinding:
    ts: str
    invariant: str
    severity: str
    passed: bool
    message: str
    symbol: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, separators=(",", ":"))


class InvariantRegistry:
    JSONL_PATH = Path("data/logs/invariants.jsonl")
    HALT_FLAG = Path("data/logs/TRADING_HALTED.flag")
    DEDUP_WINDOW = timedelta(hours=24)
    MAX_RECENT = 500
    # Size-based rotation: when the live JSONL crosses this threshold, we
    # rename it to ``.jsonl.1`` (overwriting any prior archive) and start
    # a fresh file. Keeps one backup worth of history without unbounded
    # growth — ``trade.near_economic_event`` fires on every close and
    # would otherwise reach 100s of MB over months.
    ROTATE_AT_BYTES = 50 * 1024 * 1024  # 50 MB

    def __init__(
        self,
        telegram_send: Optional[Callable[[str], None]] = None,
        jsonl_path: Optional[Path] = None,
        halt_flag: Optional[Path] = None,
    ):
        self._lock = threading.Lock()
        self._dedup: dict[str, datetime] = {}
        self._recent: deque[InvariantFinding] = deque(maxlen=self.MAX_RECENT)
        self._telegram_send = telegram_send
        self._jsonl_path = jsonl_path or self.JSONL_PATH
        self._halt_flag = halt_flag or self.HALT_FLAG
        # Guard: during pytest, the production JSONL/halt-flag paths are
        # protected. A test that routes a failing invariant to these files
        # would pollute operator telemetry and, for ALERT/CRITICAL, could
        # fire real Telegrams or write TRADING_HALTED.flag mid-suite.
        # tests/conftest.py rebinds the global registry to tmp_path; this
        # catches any bypass (ad-hoc scripts, new test dirs, direct
        # imports) by failing loud instead of silently leaking.
        if os.environ.get("PYTEST_CURRENT_TEST"):
            for label, live, attr in (
                ("JSONL", self._jsonl_path, self.JSONL_PATH),
                ("halt-flag", self._halt_flag, self.HALT_FLAG),
            ):
                try:
                    same = live.resolve() == Path(attr).resolve()
                except OSError:
                    same = str(live) == str(attr)
                if same:
                    raise RuntimeError(
                        f"InvariantRegistry refuses to bind {label} path "
                        f"to production target {live} during pytest. "
                        f"Pass jsonl_path=/halt_flag= explicitly (use "
                        f"tmp_path) or rely on tests/conftest.py to "
                        f"rebind the global registry."
                    )
        try:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("InvariantRegistry: cannot create log dir: {}", exc)

    def check(
        self,
        name: str,
        condition: bool,
        *,
        severity: Severity = Severity.WARN,
        symbol: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        dedup_key: Optional[str] = None,
        message: str = "",
    ) -> bool:
        """Record the outcome of an invariant check. Returns ``condition`` verbatim."""
        try:
            finding = InvariantFinding(
                ts=datetime.now(tz=timezone.utc).isoformat(),
                invariant=name,
                severity=severity.value,
                passed=bool(condition),
                message=message or ("ok" if condition else "violation"),
                symbol=symbol,
                context=dict(context or {}),
            )
            if condition:
                return True
            self._record_failure(finding, severity, dedup_key)
        except Exception as exc:
            logger.exception("InvariantRegistry.check failed for {}: {}", name, exc)
        return bool(condition)

    def _record_failure(
        self,
        finding: InvariantFinding,
        severity: Severity,
        dedup_key: Optional[str],
    ) -> None:
        with self._lock:
            self._recent.append(finding)
            self._append_jsonl(finding)

            log_line = (
                f"INVARIANT {finding.invariant} [{severity.value}] symbol={finding.symbol} "
                f"msg={finding.message} ctx={finding.context}"
            )
            if severity is Severity.WARN:
                logger.warning(log_line)
            else:
                logger.error(log_line)

            if severity is Severity.WARN:
                return

            key = dedup_key or f"{finding.invariant}:{finding.symbol}"
            now = datetime.now(tz=timezone.utc)
            last = self._dedup.get(key)
            if last is not None and now - last < self.DEDUP_WINDOW:
                return
            self._dedup[key] = now

            if self._telegram_send is not None:
                text = (
                    f"INVARIANT {severity.value}: {finding.invariant}\n"
                    f"Symbol: {finding.symbol or '-'}\n"
                    f"Message: {finding.message}\n"
                    f"Context: {finding.context}"
                )
                try:
                    self._telegram_send(text)
                except Exception as exc:
                    logger.warning("Invariant Telegram dispatch failed: {}", exc)

            if severity is Severity.CRITICAL:
                try:
                    self._halt_flag.parent.mkdir(parents=True, exist_ok=True)
                    self._halt_flag.write_text(
                        f"invariant={finding.invariant} at {finding.ts}",
                        encoding="utf-8",
                    )
                except Exception as exc:
                    logger.warning("Could not write halt flag: {}", exc)

    def _append_jsonl(self, finding: InvariantFinding) -> None:
        try:
            self._rotate_if_oversized()
            with self._jsonl_path.open("a", encoding="utf-8") as f:
                f.write(finding.to_json() + "\n")
        except Exception as exc:
            logger.warning("Invariant JSONL write failed: {}", exc)

    def _rotate_if_oversized(self) -> None:
        try:
            size = self._jsonl_path.stat().st_size
        except FileNotFoundError:
            return
        if size < self.ROTATE_AT_BYTES:
            return
        archive = self._jsonl_path.with_suffix(self._jsonl_path.suffix + ".1")
        try:
            if archive.exists():
                archive.unlink()
            self._jsonl_path.rename(archive)
        except Exception as exc:
            logger.warning("Invariant JSONL rotate failed: {}", exc)

    def recent(
        self,
        limit: int = 50,
        severity: Optional[Severity] = None,
    ) -> list[InvariantFinding]:
        with self._lock:
            items = list(self._recent)
        if severity is not None:
            items = [f for f in items if f.severity == severity.value]
        return list(reversed(items))[:limit]


_REGISTRY: Optional[InvariantRegistry] = None


def get_registry() -> InvariantRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = InvariantRegistry()
    return _REGISTRY


def configure_registry(
    telegram_send: Optional[Callable[[str], None]] = None,
) -> InvariantRegistry:
    global _REGISTRY
    _REGISTRY = InvariantRegistry(telegram_send=telegram_send)
    return _REGISTRY


def check(name: str, condition: bool, **kwargs: Any) -> bool:
    return get_registry().check(name, condition, **kwargs)
