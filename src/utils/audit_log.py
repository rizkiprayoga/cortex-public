"""
audit_log.py — Thread-safe CSV audit writers for paper-trading analysis.

These writers record every meaningful decision the bot makes so you can
reconstruct "what happened at 14:30 last Tuesday" after the fact. They
complement the DB persistence (which is transactional and queryable)
with plain-text CSVs that are:

  - human-readable (open in Excel / any editor)
  - append-only (no corruption risk)
  - thread-safe (locked writes)
  - durable to process crashes (flush-after-write)

Three log streams:

    data/logs/signal_audit.csv   one row per signal attempt (traded or not)
    data/logs/trade_events.csv   one row per entry/modify/close event
    data/logs/tick_summary.csv   one row per M15 tick per symbol

Rotation is NOT applied at this layer — these CSVs should be rotated
externally (by week/month) if they grow too large. Typical volume is
~1-5 MB/month for a 4-symbol bot.
"""
from __future__ import annotations

import csv
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLog:
    """Thread-safe append-only CSV writer with header auto-creation."""

    def __init__(self, path: Path, headers: list[str]):
        self.path = Path(path)
        self.headers = headers
        self._lock = threading.Lock()
        self._ensure_header()

    def _ensure_header(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.headers)

    def write(self, row: dict[str, Any]) -> None:
        """Append a row. Missing columns default to empty string; extra keys ignored."""
        values = [row.get(h, "") for h in self.headers]
        with self._lock:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(values)
                f.flush()
                os.fsync(f.fileno())


# =========================================================================
# Pre-configured logs used by the trading loop
# =========================================================================

_LOG_DIR = Path("data/logs")


SIGNAL_AUDIT = AuditLog(
    _LOG_DIR / "signal_audit.csv",
    headers=[
        "timestamp",          # ISO UTC when signal was generated
        "symbol",
        "regime",             # Crash / Bear / Neutral / Bull / Euphoria
        "regime_prob",        # confidence of the HMM state
        "lstm_prediction",    # raw LSTM output (TB label estimate)
        "combined_score",     # signed fusion score
        "direction",          # "buy" / "sell" / None
        "should_trade",       # bool — did the combiner say yes?
        "executed",           # bool — did we actually place an order?
        "news_blackout",      # bool — was entry blocked by news filter?
        "nearest_cb",         # "FOMC"/"ECB"/"BoJ"/"BoC" or ""
        "nearest_hours",      # hours from nearest CB (negative = before)
        "block_reason",       # textual reason if not executed (concat of flags)
        "cb_multiplier",      # circuit breaker size multiplier at time of signal
        "reasoning",          # full concatenated reasoning from combiner
    ],
)


TRADE_EVENTS = AuditLog(
    _LOG_DIR / "trade_events.csv",
    headers=[
        "timestamp",
        "event",              # "entry" / "modify" / "exit"
        "ticket",
        "symbol",
        "direction",
        "lot_size",
        "entry_price",
        "current_price",      # for modify/exit events
        "sl_price",
        "tp_price",
        "pnl_usd",            # realized (on close) or current (on modify)
        "r_multiple",         # PnL / initial risk
        "bars_held",          # H1 bars held
        "be_locked",          # breakeven SL lock active
        "regime_at_entry",
        "combined_score_at_entry",
        "exit_reason",        # "tp" / "sl" / "time" / "reversal" / "manual"
    ],
)


TICK_SUMMARY = AuditLog(
    _LOG_DIR / "tick_summary.csv",
    headers=[
        "timestamp",
        "symbol",
        "price",              # H4 close (or M15 close for exec)
        "atr_14",
        "regime",
        "regime_prob",
        "open_positions",     # count for this symbol
        "equity",
        "floating_pnl",
        "daily_pnl",
        "breaker_active",     # comma-separated list or "none"
        "breaker_multiplier",
    ],
)


def now_iso() -> str:
    """UTC timestamp in ISO-8601 with microseconds trimmed for CSV tidiness."""
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
