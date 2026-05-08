"""
sse.py — Server-Sent Events stream for live dashboard updates.

Yields a JSON snapshot of live state every ``interval`` seconds
until the client disconnects. Used by the ``/api/live/stream``
endpoint.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator

import numpy as np

logger = logging.getLogger(__name__)


def _serialize_value(obj):
    """JSON serializer for numpy types and datetimes."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _build_snapshot(live_state) -> dict:
    """
    Build a JSON-safe snapshot dict from LiveState.

    Reads in-memory state defensively — if any component is
    unavailable (e.g. account_monitor raises), that section
    is omitted rather than crashing the stream.
    """
    data: dict = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "bot_status": live_state.bot_control.status.value,
    }

    # Account snapshot
    try:
        snap = live_state.account_monitor.get_info()
        data["account"] = {
            "balance": snap.balance,
            "equity": snap.equity,
            "margin": snap.margin,
            "free_margin": snap.free_margin,
            "margin_level": snap.margin_level,
            "floating_pnl": snap.floating_pnl,
            "open_positions": snap.open_positions,
        }
    except Exception:
        data["account"] = None

    # Circuit breaker
    try:
        breakers = live_state.circuit_breaker.active_breakers()
        data["breaker"] = {
            "multiplier": live_state.circuit_breaker.current_size_multiplier(),
            "active_breakers": breakers,
            "is_halted": live_state.circuit_breaker.is_halted(),
        }
    except Exception:
        data["breaker"] = None

    # Peak equity
    try:
        data["peak_equity"] = live_state.risk_monitor.get_peak_equity()
    except Exception:
        data["peak_equity"] = 0.0

    # Positions count
    data["positions_count"] = len(live_state.tracked_positions)

    # Per-symbol last signal
    try:
        last_signal = live_state.combiner.last_signal
        if last_signal is not None:
            regime = last_signal.regime
            data["last_signal"] = {
                "symbol": last_signal.symbol,
                "should_trade": last_signal.should_trade,
                "direction": last_signal.direction,
                "combined_score": last_signal.combined_score,
                "confidence": last_signal.confidence,
                "regime_label": regime.regime_label if regime else None,
                "regime_probability": regime.state_probability if regime else None,
            }
    except Exception:
        pass

    return data


async def stream_live_state(
    live_state,
    interval: float = 2.0,
) -> AsyncGenerator[str, None]:
    """
    Yield SSE events with live state snapshots.

    Each event is ``data: {json}\n\n`` — the SSE wire format.
    The generator runs until the client disconnects or the
    server cancels it.
    """
    while True:
        try:
            snapshot = _build_snapshot(live_state)
            payload = json.dumps(snapshot, default=_serialize_value)
            yield f"data: {payload}\n\n"
        except Exception as exc:
            logger.warning("SSE snapshot error: %s", exc)
            error_payload = json.dumps({"error": str(exc)})
            yield f"data: {error_payload}\n\n"

        await asyncio.sleep(interval)
