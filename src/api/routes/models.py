"""
routes/models.py — Model performance dashboard endpoints.

GET  /api/models/summary              → one row per live symbol
GET  /api/models/accuracy/{symbol}    → daily accuracy time series
GET  /api/models/versions/{model_name} → retrain history for one model
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request

from src.api.auth import get_current_user
from src.api.schemas import (
    AccuracyPoint,
    AccuracyTimeSeriesResponse,
    ModelSummaryResponse,
    ModelSummaryRow,
    ModelVersionEntry,
    ModelVersionHistoryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])

# the universe sweep sprint the trading universe — DEV PREVIEW. Keep in sync with main.py::SYMBOLS and
# config/settings.yaml::trading.symbols. Production currently still on the
# 5-pair set (XAU/EUR/USDJPY/USDCAD/ETHUSD); promotion lands in Sprint 2.
LIVE_SYMBOLS: tuple[str, ...] = (
    "XAUUSD", "GBPUSD", "USDJPY", "USDCAD", "NZDUSD",
    "USDCHF", "GBPCHF", "EURAUD", "GBPAUD", "EURJPY",
)


def _get_live_state(request: Request):
    return request.app.state.live_state


# Module-level cache for /models/summary. Invalidation triggers:
#   1. Natural TTL expiry (30s — short enough that a fresh retrain is
#      visible within 30s of completion).
#   2. Monthly retrain or any call that writes to `model_versions` → the
#      caller can call `_invalidate_model_summary_cache()` directly.
# Key = account_id (one cache entry per account). Ran ~624 ms p95 before
# this cache was added (per P-1 stages 2a-2d capture); after, cached
# hits are < 1 ms.
_SUMMARY_CACHE_TTL_SEC = 30.0
_summary_cache: dict[Optional[int], tuple[float, "ModelSummaryResponse"]] = {}


def _invalidate_model_summary_cache() -> None:
    """Test + retrain-pipeline hook. Clears the TTL cache so the next
    request rebuilds from fresh DB queries. Safe to call any time."""
    _summary_cache.clear()


def _file_mtime_iso(path: Path) -> Optional[str]:
    """Return a file mtime as ISO-8601 UTC, or None if missing."""
    try:
        if not path.exists():
            return None
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _next_retrain_due() -> str:
    """
    First day of next month at 03:00 UTC — matches main.py's monthly
    retrain scheduler.
    """
    now = datetime.now(tz=timezone.utc)
    month = now.month + 1
    year = now.year
    if month > 12:
        month = 1
        year += 1
    return datetime(year, month, 1, 3, 0, 0, tzinfo=timezone.utc).isoformat()


@router.get("/summary", response_model=ModelSummaryResponse)
async def get_model_summary(
    request: Request,
    _user: str = Depends(get_current_user),
):
    """
    Per-symbol model health summary: retrain timestamps (DB + file
    mtime fallback), training-time val metrics, live rolling accuracy
    over the last 500 predictions.

    Cached for 30 seconds per account — the underlying data (model
    versions + rolling metrics) changes only on retrain or after ~500
    new predictions, so a short TTL is effectively free.
    """
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    account_id = ls.get_account_id() if hasattr(ls, "get_account_id") else None
    cached = _summary_cache.get(account_id)
    if cached is not None:
        cached_at, cached_resp = cached
        if (time.monotonic() - cached_at) < _SUMMARY_CACHE_TTL_SEC:
            return cached_resp

    ds = ls.data_store
    # Fetch latest versions + all symbols' rolling metrics concurrently.
    # Previously this was 6 sequential DB round-trips (1 + 5×rolling),
    # typically ~1.3s each → ~7.8s p95 for /api/models/summary per the
    # P-1 capture. asyncio.gather collapses that to the slowest single
    # query (~1.5s worst case).
    async def _rolling_safe(sym: str) -> dict:
        try:
            return await ds.get_rolling_metrics(sym, window=500)
        except Exception as exc:
            logger.warning("rolling metrics failed for %s: %s", sym, exc)
            return {"directional_accuracy": None, "mae": None, "n_predictions": 0}

    latest, *rolling_results = await asyncio.gather(
        ds.get_latest_model_versions(),
        *(_rolling_safe(sym) for sym in LIVE_SYMBOLS),
    )
    by_name = {row["model_name"]: row for row in latest}
    rolling_by_sym = dict(zip(LIVE_SYMBOLS, rolling_results))

    # A-8: latest drift score per symbol (best effort, must not 500 the page)
    drift_by_sym: dict[str, dict] = {}
    try:
        async with ds._session_factory() as session:
            from sqlalchemy import select
            from src.data_pipeline.data_store import DriftScoreRecord
            # For each symbol, grab the most-recent row.
            for sym in LIVE_SYMBOLS:
                stmt = (
                    select(DriftScoreRecord)
                    .where(DriftScoreRecord.symbol == sym)
                    .order_by(DriftScoreRecord.timestamp.desc())
                    .limit(1)
                )
                row = (await session.execute(stmt)).scalars().first()
                if row is not None:
                    drift_by_sym[sym] = {
                        "psi_max": row.psi_max,
                        "ks_max": row.ks_max,
                        "timestamp": row.timestamp,
                        "worst_feature": row.worst_feature,
                        "alert": bool(row.threshold_alert_breached),
                        "warn": bool(row.threshold_warn_breached),
                    }
    except Exception as exc:
        logger.debug("drift-scores query failed: %s", exc)

    out: list[ModelSummaryRow] = []
    models_dir = Path("data/models")
    for sym in LIVE_SYMBOLS:
        lstm_row = by_name.get(f"lstm_{sym}") or {}
        hmm_row  = by_name.get(f"hmm_{sym}") or {}
        rolling = rolling_by_sym[sym]

        n_pred = int(rolling.get("n_predictions") or 0)
        out.append(ModelSummaryRow(
            symbol=sym,
            lstm_version=lstm_row.get("version"),
            hmm_version=hmm_row.get("version"),
            lstm_trained_at=lstm_row.get("trained_at"),
            hmm_trained_at=hmm_row.get("trained_at"),
            lstm_val_loss=lstm_row.get("val_loss"),
            lstm_train_dir_acc=lstm_row.get("directional_accuracy"),
            live_dir_acc=(
                float(rolling["directional_accuracy"])
                if rolling.get("directional_accuracy") is not None and n_pred > 0
                else None
            ),
            live_mae=(
                float(rolling["mae"])
                if rolling.get("mae") is not None and n_pred > 0
                else None
            ),
            n_predictions=n_pred,
            lstm_file_mtime=_file_mtime_iso(models_dir / f"lstm_{sym}.pt"),
            next_retrain_due=_next_retrain_due(),
            drift_psi_max=(
                float(drift_by_sym[sym]["psi_max"])
                if sym in drift_by_sym and drift_by_sym[sym]["psi_max"] is not None
                else None
            ),
            drift_ks_max=(
                float(drift_by_sym[sym]["ks_max"])
                if sym in drift_by_sym and drift_by_sym[sym]["ks_max"] is not None
                else None
            ),
            drift_checked_at=drift_by_sym.get(sym, {}).get("timestamp"),
            drift_status=(
                "alert" if drift_by_sym.get(sym, {}).get("alert")
                else "warn" if drift_by_sym.get(sym, {}).get("warn")
                else "ok" if sym in drift_by_sym
                else None
            ),
            drift_worst_feature=drift_by_sym.get(sym, {}).get("worst_feature"),
        ))

    response = ModelSummaryResponse(symbols=out)
    _summary_cache[account_id] = (time.monotonic(), response)
    return response


@router.get("/accuracy/{symbol}", response_model=AccuracyTimeSeriesResponse)
async def get_accuracy_timeseries(
    symbol: str,
    request: Request,
    _user: str = Depends(get_current_user),
    days: int = Query(30, ge=1, le=365),
):
    """Daily directional-accuracy + MAE time series for one symbol."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    sym_norm = symbol.upper()
    rows = await ls.data_store.get_accuracy_timeseries(sym_norm, window_days=days)
    points = [
        AccuracyPoint(
            date=r["date"],
            directional_accuracy=r["directional_accuracy"],
            mae=r["mae"],
            n=r["n"],
        )
        for r in rows
    ]
    return AccuracyTimeSeriesResponse(symbol=sym_norm, points=points)


@router.get("/versions/{model_name}", response_model=ModelVersionHistoryResponse)
async def get_model_versions(
    model_name: str,
    request: Request,
    _user: str = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100),
):
    """Retrain audit trail for one model (newest version first)."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    rows = await ls.data_store.get_model_version_history(model_name, limit=limit)
    return ModelVersionHistoryResponse(
        model_name=model_name,
        versions=[ModelVersionEntry(**r) for r in rows],
    )
