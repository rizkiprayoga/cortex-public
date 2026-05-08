"""
routes/backtest.py — Walk-forward backtest endpoints.

POST  /api/backtest/submit         → start a new backtest job
GET   /api/backtest/status/{run_id} → poll job status + metrics
GET   /api/backtest/runs            → list recent backtest runs
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from src.api.auth import get_current_user
from src.api.schemas import (
    BacktestDetailResponse,
    BacktestEquityPoint,
    BacktestRunSummary,
    BacktestRunsResponse,
    BacktestStatusResponse,
    BacktestSubmitRequest,
    BacktestSubmitResponse,
    BacktestTradeRow,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


def _get_live_state(request: Request):
    return request.app.state.live_state


_INITIAL_EQUITY_ASSUMED = 10_000.0  # all CLI + ingested runs start from $10k


def _calmar_from_row(run: dict) -> float:
    """Derive Calmar = CAGR / |Max DD%| from persisted columns only.

    No new DB column needed. Returns 0.0 when DD < 0.5% (undefined
    guard — one drawdown spike dominates) or when date span is invalid.

    CAGR assumed from initial $10k equity — matches CLI + ingest
    conventions. When the portfolio/live variant arrives it should pass
    its own initial equity in separately.
    """
    try:
        max_dd = float(run.get("max_drawdown_pct") or 0.0)
        if max_dd < 0.5:
            return 0.0
        net_pnl = float(run.get("net_pnl") or 0.0)
        end_equity = _INITIAL_EQUITY_ASSUMED + net_pnl
        if end_equity <= 0:
            return 0.0
        start: Optional[str] = run.get("start_date")
        end: Optional[str] = run.get("end_date")
        if not start or not end:
            return 0.0
        # Parse as tz-aware if the string carries a zone, otherwise let
        # fromisoformat return a naive datetime. Either way the subtraction
        # yields a correct timedelta as long as both values are consistent.
        # (Earlier iteration `.split("+")[0]` silently dropped offsets like
        # "+05:30" and could shift year boundaries — caught 2026-04-19.)
        def _parse_dt(s: str) -> datetime:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        t0 = _parse_dt(str(start))
        t1 = _parse_dt(str(end))
        # Normalize: if one side is tz-aware and the other naive, strip tz
        # from the aware one so subtraction doesn't raise.
        if (t0.tzinfo is None) != (t1.tzinfo is None):
            t0 = t0.replace(tzinfo=None)
            t1 = t1.replace(tzinfo=None)
        years = (t1 - t0).total_seconds() / (365.25 * 86400)
        if years <= 0:
            return 0.0
        cagr_pct = ((end_equity / _INITIAL_EQUITY_ASSUMED) ** (1.0 / years) - 1.0) * 100.0
        return round(cagr_pct / max_dd, 3)
    except Exception:
        return 0.0


def _enrich(run: dict) -> dict:
    """Add derived fields (calmar_ratio) to a backtest_runs row dict."""
    out = dict(run)
    out.setdefault("calmar_ratio", _calmar_from_row(run))
    return out


@router.post("/submit", response_model=BacktestSubmitResponse)
async def submit_backtest(
    body: BacktestSubmitRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    _user: str = Depends(get_current_user),
):
    """Submit a new backtest job. Returns immediately with a run_id to poll."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    run_id = str(uuid.uuid4())

    # Snapshot the LSTM model in use at run creation so the result is
    # attributable even after a future retrain. Best-effort — if the file
    # is missing the columns stay null.
    from datetime import datetime as _dt, timezone as _tz
    from pathlib import Path as _Path
    model_name = f"lstm_{body.symbol}"
    model_path = _Path("data/models") / f"{model_name}.pt"
    model_trained_at = None
    model_version = None
    if model_path.exists():
        try:
            mtime = model_path.stat().st_mtime
            model_trained_at = _dt.fromtimestamp(mtime, tz=_tz.utc).isoformat()
            # Fetch the registry version too, if any
            try:
                latest = await ls.data_store.get_latest_model_versions()
                vrow = next((r for r in latest if r.get("model_name") == model_name), None)
                if vrow:
                    model_version = vrow.get("version")
            except Exception:
                pass
        except Exception:
            pass

    # Insert pending row
    await ls.data_store.create_backtest_run({
        "id": run_id,
        "status": "pending",
        "symbol": body.symbol,
        "timeframe": body.timeframe,
        "start_date": body.start_date,
        "end_date": body.end_date,
        "run_mode": body.mode,
        "model_name": model_name,
        "model_version": model_version,
        "model_trained_at": model_trained_at,
    })

    # Schedule async job
    from scripts.backtest import run_backtest_async
    background_tasks.add_task(
        run_backtest_async,
        run_id=run_id,
        symbol=body.symbol,
        timeframe=body.timeframe,
        start_date=body.start_date,
        end_date=body.end_date,
        initial_equity=body.initial_equity,
        data_store=ls.data_store,
        mode=body.mode,
    )

    return BacktestSubmitResponse(run_id=run_id, status="pending")


@router.get("/status/{run_id}", response_model=BacktestStatusResponse)
async def get_backtest_status(
    run_id: str,
    request: Request,
    _user: str = Depends(get_current_user),
):
    """Poll backtest job status and results."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    run = await ls.data_store.get_backtest_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest run {run_id} not found",
        )

    error_msg = run.pop("error_message", None)
    return BacktestStatusResponse(
        run=BacktestRunSummary(**_enrich(run)),
        error_message=error_msg,
    )


@router.get("/runs", response_model=BacktestRunsResponse)
async def list_backtest_runs(
    request: Request,
    _user: str = Depends(get_current_user),
    limit: int = 20,
):
    """List recent backtest runs."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    runs = await ls.data_store.list_backtest_runs(limit=limit)
    summaries = [BacktestRunSummary(**_enrich(r)) for r in runs]
    return BacktestRunsResponse(runs=summaries, count=len(summaries))


@router.get("/runs/{run_id}/detail", response_model=BacktestDetailResponse)
async def get_backtest_detail(
    run_id: str,
    request: Request,
    _user: str = Depends(get_current_user),
):
    """Full drill-down for a single backtest run.

    Returns the run summary plus the per-bar equity curve and every
    closed trade. Drives the F3 Backtest drawer + compare-mode equity
    overlays in the dashboard.

    Performance: each of the three DataStore calls hits a single
    indexed query (by primary key or by run_id FK). Rows can run into
    the thousands for a multi-year run but the payload compresses well
    for the dashboard's React-Query in-memory cache.
    """
    ls = _get_live_state(request)

    run = await ls.data_store.get_backtest_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")

    # Equity curve + trades — both order by timestamp for the charts
    equity_raw = await ls.data_store.get_backtest_equity(run_id)
    trades_raw = await ls.data_store.get_backtest_trades(run_id)

    equity = [
        BacktestEquityPoint(
            bar_timestamp=str(p.get("bar_timestamp", "")),
            equity=float(p.get("equity", 0.0) or 0.0),
            drawdown_pct=float(p.get("drawdown_pct", 0.0) or 0.0),
        )
        for p in (equity_raw or [])
    ]

    trades: list[BacktestTradeRow] = []
    for t in (trades_raw or []):
        trades.append(BacktestTradeRow(
            symbol=str(t.get("symbol", "")),
            direction=str(t.get("direction", "")),
            entry_time=str(t.get("entry_time", "")) if t.get("entry_time") else None,
            exit_time=str(t.get("exit_time", "")) if t.get("exit_time") else None,
            entry_price=float(t.get("entry_price", 0.0) or 0.0),
            exit_price=float(t.get("exit_price", 0.0) or 0.0),
            pnl=float(t.get("pnl", 0.0) or 0.0),
            r_multiple=float(t.get("r_multiple", 0.0) or 0.0),
            exit_reason=str(t.get("exit_reason", "") or ""),
            strategy_name=t.get("strategy_name"),
            regime_label=t.get("regime_label"),
            combined_score=(
                float(t["combined_score"])
                if t.get("combined_score") is not None else None
            ),
        ))

    # A-7 overfitting diagnostics — need the trade series.
    dsr: Optional[float] = None
    stab: Optional[float] = None
    try:
        import numpy as _np
        from scipy import stats as _sstats
        from src.ml.overfitting import (
            deflated_sharpe_ratio, sharpe_from_returns, sharpe_stability,
        )
        # Use trade-level pnl as the return series. Normalize by initial
        # equity so Sharpe is scale-invariant vs the size of each trade.
        pnl_arr = _np.array(
            [t.pnl for t in trades if t.pnl is not None and _np.isfinite(t.pnl)],
            dtype=float,
        )
        if pnl_arr.size >= 10:
            returns = pnl_arr / 10000.0   # matches _INITIAL_EQUITY_ASSUMED
            obs_sr = sharpe_from_returns(returns)
            skew = float(_sstats.skew(returns, bias=False))
            # scipy default is *excess* kurtosis; DSR expects raw kurtosis
            kurt = float(_sstats.kurtosis(returns, bias=False, fisher=False))
            # Count total backtest_runs as a coarse proxy for "trials tried".
            # Better than 1 (which under-penalizes); worse than a proper
            # hparam-search tally. Documented in SYSTEM_AUDIT / BACKLOG.
            try:
                all_runs = await ls.data_store.get_backtest_runs(limit=500)
                n_trials = max(1, len(all_runs or []))
            except Exception:
                n_trials = 1
            dsr = round(deflated_sharpe_ratio(
                sharpe=obs_sr, n_obs=pnl_arr.size,
                skewness=skew, kurtosis=kurt, n_trials=n_trials,
            ), 4)
            stab = round(sharpe_stability(returns, n_windows=4), 4)
    except Exception as _exc:
        logger.debug("overfitting diagnostics skipped: %s", _exc)

    enriched = _enrich(run)
    enriched["deflated_sharpe"] = dsr
    enriched["sharpe_stability"] = stab

    return BacktestDetailResponse(
        summary=BacktestRunSummary(**enriched),
        equity_curve=equity,
        trades=trades,
    )
