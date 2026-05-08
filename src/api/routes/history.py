"""
routes/history.py — Historical data query endpoints.

GET  /api/history/trades     → paginated trade history
GET  /api/history/equity     → equity curve points
GET  /api/history/signals    → paginated signal log
GET  /api/history/accuracy   → rolling model accuracy metrics
GET  /api/history/metrics    → computed trading performance metrics
"""

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request

from src.api.auth import get_current_user
from src.api.schemas import (
    BalanceOperation,
    BalanceOperationsResponse,
    EquityCurveResponse,
    EquityPoint,
    ModelAccuracyResponse,
    SignalAuditItem,
    SignalAuditResponse,
    SignalLogItem,
    SignalLogResponse,
    TradeEventItem,
    TradeHistoryItem,
    TradeHistoryResponse,
    TradeTimelineResponse,
    TradingMetricsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/history", tags=["history"])


def _get_live_state(request: Request):
    return request.app.state.live_state


@router.get("/trades", response_model=TradeHistoryResponse)
async def get_trades(
    request: Request,
    _user: str = Depends(get_current_user),
    symbol: Optional[str] = Query(None),
    since: Optional[str] = Query(None, description="ISO 8601 start date"),
    until: Optional[str] = Query(None, description="ISO 8601 end date"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=2000),
):
    """Paginated trade history with optional symbol/date filters."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    since_dt = datetime.fromisoformat(since) if since else None
    until_dt = datetime.fromisoformat(until) if until else None
    offset = (page - 1) * page_size

    # Audit H9: don't leak cross-account data when current_account_id
    # is unset. Return empty result instead of ALL accounts' data.
    # Audit H8: atomic read via helper.
    account_id = ls.get_account_id()
    if account_id is None:
        return TradeHistoryResponse(
            trades=[], total=0, page=page, page_size=page_size,
        )

    cache_key = (account_id, symbol or "", since or "", until or "", page, page_size)
    _now = datetime.now(tz=timezone.utc)
    cached = _TRADES_CACHE.get(cache_key)
    if cached is not None and (_now - cached[0]).total_seconds() < _HISTORY_CACHE_TTL_SEC:
        return cached[1]

    items, total = await ls.data_store.get_trades_paginated(
        symbol=symbol, since=since_dt, until=until_dt,
        offset=offset, limit=page_size,
        mt5_account=account_id,
    )
    trades = [TradeHistoryItem(**t) for t in items]
    response = TradeHistoryResponse(
        trades=trades, total=total, page=page, page_size=page_size,
    )
    _TRADES_CACHE[cache_key] = (_now, response)
    return response


@router.get("/equity", response_model=EquityCurveResponse)
async def get_equity(
    request: Request,
    _user: str = Depends(get_current_user),
    limit: int = Query(500, ge=1, le=5000),
):
    """Equity curve data points."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    # Audit H9: empty result if no account set, instead of leaking all.
    account_id = ls.get_account_id()
    if account_id is None:
        return EquityCurveResponse(points=[], count=0)
    df = await ls.data_store.get_equity_history(
        limit=limit, mt5_account=account_id,
    )
    points = []
    for ts, row in df.iterrows():
        points.append(EquityPoint(
            timestamp=ts.isoformat(),
            equity=row["equity"],
            balance=row.get("balance"),
            floating_pnl=row.get("floating_pnl"),
        ))
    return EquityCurveResponse(points=points, count=len(points))


@router.get("/signals", response_model=SignalLogResponse)
async def get_signals(
    request: Request,
    _user: str = Depends(get_current_user),
    symbol: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """Paginated signal log."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    # Audit H9: empty result if no account set.
    account_id = ls.get_account_id()
    if account_id is None:
        return SignalLogResponse(
            signals=[], total=0, page=page, page_size=page_size,
        )
    offset = (page - 1) * page_size
    items, total = await ls.data_store.get_signals_paginated(
        symbol=symbol, offset=offset, limit=page_size,
        mt5_account=account_id,
    )
    signals = [SignalLogItem(**s) for s in items]
    return SignalLogResponse(
        signals=signals, total=total, page=page, page_size=page_size,
    )


@router.get("/accuracy", response_model=ModelAccuracyResponse)
async def get_accuracy(
    request: Request,
    _user: str = Depends(get_current_user),
    symbol: str = Query("XAUUSD"),
    window: int = Query(500, ge=10, le=5000),
):
    """Rolling model accuracy metrics."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    metrics = await ls.data_store.get_rolling_metrics(symbol=symbol, window=window)
    return ModelAccuracyResponse(symbol=symbol, window=window, **metrics)


# Per-endpoint TTL caches — the History page fires /metrics,
# /trades, and /account-ledger in rapid succession on every navigation,
# and React Query sometimes retries on focus/reconnect. Each cold call
# is 2-3s p95 per the P-1 capture (metrics: 10K-trade aggregate;
# trades: paginated scan; ledger: full MT5 round-trip). A 30s TTL per
# (account, query-shape) absorbs the burst without touching semantics
# — trade data changes slowly enough that half-a-minute stale is fine.
_HISTORY_CACHE_TTL_SEC = 30
_METRICS_CACHE: dict[tuple, tuple[datetime, "TradingMetricsResponse"]] = {}
_TRADES_CACHE: dict[tuple, tuple[datetime, "TradeHistoryResponse"]] = {}
_LEDGER_CACHE: dict[tuple, tuple[datetime, "BalanceOperationsResponse"]] = {}


def _metrics_cache_clear() -> None:
    """Test hook — clears the caches between runs so tests stay
    deterministic. No production caller."""
    _METRICS_CACHE.clear()
    _TRADES_CACHE.clear()
    _LEDGER_CACHE.clear()


@router.get("/metrics", response_model=TradingMetricsResponse)
async def get_metrics(
    request: Request,
    _user: str = Depends(get_current_user),
    symbol: Optional[str] = Query(None),
):
    """Server-computed trading performance metrics."""
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    import pandas as pd

    account_id = ls.get_account_id()
    if account_id is None:
        return TradingMetricsResponse(
            win_rate=0.0, profit_factor=0.0, sharpe_daily=0.0,
            max_drawdown_pct=0.0, net_pnl=0.0, total_r=0.0, total_trades=0,
        )

    # Cache check — key is (account, symbol) so a symbol filter doesn't
    # serve another symbol's cached aggregate.
    cache_key = (account_id, symbol or "")
    _now = datetime.now(tz=timezone.utc)
    cached = _METRICS_CACHE.get(cache_key)
    if cached is not None and (_now - cached[0]).total_seconds() < _HISTORY_CACHE_TTL_SEC:
        return cached[1]

    trades_list, total = await ls.data_store.get_trades_paginated(
        symbol=symbol, offset=0, limit=10000, mt5_account=account_id,
    )

    if not trades_list:
        return TradingMetricsResponse(
            win_rate=0.0, profit_factor=0.0, sharpe_daily=0.0,
            max_drawdown_pct=0.0, net_pnl=0.0, total_r=0.0, total_trades=0,
        )

    df = pd.DataFrame(trades_list)
    # Paginated query returns DESC by close — re-sort ASC so cumulative
    # math (cumsum, drawdown) is computed on the true equity curve.
    if "timestamp_close" in df.columns:
        df = df.sort_values("timestamp_close", ascending=True, kind="stable")
    pnls = df["pnl_usd"].dropna()
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate = len(wins) / n if n > 0 else 0.0
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    # Profit factor = gross_profit / gross_loss.
    #   • No losses AND some wins  → sentinel 99.99 (effectively infinite;
    #     clearly distinguished from 0 so the UI doesn't paint the card red).
    #   • No wins (loss-only or no trades) → 0.0.
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = 99.99
    else:
        profit_factor = 0.0
    net_pnl = float(pnls.sum())

    # Annualized Sharpe (forex convention: 252 trading days). Computed from
    # daily-aggregated trade PnL as a proxy for daily returns. The previous
    # version returned the raw daily ratio, which under-reports by ~√252 —
    # a real annualized Sharpe of 1.5 showed up as 0.09 on the dashboard.
    # ddof=1 (sample std) is pandas default; kept explicit for clarity.
    if "timestamp_close" in df.columns and n > 1:
        df["close_dt"] = pd.to_datetime(df["timestamp_close"], errors="coerce")
        daily = df.groupby(df["close_dt"].dt.date)["pnl_usd"].sum()
        daily_std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
        if len(daily) > 1 and daily_std > 0:
            sharpe_daily = float(daily.mean()) / daily_std * math.sqrt(252)
        else:
            sharpe_daily = 0.0
    else:
        sharpe_daily = 0.0

    # Max drawdown computed on the true EQUITY curve (starting_balance +
    # cumulative_pnl), not on raw cumulative PnL. Dividing DD by
    # peak(cum_pnl) gives nonsense when the trade sample is small or
    # net-negative (e.g. peak_cum=$13 with $63 DD yields 464.9%).
    try:
        eq_hist = await ls.data_store.get_equity_history(limit=5000, mt5_account=account_id)
        starting_balance = (
            float(eq_hist["balance"].iloc[-1])  # oldest snapshot
            if not eq_hist.empty and "balance" in eq_hist.columns
            else 10000.0
        )
    except Exception:
        starting_balance = 10000.0
    # Seed the equity curve with the pre-trade balance so the opening
    # drawdown is visible. Without the leading zero, a single losing
    # trade produces a one-point series where peak == trough and DD = 0.
    cum = pd.concat([pd.Series([0.0]), pnls.cumsum()], ignore_index=True)
    equity_curve = starting_balance + cum
    running_peak = equity_curve.cummax()
    dd_abs = running_peak - equity_curve
    max_dd_abs = float(dd_abs.max()) if len(dd_abs) > 0 else 0.0
    # Denominator is the equity at the peak before the worst drawdown
    peak_equity = float(running_peak.max()) if len(running_peak) > 0 else starting_balance
    max_drawdown_pct = (
        (max_dd_abs / peak_equity * 100) if peak_equity > 0 else 0.0
    )

    # ---- Rolling 90d Calmar (A-3 live half) ----
    # Subset the last 90d of trades, rebuild equity + DD on that slice
    # anchored at the equity value 90 days ago. Undefined when: fewer
    # than 10 trades in the window (low-N noise), or DD < 0.5%
    # (one-spike dominance). Window span is clamped to 90d regardless
    # of how many trades we have — this is what "annualized" means.
    calmar_90d = 0.0
    try:
        if "timestamp_close" in df.columns and n >= 10:
            # Compute on closed trades only so `mask` and `pnls_closed`
            # share the same index. Mixing an index-aligned mask with
            # the pre-dropna `pnls` under-counted pre-window PnL whenever
            # open positions had NaN `pnl_usd` rows inside the window.
            df_closed = df[df["pnl_usd"].notna()]
            closes = pd.to_datetime(df_closed["timestamp_close"], errors="coerce")
            pnls_closed = df_closed["pnl_usd"]
            cutoff = _now - pd.Timedelta(days=90)
            mask = closes >= cutoff
            window_trades = df_closed.loc[mask]
            if len(window_trades) >= 10:
                w_pnls = window_trades["pnl_usd"]
                w_cum = w_pnls.cumsum()
                # Anchor: equity at the start of the window = cumulative
                # PnL of all trades BEFORE the window + starting_balance.
                pre_window_pnl = float(pnls_closed.loc[~mask].sum()) if mask.any() else 0.0
                start_eq_90d = starting_balance + pre_window_pnl
                if start_eq_90d > 0:
                    eq_90d = start_eq_90d + w_cum
                    peak_90d = eq_90d.cummax()
                    dd_90d_pct = float(((peak_90d - eq_90d) / peak_90d * 100).max())
                    end_eq_90d = float(eq_90d.iloc[-1])
                    # CAGR over the actual realized span (may be <90d if
                    # account is newer than 90 days). Using 90 hardcoded
                    # would over-annualize a 10-day sample.
                    span_days = (closes.max() - closes.loc[mask].min()).total_seconds() / 86400
                    years = max(span_days / 365.25, 1 / 365.25)
                    if end_eq_90d > 0 and dd_90d_pct >= 0.5:
                        cagr_pct = ((end_eq_90d / start_eq_90d) ** (1.0 / years) - 1.0) * 100.0
                        calmar_90d = round(cagr_pct / dd_90d_pct, 3)
    except Exception:
        calmar_90d = 0.0

    # Total R = sum of r_multiple_at_exit across closed trades (nullable
    # column — OrderManager populates it on every broker close, but some
    # legacy rows have NULL, so dropna() before summing).
    total_r = (
        float(df["r_multiple_at_exit"].dropna().sum())
        if "r_multiple_at_exit" in df.columns
        else 0.0
    )

    response = TradingMetricsResponse(
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
        sharpe_daily=round(sharpe_daily, 4),
        max_drawdown_pct=round(max_drawdown_pct, 2),
        net_pnl=round(net_pnl, 2),
        total_r=round(total_r, 4),
        total_trades=n,
        calmar_90d=calmar_90d,
    )
    _METRICS_CACHE[cache_key] = (_now, response)
    return response


# ---------------------------------------------------------------------------
# Signal audit feed — reads the CSV at data/logs/signal_audit.csv
# (written by main.py's trading loop). Unlike /api/history/signals which
# reads the `signals` DB table, this endpoint carries the full reasoning
# string + block_reason + news-blackout context so the dashboard can show
# WHY each signal fired or didn't. Used by the F3 "Signals Log" screen.
# ---------------------------------------------------------------------------

_SIGNAL_AUDIT_CSV = Path("data/logs/signal_audit.csv")


def _parse_bool(v: str) -> bool:
    return v.strip().lower() in ("true", "1", "yes")


def _parse_float(v: str):
    try:
        return float(v) if v not in ("", None) else None
    except (ValueError, TypeError):
        return None


@router.get("/signal-audit", response_model=SignalAuditResponse)
async def get_signal_audit(
    request: Request,
    _user: str = Depends(get_current_user),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    executed: Optional[bool] = Query(
        None, description="True=only executed, False=only blocked, None=both",
    ),
    block_reason: Optional[str] = Query(
        None, description="Filter by block_reason substring match",
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=2000),
):
    """Paginated view of data/logs/signal_audit.csv.

    Each row is one signal attempt with the full gate-by-gate reasoning,
    whether it executed, and if not, why. Returns most recent first.
    """
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    if not _SIGNAL_AUDIT_CSV.exists():
        return SignalAuditResponse(items=[], total=0, page=page, page_size=page_size)

    # Read + filter. Files in practice stay <10 MB; full read is fine.
    import csv
    try:
        with open(_SIGNAL_AUDIT_CSV, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        logger.warning("signal_audit.csv read failed: %s", exc)
        return SignalAuditResponse(items=[], total=0, page=page, page_size=page_size)

    # Newest first
    rows.reverse()

    # Filters
    if symbol:
        sym = symbol.upper()
        rows = [r for r in rows if (r.get("symbol") or "").upper() == sym]
    if executed is not None:
        rows = [r for r in rows if _parse_bool(r.get("executed") or "") == executed]
    if block_reason:
        needle = block_reason.lower()
        rows = [r for r in rows if needle in (r.get("block_reason") or "").lower()]

    total = len(rows)
    start = (page - 1) * page_size
    page_rows = rows[start:start + page_size]

    items: list[SignalAuditItem] = []
    for r in page_rows:
        items.append(SignalAuditItem(
            timestamp=r.get("timestamp", ""),
            symbol=r.get("symbol", ""),
            regime=r.get("regime") or None,
            regime_prob=_parse_float(r.get("regime_prob", "")),
            lstm_prediction=_parse_float(r.get("lstm_prediction", "")),
            combined_score=_parse_float(r.get("combined_score", "")),
            direction=r.get("direction") or None,
            should_trade=_parse_bool(r.get("should_trade") or ""),
            executed=_parse_bool(r.get("executed") or ""),
            news_blackout=_parse_bool(r.get("news_blackout") or ""),
            nearest_cb=r.get("nearest_cb") or None,
            nearest_hours=_parse_float(r.get("nearest_hours", "")),
            block_reason=r.get("block_reason") or None,
            cb_multiplier=_parse_float(r.get("cb_multiplier", "")),
            reasoning=r.get("reasoning") or None,
        ))

    return SignalAuditResponse(
        items=items, total=total, page=page, page_size=page_size,
    )


# --------------------------------------------------------------------------
# Trade timeline — per-ticket stitched lifecycle (events + signals)
# Reads data/logs/trade_events.csv and signal_audit.csv directly.
# --------------------------------------------------------------------------

_TRADE_EVENTS_CSV = Path("data/logs/trade_events.csv")


def _parse_int(v: str):
    try:
        return int(v) if v not in ("", None) else None
    except (ValueError, TypeError):
        try:
            return int(float(v))
        except Exception:
            return None


@router.get(
    "/trade-timeline/{ticket}",
    response_model=TradeTimelineResponse,
)
async def get_trade_timeline(
    ticket: int,
    request: Request,
    _user: str = Depends(get_current_user),
    signal_window_min: int = Query(
        30, ge=0, le=1440,
        description="Include signal_audit rows within ±N minutes of entry",
    ),
):
    """
    Per-ticket activity stream.

    Returns the chronological list of trade events (entry, modify,
    partial_close, exit) for this ticket from trade_events.csv, plus
    any signals from signal_audit.csv that fired on the same symbol
    within ±signal_window_min of the entry event. Gives operators a
    "what did the bot do (and why)?" view for a single trade.
    """
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    import csv
    events: list[TradeEventItem] = []
    symbol: Optional[str] = None
    entry_ts_iso: Optional[str] = None

    # 1. Events ------------------------------------------------------------
    if _TRADE_EVENTS_CSV.exists():
        try:
            with open(_TRADE_EVENTS_CSV, "r", encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    tkt = _parse_int(r.get("ticket") or "")
                    if tkt != ticket:
                        continue
                    ev = TradeEventItem(
                        timestamp=r.get("timestamp", ""),
                        event=r.get("event", ""),
                        ticket=tkt,
                        symbol=r.get("symbol") or None,
                        direction=r.get("direction") or None,
                        lot_size=_parse_float(r.get("lot_size", "")),
                        entry_price=_parse_float(r.get("entry_price", "")),
                        current_price=_parse_float(r.get("current_price", "")),
                        sl_price=_parse_float(r.get("sl_price", "")),
                        tp_price=_parse_float(r.get("tp_price", "")),
                        pnl_usd=_parse_float(r.get("pnl_usd", "")),
                        r_multiple=_parse_float(r.get("r_multiple", "")),
                        bars_held=_parse_int(r.get("bars_held", "")),
                        be_locked=_parse_bool(r.get("be_locked") or ""),
                        regime_at_entry=r.get("regime_at_entry") or None,
                        combined_score_at_entry=_parse_float(
                            r.get("combined_score_at_entry", ""),
                        ),
                        exit_reason=r.get("exit_reason") or None,
                    )
                    events.append(ev)
                    if ev.event == "entry" and entry_ts_iso is None:
                        entry_ts_iso = ev.timestamp
                        symbol = ev.symbol
        except OSError as exc:
            logger.warning("trade_events.csv read failed: %s", exc)

    events.sort(key=lambda e: e.timestamp)

    # 2. Related signals ---------------------------------------------------
    # Only look up signals if we found an entry event and symbol.
    signals: list[SignalAuditItem] = []
    if symbol and entry_ts_iso and _SIGNAL_AUDIT_CSV.exists() and signal_window_min > 0:
        try:
            entry_dt = datetime.fromisoformat(entry_ts_iso.replace("Z", "+00:00"))
        except ValueError:
            entry_dt = None
        if entry_dt is not None:
            try:
                from datetime import timedelta
                window = timedelta(minutes=signal_window_min)
                with open(_SIGNAL_AUDIT_CSV, "r", encoding="utf-8", newline="") as f:
                    for r in csv.DictReader(f):
                        if (r.get("symbol") or "").upper() != symbol.upper():
                            continue
                        ts_str = r.get("timestamp", "")
                        try:
                            row_dt = datetime.fromisoformat(
                                ts_str.replace("Z", "+00:00"),
                            )
                        except ValueError:
                            continue
                        if abs(row_dt - entry_dt) > window:
                            continue
                        signals.append(SignalAuditItem(
                            timestamp=ts_str,
                            symbol=r.get("symbol", ""),
                            regime=r.get("regime") or None,
                            regime_prob=_parse_float(r.get("regime_prob", "")),
                            lstm_prediction=_parse_float(r.get("lstm_prediction", "")),
                            combined_score=_parse_float(r.get("combined_score", "")),
                            direction=r.get("direction") or None,
                            should_trade=_parse_bool(r.get("should_trade") or ""),
                            executed=_parse_bool(r.get("executed") or ""),
                            news_blackout=_parse_bool(r.get("news_blackout") or ""),
                            nearest_cb=r.get("nearest_cb") or None,
                            nearest_hours=_parse_float(r.get("nearest_hours", "")),
                            block_reason=r.get("block_reason") or None,
                            cb_multiplier=_parse_float(r.get("cb_multiplier", "")),
                            reasoning=r.get("reasoning") or None,
                        ))
            except OSError as exc:
                logger.warning("signal_audit.csv read failed: %s", exc)
    signals.sort(key=lambda s: s.timestamp)

    return TradeTimelineResponse(
        ticket=ticket, events=events, signals=signals,
    )


# --------------------------------------------------------------------------
# Balance operations (deposits / withdrawals / credits)
# --------------------------------------------------------------------------

@router.get("/balance-operations", response_model=BalanceOperationsResponse)
async def get_balance_operations(
    request: Request,
    _user: str = Depends(get_current_user),
    days: int = Query(365, ge=1, le=3650),
):
    """
    Return the MT5 balance-operation history (deposits, withdrawals,
    credits) for the last ``days`` days.

    Pulled live from MT5 via account_monitor; no DB persistence. Safe
    to call repeatedly — response set is small.
    """
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    ops_raw = []
    monitor = getattr(ls, "account_monitor", None)
    if monitor is not None and hasattr(monitor, "fetch_balance_operations"):
        ops_raw = monitor.fetch_balance_operations(days=days)

    operations = [
        BalanceOperation(
            time=op.time,
            type=op.type,
            amount=op.amount,
            comment=op.comment,
            ticket=op.ticket,
        )
        for op in ops_raw
    ]
    return BalanceOperationsResponse(
        operations=operations, count=len(operations),
    )


@router.get("/account-ledger", response_model=BalanceOperationsResponse)
async def get_account_ledger(
    request: Request,
    _user: str = Depends(get_current_user),
    days: int = Query(365, ge=1, le=3650),
):
    """
    Full MT5-style account history: deposits, withdrawals, credits AND
    closed-trade P/L (one entry per closing deal). Useful when the user
    wants a single chronological ledger with running balance.

    For cash-flow only, see /balance-operations.
    """
    ls = _get_live_state(request)
    ls.dashboard_lock.touch()

    # Cache lookup — fetch_account_ledger() hits MT5 directly (no DB
    # shortcut), so each cold call pays MT5 round-trip latency. Keyed
    # per-account so a switch doesn't serve the prior account's data.
    account_id = ls.get_account_id() or 0
    cache_key = (account_id, days)
    _now = datetime.now(tz=timezone.utc)
    cached = _LEDGER_CACHE.get(cache_key)
    if cached is not None and (_now - cached[0]).total_seconds() < _HISTORY_CACHE_TTL_SEC:
        return cached[1]

    monitor = getattr(ls, "account_monitor", None)
    ops_raw = []
    if monitor is not None and hasattr(monitor, "fetch_account_ledger"):
        ops_raw = monitor.fetch_account_ledger(days=days)

    operations = [
        BalanceOperation(
            time=op.time,
            type=op.type,
            amount=op.amount,
            comment=op.comment,
            ticket=op.ticket,
        )
        for op in ops_raw
    ]
    response = BalanceOperationsResponse(
        operations=operations, count=len(operations),
    )
    _LEDGER_CACHE[cache_key] = (_now, response)
    return response
