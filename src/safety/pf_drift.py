"""
pf_drift.py — Live-vs-backtest profit-factor drift monitor (A-4).

Fires the ``strategy.live_pf_drift`` invariant when rolling live PF per
symbol deviates materially below its backtest baseline. Early warning
for regime shift, feature staleness, data-pipeline regression, or
broker execution degradation — well before cumulative P&L tells the
same story.

Design
------
Baseline PF comes from the most recent ``run_mode='full'`` backtest run
for the symbol (table ``backtest_runs``). Live PF comes from the last
``window_days`` of closed trades (table ``trades``), optionally
account-scoped.

The check is asymmetric: we only alarm on *downside* deviation. Live
outperforming backtest is good news, not a drift event.

Thresholds (downside ratio = live_pf / baseline_pf):
  * ratio >= 0.80 → ok, no invariant
  * 0.70 ≤ ratio < 0.80 → WARN
  * ratio <  0.70 → ALERT (Telegram once per 24h per symbol)

Skip conditions (silent, log info only — no invariant fires):
  * No baseline found for symbol (never run a full backtest)
  * Fewer than ``MIN_TRADES_TO_CHECK`` closed trades in the window
    (low-N noise dominates; waiting for more signal)
  * Live PF is undefined (gross_loss == 0 — either no losses at all or
    no trades; either way, not a drift event)

Deliberately not addressed in v1:
  * Regime-stratified comparison (live in chop vs backtest in trend).
    Future work once R-4 (trade attribution) lands.
  * Variance-adjusted thresholds. Current fixed ratios will false-positive
    on small-N windows in volatile regimes; the 10-trade floor is a
    rough compensation.
  * Baseline freshness guard. If the last backtest was 6+ months ago the
    comparison is noisy but still useful; ops should re-run backtests
    after every retrain anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from loguru import logger
from sqlalchemy import select, and_

from src.data_pipeline.data_store import BacktestRun, TradeRecord, _dt_to_iso
from src.safety.invariants import Severity, check as _inv_check


WARN_RATIO = 0.80
ALERT_RATIO = 0.70
MIN_TRADES_TO_CHECK = 10
DEFAULT_WINDOW_DAYS = 30


@dataclass(frozen=True)
class DriftCheckResult:
    """Outcome of a single drift check. ``severity`` is None when the
    check was skipped (insufficient data) or passed."""
    symbol: str
    baseline_pf: Optional[float]
    live_pf: Optional[float]
    ratio: Optional[float]
    n_trades: int
    severity: Optional[Severity]
    reason: str


def compute_pf_from_pnls(pnls: list[float]) -> Optional[float]:
    """PF = gross_profit / gross_loss. None when gross_loss == 0 (undefined)."""
    gross_profit = sum(p for p in pnls if p and p > 0)
    gross_loss = sum(-p for p in pnls if p and p < 0)
    if gross_loss <= 0:
        return None
    return gross_profit / gross_loss


async def get_baseline_pf(data_store, symbol: str) -> Optional[float]:
    """
    Most recent completed ``run_mode='full'`` backtest PF for ``symbol``.
    Returns None if no baseline exists.
    """
    stmt = (
        select(BacktestRun.profit_factor)
        .where(
            and_(
                BacktestRun.symbol == symbol,
                BacktestRun.status == "done",
                BacktestRun.run_mode == "full",
                BacktestRun.profit_factor > 0,
            )
        )
        .order_by(BacktestRun.created_at.desc())
        .limit(1)
    )
    async with data_store._session_factory() as session:
        result = await session.execute(stmt)
        pf = result.scalar_one_or_none()
    return float(pf) if pf is not None else None


async def compute_live_pf(
    data_store,
    symbol: str,
    account_id: Optional[int] = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> tuple[Optional[float], int]:
    """
    (live_pf, n_trades) for closed trades in the last ``window_days``.
    When ``account_id`` is set, scopes to that MT5 account.

    PF is None when no losses exist in the window (undefined); ``n`` is
    still reported so the caller can distinguish "nothing happened" from
    "only winners."
    """
    since = datetime.now(tz=timezone.utc) - timedelta(days=window_days)
    conditions = [
        TradeRecord.symbol == symbol,
        TradeRecord.timestamp_close.isnot(None),
        TradeRecord.timestamp_close >= _dt_to_iso(since),
    ]
    if account_id is not None:
        conditions.append(TradeRecord.mt5_account == account_id)

    stmt = select(TradeRecord.pnl_usd).where(and_(*conditions))
    async with data_store._session_factory() as session:
        result = await session.execute(stmt)
        pnls = [float(p) for p in result.scalars().all() if p is not None]

    return compute_pf_from_pnls(pnls), len(pnls)


async def check_pf_drift(
    data_store,
    symbol: str,
    account_id: Optional[int] = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> DriftCheckResult:
    """
    Run one drift check. Fires ``strategy.live_pf_drift`` invariant if
    live PF is materially below baseline. Safe to call repeatedly —
    the invariant registry deduplicates ALERT Telegrams per-symbol per-24h.
    """
    baseline = await get_baseline_pf(data_store, symbol)
    live, n = await compute_live_pf(data_store, symbol, account_id, window_days)

    if baseline is None or baseline <= 0:
        return DriftCheckResult(
            symbol, baseline, live, None, n, None, "no_baseline",
        )
    if n < MIN_TRADES_TO_CHECK:
        return DriftCheckResult(
            symbol, baseline, live, None, n, None,
            f"insufficient_trades({n}<{MIN_TRADES_TO_CHECK})",
        )
    if live is None:
        return DriftCheckResult(
            symbol, baseline, live, None, n, None, "no_losses_in_window",
        )

    ratio = live / baseline
    if ratio >= WARN_RATIO:
        return DriftCheckResult(
            symbol, baseline, live, ratio, n, None,
            f"ok (ratio={ratio:.2f})",
        )

    severity = Severity.ALERT if ratio < ALERT_RATIO else Severity.WARN
    msg = (
        f"live PF {live:.2f} is {ratio:.0%} of backtest baseline "
        f"{baseline:.2f} over last {window_days}d / {n} trades"
    )
    _inv_check(
        "strategy.live_pf_drift",
        condition=False,
        severity=severity,
        symbol=symbol,
        context={
            "baseline_pf": round(baseline, 3),
            "live_pf": round(live, 3),
            "ratio": round(ratio, 3),
            "n_trades": n,
            "window_days": window_days,
        },
        dedup_key=f"strategy.live_pf_drift:{symbol}",
        message=msg,
    )
    return DriftCheckResult(
        symbol, baseline, live, ratio, n, severity, msg,
    )


async def run_drift_checks(
    data_store,
    symbols: list[str],
    account_id: Optional[int] = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> list[DriftCheckResult]:
    """
    Run drift check for every symbol in ``symbols``. One bad symbol
    doesn't halt the rest — per-symbol exceptions are swallowed and
    logged so the scheduler job always completes.
    """
    results: list[DriftCheckResult] = []
    for sym in symbols:
        try:
            r = await check_pf_drift(data_store, sym, account_id, window_days)
        except Exception as exc:
            logger.warning(
                "pf_drift: check failed for {}: {}", sym, exc,
            )
            r = DriftCheckResult(
                sym, None, None, None, 0, None, f"error: {exc}",
            )
        results.append(r)
    return results
