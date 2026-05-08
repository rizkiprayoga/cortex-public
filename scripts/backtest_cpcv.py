"""
backtest_cpcv.py — Combinatorial Purged Cross-Validation orchestrator (A-6).

Implements the hybrid CPCV variant suitable for LSTM primary models:
 - N chronological groups (default 6), k_test per fold (default 2)
 - C(N, k) = 15 folds total
 - For each fold, LSTM is retrained on the LARGEST CONTIGUOUS PRE-TEST
   BLOCK (purged) — LSTMs can't train on holey data. HMM stays fixed.
 - Test set = the combo's test groups; metrics aggregated with mean ± σ.

The compromise: not canonical tree-CPCV (which trains on non-contiguous
row sets), but the LSTM-compatible equivalent. Gives you combinatorial
test-configuration coverage AND honest OOS scores per fold. See
``docs/BACKLOG.md`` A-6 archive for the full rationale.

Usage:
    python scripts/backtest_cpcv.py --symbols XAUUSD \\
        --n-groups 6 --k-test 2 \\
        --start 2021-01-01 --end 2026-04-01

Produces a markdown report at ``data/logs/cpcv/<symbol>_<ts>.md`` and
logs each fold to MLflow under experiment ``lstm_cpcv``.
"""
from __future__ import annotations

import argparse
import atexit
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backtest_cpcv")


LIVE_SYMBOLS = ("XAUUSD", "EURUSD", "USDJPY", "USDCAD", "ETHUSD")
_RUN_ID_RE = re.compile(r"run_id=([0-9a-f\-]{36})")


@dataclass
class FoldResult:
    combo: tuple[int, ...]
    train_start: str
    train_end: str
    test_windows: list[tuple[str, str]]
    profit_factor: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    win_rate: Optional[float] = None
    total_trades: Optional[int] = None
    net_pnl: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=list(LIVE_SYMBOLS))
    p.add_argument("--n-groups", type=int, default=6,
                   help="Number of chronological groups (default 6).")
    p.add_argument("--k-test", type=int, default=2,
                   help="Number of test groups per fold (default 2 → 15 combos).")
    p.add_argument("--start", default="2021-01-01",
                   help="Overall backtest window start (YYYY-MM-DD).")
    p.add_argument("--end", default="2026-04-01",
                   help="Overall backtest window end (YYYY-MM-DD).")
    p.add_argument("--purge-bars", type=int, default=80,
                   help="Training bars dropped BEFORE each test group "
                        "(default 80 H4 bars ~= 13 days; covers Triple-Barrier "
                        "max time_h4 label horizon).")
    p.add_argument("--embargo-bars", type=int, default=20,
                   help="Training bars dropped AFTER each test group.")
    p.add_argument("--out-dir", type=Path, default=Path("data/logs/cpcv"))
    p.add_argument("--epochs", type=int, default=60,
                   help="LSTM epochs per fold (lower than default to keep "
                        "compute bounded; early stopping usually trips sooner).")
    p.add_argument("--dry-run-combos", type=int, default=0,
                   help="If >0, run only the first N combinations. Use 2 for "
                        "a quick end-to-end smoke test.")
    return p.parse_args()


def _snapshot_live_models(label: str) -> None:
    from scripts.model_snapshot import cmd_save
    cmd_save(label, note="CPCV pre-batch safety snapshot", force=False)


def _restore_live_models(label: str) -> None:
    from scripts.model_snapshot import cmd_restore, SNAPSHOTS_DIR
    snap_dir = SNAPSHOTS_DIR / label
    if not snap_dir.exists():
        return
    cmd_restore(label, no_prompt=True)


def build_group_boundaries(
    h4_index: pd.DatetimeIndex, n_groups: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, int, int]]:
    """Partition the H4 bar index into ``n_groups`` contiguous chunks.

    Returns ``[(start_ts, end_ts, start_row, end_row), ...]`` so the CPCV
    loop can translate combo → concrete date ranges for train/test.
    """
    from src.ml.cpcv import split_index_into_groups
    groups = split_index_into_groups(h4_index, n_groups)
    out: list[tuple[pd.Timestamp, pd.Timestamp, int, int]] = []
    for rows in groups:
        out.append((
            h4_index[rows[0]], h4_index[rows[-1]],
            rows[0], rows[-1],
        ))
    return out


def _largest_contiguous_pre_test_block(
    group_bounds: list[tuple[pd.Timestamp, pd.Timestamp, int, int]],
    test_group_ids: tuple[int, ...], purge_bars: int,
) -> tuple[Optional[int], Optional[int]]:
    """Training window for one fold: all rows from row 0 up to (purged)
    the first test group's start. Returns (train_start_row, train_end_row)
    or (None, None) if no valid training block exists.
    """
    first_test_start_row = min(group_bounds[gid][2] for gid in test_group_ids)
    train_end_row = first_test_start_row - 1 - purge_bars
    train_start_row = 0
    if train_end_row - train_start_row < 200:   # too little data
        return None, None
    return train_start_row, train_end_row


def _row_to_date(h4_index: pd.DatetimeIndex, row: int) -> str:
    return h4_index[row].strftime("%Y-%m-%d")


def train_lstm_for_fold(
    symbol: str, train_start: str, train_end: str, epochs: int,
    timeout_s: int = 1800,
) -> bool:
    """Invoke train_deep_learning.py for one fold. Returns True on success.

    PyTorch on Windows can crash on CUDA shutdown with returncode
    STATUS_STACK_BUFFER_OVERRUN (0xC0000409 = 3221226505) AFTER the model
    has already been saved successfully. To work around that, we check
    model-file freshness as the ground-truth success signal, and only
    treat returncode as informational.
    """
    cmd = [
        sys.executable, "scripts/train_deep_learning.py",
        "--symbols", symbol,
        "--triple-barrier",
        "--pca-components", "25",
        "--no-snapshot",
        "--start-date", train_start,
        "--end-date", train_end,
        "--epochs", str(epochs),
    ]
    model_file = Path("data/models") / f"lstm_{symbol}.pt"
    mtime_before = model_file.stat().st_mtime if model_file.exists() else 0

    logger.info("  [train] %s %s..%s", symbol, train_start, train_end)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, check=False,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        logger.error("  [train] %s TIMED OUT after %ds", symbol, timeout_s)
        return False
    elapsed = time.monotonic() - t0

    # Success check: model file updated after we started the subprocess.
    if model_file.exists() and model_file.stat().st_mtime > mtime_before:
        if proc.returncode != 0:
            logger.warning(
                "  [train] %s model saved but subprocess exited %d "
                "(probable PyTorch CUDA-shutdown crash on Windows — benign)",
                symbol, proc.returncode,
            )
        logger.info("  [train] %s done in %.1fs", symbol, elapsed)
        return True

    logger.error(
        "  [train] %s failed: no fresh model file. exit=%d stderr tail:\n%s",
        symbol, proc.returncode, (proc.stderr or "")[-1500:],
    )
    return False


def run_backtest_slice(
    symbol: str, start: str, end: str, timeout_s: int = 600,
) -> Optional[str]:
    """Run scripts/backtest.py over a slice. Returns run_id or None."""
    cmd = [
        sys.executable, "scripts/backtest.py",
        "--mode", "full", "--symbols", symbol,
        "--start", start, "--end", end,
    ]
    logger.info("  [backtest] %s %s..%s", symbol, start, end)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, check=False,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        logger.error("  [backtest] %s TIMEOUT", symbol)
        return None
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        logger.error("  [backtest] %s exit %d: %s",
                     symbol, proc.returncode, combined[-1500:])
        return None
    m = _RUN_ID_RE.search(combined)
    return m.group(1) if m else None


def _compute_fold_pf_from_trades(
    symbol: str, test_windows: list[tuple[str, str]],
) -> Optional[float]:
    """Proper PF computed from trade-level pnls across all test windows.

    PF = sum(winners) / |sum(losers)|. Capped at 20.0 when there are no
    losers (distinguishes "very good" from "undefined").
    """
    import asyncio, asyncpg

    if not test_windows:
        return None

    async def _fetch() -> list[float]:
        dsn = os.environ["POSTGRES_DSN"].replace(
            "postgresql+asyncpg://", "postgresql://",
        )
        conn = await asyncpg.connect(dsn)
        try:
            all_pnls: list[float] = []
            for start, end in test_windows:
                rows = await conn.fetch(
                    """
                    SELECT pnl FROM backtest_trades
                    WHERE symbol = $1
                      AND entry_time >= $2
                      AND entry_time <= $3
                    """, symbol, start, end + " 23:59:59",
                )
                all_pnls.extend(float(r["pnl"]) for r in rows if r["pnl"] is not None)
            return all_pnls
        finally:
            await conn.close()

    try:
        pnls = asyncio.run(_fetch())
    except Exception as exc:
        logger.warning("[%s] fold PF query failed: %s", symbol, exc)
        return None
    if not pnls:
        return None
    wins = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses <= 0:
        return 20.0 if wins > 0 else 0.0
    return round(wins / losses, 3)


def fetch_backtest_metrics(run_id: str) -> Optional[dict]:
    import asyncio, asyncpg

    async def _fetch() -> Optional[dict]:
        dsn = os.environ["POSTGRES_DSN"].replace(
            "postgresql+asyncpg://", "postgresql://",
        )
        conn = await asyncpg.connect(dsn)
        try:
            row = await conn.fetchrow(
                """
                SELECT profit_factor, max_drawdown_pct, win_rate,
                       total_trades, net_pnl, sharpe_ratio
                FROM backtest_runs WHERE id = $1
                """, run_id,
            )
        finally:
            await conn.close()
        if row is None:
            return None
        return {k: row[k] for k in row.keys()}

    return asyncio.run(_fetch())


def write_report(
    symbol: str, fold_results: list[FoldResult], out_dir: Path,
    n_groups: int, k_test: int, purge: int, embargo: int,
    overall_start: str, overall_end: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"{symbol}_{ts}.md"

    from src.ml.cpcv import aggregate_fold_metrics
    successes = [r for r in fold_results if r.error is None and r.profit_factor is not None]
    agg = aggregate_fold_metrics([
        {
            "profit_factor": r.profit_factor,
            "max_drawdown_pct": r.max_drawdown_pct,
            "win_rate": r.win_rate,
            "total_trades": r.total_trades,
            "net_pnl": r.net_pnl,
            "sharpe_ratio": r.sharpe_ratio,
        }
        for r in successes
    ])

    lines = [
        f"# CPCV report — {symbol}",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}Z",
        f"- Overall window: {overall_start} → {overall_end}",
        f"- N groups = {n_groups}, k test = {k_test} → {len(fold_results)} folds",
        f"- Purge = {purge} H4 bars, Embargo = {embargo} H4 bars",
        f"- Successful folds: {len(successes)} / {len(fold_results)}",
        "",
        "## Aggregate (mean ± std across successful folds)",
        "",
        "| metric | mean | std | n |",
        "| --- | ---: | ---: | ---: |",
    ]
    if agg:
        for key, stats in agg.items():
            lines.append(
                f"| {key} | {stats['mean']:.3f} | {stats['std']:.3f} | {int(stats['n_folds'])} |"
            )
    else:
        lines.append("| _no successful folds_ |  |  |  |")
    lines.append("")
    lines.append("## Per-fold detail")
    lines.append("")
    lines.append("| combo | train range | test ranges | PF | DD % | WR | trades | error |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for r in fold_results:
        test_ranges = " + ".join(f"{a}..{b}" for a, b in r.test_windows)
        lines.append(
            f"| {r.combo} | {r.train_start}..{r.train_end} | {test_ranges} | "
            f"{r.profit_factor if r.profit_factor is not None else '—'} | "
            f"{r.max_drawdown_pct if r.max_drawdown_pct is not None else '—'} | "
            f"{'' if r.win_rate is None else f'{r.win_rate * 100:.1f}%'} | "
            f"{r.total_trades if r.total_trades is not None else '—'} | "
            f"{r.error or ''} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("CPCV report: %s", path)
    return path


def run_symbol(
    symbol: str, args, group_bounds,
) -> tuple[list[FoldResult], Path]:
    from src.ml.cpcv import enumerate_cpcv_combos

    combos = enumerate_cpcv_combos(n_groups=args.n_groups, k_test=args.k_test)
    if args.dry_run_combos and args.dry_run_combos < len(combos):
        # Take the LAST N — those have plenty of pre-test data. The early
        # combos (where group 0 is a test group) get skipped for lack of
        # pre-test training data, which is accurate but not informative
        # for a smoke test.
        combos = combos[-args.dry_run_combos:]
    logger.info("=== %s: %d combos ===", symbol, len(combos))

    fold_results: list[FoldResult] = []

    for i, combo in enumerate(combos, 1):
        logger.info("[%s fold %d/%d] combo=%s",
                    symbol, i, len(combos), combo)

        first_test_row = min(group_bounds[gid][2] for gid in combo)
        train_end_row = first_test_row - 1 - args.purge_bars
        if train_end_row < 200:
            logger.warning("[%s fold %s] insufficient pre-test data; skipping",
                           symbol, combo)
            fold_results.append(FoldResult(
                combo=combo,
                train_start="-", train_end="-",
                test_windows=[(
                    group_bounds[gid][0].strftime("%Y-%m-%d"),
                    group_bounds[gid][1].strftime("%Y-%m-%d"),
                ) for gid in combo],
                error="insufficient pre-test data",
            ))
            continue

        train_start = args.start   # always start from overall window start
        train_end = group_bounds[0][1].strftime("%Y-%m-%d")   # placeholder
        first_test_ts = min(group_bounds[gid][0] for gid in combo)
        train_end = (first_test_ts - pd.Timedelta(days=args.purge_bars // 6 + 1)).strftime("%Y-%m-%d")

        # Train LSTM on the pre-test contiguous block
        if not train_lstm_for_fold(symbol, train_start, train_end, args.epochs):
            fold_results.append(FoldResult(
                combo=combo, train_start=train_start, train_end=train_end,
                test_windows=[(
                    group_bounds[gid][0].strftime("%Y-%m-%d"),
                    group_bounds[gid][1].strftime("%Y-%m-%d"),
                ) for gid in combo],
                error="training failed",
            ))
            continue

        # Backtest each test window separately and aggregate
        fold_pnls: list[float] = []
        fold_trades = 0
        fold_wins = 0
        fold_max_dd = 0.0
        test_wins: list[tuple[str, str]] = []
        for gid in combo:
            g_start = group_bounds[gid][0].strftime("%Y-%m-%d")
            g_end = group_bounds[gid][1].strftime("%Y-%m-%d")
            test_wins.append((g_start, g_end))
            run_id = run_backtest_slice(symbol, g_start, g_end)
            if run_id is None:
                continue
            metrics = fetch_backtest_metrics(run_id)
            if metrics is None:
                continue
            trades_n = int(metrics.get("total_trades") or 0)
            wr = float(metrics.get("win_rate") or 0.0)
            pnl = float(metrics.get("net_pnl") or 0.0)
            dd = float(metrics.get("max_drawdown_pct") or 0.0)
            fold_trades += trades_n
            fold_wins += int(round(wr * trades_n))
            fold_pnls.append(pnl)
            if dd > fold_max_dd:
                fold_max_dd = dd

        # Aggregate fold-level metrics. Query backtest_trades for a proper
        # trade-level PF (not the coarse per-window sum which can yield
        # inf when one window has no losers).
        total_pnl = sum(fold_pnls) if fold_pnls else 0.0
        pf = _compute_fold_pf_from_trades(symbol, test_wins)
        wr_fold = (fold_wins / fold_trades) if fold_trades else None

        fold_results.append(FoldResult(
            combo=combo,
            train_start=train_start, train_end=train_end,
            test_windows=test_wins,
            profit_factor=pf,
            max_drawdown_pct=fold_max_dd or None,
            win_rate=wr_fold,
            total_trades=fold_trades or None,
            net_pnl=total_pnl or None,
            sharpe_ratio=None,   # per-window sharpe is not easily combined
        ))

    report_path = write_report(
        symbol, fold_results, args.out_dir,
        args.n_groups, args.k_test,
        args.purge_bars, args.embargo_bars,
        args.start, args.end,
    )

    # Log aggregate to MLflow for the registry
    try:
        import mlflow
        from src.ml.registry import start_run
        from src.ml.cpcv import aggregate_fold_metrics

        successes = [
            {
                "profit_factor": r.profit_factor,
                "max_drawdown_pct": r.max_drawdown_pct,
                "win_rate": r.win_rate,
                "total_trades": r.total_trades,
            }
            for r in fold_results if r.error is None and r.profit_factor is not None
        ]
        agg = aggregate_fold_metrics(successes)
        run_name = f"{symbol}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        with start_run(
            experiment="lstm_cpcv",
            run_name=run_name,
            tags={"symbol": symbol, "training_script": "backtest_cpcv.py"},
        ):
            mlflow.log_params({
                "symbol": symbol, "n_groups": args.n_groups,
                "k_test": args.k_test, "purge_bars": args.purge_bars,
                "embargo_bars": args.embargo_bars,
                "n_combos_total": len(fold_results),
                "n_combos_successful": len(successes),
            })
            for k, stats in agg.items():
                mlflow.log_metric(f"{k}_mean", stats["mean"])
                mlflow.log_metric(f"{k}_std", stats["std"])
            mlflow.log_artifact(str(report_path))
    except Exception as _exc:
        logger.warning("MLflow logging skipped: %s", _exc)

    return fold_results, report_path


async def _build_groups_db_direct(args: argparse.Namespace) -> dict:
    """Read H4 OHLCV from DataStore (no MT5) and build per-symbol group
    boundaries. Refactored 2026-04-29 from sync MT5 path that was
    hijacking the prod MT5 terminal — see the the universe sweep sprint incident in
    ``project_sprint_1_5_verdict.md``. CPCV is now MT5-free; safe to
    run while prod bot is live.
    """
    from src.data_pipeline.data_store import DataStore
    from src.data_pipeline.mt5_feed import MT5DataFeed

    store = DataStore()
    await store.connect()
    try:
        feed = MT5DataFeed(connector=None, data_store=store)
        per_symbol_groups: dict[str, list] = {}
        for symbol in args.symbols:
            h4 = await feed.get_historical_db_only(symbol, "H4", bars=99999)
            if h4 is None or len(h4) < 500:
                logger.warning("[%s] too little H4 history; skipping", symbol)
                continue
            start_ts = pd.Timestamp(args.start)
            end_ts = pd.Timestamp(args.end)
            h4 = h4[(h4.index >= start_ts) & (h4.index <= end_ts)]
            if len(h4) < 500:
                logger.warning("[%s] after window filter only %d bars; skipping",
                               symbol, len(h4))
                continue
            per_symbol_groups[symbol] = build_group_boundaries(
                h4.index, args.n_groups,
            )
            logger.info("[%s] H4 window: %s → %s (%d bars), groups=%d",
                        symbol, h4.index[0], h4.index[-1],
                        len(h4), args.n_groups)
        return per_symbol_groups
    finally:
        await store.close()


def main() -> int:
    import asyncio
    args = parse_args()

    # Safety snapshot (atexit + finally, mirrors model_bench). Per-fold
    # training overwrites live model files, so snapshot/restore stays.
    snap_label = f"pre-cpcv-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    _snapshot_live_models(snap_label)
    atexit.register(_restore_live_models, snap_label)
    logger.info("Live models snapshotted as '%s'", snap_label)

    try:
        # DB-direct boundary read — no MT5 contact, safe with prod live.
        per_symbol_groups = asyncio.run(_build_groups_db_direct(args))

        all_symbol_summaries: list[tuple[str, Path, int, int]] = []
        for symbol, group_bounds in per_symbol_groups.items():
            fold_results, report_path = run_symbol(symbol, args, group_bounds)
            successes = sum(1 for r in fold_results
                            if r.error is None and r.profit_factor is not None)
            all_symbol_summaries.append(
                (symbol, report_path, successes, len(fold_results))
            )

        logger.info("")
        logger.info("===== CPCV complete =====")
        for sym, path, succ, total in all_symbol_summaries:
            logger.info("  %s: %d/%d successful folds — %s",
                        sym, succ, total, path)

    finally:
        _restore_live_models(snap_label)

    return 0


if __name__ == "__main__":
    sys.exit(main())
