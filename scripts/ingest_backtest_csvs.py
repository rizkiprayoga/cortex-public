"""
ingest_backtest_csvs.py — one-shot import of historical CSV backtests into DB.

Historical backtests run via `python scripts/backtest.py` only wrote
`data/logs/backtest_equity_<SYMBOL>.csv` + `backtest_trades_<SYMBOL>.csv`
— never the `backtest_runs` DB table — so the dashboard never showed
them. This script reads every existing CSV pair and creates matching
DB rows so they appear in the Backtest screen.

Usage:
    python scripts/ingest_backtest_csvs.py --dry-run       # preview
    python scripts/ingest_backtest_csvs.py                 # do it
    python scripts/ingest_backtest_csvs.py --force         # re-import
    python scripts/ingest_backtest_csvs.py --timeframe H1  # override

Idempotent: a symbol+start+end combo that already exists in DB is
skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Project imports (make sure repo root is importable)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data_pipeline.data_store import DataStore  # noqa: E402
from scripts.backtest import compute_summary  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_backtest_csvs")

LOGS_DIR = REPO / "data" / "logs"


def pair_csvs() -> list[tuple[str, Path, Path]]:
    """Return (symbol, trades_csv, equity_csv) triples found in data/logs."""
    triples: list[tuple[str, Path, Path]] = []
    for trades_path in sorted(LOGS_DIR.glob("backtest_trades_*.csv")):
        symbol = trades_path.stem.replace("backtest_trades_", "")
        equity_path = LOGS_DIR / f"backtest_equity_{symbol}.csv"
        if not equity_path.exists():
            logger.warning("skip %s: no matching equity CSV", symbol)
            continue
        triples.append((symbol, trades_path, equity_path))
    return triples


_TRADE_COLUMNS = {
    "symbol", "direction", "entry_time", "exit_time", "entry_price",
    "exit_price", "pnl", "r_multiple", "exit_reason", "strategy_name",
    "regime_label", "combined_score",
}


def read_pair(trades_path: Path, equity_path: Path):
    """Return (equity_curve_list, trades_list) in the shape compute_summary expects."""
    trades_df = pd.read_csv(trades_path)
    equity_df = pd.read_csv(equity_path)

    # Drop columns not in BacktestTrade ORM (e.g. `commission` from R-1
    # friction output). Keep the CSV itself as the forensic source of
    # truth; the DB only needs what the dashboard renders.
    drop_cols = [c for c in trades_df.columns if c not in _TRADE_COLUMNS]
    if drop_cols:
        trades_df = trades_df.drop(columns=drop_cols)

    trades = trades_df.to_dict(orient="records")
    equity_curve = equity_df.to_dict(orient="records")
    return equity_curve, trades


def derive_span(trades: list[dict]) -> tuple[str, str]:
    """Earliest entry_time and latest exit_time as ISO strings."""
    if not trades:
        return "", ""
    entry_times = [t.get("entry_time") for t in trades if t.get("entry_time")]
    exit_times = [t.get("exit_time") for t in trades if t.get("exit_time")]
    start = min(entry_times) if entry_times else ""
    end = max(exit_times) if exit_times else ""
    return str(start), str(end)


async def already_imported(
    ds: DataStore, symbol: str, start_date: str, end_date: str, timeframe: str,
) -> bool:
    """True if a non-failed row with matching symbol/timeframe/period exists."""
    runs = await ds.list_backtest_runs(limit=500)
    for r in runs:
        if (
            r["symbol"] == symbol
            and r["timeframe"] == timeframe
            and (r.get("start_date") or "").startswith(start_date[:10])
            and (r.get("end_date") or "").startswith(end_date[:10])
            and r["status"] != "failed"
        ):
            return True
    return False


async def ingest_one(
    ds: DataStore,
    symbol: str,
    trades_path: Path,
    equity_path: Path,
    timeframe: str,
    mode: str,
    dry_run: bool,
    force: bool,
) -> str | None:
    equity_curve, trades = read_pair(trades_path, equity_path)
    if not trades:
        logger.warning("%s: trades CSV empty, skip", symbol)
        return None

    start_date, end_date = derive_span(trades)
    summary = compute_summary(equity_curve, trades)

    # Human-readable preview line
    logger.info(
        "%s %s→%s [%s] trades=%d pnl=%+.2f win=%.1f%% pf=%.2f dd=%.2f%% sharpe=%.3f",
        symbol, start_date[:10], end_date[:10], timeframe,
        summary["total_trades"], summary["net_pnl"],
        summary["win_rate"] * 100.0, summary["profit_factor"],
        summary["max_drawdown_pct"], summary["sharpe_ratio"],
    )

    if not force and await already_imported(ds, symbol, start_date, end_date, timeframe):
        logger.info("  ↳ already imported, skipping (pass --force to re-import)")
        return None

    if dry_run:
        return None

    run_id = str(uuid.uuid4())
    # calmar_ratio is a derived field — no DB column yet; API computes
    # it on read from persisted fields.
    db_summary = {k: v for k, v in summary.items() if k != "calmar_ratio"}
    await ds.create_backtest_run({
        "id": run_id,
        "status": "done",
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "run_mode": mode,
        "finished_at": end_date,
        **db_summary,
    })
    eq_rows = [{"run_id": run_id, **e} for e in equity_curve]
    tr_rows = [{"run_id": run_id, **t} for t in trades]
    await ds.bulk_insert_backtest_equity(eq_rows)
    await ds.bulk_insert_backtest_trades(tr_rows)
    logger.info("  ↳ imported as run_id=%s (%d equity, %d trades)",
                 run_id, len(eq_rows), len(tr_rows))
    return run_id


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only, no DB writes.")
    parser.add_argument("--force", action="store_true",
                        help="Re-import even if a matching run already exists.")
    parser.add_argument("--timeframe", default="H4",
                        help="Timeframe to tag these runs with (default H4).")
    parser.add_argument("--mode", default="full",
                        help="Run mode to record (default 'full').")
    parser.add_argument("--symbol", default=None,
                        help="Ingest only this symbol; otherwise all pairs.")
    args = parser.parse_args()

    load_dotenv()

    triples = pair_csvs()
    if args.symbol:
        triples = [t for t in triples if t[0] == args.symbol.upper()]
    if not triples:
        logger.error("No CSV pairs found in %s", LOGS_DIR)
        return

    ds = DataStore()
    await ds.connect()

    imported = 0
    try:
        for symbol, trades_path, equity_path in triples:
            run_id = await ingest_one(
                ds, symbol, trades_path, equity_path,
                timeframe=args.timeframe, mode=args.mode,
                dry_run=args.dry_run, force=args.force,
            )
            if run_id:
                imported += 1
    finally:
        await ds.close()

    if args.dry_run:
        logger.info("DRY RUN complete — %d pair(s) inspected.", len(triples))
    else:
        logger.info("Done. Imported %d new run(s).", imported)


if __name__ == "__main__":
    asyncio.run(main())
