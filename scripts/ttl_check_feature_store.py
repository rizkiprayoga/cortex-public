"""
ttl_check_feature_store.py — weekly safety-net cron for feature_store.

Phase 1G second half. ``feature_store`` rows are immutable-by-timestamp:
once written, the bulk path uses ``ON CONFLICT DO NOTHING`` so re-runs
of ``backfill_feature_store.py`` are cheap no-ops.

But upstream sources occasionally REVISE published values:
  - FRED revises GDP/CPI estimates 1-3 months after first release
  - CFTC corrects clerical errors in COT reports
  - Yahoo backfills missing ticker data
  - ECB occasionally republishes corrected curve points

A skip-on-conflict cache silently misses these revisions, leaving stale
values in feature_store potentially for months. This script re-fetches
the last N days from each source and writes with ``DO UPDATE``, so any
revisions land. Operator-visible row counts let you spot when a source
revised something.

Usage:

    # Default: re-check the last 7 days for all 5 sources × all symbols
    python -m scripts.ttl_check_feature_store

    # Tighter window after a known revision event
    python -m scripts.ttl_check_feature_store --days 14

    # Subset of symbols / sources
    python -m scripts.ttl_check_feature_store --symbols GBPUSD --sources fred_macro,cot

    # Dry-run (skip writes; show counts only)
    python -m scripts.ttl_check_feature_store --dry-run

Schedule: weekly, e.g. Sundays at 04:00 UTC via Windows Scheduled Task.
Exit code 0 always (rows-touched is informational, not pass/fail).
Pipe stdout to a daily log file for grep-able row-touched history.

Note: each persist method's ``force=True`` flag activates DO UPDATE
on the bulk insert. ``written_at`` is refreshed on every TTL pass even
for unchanged rows — that's the cost of catching revisions without
per-row diff logic. Operators inspecting feature_store can ignore
``written_at`` for staleness analysis (use the row's ``timestamp``).

Environment:
    POSTGRES_DSN — required.
    FRED_API_KEY — required for fred_macro source. Skipped if missing.
    STOOQ_API_KEY — required for stooq source. Skipped if missing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.data_store import DataStore  # noqa: E402
from scripts._feature_store_common import (  # noqa: E402
    SOURCES, resolve_sources, resolve_symbols,
)

logger = logging.getLogger(__name__)

_DEFAULT_DAYS = 7


async def _check_one(
    source: str, symbol: str, store: DataStore, days: int, dry_run: bool,
) -> tuple[int, str]:
    """Run one (source, symbol) TTL check. Returns (rows_touched, status)."""
    try:
        if source == "ecb":
            from src.data_pipeline.market.ecb_data import ECBDataFetcher
            fetcher = ECBDataFetcher()
            if dry_run:
                return (0, "would TTL-check (_GLOBAL)")
            n = await fetcher.persist_raw_history_to_feature_store(
                store, force=True, lookback_days=days,
            )
            return (n, f"{n} rows touched ({fetcher.FEATURE_GROUP}, _GLOBAL)")

        if source == "stooq":
            from src.data_pipeline.market.stooq_data import StooqFetcher
            fetcher = StooqFetcher()
            if dry_run:
                return (0, "would TTL-check")
            n = await fetcher.persist_raw_history_to_feature_store(
                store, symbol, force=True, lookback_days=days,
            )
            return (n, f"{n} rows touched ({fetcher.FEATURE_GROUP})")

        if source == "fred_macro":
            from src.data_pipeline.fundamental.macro_data import MacroDataFetcher
            try:
                fetcher = MacroDataFetcher()
            except EnvironmentError as exc:
                return (0, f"skipped — {exc}")
            if dry_run:
                return (0, "would TTL-check")
            n = await fetcher.persist_raw_history_to_feature_store(
                store, symbol, force=True, lookback_days=days,
            )
            return (n, f"{n} rows touched ({fetcher.FEATURE_GROUP})")

        if source == "cot":
            from src.data_pipeline.fundamental.cot_data import COTDataFetcher
            fetcher = COTDataFetcher()
            if dry_run:
                return (0, "would TTL-check")
            n = await fetcher.persist_raw_history_to_feature_store(
                store, symbol, force=True, lookback_days=days,
            )
            grp = (fetcher.FEATURE_GROUP_XAU if symbol.upper().startswith("XAU")
                   else fetcher.FEATURE_GROUP_FX)
            return (n, f"{n} rows touched ({grp})")

        if source == "yfinance":
            from src.data_pipeline.market.cross_asset import CrossAssetFetcher
            fetcher = CrossAssetFetcher()
            if dry_run:
                return (0, "would TTL-check")
            n = await fetcher.persist_raw_history_to_feature_store(
                store, symbol, force=True, lookback_days=days,
            )
            return (n, f"{n} rows touched ({fetcher.FEATURE_GROUP})")

        return (0, f"unknown source: {source}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("TTL check %s/%s failed", source, symbol)
        return (0, f"FAILED: {exc.__class__.__name__}: {exc}")


async def _run(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv
    load_dotenv()

    if not os.environ.get("POSTGRES_DSN"):
        print("ERROR: POSTGRES_DSN not set", file=sys.stderr)
        return 1

    symbols = resolve_symbols(args.symbols)
    sources = resolve_sources(args.sources)

    print(f"feature_store TTL check (lookback={args.days} days)")
    print(f"  symbols: {', '.join(symbols)}")
    print(f"  sources: {', '.join(sources)}")
    print(f"  dry_run: {args.dry_run}")
    print(f"  started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print()

    store = DataStore()
    await store.connect()

    grand_total = 0
    try:
        ecb_done = False
        for source in sources:
            for symbol in symbols:
                # ECB is _GLOBAL; only call once.
                if source == "ecb" and ecb_done:
                    continue
                tag = f"{source:>12s} / {symbol:<6s}"
                rows, msg = await _check_one(source, symbol, store, args.days, args.dry_run)
                grand_total += rows
                print(f"  {tag}  {msg}")
                if source == "ecb":
                    ecb_done = True
    finally:
        await store.close()

    print()
    print(f"Total rows touched: {grand_total}")
    print(f"  finished: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    # Always exit 0 — rows-touched is informational, not pass/fail.
    # An operator alarm should fire on log-line patterns (FAILED/skipped),
    # not on exit code.
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Weekly TTL safety-net for feature_store — picks up upstream revisions.",
    )
    parser.add_argument(
        "--symbols", default="ALL",
        help="Comma-list, or 'ALL_LIVE'/'ALL_NEW'/'ALL'. Default: ALL.",
    )
    parser.add_argument(
        "--sources", default="all",
        help=f"Comma-list from {SOURCES} or 'all' (default).",
    )
    parser.add_argument(
        "--days", type=int, default=_DEFAULT_DAYS,
        help=f"Lookback window in days (default: {_DEFAULT_DAYS}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip writes; print what each pair would TTL-check.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
