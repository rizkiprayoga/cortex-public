"""
backfill_feature_store.py — populate `feature_store` from external collectors.

Phase 1F. Walks the 5 external fetchers' cached series and writes raw
observations to the partitioned `feature_store` table. Writes are
idempotent (PK conflict → skip) so re-runs are safe and incremental.

This is the **only** code path that writes to feature_store today. The
live bot's H4 tick is unaffected — fetcher in-memory caches still
serve the live read path.

Usage examples:

    # Backfill every source for one symbol
    python -m scripts.backfill_feature_store --symbols GBPUSD

    # Multiple symbols, one source
    python -m scripts.backfill_feature_store \\
        --symbols GBPUSD,EURUSD,AUDUSD --sources stooq

    # All live + new symbols, every source
    python -m scripts.backfill_feature_store --symbols ALL_LIVE_AND_NEW

    # Dry-run (skip the writes; show what would be done)
    python -m scripts.backfill_feature_store --symbols GBPUSD --dry-run

Environment:
    POSTGRES_DSN — required. Targets `trading_bot_dev` in dev workspace,
                    `trading_bot` in prod (after promotion).
    FRED_API_KEY  — required for fred_macro source. Skipped if missing.
    STOOQ_API_KEY — required for stooq source. Skipped if missing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.data_store import DataStore  # noqa: E402
from scripts._feature_store_common import (  # noqa: E402
    resolve_sources, resolve_symbols,
)

logger = logging.getLogger(__name__)


async def _persist_one(
    source: str,
    symbol: str,
    store: DataStore,
    dry_run: bool,
) -> tuple[int, str]:
    """
    Run one (source, symbol) backfill. Returns (rows_written, status_msg).
    Lazy-imports the fetcher so a missing optional dep (e.g. yfinance
    gone, no FRED key) doesn't crash the whole run.
    """
    try:
        if source == "ecb":
            # ECB writes _GLOBAL once; symbol arg ignored. Run only on the
            # first symbol in the iteration to avoid redundant rewrites.
            from src.data_pipeline.market.ecb_data import ECBDataFetcher
            fetcher = ECBDataFetcher()
            if dry_run:
                return (0, "would persist (_GLOBAL)")
            n = await fetcher.persist_raw_history_to_feature_store(store)
            return (n, f"{n} rows ({fetcher.FEATURE_GROUP}, _GLOBAL)")

        if source == "stooq":
            from src.data_pipeline.market.stooq_data import StooqFetcher
            fetcher = StooqFetcher()
            if dry_run:
                return (0, "would persist")
            n = await fetcher.persist_raw_history_to_feature_store(store, symbol)
            return (n, f"{n} rows ({fetcher.FEATURE_GROUP})")

        if source == "fred_macro":
            from src.data_pipeline.fundamental.macro_data import MacroDataFetcher
            try:
                fetcher = MacroDataFetcher()
            except EnvironmentError as exc:
                return (0, f"skipped — {exc}")
            if dry_run:
                return (0, "would persist")
            n = await fetcher.persist_raw_history_to_feature_store(store, symbol)
            return (n, f"{n} rows ({fetcher.FEATURE_GROUP})")

        if source == "cot":
            from src.data_pipeline.fundamental.cot_data import COTDataFetcher
            fetcher = COTDataFetcher()
            if dry_run:
                return (0, "would persist")
            n = await fetcher.persist_raw_history_to_feature_store(store, symbol)
            grp = (fetcher.FEATURE_GROUP_XAU if symbol.upper().startswith("XAU")
                   else fetcher.FEATURE_GROUP_FX)
            return (n, f"{n} rows ({grp})")

        if source == "yfinance":
            from src.data_pipeline.market.cross_asset import CrossAssetFetcher
            fetcher = CrossAssetFetcher()
            if dry_run:
                return (0, "would persist")
            n = await fetcher.persist_raw_history_to_feature_store(store, symbol)
            return (n, f"{n} rows ({fetcher.FEATURE_GROUP})")

        return (0, f"unknown source: {source}")
    except Exception as exc:  # noqa: BLE001 — keep one bad source from killing the rest
        logger.exception("persist %s/%s failed", source, symbol)
        return (0, f"FAILED: {exc.__class__.__name__}: {exc}")


async def _main(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv
    load_dotenv()

    if not os.environ.get("POSTGRES_DSN"):
        print("ERROR: POSTGRES_DSN not set in environment.", file=sys.stderr)
        return 1

    symbols = resolve_symbols(args.symbols)
    sources = resolve_sources(args.sources)
    if not symbols:
        print("ERROR: no symbols resolved from --symbols", file=sys.stderr)
        return 1
    if not sources:
        print("ERROR: no sources resolved from --sources", file=sys.stderr)
        return 1

    print("Backfilling feature_store")
    print(f"  symbols: {', '.join(symbols)}")
    print(f"  sources: {', '.join(sources)}")
    print(f"  dry_run: {args.dry_run}")
    print()

    store = DataStore()
    await store.connect()

    grand_total = 0
    try:
        ecb_done = False
        for source in sources:
            for symbol in symbols:
                # ECB is symbol-independent — only call once.
                if source == "ecb" and ecb_done:
                    continue
                tag = f"{source:>12s} / {symbol:<6s}"
                rows, msg = await _persist_one(source, symbol, store, args.dry_run)
                grand_total += rows
                print(f"  {tag}  {msg}")
                if source == "ecb":
                    ecb_done = True
    finally:
        await store.close()

    print()
    print(f"Total rows inserted: {grand_total}")
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Populate the feature_store table from external collectors.",
    )
    parser.add_argument(
        "--symbols", required=True,
        help="Comma-separated symbols, or 'ALL_LIVE' / 'ALL_NEW' / 'ALL'.",
    )
    parser.add_argument(
        "--sources", default="all",
        help=("Comma-separated sources from {fred_macro, cot, ecb, stooq, "
              "yfinance}, or 'all' (default)."),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Do not write — print what each (source, symbol) pair would persist.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
