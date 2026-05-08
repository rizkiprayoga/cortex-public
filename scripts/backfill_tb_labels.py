"""
backfill_tb_labels.py — populate actual_outcomes.actual_tb_label for
existing rows that predate the metric fix.

After backfilling labels, clears existing prediction_errors so they
get recomputed against the correct (TB-label) ground truth on the
next FeedbackLoop tick — avoids stuck "0% accuracy" rows that compared
TB-scale predictions against log-return-scale actuals.

Usage:
    python scripts/backfill_tb_labels.py                 # all symbols, dry safe
    python scripts/backfill_tb_labels.py --symbol EURUSD
    python scripts/backfill_tb_labels.py --no-clear      # don't drop prediction_errors
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data_pipeline.data_store import DataStore  # noqa: E402
from src.data_pipeline.feedback_loop import FeedbackLoop  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_tb_labels")

LIVE_SYMBOLS = ("XAUUSD", "EURUSD", "USDJPY", "USDCAD", "ETHUSD")


async def backfill_one(ds: DataStore, fb: FeedbackLoop, symbol: str) -> tuple[int, int]:
    """Returns (filled_count, skipped_count) for the symbol."""
    async with ds._session_factory() as session:
        rows = await session.execute(text(
            "SELECT id, bar_timestamp FROM actual_outcomes "
            "WHERE symbol = :s AND actual_tb_label IS NULL"
        ), {"s": symbol})
        pending = rows.fetchall()

    filled = 0
    skipped = 0
    for outcome_id, bar_timestamp in pending:
        label = await fb._compute_tb_label(symbol, bar_timestamp, timeframe="H4")
        if label is None:
            skipped += 1
            continue
        await ds.backfill_tb_label(int(outcome_id), float(label))
        filled += 1
    logger.info("[%s] filled %d, skipped %d (no future bars yet)",
                 symbol, filled, skipped)
    return filled, skipped


async def clear_prediction_errors(ds: DataStore, symbol: str) -> int:
    """Remove all prediction_errors for a symbol so they recompute fresh."""
    async with ds._engine.begin() as conn:
        result = await conn.execute(text(
            "DELETE FROM prediction_errors USING model_predictions mp "
            "WHERE prediction_errors.prediction_id = mp.id AND mp.symbol = :s"
        ), {"s": symbol})
        return int(result.rowcount or 0)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=None,
                        help="Single symbol; otherwise all 4 live symbols.")
    parser.add_argument("--no-clear", action="store_true",
                        help="Skip clearing prediction_errors after backfill.")
    args = parser.parse_args()

    load_dotenv()
    syms = [args.symbol.upper()] if args.symbol else list(LIVE_SYMBOLS)

    ds = DataStore()
    await ds.connect()
    fb = FeedbackLoop(ds, hmm=None)
    try:
        for sym in syms:
            await backfill_one(ds, fb, sym)
            if not args.no_clear:
                cleared = await clear_prediction_errors(ds, sym)
                logger.info("[%s] cleared %d prediction_errors rows", sym, cleared)
            # Recompute against the new TB labels
            n = await fb.compute_prediction_errors(sym)
            logger.info("[%s] recomputed %d prediction_errors", sym, n)
    finally:
        await ds.close()


if __name__ == "__main__":
    asyncio.run(main())
