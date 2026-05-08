"""Post-backfill sanity check.

Runs six invariants against ohlcv_bars:
    1. OHLC integrity: high >= max(open, close) AND low <= min(open, close)
    2. Positivity: open/high/low/close all > 0
    3. No duplicate (symbol, tf, bar_timestamp) — unique constraint is enforced
       by the DB schema, but we re-count as a safety belt
    4. Row counts per (symbol, tf) — spot compare vs expected
    5. Max gap in hours per (symbol, tf) — must match weekend/holiday shape
    6. Sample spot check: recent ETH bars look like real ETH prices
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

from src.data_pipeline.data_store import DataStore  # noqa: E402

WINDOW_START = datetime(2021, 1, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 4, 21, tzinfo=timezone.utc)

SYMBOLS = ["XAUUSD", "EURUSD", "USDJPY", "USDCAD", "ETHUSD"]
TIMEFRAMES = ["D1", "H4", "H1"]


async def run() -> None:
    dsn = os.environ["POSTGRES_DSN"]
    store = DataStore(dsn=dsn)
    await store.connect()
    try:
        print("=" * 70)
        print("INVARIANT 1-3: integrity, positivity, duplicates (SQL)")
        print("=" * 70)
        async with store._session_factory() as session:
            # OHLC integrity violations
            bad_ohlc = await session.execute(text("""
                SELECT symbol, timeframe, COUNT(*) AS n
                FROM ohlcv_bars
                WHERE high < GREATEST(open, close)
                   OR low  > LEAST(open, close)
                   OR open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
                GROUP BY symbol, timeframe
            """))
            rows = bad_ohlc.fetchall()
            if rows:
                print("FAIL: bad OHLC rows found:")
                for r in rows:
                    print(f"  {r.symbol:<8} {r.timeframe:<4} {r.n} rows")
            else:
                print("PASS: all OHLC rows well-formed, all prices positive")

            # Duplicate check (unique constraint should prevent, but verify)
            dup = await session.execute(text("""
                SELECT symbol, timeframe, bar_timestamp, COUNT(*) AS n
                FROM ohlcv_bars
                GROUP BY symbol, timeframe, bar_timestamp
                HAVING COUNT(*) > 1
                LIMIT 10
            """))
            dup_rows = dup.fetchall()
            if dup_rows:
                print(f"FAIL: {len(dup_rows)} duplicate keys found:")
                for r in dup_rows:
                    print(f"  {r.symbol} {r.timeframe} {r.bar_timestamp} ×{r.n}")
            else:
                print("PASS: no duplicate (symbol, tf, bar_timestamp) keys")

        print()
        print("=" * 70)
        print("INVARIANT 4: row counts per (symbol, tf)")
        print("=" * 70)
        print(f"{'symbol':<8} {'tf':<4} {'rows':>7}")
        for sym in SYMBOLS:
            for tf in TIMEFRAMES:
                df = await store.get_ohlcv_range(
                    sym, tf, start=WINDOW_START, end=WINDOW_END
                )
                print(f"{sym:<8} {tf:<4} {len(df):>7}")

        print()
        print("=" * 70)
        print("INVARIANT 5: max-gap-hours per (symbol, tf)")
        print("=" * 70)
        print(f"{'symbol':<8} {'tf':<4} {'max_gap_h':>10} {'verdict'}")
        for sym in SYMBOLS:
            for tf in TIMEFRAMES:
                df = await store.get_ohlcv_range(
                    sym, tf, start=WINDOW_START, end=WINDOW_END
                )
                if df.empty:
                    continue
                idx = pd.to_datetime(df.index, utc=True)
                max_gap_h = idx.to_series().diff().max().total_seconds() / 3600.0
                # Expected max-gap tolerances. Forex weekend closure ~65h,
                # XAU weekend ~65h, ETH never closes (~4h on D1 = none).
                # Padding for holidays.
                if sym == "ETHUSD":
                    limit = 72 if tf == "H1" else 96
                elif sym == "XAUUSD":
                    limit = 96
                else:  # forex
                    limit = 96
                verdict = "ok" if max_gap_h <= limit else f"WARN (> {limit}h)"
                print(f"{sym:<8} {tf:<4} {max_gap_h:>10.1f} {verdict}")

        print()
        print("=" * 70)
        print("INVARIANT 6: ETH spot check — last 3 H4 bars")
        print("=" * 70)
        eth = await store.get_ohlcv_range(
            "ETHUSD", "H4", start=WINDOW_START, end=WINDOW_END, limit=3
        )
        print(eth.to_string())

        print()
        print("=" * 70)
        print("INVARIANT 7: other symbols' D1 row counts match pre-backfill")
        print("=" * 70)
        # From the pre-backfill coverage snapshot: XAU 1367, EUR 1375, JPY 1375, CAD 1375
        expected = {"XAUUSD": 1367, "EURUSD": 1375, "USDJPY": 1375, "USDCAD": 1375}
        for sym, exp in expected.items():
            df = await store.get_ohlcv_range(sym, "D1", start=WINDOW_START, end=WINDOW_END)
            delta = len(df) - exp
            verdict = "ok" if abs(delta) <= 1 else f"DIFF ({delta:+d})"
            print(f"{sym:<8} D1  before={exp} after={len(df)} {verdict}")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(run())
