"""
backfill_ohlcv.py — One-time import of MT5 historical OHLCV into PostgreSQL.

Motivation
----------
MT5 brokers can (and do) rotate historical data. the broker, for example,
keeps ~15 years for majors but shorter windows for less-liquid symbols.
Having our own persistent copy in PostgreSQL means:
  - Retrains stay reproducible if the broker purges old bars
  - Broker switches don't erase history
  - DB reads are ~10x faster than MT5 IPC during retrain cycles
  - Backtests become deterministic across runs

What this script does
---------------------
For each (symbol, timeframe) combination we care about, it:
  1. Fetches all available bars from MT5 (sync API)
  2. Converts to DataStore's bulk-insert format
  3. Upserts into ``ohlcv_bars`` (ON CONFLICT DO NOTHING on unique key)
  4. Reports bars imported per symbol/timeframe

Idempotent: running twice is safe (duplicates ignored by constraint).

Usage
-----
    python scripts/backfill_ohlcv.py
    python scripts/backfill_ohlcv.py --symbols XAUUSD --timeframes H4
    python scripts/backfill_ohlcv.py --max-bars 50000
"""
import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import MetaTrader5 as mt5
import pandas as pd

from src.broker.mt5_connector import MT5Connector
from src.data_pipeline.data_store import DataStore
from src.data_pipeline.mt5_feed import MT5DataFeed, _broker_ts_to_utc

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["XAUUSD", "USDJPY", "EURUSD", "USDCAD"]
DEFAULT_TIMEFRAMES = ["M15", "H1", "H4", "D1", "W1"]

# Allowlists prevent typos from silently creating bogus rows in the DB
# (audit HIGH-2). These match the symbol/TF sets the rest of the system
# knows about.
VALID_SYMBOLS = {
    # Live production set
    "XAUUSD", "USDJPY", "EURUSD", "USDCAD", "ETHUSD",
    # Forex expansion Phase 1 (2026-04-24)
    "GBPUSD", "AUDUSD", "EURGBP", "EURJPY", "GBPJPY", "AUDNZD",
    # an earlier sprint Phase B expansion (2026-04-29)
    "USDCHF", "NZDUSD", "EURCHF", "EURAUD",
    "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
    "GBPCHF", "GBPAUD",
    # Legacy
    "BTCUSD",
}
VALID_TIMEFRAMES = {"M1", "M5", "M15", "H1", "H4", "D1", "W1"}

# PostgreSQL caps prepared-statement parameters at 32767. With 9 columns
# per bar, the safe chunk size is floor(32767/9) = 3640. Round down.
CHUNK_SIZE = 3000

# MT5's copy_rates_range has a hard span cap per timeframe — requests wider
# than this return empty with "Terminal: Invalid params". Chunking by these
# windows lets us walk backwards through full broker history. Values probed
# empirically against the broker-Demo 2026-04-24.
MT5_CHUNK_DAYS = {
    "M1":  90,
    "M5":  180,
    "M15": 540,    # ~1.5yr (probed cap ≈ 2yr)
    "H1":  2920,   # ~8yr (probed cap ≈ 11yr from 2015)
    "H4":  11000,  # ~30yr — no practical cap observed
    "D1":  36500,  # no cap
    "W1":  36500,  # no cap
}

MT5_TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
}


def fetch_chunked(symbol: str, timeframe: str, start_limit: datetime):
    """
    Walk backwards from now in chunks sized by ``MT5_CHUNK_DAYS`` until the
    broker returns empty (= no more history) or we pass ``start_limit``.

    Works around two MT5 API limits:
      - ``copy_rates_range`` rejects wide spans with 'Invalid params'
      - ``copy_rates_from_pos`` silently caps at 99,999 bars

    Each bar's broker-local epoch is converted to true-UTC naive datetime
    via ``_broker_ts_to_utc`` — the SAME helper the live writer uses
    (``mt5_feed._rates_to_dicts``). Bit-identical output is the invariant
    that keeps this in lockstep with the DB's naive-UTC ``bar_timestamp``
    convention. Violating it was the root cause of the 2026-04-24 backfill
    incident — pandas ``pd.to_datetime(unit='s')`` treated the broker-local
    epoch as if it were UTC epoch and shifted every row by the broker
    TZ offset (2-3h depending on EET/EEST).

    Returns a DataFrame with deduplicated bars indexed by naive-UTC
    datetime, or None if no data was returned at all.
    """
    tf_mt5 = MT5_TF_MAP[timeframe]
    chunk_td = timedelta(days=MT5_CHUNK_DAYS[timeframe])
    end = datetime.now(timezone.utc)
    rows: list[dict] = []
    empty_streak = 0
    while end > start_limit:
        start = max(end - chunk_td, start_limit)
        r = mt5.copy_rates_range(symbol, tf_mt5, start, end)
        n = 0 if r is None else len(r)
        if n == 0:
            # Two consecutive empties = broker genuinely has no more history.
            # One empty could be a weekend/holiday gap on intraday TFs.
            empty_streak += 1
            if empty_streak >= 2:
                break
        else:
            empty_streak = 0
            for bar in r:
                rows.append({
                    "time":        _broker_ts_to_utc(int(bar["time"])),
                    "open":        float(bar["open"]),
                    "high":        float(bar["high"]),
                    "low":         float(bar["low"]),
                    "close":       float(bar["close"]),
                    "tick_volume": float(bar["tick_volume"]),
                })
        end = start
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").set_index("time")
    return df


def df_to_bars(symbol: str, timeframe: str, df) -> list[dict]:
    """
    Convert OHLCV DataFrame to bulk_insert_ohlcv dict format.

    bar_timestamp is emitted as NAIVE ISO8601 ("YYYY-MM-DDTHH:MM:SS")
    at second precision — matches the format that ``mt5_feed._rates_to_dicts``,
    ``_dt_to_iso``, and the rest of the pipeline use. Mixing naive and
    tz-aware timestamps in this column breaks ``pd.to_datetime`` on
    read (inferred format locks to the first row).

    Post-broker-ts-fix: ``df.index`` is already naive-UTC datetime
    (converted in ``fetch_chunked``), so the tz strip below is a
    defensive no-op — kept to keep this function robust if another
    caller passes a tz-aware index.
    """
    bars = []
    for ts, row in df.iterrows():
        if hasattr(ts, "to_pydatetime"):
            dt = ts.to_pydatetime()
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            bar_ts = dt.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            bar_ts = str(ts)
        bars.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "bar_timestamp": bar_ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("tick_volume", row.get("volume", 0.0))),
        })
    return bars


async def backfill_one(store: DataStore, feed: MT5DataFeed,
                        symbol: str, timeframe: str,
                        max_bars: int) -> tuple[int, int]:
    """Backfill one (symbol, tf). Returns (fetched, inserted)."""
    logger.info("  [%s %s] Fetching from MT5...", symbol, timeframe)

    # Chunked fetch walks backwards from now, bounded by MT5's per-timeframe
    # range caps. 1990-01-01 is the earliest plausible broker history — MT5
    # simply stops returning data before the broker's actual start.
    start_limit = datetime(1990, 1, 1, tzinfo=timezone.utc)
    df = fetch_chunked(symbol, timeframe, start_limit)
    if df is None or df.empty:
        logger.warning("  [%s %s] No data available", symbol, timeframe)
        return 0, 0

    fetched = len(df)
    bars = df_to_bars(symbol, timeframe, df)

    logger.info("  [%s %s] Upserting %d bars...", symbol, timeframe, fetched)
    # Chunk small enough to fit under PostgreSQL's 32767-parameter cap.
    inserted = 0
    for i in range(0, len(bars), CHUNK_SIZE):
        chunk = bars[i:i + CHUNK_SIZE]
        inserted += await store.bulk_insert_ohlcv(chunk)

    logger.info("  [%s %s] DONE: fetched=%d inserted=%d (skipped duplicates=%d)",
                 symbol, timeframe, fetched, inserted, fetched - inserted)
    return fetched, inserted


async def main_async(args):
    # Attach-only: no mt5.login() issued. Safe to run while the live bot
    # is attached to another account — the terminal's current login is
    # preserved. Market-data reads (copy_rates_range) don't need a
    # specific account.
    connector = MT5Connector()
    if not connector.connect_attach_only():
        logger.error("MT5 attach failed")
        return 1

    feed = MT5DataFeed(connector)
    store = DataStore()
    await store.connect()

    grand_fetched = 0
    grand_inserted = 0

    for symbol in args.symbols:
        logger.info("=== %s ===", symbol)
        for tf in args.timeframes:
            try:
                fetched, inserted = await backfill_one(
                    store, feed, symbol, tf, args.max_bars,
                )
                grand_fetched += fetched
                grand_inserted += inserted
            except Exception as exc:
                logger.error("  [%s %s] FAILED: %s", symbol, tf, exc)

    connector.disconnect()
    await store.close()

    logger.info("=" * 60)
    logger.info("Backfill complete: %d bars fetched, %d new rows inserted",
                grand_fetched, grand_inserted)
    logger.info("=" * 60)
    return 0


def main():
    # Belt-and-braces: refuse to run if prod bot is live, since this script
    # genuinely needs MT5 (writes to ohlcv_bars from copy_rates_*) and would
    # repoint the shared terminal to dev's MT5_LOGIN. See
    # memory/feedback_dev_mt5_steals_prod_terminal.md.
    from scripts._assert_prod_idle import assert_prod_idle
    assert_prod_idle()

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES)
    # Validate against allowlist — prevents typos from silently inserting
    # malformed rows (audit HIGH-2).
    args_tmp, _ = parser.parse_known_args()
    bad_syms = set(args_tmp.symbols) - VALID_SYMBOLS
    bad_tfs = set(args_tmp.timeframes) - VALID_TIMEFRAMES
    if bad_syms:
        parser.error(f"Unknown symbols: {sorted(bad_syms)}. "
                     f"Allowed: {sorted(VALID_SYMBOLS)}")
    if bad_tfs:
        parser.error(f"Unknown timeframes: {sorted(bad_tfs)}. "
                     f"Allowed: {sorted(VALID_TIMEFRAMES)}")
    parser.add_argument("--max-bars", type=int, default=200000,
                        help="Max bars per (symbol, tf) to request from MT5. "
                             "Use 200000 (~55 years H4, ~13 years H1) to get all.")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
