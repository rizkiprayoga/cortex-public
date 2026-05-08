"""One-shot ETHUSD backfill into ohlcv_bars.

Prereq: bot must be paused (dashboard System tab → pause). This script calls
MT5Connector.connect() which repoints the terminal to MT5_LOGIN; it snapshots
and restores the pre-run login on exit so resuming the bot lands on the
operator's active account.

Pulls D1/H4/H1 from 2021-01-01 → now for ETHUSD and persists via DataStore.
Uses raw mt5.copy_rates_range + MT5DataFeed._rates_to_dicts because
get_historical_async's start_date path has a typing bug (passes strings to
DataStore.get_ohlcv_range which expects datetimes).
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_ethusd")

import MetaTrader5 as mt5  # noqa: E402

from scripts._mt5_safety import _restore_mt5_login, _snapshot_mt5_login  # noqa: E402
from src.broker.mt5_connector import MT5Connector  # noqa: E402
from src.data_pipeline.data_store import DataStore  # noqa: E402
from src.data_pipeline.mt5_feed import MT5DataFeed, TIMEFRAME_MAP  # noqa: E402

SYMBOL = "ETHUSD"
TIMEFRAMES = ["D1", "H4", "H1"]
START = datetime(2021, 1, 1)


async def run() -> None:
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise SystemExit("POSTGRES_DSN missing from environment")

    store = DataStore(dsn=dsn)
    await store.connect()

    mt5_snapshot = _snapshot_mt5_login()
    if mt5_snapshot is not None:
        logger.info(
            "MT5 snapshot: login=%s server=%s source=%s",
            mt5_snapshot["login"], mt5_snapshot["server"], mt5_snapshot["source"],
        )
    atexit.register(_restore_mt5_login, mt5_snapshot)

    connector = MT5Connector()
    connector.connect()
    feed = MT5DataFeed(connector, data_store=store)

    end = datetime.utcnow()
    try:
        for tf in TIMEFRAMES:
            tf_const = TIMEFRAME_MAP[tf]
            logger.info("[%s %s] fetching %s → %s", SYMBOL, tf, START.date(), end.date())
            rates = mt5.copy_rates_range(SYMBOL, tf_const, START, end)
            if rates is None or len(rates) == 0:
                logger.warning("[%s %s] MT5 returned no bars", SYMBOL, tf)
                continue

            bar_dicts = feed._rates_to_dicts(rates, SYMBOL, tf)

            # Dedupe by (symbol, timeframe, bar_timestamp) — MT5 can return
            # duplicate rows on H1 around DST transitions after
            # _broker_ts_to_utc collapses two broker-time bars to the same UTC.
            seen: dict[tuple, dict] = {}
            for b in bar_dicts:
                key = (b["symbol"], b["timeframe"], b["bar_timestamp"])
                seen[key] = b  # last-write-wins (deterministic)
            deduped = list(seen.values())
            if len(deduped) != len(bar_dicts):
                logger.info(
                    "[%s %s] deduped %d → %d rows (MT5 DST dupes)",
                    SYMBOL, tf, len(bar_dicts), len(deduped),
                )

            # asyncpg bind-param limit is 32767. With 9 columns per row the
            # safe upper bound is ~3600 rows/batch — use 2000 for headroom.
            CHUNK = 2000
            inserted_total = 0
            for i in range(0, len(deduped), CHUNK):
                chunk = deduped[i : i + CHUNK]
                inserted_total += await store.bulk_insert_ohlcv(chunk)
            logger.info(
                "[%s %s] fetched=%d inserted=%d (chunked; dupes skipped by unique constraint)",
                SYMBOL, tf, len(rates), inserted_total,
            )
    finally:
        try:
            connector.disconnect()
        except Exception:
            pass
        await store.close()


if __name__ == "__main__":
    asyncio.run(run())
