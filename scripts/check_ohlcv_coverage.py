"""One-off coverage check for ohlcv_bars before switching backtest.py to DB-only.

Reports per (symbol, timeframe):
    - first bar, last bar (UTC)
    - row count
    - expected count over the window (trading-session aware)
    - largest single gap in bars
    - coverage ratio

Focus window: 2021-01-01 → 2026-04-21 (matches the 5yr baseline).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.data_pipeline.data_store import DataStore  # noqa: E402

# All 21 symbols: 5 live + 6 Phase 1A expansion + 10 an earlier sprint Phase B
# expansion. Update here when adding more pairs.
LIVE_SYMBOLS = ["XAUUSD", "EURUSD", "USDJPY", "USDCAD", "ETHUSD"]
NEW_SYMBOLS  = ["GBPUSD", "AUDUSD", "EURGBP", "EURJPY", "GBPJPY", "AUDNZD"]
PHASE_2_SYMBOLS = [
    "USDCHF", "NZDUSD", "EURCHF", "EURAUD",
    "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
    "GBPCHF", "GBPAUD",
]
SYMBOLS = LIVE_SYMBOLS + NEW_SYMBOLS + PHASE_2_SYMBOLS
TIMEFRAMES = ["W1", "D1", "H4", "H1", "M15"]

WINDOW_START = datetime(2021, 1, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 4, 25, tzinfo=timezone.utc)

# Hours per bar for gap math.
TF_HOURS = {"W1": 168, "D1": 24, "H4": 4, "H1": 1, "M15": 0.25}

# 5-day weekday-only symbols (Mon-Fri trading sessions). Everything except
# ETHUSD which trades 7 days at the broker. The 5/7 ratio scales calendar-
# expected bar count to trading-day-expected count; doesn't apply to W1
# since one weekly bar represents one calendar week regardless of session.
FIVE_DAY_SYMBOLS = frozenset({
    "EURUSD", "USDJPY", "USDCAD", "XAUUSD",
    "GBPUSD", "AUDUSD", "EURGBP", "EURJPY", "GBPJPY", "AUDNZD",
    "USDCHF", "NZDUSD", "EURCHF", "EURAUD",
    "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
    "GBPCHF", "GBPAUD",
})


async def main() -> None:
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise SystemExit("POSTGRES_DSN missing from environment")

    store = DataStore(dsn=dsn)
    await store.connect()
    try:
        print(
            f"{'symbol':<7} {'tf':<4} "
            f"{'first':<20} {'last':<20} "
            f"{'rows':>7} {'expected':>9} {'cov%':>6} {'max_gap_bars':>13}"
        )
        print("-" * 100)

        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                df = await store.get_ohlcv_range(
                    symbol, tf, start=WINDOW_START, end=WINDOW_END
                )
                if df.empty:
                    print(
                        f"{symbol:<7} {tf:<4} "
                        f"{'(no data)':<20} {'(no data)':<20} "
                        f"{0:>7} {'-':>9} {'-':>6} {'-':>13}"
                    )
                    continue

                idx = pd.to_datetime(df.index, utc=True)
                first = idx.min()
                last = idx.max()
                rows = len(df)

                # 5-day weekday vs 7-day session heuristic for expected count.
                # Apply only to intraday / daily TFs — weekly bar produces one
                # bar per calendar week regardless of session.
                span_hours = (last - first).total_seconds() / 3600.0
                hours_per_bar = TF_HOURS[tf]
                if tf != "W1" and symbol in FIVE_DAY_SYMBOLS:
                    session_ratio = 5.0 / 7.0
                else:
                    session_ratio = 1.0
                expected = int(span_hours / hours_per_bar * session_ratio)
                coverage = (rows / expected * 100.0) if expected > 0 else 0.0

                # Largest single gap (in bar-widths).
                deltas = idx.to_series().diff().dropna()
                max_gap_hours = deltas.max().total_seconds() / 3600.0 if not deltas.empty else 0.0
                max_gap_bars = max_gap_hours / hours_per_bar

                print(
                    f"{symbol:<7} {tf:<4} "
                    f"{first.strftime('%Y-%m-%d %H:%M'):<20} "
                    f"{last.strftime('%Y-%m-%d %H:%M'):<20} "
                    f"{rows:>7} {expected:>9} {coverage:>5.1f}% {max_gap_bars:>13.1f}"
                )
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
