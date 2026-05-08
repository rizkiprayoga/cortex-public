"""
mt5_feed.py — MetaTrader 5 OHLCV Data Feed (DB-first cache)

Fetches historical and real-time OHLCV (Open, High, Low, Close, Volume)
data from the MetaTrader 5 terminal for XAUUSD and BTCUSD.

DB-first caching strategy
-------------------------
Historical data fetched from MT5 is persisted into the ``ohlcv_bars`` table
via ``DataStore.bulk_insert_ohlcv()``. Subsequent calls to ``get_historical()``
check the DB first and only fetch missing date gaps from MT5. This is the
single source of truth for all downstream consumers (HMM, LSTM, backtests,
feature engineering, feedback loop).

Key MT5 functions used:
    mt5.copy_rates_from()      — Historical bars from a date
    mt5.copy_rates_from_pos()  — Historical bars from position index
    mt5.copy_rates_range()     — Bars within a date range
    mt5.copy_ticks_from()      — Real-time tick data

Timeframes fetched:
    D1  — for HMM regime classification
    H4  — for LSTM signal generation
    M15 — for execution timing
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# the broker runs EET (UTC+2 winter) / EEST (UTC+3 summer DST). MT5 returns
# bar timestamps in broker-server time. Convert to UTC at ingest so all
# downstream code (news blackout, calendar features, LSTM training) sees
# true UTC and aligns with TradingView / macro data. zoneinfo is in the
# stdlib since Python 3.9 and handles DST transitions automatically.
_BROKER_TZ = ZoneInfo("Europe/Helsinki")


def _broker_ts_to_utc(ts_int: int) -> datetime:
    """Convert MT5 broker-local epoch seconds → true-UTC naive datetime."""
    # rate["time"] is int seconds since epoch BUT measured against broker
    # clock, not real UTC. Walk it through the broker TZ then convert.
    broker_naive = datetime.fromtimestamp(int(ts_int), tz=timezone.utc).replace(tzinfo=None)
    broker_aware = broker_naive.replace(tzinfo=_BROKER_TZ)
    utc_aware = broker_aware.astimezone(timezone.utc)
    return utc_aware.replace(tzinfo=None)

from src.broker.mt5_connector import MT5Connector

if TYPE_CHECKING:
    from src.data_pipeline.data_store import DataStore

logger = logging.getLogger(__name__)

# MT5 timeframe constants
TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
}

# Bar width per timeframe — used to advance past the last-stored bar during
# gap backfill so the unique bar is not re-fetched.
TIMEFRAME_DELTA = {
    "M1":  timedelta(minutes=1),
    "M5":  timedelta(minutes=5),
    "M15": timedelta(minutes=15),
    "H1":  timedelta(hours=1),
    "H4":  timedelta(hours=4),
    "D1":  timedelta(days=1),
    "W1":  timedelta(weeks=1),
}


class MT5DataFeed:
    """
    Fetches OHLCV bars from MetaTrader 5 for any symbol and timeframe,
    with a PostgreSQL-backed cache.

    Usage:
        feed = MT5DataFeed(connector, data_store=store)
        df = await feed.get_historical_async("XAUUSD", "D1", bars=500)
        latest = feed.get_latest("XAUUSD", "H4", bars=60)

    The async variants (``_async`` suffix) should be preferred from the
    main trading loop since they reuse the DB cache. Sync variants still
    exist for scripts that don't run inside an event loop (backtests,
    one-off training jobs).
    """

    def __init__(
        self,
        connector: Optional[MT5Connector] = None,
        data_store: Optional["DataStore"] = None,
    ):
        """
        Args:
            connector:  An MT5Connector (need not be already-connected; the
                        per-method MT5 calls check ``connector.is_connected()``).
                        Pass ``None`` for MT5-free workflows that only use
                        ``get_historical_db_only`` — training/backtest scripts
                        should do this to avoid the shared-terminal hijack risk.
            data_store: Optional async DataStore for DB-first caching.
                        If None, bars are fetched directly from MT5 each call.
        """
        self.connector = connector
        self.data_store = data_store

    # -------------------------------------------------------------------------
    # Sync API (backtests, one-off scripts)
    # -------------------------------------------------------------------------

    def get_historical(
        self,
        symbol: str,
        timeframe: str = "D1",
        bars: int = 500,
        start_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars from MT5 (no DB caching).

        Args:
            symbol:     MT5 symbol name (e.g. "XAUUSD")
            timeframe:  Timeframe string ("M15", "H4", "D1", etc.)
            bars:       Number of bars to fetch (if start_date is None)
            start_date: Fetch from this date to now (overrides bars)

        Returns:
            DataFrame indexed by datetime with columns:
            [open, high, low, close, tick_volume]
        """
        if not self.connector.is_connected():
            raise RuntimeError(
                "MT5 terminal not connected. Call connector.connect() first."
            )

        tf_const = TIMEFRAME_MAP.get(timeframe)
        if tf_const is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        if start_date is not None:
            rates = mt5.copy_rates_range(
                symbol, tf_const, start_date, datetime.utcnow()
            )
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars)

        if rates is None or len(rates) == 0:
            logger.warning(f"[{symbol} {timeframe}] MT5 returned no bars")
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "tick_volume"]
            )

        # MT5 ``rate["time"]`` is broker-local epoch (the broker = EET/EEST,
        # UTC+2/+3). Treating it as UTC (the old ``pd.to_datetime(unit='s')``
        # path) labels broker-wall-clock as UTC — bars come back 2-3h off true
        # UTC. This mismatched live (DB cache via ``_rates_to_dicts``, true UTC)
        # against training (sync ``get_historical``, broker-time-as-UTC) so the
        # LSTM was silently learning a 2-3h-shifted alignment to externals.
        # Same fix as ``fetch_chunked`` (Phase 1A backfill, commit 5b9fb2e):
        # walk each broker epoch through ``_broker_ts_to_utc`` so the returned
        # DataFrame index is naive true-UTC, matching the DB convention.
        df = pd.DataFrame(rates)
        df["time"] = [_broker_ts_to_utc(int(t)) for t in df["time"]]
        df.set_index("time", inplace=True)
        df = df[["open", "high", "low", "close", "tick_volume"]]
        logger.debug(
            f"[{symbol} {timeframe}] Fetched {len(df)} bars from MT5"
        )
        return df

    def get_latest(
        self,
        symbol: str,
        timeframe: str = "H4",
        bars: int = 60,
    ) -> pd.DataFrame:
        """
        Fetch the most recent N bars (used in live trading loop).

        Returns the last ``bars`` closed bars for the given symbol/timeframe.
        """
        return self.get_historical(symbol, timeframe, bars=bars)

    def get_current_price(self, symbol: str) -> dict:
        """
        Return current bid/ask prices for a symbol.

        Returns:
            {"bid": float, "ask": float, "spread": float, "time": datetime}
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(
                f"MT5 returned no tick data for {symbol}. "
                f"Ensure the symbol is visible in Market Watch."
            )
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": round(tick.ask - tick.bid, 6),
            "time": _broker_ts_to_utc(int(tick.time)),
        }

    def get_multi_timeframe(
        self, symbol: str, timeframes: list[str], bars: int = 500
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch multiple timeframes at once.

        Returns:
            Dict mapping timeframe string → OHLCV DataFrame.
        """
        result = {}
        for tf in timeframes:
            result[tf] = self.get_historical(symbol, tf, bars=bars)
        return result

    # -------------------------------------------------------------------------
    # DB-only API (training / backtest — no MT5 contact, no terminal hijack)
    # -------------------------------------------------------------------------

    async def get_historical_db_only(
        self,
        symbol: str,
        timeframe: str = "D1",
        bars: int = 500,
        start_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars from DataStore ONLY — never touches MT5.

        Used by training and backtest scripts so they are MT5-free, eliminating
        the shared-terminal hijack risk that polluted prod equity_history on
        2026-04-25 (see feedback_dev_mt5_steals_prod_terminal.md). The Python
        ``MetaTrader5`` package is a single global Windows binding — any
        ``mt5.initialize(...)`` call from a dev script silently switches the
        live prod terminal's logged-in account. The only structural fix is to
        not call MT5 from training scripts at all.

        DataStore is the source of truth for historical bars (true UTC,
        populated by the live writer ``_rates_to_dicts`` + Phase 1A backfill
        ``fetch_chunked``, both of which route through ``_broker_ts_to_utc``).
        Every bar the live bot has ever seen is here.

        Args:
            symbol:     Symbol name (e.g. "XAUUSD").
            timeframe:  TF string ("M15", "H4", "D1", "W1").
            bars:       How many most recent bars to return. Ignored if
                        ``start_date`` is given.
            start_date: Optional naive UTC lower bound. When set, returns
                        ALL bars in DB whose timestamp >= start_date.

        Returns:
            DataFrame indexed by naive-UTC bar_timestamp with columns
            [open, high, low, close, volume]. Empty DataFrame if no DB rows.
            FeatureEngineer.transform() handles the volume/tick_volume rename
            internally so callers do not need to.

        Raises:
            RuntimeError: when self.data_store is None — pure DB read requires
                          a connected store, no MT5 fallback by design.
            ValueError: when timeframe is not in TIMEFRAME_MAP.
        """
        if self.data_store is None:
            raise RuntimeError(
                "get_historical_db_only requires self.data_store to be "
                "connected. Construct MT5DataFeed(connector, data_store=...) "
                "or set feed.data_store before calling. There is no MT5 "
                "fallback by design — see "
                "memory/feedback_dev_mt5_steals_prod_terminal.md."
            )
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        if start_date is not None:
            df = await self.data_store.get_ohlcv_range(
                symbol, timeframe, start=start_date,
            )
        else:
            df = await self.data_store.get_ohlcv_range(
                symbol, timeframe, limit=bars,
            )

        if df.empty:
            logger.warning(
                "[%s %s] get_historical_db_only: 0 rows in DB. Has the live "
                "bot or backfill ever populated this symbol/timeframe?",
                symbol, timeframe,
            )
        else:
            logger.debug(
                "[%s %s] get_historical_db_only: %d bars from DB",
                symbol, timeframe, len(df),
            )
        return df

    # -------------------------------------------------------------------------
    # Async API (DB-first cache — preferred in main trading loop)
    # -------------------------------------------------------------------------

    async def get_historical_async(
        self,
        symbol: str,
        timeframe: str = "D1",
        bars: int = 500,
        start_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars using the DB-first caching strategy.

        Algorithm:
            1. Query DataStore for existing bars in the requested range
            2. Compute date gaps (missing periods)
            3. Fetch only the gaps from MT5
            4. Persist new bars via DataStore.bulk_insert_ohlcv()
            5. Return the full concatenated DataFrame

        Falls back to a plain MT5 fetch if ``self.data_store`` is None.
        """
        if self.data_store is None:
            return self.get_historical(symbol, timeframe, bars, start_date)

        tf_const = TIMEFRAME_MAP.get(timeframe)
        if tf_const is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        end = datetime.utcnow()
        if start_date is not None:
            start = start_date
        else:
            delta = TIMEFRAME_DELTA.get(timeframe, timedelta(days=1))
            start = end - delta * bars

        # 1. Query DB for existing bars
        existing = await self.data_store.get_ohlcv_range(
            symbol, timeframe,
            start.strftime("%Y-%m-%dT%H:%M:%S"),
            end.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        if existing is not None and len(existing) >= bars:
            logger.debug(
                f"[{symbol} {timeframe}] DB cache hit: {len(existing)} bars"
            )
            return existing.tail(bars)

        # 2-3. Find gaps and fetch from MT5
        gaps = self._find_date_gaps(
            existing if existing is not None else pd.DataFrame(),
            start, end, timeframe,
        )

        new_bars_total = 0
        for gap_start, gap_end in gaps:
            rates = mt5.copy_rates_range(symbol, tf_const, gap_start, gap_end)
            if rates is not None and len(rates) > 0:
                bar_dicts = self._rates_to_dicts(rates, symbol, timeframe)
                inserted = await self.data_store.bulk_insert_ohlcv(bar_dicts)
                new_bars_total += inserted
                logger.debug(
                    f"[{symbol} {timeframe}] Filled gap "
                    f"{gap_start} → {gap_end}: {inserted} bars"
                )

        # 5. Re-query DB for the full range
        # get_ohlcv_range expects datetime objects (it converts them to
        # ISO strings internally via _dt_to_iso). Passing strings here
        # crashes with 'str has no attribute tzinfo'.
        result = await self.data_store.get_ohlcv_range(
            symbol, timeframe, start, end,
        )

        if result is None or len(result) == 0:
            logger.warning(
                f"[{symbol} {timeframe}] No data after DB+MT5 fetch"
            )
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "tick_volume"]
            )

        logger.info(
            f"[{symbol} {timeframe}] Returning {len(result)} bars "
            f"({new_bars_total} newly fetched)"
        )
        return result.tail(bars)

    async def get_latest_async(
        self,
        symbol: str,
        timeframe: str = "H4",
        bars: int = 60,
    ) -> pd.DataFrame:
        """
        Fetch most recent N bars with DB caching.

        The latest-closed bars are fetched from MT5 and appended to the
        ``ohlcv_bars`` cache before returning. Old bars already in the DB
        are not re-fetched.
        """
        if self.data_store is None:
            return self.get_latest(symbol, timeframe, bars)

        tf_const = TIMEFRAME_MAP.get(timeframe)
        if tf_const is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Fetch latest bars from MT5
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars)
        if rates is not None and len(rates) > 0:
            bar_dicts = self._rates_to_dicts(rates, symbol, timeframe)
            inserted = await self.data_store.bulk_insert_ohlcv(bar_dicts)
            if inserted > 0:
                logger.debug(
                    f"[{symbol} {timeframe}] Cached {inserted} new bars"
                )

        # Return from DB for consistency
        delta = TIMEFRAME_DELTA.get(timeframe, timedelta(days=1))
        start = datetime.utcnow() - delta * (bars + 10)  # small padding
        end = datetime.utcnow()

        # get_ohlcv_range expects datetime objects (it converts them to
        # ISO strings internally via _dt_to_iso). Passing strings here
        # crashes with 'str has no attribute tzinfo'.
        result = await self.data_store.get_ohlcv_range(
            symbol, timeframe, start, end,
        )

        if result is None or len(result) == 0:
            return self.get_latest(symbol, timeframe, bars)

        # The DB stores the column as 'volume'; the rest of the system
        # (feature engineering, indicators) expects 'tick_volume' to match
        # the MT5-direct path. Rename here so DB-cached and MT5-direct
        # paths produce identical DataFrame schemas.
        if "volume" in result.columns and "tick_volume" not in result.columns:
            result = result.rename(columns={"volume": "tick_volume"})

        return result.tail(bars)

    async def backfill_gaps(self, symbol: str, timeframe: str) -> int:
        """
        Detect and fill OHLCV bars missing from the DB between the most
        recently stored bar and 'now'. Self-healing on startup after
        downtime (blackout, reboot, crash).

        Idempotent — relies on the unique constraint
        (symbol, timeframe, bar_timestamp) in ohlcv_bars so re-running
        it is safe. Returns the number of new bars persisted.

        Behaviour:
            - If the DB has no prior bars for this (symbol, timeframe),
              returns 0 and defers the initial bulk fetch to
              ``load_or_train_async()``.
            - Otherwise fetches from (last_ts + 1 bar) through utcnow()
              via ``mt5.copy_rates_range()`` and bulk-inserts them.
        """
        if self.data_store is None:
            logger.warning(
                f"[{symbol} {timeframe}] backfill_gaps called without data_store — skipping"
            )
            return 0

        last_ts = await self.data_store.get_latest_bar_timestamp(symbol, timeframe)
        if last_ts is None:
            logger.info(
                f"[{symbol} {timeframe}] No prior data in DB — "
                f"initial fetch deferred to load_or_train"
            )
            return 0

        delta = TIMEFRAME_DELTA.get(timeframe)
        if delta is None:
            logger.error(f"[{symbol} {timeframe}] Unknown timeframe for backfill")
            return 0

        # Advance past the last stored bar so we don't refetch it
        start = datetime.fromisoformat(last_ts) + delta
        end = datetime.utcnow()
        if start >= end:
            logger.debug(f"[{symbol} {timeframe}] Already up to date (last={last_ts})")
            return 0

        tf_const = TIMEFRAME_MAP.get(timeframe)
        if tf_const is None:
            logger.error(f"[{symbol} {timeframe}] Unsupported timeframe constant")
            return 0

        rates = mt5.copy_rates_range(symbol, tf_const, start, end)
        if rates is None or len(rates) == 0:
            logger.info(
                f"[{symbol} {timeframe}] MT5 returned no new bars since {last_ts}"
            )
            return 0

        bars = self._rates_to_dicts(rates, symbol, timeframe)
        inserted = await self.data_store.bulk_insert_ohlcv(bars)
        logger.info(
            f"[{symbol} {timeframe}] Backfilled {inserted} bars since {last_ts}"
        )
        return inserted

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _rates_to_dicts(
        self, rates: np.ndarray, symbol: str, timeframe: str
    ) -> list[dict]:
        """
        Convert an MT5 rates numpy structured array into the row dict
        format expected by ``DataStore.bulk_insert_ohlcv()``.

        MT5 structured array fields:
            time, open, high, low, close, tick_volume, spread, real_volume
        ``time`` is a Unix epoch in seconds.
        """
        out: list[dict] = []
        for rate in rates:
            ts = _broker_ts_to_utc(int(rate["time"]))
            out.append({
                "symbol":        symbol,
                "timeframe":     timeframe,
                "bar_timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "open":          float(rate["open"]),
                "high":          float(rate["high"]),
                "low":           float(rate["low"]),
                "close":         float(rate["close"]),
                "volume":        float(rate["tick_volume"]),
            })
        return out

    def _find_date_gaps(
        self,
        existing: pd.DataFrame,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> list[tuple[datetime, datetime]]:
        """
        Given existing bars in ``existing`` and a requested [start, end] range,
        return a list of (gap_start, gap_end) tuples representing contiguous
        periods that are missing from the DB and must be fetched from MT5.

        Algorithm:
            - If existing is empty, the entire [start, end] range is one gap.
            - Otherwise, walk through existing timestamps sorted ascending.
              Gaps exist before the first bar, between non-consecutive bars,
              and after the last bar up to ``end``.
            - Two bars are "consecutive" when their delta equals the
              timeframe's bar width (TIMEFRAME_DELTA). A wider gap means
              bars are missing in between.
        """
        if existing is None or len(existing) == 0:
            return [(start, end)]

        delta = TIMEFRAME_DELTA.get(timeframe, timedelta(days=1))

        # Ensure we have a sorted list of bar timestamps
        if isinstance(existing.index, pd.DatetimeIndex):
            timestamps = sorted(existing.index.to_pydatetime())
        else:
            timestamps = sorted(
                pd.to_datetime(existing.index).to_pydatetime()
            )

        gaps: list[tuple[datetime, datetime]] = []

        # Gap before the first stored bar
        first_ts = timestamps[0]
        if start < first_ts - delta:
            gaps.append((start, first_ts - delta))

        # Gaps between consecutive stored bars
        for i in range(len(timestamps) - 1):
            expected_next = timestamps[i] + delta
            actual_next = timestamps[i + 1]
            if actual_next > expected_next + timedelta(seconds=30):
                gaps.append((expected_next, actual_next - delta))

        # Gap after the last stored bar
        last_ts = timestamps[-1]
        if last_ts + delta < end:
            gaps.append((last_ts + delta, end))

        return gaps
