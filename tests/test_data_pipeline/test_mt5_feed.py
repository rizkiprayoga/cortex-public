"""Tests for MT5DataFeed (mocked MT5 terminal)."""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.data_pipeline.mt5_feed import MT5DataFeed


@pytest.fixture
def mock_connector():
    connector = MagicMock()
    connector.is_connected.return_value = True
    return connector


@pytest.fixture
def mock_rates():
    """Synthetic MT5 rates array as returned by copy_rates_from_pos."""
    n = 100
    return np.array(
        [(1700000000 + i * 3600, 1900.0 + i * 0.1, 1905.0, 1895.0, 1902.0, 1000)
         for i in range(n)],
        dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"),
               ("low", "f8"), ("close", "f8"), ("tick_volume", "i8")]
    )


class TestMT5DataFeed:

    def test_get_historical_returns_dataframe(self, mock_connector, mock_rates):
        """get_historical() should return a DataFrame with OHLCV columns."""
        with patch("MetaTrader5.copy_rates_from_pos", return_value=mock_rates):
            feed = MT5DataFeed(mock_connector)
            df = feed.get_historical("XAUUSD", "H4", bars=100)
            assert isinstance(df, pd.DataFrame)
            assert set(["open", "high", "low", "close", "tick_volume"]).issubset(df.columns)
            assert len(df) == 100

    def test_get_latest_returns_n_bars(self, mock_connector, mock_rates):
        """get_latest() should return exactly the requested number of bars."""
        with patch("MetaTrader5.copy_rates_from_pos", return_value=mock_rates[:60]):
            feed = MT5DataFeed(mock_connector)
            df = feed.get_latest("XAUUSD", "H4", bars=60)
            assert len(df) == 60

    def test_raises_on_disconnected(self, mock_connector):
        """Should raise RuntimeError if MT5 is not connected."""
        mock_connector.is_connected.return_value = False
        feed = MT5DataFeed(mock_connector)
        with pytest.raises(RuntimeError):
            feed.get_latest("XAUUSD")


class TestGetHistoricalAppliesBrokerTsConversion:
    """Phase 2A correctness fix: sync get_historical() must walk MT5 broker
    epoch through _broker_ts_to_utc, identical to _rates_to_dicts (live writer)
    and fetch_chunked (Phase 1A backfill writer). Without this, training reads
    bars 2-3h off true UTC while live reads (DB cache) are true UTC — silent
    train/serve skew that lookahead-leaks externals during retraining.
    """

    def test_index_matches_rates_to_dicts_per_bar(self, mock_connector, mock_rates):
        """For each broker epoch, get_historical()'s index timestamp must match
        _rates_to_dicts() exactly (the live-writer reference)."""
        from src.data_pipeline.mt5_feed import _broker_ts_to_utc
        with patch("MetaTrader5.copy_rates_from_pos", return_value=mock_rates):
            feed = MT5DataFeed(mock_connector)
            df = feed.get_historical("XAUUSD", "H4", bars=100)

        for i, ts in enumerate(df.index):
            broker_epoch = int(mock_rates[i]["time"])
            expected = _broker_ts_to_utc(broker_epoch)
            assert ts.to_pydatetime() == expected, (
                f"bar {i}: get_historical={ts}, expected={expected} — "
                "broker-ts conversion mismatch"
            )

    def test_index_is_naive_utc(self, mock_connector, mock_rates):
        """Index must be tz-naive (DB convention). Mixing tz-aware + naive
        is what caused the 2026-04-24 production cascade."""
        with patch("MetaTrader5.copy_rates_from_pos", return_value=mock_rates):
            feed = MT5DataFeed(mock_connector)
            df = feed.get_historical("XAUUSD", "H4", bars=100)
        assert df.index.tz is None, (
            "DataFrame index must be tz-naive UTC; tz-aware would mix with "
            "feature_store reads (naive) and break ffill alignment."
        )

    def test_dst_transition_subtracts_correct_offset(self, mock_connector):
        """A broker-epoch in EET (winter, UTC+2) should produce true-UTC -2h.
        A broker-epoch in EEST (summer DST, UTC+3) should produce true-UTC -3h.
        """
        from src.data_pipeline.mt5_feed import _broker_ts_to_utc
        # 2024-01-15 12:00:00 broker-wall-clock (winter EET, UTC+2)
        # broker thinks this is 12:00; the epoch they emit is the one for
        # which datetime.fromtimestamp(epoch, tz=UTC) = 12:00:00 UTC.
        # That means the broker epoch = epoch_of("2024-01-15 12:00:00") naive.
        from datetime import datetime, timezone, timedelta
        winter_broker_epoch = int(
            datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        )
        winter_true_utc = _broker_ts_to_utc(winter_broker_epoch)
        # EET = UTC+2, so true UTC is broker-wall - 2h = 10:00:00
        assert winter_true_utc == datetime(2024, 1, 15, 10, 0, 0)

        # 2024-07-15 12:00:00 broker-wall-clock (summer EEST, UTC+3)
        summer_broker_epoch = int(
            datetime(2024, 7, 15, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        )
        summer_true_utc = _broker_ts_to_utc(summer_broker_epoch)
        # EEST = UTC+3, so true UTC is broker-wall - 3h = 09:00:00
        assert summer_true_utc == datetime(2024, 7, 15, 9, 0, 0)


# ============================================================================
# get_historical_db_only — DB-only path for training/backtest scripts.
# Eliminates the shared MT5 terminal hijack risk by never calling mt5 APIs.
# Tests use a mock DataStore — no real DB connection required.
# ============================================================================

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock


class TestGetHistoricalDbOnly:
    """Pure-DB OHLCV read path — no mt5.initialize() calls under any branch."""

    @staticmethod
    def _store_returning(df: pd.DataFrame) -> AsyncMock:
        """Mock DataStore whose get_ohlcv_range always returns ``df``."""
        store = AsyncMock()
        captured: dict = {}

        async def _read(symbol, timeframe, start=None, end=None, limit=None):
            captured["symbol"] = symbol
            captured["timeframe"] = timeframe
            captured["start"] = start
            captured["end"] = end
            captured["limit"] = limit
            return df.copy()

        store.get_ohlcv_range = _read
        store.captured = captured
        return store

    @staticmethod
    def _build_db_df(n_bars: int = 100, freq: str = "4h") -> pd.DataFrame:
        """Build a DataFrame matching DataStore.get_ohlcv_range output shape."""
        idx = pd.date_range(
            end=datetime(2024, 6, 1), periods=n_bars, freq=freq,
        )
        return pd.DataFrame({
            "open":   [1900.0 + i * 0.1 for i in range(n_bars)],
            "high":   [1905.0 + i * 0.1 for i in range(n_bars)],
            "low":    [1895.0 + i * 0.1 for i in range(n_bars)],
            "close":  [1902.0 + i * 0.1 for i in range(n_bars)],
            "volume": [1000.0] * n_bars,
        }, index=idx)

    def test_raises_when_data_store_not_connected(self, mock_connector):
        """No MT5 fallback by design — must raise loud rather than silently
        repoint the terminal."""
        feed = MT5DataFeed(mock_connector)  # no data_store
        with pytest.raises(RuntimeError, match="requires self.data_store"):
            asyncio.run(feed.get_historical_db_only("XAUUSD", "H4", bars=100))

    def test_does_not_call_mt5_initialize(self, mock_connector):
        """Critical safety invariant: this path must never call mt5.* APIs.
        Patches MetaTrader5.initialize to crash if invoked — proves the DB-only
        path doesn't go anywhere near the shared terminal."""
        feed = MT5DataFeed(mock_connector)
        feed.data_store = self._store_returning(self._build_db_df())

        with patch(
            "MetaTrader5.initialize",
            side_effect=AssertionError("get_historical_db_only must not call mt5.initialize"),
        ):
            with patch(
                "MetaTrader5.copy_rates_from_pos",
                side_effect=AssertionError("get_historical_db_only must not call mt5.copy_rates_*"),
            ):
                with patch(
                    "MetaTrader5.copy_rates_range",
                    side_effect=AssertionError("get_historical_db_only must not call mt5.copy_rates_*"),
                ):
                    df = asyncio.run(
                        feed.get_historical_db_only("XAUUSD", "H4", bars=100),
                    )
        assert len(df) == 100

    def test_returns_naive_utc_index(self, mock_connector):
        """DB convention is naive UTC — no tz-aware leakage onto callers."""
        feed = MT5DataFeed(mock_connector)
        feed.data_store = self._store_returning(self._build_db_df())
        df = asyncio.run(feed.get_historical_db_only("XAUUSD", "H4", bars=50))
        assert df.index.tz is None

    def test_passes_limit_to_data_store(self, mock_connector):
        """When start_date is None, bars maps to DataStore limit param."""
        feed = MT5DataFeed(mock_connector)
        store = self._store_returning(self._build_db_df())
        feed.data_store = store
        asyncio.run(feed.get_historical_db_only("XAUUSD", "H4", bars=42))
        assert store.captured["limit"] == 42
        assert store.captured["start"] is None

    def test_passes_start_date_when_provided(self, mock_connector):
        """When start_date is given, query becomes range-bounded, not limited."""
        feed = MT5DataFeed(mock_connector)
        store = self._store_returning(self._build_db_df())
        feed.data_store = store
        cutoff = datetime(2024, 1, 1)
        asyncio.run(feed.get_historical_db_only(
            "XAUUSD", "D1", bars=99999, start_date=cutoff,
        ))
        assert store.captured["start"] == cutoff
        assert store.captured["limit"] is None

    def test_empty_db_returns_empty_dataframe(self, mock_connector):
        """No bars in DB → empty DataFrame, not crash."""
        feed = MT5DataFeed(mock_connector)
        feed.data_store = self._store_returning(pd.DataFrame())
        df = asyncio.run(feed.get_historical_db_only("UNKNOWN", "H4", bars=100))
        assert df.empty

    def test_unsupported_timeframe_raises(self, mock_connector):
        feed = MT5DataFeed(mock_connector)
        feed.data_store = self._store_returning(pd.DataFrame())
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            asyncio.run(feed.get_historical_db_only("XAUUSD", "BOGUS", bars=100))

    def test_disconnected_mt5_does_not_block_db_read(self, mock_connector):
        """The whole point: MT5 connectivity is irrelevant for this path."""
        mock_connector.is_connected.return_value = False
        feed = MT5DataFeed(mock_connector)
        feed.data_store = self._store_returning(self._build_db_df(n_bars=20))
        df = asyncio.run(feed.get_historical_db_only("XAUUSD", "H4", bars=20))
        assert len(df) == 20
