"""
Phase 1D — feature_store tests.

Two layers:

1. Pure-unit tests for the partition planner. No DB needed; runs in CI
   alongside every other unit test.

2. Opt-in integration tests against a real Postgres cluster. Skipped
   unless ``CORTEX_POSTGRES_TEST_DSN`` is set (e.g. on a developer
   machine that has Postgres running). Spins up a transient
   ``trading_bot_test_fs`` database, runs the full migration, exercises
   upsert/read/watermark, then drops the database. Never touches
   ``trading_bot`` or ``trading_bot_dev``.

Async tests follow the repo convention (asyncio.run inside a sync test
body — see test_feedback_loop.py) so we don't add a pytest-asyncio
dependency.

To run integration tests locally (PowerShell):

    $env:CORTEX_POSTGRES_TEST_DSN = "postgresql+asyncpg://USER:PW@localhost:5432/postgres"
    pytest tests/test_data_pipeline/test_feature_store.py -v
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

import pytest

from src.data_pipeline.db_migrations import (
    FEATURE_STORE_PARTITION_END,
    FEATURE_STORE_PARTITION_START,
    _iter_partition_months,
    _partition_ddl,
)


# ---------------------------------------------------------------------------
# Layer 1 — Partition planner unit tests
# ---------------------------------------------------------------------------


class TestPartitionPlanner:
    """No DB. Validates month iteration + DDL generation."""

    def test_iter_inclusive_start_exclusive_end(self):
        months = list(_iter_partition_months((2024, 1), (2024, 4)))
        assert months == [(2024, 1), (2024, 2), (2024, 3)]

    def test_iter_handles_year_rollover(self):
        months = list(_iter_partition_months((2025, 11), (2026, 3)))
        assert months == [(2025, 11), (2025, 12), (2026, 1), (2026, 2)]

    def test_iter_returns_empty_when_start_equals_end(self):
        assert list(_iter_partition_months((2024, 6), (2024, 6))) == []

    def test_default_window_covers_36_years(self):
        count = len(list(_iter_partition_months(
            FEATURE_STORE_PARTITION_START, FEATURE_STORE_PARTITION_END,
        )))
        assert count == 432  # 36 years × 12 months

    def test_partition_ddl_within_year(self):
        sql = _partition_ddl(2026, 4)
        assert "feature_store_2026_04" in sql
        assert "FROM ('2026-04-01') TO ('2026-05-01')" in sql
        assert "IF NOT EXISTS" in sql

    def test_partition_ddl_december_rollover(self):
        sql = _partition_ddl(2025, 12)
        assert "feature_store_2025_12" in sql
        assert "FROM ('2025-12-01') TO ('2026-01-01')" in sql

    def test_partition_ddl_january_pad(self):
        sql = _partition_ddl(2024, 1)
        assert "feature_store_2024_01" in sql
        assert "FROM ('2024-01-01') TO ('2024-02-01')" in sql


# ---------------------------------------------------------------------------
# Layer 2 — Live-Postgres integration (opt-in)
# ---------------------------------------------------------------------------

_PG_TEST_DSN = os.environ.get("CORTEX_POSTGRES_TEST_DSN")
_TEST_DBNAME = "trading_bot_test_fs"

requires_postgres = pytest.mark.skipif(
    _PG_TEST_DSN is None,
    reason="set CORTEX_POSTGRES_TEST_DSN to run feature_store integration tests",
)


def _swap_dbname(dsn: str, new_dbname: str) -> str:
    """Replace the database name in a DSN with ``new_dbname``."""
    parts = urlsplit(dsn)
    return urlunsplit((parts.scheme, parts.netloc, "/" + new_dbname, parts.query, parts.fragment))


async def _setup_test_db():
    """Drop+recreate the test DB, run the migration, return an engine."""
    from sqlalchemy import text

    from src.data_pipeline.data_store import build_engine
    from src.data_pipeline.db_migrations import (
        count_feature_store_partitions,
        create_all_tables,
    )

    admin_dsn = _swap_dbname(_PG_TEST_DSN, "postgres")
    test_dsn = _swap_dbname(_PG_TEST_DSN, _TEST_DBNAME)

    admin = build_engine(admin_dsn, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f"DROP DATABASE IF EXISTS {_TEST_DBNAME}"))
        await conn.execute(text(f"CREATE DATABASE {_TEST_DBNAME}"))
    await admin.dispose()

    engine = build_engine(test_dsn)
    await create_all_tables(engine)
    n = await count_feature_store_partitions(engine)
    assert n == 432, f"expected 432 partitions, got {n}"
    return engine


async def _teardown_test_db(engine):
    from sqlalchemy import text
    from src.data_pipeline.data_store import build_engine

    await engine.dispose()
    admin = build_engine(_swap_dbname(_PG_TEST_DSN, "postgres"), isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f"DROP DATABASE IF EXISTS {_TEST_DBNAME}"))
    await admin.dispose()


def _make_store(engine):
    """Create a DataStore wired to a pre-built engine (bypasses __init__).

    We can't call ``DataStore(dsn=...)`` here because that path builds a
    second engine and reads ``POSTGRES_DSN`` from the environment when the
    arg is omitted. ``__new__`` skips ``__init__`` entirely so we can graft
    the test engine on directly.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.data_pipeline.data_store import DataStore

    store = DataStore.__new__(DataStore)
    store._engine = engine
    store._session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return store


@requires_postgres
class TestFeatureStoreIntegration:
    """
    Exercises the real partitioned table — needs Postgres.

    Each test owns the test DB end-to-end (setup + teardown). Module-scoped
    setup would be cheaper, but the repo doesn't use pytest-asyncio fixtures,
    so per-test is simpler than juggling a long-lived event loop.
    """

    def test_partitions_present(self):
        async def _run():
            engine = await _setup_test_db()
            try:
                from src.data_pipeline.db_migrations import count_feature_store_partitions
                assert await count_feature_store_partitions(engine) == 432
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_upsert_then_read_roundtrip(self):
        async def _run():
            engine = await _setup_test_db()
            try:
                store = _make_store(engine)
                ts = datetime(2026, 4, 15, 12, 0, 0)
                await store.upsert_feature_store(
                    symbol="GBPUSD", timestamp=ts, feature_group="fred_macro",
                    values={"uk_10y_yield": 4.32, "us_10y_yield": 4.51},
                )
                df = await store.read_feature_store("GBPUSD", "fred_macro")
                assert len(df) == 1
                assert df.index[0].to_pydatetime() == ts
                assert df.iloc[0]["uk_10y_yield"] == pytest.approx(4.32)
                assert df.iloc[0]["us_10y_yield"] == pytest.approx(4.51)
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_upsert_overwrites_on_pk_conflict(self):
        async def _run():
            engine = await _setup_test_db()
            try:
                store = _make_store(engine)
                ts = datetime(2026, 5, 1, 0, 0, 0)
                await store.upsert_feature_store(
                    symbol="EURUSD", timestamp=ts, feature_group="cot_tff",
                    values={"net_pos": 100.0},
                )
                await store.upsert_feature_store(
                    symbol="EURUSD", timestamp=ts, feature_group="cot_tff",
                    values={"net_pos": 200.0},
                )
                df = await store.read_feature_store("EURUSD", "cot_tff")
                assert len(df) == 1
                assert df.iloc[0]["net_pos"] == pytest.approx(200.0)
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_partition_routing_across_months(self):
        """
        Insert rows for three different months and verify each lands in
        the correct child partition. Partition routing is opaque to SQL
        callers (Postgres handles it), but a SELECT against the child
        table directly proves the row is physically there.
        """
        async def _run():
            engine = await _setup_test_db()
            try:
                from sqlalchemy import text
                store = _make_store(engine)
                triples = [
                    (datetime(2024, 6, 15, 0, 0, 0), "feature_store_2024_06"),
                    (datetime(2025, 1,  1, 0, 0, 0), "feature_store_2025_01"),
                    (datetime(2025, 12, 31, 23, 0, 0), "feature_store_2025_12"),
                ]
                for ts, _child in triples:
                    await store.upsert_feature_store(
                        symbol="USDJPY", timestamp=ts,
                        feature_group="ecb_yield_curve",
                        values={"y10": 1.23},
                    )
                async with engine.connect() as conn:
                    for ts, child in triples:
                        row = await conn.execute(
                            text(f"SELECT count(*) FROM {child} WHERE timestamp = :ts"),
                            {"ts": ts},
                        )
                        assert row.scalar() == 1, f"row not in {child}"
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_watermark_returns_latest(self):
        async def _run():
            engine = await _setup_test_db()
            try:
                store = _make_store(engine)
                for day in (10, 20, 5):
                    await store.upsert_feature_store(
                        symbol="AUDUSD",
                        timestamp=datetime(2026, 3, day),
                        feature_group="stooq_yields",
                        values={"au_10y": 4.0 + day / 100.0},
                    )
                wm = await store.get_feature_store_watermark("AUDUSD", "stooq_yields")
                assert wm == datetime(2026, 3, 20)
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_bulk_upsert_skips_existing(self):
        async def _run():
            engine = await _setup_test_db()
            try:
                store = _make_store(engine)
                ts = datetime(2025, 7, 1)
                await store.upsert_feature_store(
                    symbol="NZDUSD", timestamp=ts,
                    feature_group="yfinance_cross_asset",
                    values={"vix": 15.0},
                )
                inserted = await store.upsert_feature_store_bulk([
                    {
                        "symbol": "NZDUSD", "timestamp": ts,
                        "feature_group": "yfinance_cross_asset",
                        "values": {"vix": 99.0},
                    },
                    {
                        "symbol": "NZDUSD", "timestamp": datetime(2025, 7, 2),
                        "feature_group": "yfinance_cross_asset",
                        "values": {"vix": 16.0},
                    },
                ])
                assert inserted == 1
                df = await store.read_feature_store(
                    "NZDUSD", "yfinance_cross_asset",
                )
                assert df.loc[df.index == ts, "vix"].iloc[0] == pytest.approx(15.0)
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_bulk_upsert_overwrite_mode_replaces_values(self):
        """
        Regression test for the TTL-cron path. ``mode="overwrite"`` issues
        ``ON CONFLICT DO UPDATE`` (not DO NOTHING). When a row's PK already
        exists, the new ``values`` blob overwrites the old one. The
        ``stmt.excluded["values"]`` bracket-access fix is exercised here —
        regression for the dict-method-collision bug found during the
        first TTL run.
        """
        async def _run():
            engine = await _setup_test_db()
            try:
                store = _make_store(engine)
                ts = datetime(2025, 8, 1)
                await store.upsert_feature_store(
                    symbol="GBPUSD", timestamp=ts, feature_group="cot_tff",
                    values={"net_pos": 100.0, "open_interest": 1000.0},
                )
                touched = await store.upsert_feature_store_bulk(
                    [
                        {
                            "symbol": "GBPUSD", "timestamp": ts,
                            "feature_group": "cot_tff",
                            "values": {"net_pos": 200.0, "open_interest": 1500.0},
                        },
                    ],
                    mode="overwrite",
                )
                assert touched == 1
                df = await store.read_feature_store("GBPUSD", "cot_tff")
                assert len(df) == 1
                assert df.iloc[0]["net_pos"] == pytest.approx(200.0)
                assert df.iloc[0]["open_interest"] == pytest.approx(1500.0)
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_bulk_drops_rows_outside_partition_window(self):
        """
        Regression test for the partition-window filter. Rows with timestamp
        outside [PARTITION_START, PARTITION_END) are silently dropped with a
        WARNING log — Stooq US 10Y series goes back to 1871 and would
        otherwise crash on insert with "no partition for given key".
        """
        async def _run():
            engine = await _setup_test_db()
            try:
                store = _make_store(engine)
                rows = [
                    {
                        "symbol": "GBPUSD",
                        "timestamp": datetime(1900, 1, 1),   # pre-window
                        "feature_group": "stooq_yields",
                        "values": {"us_10y": 4.0},
                    },
                    {
                        "symbol": "GBPUSD",
                        "timestamp": datetime(2024, 6, 1),   # in-window
                        "feature_group": "stooq_yields",
                        "values": {"us_10y": 4.5},
                    },
                ]
                inserted = await store.upsert_feature_store_bulk(rows)
                assert inserted == 1   # only the in-window row
                df = await store.read_feature_store("GBPUSD", "stooq_yields")
                assert len(df) == 1
                assert df.index[0].to_pydatetime() == datetime(2024, 6, 1)
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_bulk_rejects_invalid_mode(self):
        """``mode`` is typed as Literal['skip','overwrite']; runtime guard."""
        async def _run():
            engine = await _setup_test_db()
            try:
                store = _make_store(engine)
                with pytest.raises(ValueError, match="mode must be"):
                    await store.upsert_feature_store_bulk(
                        [{
                            "symbol": "X", "timestamp": datetime(2024, 1, 1),
                            "feature_group": "test", "values": {"a": 1.0},
                        }],
                        mode="bogus",  # type: ignore[arg-type]
                    )
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_read_filters_by_time_range(self):
        async def _run():
            engine = await _setup_test_db()
            try:
                store = _make_store(engine)
                for month in (1, 2, 3, 4, 5):
                    await store.upsert_feature_store(
                        symbol="USDCAD", timestamp=datetime(2026, month, 1),
                        feature_group="fred_macro", values={"x": float(month)},
                    )
                df = await store.read_feature_store(
                    "USDCAD", "fred_macro",
                    start=datetime(2026, 2, 1),
                    end=datetime(2026, 4, 1),
                )
                assert len(df) == 3
                assert list(df["x"]) == [2.0, 3.0, 4.0]
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Layer 3 — Collector persist_raw_history_to_feature_store (Phase 1F)
#
# Each fetcher's persist method is exercised end-to-end against the live
# Postgres test DB, but the EXTERNAL HTTP layer is monkey-patched so tests
# don't hit FRED / CFTC / ECB / Stooq / Yahoo. We replace the in-memory
# cache layer with synthetic data, then assert the resulting feature_store
# rows have the right shape, feature_group, and symbol routing.
# ---------------------------------------------------------------------------


def _series(start: str, n: int, step_days: int = 1, base: float = 1.0):
    """Build a small numeric pd.Series indexed by date for mocking."""
    import pandas as pd  # local import — module-level pandas not loaded for unit-only runs
    idx = pd.date_range(start=start, periods=n, freq=f"{step_days}D")
    return pd.Series([base + i * 0.01 for i in range(n)], index=idx)


def _df_close(start: str, n: int, base: float = 1.0):
    """Build a DataFrame with a Date index and 'Close' column."""
    import pandas as pd
    idx = pd.date_range(start=start, periods=n, freq="D")
    return pd.DataFrame({"Close": [base + i * 0.01 for i in range(n)]}, index=idx)


@requires_postgres
class TestFetcherPersistIntegration:
    """
    End-to-end persist tests for each of the 5 collectors.

    Pattern: spin up the test DB, monkey-patch the fetcher's internal
    cache to return a synthetic series, call persist, then read back via
    DataStore.read_feature_store and assert.
    """

    def test_ecb_persists_under_global_with_curve_keys(self):
        async def _run():
            engine = await _setup_test_db()
            try:
                from src.data_pipeline.market.ecb_data import (
                    ECB_SERIES, ECBDataFetcher,
                )
                store = _make_store(engine)
                fetcher = ECBDataFetcher()

                # Stub each tenor with a small synthetic series.
                synthetic = {label: _df_close("2024-01-01", 10, base=1.0 + i)
                              for i, label in enumerate(ECB_SERIES)}
                fetcher.get_series = lambda label: synthetic.get(label)  # type: ignore[assignment]

                n = await fetcher.persist_raw_history_to_feature_store(store)
                assert n == 10, f"expected 10 rows (one per date), got {n}"

                df = await store.read_feature_store("_GLOBAL", "ecb_yield_curve")
                assert len(df) == 10
                # Every tenor + the two derived slopes should land as columns.
                for label in ECB_SERIES:
                    assert f"eu_aaa_{label}_daily" in df.columns
                assert "eu_aaa_slope_2y10y" in df.columns
                assert "eu_aaa_slope_3m10y" in df.columns
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_stooq_persists_per_symbol_with_routed_countries(self):
        async def _run():
            engine = await _setup_test_db()
            try:
                from src.data_pipeline.market.stooq_data import (
                    STOOQ_SERIES, StooqFetcher,
                )
                store = _make_store(engine)
                fetcher = StooqFetcher()

                synthetic = {label: _df_close("2024-01-01", 8, base=1.0 + i)
                              for i, label in enumerate(STOOQ_SERIES)}
                fetcher.get_series = lambda label: synthetic.get(label, _df_close("2024-01-01", 0))  # type: ignore[assignment]

                # GBPUSD routes to US-axis + UK block.
                n = await fetcher.persist_raw_history_to_feature_store(store, "GBPUSD")
                assert n == 8

                df = await store.read_feature_store("GBPUSD", "stooq_yields")
                assert len(df) == 8
                # US-axis baseline (always emitted)
                assert "us_2y_daily" in df.columns
                assert "us_10y_daily" in df.columns
                assert "us_slope_daily" in df.columns
                # UK block (GBP exposure)
                assert "uk_2y_daily" in df.columns
                assert "uk_10y_daily" in df.columns
                # No EUR / JPY / AUD blocks for a pure GBP pair
                assert "de_2y_daily" not in df.columns
                assert "jp_2y_daily" not in df.columns
                assert "au_2y_daily" not in df.columns
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_fred_macro_persists_routed_series_for_gbp(self):
        async def _run():
            engine = await _setup_test_db()
            try:
                from src.data_pipeline.fundamental.macro_data import MacroDataFetcher

                store = _make_store(engine)
                # Construct without HTTP — bypass __init__'s FRED key check.
                fetcher = MacroDataFetcher.__new__(MacroDataFetcher)
                fetcher._cache = {}
                fetcher._cache_ts = None

                # Stub get_series (the persist path calls it directly with
                # a 25-yr lookback, bypassing the live _get_cached).
                def _stub(series_id: str, lookback_days: int = 365):
                    return _series("2024-01-01", 6, base=hash(series_id) % 10)
                fetcher.get_series = _stub  # type: ignore[assignment]

                n = await fetcher.persist_raw_history_to_feature_store(store, "GBPUSD")
                assert n == 6

                df = await store.read_feature_store("GBPUSD", "fred_macro")
                assert len(df) == 6
                # Common (USD-axis) series always present
                assert "fed_funds" in df.columns
                assert "dxy" in df.columns
                # GBP-block series present (per routing)
                assert "boe_rate" in df.columns
                assert "uk_10y" in df.columns
                # AUD-block / NZD-block series NOT present
                assert "rba_rate" not in df.columns
                assert "rbnz_rate" not in df.columns
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_cot_persists_xau_disagg_and_fx_tff(self):
        async def _run():
            import pandas as pd

            engine = await _setup_test_db()
            try:
                from src.data_pipeline.fundamental.cot_data import COTDataFetcher
                store = _make_store(engine)
                fetcher = COTDataFetcher()

                # Stub the multi-year fetcher (persist bypasses the live
                # _get_gold_data cache and pulls explicit yearly zips).
                gold = pd.DataFrame({
                    "date": pd.to_datetime(["2024-01-09", "2024-01-16", "2024-01-23"]),
                    "mm_long": [100, 110, 120],
                    "mm_short": [50, 55, 60],
                    "comm_long": [200, 210, 220],
                    "comm_short": [150, 145, 140],
                    "open_interest": [1000, 1010, 1020],
                    "net_spec": [50, 55, 60],
                    "net_comm": [50, 65, 80],
                })
                fetcher._fetch_multi_year_disagg = lambda years: gold  # type: ignore[assignment]

                n_xau = await fetcher.persist_raw_history_to_feature_store(store, "XAUUSD")
                assert n_xau == 3

                df_xau = await store.read_feature_store("XAUUSD", "cot_disagg")
                assert len(df_xau) == 3
                assert "mm_long" in df_xau.columns
                assert "open_interest" in df_xau.columns
                assert "net_spec" in df_xau.columns

                # Stub TFF cache with EUR + GBP rows for two weeks.
                tff = pd.DataFrame({
                    "currency": ["EUR", "EUR", "GBP", "GBP"],
                    "date": pd.to_datetime([
                        "2024-01-09", "2024-01-16",
                        "2024-01-09", "2024-01-16",
                    ]),
                    "dealer_long":  [10, 11, 20, 21],
                    "dealer_short": [5,  6,  10, 11],
                    "lev_long":     [30, 31, 40, 41],
                    "lev_short":    [15, 16, 20, 21],
                    "open_interest": [100, 110, 200, 210],
                    "net_spec":      [15, 15, 20, 20],
                    "net_dealer":    [5, 5, 10, 10],
                })
                fetcher._fetch_multi_year_tff = lambda years: tff  # type: ignore[assignment]

                # EURGBP routes to BOTH EUR and GBP — should emit one row
                # per date with both currencies' columns merged.
                n_fx = await fetcher.persist_raw_history_to_feature_store(store, "EURGBP")
                assert n_fx == 2

                df_fx = await store.read_feature_store("EURGBP", "cot_tff")
                assert len(df_fx) == 2
                assert "eur_lev_long" in df_fx.columns
                assert "gbp_lev_long" in df_fx.columns
                assert "eur_dealer_long" in df_fx.columns
                assert "gbp_dealer_long" in df_fx.columns
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_yfinance_persists_via_historical_helper(self):
        async def _run():
            import pandas as pd

            engine = await _setup_test_db()
            try:
                from src.data_pipeline.market.cross_asset import CrossAssetFetcher
                store = _make_store(engine)
                fetcher = CrossAssetFetcher()

                idx = pd.date_range("2024-02-01", periods=5, freq="D")
                synthetic_df = pd.DataFrame({
                    "dxy_log_return": [0.01] * 5,
                    "vix_level":      [15.0] * 5,
                    "spx_zscore":     [0.5] * 5,
                    "ftse_log_return": [0.002] * 5,
                }, index=idx)
                fetcher.get_historical_cross_asset_features = (
                    lambda symbol, start, end: synthetic_df  # type: ignore[assignment]
                )

                n = await fetcher.persist_raw_history_to_feature_store(store, "GBPUSD")
                assert n == 5

                df = await store.read_feature_store("GBPUSD", "yfinance_cross_asset")
                assert len(df) == 5
                assert "dxy_log_return" in df.columns
                assert "vix_level" in df.columns
                assert "ftse_log_return" in df.columns
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_persist_is_idempotent_on_rerun(self):
        async def _run():
            engine = await _setup_test_db()
            try:
                from src.data_pipeline.market.ecb_data import (
                    ECB_SERIES, ECBDataFetcher,
                )
                store = _make_store(engine)
                fetcher = ECBDataFetcher()
                synthetic = {label: _df_close("2024-01-01", 7) for label in ECB_SERIES}
                fetcher.get_series = lambda label: synthetic.get(label)  # type: ignore[assignment]

                first = await fetcher.persist_raw_history_to_feature_store(store)
                second = await fetcher.persist_raw_history_to_feature_store(store)
                assert first == 7
                assert second == 0   # everything already present, DO NOTHING
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())
