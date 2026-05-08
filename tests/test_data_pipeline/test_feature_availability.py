"""
Phase 1E — feature-availability schema tests.

Two layers, mirroring test_feature_store.py:

1. Pure-unit tests for config loading + lag math + error paths.
   No DB. Runs in CI alongside every other unit test.

2. Opt-in integration test against a real Postgres cluster, gated on
   ``CORTEX_POSTGRES_TEST_DSN``. Verifies that read_feature_store_safe
   actually filters out rows that would leak future information.

Async tests follow the repo convention (asyncio.run inside sync test
body — see test_feature_store.py).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import pytest

from src.data_pipeline.feature_engineering import (
    _DATA_FEEDS_PATH,
    _load_data_feeds_yaml,
    read_feature_store_safe,
)


# ---------------------------------------------------------------------------
# Layer 1 — Config + lag-math unit tests (no DB)
# ---------------------------------------------------------------------------


class TestDataFeedsConfig:
    """Loading + structure of config/data_feeds.yaml."""

    def test_config_loads_from_canonical_path(self):
        cfg = _load_data_feeds_yaml(force=True)
        assert "sources" in cfg

    def test_all_six_known_sources_present(self):
        """Every feature_group emitted by the 5 collectors must have a config block."""
        cfg = _load_data_feeds_yaml(force=True)
        expected = {
            "fred_macro",
            "cot_disagg",
            "cot_tff",
            "ecb_yield_curve",
            "stooq_yields",
            "yfinance_cross_asset",
        }
        assert expected.issubset(cfg["sources"].keys())

    def test_each_source_has_release_lag(self):
        cfg = _load_data_feeds_yaml(force=True)
        for name, src in cfg["sources"].items():
            assert "release_lag_hours" in src, f"{name} missing release_lag_hours"
            assert isinstance(src["release_lag_hours"], (int, float))
            assert src["release_lag_hours"] >= 0


class TestLagApplication:
    """Verify read_feature_store_safe applies the right lag to as_of."""

    @staticmethod
    def _stub_store(captured: dict):
        """Build a mock DataStore that records the bounds it was queried with."""
        store = AsyncMock()
        async def _capture(symbol, feature_group, start, end):
            captured["symbol"] = symbol
            captured["feature_group"] = feature_group
            captured["start"] = start
            captured["end"] = end
            return pd.DataFrame()
        store.read_feature_store = _capture
        return store

    def test_lag_subtracted_from_as_of(self):
        """A 24h-lag source queried at as_of=T should hit feature_store with end = T - 24h."""
        captured: dict = {}
        store = self._stub_store(captured)
        cfg = {"sources": {"ecb_yield_curve": {"release_lag_hours": 24}}}
        as_of = datetime(2026, 4, 25, 12, 0, 0)
        asyncio.run(read_feature_store_safe(
            store, "_GLOBAL", "ecb_yield_curve", as_of, feeds_config=cfg,
        ))
        assert captured["end"] == as_of - timedelta(hours=24)

    def test_default_start_is_one_year_before_effective_end(self):
        captured: dict = {}
        store = self._stub_store(captured)
        cfg = {"sources": {"fred_macro": {"release_lag_hours": 504}}}
        as_of = datetime(2026, 4, 25, 12, 0, 0)
        asyncio.run(read_feature_store_safe(
            store, "GBPUSD", "fred_macro", as_of, feeds_config=cfg,
        ))
        effective_end = as_of - timedelta(hours=504)
        assert captured["start"] == effective_end - timedelta(days=365)
        assert captured["end"] == effective_end

    def test_explicit_start_overrides_default(self):
        captured: dict = {}
        store = self._stub_store(captured)
        cfg = {"sources": {"stooq_yields": {"release_lag_hours": 24}}}
        as_of = datetime(2026, 4, 25, 12, 0, 0)
        explicit_start = datetime(2020, 1, 1)
        asyncio.run(read_feature_store_safe(
            store, "GBPUSD", "stooq_yields", as_of,
            start=explicit_start, feeds_config=cfg,
        ))
        assert captured["start"] == explicit_start

    def test_zero_lag_passes_as_of_through_unchanged(self):
        captured: dict = {}
        store = self._stub_store(captured)
        cfg = {"sources": {"ohlcv_bars": {"release_lag_hours": 0}}}
        as_of = datetime(2026, 4, 25, 12, 0, 0)
        asyncio.run(read_feature_store_safe(
            store, "GBPUSD", "ohlcv_bars", as_of, feeds_config=cfg,
        ))
        assert captured["end"] == as_of


class TestErrorPaths:
    """Refuse-to-query semantics for unknown / malformed sources."""

    def test_unknown_feature_group_raises(self):
        store = AsyncMock()
        cfg = {"sources": {"fred_macro": {"release_lag_hours": 504}}}
        with pytest.raises(ValueError, match="not in data_feeds.yaml"):
            asyncio.run(read_feature_store_safe(
                store, "GBPUSD", "made_up_source", datetime(2026, 1, 1),
                feeds_config=cfg,
            ))

    def test_missing_release_lag_raises(self):
        store = AsyncMock()
        cfg = {"sources": {"weird_source": {"cutoff_utc": "12:00"}}}  # no lag key
        with pytest.raises(ValueError, match="release_lag_hours missing"):
            asyncio.run(read_feature_store_safe(
                store, "GBPUSD", "weird_source", datetime(2026, 1, 1),
                feeds_config=cfg,
            ))

    def test_non_numeric_release_lag_raises(self):
        store = AsyncMock()
        cfg = {"sources": {"weird_source": {"release_lag_hours": "twenty four"}}}
        with pytest.raises(ValueError, match="missing or invalid"):
            asyncio.run(read_feature_store_safe(
                store, "GBPUSD", "weird_source", datetime(2026, 1, 1),
                feeds_config=cfg,
            ))


# ---------------------------------------------------------------------------
# Layer 2 — Live-Postgres integration (opt-in)
# ---------------------------------------------------------------------------

_PG_TEST_DSN = os.environ.get("CORTEX_POSTGRES_TEST_DSN")
_TEST_DBNAME = "trading_bot_test_fa"

requires_postgres = pytest.mark.skipif(
    _PG_TEST_DSN is None,
    reason="set CORTEX_POSTGRES_TEST_DSN to run feature_availability integration tests",
)


def _swap_dbname(dsn: str, new: str) -> str:
    parts = urlsplit(dsn)
    return urlunsplit((parts.scheme, parts.netloc, "/" + new, parts.query, parts.fragment))


async def _setup_test_db():
    """Create+migrate a transient DB, return its engine."""
    from sqlalchemy import text
    from src.data_pipeline.data_store import build_engine
    from src.data_pipeline.db_migrations import (
        count_feature_store_partitions, create_all_tables,
    )
    admin = build_engine(_swap_dbname(_PG_TEST_DSN, "postgres"), isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f"DROP DATABASE IF EXISTS {_TEST_DBNAME}"))
        await conn.execute(text(f"CREATE DATABASE {_TEST_DBNAME}"))
    await admin.dispose()
    engine = build_engine(_swap_dbname(_PG_TEST_DSN, _TEST_DBNAME))
    await create_all_tables(engine)
    assert await count_feature_store_partitions(engine) == 432
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
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from src.data_pipeline.data_store import DataStore
    store = DataStore.__new__(DataStore)
    store._engine = engine
    store._session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return store


@requires_postgres
class TestLookaheadEnforcement:
    """End-to-end: insert a row, query with as_of < release_time, verify excluded."""

    def test_value_published_after_as_of_is_excluded(self):
        """
        ECB yield curve has 24h release lag. A row written at 2026-04-15 12:00
        (source-timestamp) becomes "knowable" at 2026-04-16 12:00. Querying
        with as_of=2026-04-16 06:00 must exclude it.
        """
        async def _run():
            engine = await _setup_test_db()
            try:
                store = _make_store(engine)
                # Insert two rows: one safely "old", one too-recent
                await store.upsert_feature_store(
                    symbol="_GLOBAL",
                    timestamp=datetime(2026, 4, 10, 12, 0, 0),
                    feature_group="ecb_yield_curve",
                    values={"y10": 1.0},
                )
                await store.upsert_feature_store(
                    symbol="_GLOBAL",
                    timestamp=datetime(2026, 4, 15, 12, 0, 0),
                    feature_group="ecb_yield_curve",
                    values={"y10": 1.5},
                )

                # as_of = 2026-04-16 06:00 → effective_end = 2026-04-15 06:00
                # → only the 2026-04-10 row qualifies.
                cfg = {"sources": {"ecb_yield_curve": {"release_lag_hours": 24}}}
                df = await read_feature_store_safe(
                    store, "_GLOBAL", "ecb_yield_curve",
                    as_of=datetime(2026, 4, 16, 6, 0, 0),
                    feeds_config=cfg,
                )
                assert len(df) == 1
                assert df.iloc[0]["y10"] == pytest.approx(1.0)

                # Now the same row is OK at as_of = 2026-04-17
                df2 = await read_feature_store_safe(
                    store, "_GLOBAL", "ecb_yield_curve",
                    as_of=datetime(2026, 4, 17, 0, 0, 0),
                    feeds_config=cfg,
                )
                assert len(df2) == 2
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())

    def test_works_with_canonical_data_feeds_yaml(self):
        """End-to-end with the real config — no test override."""
        async def _run():
            engine = await _setup_test_db()
            try:
                store = _make_store(engine)
                # FRED macro has 504h (21d) lag. Row with timestamp 2026-04-01
                # becomes knowable at 2026-04-22. Query as_of = 2026-04-30.
                await store.upsert_feature_store(
                    symbol="GBPUSD",
                    timestamp=datetime(2026, 4, 1),
                    feature_group="fred_macro",
                    values={"fed_funds": 5.25},
                )
                df = await read_feature_store_safe(
                    store, "GBPUSD", "fred_macro",
                    as_of=datetime(2026, 4, 30),
                )
                assert len(df) == 1

                # Same row at as_of = 2026-04-15 (before release window)
                df2 = await read_feature_store_safe(
                    store, "GBPUSD", "fred_macro",
                    as_of=datetime(2026, 4, 15),
                )
                assert len(df2) == 0
            finally:
                await _teardown_test_db(engine)
        asyncio.run(_run())
