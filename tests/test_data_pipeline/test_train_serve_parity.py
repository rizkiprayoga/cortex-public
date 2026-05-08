"""
test_train_serve_parity.py — Phase 2A wiring smoke tests.

These tests verify the most important invariant of the Phase 2A wiring:
the SET of feature columns produced by the training-time historical
readers must be a subset of the SET emitted by the live ``FundamentalDataManager.get_all_features()``.

Why this matters: any column that exists in training but not at live time
gets zero-filled by ``align_to_manifest`` at inference, silently. The LSTM
learned a real value during training; live inference sees zero. This is
the train/serve skew that broke implicit assumptions in past sessions.

Tests cover:

1. **Schema parity per symbol** — for each of the 4 historical readers,
   the column set returned for a given symbol must be a SUBSET of what
   live ``get_*_features(symbol)`` would emit. Strict equality is checked
   per-source where the live API is well-defined.

2. **Timezone invariant** — every historical reader's index is naive UTC.
   No tz-aware leakage that would silently shift values when joined onto
   the bar grid (the 2026-04-24 incident class).

3. **Bar-grid alignment** — when external features are reindexed onto a
   naive-UTC H4 bar grid via the engineer's helper, no NaN row gets a
   future-dated lookup (point-in-time integrity).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from src.data_pipeline.feature_engineering import FeatureEngineer
from src.data_pipeline.fundamental.cot_data import COTDataFetcher
from src.data_pipeline.fundamental.macro_data import MacroDataFetcher
from src.data_pipeline.market.ecb_data import ECBDataFetcher
from src.data_pipeline.market.stooq_data import StooqFetcher


# ---------------------------------------------------------------------------
# Fixtures — wide raw DataFrames matching what feature_store would emit.
# Imported from the per-fetcher test files would be ideal, but those use
# module-local fixtures; copy minimal versions to keep this test file
# self-contained.
# ---------------------------------------------------------------------------

def _raw_macro(n_days: int = 800) -> pd.DataFrame:
    end = datetime(2024, 12, 31)
    idx = pd.date_range(end=end, periods=n_days, freq="D")
    return pd.DataFrame({
        "fed_funds":         [5.0] * n_days,
        "cpi_yoy":           [3.0] * n_days,
        "real_yield":        [2.0] * n_days,
        "m2":                [21000.0] * n_days,
        "yield_curve":       [-0.5] * n_days,
        "breakeven_inflation": [2.3] * n_days,
        "hy_spread":         [4.2] * n_days,
        "initial_claims":    [220000.0] * n_days,
        "fed_balance_sheet": [7_500_000.0] * n_days,
        "dxy":               [105.0] * n_days,
        "vix":               [18.0] * n_days,
        "ecb_rate":          [3.5] * n_days,
        "eu_cpi_yoy":        [2.5] * n_days,
        "eu_unemployment":   [6.0] * n_days,
        "boj_rate":          [0.25] * n_days,
        "japan_cpi_yoy":     [2.0] * n_days,
        "boc_rate":          [4.5] * n_days,
        "canada_cpi_yoy":    [3.1] * n_days,
        "wti_oil":           [78.0] * n_days,
        "boe_rate":          [4.5] * n_days,
        "uk_10y":            [4.2] * n_days,
        "uk_cpi_yoy":        [3.8] * n_days,
        "rba_rate":          [4.35] * n_days,
        "au_10y":            [4.5] * n_days,
        "au_cpi_yoy":        [3.4] * n_days,
        "iron_ore":          [120.0] * n_days,
        "china_cpi_yoy":     [0.5] * n_days,
        "cny_usd":           [7.2] * n_days,
        "rbnz_rate":         [5.5] * n_days,
        "nz_10y":            [4.8] * n_days,
        "nz_cpi_yoy":        [4.0] * n_days,
    }, index=idx)


def _raw_yield(n_days: int = 200) -> pd.DataFrame:
    end = datetime(2024, 12, 31)
    idx = pd.date_range(end=end, periods=n_days, freq="D")
    cols = {}
    for c in ("us", "uk", "de", "jp", "au", "nz"):
        cols[f"{c}_2y_daily"] = [4.0] * n_days
        cols[f"{c}_10y_daily"] = [4.5] * n_days
        cols[f"{c}_slope_daily"] = [0.5] * n_days
    return pd.DataFrame(cols, index=idx)


def _raw_curve(n_days: int = 200) -> pd.DataFrame:
    end = datetime(2024, 12, 31)
    idx = pd.date_range(end=end, periods=n_days, freq="D")
    cols = {}
    for tenor, lvl in [
        ("3m", 3.50), ("6m", 3.45), ("1y", 3.40), ("2y", 3.30), ("3y", 3.25),
        ("5y", 3.10), ("7y", 3.05), ("10y", 3.00), ("20y", 3.20), ("30y", 3.30),
    ]:
        cols[f"eu_aaa_{tenor}_daily"] = [lvl] * n_days
    cols["eu_aaa_slope_2y10y"] = [-0.30] * n_days
    cols["eu_aaa_slope_3m10y"] = [-0.50] * n_days
    return pd.DataFrame(cols, index=idx)


def _raw_xau_cot(n_weeks: int = 80) -> pd.DataFrame:
    end = datetime(2024, 12, 31)
    idx = pd.date_range(end=end, periods=n_weeks, freq="W-TUE")
    return pd.DataFrame({
        "mm_long":       [180000.0] * n_weeks,
        "mm_short":      [80000.0] * n_weeks,
        "comm_long":     [120000.0] * n_weeks,
        "comm_short":    [220000.0] * n_weeks,
        "open_interest": [500000.0] * n_weeks,
        "net_spec":      [100000.0] * n_weeks,
        "net_comm":      [-100000.0] * n_weeks,
    }, index=idx)


def _raw_fx_cot(n_weeks: int = 80) -> pd.DataFrame:
    end = datetime(2024, 12, 31)
    idx = pd.date_range(end=end, periods=n_weeks, freq="W-TUE")
    cols = {}
    for c in ("eur", "jpy", "cad", "gbp", "aud", "nzd"):
        cols[f"{c}_dealer_long"] = [80000.0] * n_weeks
        cols[f"{c}_dealer_short"] = [40000.0] * n_weeks
        cols[f"{c}_lev_long"] = [100000.0] * n_weeks
        cols[f"{c}_lev_short"] = [60000.0] * n_weeks
        cols[f"{c}_open_interest"] = [200000.0] * n_weeks
        cols[f"{c}_net_spec"] = [40000.0] * n_weeks
        cols[f"{c}_net_dealer"] = [40000.0] * n_weeks
    return pd.DataFrame(cols, index=idx)


def _store_returning_per_group(group_to_df: dict[str, pd.DataFrame]) -> AsyncMock:
    store = AsyncMock()
    async def _read(symbol, feature_group, start=None, end=None):
        return group_to_df.get(feature_group, pd.DataFrame()).copy()
    store.read_feature_store = _read
    return store


# ---------------------------------------------------------------------------
# 1. Schema parity per source — historical column SET ⊆ live-API SET.
# ---------------------------------------------------------------------------

class TestMacroSchemaIsSubsetOfLive:
    """get_historical_macro_features columns must match live get_macro_features keys."""

    @pytest.mark.parametrize("symbol", [
        "XAUUSD", "EURUSD", "USDJPY", "USDCAD",
        "GBPUSD", "AUDUSD", "EURGBP", "EURJPY", "GBPJPY", "AUDNZD",
    ])
    def test_macro_columns_subset_of_live(self, symbol):
        cfg = {"sources": {"fred_macro": {"release_lag_hours": 504}}}
        store = _store_returning_per_group({"fred_macro": _raw_macro()})
        # Construct fetcher without FRED_API_KEY (test path only uses
        # historical reader + helpers).
        fetcher = MacroDataFetcher.__new__(MacroDataFetcher)
        fetcher._cache = {}
        fetcher._cache_ts = datetime.utcnow()
        fetcher._cache_ttl = timedelta(hours=4)
        # Stub _get_cached so live get_macro_features() works for parity check
        def _stub_cached(key: str):
            idx = pd.date_range(end=datetime.utcnow(), periods=90, freq="D")
            return pd.Series([1.0] * 90, index=idx)
        fetcher._get_cached = _stub_cached  # type: ignore

        live_keys = set(fetcher.get_macro_features(symbol).keys())
        df = asyncio.run(fetcher.get_historical_macro_features(
            store, symbol,
            datetime(2023, 1, 1), datetime(2024, 12, 1),
            feeds_config=cfg,
        ))
        hist_cols = set(df.columns)
        assert hist_cols == live_keys, (
            f"{symbol}: hist-only={hist_cols - live_keys}, "
            f"live-only={live_keys - hist_cols}"
        )


class TestYieldSchemaIsSubsetOfDefault:
    """Stooq historical columns == default_yield_features keys (live API
    hits the network; default_yield_features is the operative reference)."""

    @pytest.mark.parametrize("symbol", [
        "XAUUSD", "EURUSD", "USDJPY", "USDCAD",
        "GBPUSD", "AUDUSD", "EURGBP", "EURJPY", "GBPJPY", "AUDNZD",
    ])
    def test_yield_columns_match_default(self, symbol):
        fetcher = StooqFetcher()
        cfg = {"sources": {"stooq_yields": {"release_lag_hours": 24}}}
        store = _store_returning_per_group({"stooq_yields": _raw_yield()})
        live_keys = set(fetcher.default_yield_features(symbol).keys())
        df = asyncio.run(fetcher.get_historical_yield_features(
            store, symbol,
            datetime(2023, 1, 1), datetime(2024, 12, 1),
            feeds_config=cfg,
        ))
        assert set(df.columns) == live_keys, (
            f"{symbol}: hist={set(df.columns)}, default={live_keys}"
        )


class TestECBSchemaIsSubsetOfDefault:
    """ECB historical columns must match default_yield_curve_features keys
    for EUR pairs; non-EUR pairs return empty (matches live early return)."""

    @pytest.mark.parametrize("symbol", ["EURUSD", "EURGBP", "EURJPY"])
    def test_eur_columns_match_default(self, symbol):
        fetcher = ECBDataFetcher()
        cfg = {"sources": {"ecb_yield_curve": {"release_lag_hours": 24}}}
        store = _store_returning_per_group({"ecb_yield_curve": _raw_curve()})
        live_keys = set(fetcher.default_yield_curve_features(symbol).keys())
        df = asyncio.run(fetcher.get_historical_curve_features(
            store, symbol,
            datetime(2023, 1, 1), datetime(2024, 12, 1),
            feeds_config=cfg,
        ))
        assert set(df.columns) == live_keys

    @pytest.mark.parametrize("symbol", [
        "XAUUSD", "USDJPY", "USDCAD", "GBPUSD", "AUDUSD", "AUDNZD",
    ])
    def test_non_eur_pairs_return_empty(self, symbol):
        fetcher = ECBDataFetcher()
        cfg = {"sources": {"ecb_yield_curve": {"release_lag_hours": 24}}}
        store = _store_returning_per_group({"ecb_yield_curve": _raw_curve()})
        df = asyncio.run(fetcher.get_historical_curve_features(
            store, symbol,
            datetime(2023, 1, 1), datetime(2024, 12, 1),
            feeds_config=cfg,
        ))
        assert df.empty


class TestCOTSchemaIsSubsetOfDefault:
    """COT historical columns match default keys (XAU disagg + per-currency TFF)."""

    def test_xau_columns_match_default(self):
        fetcher = COTDataFetcher()
        cfg = {"sources": {
            "cot_disagg": {"release_lag_hours": 504},
            "cot_tff":    {"release_lag_hours": 504},
        }}
        store = _store_returning_per_group({"cot_disagg": _raw_xau_cot()})
        live_keys = set(COTDataFetcher._default_xau_features().keys())
        df = asyncio.run(fetcher.get_historical_cot_features(
            store, "XAUUSD",
            datetime(2023, 1, 1), datetime(2024, 12, 1),
            feeds_config=cfg,
        ))
        assert set(df.columns) == live_keys

    @pytest.mark.parametrize("symbol,currencies", [
        ("EURUSD", ("EUR",)),
        ("USDJPY", ("JPY",)),
        ("EURGBP", ("EUR", "GBP")),
        ("AUDNZD", ("AUD", "NZD")),
    ])
    def test_fx_columns_match_default(self, symbol, currencies):
        fetcher = COTDataFetcher()
        cfg = {"sources": {
            "cot_disagg": {"release_lag_hours": 504},
            "cot_tff":    {"release_lag_hours": 504},
        }}
        store = _store_returning_per_group({"cot_tff": _raw_fx_cot()})
        live_keys = set(COTDataFetcher._default_fx_features(currencies).keys())
        df = asyncio.run(fetcher.get_historical_cot_features(
            store, symbol,
            datetime(2023, 1, 1), datetime(2024, 12, 1),
            feeds_config=cfg,
        ))
        assert set(df.columns) == live_keys


# ---------------------------------------------------------------------------
# 2. Timezone invariant — every historical output is naive UTC.
# ---------------------------------------------------------------------------

class TestHistoricalOutputsAreNaiveUTC:
    """Critical invariant: every historical reader returns a tz-naive
    DatetimeIndex. Tz-aware leakage joined onto a naive bar grid silently
    shifts values by hours — same class of bug as the 2026-04-24 incident.
    """

    def test_macro_index_is_naive(self):
        fetcher = MacroDataFetcher.__new__(MacroDataFetcher)
        fetcher._cache = {}
        fetcher._cache_ts = datetime.utcnow()
        fetcher._cache_ttl = timedelta(hours=4)
        cfg = {"sources": {"fred_macro": {"release_lag_hours": 504}}}
        store = _store_returning_per_group({"fred_macro": _raw_macro()})
        df = asyncio.run(fetcher.get_historical_macro_features(
            store, "EURUSD",
            datetime(2023, 1, 1), datetime(2024, 12, 1),
            feeds_config=cfg,
        ))
        assert df.index.tz is None

    def test_yield_index_is_naive(self):
        fetcher = StooqFetcher()
        cfg = {"sources": {"stooq_yields": {"release_lag_hours": 24}}}
        store = _store_returning_per_group({"stooq_yields": _raw_yield()})
        df = asyncio.run(fetcher.get_historical_yield_features(
            store, "EURUSD",
            datetime(2023, 1, 1), datetime(2024, 12, 1),
            feeds_config=cfg,
        ))
        assert df.index.tz is None

    def test_curve_index_is_naive(self):
        fetcher = ECBDataFetcher()
        cfg = {"sources": {"ecb_yield_curve": {"release_lag_hours": 24}}}
        store = _store_returning_per_group({"ecb_yield_curve": _raw_curve()})
        df = asyncio.run(fetcher.get_historical_curve_features(
            store, "EURUSD",
            datetime(2023, 1, 1), datetime(2024, 12, 1),
            feeds_config=cfg,
        ))
        assert df.index.tz is None

    def test_xau_cot_index_is_naive(self):
        fetcher = COTDataFetcher()
        cfg = {"sources": {
            "cot_disagg": {"release_lag_hours": 504},
            "cot_tff":    {"release_lag_hours": 504},
        }}
        store = _store_returning_per_group({"cot_disagg": _raw_xau_cot()})
        df = asyncio.run(fetcher.get_historical_cot_features(
            store, "XAUUSD",
            datetime(2023, 1, 1), datetime(2024, 12, 1),
            feeds_config=cfg,
        ))
        assert df.index.tz is None

    def test_fx_cot_index_is_naive(self):
        fetcher = COTDataFetcher()
        cfg = {"sources": {
            "cot_disagg": {"release_lag_hours": 504},
            "cot_tff":    {"release_lag_hours": 504},
        }}
        store = _store_returning_per_group({"cot_tff": _raw_fx_cot()})
        df = asyncio.run(fetcher.get_historical_cot_features(
            store, "EURUSD",
            datetime(2023, 1, 1), datetime(2024, 12, 1),
            feeds_config=cfg,
        ))
        assert df.index.tz is None


# ---------------------------------------------------------------------------
# 3. Bar-grid alignment — _align_external_to_bar_grid normalizes TZ.
# ---------------------------------------------------------------------------

class TestAlignExternalToBarGrid:
    """The helper must defensively normalize tz-aware indices to naive UTC
    so a stray yfinance / SDMX feed with an explicit TZ doesn't silently
    shift values when ffilled onto a naive bar grid."""

    def test_naive_external_naive_bar_passes_through(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        ext = pd.DataFrame({"x": range(10)}, index=idx)
        bar_idx = pd.date_range("2024-01-05", periods=5, freq="D")
        out = FeatureEngineer._align_external_to_bar_grid(ext, bar_idx)
        assert list(out["x"]) == [4, 5, 6, 7, 8]
        assert out.index.tz is None

    def test_tz_aware_external_dropped_to_naive(self):
        """yfinance returns Etc/UTC tz-aware; we must strip to naive without
        shifting clock values."""
        idx = pd.date_range(
            "2024-01-01", periods=10, freq="D", tz="UTC",
        )
        ext = pd.DataFrame({"x": range(10)}, index=idx)
        bar_idx = pd.date_range("2024-01-05", periods=5, freq="D")  # naive
        out = FeatureEngineer._align_external_to_bar_grid(ext, bar_idx)
        # values should not shift — index just gets stripped
        assert list(out["x"]) == [4, 5, 6, 7, 8]
        assert out.index.tz is None

    def test_ffill_does_not_leak_future_data(self):
        """ffill may only carry data FORWARD, never backward. A bar at
        2024-01-10 with ext data starting only 2024-01-15 must be NaN
        (filled with 0.0 by the helper), not pulled from the future.
        """
        ext = pd.DataFrame({"x": [99.0]}, index=pd.DatetimeIndex(["2024-01-15"]))
        bar_idx = pd.date_range("2024-01-10", periods=10, freq="D")
        out = FeatureEngineer._align_external_to_bar_grid(ext, bar_idx)
        # bars before 2024-01-15 have nothing to ffill — should be 0.0
        assert (out.loc[:"2024-01-14", "x"] == 0.0).all()
        # bars from 2024-01-15 onward see the value
        assert (out.loc["2024-01-15":, "x"] == 99.0).all()


# ---------------------------------------------------------------------------
# 4. Zero-fill list shrunk — Phase 2A removed per-symbol macro features.
# ---------------------------------------------------------------------------

class TestZeroFillListShrunk:
    """get_zero_fill_feature_names() should NO LONGER list per-currency macro
    features (they're now backfilled via feature_store). Only true
    no-historical-source columns remain."""

    @pytest.mark.parametrize("symbol", [
        "EURUSD", "USDJPY", "USDCAD", "GBPUSD", "AUDUSD",
        "EURGBP", "EURJPY", "GBPJPY", "AUDNZD", "XAUUSD", "ETHUSD",
    ])
    def test_no_per_currency_macro_in_zero_fill(self, symbol):
        zf = set(FeatureEngineer.get_zero_fill_feature_names(symbol))
        # These were zero-filled before Phase 2A, MUST NOT be after.
        for col in (
            "ecb_rate_level", "eur_usd_rate_diff", "eu_cpi_yoy_zscore",
            "eu_unemployment_level",
            "boj_rate_level", "usd_jpy_rate_diff", "carry_trade_indicator",
            "japan_cpi_yoy_zscore",
            "boc_rate_level", "usd_cad_rate_diff", "canada_cpi_yoy_zscore",
            "wti_oil_zscore",
        ):
            assert col not in zf, (
                f"{symbol}: {col} should be backfilled via feature_store, "
                f"not zero-filled. Phase 2A retraining will leak this column "
                f"as zero in training but live will populate it."
            )

    def test_legacy_scalar_scores_still_zero_filled(self):
        """The 4 backward-compat scalar scores have no historical backfill
        and must remain in the zero-fill list."""
        zf = set(FeatureEngineer.get_zero_fill_feature_names("EURUSD"))
        for col in ("sentiment_score", "onchain_score", "macro_score", "cot_score"):
            assert col in zf


# ---------------------------------------------------------------------------
# 5. End-to-end — full transform_with_externals async path.
# ---------------------------------------------------------------------------

def _mock_h4_ohlcv(n_bars: int = 100, end: datetime | None = None) -> pd.DataFrame:
    """Build a mock H4 OHLCV DataFrame indexed by naive UTC bar timestamps.

    Uses pseudo-random walk so rolling stats (skew/kurt/std) don't degenerate
    to NaN on a perfectly linear series — that would empty out transform()'s
    NaN-drop step.
    """
    import numpy as np
    if end is None:
        end = datetime(2024, 12, 1)
    idx = pd.date_range(end=end, periods=n_bars, freq="4h")
    rng = np.random.default_rng(seed=42)
    closes = 1900.0 + np.cumsum(rng.standard_normal(n_bars) * 2.0)
    highs = closes + rng.uniform(0.5, 5.0, n_bars)
    lows = closes - rng.uniform(0.5, 5.0, n_bars)
    opens = closes + rng.standard_normal(n_bars) * 1.0
    return pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": rng.uniform(800, 1200, n_bars),
        "tick_volume": rng.uniform(800, 1200, n_bars),
    }, index=idx)


class TestEndToEndTransformWithExternals:
    """Drives the async transform path end-to-end. Post Phase A revert
    (spec §1 anchor 7), the LSTM input path no longer joins the 4
    feature_store-backed historical readers — only cross-asset + calendar
    + zero-fill stay. The historical-reader regression guard lives in
    test_phase_a_revert.py; what survives here is the no-NaN invariant."""

    def _store_for_eurusd(self) -> AsyncMock:
        """Mock store covering all feature_groups EURUSD touches."""
        store = AsyncMock()
        async def _read(symbol, feature_group, start=None, end=None):
            if feature_group == "fred_macro":
                return _raw_macro()
            if feature_group == "stooq_yields":
                return _raw_yield()
            if feature_group == "ecb_yield_curve":
                return _raw_curve()
            if feature_group == "cot_tff":
                return _raw_fx_cot()
            return pd.DataFrame()
        store.read_feature_store = _read
        return store

    def test_async_path_no_nans_in_output(self, monkeypatch):
        """After fillna(0.0), result must contain no NaN — alignment
        helper + final fillna jointly cover every source."""
        from src.data_pipeline import feature_engineering as fe_mod
        monkeypatch.setattr(fe_mod, "_data_feeds_cache", {
            "sources": {
                "fred_macro":      {"release_lag_hours": 504},
                "stooq_yields":    {"release_lag_hours": 24},
                "ecb_yield_curve": {"release_lag_hours": 24},
                "cot_disagg":      {"release_lag_hours": 504},
                "cot_tff":         {"release_lag_hours": 504},
            },
        })
        monkeypatch.setattr(
            "src.data_pipeline.market.cross_asset.CrossAssetFetcher."
            "get_historical_cross_asset_features",
            lambda self, symbol, start, end: pd.DataFrame(),
        )
        from src.data_pipeline.fundamental.macro_data import MacroDataFetcher
        monkeypatch.setattr(MacroDataFetcher, "__init__", lambda self: None)

        store = self._store_for_eurusd()
        engineer = FeatureEngineer(data_store=store)
        ohlcv = _mock_h4_ohlcv(n_bars=300)

        zf = FeatureEngineer.get_zero_fill_feature_names("EURUSD")
        result = engineer.transform_with_externals(ohlcv, "EURUSD", zero_fill_cols=zf)
        assert not result.isna().any().any(), (
            "transform_with_externals output contains NaN — alignment helper "
            "or final fillna(0.0) failed to cover some source"
        )
