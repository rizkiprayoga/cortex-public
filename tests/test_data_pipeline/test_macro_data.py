"""Tests for macro_data per-symbol routing (Forex Phase 1)."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.data_pipeline.fundamental.macro_data import (
    FRED_SERIES,
    MacroDataFetcher,
    _AUD_EXPOSURE,
    _CAD_EXPOSURE,
    _EUR_EXPOSURE,
    _GBP_EXPOSURE,
    _JPY_EXPOSURE,
    _NZD_EXPOSURE,
)


# -------- Test fixtures ---------------------------------------------------

def _dummy_series(value: float = 1.0, n: int = 90) -> pd.Series:
    """Build a pd.Series with n daily observations of `value`."""
    idx = pd.date_range(end=datetime.utcnow(), periods=n, freq="D")
    return pd.Series([value] * n, index=idx)


def _make_fetcher(fake_values: dict[str, float] | None = None) -> MacroDataFetcher:
    """
    Construct a MacroDataFetcher that bypasses FRED API entirely.

    _get_cached is monkey-patched to return a constant-value dummy series
    per key. Default value is 1.0; override specific keys via fake_values.
    """
    fetcher = MacroDataFetcher.__new__(MacroDataFetcher)
    fetcher._cache = {}
    fetcher._cache_ts = datetime.utcnow()
    fetcher._cache_ttl = timedelta(hours=4)

    values = fake_values or {}

    def fake_get_cached(key: str) -> pd.Series:
        return _dummy_series(values.get(key, 1.0))

    fetcher._get_cached = fake_get_cached  # type: ignore[attr-defined]
    return fetcher


# -------- Exposure set sanity --------------------------------------------

class TestExposureSets:
    """Each currency-exposure set must include every pair with that currency."""

    def test_eur_exposure_covers_all_eur_pairs(self):
        for sym in ("EURUSD", "EURGBP", "EURJPY"):
            assert sym in _EUR_EXPOSURE

    def test_jpy_exposure_covers_all_jpy_pairs(self):
        for sym in ("USDJPY", "EURJPY", "GBPJPY"):
            assert sym in _JPY_EXPOSURE

    def test_gbp_exposure_covers_all_gbp_pairs(self):
        for sym in ("GBPUSD", "EURGBP", "GBPJPY"):
            assert sym in _GBP_EXPOSURE

    def test_aud_exposure_covers_all_aud_pairs(self):
        for sym in ("AUDUSD", "AUDNZD"):
            assert sym in _AUD_EXPOSURE

    def test_nzd_exposure_covers_only_audnzd(self):
        assert "AUDNZD" in _NZD_EXPOSURE
        assert "AUDUSD" not in _NZD_EXPOSURE
        assert "USDCAD" not in _NZD_EXPOSURE

    def test_cad_exposure_is_usdcad_only(self):
        assert "USDCAD" in _CAD_EXPOSURE
        assert "AUDUSD" not in _CAD_EXPOSURE


class TestFredSeriesCoverage:
    """New FRED series IDs exist for all currencies we now support."""

    def test_gbp_series_present(self):
        for key in ("boe_rate", "uk_10y", "uk_cpi_yoy"):
            assert key in FRED_SERIES

    def test_aud_series_present(self):
        for key in ("rba_rate", "au_10y", "au_cpi_yoy"):
            assert key in FRED_SERIES

    def test_aud_china_linkage_series_present(self):
        """AU's #1 trading partner is China — iron ore/CNY/China CPI."""
        for key in ("iron_ore", "china_cpi_yoy", "cny_usd"):
            assert key in FRED_SERIES

    def test_nzd_series_present(self):
        for key in ("rbnz_rate", "nz_10y", "nz_cpi_yoy"):
            assert key in FRED_SERIES


# -------- Feature emission per symbol -------------------------------------

class TestLiveSymbolsUnchanged:
    """The 5 live pairs must still emit their existing feature sets."""

    def test_eurusd_emits_eur_features(self):
        feats = _make_fetcher().get_macro_features("EURUSD")
        for key in ("ecb_rate_level", "eu_cpi_yoy_zscore",
                     "eu_unemployment_level", "eur_usd_rate_diff"):
            assert key in feats, f"EURUSD missing {key}"

    def test_usdjpy_emits_jpy_features(self):
        feats = _make_fetcher().get_macro_features("USDJPY")
        for key in ("boj_rate_level", "japan_cpi_yoy_zscore",
                     "usd_jpy_rate_diff", "carry_trade_indicator"):
            assert key in feats, f"USDJPY missing {key}"

    def test_usdcad_emits_cad_features(self):
        feats = _make_fetcher().get_macro_features("USDCAD")
        for key in ("boc_rate_level", "canada_cpi_yoy_zscore",
                     "usd_cad_rate_diff", "wti_oil_zscore"):
            assert key in feats, f"USDCAD missing {key}"

    def test_xau_gets_only_common_features(self):
        feats = _make_fetcher().get_macro_features("XAUUSD")
        # Must not pick up any currency-block features
        for key in ("ecb_rate_level", "boj_rate_level", "boc_rate_level",
                     "boe_rate_level", "rba_rate_level", "rbnz_rate_level"):
            assert key not in feats, f"XAUUSD should not have {key}"

    def test_eth_gets_only_common_features(self):
        feats = _make_fetcher().get_macro_features("ETHUSD")
        for key in ("ecb_rate_level", "boj_rate_level", "boe_rate_level",
                     "rba_rate_level"):
            assert key not in feats, f"ETHUSD should not have {key}"


class TestNewPairFeatures:
    """The 6 new pairs must emit their currency-specific blocks + cross diffs."""

    def test_gbpusd_gets_gbp_block_and_usd_diff(self):
        feats = _make_fetcher().get_macro_features("GBPUSD")
        for key in ("boe_rate_level", "uk_10y_level", "uk_cpi_yoy_zscore",
                     "gbp_usd_rate_diff"):
            assert key in feats, f"GBPUSD missing {key}"
        # No EUR/JPY/AUD/NZD side-effects
        assert "ecb_rate_level" not in feats
        assert "boj_rate_level" not in feats

    def test_audusd_gets_aud_block_and_usd_diff(self):
        feats = _make_fetcher().get_macro_features("AUDUSD")
        for key in ("rba_rate_level", "au_10y_level", "au_cpi_yoy_zscore",
                     "aud_usd_rate_diff",
                     # China-demand channel
                     "iron_ore_zscore", "china_cpi_yoy_zscore", "cny_usd_zscore"):
            assert key in feats, f"AUDUSD missing {key}"
        assert "boe_rate_level" not in feats
        assert "rbnz_rate_level" not in feats

    def test_audnzd_gets_aud_china_channel_too(self):
        """AUDNZD exposes AUD, so it should also get the China-demand channel."""
        feats = _make_fetcher().get_macro_features("AUDNZD")
        for key in ("iron_ore_zscore", "china_cpi_yoy_zscore", "cny_usd_zscore"):
            assert key in feats, f"AUDNZD missing AU-China series {key}"

    def test_gbpusd_does_not_get_china_channel(self):
        """Non-AUD pairs must NOT pick up iron ore / China CPI / CNY."""
        feats = _make_fetcher().get_macro_features("GBPUSD")
        for key in ("iron_ore_zscore", "china_cpi_yoy_zscore", "cny_usd_zscore"):
            assert key not in feats, f"GBPUSD should not have {key}"

    def test_eurgbp_gets_both_blocks_and_cross_diff(self):
        feats = _make_fetcher().get_macro_features("EURGBP")
        # EUR block
        assert "ecb_rate_level" in feats
        assert "eu_cpi_yoy_zscore" in feats
        # GBP block
        assert "boe_rate_level" in feats
        assert "uk_cpi_yoy_zscore" in feats
        # Cross-pair diff
        assert "eur_gbp_rate_diff" in feats
        # No USD-side diff
        assert "eur_usd_rate_diff" not in feats
        assert "gbp_usd_rate_diff" not in feats

    def test_eurjpy_gets_both_blocks_and_carry(self):
        feats = _make_fetcher().get_macro_features("EURJPY")
        assert "ecb_rate_level" in feats
        assert "boj_rate_level" in feats
        assert "eur_jpy_rate_diff" in feats
        # carry_trade_indicator should be set to EUR-JPY diff for this pair
        assert "carry_trade_indicator" in feats
        # Not the USDJPY version
        assert "usd_jpy_rate_diff" not in feats

    def test_gbpjpy_gets_both_blocks_and_carry(self):
        feats = _make_fetcher().get_macro_features("GBPJPY")
        assert "boe_rate_level" in feats
        assert "boj_rate_level" in feats
        assert "gbp_jpy_rate_diff" in feats
        assert "carry_trade_indicator" in feats
        # No EUR contamination
        assert "ecb_rate_level" not in feats

    def test_audnzd_gets_both_blocks_and_cross_diff(self):
        feats = _make_fetcher().get_macro_features("AUDNZD")
        # AUD block
        assert "rba_rate_level" in feats
        assert "au_cpi_yoy_zscore" in feats
        # NZD block
        assert "rbnz_rate_level" in feats
        assert "nz_10y_level" in feats
        assert "nz_cpi_yoy_zscore" in feats
        # Cross-pair diff
        assert "aud_nzd_rate_diff" in feats
        # No USD or any other currency leakage
        assert "boe_rate_level" not in feats
        assert "ecb_rate_level" not in feats


class TestRateDiffsCorrectSign:
    """Cross-pair rate diffs should be first_currency - second_currency."""

    def test_eur_gbp_diff_is_ecb_minus_boe(self):
        # ECB rate = 3.0, BoE rate = 4.0 → diff = -1.0
        fetcher = _make_fetcher(fake_values={"ecb_rate": 3.0, "boe_rate": 4.0})
        feats = fetcher.get_macro_features("EURGBP")
        assert feats["eur_gbp_rate_diff"] == pytest.approx(3.0 - 4.0)

    def test_gbp_jpy_diff_is_boe_minus_boj(self):
        fetcher = _make_fetcher(fake_values={"boe_rate": 4.0, "boj_rate": 0.5})
        feats = fetcher.get_macro_features("GBPJPY")
        assert feats["gbp_jpy_rate_diff"] == pytest.approx(4.0 - 0.5)
        # Carry indicator equals the same diff
        assert feats["carry_trade_indicator"] == pytest.approx(4.0 - 0.5)

    def test_aud_nzd_diff_is_rba_minus_rbnz(self):
        fetcher = _make_fetcher(fake_values={"rba_rate": 4.5, "rbnz_rate": 3.25})
        feats = fetcher.get_macro_features("AUDNZD")
        assert feats["aud_nzd_rate_diff"] == pytest.approx(4.5 - 3.25)


# ===========================================================================
# Phase 2A — historical feature reader from feature_store
# ===========================================================================

import asyncio
from unittest.mock import AsyncMock

import numpy as np


def _build_raw_macro_df(
    *,
    n_days: int = 800,
    end: datetime | None = None,
    overrides: dict[str, list[float] | float] | None = None,
) -> pd.DataFrame:
    """Build a wide raw-observation DataFrame as the bare reader would emit.

    Default values match the live ``_default_features()`` so a smoke test of
    historical engineering with no overrides emits the same defaults as live.
    """
    if end is None:
        end = datetime(2026, 4, 1)
    idx = pd.date_range(end=end, periods=n_days, freq="D")
    cols: dict[str, list[float]] = {
        # Keep the values realistic but constant so rolling z-score = 0,
        # and any mismatch in column-naming surfaces immediately.
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
    }
    if overrides:
        for k, v in overrides.items():
            cols[k] = list(v) if hasattr(v, "__len__") else [float(v)] * n_days
    return pd.DataFrame(cols, index=idx)


def _build_store_returning(raw_df: pd.DataFrame) -> AsyncMock:
    """Mock DataStore whose read_feature_store always returns ``raw_df``."""
    store = AsyncMock()
    async def _read(symbol, feature_group, start=None, end=None):
        return raw_df.copy()
    store.read_feature_store = _read
    return store


def _historical_call(
    fetcher: MacroDataFetcher,
    symbol: str,
    raw_df: pd.DataFrame,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    lag_hours: float = 0.0,
) -> pd.DataFrame:
    """Run get_historical_macro_features synchronously for a test."""
    store = _build_store_returning(raw_df)
    cfg = {"sources": {"fred_macro": {"release_lag_hours": lag_hours}}}
    if start is None:
        start = raw_df.index.min().to_pydatetime()
    if end is None:
        end = raw_df.index.max().to_pydatetime()
    return asyncio.run(fetcher.get_historical_macro_features(
        store, symbol, start, end, feeds_config=cfg,
    ))


class TestHistoricalSchemaMatchesLive:
    """The KEY invariant: historical column set == live dict keys per symbol."""

    @pytest.mark.parametrize("symbol", [
        "XAUUSD",                                           # live, common-only
        "EURUSD", "USDJPY", "USDCAD",                       # live forex
        "GBPUSD", "AUDUSD", "EURGBP", "EURJPY",             # expansion
        "GBPJPY", "AUDNZD",                                 # expansion crosses
    ])
    def test_historical_columns_match_live_keys(self, symbol):
        fetcher = _make_fetcher()
        # Live keys
        live_keys = set(fetcher.get_macro_features(symbol).keys())
        # Historical columns
        raw = _build_raw_macro_df()
        hist_df = _historical_call(fetcher, symbol, raw)
        hist_cols = set(hist_df.columns)
        assert live_keys == hist_cols, (
            f"{symbol} schema drift: live-only={live_keys - hist_cols}, "
            f"hist-only={hist_cols - live_keys}"
        )

    def test_eth_gets_only_common_columns(self):
        """ETHUSD has no FX block — historical should match."""
        fetcher = _make_fetcher()
        raw = _build_raw_macro_df()
        hist_df = _historical_call(fetcher, "ETHUSD", raw)
        # No currency-block columns
        for col in ("ecb_rate_level", "boj_rate_level", "boc_rate_level",
                    "boe_rate_level", "rba_rate_level", "rbnz_rate_level"):
            assert col not in hist_df.columns


class TestHistoricalReturnsEmptyOnMissingData:
    def test_empty_raw_df_returns_empty(self):
        fetcher = _make_fetcher()
        empty_raw = pd.DataFrame()
        hist_df = _historical_call(
            fetcher, "EURUSD", empty_raw,
            start=datetime(2024, 1, 1), end=datetime(2024, 12, 31),
        )
        assert hist_df.empty


class TestHistoricalReleaseLag:
    """Release lag must shift each raw observation forward in time."""

    def test_step_change_appears_lag_days_later(self):
        """A jump at raw-day D should show up at engineered-day D+lag, not D.

        This is THE lookahead-safety invariant: the LSTM at time t may only
        see data that would have been publicly published by t.
        """
        fetcher = _make_fetcher()
        n = 400
        # Step change halfway: ecb_rate jumps from 3.0 to 5.0 at day 200
        change_day_idx = 200
        ecb_series = [3.0] * change_day_idx + [5.0] * (n - change_day_idx)
        raw = _build_raw_macro_df(
            n_days=n,
            end=datetime(2024, 12, 31),
            overrides={"ecb_rate": ecb_series},
        )
        change_date = raw.index[change_day_idx]

        # 504h = 21 days
        hist_df = _historical_call(
            fetcher, "EURUSD", raw,
            start=raw.index[0].to_pydatetime(),
            end=raw.index[-1].to_pydatetime(),
            lag_hours=504.0,
        )
        # Engineered ecb_rate_level at change_date should still be 3.0
        # (the new 5.0 observation hasn't been published yet at this point)
        if change_date in hist_df.index:
            assert hist_df.loc[change_date, "ecb_rate_level"] == pytest.approx(3.0)
        # 21 days later, the published value is now 5.0
        post_lag_date = change_date + pd.Timedelta(days=21)
        if post_lag_date in hist_df.index:
            assert hist_df.loc[post_lag_date, "ecb_rate_level"] == pytest.approx(5.0)

    def test_zero_lag_passes_through_unchanged(self):
        """With release_lag=0, engineered values match raw observations directly."""
        fetcher = _make_fetcher()
        raw = _build_raw_macro_df(overrides={"ecb_rate": 3.5})
        hist_df = _historical_call(fetcher, "EURUSD", raw, lag_hours=0.0)
        # ecb_rate_level should equal 3.5 throughout (same day as raw)
        assert (hist_df["ecb_rate_level"] == 3.5).all()


class TestHistoricalEngineeringCorrectness:
    """Engineered columns should compute the right values from raw."""

    def test_constant_input_yields_zero_zscore(self):
        """A constant series has zero variance → z-score = 0."""
        fetcher = _make_fetcher()
        raw = _build_raw_macro_df()  # all values constant
        hist_df = _historical_call(fetcher, "EURUSD", raw)
        # After warmup (>= 60 obs), z-score should be 0
        warmup_idx = hist_df.index[100:]
        assert (hist_df.loc[warmup_idx, "cpi_yoy_zscore"].abs() < 1e-9).all()
        assert (hist_df.loc[warmup_idx, "dxy_zscore"].abs() < 1e-9).all()

    def test_step_change_produces_nonzero_zscore_within_window(self):
        """A jump-up should produce a positive z-score while inside the
        60-obs rolling window. Far past the window, std collapses back to
        zero (constant high values) and z returns to 0 — that's correct
        rolling-window behavior, not a bug."""
        fetcher = _make_fetcher()
        n = 800
        change_idx = n // 2
        cpi_series = [2.0] * change_idx + [5.0] * (n - change_idx)
        raw = _build_raw_macro_df(n_days=n, overrides={"cpi_yoy": cpi_series})
        hist_df = _historical_call(fetcher, "EURUSD", raw)
        # 10 bars after the step: window is mostly 2.0s with a few 5.0s →
        # the current 5.0 value sits well above the rolling mean.
        probe_date = raw.index[change_idx + 10]
        z_at_probe = hist_df.loc[probe_date, "cpi_yoy_zscore"]
        assert z_at_probe > 0, f"expected positive z near step-up, got {z_at_probe}"
        # Clipped to [-1, 1]
        assert -1.0 <= z_at_probe <= 1.0

    def test_zscore_clipped_to_unit_interval(self):
        """Even an extreme step should clip at +1.0 / -1.0."""
        fetcher = _make_fetcher()
        n = 800
        cpi_series = [2.0] * (n // 2) + [200.0] * (n // 2)
        raw = _build_raw_macro_df(n_days=n, overrides={"cpi_yoy": cpi_series})
        hist_df = _historical_call(fetcher, "EURUSD", raw)
        assert hist_df["cpi_yoy_zscore"].max() <= 1.0
        assert hist_df["cpi_yoy_zscore"].min() >= -1.0

    def test_rate_diffs_computed_from_levels(self):
        """eur_usd_rate_diff at any bar = fed_funds_level - ecb_rate_level."""
        fetcher = _make_fetcher()
        raw = _build_raw_macro_df(overrides={"fed_funds": 5.0, "ecb_rate": 3.5})
        hist_df = _historical_call(fetcher, "EURUSD", raw)
        # Pick a post-warmup row
        row = hist_df.iloc[-1]
        assert row["eur_usd_rate_diff"] == pytest.approx(
            row["fed_funds_level"] - row["ecb_rate_level"]
        )

    def test_carry_trade_indicator_matches_rate_diff(self):
        """For USDJPY, carry_trade_indicator == usd_jpy_rate_diff."""
        fetcher = _make_fetcher()
        raw = _build_raw_macro_df(overrides={"fed_funds": 5.0, "boj_rate": 0.25})
        hist_df = _historical_call(fetcher, "USDJPY", raw)
        row = hist_df.iloc[-1]
        assert row["carry_trade_indicator"] == pytest.approx(row["usd_jpy_rate_diff"])

    def test_missing_column_falls_back_to_default(self):
        """Live default for fed_funds_level is 5.0 — missing column → emit 5.0."""
        fetcher = _make_fetcher()
        raw = _build_raw_macro_df()
        del raw["fed_funds"]  # simulate missing series
        hist_df = _historical_call(fetcher, "EURUSD", raw)
        assert (hist_df["fed_funds_level"] == 5.0).all()


class TestHistoricalRefusesUnknownSource:
    """Lookahead-safety contract: refuse to query without release_lag."""

    def test_missing_source_in_config_raises(self):
        fetcher = _make_fetcher()
        store = AsyncMock()
        cfg = {"sources": {}}  # no fred_macro entry
        with pytest.raises(ValueError, match="not in data_feeds.yaml"):
            asyncio.run(fetcher.get_historical_macro_features(
                store, "EURUSD",
                datetime(2024, 1, 1), datetime(2024, 12, 31),
                feeds_config=cfg,
            ))
