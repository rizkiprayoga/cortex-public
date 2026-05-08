"""Tests for stooq_data — HTTP-mocked, never hits the live API."""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.data_pipeline.market.stooq_data import (
    STOOQ_SERIES,
    StooqFetcher,
    _AUD_EXPOSURE,
    _EUR_EXPOSURE,
    _GBP_EXPOSURE,
    _JPY_EXPOSURE,
    _NZD_EXPOSURE,
)


# ----- Fixtures -----------------------------------------------------------

_GOOD_CSV = """Date,Open,High,Low,Close,Volume
2026-04-20,4.10,4.15,4.08,4.12,1000
2026-04-21,4.12,4.20,4.10,4.18,1200
2026-04-22,4.18,4.22,4.15,4.20,1500
2026-04-23,4.20,4.25,4.18,4.22,1100
2026-04-24,4.22,4.28,4.20,4.25,1300
"""

_APIKEY_PROMPT_BODY = """
Get your apikey:
1. Open https://stooq.com/q/d/?s=10yuky.b&get_apikey
2. Enter the captcha code.
"""


def _mock_response(text: str, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    return resp


# ----- Catalog -----------------------------------------------------------

class TestSeriesCatalog:
    """Plan line 76 requires UK/AU/NZ/JP 10Y — breadth adds DE/US + 2Y tenors."""

    def test_plan_countries_present(self):
        countries = {c for (_, _, c) in STOOQ_SERIES.values()}
        for required in ("UK", "AU", "NZ", "JP"):
            assert required in countries, f"Missing {required}"

    def test_2y_and_10y_tenors(self):
        """Both tenors for every country → per-country slope."""
        tenors_by_country: dict[str, set] = {}
        for (_, tenor, country) in STOOQ_SERIES.values():
            tenors_by_country.setdefault(country, set()).add(tenor)
        for country, tenors in tenors_by_country.items():
            assert 2 in tenors, f"{country} missing 2Y"
            assert 10 in tenors, f"{country} missing 10Y"

    def test_stooq_symbol_format(self):
        """All symbols follow the {tenor}{country}Y.B convention."""
        for (sym, tenor, country) in STOOQ_SERIES.values():
            assert sym.endswith("Y.B"), f"Bad suffix: {sym}"
            assert str(tenor) in sym
            assert country in sym


# ----- Fetch path --------------------------------------------------------

class TestGetSeriesHttpMocked:
    """get_series uses requests.get — mock it to verify parse/cache/errors."""

    def test_parses_csv_response(self):
        fetcher = StooqFetcher()
        fetcher.api_key = "dummy"  # skip the "no key" warning branch
        with patch("src.data_pipeline.market.stooq_data.requests.get",
                    return_value=_mock_response(_GOOD_CSV)):
            df = fetcher.get_series("uk_10y")
        assert not df.empty
        assert "Close" in df.columns
        assert len(df) == 5
        assert df["Close"].iloc[-1] == pytest.approx(4.25)

    def test_apikey_prompt_returns_empty(self):
        """When Stooq returns the apikey landing page, treat as empty."""
        fetcher = StooqFetcher()
        fetcher.api_key = ""
        with patch("src.data_pipeline.market.stooq_data.requests.get",
                    return_value=_mock_response(_APIKEY_PROMPT_BODY)):
            df = fetcher.get_series("uk_10y")
        assert df.empty

    def test_http_error_returns_empty(self):
        fetcher = StooqFetcher()
        fetcher.api_key = "dummy"
        import requests as req
        with patch("src.data_pipeline.market.stooq_data.requests.get",
                    side_effect=req.ConnectionError("network down")):
            df = fetcher.get_series("au_10y")
        assert df.empty  # must not raise

    def test_unknown_label_returns_empty(self):
        fetcher = StooqFetcher()
        df = fetcher.get_series("nonexistent_label")
        assert df.empty

    def test_cache_reuses_result(self):
        """Second call within TTL must not re-hit HTTP."""
        fetcher = StooqFetcher(cache_ttl_hours=24)
        fetcher.api_key = "dummy"
        with patch("src.data_pipeline.market.stooq_data.requests.get",
                    return_value=_mock_response(_GOOD_CSV)) as mock_get:
            fetcher.get_series("uk_10y")
            fetcher.get_series("uk_10y")
            fetcher.get_series("uk_10y")
        assert mock_get.call_count == 1


# ----- Per-symbol routing ------------------------------------------------

class TestYieldFeatureRouting:
    """get_yield_features emits the correct country blocks per symbol."""

    def _patched_fetcher(self, close_value: float = 4.25) -> StooqFetcher:
        """Fetcher whose get_series always returns the same sample DF."""
        sample = pd.read_csv(
            pd.io.common.StringIO(_GOOD_CSV.replace("4.25", str(close_value)))
        )
        sample["Date"] = pd.to_datetime(sample["Date"])
        sample = sample.set_index("Date")

        fetcher = StooqFetcher()
        fetcher.get_series = lambda label: sample.copy()  # type: ignore[assignment]
        return fetcher

    def test_usdjpy_gets_us_and_jp(self):
        feats = self._patched_fetcher().get_yield_features("USDJPY")
        # US baseline always present
        assert "us_10y_daily" in feats
        assert "us_slope_daily" in feats
        # JP block
        assert "jp_10y_daily" in feats
        assert "jp_slope_daily" in feats
        # No other country leakage
        assert "uk_10y_daily" not in feats
        assert "au_10y_daily" not in feats

    def test_gbpusd_gets_uk_block(self):
        feats = self._patched_fetcher().get_yield_features("GBPUSD")
        for key in ("uk_2y_daily", "uk_10y_daily", "uk_slope_daily"):
            assert key in feats

    def test_audusd_gets_au_block(self):
        feats = self._patched_fetcher().get_yield_features("AUDUSD")
        for key in ("au_2y_daily", "au_10y_daily", "au_slope_daily"):
            assert key in feats
        assert "nz_10y_daily" not in feats  # no NZD exposure

    def test_audnzd_gets_au_plus_nz(self):
        feats = self._patched_fetcher().get_yield_features("AUDNZD")
        for country in ("au", "nz"):
            for tenor in ("2y", "10y", "slope"):
                assert f"{country}_{tenor}_daily" in feats
        # No USD-side country blocks beyond US baseline
        assert "uk_10y_daily" not in feats

    def test_eurgbp_gets_de_uk_no_jp(self):
        feats = self._patched_fetcher().get_yield_features("EURGBP")
        assert "de_10y_daily" in feats
        assert "uk_10y_daily" in feats
        assert "jp_10y_daily" not in feats

    def test_xau_gets_only_us_baseline(self):
        """XAU has no FX-currency routing — just US baseline."""
        feats = self._patched_fetcher().get_yield_features("XAUUSD")
        assert "us_10y_daily" in feats
        for key in ("uk_10y_daily", "jp_10y_daily", "au_10y_daily",
                     "de_10y_daily", "nz_10y_daily"):
            assert key not in feats

    def test_slope_is_10y_minus_2y(self):
        """slope_daily should equal 10Y - 2Y (both set to 4.25 in fixture)."""
        feats = self._patched_fetcher(4.25).get_yield_features("GBPUSD")
        # Both 2Y and 10Y read the same sample (close=4.25) — slope is 0.0
        assert feats["uk_slope_daily"] == pytest.approx(0.0, abs=1e-6)


# ----- Defaults -----------------------------------------------------------

class TestDefaults:
    """default_yield_features mirrors get_yield_features routing."""

    def test_usdjpy_defaults(self):
        d = StooqFetcher.default_yield_features("USDJPY")
        for key in ("us_2y_daily", "us_10y_daily", "us_slope_daily",
                     "jp_2y_daily", "jp_10y_daily", "jp_slope_daily"):
            assert key in d
            assert d[key] == 0.0
        assert "uk_10y_daily" not in d

    def test_audnzd_defaults(self):
        d = StooqFetcher.default_yield_features("AUDNZD")
        for country in ("us", "au", "nz"):
            assert f"{country}_10y_daily" in d
        assert "uk_10y_daily" not in d


# ===========================================================================
# Phase 2A — historical feature reader from feature_store
# ===========================================================================

import asyncio
from unittest.mock import AsyncMock


def _build_raw_yield_df(
    *,
    n_days: int = 200,
    end: datetime | None = None,
    overrides: dict[str, list[float] | float] | None = None,
    countries: tuple[str, ...] = ("us", "uk", "de", "jp", "au", "nz"),
    include_slope: bool = True,
) -> pd.DataFrame:
    """Build a wide raw-observation DataFrame as the bare reader would emit."""
    if end is None:
        end = datetime(2026, 4, 1)
    idx = pd.date_range(end=end, periods=n_days, freq="D")
    cols: dict[str, list[float]] = {}
    for c in countries:
        cols[f"{c}_2y_daily"] = [4.0] * n_days
        cols[f"{c}_10y_daily"] = [4.5] * n_days
        if include_slope:
            cols[f"{c}_slope_daily"] = [0.5] * n_days
    if overrides:
        for k, v in overrides.items():
            cols[k] = list(v) if hasattr(v, "__len__") else [float(v)] * n_days
    return pd.DataFrame(cols, index=idx)


def _stooq_store(raw_df: pd.DataFrame) -> AsyncMock:
    store = AsyncMock()
    async def _read(symbol, feature_group, start=None, end=None):
        return raw_df.copy()
    store.read_feature_store = _read
    return store


def _stooq_historical(
    symbol: str,
    raw_df: pd.DataFrame,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    lag_hours: float = 0.0,
) -> pd.DataFrame:
    fetcher = StooqFetcher()
    store = _stooq_store(raw_df)
    cfg = {"sources": {"stooq_yields": {"release_lag_hours": lag_hours}}}
    if start is None:
        start = raw_df.index.min().to_pydatetime()
    if end is None:
        end = raw_df.index.max().to_pydatetime()
    return asyncio.run(fetcher.get_historical_yield_features(
        store, symbol, start, end, feeds_config=cfg,
    ))


class TestHistoricalYieldSchemaMatchesLive:
    """Historical column set == live default-features keys per symbol."""

    @pytest.mark.parametrize("symbol", [
        "XAUUSD",
        "EURUSD", "USDJPY", "USDCAD",
        "GBPUSD", "AUDUSD", "EURGBP", "EURJPY", "GBPJPY", "AUDNZD",
    ])
    def test_historical_columns_match_default_keys(self, symbol):
        # Live get_yield_features hits the network; default_yield_features
        # is the operative reference (same routing, same column set).
        live_keys = set(StooqFetcher.default_yield_features(symbol).keys())
        raw = _build_raw_yield_df()
        hist_df = _stooq_historical(symbol, raw)
        hist_cols = set(hist_df.columns)
        assert live_keys == hist_cols, (
            f"{symbol} schema drift: live-only={live_keys - hist_cols}, "
            f"hist-only={hist_cols - live_keys}"
        )


class TestHistoricalYieldEmpty:
    def test_empty_raw_returns_empty(self):
        empty = pd.DataFrame()
        df = _stooq_historical(
            "EURUSD", empty,
            start=datetime(2024, 1, 1), end=datetime(2024, 12, 31),
        )
        assert df.empty


class TestHistoricalYieldEngineering:
    """Slope reconstruction + leg defaulting match live behavior."""

    def test_slope_recomputed_when_missing_from_persisted_row(self):
        """A row missing slope but with both legs should reconstruct it."""
        raw = _build_raw_yield_df(include_slope=False)
        df = _stooq_historical("USDJPY", raw)
        # us_slope = us_10y - us_2y = 4.5 - 4.0 = 0.5
        assert (df["us_slope_daily"] - 0.5).abs().max() < 1e-9
        # jp_slope = same
        assert (df["jp_slope_daily"] - 0.5).abs().max() < 1e-9

    def test_slope_preserved_when_persisted(self):
        raw = _build_raw_yield_df(
            overrides={"us_slope_daily": 0.75},  # override persisted slope
            include_slope=True,
        )
        df = _stooq_historical("USDJPY", raw)
        # Should use the persisted 0.75, not recompute from legs
        assert (df["us_slope_daily"] - 0.75).abs().max() < 1e-9

    def test_slope_zero_when_either_leg_missing(self):
        """Live ``_emit_country_block`` defaults slope=0 when a leg is missing.
        After ffill across NaNs this becomes leg=0.0 → slope must be 0."""
        # Build a frame where us_2y is all NaN
        raw = _build_raw_yield_df(include_slope=False)
        raw["us_2y_daily"] = float("nan")
        df = _stooq_historical("EURUSD", raw)
        # us_2y ffilled then fillna(0) = 0; slope guard returns 0 when leg=0
        assert (df["us_2y_daily"] == 0.0).all()
        assert (df["us_slope_daily"] == 0.0).all()

    def test_xau_gets_only_us_block(self):
        raw = _build_raw_yield_df()
        df = _stooq_historical("XAUUSD", raw)
        for col in ("us_2y_daily", "us_10y_daily", "us_slope_daily"):
            assert col in df.columns
        # No country-conditional blocks
        for col in ("uk_10y_daily", "de_10y_daily", "jp_10y_daily",
                    "au_10y_daily", "nz_10y_daily"):
            assert col not in df.columns

    def test_audnzd_gets_us_au_nz_blocks(self):
        raw = _build_raw_yield_df()
        df = _stooq_historical("AUDNZD", raw)
        for c in ("us", "au", "nz"):
            for tenor in ("2y", "10y", "slope"):
                assert f"{c}_{tenor}_daily" in df.columns
        # No EUR/GBP/JPY leakage
        for c in ("uk", "de", "jp"):
            assert f"{c}_10y_daily" not in df.columns


class TestHistoricalYieldReleaseLag:
    def test_step_change_appears_lag_days_later(self):
        n = 200
        change_idx = 100
        # us_10y jumps from 4.5 to 5.5 mid-window
        us_10y_series = [4.5] * change_idx + [5.5] * (n - change_idx)
        raw = _build_raw_yield_df(
            n_days=n,
            end=datetime(2024, 12, 31),
            overrides={"us_10y_daily": us_10y_series},
            include_slope=False,
        )
        change_date = raw.index[change_idx]
        # 24h lag (typical for stooq_yields per data_feeds.yaml)
        df = _stooq_historical(
            "EURUSD", raw,
            start=raw.index[0].to_pydatetime(),
            end=raw.index[-1].to_pydatetime(),
            lag_hours=24.0,
        )
        # On change_date itself, the 4.5 value is still showing
        if change_date in df.index:
            assert df.loc[change_date, "us_10y_daily"] == pytest.approx(4.5)
        # 1 day later, the 5.5 value has now landed
        post_lag_date = change_date + pd.Timedelta(days=1)
        if post_lag_date in df.index:
            assert df.loc[post_lag_date, "us_10y_daily"] == pytest.approx(5.5)


class TestHistoricalYieldRefusesUnknownSource:
    def test_missing_source_in_config_raises(self):
        fetcher = StooqFetcher()
        store = AsyncMock()
        cfg = {"sources": {}}
        with pytest.raises(ValueError, match="not in data_feeds.yaml"):
            asyncio.run(fetcher.get_historical_yield_features(
                store, "EURUSD",
                datetime(2024, 1, 1), datetime(2024, 12, 31),
                feeds_config=cfg,
            ))
