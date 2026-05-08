"""Tests for ecb_data — HTTP-mocked, never hits the live ECB endpoint."""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.data_pipeline.market.ecb_data import (
    ECB_SERIES,
    ECBDataFetcher,
    _EUR_EXPOSURE,
)


# Snippet of the actual ECB CSV format. The real payload has ~40 columns;
# only TIME_PERIOD + OBS_VALUE are required for parsing.
_GOOD_CSV = """KEY,FREQ,REF_AREA,CURRENCY,PROVIDER_FM,INSTRUMENT_FM,PROVIDER_FM_ID,DATA_TYPE_FM,TIME_PERIOD,OBS_VALUE,OBS_STATUS
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,B,U2,EUR,4F,G_N_A,SV_C_YM,SR_10Y,2026-04-20,3.0100,A
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,B,U2,EUR,4F,G_N_A,SV_C_YM,SR_10Y,2026-04-21,3.0250,A
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,B,U2,EUR,4F,G_N_A,SV_C_YM,SR_10Y,2026-04-22,3.0400,A
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,B,U2,EUR,4F,G_N_A,SV_C_YM,SR_10Y,2026-04-23,3.0675,A
"""


def _mock_response(text: str, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    return resp


class TestSeriesCatalog:
    """Plan line 75 requires the full tenor range (3M-30Y)."""

    def test_all_tenors_present(self):
        """3M through 30Y, 10 tenors total."""
        expected = {"3m", "6m", "1y", "2y", "3y", "5y", "7y", "10y", "20y", "30y"}
        assert set(ECB_SERIES.keys()) == expected

    def test_series_key_format(self):
        """Key omits 'YC.' prefix — flow goes in URL path, not key segment."""
        for label, key in ECB_SERIES.items():
            assert key.startswith("B.U2.EUR.4F.G_N_A.SV_C_YM."), \
                f"Bad prefix for {label}: {key}"
            assert not key.startswith("YC."), \
                f"{label} has YC. prefix — REST API returns 400 with it"


class TestGetSeriesHttpMocked:
    """Parsing, error handling, caching — all with patched requests.get."""

    def test_parses_csv_response(self):
        fetcher = ECBDataFetcher()
        with patch("src.data_pipeline.market.ecb_data.requests.get",
                    return_value=_mock_response(_GOOD_CSV)):
            df = fetcher.get_series("10y")
        assert not df.empty
        assert "Close" in df.columns
        assert len(df) == 4
        assert df["Close"].iloc[-1] == pytest.approx(3.0675)

    def test_drops_missing_obs_values(self):
        """NaN OBS_VALUE rows must be dropped (ECB sometimes emits blanks)."""
        csv_with_nan = _GOOD_CSV + (
            "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,B,U2,EUR,4F,G_N_A,SV_C_YM,SR_10Y,2026-04-24,,A\n"
        )
        fetcher = ECBDataFetcher()
        with patch("src.data_pipeline.market.ecb_data.requests.get",
                    return_value=_mock_response(csv_with_nan)):
            df = fetcher.get_series("10y")
        # 4 valid rows + 1 NaN row → 4 after dropna
        assert len(df) == 4
        assert df["Close"].iloc[-1] == pytest.approx(3.0675)

    def test_http_error_returns_empty(self):
        fetcher = ECBDataFetcher()
        import requests as req
        with patch("src.data_pipeline.market.ecb_data.requests.get",
                    side_effect=req.ConnectionError("network down")):
            df = fetcher.get_series("10y")
        assert df.empty

    def test_400_returns_empty(self):
        """Bad URL gets 400 — must not raise."""
        fetcher = ECBDataFetcher()
        bad_resp = _mock_response("error body", status=400)
        import requests as req
        bad_resp.raise_for_status.side_effect = req.HTTPError("400")
        with patch("src.data_pipeline.market.ecb_data.requests.get",
                    return_value=bad_resp):
            df = fetcher.get_series("10y")
        assert df.empty

    def test_unexpected_schema_returns_empty(self):
        fetcher = ECBDataFetcher()
        with patch("src.data_pipeline.market.ecb_data.requests.get",
                    return_value=_mock_response("col1,col2\nfoo,bar\n")):
            df = fetcher.get_series("10y")
        assert df.empty

    def test_unknown_tenor_returns_empty(self):
        fetcher = ECBDataFetcher()
        df = fetcher.get_series("42y")   # not in _TENORS
        assert df.empty

    def test_cache_reuses_result(self):
        """Second fetch within TTL hits cache, not HTTP."""
        fetcher = ECBDataFetcher(cache_ttl_hours=24)
        with patch("src.data_pipeline.market.ecb_data.requests.get",
                    return_value=_mock_response(_GOOD_CSV)) as mock_get:
            fetcher.get_series("10y")
            fetcher.get_series("10y")
            fetcher.get_series("10y")
        assert mock_get.call_count == 1


class TestPerSymbolRouting:
    """get_yield_curve_features only emits for EUR-exposed symbols."""

    def _patched_fetcher(self):
        """Fetcher whose get_series returns the same sample DF for any label."""
        sample = pd.DataFrame(
            {"Close": [2.0, 2.1, 2.2, 2.5]},
            index=pd.to_datetime(["2026-04-20", "2026-04-21",
                                    "2026-04-22", "2026-04-23"]),
        )
        sample.index.name = "Date"
        fetcher = ECBDataFetcher()
        fetcher.get_series = lambda label: sample.copy()  # type: ignore[assignment]
        return fetcher

    def test_eur_pairs_get_full_curve(self):
        fetcher = self._patched_fetcher()
        for sym in ("EURUSD", "EURGBP", "EURJPY"):
            feats = fetcher.get_yield_curve_features(sym)
            # 10 tenors + 2 slope features
            assert len(feats) == 12, f"{sym} feature count wrong"
            for tenor in ("3m", "6m", "1y", "2y", "3y", "5y", "7y",
                           "10y", "20y", "30y"):
                assert f"eu_aaa_{tenor}_daily" in feats
            assert "eu_aaa_slope_2y10y" in feats
            assert "eu_aaa_slope_3m10y" in feats

    def test_non_eur_pairs_get_nothing(self):
        """XAU, ETH, USD-only pairs (USDJPY/USDCAD), and other crosses."""
        fetcher = self._patched_fetcher()
        for sym in ("XAUUSD", "ETHUSD", "USDJPY", "USDCAD",
                     "GBPUSD", "AUDUSD", "GBPJPY", "AUDNZD"):
            feats = fetcher.get_yield_curve_features(sym)
            assert feats == {}, f"{sym} should get no ECB features"

    def test_slopes_arithmetic(self):
        """slope_2y10y = 10Y - 2Y; slope_3m10y = 10Y - 3M. Fixture = 2.5 for all."""
        feats = self._patched_fetcher().get_yield_curve_features("EURUSD")
        # All tenors = 2.5 → both slopes = 0
        assert feats["eu_aaa_slope_2y10y"] == pytest.approx(0.0, abs=1e-6)
        assert feats["eu_aaa_slope_3m10y"] == pytest.approx(0.0, abs=1e-6)


class TestDefaults:
    """default_yield_curve_features mirrors get_yield_curve_features."""

    def test_eur_pairs_default_all_zeros(self):
        d = ECBDataFetcher.default_yield_curve_features("EURUSD")
        assert len(d) == 12
        for v in d.values():
            assert v == 0.0

    def test_non_eur_pairs_default_empty(self):
        for sym in ("XAUUSD", "USDJPY", "GBPUSD", "AUDNZD"):
            d = ECBDataFetcher.default_yield_curve_features(sym)
            assert d == {}


class TestExposureSet:
    """_EUR_EXPOSURE matches the three EUR pairs."""

    def test_eur_exposure_contents(self):
        for sym in ("EURUSD", "EURGBP", "EURJPY"):
            assert sym in _EUR_EXPOSURE

    def test_slashed_forms_also_match(self):
        for sym in ("EUR/USD", "EUR/GBP", "EUR/JPY"):
            assert sym in _EUR_EXPOSURE


# ===========================================================================
# Phase 2A — historical feature reader from feature_store
# ===========================================================================

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock


def _build_raw_curve_df(
    *,
    n_days: int = 200,
    end: datetime | None = None,
    overrides: dict[str, list[float] | float] | None = None,
    include_slopes: bool = True,
) -> pd.DataFrame:
    """Build a wide raw-observation DataFrame for the ECB curve."""
    if end is None:
        end = datetime(2026, 4, 1)
    idx = pd.date_range(end=end, periods=n_days, freq="D")
    cols: dict[str, list[float]] = {}
    base = {
        "3m": 3.50, "6m": 3.45, "1y": 3.40, "2y": 3.30, "3y": 3.25,
        "5y": 3.10, "7y": 3.05, "10y": 3.00, "20y": 3.20, "30y": 3.30,
    }
    for label, lvl in base.items():
        cols[f"eu_aaa_{label}_daily"] = [lvl] * n_days
    if include_slopes:
        cols["eu_aaa_slope_2y10y"] = [base["10y"] - base["2y"]] * n_days
        cols["eu_aaa_slope_3m10y"] = [base["10y"] - base["3m"]] * n_days
    if overrides:
        for k, v in overrides.items():
            cols[k] = list(v) if hasattr(v, "__len__") else [float(v)] * n_days
    return pd.DataFrame(cols, index=idx)


def _ecb_store(raw_df: pd.DataFrame) -> AsyncMock:
    store = AsyncMock()
    captured: dict = {}

    async def _read(symbol, feature_group, start=None, end=None):
        captured["symbol"] = symbol
        return raw_df.copy()

    store.read_feature_store = _read
    store.captured = captured
    return store


def _ecb_historical(
    symbol: str,
    raw_df: pd.DataFrame,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    lag_hours: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    fetcher = ECBDataFetcher()
    store = _ecb_store(raw_df)
    cfg = {"sources": {"ecb_yield_curve": {"release_lag_hours": lag_hours}}}
    if start is None:
        start = raw_df.index.min().to_pydatetime() if not raw_df.empty else datetime(2024, 1, 1)
    if end is None:
        end = raw_df.index.max().to_pydatetime() if not raw_df.empty else datetime(2024, 12, 31)
    df = asyncio.run(fetcher.get_historical_curve_features(
        store, symbol, start, end, feeds_config=cfg,
    ))
    return df, store.captured


class TestHistoricalCurveSchemaMatchesLive:
    """Historical column set == default_yield_curve_features keys per symbol."""

    @pytest.mark.parametrize("symbol", ["EURUSD", "EURGBP", "EURJPY"])
    def test_eur_pairs_columns_match_default_keys(self, symbol):
        live_keys = set(ECBDataFetcher.default_yield_curve_features(symbol).keys())
        raw = _build_raw_curve_df()
        df, _ = _ecb_historical(symbol, raw)
        assert set(df.columns) == live_keys

    @pytest.mark.parametrize("symbol", [
        "XAUUSD", "USDJPY", "USDCAD", "GBPUSD",
        "AUDUSD", "AUDNZD", "GBPJPY",
    ])
    def test_non_eur_pairs_get_empty_dataframe(self, symbol):
        """Live get_yield_curve_features returns {} for non-EUR — historical
        must return an empty DataFrame for parity."""
        raw = _build_raw_curve_df()
        df, _ = _ecb_historical(symbol, raw)
        assert df.empty


class TestHistoricalCurveReadsGlobalSymbol:
    """ECB curve is symbol-independent — reads must use _GLOBAL."""

    def test_global_symbol_used_regardless_of_caller(self):
        raw = _build_raw_curve_df()
        # Caller asks for EURUSD; backend must query _GLOBAL.
        _, captured = _ecb_historical("EURUSD", raw)
        assert captured["symbol"] == "_GLOBAL"

    def test_global_symbol_used_for_eurgbp_too(self):
        raw = _build_raw_curve_df()
        _, captured = _ecb_historical("EURGBP", raw)
        assert captured["symbol"] == "_GLOBAL"


class TestHistoricalCurveEmpty:
    def test_empty_raw_returns_empty(self):
        df, _ = _ecb_historical(
            "EURUSD", pd.DataFrame(),
            start=datetime(2024, 1, 1), end=datetime(2024, 12, 31),
        )
        assert df.empty


class TestHistoricalCurveEngineering:
    """Slope reconstruction + tenor defaulting match live behavior."""

    def test_slope_reconstructed_when_missing(self):
        raw = _build_raw_curve_df(include_slopes=False)
        df, _ = _ecb_historical("EURUSD", raw)
        # 2y10y slope = 10y - 2y = 3.00 - 3.30 = -0.30
        assert (df["eu_aaa_slope_2y10y"] - (3.00 - 3.30)).abs().max() < 1e-9
        # 3m10y slope = 10y - 3m = 3.00 - 3.50 = -0.50
        assert (df["eu_aaa_slope_3m10y"] - (3.00 - 3.50)).abs().max() < 1e-9

    def test_slope_preserved_when_persisted(self):
        raw = _build_raw_curve_df(
            include_slopes=True,
            overrides={"eu_aaa_slope_2y10y": 0.85},
        )
        df, _ = _ecb_historical("EURUSD", raw)
        assert (df["eu_aaa_slope_2y10y"] - 0.85).abs().max() < 1e-9

    def test_zero_leg_yields_zero_slope(self):
        """Live rule: slope = 0 when either leg is zero/missing."""
        raw = _build_raw_curve_df(include_slopes=False)
        raw["eu_aaa_2y_daily"] = float("nan")  # missing 2y
        df, _ = _ecb_historical("EURUSD", raw)
        assert (df["eu_aaa_2y_daily"] == 0.0).all()
        assert (df["eu_aaa_slope_2y10y"] == 0.0).all()
        # 3m10y unaffected — different leg
        assert (df["eu_aaa_slope_3m10y"] - (3.00 - 3.50)).abs().max() < 1e-9

    def test_all_10_tenors_present(self):
        raw = _build_raw_curve_df()
        df, _ = _ecb_historical("EURUSD", raw)
        for label in ("3m", "6m", "1y", "2y", "3y", "5y", "7y", "10y", "20y", "30y"):
            assert f"eu_aaa_{label}_daily" in df.columns


class TestHistoricalCurveReleaseLag:
    def test_step_change_appears_lag_days_later(self):
        n = 200
        change_idx = 100
        y10_series = [3.0] * change_idx + [4.0] * (n - change_idx)
        raw = _build_raw_curve_df(
            n_days=n,
            end=datetime(2024, 12, 31),
            overrides={"eu_aaa_10y_daily": y10_series},
            include_slopes=False,
        )
        change_date = raw.index[change_idx]
        df, _ = _ecb_historical(
            "EURUSD", raw,
            start=raw.index[0].to_pydatetime(),
            end=raw.index[-1].to_pydatetime(),
            lag_hours=24.0,
        )
        if change_date in df.index:
            assert df.loc[change_date, "eu_aaa_10y_daily"] == pytest.approx(3.0)
        post_lag_date = change_date + pd.Timedelta(days=1)
        if post_lag_date in df.index:
            assert df.loc[post_lag_date, "eu_aaa_10y_daily"] == pytest.approx(4.0)


class TestHistoricalCurveRefusesUnknownSource:
    def test_missing_source_in_config_raises(self):
        fetcher = ECBDataFetcher()
        store = AsyncMock()
        cfg = {"sources": {}}
        with pytest.raises(ValueError, match="not in data_feeds.yaml"):
            asyncio.run(fetcher.get_historical_curve_features(
                store, "EURUSD",
                datetime(2024, 1, 1), datetime(2024, 12, 31),
                feeds_config=cfg,
            ))

    def test_non_eur_short_circuits_before_config_check(self):
        """Non-EUR returns empty without reading config — that's fine."""
        fetcher = ECBDataFetcher()
        store = AsyncMock()
        cfg = {"sources": {}}  # broken config
        df = asyncio.run(fetcher.get_historical_curve_features(
            store, "USDJPY",
            datetime(2024, 1, 1), datetime(2024, 12, 31),
            feeds_config=cfg,
        ))
        assert df.empty
