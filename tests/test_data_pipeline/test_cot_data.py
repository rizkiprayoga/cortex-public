"""Tests for cot_data — XAU Disaggregated + FX TFF routing + parsing."""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.data_pipeline.fundamental.cot_data import (
    COTDataFetcher,
    FX_CONTRACT_CODES,
    GOLD_CONTRACT_CODE,
    _SYMBOL_CURRENCIES,
)


# ---------- Catalog ------------------------------------------------------

class TestContractCatalog:
    """Plan line 79 requires GBP/AUD/NZD codes; breadth adds EUR/JPY/CAD."""

    def test_xau_code_unchanged(self):
        assert GOLD_CONTRACT_CODE == "088691"

    def test_plan_required_fx_codes(self):
        """Plan line 79 names these 3 explicitly."""
        assert FX_CONTRACT_CODES["GBP"] == "096742"
        assert FX_CONTRACT_CODES["AUD"] == "232741"
        assert FX_CONTRACT_CODES["NZD"] == "112741"

    def test_all_six_currencies_covered(self):
        """Breadth: EUR, JPY, CAD added so cross pairs route correctly."""
        for ccy in ("EUR", "JPY", "CAD", "GBP", "AUD", "NZD"):
            assert ccy in FX_CONTRACT_CODES


class TestSymbolRoutingMap:
    """_SYMBOL_CURRENCIES must match the exposure sets used elsewhere."""

    def test_usd_pairs_route_to_other_side(self):
        """USDxxx pairs route to the non-USD currency."""
        assert _SYMBOL_CURRENCIES["EURUSD"] == ("EUR",)
        assert _SYMBOL_CURRENCIES["USDJPY"] == ("JPY",)
        assert _SYMBOL_CURRENCIES["USDCAD"] == ("CAD",)
        assert _SYMBOL_CURRENCIES["GBPUSD"] == ("GBP",)
        assert _SYMBOL_CURRENCIES["AUDUSD"] == ("AUD",)

    def test_cross_pairs_route_to_both_currencies(self):
        assert _SYMBOL_CURRENCIES["EURGBP"] == ("EUR", "GBP")
        assert _SYMBOL_CURRENCIES["EURJPY"] == ("EUR", "JPY")
        assert _SYMBOL_CURRENCIES["GBPJPY"] == ("GBP", "JPY")
        assert _SYMBOL_CURRENCIES["AUDNZD"] == ("AUD", "NZD")

    def test_xau_not_in_fx_map(self):
        assert "XAUUSD" not in _SYMBOL_CURRENCIES
        assert "ETHUSD" not in _SYMBOL_CURRENCIES


# ---------- Dispatch -----------------------------------------------------

class TestGetCotFeaturesDispatch:
    """get_cot_features dispatches XAU vs FX vs others, never raises."""

    def test_xau_path_runs_gold_fetch_returns_defaults_on_error(self):
        fetcher = COTDataFetcher()
        with patch.object(fetcher, "_get_gold_data",
                           side_effect=Exception("network down")):
            feats = fetcher.get_cot_features("XAUUSD")
        assert set(feats.keys()) == {
            "cot_net_position", "cot_net_zscore_52w", "cot_wow_change",
            "cot_commercial_ratio", "cot_extreme_flag",
        }
        assert all(v == 0.0 for v in feats.values())

    def test_fx_path_single_currency(self):
        fetcher = COTDataFetcher()
        with patch.object(fetcher, "_get_tff_data",
                           return_value=pd.DataFrame()):
            feats = fetcher.get_cot_features("GBPUSD")
        expected = {"cot_gbp_net_position", "cot_gbp_net_zscore_52w",
                     "cot_gbp_wow_change", "cot_gbp_dealer_ratio",
                     "cot_gbp_extreme_flag"}
        assert set(feats.keys()) == expected

    def test_fx_path_cross_pair_gets_both(self):
        fetcher = COTDataFetcher()
        with patch.object(fetcher, "_get_tff_data",
                           return_value=pd.DataFrame()):
            feats = fetcher.get_cot_features("EURJPY")
        for ccy in ("eur", "jpy"):
            for suffix in ("net_position", "net_zscore_52w", "wow_change",
                             "dealer_ratio", "extreme_flag"):
                assert f"cot_{ccy}_{suffix}" in feats

    def test_non_fx_non_xau_returns_empty(self):
        feats = COTDataFetcher().get_cot_features("ETHUSD")
        assert feats == {}

    def test_fx_path_never_raises_on_exception(self):
        fetcher = COTDataFetcher()
        with patch.object(fetcher, "_get_tff_data",
                           side_effect=Exception("boom")):
            feats = fetcher.get_cot_features("AUDNZD")
        # Falls back to defaults for (AUD, NZD)
        assert "cot_aud_net_zscore_52w" in feats
        assert "cot_nzd_net_zscore_52w" in feats


# ---------- TFF parsing --------------------------------------------------

# Minimal TFF CSV — only the columns we actually consume, plus a few
# contract codes (GBP, AUD, EUR) across multiple dates. Two weeks per
# currency so net-change + z-score math exercises.
_TFF_CSV = """CFTC_Contract_Market_Code,Report_Date_as_YYYY-MM-DD,Dealer_Positions_Long_All,Dealer_Positions_Short_All,Lev_Money_Positions_Long_All,Lev_Money_Positions_Short_All,Open_Interest_All
096742,2026-04-15,100000,50000,80000,40000,200000
096742,2026-04-22,110000,55000,90000,35000,210000
232741,2026-04-15, 70000,30000,50000,20000,150000
232741,2026-04-22, 72000,28000,55000,22000,155000
099741,2026-04-15, 80000,40000,70000,30000,180000
099741,2026-04-22, 85000,42000,75000,33000,185000
"""


class TestTffParse:
    """_parse_tff_csv must filter to FX contract codes and shape rows."""

    def _fetcher_with_tff(self) -> COTDataFetcher:
        fetcher = COTDataFetcher()
        df = fetcher._parse_tff_csv(_TFF_CSV)
        return fetcher, df

    def test_parse_filters_to_fx_contracts(self):
        _, df = self._fetcher_with_tff()
        assert not df.empty
        # 3 currencies × 2 weeks = 6 rows
        assert len(df) == 6
        assert set(df["currency"].unique()) == {"GBP", "AUD", "EUR"}

    def test_parse_computes_net_spec(self):
        """net_spec = leveraged_long - leveraged_short."""
        _, df = self._fetcher_with_tff()
        gbp_rows = df[df["currency"] == "GBP"].sort_values("date")
        # Week 1: 80000 - 40000 = 40000
        # Week 2: 90000 - 35000 = 55000
        assert gbp_rows["net_spec"].iloc[0] == 40000
        assert gbp_rows["net_spec"].iloc[1] == 55000

    def test_parse_computes_net_dealer(self):
        """net_dealer = dealer_long - dealer_short."""
        _, df = self._fetcher_with_tff()
        gbp_rows = df[df["currency"] == "GBP"].sort_values("date")
        # Week 1: 100000 - 50000 = 50000
        assert gbp_rows["net_dealer"].iloc[0] == 50000

    def test_parse_drops_non_fx_contracts(self):
        """Gold contract (088691) should not appear in TFF filter result."""
        mixed_csv = (
            _TFF_CSV
            + "088691,2026-04-22,0,0,0,0,0\n"
        )
        df = COTDataFetcher()._parse_tff_csv(mixed_csv)
        assert 0 == (df["code"] == "088691").sum()

    def test_parse_malformed_csv_returns_empty(self):
        df = COTDataFetcher()._parse_tff_csv("not,a,real,csv\n")
        assert df.empty


# ---------- FX feature computation --------------------------------------

class TestFxFeatureComputation:
    """_compute_features_fx emits correct per-currency keys + values."""

    def _df(self) -> pd.DataFrame:
        """Build a sample GBP TFF DataFrame (requires ≥2 weeks)."""
        return pd.DataFrame({
            "currency":      ["GBP", "GBP"],
            "code":          ["096742", "096742"],
            "date":          pd.to_datetime(["2026-04-15", "2026-04-22"]),
            "dealer_long":   [100000, 110000],
            "dealer_short":  [50000, 55000],
            "lev_long":      [80000, 90000],
            "lev_short":     [40000, 35000],
            "open_interest": [200000, 210000],
            "net_spec":      [40000, 55000],
            "net_dealer":    [50000, 55000],
        })

    def test_emits_5_features_for_one_currency(self):
        feats = COTDataFetcher()._compute_features_fx(self._df(), "GBP")
        assert set(feats.keys()) == {
            "cot_gbp_net_position", "cot_gbp_net_zscore_52w",
            "cot_gbp_wow_change", "cot_gbp_dealer_ratio",
            "cot_gbp_extreme_flag",
        }

    def test_net_position_normalized(self):
        """Latest net (55000) / max abs net in sample (55000) = 1.0."""
        feats = COTDataFetcher()._compute_features_fx(self._df(), "GBP")
        assert feats["cot_gbp_net_position"] == pytest.approx(1.0)

    def test_wow_change_matches_formula(self):
        """wow_change = (latest_net - prev_net) / OI = (55000 - 40000) / 210000."""
        feats = COTDataFetcher()._compute_features_fx(self._df(), "GBP")
        expected = (55000 - 40000) / 210000
        assert feats["cot_gbp_wow_change"] == pytest.approx(expected)

    def test_dealer_ratio_matches_formula(self):
        """dealer_ratio = latest_net_dealer / OI = 55000 / 210000."""
        feats = COTDataFetcher()._compute_features_fx(self._df(), "GBP")
        assert feats["cot_gbp_dealer_ratio"] == pytest.approx(55000 / 210000)

    def test_missing_currency_returns_defaults(self):
        """Asking for a currency not in the DF → default zeros."""
        feats = COTDataFetcher()._compute_features_fx(self._df(), "AUD")
        assert feats["cot_aud_net_zscore_52w"] == 0.0


# ===========================================================================
# Phase 2A — historical feature reader from feature_store
# ===========================================================================

import asyncio
from unittest.mock import AsyncMock


def _build_raw_xau_df(
    *,
    n_weeks: int = 80,
    end: datetime | None = None,
    overrides: dict[str, list[float] | float] | None = None,
) -> pd.DataFrame:
    """Build a wide raw-observation DataFrame for XAU disagg."""
    if end is None:
        end = datetime(2026, 4, 1)
    idx = pd.date_range(end=end, periods=n_weeks, freq="W-TUE")
    cols: dict[str, list[float]] = {
        "mm_long":       [180000.0] * n_weeks,
        "mm_short":      [80000.0] * n_weeks,
        "comm_long":     [120000.0] * n_weeks,
        "comm_short":    [220000.0] * n_weeks,
        "open_interest": [500000.0] * n_weeks,
        "net_spec":      [100000.0] * n_weeks,
        "net_comm":      [-100000.0] * n_weeks,
    }
    if overrides:
        for k, v in overrides.items():
            cols[k] = list(v) if hasattr(v, "__len__") else [float(v)] * n_weeks
    return pd.DataFrame(cols, index=idx)


def _build_raw_tff_df(
    *,
    n_weeks: int = 80,
    end: datetime | None = None,
    currencies: tuple[str, ...] = ("eur", "jpy", "cad", "gbp", "aud", "nzd"),
    overrides: dict[str, list[float] | float] | None = None,
) -> pd.DataFrame:
    """Build a wide raw-observation DataFrame for TFF FX (multi-currency)."""
    if end is None:
        end = datetime(2026, 4, 1)
    idx = pd.date_range(end=end, periods=n_weeks, freq="W-TUE")
    cols: dict[str, list[float]] = {}
    for c in currencies:
        cols[f"{c}_dealer_long"] = [80000.0] * n_weeks
        cols[f"{c}_dealer_short"] = [40000.0] * n_weeks
        cols[f"{c}_lev_long"] = [100000.0] * n_weeks
        cols[f"{c}_lev_short"] = [60000.0] * n_weeks
        cols[f"{c}_open_interest"] = [200000.0] * n_weeks
        cols[f"{c}_net_spec"] = [40000.0] * n_weeks
        cols[f"{c}_net_dealer"] = [40000.0] * n_weeks
    if overrides:
        for k, v in overrides.items():
            cols[k] = list(v) if hasattr(v, "__len__") else [float(v)] * n_weeks
    return pd.DataFrame(cols, index=idx)


def _cot_store(raw_df: pd.DataFrame, expected_group: str) -> AsyncMock:
    """Mock store that returns raw_df only for the requested feature_group."""
    store = AsyncMock()
    captured: dict = {}

    async def _read(symbol, feature_group, start=None, end=None):
        captured["symbol"] = symbol
        captured["feature_group"] = feature_group
        if feature_group == expected_group:
            return raw_df.copy()
        return pd.DataFrame()

    store.read_feature_store = _read
    store.captured = captured
    return store


def _cot_historical(
    symbol: str,
    raw_df: pd.DataFrame,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    lag_hours: float = 0.0,
    expected_group: str = "cot_disagg",
) -> tuple[pd.DataFrame, dict]:
    fetcher = COTDataFetcher()
    store = _cot_store(raw_df, expected_group)
    cfg = {
        "sources": {
            "cot_disagg": {"release_lag_hours": lag_hours},
            "cot_tff":    {"release_lag_hours": lag_hours},
        },
    }
    if start is None:
        start = raw_df.index.min().to_pydatetime() if not raw_df.empty else datetime(2024, 1, 1)
    if end is None:
        end = raw_df.index.max().to_pydatetime() if not raw_df.empty else datetime(2024, 12, 31)
    df = asyncio.run(fetcher.get_historical_cot_features(
        store, symbol, start, end, feeds_config=cfg,
    ))
    return df, store.captured


class TestHistoricalCOTSchemaMatchesLive:
    """Historical column set == live default keys per symbol."""

    def test_xauusd_columns_match_default_xau(self):
        live_keys = set(COTDataFetcher._default_xau_features().keys())
        raw = _build_raw_xau_df()
        df, _ = _cot_historical("XAUUSD", raw, expected_group="cot_disagg")
        assert set(df.columns) == live_keys

    @pytest.mark.parametrize("symbol,currencies", [
        ("EURUSD", ("EUR",)),
        ("USDJPY", ("JPY",)),
        ("USDCAD", ("CAD",)),
        ("GBPUSD", ("GBP",)),
        ("AUDUSD", ("AUD",)),
        ("EURGBP", ("EUR", "GBP")),
        ("EURJPY", ("EUR", "JPY")),
        ("GBPJPY", ("GBP", "JPY")),
        ("AUDNZD", ("AUD", "NZD")),
    ])
    def test_fx_columns_match_default_fx(self, symbol, currencies):
        live_keys = set(COTDataFetcher._default_fx_features(currencies).keys())
        raw = _build_raw_tff_df()
        df, _ = _cot_historical(symbol, raw, expected_group="cot_tff")
        assert set(df.columns) == live_keys

    @pytest.mark.parametrize("symbol", ["ETHUSD", "BTCUSD", "UNKNOWN"])
    def test_non_classified_symbols_get_empty_dataframe(self, symbol):
        """Symbols with no FX exposure return empty (not in _SYMBOL_CURRENCIES)."""
        raw = _build_raw_tff_df()
        df, _ = _cot_historical(symbol, raw, expected_group="cot_tff")
        assert df.empty


class TestHistoricalCOTDispatch:
    """XAU vs FX dispatch must hit the right feature_group."""

    def test_xau_hits_cot_disagg(self):
        raw = _build_raw_xau_df()
        _, captured = _cot_historical("XAUUSD", raw, expected_group="cot_disagg")
        assert captured["feature_group"] == "cot_disagg"

    def test_fx_hits_cot_tff(self):
        raw = _build_raw_tff_df()
        _, captured = _cot_historical("EURUSD", raw, expected_group="cot_tff")
        assert captured["feature_group"] == "cot_tff"


class TestHistoricalXAUEngineering:
    """XAU 5-feature engineering matches live."""

    def test_constant_input_yields_zero_zscore(self):
        raw = _build_raw_xau_df()  # all values constant
        df, _ = _cot_historical("XAUUSD", raw, expected_group="cot_disagg")
        # After warmup z should be 0
        assert (df["cot_net_zscore_52w"].iloc[60:].abs() < 1e-9).all()
        assert (df["cot_extreme_flag"].iloc[60:].abs() < 1e-9).all()

    def test_zscore_clipped_to_pm_2(self):
        n = 80
        ns = [100000.0] * (n // 2) + [10_000_000.0] * (n // 2)
        raw = _build_raw_xau_df(n_weeks=n, overrides={"net_spec": ns})
        df, _ = _cot_historical("XAUUSD", raw, expected_group="cot_disagg")
        assert df["cot_net_zscore_52w"].max() <= 2.0
        assert df["cot_net_zscore_52w"].min() >= -2.0

    def test_extreme_flag_contrarian_polarity(self):
        """zscore > +1.5 → -1 (specs over-long → bearish contrarian)."""
        n = 80
        ns = [100000.0] * (n - 5) + [10_000_000.0] * 5
        raw = _build_raw_xau_df(n_weeks=n, overrides={"net_spec": ns})
        df, _ = _cot_historical("XAUUSD", raw, expected_group="cot_disagg")
        last_z = df["cot_net_zscore_52w"].iloc[-1]
        last_flag = df["cot_extreme_flag"].iloc[-1]
        assert last_z > 1.5 - 1e-9
        assert last_flag == -1.0

    def test_wow_change_normalized_by_oi(self):
        """wow_change = (net - net.shift(1)) / open_interest."""
        n = 10
        ns = [100000.0, 110000.0, 120000.0, 130000.0, 140000.0,
              150000.0, 160000.0, 170000.0, 180000.0, 190000.0]
        oi = [500000.0] * n
        raw = _build_raw_xau_df(
            n_weeks=n, overrides={"net_spec": ns, "open_interest": oi},
        )
        df, _ = _cot_historical("XAUUSD", raw, expected_group="cot_disagg")
        assert df["cot_wow_change"].iloc[1] == pytest.approx(0.02)

    def test_commercial_ratio_uses_oi(self):
        raw = _build_raw_xau_df(overrides={
            "net_comm": -150000.0, "open_interest": 500000.0,
        })
        df, _ = _cot_historical("XAUUSD", raw, expected_group="cot_disagg")
        assert (df["cot_commercial_ratio"] - (-150000.0 / 500000.0)).abs().max() < 1e-9


class TestHistoricalFXEngineering:
    """FX 5-feature-per-currency engineering matches live."""

    def test_eurusd_only_emits_eur_block(self):
        raw = _build_raw_tff_df()
        df, _ = _cot_historical("EURUSD", raw, expected_group="cot_tff")
        for k in ("net_position", "net_zscore_52w", "wow_change",
                  "dealer_ratio", "extreme_flag"):
            assert f"cot_eur_{k}" in df.columns
        for ccy in ("jpy", "cad", "gbp", "aud", "nzd"):
            assert f"cot_{ccy}_net_zscore_52w" not in df.columns

    def test_eurgbp_emits_both_eur_and_gbp_blocks(self):
        raw = _build_raw_tff_df()
        df, _ = _cot_historical("EURGBP", raw, expected_group="cot_tff")
        assert "cot_eur_net_zscore_52w" in df.columns
        assert "cot_gbp_net_zscore_52w" in df.columns
        assert "cot_jpy_net_zscore_52w" not in df.columns

    def test_audnzd_emits_aud_and_nzd_blocks(self):
        raw = _build_raw_tff_df()
        df, _ = _cot_historical("AUDNZD", raw, expected_group="cot_tff")
        assert "cot_aud_net_zscore_52w" in df.columns
        assert "cot_nzd_net_zscore_52w" in df.columns
        assert "cot_eur_net_zscore_52w" not in df.columns

    def test_missing_currency_in_raw_emits_zeros(self):
        """If TFF rows miss data for a routed currency, schema still emits zeros."""
        raw = _build_raw_tff_df(currencies=("eur",))  # only EUR data
        df, _ = _cot_historical("EURGBP", raw, expected_group="cot_tff")
        assert "cot_gbp_net_zscore_52w" in df.columns
        assert (df["cot_gbp_net_zscore_52w"] == 0.0).all()
        assert (df["cot_gbp_dealer_ratio"] == 0.0).all()


class TestHistoricalCOTEmpty:
    def test_empty_raw_returns_empty(self):
        df, _ = _cot_historical(
            "XAUUSD", pd.DataFrame(),
            start=datetime(2024, 1, 1), end=datetime(2024, 12, 31),
            expected_group="cot_disagg",
        )
        assert df.empty

    def test_eth_returns_empty_without_querying_store(self):
        fetcher = COTDataFetcher()
        store = AsyncMock()
        cfg = {"sources": {"cot_disagg": {"release_lag_hours": 504},
                            "cot_tff":    {"release_lag_hours": 504}}}
        df = asyncio.run(fetcher.get_historical_cot_features(
            store, "ETHUSD",
            datetime(2024, 1, 1), datetime(2024, 12, 31),
            feeds_config=cfg,
        ))
        assert df.empty
        store.read_feature_store.assert_not_called()


class TestHistoricalCOTReleaseLag:
    def test_xau_step_change_appears_lag_days_later(self):
        n = 80
        change_idx = 40
        ns = [100000.0] * change_idx + [200000.0] * (n - change_idx)
        raw = _build_raw_xau_df(n_weeks=n, overrides={"net_spec": ns})
        change_date = raw.index[change_idx]
        df, _ = _cot_historical(
            "XAUUSD", raw, lag_hours=72.0, expected_group="cot_disagg",
        )
        post_lag_date = change_date + pd.Timedelta(hours=72)
        if post_lag_date in df.index:
            assert df.loc[post_lag_date, "cot_net_position"] > 0.9


class TestHistoricalCOTRefusesUnknownSource:
    def test_xau_with_missing_disagg_config_raises(self):
        fetcher = COTDataFetcher()
        store = AsyncMock()
        cfg = {"sources": {}}
        with pytest.raises(ValueError, match="not in data_feeds.yaml"):
            asyncio.run(fetcher.get_historical_cot_features(
                store, "XAUUSD",
                datetime(2024, 1, 1), datetime(2024, 12, 31),
                feeds_config=cfg,
            ))

    def test_fx_with_missing_tff_config_raises(self):
        fetcher = COTDataFetcher()
        store = AsyncMock()
        cfg = {"sources": {}}
        with pytest.raises(ValueError, match="not in data_feeds.yaml"):
            asyncio.run(fetcher.get_historical_cot_features(
                store, "EURUSD",
                datetime(2024, 1, 1), datetime(2024, 12, 31),
                feeds_config=cfg,
            ))
