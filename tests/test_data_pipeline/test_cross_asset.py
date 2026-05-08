"""Tests for cross_asset routing (Forex Phase 1B yfinance extension)."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.market.cross_asset import (
    CROSS_ASSET_TICKERS,
    CrossAssetFetcher,
    _AUD_EXPOSURE,
    _EUR_EXPOSURE,
    _GBP_EXPOSURE,
    _JPY_EXPOSURE,
    _NZD_EXPOSURE,
)


# Expected tickers per plan line 77.
_PLAN_TICKERS = {
    "^FTSE", "^AXJO", "^NZ50", "^SSEC", "^HSI",
    "^N225", "^GDAXI", "BHP.AX", "RIO.L", "VALE",
    "HG=F", "CNH=X",
}


class TestTickerCoverage:
    """Plan line 77 requires all 12 tickers in CROSS_ASSET_TICKERS."""

    def test_all_plan_tickers_present(self):
        for ticker in _PLAN_TICKERS:
            assert ticker in CROSS_ASSET_TICKERS, f"Missing ticker: {ticker}"

    def test_original_tickers_retained(self):
        """Don't drop the USD-axis tickers during extension."""
        for ticker in ("DX-Y.NYB", "^GSPC", "^VIX", "CL=F", "^TNX", "GC=F"):
            assert ticker in CROSS_ASSET_TICKERS


class TestDefaultFeatureRouting:
    """_default_features must emit the right keys per currency."""

    @pytest.fixture
    def fetcher(self):
        return CrossAssetFetcher()

    def test_xau_unchanged(self, fetcher):
        """XAU still gets only USD-axis + gold/silver."""
        defaults = fetcher._default_features("XAUUSD")
        # No country-specific leakage
        for key in ("ftse_log_return", "dax_log_return", "nikkei_log_return",
                     "axjo_log_return", "nz50_log_return"):
            assert key not in defaults

    def test_eurusd_gets_dax(self, fetcher):
        defaults = fetcher._default_features("EURUSD")
        assert "dax_log_return" in defaults
        assert "dax_zscore" in defaults
        # No other country blocks
        assert "ftse_log_return" not in defaults
        assert "nikkei_log_return" not in defaults
        assert "axjo_log_return" not in defaults

    def test_gbpusd_gets_ftse_and_rio(self, fetcher):
        defaults = fetcher._default_features("GBPUSD")
        for key in ("ftse_log_return", "ftse_zscore",
                     "rio_l_log_return", "rio_l_zscore"):
            assert key in defaults
        assert "dax_log_return" not in defaults

    def test_audusd_gets_china_block(self, fetcher):
        """AUD pairs get the full AU+China commodity stack."""
        defaults = fetcher._default_features("AUDUSD")
        for prefix in ("axjo", "ssec", "hsi", "bhp", "vale", "copper", "cnh"):
            assert f"{prefix}_log_return" in defaults, f"AUDUSD missing {prefix}"
            assert f"{prefix}_zscore" in defaults
        # No GBP/EUR/JPY contamination
        assert "ftse_log_return" not in defaults
        assert "dax_log_return" not in defaults

    def test_eurgbp_gets_both_blocks(self, fetcher):
        defaults = fetcher._default_features("EURGBP")
        # EUR block
        assert "dax_log_return" in defaults
        # GBP block
        assert "ftse_log_return" in defaults
        assert "rio_l_log_return" in defaults
        # No USD-side extras
        assert "axjo_log_return" not in defaults

    def test_eurjpy_gets_dax_and_nikkei(self, fetcher):
        defaults = fetcher._default_features("EURJPY")
        assert "dax_log_return" in defaults
        assert "nikkei_log_return" in defaults
        assert "ftse_log_return" not in defaults

    def test_gbpjpy_gets_ftse_and_nikkei(self, fetcher):
        defaults = fetcher._default_features("GBPJPY")
        assert "ftse_log_return" in defaults
        assert "nikkei_log_return" in defaults
        assert "dax_log_return" not in defaults

    def test_audnzd_gets_full_aud_block_plus_nz50(self, fetcher):
        """AUDNZD is the only NZD-exposed pair — gets full AUD block + NZ50."""
        defaults = fetcher._default_features("AUDNZD")
        # AUD block
        for prefix in ("axjo", "ssec", "hsi", "bhp", "vale", "copper", "cnh"):
            assert f"{prefix}_log_return" in defaults
        # NZD block
        assert "nz50_log_return" in defaults
        assert "nz50_zscore" in defaults

    def test_ethusd_no_country_blocks(self, fetcher):
        """Crypto gets only USD-axis."""
        defaults = fetcher._default_features("ETHUSD")
        for key in ("ftse_log_return", "dax_log_return", "nikkei_log_return",
                     "axjo_log_return", "nz50_log_return"):
            assert key not in defaults


class TestComputeFeaturesWithStubbedData:
    """Pipe stub data through _compute_features and verify routing."""

    def _stub_data(self, tickers: list[str] = None) -> dict:
        """Build a 30-day synthetic OHLCV dict for every ticker."""
        tickers = tickers or list(set(CROSS_ASSET_TICKERS.values()))
        idx = pd.date_range(end=datetime.utcnow(), periods=30, freq="D")
        data = {}
        for label in tickers:
            close = pd.Series(np.linspace(100.0, 110.0, 30), index=idx)
            df = pd.DataFrame({"Close": close, "Open": close, "High": close,
                                "Low": close, "Volume": 1e6}, index=idx)
            data[label] = df
        return data

    def test_audusd_compute_emits_china_block(self):
        fetcher = CrossAssetFetcher()
        data = self._stub_data()
        feats = fetcher._compute_features(data, "AUDUSD")
        for prefix in ("axjo", "ssec", "hsi", "bhp", "vale", "copper", "cnh"):
            assert f"{prefix}_log_return" in feats
            assert f"{prefix}_zscore" in feats

    def test_gbpusd_compute_emits_ftse_rio(self):
        fetcher = CrossAssetFetcher()
        data = self._stub_data()
        feats = fetcher._compute_features(data, "GBPUSD")
        assert "ftse_log_return" in feats
        assert "rio_l_log_return" in feats
        # GBP pair should NOT get DAX
        assert "dax_log_return" not in feats

    def test_audnzd_compute_emits_both_blocks(self):
        fetcher = CrossAssetFetcher()
        data = self._stub_data()
        feats = fetcher._compute_features(data, "AUDNZD")
        # AUD side
        assert "axjo_log_return" in feats
        assert "copper_log_return" in feats
        # NZ side
        assert "nz50_log_return" in feats
        # No USD-side country blocks
        assert "ftse_log_return" not in feats

    def test_xauusd_unchanged(self):
        """Regression — XAU still gets only USD-axis + gold/silver."""
        fetcher = CrossAssetFetcher()
        data = self._stub_data()
        feats = fetcher._compute_features(data, "XAUUSD")
        # Core features present
        assert "dxy_log_return" in feats
        assert "gold_silver_ratio" in feats
        # No country-specific leakage
        assert "ftse_log_return" not in feats
        assert "axjo_log_return" not in feats
