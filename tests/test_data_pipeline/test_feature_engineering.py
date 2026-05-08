"""Tests for FeatureEngineer — expanded 56-feature pipeline."""

import pandas as pd
import numpy as np
import pytest

from src.data_pipeline.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_ohlcv():
    """Synthetic OHLCV DataFrame with 500 bars (need 200+ for warmup)."""
    np.random.seed(0)
    n = 500
    close = 1900 + np.cumsum(np.random.randn(n) * 5)
    return pd.DataFrame({
        "open":        close + np.random.randn(n),
        "high":        close + abs(np.random.randn(n)) * 3,
        "low":         close - abs(np.random.randn(n)) * 3,
        "close":       close,
        "tick_volume": np.random.randint(500, 2000, n).astype(float),
    }, index=pd.date_range("2023-01-01", periods=n, freq="4h"))


class TestFeatureEngineer:

    def setup_method(self):
        self.eng = FeatureEngineer()

    def test_transform_returns_dataframe(self, sample_ohlcv):
        """transform() should return a DataFrame with no NaN values."""
        result = self.eng.transform(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert result.isnull().sum().sum() == 0

    def test_feature_count(self, sample_ohlcv):
        """Should produce 57 technical features (no fundamentals)."""
        result = self.eng.transform(sample_ohlcv)
        # 57 = 56 + 1 (efficiency_ratio added in Task 6 for E-7 trend-mode)
        assert len(result.columns) == 57

    def test_feature_count_with_fundamentals(self, sample_ohlcv):
        """Should produce 57 + 4 = 61 features with fundamental scores."""
        scores = {
            "macro_score": 0.3,
            "sentiment_score": -0.2,
            "onchain_score": 0.1,
            "cot_score": 0.0,
        }
        result = self.eng.transform(sample_ohlcv, fundamental_scores=scores)
        # 61 = 60 + 1 (efficiency_ratio added in Task 6 for E-7 trend-mode)
        assert len(result.columns) == 61

    def test_log_return_column_present(self, sample_ohlcv):
        """Feature DataFrame must include all log_return variants."""
        result = self.eng.transform(sample_ohlcv)
        for col in ["log_return", "log_return_5", "log_return_10", "log_return_20"]:
            assert col in result.columns

    def test_rsi_bounded(self, sample_ohlcv):
        """RSI values must be in [0, 100]."""
        result = self.eng.transform(sample_ohlcv)
        assert result["rsi_14"].between(0, 100).all()
        assert result["rsi_7"].between(0, 100).all()

    def test_mfi_bounded(self, sample_ohlcv):
        """MFI values must be in [0, 100]."""
        result = self.eng.transform(sample_ohlcv)
        assert result["mfi_14"].between(0, 100).all()

    def test_adx_bounded(self, sample_ohlcv):
        """ADX values must be in [0, 100]."""
        result = self.eng.transform(sample_ohlcv)
        assert result["adx"].between(0, 100).all()

    def test_bb_pct_b_reasonable(self, sample_ohlcv):
        """bb_pct_b should mostly be in [0, 1] (can exceed during extremes)."""
        result = self.eng.transform(sample_ohlcv)
        # At least 90% of values should be in [0, 1]
        in_range = result["bb_pct_b"].between(0, 1).mean()
        assert in_range > 0.85

    def test_close_position_bounded(self, sample_ohlcv):
        """close_position_in_range should be in [0, 1]."""
        result = self.eng.transform(sample_ohlcv)
        assert result["close_position_in_range"].between(0, 1).all()

    def test_hurst_exponent_bounded(self, sample_ohlcv):
        """Hurst exponent should be in (0, 1)."""
        result = self.eng.transform(sample_ohlcv)
        assert result["hurst_exponent"].between(0, 1.5).all()

    def test_sma_rel_symbol_agnostic(self, sample_ohlcv):
        """SMA relative features should be small numbers (not absolute prices)."""
        result = self.eng.transform(sample_ohlcv)
        # Relative features should be << 1 in absolute value for typical data
        for col in ["sma_10_rel", "sma_20_rel", "sma_50_rel"]:
            assert result[col].abs().max() < 1.0, f"{col} too large — not relative?"

    def test_to_matrix_shape(self, sample_ohlcv):
        """to_matrix() should return float64 array with correct shape."""
        df = self.eng.transform(sample_ohlcv)
        matrix = self.eng.to_matrix(df)
        assert matrix.dtype == np.float64
        assert matrix.ndim == 2
        assert matrix.shape[0] == len(df)
        # 57 = 56 + 1 (efficiency_ratio added in Task 6 for E-7 trend-mode)
        assert matrix.shape[1] == 57

    def test_to_matrix_columns_sorted(self, sample_ohlcv):
        """to_matrix() output column order should be deterministic (sorted)."""
        df = self.eng.transform(sample_ohlcv)
        # Shuffled columns should produce same matrix
        shuffled = df[np.random.default_rng(42).permutation(df.columns)]
        mat1 = self.eng.to_matrix(df)
        mat2 = self.eng.to_matrix(shuffled)
        np.testing.assert_array_equal(mat1, mat2)

    def test_empty_input(self, sample_ohlcv):
        """transform() with empty DataFrame should return empty."""
        result = self.eng.transform(pd.DataFrame())
        assert result.empty

    def test_no_raw_ohlcv_in_output(self, sample_ohlcv):
        """Raw OHLCV columns should not appear in output."""
        result = self.eng.transform(sample_ohlcv)
        for col in ["open", "high", "low", "close", "tick_volume"]:
            assert col not in result.columns

    def test_efficiency_ratio_present_and_bounded(self, sample_ohlcv):
        """Kaufman's Efficiency Ratio must be exposed in transform() output
        and constrained to [0, 1] (Task 6 / E-7 trend-mode)."""
        result = self.eng.transform(sample_ohlcv)
        assert "efficiency_ratio" in result.columns
        er = result["efficiency_ratio"].dropna()
        assert (er >= 0.0).all()
        assert (er <= 1.0).all()

    def test_d1_w1_subsets_exclude_efficiency_ratio(self, sample_ohlcv):
        """ER is H4-only by spec §3.3 — D1/W1 subsets must NOT add it.

        Guards against a future engineer accidentally inserting
        ``efficiency_ratio`` into the D1/W1 allow-lists in
        ``TIMEFRAME_FEATURE_SUBSETS`` (feature_engineering.py lines ~442-454).
        """
        # Build a multi-TF dict: H4 = primary (full set), plus D1 + W1 resampled.
        h4 = sample_ohlcv
        d1 = h4.resample("1D").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "tick_volume": "sum",
        }).dropna()
        w1 = h4.resample("1W").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "tick_volume": "sum",
        }).dropna()

        result = self.eng.transform_multi_timeframe(
            {"H4": h4, "D1": d1, "W1": w1}, primary_tf="H4",
        )

        # Primary H4 keeps the unprefixed column.
        assert "efficiency_ratio" in result.columns
        # D1 + W1 subsets must NOT carry ER through the prefix path.
        assert "D1_efficiency_ratio" not in result.columns
        assert "W1_efficiency_ratio" not in result.columns


class TestTripleBarrierLabels:
    """Phase B.3 — compute_triple_barrier_labels."""

    def test_empty_returns_empty(self):
        result = FeatureEngineer.compute_triple_barrier_labels(pd.DataFrame())
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_labels_in_valid_range(self):
        """All labels must be in {-1, 0, +1}."""
        np.random.seed(42)
        n = 300
        close = 1900 + np.cumsum(np.random.randn(n) * 5)
        ohlcv = pd.DataFrame({
            "open":  close,
            "high":  close + abs(np.random.randn(n)) * 2,
            "low":   close - abs(np.random.randn(n)) * 2,
            "close": close,
        }, index=pd.date_range("2023-01-01", periods=n, freq="4h"))

        labels = FeatureEngineer.compute_triple_barrier_labels(
            ohlcv, tp_r_mult=2.5, sl_atr_mult=2.0, time_limit_bars=20,
        )
        assert len(labels) == n
        assert set(np.unique(labels)).issubset({-1.0, 0.0, 1.0})

    def test_tp_hit_recognized(self):
        """Strong upward series should produce predominantly +1 labels."""
        n = 100
        # Monotonic rising series — TP should always hit
        close = np.linspace(100, 150, n)
        ohlcv = pd.DataFrame({
            "open": close, "high": close + 0.5,
            "low": close - 0.1, "close": close,
        }, index=pd.date_range("2023-01-01", periods=n, freq="4h"))

        labels = FeatureEngineer.compute_triple_barrier_labels(
            ohlcv, tp_r_mult=2.0, sl_atr_mult=1.0, time_limit_bars=20,
        )
        # Exclude the last few bars (can't look forward)
        core = labels[20:-25]
        assert (core == 1.0).sum() > (core == -1.0).sum()

    def test_sl_hit_recognized(self):
        """Monotonic falling series should produce predominantly -1 labels."""
        n = 100
        close = np.linspace(150, 100, n)
        ohlcv = pd.DataFrame({
            "open": close, "high": close + 0.1,
            "low": close - 0.5, "close": close,
        }, index=pd.date_range("2023-01-01", periods=n, freq="4h"))

        labels = FeatureEngineer.compute_triple_barrier_labels(
            ohlcv, tp_r_mult=2.0, sl_atr_mult=1.0, time_limit_bars=20,
        )
        core = labels[20:-25]
        assert (core == -1.0).sum() > (core == 1.0).sum()

    def test_conservative_sl_when_both_hit(self):
        """When TP and SL both hit in the same bar, SL should win (conservative)."""
        # Construct OHLCV where on bar 1 both TP and SL are touched
        ohlcv = pd.DataFrame({
            "open":  [100.0, 100.0],
            "high":  [100.0, 110.0],   # reaches +10 (TP)
            "low":   [100.0,  90.0],   # reaches -10 (SL)
            "close": [100.0, 100.0],
        }, index=pd.date_range("2023-01-01", periods=2, freq="4h"))

        # Manually pass a simple ATR = 5.0 — then TP=+10, SL=-10 with
        # sl_mult=2, tp_r=1 → SL=10 below, TP=10 above
        atr = pd.Series([5.0, 5.0], index=ohlcv.index)

        labels = FeatureEngineer.compute_triple_barrier_labels(
            ohlcv, atr=atr, tp_r_mult=1.0, sl_atr_mult=2.0,
            time_limit_bars=5,
        )
        assert labels[0] == -1.0  # SL wins tie
