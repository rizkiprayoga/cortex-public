"""Unit tests for Kaufman Efficiency Ratio used by E-7 trend-mode detector."""
import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.feature_engineering import FeatureEngineer


@pytest.fixture
def engineer():
    """FeatureEngineer instance — ER doesn't touch the data store, so None is fine."""
    return FeatureEngineer(data_store=None)


class TestEfficiencyRatio:
    def test_monotonic_uptrend_er_close_to_one(self, engineer):
        """A perfectly straight uptrend has gross movement == net movement -> ER ~= 1.0."""
        closes = pd.Series(np.arange(100, 200, dtype=float))  # +1 every bar
        er = engineer._compute_efficiency_ratio(closes, n=20)
        # Last value: net=20, vol=20 -> ER=1.0
        assert er.iloc[-1] == pytest.approx(1.0, abs=1e-9)

    def test_round_trip_er_close_to_zero(self, engineer):
        """A round-trip (same close after n bars) has net=0 → ER=0.

        Uses n=10 with 11 anchor bars + 10 round-trip bars (21 total) so
        shift(n) resolves to a valid anchor — the previous version of this
        test passed because shift(20) on a 20-bar series produced NaN, not
        because ER actually evaluated zero on a real round-trip.
        """
        # NB: the reviewer-suggested arange bounds (101..106 up, 105..100 down)
        # don't actually return to the anchor — np.arange(105, 100, -1) ends at
        # 101, not 100, so the last close is 101 vs anchor 100 → ER ≠ 0.
        # Corrected to up=[101..105], down=[104..100] so close[-1] == close[-(n+1)] == 100.
        n = 10
        anchor = [100.0] * (n + 1)
        round_trip = list(np.arange(101, 106, dtype=float)) + list(np.arange(104, 99, -1, dtype=float))
        closes = pd.Series(anchor + round_trip)
        # Sanity: confirm it's actually a round-trip at the last bar.
        assert closes.iloc[-1] == closes.iloc[-(n + 1)] == 100.0
        er = engineer._compute_efficiency_ratio(closes, n=n)
        assert er.iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_first_n_bars_are_nan_safe(self, engineer):
        """First n−1 bars cannot have ER (no full window). Must be 0.0."""
        closes = pd.Series(np.arange(100, 130, dtype=float))
        er = engineer._compute_efficiency_ratio(closes, n=20)
        assert len(er) == len(closes)
        assert not er.isna().any()
        # Explicitly verify early bars are 0.0 (the fillna path) so
        # downstream `er > 0.30` comparisons in the detector don't error.
        assert (er.iloc[:19] == 0.0).all()

    def test_flat_segment_returns_zero(self, engineer):
        """A perfectly flat price series has both direction and volatility = 0.
        The replace(0, np.nan) + fillna(0.0) guard must return 0.0, not NaN
        or div-by-zero error. This is the only non-trivial defensive line
        in the implementation."""
        closes = pd.Series([100.0] * 30)
        er = engineer._compute_efficiency_ratio(closes, n=20)
        assert not er.isna().any()
        assert er.iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_er_bounded_zero_to_one(self, engineer):
        """ER must lie in [0, 1] for any non-degenerate price path."""
        rng = np.random.default_rng(42)
        closes = pd.Series(100 + rng.normal(0, 1, 200).cumsum())
        er = engineer._compute_efficiency_ratio(closes, n=20)
        assert er.min() >= 0.0
        assert er.max() <= 1.0

    def test_kaufman_published_example(self, engineer):
        """Synthetic example matches the canonical pandas-ta/Kaufman formula:
            direction  = |close[t] - close[t-n]|     (n-bar lag via .shift(n))
            volatility = sum(|Δclose|, n diffs)      (.rolling(n).sum() over .diff().abs())

        n=5, prices = [100, 102, 101, 103, 102, 104]
        At position 5:
            direction  = |104 − 100| = 4              (shift(5) = close[0])
            volatility = |2|+|-1|+|2|+|-1|+|2| = 8    (5 diffs covered by rolling(5))
            ER         = 4 / 8 = 0.5
        """
        closes = pd.Series([100.0, 102.0, 101.0, 103.0, 102.0, 104.0])
        er = engineer._compute_efficiency_ratio(closes, n=5)
        assert er.iloc[-1] == pytest.approx(0.5, abs=1e-9)
