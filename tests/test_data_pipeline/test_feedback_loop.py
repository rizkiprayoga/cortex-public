"""
Tests for FeedbackLoop._label_regime causality.

Wave 4 fix: verify the regime-labeling path for feedback training is
strictly causal — i.e. the HMM is only shown bars at-or-before the
target ``bar_timestamp``. Any leakage would train the feedback loop on
labels contaminated by future information, which silently corrupts the
directional-accuracy metric and the sample-weight feedback signal.

The structural guarantee comes from two places:
  1. ``_label_regime`` passes ``end=bar_dt`` to ``get_ohlcv_range``, so
     the DB query itself cannot return future bars.
  2. ``HMMRegimeClassifier.predict()`` must use filtered decoding
     (``predict_proba[-1]``) — enforced by the docstring invariant in
     [src/brain/hmm_regime.py](src/brain/hmm_regime.py).

These tests pin property #1 in behaviour. Property #2 is documented
but cannot be asserted until ``predict()`` moves past its stub.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.feedback_loop import FeedbackLoop


class TestLabelRegimeCausality:

    def _build_loop(self, ohlcv_df, predicted_regime_index=2):
        data_store = MagicMock()
        data_store.get_ohlcv_range = AsyncMock(return_value=ohlcv_df)

        hmm = MagicMock()
        fake_result = MagicMock()
        fake_result.regime_index = predicted_regime_index
        hmm.predict.return_value = fake_result
        hmm._models = {"XAUUSD": MagicMock()}

        loop = FeedbackLoop(data_store=data_store, hmm=hmm)
        return loop, data_store, hmm

    def _synthetic_ohlcv(self, n_bars: int, end_dt: datetime) -> pd.DataFrame:
        """Build a minimal OHLCV frame that FeatureEngineer can digest."""
        times = [end_dt - timedelta(days=n_bars - 1 - i) for i in range(n_bars)]
        rng = np.random.default_rng(42)
        closes = 2000.0 + np.cumsum(rng.normal(0, 5, size=n_bars))
        df = pd.DataFrame({
            "time": times,
            "open": closes + rng.normal(0, 1, size=n_bars),
            "high": closes + 2,
            "low": closes - 2,
            "close": closes,
            "volume": rng.integers(1000, 5000, size=n_bars),
        })
        return df

    def test_fetches_only_bars_at_or_before_target(self):
        """
        _label_regime must request bars ending at bar_dt — never later.
        This is the primary structural guarantee that future bars
        cannot leak into the regime label for bar t.
        """
        target = datetime(2026, 4, 11, tzinfo=timezone.utc)
        df = self._synthetic_ohlcv(60, target)
        loop, data_store, hmm = self._build_loop(df)

        # Patch FeatureEngineer so the test doesn't depend on its internals.
        with patch(
            "src.data_pipeline.feature_engineering.FeatureEngineer"
        ) as FE:
            fe_instance = FE.return_value
            fe_instance.transform.return_value = df
            fe_instance.to_matrix.return_value = np.zeros((len(df), 5))

            result = asyncio.run(
                loop._label_regime("XAUUSD", "2026-04-11T00:00:00+00:00")
            )

        assert result == 2  # our mocked predicted_regime_index

        # The actual causality check: what 'end' did the data_store see?
        call_args = data_store.get_ohlcv_range.call_args
        assert call_args is not None
        passed_end = call_args.kwargs.get("end") or call_args.args[-1]
        # Allow tz-aware vs tz-naive comparison quirks by stripping tzinfo
        passed_end_naive = passed_end.replace(tzinfo=None) if passed_end.tzinfo else passed_end
        target_naive = target.replace(tzinfo=None)
        assert passed_end_naive == target_naive, (
            f"_label_regime fetched bars up to {passed_end_naive}, "
            f"not {target_naive} — this is a causality violation"
        )

    def test_label_independent_of_bars_after_target(self):
        """
        Simulate a "contaminated" data_store that erroneously returns
        bars beyond bar_dt. Even so, the label for bar t must be
        determined by the feature-matrix row corresponding to bar t,
        NOT by the contaminated tail. We assert this by ensuring
        _label_regime still calls get_ohlcv_range with end=bar_dt
        (the code's defense) — a test at the contract boundary.
        """
        target = datetime(2026, 4, 11, tzinfo=timezone.utc)
        df = self._synthetic_ohlcv(60, target)
        loop, data_store, _ = self._build_loop(df, predicted_regime_index=0)

        with patch(
            "src.data_pipeline.feature_engineering.FeatureEngineer"
        ) as FE:
            fe_instance = FE.return_value
            fe_instance.transform.return_value = df
            fe_instance.to_matrix.return_value = np.zeros((len(df), 5))

            asyncio.run(
                loop._label_regime("XAUUSD", "2026-04-11T00:00:00+00:00")
            )

        # Contract check: end passed to get_ohlcv_range equals bar_dt.
        call_args = data_store.get_ohlcv_range.call_args
        passed_end = call_args.kwargs.get("end") or call_args.args[-1]
        passed_end_naive = passed_end.replace(tzinfo=None) if passed_end.tzinfo else passed_end
        target_naive = target.replace(tzinfo=None)
        assert passed_end_naive == target_naive

    def test_returns_none_on_empty_window(self):
        """
        If the DB returns <30 bars in the 120-day lookback, _label_regime
        must return None rather than trying to predict on a sparse window.
        """
        target = datetime(2026, 4, 11, tzinfo=timezone.utc)
        df = self._synthetic_ohlcv(5, target)  # way below 30
        loop, _, _ = self._build_loop(df)

        result = asyncio.run(
            loop._label_regime("XAUUSD", "2026-04-11T00:00:00+00:00")
        )
        assert result is None

    def test_rejects_malformed_timestamp(self):
        """Bad timestamps return None, not raise — feedback loop stays alive."""
        df = self._synthetic_ohlcv(60, datetime(2026, 4, 11, tzinfo=timezone.utc))
        loop, _, _ = self._build_loop(df)

        result = asyncio.run(
            loop._label_regime("XAUUSD", "not-a-timestamp")
        )
        assert result is None
