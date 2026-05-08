"""Tests for HMMRegimeClassifier."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.brain.hmm_regime import HMMRegimeClassifier, RegimeResult, REGIME_LABELS


class TestHMMRegimeClassifier:

    def setup_method(self):
        self.clf = HMMRegimeClassifier(n_components=5, n_init=2)
        # Synthetic feature matrix: 200 bars × 5 features
        np.random.seed(42)
        self.features = np.random.randn(200, 5).astype(np.float32)

    def test_train_and_predict_returns_regime_result(self):
        """After training, predict() should return a valid RegimeResult."""
        self.clf.train("XAUUSD", self.features)
        result = self.clf.predict("XAUUSD", self.features[-60:])
        assert isinstance(result, RegimeResult)
        assert result.symbol == "XAUUSD"
        assert result.regime_label in REGIME_LABELS.values()
        assert 0.0 <= result.state_probability <= 1.0
        assert result.position_multiplier in [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_all_probabilities_sum_to_one(self):
        """Posterior probabilities should sum to 1."""
        self.clf.train("XAUUSD", self.features)
        result = self.clf.predict("XAUUSD", self.features[-60:])
        assert abs(result.all_probabilities.sum() - 1.0) < 1e-5

    def test_predict_raises_if_not_trained(self):
        """predict() should raise RuntimeError if model has not been trained."""
        with pytest.raises((RuntimeError, KeyError)):
            self.clf.predict("XAUUSD", self.features[-60:])

    def test_multipliers_are_symmetric(self):
        """
        Symmetric MULTIPLIERS (the trading universe, 2026-04-29). All 10 live pairs trade
        bidirectionally, so Crash deserves the same conviction (1.0) as
        Euphoria, and Bear matches Bull. Asserts the mirror invariant so a
        future drift back to the asymmetric long-only ramp is caught here.
        """
        m = HMMRegimeClassifier.MULTIPLIERS
        assert m[0] == 1.0  # Crash
        assert m[1] == 0.75  # Bear
        assert m[2] == 0.50  # Neutral
        assert m[3] == 0.75  # Bull
        assert m[4] == 1.0  # Euphoria
        assert m[0] == m[4]  # mirror at extremes
        assert m[1] == m[3]  # mirror at directional

    def test_save_and_load(self, tmp_path, monkeypatch):
        """Saved model should reload and produce the same predictions."""
        monkeypatch.setattr(
            "src.brain.hmm_regime.MODEL_PATH",
            tmp_path / "hmm_{symbol}.pkl"
        )
        self.clf.train("XAUUSD", self.features)
        self.clf.save("XAUUSD")

        clf2 = HMMRegimeClassifier()
        assert clf2.load("XAUUSD")
        r1 = self.clf.predict("XAUUSD", self.features[-60:])
        r2 = clf2.predict("XAUUSD", self.features[-60:])
        assert r1.regime_index == r2.regime_index


class TestSortStatesByMeanReturn:
    """
    Tests for the state-canonicalization helper. This is the Wave 4
    fix for HMM categorical-collapse risk: regardless of the order in
    which hmmlearn returns hidden states, the canonical index 0 must
    always be the lowest-mean-return state (Crash) and index n-1 must
    be the highest (Euphoria). Without this, consecutive retrains
    produce state 0=Bull on Monday, state 0=Crash on Tuesday, and the
    MULTIPLIERS table silently breaks.
    """

    def setup_method(self):
        self.clf = HMMRegimeClassifier(n_components=5, n_init=1)

    def test_sorts_strictly_increasing_means(self):
        """Lowest-mean state gets canonical 0; highest gets canonical n-1."""
        # raw state 0 = highest return, raw 1 = lowest, raw 2 = middling
        means = np.array([
            [0.020, 0.1, 0.2],
            [-0.030, 0.1, 0.2],
            [0.001, 0.1, 0.2],
        ])
        label_map = self.clf._sort_states_by_mean_return(means, log_return_col=0)
        # raw 1 (lowest) → canonical 0, raw 2 (middle) → canonical 1, raw 0 (highest) → canonical 2
        assert label_map == {1: 0, 2: 1, 0: 2}

    def test_stable_across_shuffled_state_order(self):
        """
        The same underlying states — presented to hmmlearn in a different
        order — must canonicalize to the same labels. This is the core
        stability property that prevents categorical collapse across
        weekly retrains.
        """
        # Canonical-order state means
        canonical_means = np.array([
            [-0.05, 0.1, 0.2],  # Crash
            [-0.01, 0.1, 0.2],  # Bear
            [0.00,  0.1, 0.2],  # Neutral
            [0.01,  0.1, 0.2],  # Bull
            [0.05,  0.1, 0.2],  # Euphoria
        ])

        # Run 1: states presented in canonical order
        map1 = self.clf._sort_states_by_mean_return(canonical_means)
        # After applying map1, raw-state i should land at canonical i.
        remapped1 = sorted(map1.items(), key=lambda kv: kv[1])
        ordered_canonical_means1 = canonical_means[[r for r, _ in remapped1], 0]

        # Run 2: same states in a shuffled order
        permutation = np.array([3, 0, 4, 1, 2])  # arbitrary shuffle
        shuffled_means = canonical_means[permutation]
        map2 = self.clf._sort_states_by_mean_return(shuffled_means)
        remapped2 = sorted(map2.items(), key=lambda kv: kv[1])
        ordered_canonical_means2 = shuffled_means[[r for r, _ in remapped2], 0]

        # Both runs must produce the same ordered-mean-return vector —
        # that's the invariant the downstream MULTIPLIERS table depends on.
        assert np.allclose(ordered_canonical_means1, ordered_canonical_means2)

    def test_rejects_non_2d_means(self):
        """Means must be 2D (n_components, n_features)."""
        with pytest.raises(ValueError, match="2D"):
            self.clf._sort_states_by_mean_return(np.array([0.0, 0.1, 0.2]))

    def test_rejects_bad_log_return_col(self):
        """log_return_col must be in-bounds for the feature dimension."""
        means = np.zeros((5, 3))
        with pytest.raises(ValueError, match="out of range"):
            self.clf._sort_states_by_mean_return(means, log_return_col=5)
