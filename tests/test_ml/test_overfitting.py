"""Unit tests for src/ml/overfitting.py — Deflated Sharpe + Sharpe stability."""
from __future__ import annotations

import math

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────
# Deflated Sharpe
# ──────────────────────────────────────────────────────────────────────────

def test_deflated_sharpe_identical_to_observed_when_single_trial():
    """With n_trials=1, the "expected max under null" term collapses to 0,
    so DSR reduces to a straight z-test on the observed Sharpe."""
    from src.ml.overfitting import deflated_sharpe_ratio

    # Observed SR=2 over 100 bars of returns, normal-ish
    dsr = deflated_sharpe_ratio(
        sharpe=2.0, n_obs=100, skewness=0.0, kurtosis=3.0, n_trials=1,
    )
    assert 0.9 < dsr <= 1.0   # very significant (norm.cdf may saturate at 1.0)


def test_deflated_sharpe_penalizes_many_trials():
    """A Sharpe of 2 from 1 trial should look significant; from 1000 trials,
    much less so — DSR should drop toward 0."""
    from src.ml.overfitting import deflated_sharpe_ratio

    dsr1 = deflated_sharpe_ratio(2.0, 100, 0.0, 3.0, n_trials=1)
    dsr_many = deflated_sharpe_ratio(2.0, 100, 0.0, 3.0, n_trials=1000)
    assert dsr_many < dsr1
    assert 0.0 <= dsr_many <= 1.0


def test_deflated_sharpe_low_for_below_null_sharpe():
    """A Sharpe lower than expected-under-null should produce DSR ~ 0."""
    from src.ml.overfitting import deflated_sharpe_ratio

    # Very weak Sharpe + many trials → expected-max-under-null dominates
    dsr = deflated_sharpe_ratio(0.1, 50, 0.0, 3.0, n_trials=100)
    assert dsr < 0.5   # doesn't clear the null bar


def test_deflated_sharpe_handles_zero_obs_gracefully():
    from src.ml.overfitting import deflated_sharpe_ratio
    assert deflated_sharpe_ratio(1.0, 0, 0.0, 3.0, n_trials=1) == 0.0
    assert deflated_sharpe_ratio(1.0, 1, 0.0, 3.0, n_trials=1) == 0.0


# ──────────────────────────────────────────────────────────────────────────
# Sharpe stability
# ──────────────────────────────────────────────────────────────────────────

def test_sharpe_stability_one_when_all_windows_positive():
    """Strictly increasing equity → every sub-window is positive-Sharpe → 1.0."""
    from src.ml.overfitting import sharpe_stability

    rng = np.random.default_rng(1)
    returns = np.abs(rng.normal(0.01, 0.005, 200))   # always positive
    stab = sharpe_stability(returns, n_windows=5)
    assert stab == pytest.approx(1.0)


def test_sharpe_stability_zero_when_all_windows_negative():
    from src.ml.overfitting import sharpe_stability

    rng = np.random.default_rng(1)
    returns = -np.abs(rng.normal(0.01, 0.005, 200))   # always negative
    stab = sharpe_stability(returns, n_windows=5)
    assert stab == pytest.approx(0.0)


def test_sharpe_stability_mixed():
    """Half positive Sharpes + half negative → around 0.5."""
    from src.ml.overfitting import sharpe_stability

    rng = np.random.default_rng(7)
    pos = np.abs(rng.normal(0.01, 0.005, 100))
    neg = -np.abs(rng.normal(0.01, 0.005, 100))
    returns = np.concatenate([pos, neg])
    stab = sharpe_stability(returns, n_windows=4)
    assert 0.4 <= stab <= 0.6


def test_sharpe_stability_too_few_obs_returns_nan_sentinel():
    """Less than 2 × n_windows observations → can't split meaningfully."""
    from src.ml.overfitting import sharpe_stability
    assert sharpe_stability(np.array([1.0, 2.0, 3.0]), n_windows=4) == 0.0


def test_sharpe_stability_zero_std_window_treated_as_nonpositive():
    """A window with zero variance has undefined Sharpe — treat as not positive."""
    from src.ml.overfitting import sharpe_stability
    zeros = np.zeros(100)
    assert sharpe_stability(zeros, n_windows=5) == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────
# Helper: compute Sharpe from a return series
# ──────────────────────────────────────────────────────────────────────────

def test_sharpe_from_returns_matches_definition():
    """Sanity on the Sharpe helper used internally."""
    from src.ml.overfitting import sharpe_from_returns

    rets = np.array([0.01, 0.02, -0.005, 0.015, 0.0])
    mean = rets.mean()
    std = rets.std(ddof=1)
    expected = mean / std * math.sqrt(len(rets))
    assert sharpe_from_returns(rets) == pytest.approx(expected)


def test_sharpe_from_returns_zero_std_returns_zero():
    from src.ml.overfitting import sharpe_from_returns
    assert sharpe_from_returns(np.array([0.01, 0.01, 0.01])) == 0.0
