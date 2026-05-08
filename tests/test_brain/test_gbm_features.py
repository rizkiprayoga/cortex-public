"""
Tests for src/brain/gbm/gbm_features.py — the model bake-off GBM track.

Spec §3 invariants:
  #6  lag/rolling features must be lookahead-safe (current bar EXCLUDED
      from its own statistic)
  #7  regime feature comes from prior bars only (HMM is run on D1 prior
      to current H4 bar — enforced upstream by inject_regime_features)
  #8  build_features (vectorized training path) and build_feature_row
      (single-row serving path) must produce bit-identical output for
      the same input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_h4_with_regime() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=200, freq="4h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "open":   100 + rng.standard_normal(200).cumsum(),
            "high":   100 + rng.standard_normal(200).cumsum() + 0.5,
            "low":    100 + rng.standard_normal(200).cumsum() - 0.5,
            "close":  100 + rng.standard_normal(200).cumsum(),
            "volume": rng.integers(100, 1000, 200).astype(float),
            # 6 regime features (5 one-hot + 1 prob) — what
            # inject_regime_features produces upstream.
            "regime_0": rng.choice([0, 1], 200).astype(float),
            "regime_1": rng.choice([0, 1], 200).astype(float),
            "regime_2": rng.choice([0, 1], 200).astype(float),
            "regime_3": rng.choice([0, 1], 200).astype(float),
            "regime_4": rng.choice([0, 1], 200).astype(float),
            "regime_probability": rng.uniform(0.2, 1.0, 200),
        },
        index=idx,
    )
    return df


def test_lag_1_equals_prior_close(sample_h4_with_regime):
    """close_lag_1 at row t MUST equal close at row t-1 (invariant #6)."""
    from src.brain.gbm.gbm_features import build_features

    out = build_features(sample_h4_with_regime)

    for t in range(1, 50):
        ts = sample_h4_with_regime.index[t]
        prior_close = sample_h4_with_regime["close"].iloc[t - 1]
        assert out.loc[ts, "close_lag_1"] == prior_close, (
            f"close_lag_1 mismatch at t={t}: got {out.loc[ts, 'close_lag_1']}, "
            f"expected {prior_close}"
        )


def test_rolling_mean_excludes_current_row(sample_h4_with_regime):
    """close_mean_20 at row t MUST equal mean of close[t-20:t] EXCLUSIVE of t.

    The .shift(1) on the rolling output is what enforces this — without
    it, the bar's own close would be in its own 20-bar mean, which is a
    classic lookahead leak that inflates train metrics.
    """
    from src.brain.gbm.gbm_features import build_features

    out = build_features(sample_h4_with_regime)

    for t in range(20, 50):
        ts = sample_h4_with_regime.index[t]
        expected = sample_h4_with_regime["close"].iloc[t - 20:t].mean()
        actual = out.loc[ts, "close_mean_20"]
        assert abs(actual - expected) < 1e-9, (
            f"close_mean_20 mismatch at t={t}: got {actual}, expected {expected}"
        )


def test_regime_features_pass_through(sample_h4_with_regime):
    """The 6 regime cols (regime_0..4 + regime_probability) carry through unchanged."""
    from src.brain.gbm.gbm_features import build_features

    out = build_features(sample_h4_with_regime)

    for col in ("regime_0", "regime_1", "regime_2", "regime_3", "regime_4",
                "regime_probability"):
        assert col in out.columns, f"{col} missing from output"
        assert (out[col].values == sample_h4_with_regime[col].values).all(), (
            f"{col} values modified during feature build"
        )


def test_regime_cross_feature_present(sample_h4_with_regime):
    """At least one regime × technical feature cross should be in the output.

    Trees benefit from explicit regime × signal interactions (NeurIPS '23
    quant lit consensus).
    """
    from src.brain.gbm.gbm_features import build_features

    out = build_features(sample_h4_with_regime)

    cross_cols = [c for c in out.columns if "_x_regime_" in c or "_regime_x_" in c]
    assert len(cross_cols) >= 1, "no regime × feature cross columns found"


def test_no_nan_in_output_after_warmup(sample_h4_with_regime):
    """Output should be NaN-free after the warmup period (60 bars — the
    longest lag horizon). Warmup rows themselves can be NaN."""
    from src.brain.gbm.gbm_features import build_features

    out = build_features(sample_h4_with_regime).iloc[60:]
    nan_cols = out.columns[out.isna().any()].tolist()
    assert not nan_cols, f"NaN found in columns after warmup: {nan_cols}"


def test_train_serve_parity_single_row(sample_h4_with_regime):
    """build_features (vectorized) must agree with build_feature_row
    (single-row serving) at any t past the warmup. Spec §3 invariant #8."""
    from src.brain.gbm.gbm_features import build_features, build_feature_row

    full = build_features(sample_h4_with_regime)
    t = 100
    row = build_feature_row(sample_h4_with_regime.iloc[: t + 1])
    expected = full.iloc[t].to_dict()

    for k, v in row.items():
        if isinstance(v, float) and np.isnan(v):
            assert np.isnan(expected[k]), f"{k}: serving={v} train={expected[k]}"
        else:
            assert abs(v - expected[k]) < 1e-9, (
                f"{k}: serving={v} train={expected[k]}"
            )
