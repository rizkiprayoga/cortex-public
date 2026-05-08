"""
GBM train/serve parity (spec §3 invariant #8).

The flat-row feature builder must produce IDENTICAL output between:
- training path (vectorized over a DataFrame), and
- serving path (single-row computation on a sliding window).

Mirrors test_train_serve_parity.py for the GBM track. The contract is
guaranteed by construction (build_feature_row delegates to
build_features) but parametrized indices catch silent regressions if
that delegation is ever broken.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synth_h4_with_regime(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    df = pd.DataFrame(
        {
            "open":   100 + rng.standard_normal(n).cumsum() * 0.1,
            "high":   100 + rng.standard_normal(n).cumsum() * 0.1 + 0.5,
            "low":    100 + rng.standard_normal(n).cumsum() * 0.1 - 0.5,
            "close":  100 + rng.standard_normal(n).cumsum() * 0.1,
            "volume": rng.integers(100, 1000, n).astype(float),
            "regime_0": rng.choice([0, 1], n).astype(float),
            "regime_1": rng.choice([0, 1], n).astype(float),
            "regime_2": rng.choice([0, 1], n).astype(float),
            "regime_3": rng.choice([0, 1], n).astype(float),
            "regime_4": rng.choice([0, 1], n).astype(float),
            "regime_probability": rng.uniform(0.2, 1.0, n),
        },
        index=idx,
    )
    return df


@pytest.mark.parametrize("t", [80, 100, 150, 250])
def test_serving_row_equals_training_row(t):
    """Serving path at index t must equal training row t bit-for-bit."""
    from src.brain.gbm.gbm_features import build_features, build_feature_row

    df = _synth_h4_with_regime()
    full = build_features(df)
    row = build_feature_row(df.iloc[: t + 1])

    expected = full.iloc[t].to_dict()
    for k, v_serve in row.items():
        v_train = expected[k]
        if isinstance(v_serve, float) and np.isnan(v_serve):
            assert np.isnan(v_train), f"row {t} col {k}: serve=NaN train={v_train}"
        else:
            assert abs(v_serve - v_train) < 1e-9, (
                f"row {t} col {k}: serve={v_serve} train={v_train}"
            )


def test_column_order_stable():
    """build_features must produce a stable column order across runs.

    LightGBM's feature names depend on the column order of the input
    matrix at training time; if it shifts between train and serve, the
    inference will read wrong values.
    """
    from src.brain.gbm.gbm_features import build_features

    df1 = _synth_h4_with_regime(200)
    df2 = _synth_h4_with_regime(200)
    cols1 = list(build_features(df1).columns)
    cols2 = list(build_features(df2).columns)
    assert cols1 == cols2, "column order is not deterministic"
