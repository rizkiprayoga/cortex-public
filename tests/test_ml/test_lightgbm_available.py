"""Smoke test: LightGBM is installed and a toy classifier trains."""
from __future__ import annotations


def test_lightgbm_importable():
    import lightgbm as lgb
    assert hasattr(lgb, "LGBMClassifier")


def test_lightgbm_trains_toy_binary_classifier():
    """Sanity-check that fit/predict works on a 2-feature 2-class toy set."""
    import numpy as np
    from lightgbm import LGBMClassifier

    rng = np.random.default_rng(42)
    x_pos = rng.normal(loc=1.0, scale=0.3, size=(50, 2))
    x_neg = rng.normal(loc=-1.0, scale=0.3, size=(50, 2))
    X = np.vstack([x_pos, x_neg])
    y = np.array([1] * 50 + [0] * 50)

    clf = LGBMClassifier(n_estimators=20, verbosity=-1)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert (preds == y).mean() > 0.9   # lazy but enough
