"""
Tests for src/brain/gbm/gbm_model.py — GBMPredictor wrapper.

Spec §4.2: GBMPredictor exposes the same .predict(features) -> directional
score interface as LSTMPricePredictor so signal_combiner can route to
either model without branching past load time.

Single head type per anchor 7: 3-class multiclass classifier on TB labels
mapped {-1 -> 0, 0 -> 1, +1 -> 2}. Inference output is the directional
score P(class +1) - P(class -1) ∈ [-1.0, 1.0].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def trained_gbm(tmp_path):
    """Train a tiny GBM on synthetic 3-class TB labels and save it."""
    import lightgbm as lgb
    from src.brain.gbm.gbm_model import GBMPredictor

    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 5))
    # 3-class labels (TB convention -1=0, 0=1, +1=2)
    y = rng.integers(0, 3, 300)
    booster = lgb.train(
        params={
            "objective": "multiclass",
            "num_class": 3,
            "verbose": -1,
            "num_leaves": 7,
        },
        train_set=lgb.Dataset(X, label=y),
        num_boost_round=20,
    )
    feature_names = [f"f{i}" for i in range(5)]
    pkl = tmp_path / "gbm_TEST.pkl"
    GBMPredictor.save(booster, feature_names, pkl)
    return pkl, feature_names


def test_save_then_load_roundtrip(trained_gbm):
    pkl, feature_names = trained_gbm
    from src.brain.gbm.gbm_model import GBMPredictor

    pred = GBMPredictor.load(pkl)
    assert pred.feature_names == feature_names
    assert pred.num_class == 3


def test_predict_returns_scalar_directional_score(trained_gbm):
    """predict(symbol, {...}) should return a single float directional
    score P(class +1) - P(class -1) where -1->idx 0, 0->idx 1, +1->idx 2."""
    pkl, feature_names = trained_gbm
    from src.brain.gbm.gbm_model import GBMPredictor

    pred = GBMPredictor.load(pkl)
    feature_row = {name: 0.5 for name in feature_names}
    score = pred.predict("TEST", feature_row)
    assert isinstance(score, float), f"got {type(score)}"
    assert -1.0 <= score <= 1.0


def test_predict_dataframe_returns_array(trained_gbm):
    """predict(symbol, DataFrame) returns ndarray of length N>1."""
    pkl, feature_names = trained_gbm
    from src.brain.gbm.gbm_model import GBMPredictor

    pred = GBMPredictor.load(pkl)
    df = pd.DataFrame({n: np.zeros(10) for n in feature_names})
    scores = pred.predict("TEST", df)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (10,)
    assert ((scores >= -1.0) & (scores <= 1.0)).all()


def test_predict_1d_ndarray_returns_scalar(trained_gbm):
    """predict(symbol, 1D ndarray) is the contract backtest_full uses
    when slicing one row out of the GBM feature matrix per H4 bar.
    Must return a Python float (signal_combiner expects a scalar)."""
    pkl, feature_names = trained_gbm
    from src.brain.gbm.gbm_model import GBMPredictor

    pred = GBMPredictor.load(pkl)
    row = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    score = pred.predict("TEST", row)
    assert isinstance(score, float), f"got {type(score)}"
    assert -1.0 <= score <= 1.0


def test_predict_pandas_series_returns_scalar(trained_gbm):
    """pd.Series with feature_names index is convenient for the call
    site that does ``feat_df.iloc[h4_idx]`` — verify it works."""
    pkl, feature_names = trained_gbm
    from src.brain.gbm.gbm_model import GBMPredictor

    pred = GBMPredictor.load(pkl)
    series = pd.Series({n: 0.25 for n in feature_names})
    score = pred.predict("TEST", series)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_predict_symbol_arg_is_ignored(trained_gbm):
    """The symbol arg is purely interface symmetry with LSTM. A GBM
    predictor instance is per-file; passing different symbols must
    return identical scores given identical features."""
    pkl, feature_names = trained_gbm
    from src.brain.gbm.gbm_model import GBMPredictor

    pred = GBMPredictor.load(pkl)
    row = {name: 0.5 for name in feature_names}
    score_a = pred.predict("XAUUSD", row)
    score_b = pred.predict("EURUSD", row)
    assert score_a == score_b


def test_load_missing_file_raises(tmp_path):
    from src.brain.gbm.gbm_model import GBMPredictor

    with pytest.raises(FileNotFoundError):
        GBMPredictor.load(tmp_path / "does_not_exist.pkl")
