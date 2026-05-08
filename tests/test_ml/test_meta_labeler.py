"""Unit tests for src/ml/meta_labeler.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_trades(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Build a toy backtest_trades-shaped DataFrame where pnl is correlated
    with combined_score so LightGBM actually has a signal to learn."""
    rng = np.random.default_rng(seed)
    ts_start = pd.Timestamp("2023-01-01T00:00:00+00:00")
    entry_time = [ts_start + pd.Timedelta(hours=int(h)) for h in range(n)]
    combined_score = rng.uniform(-1.0, 1.0, n)
    # Winners more likely when combined_score is large-magnitude positive
    win_prob = 1.0 / (1.0 + np.exp(-3.0 * combined_score))
    win = rng.random(n) < win_prob
    pnl = np.where(win, rng.uniform(50, 200, n), rng.uniform(-200, -50, n))
    regimes = rng.choice(["Bull", "Bear", "Neutral"], n)
    directions = rng.choice(["buy", "sell"], n)
    return pd.DataFrame({
        "entry_time": entry_time,
        "combined_score": combined_score,
        "regime_label": regimes,
        "direction": directions,
        "pnl": pnl,
    })


def test_build_feature_matrix_shape_and_encoding():
    from src.ml.meta_labeler import build_feature_matrix, FEATURE_NAMES

    df = pd.DataFrame({
        "entry_time": ["2024-01-01T00:00:00", "2024-01-01T05:00:00"],
        "combined_score": [0.8, -0.3],
        "regime_label": ["Bull", "Bear"],
        "direction": ["buy", "sell"],
    })
    X = build_feature_matrix(df)
    assert X.shape == (2, len(FEATURE_NAMES))
    # Bull is index 3 in REGIME_VALUES = ("Crash","Bear","Neutral","Bull","Euphoria")
    assert X[0, 1] == 3
    # Bear is index 1
    assert X[1, 1] == 1
    # buy is 0, sell is 1
    assert X[0, 2] == 0
    assert X[1, 2] == 1


def test_build_labels_binary_from_pnl():
    from src.ml.meta_labeler import build_labels
    df = pd.DataFrame({"pnl": [100.0, -50.0, 0.0, 25.0]})
    labels = build_labels(df)
    # 0.0 is NOT > 0, so it's a loss-label (0)
    assert list(labels) == [1, 0, 0, 1]


def test_train_raises_on_insufficient_trades():
    from src.ml.meta_labeler import train_meta_labeler
    with pytest.raises(ValueError, match="too few trades"):
        train_meta_labeler("XAUUSD", _synthetic_trades(n=10))


def test_train_end_to_end_learns_the_signal():
    """With a synthetic dataset where pnl is deliberately correlated
    with combined_score, the trained labeler should beat random (0.5)
    on accuracy and produce a coverage < 1.0 at threshold 0.5."""
    from src.ml.meta_labeler import train_meta_labeler

    df = _synthetic_trades(n=500)
    clf, result = train_meta_labeler("XAUUSD", df, val_fraction=0.2, threshold=0.5)

    assert result.symbol == "XAUUSD"
    assert result.n_train + result.n_val == 500
    assert 0.55 <= result.val_accuracy <= 1.0, \
        f"labeler should learn something, got {result.val_accuracy}"
    assert 0.0 < result.coverage_at_default_threshold < 1.0, \
        f"labeler should filter some but not all trades, got coverage {result.coverage_at_default_threshold}"


def test_predict_proba_stays_in_unit_interval():
    from src.ml.meta_labeler import train_meta_labeler, predict_proba

    df = _synthetic_trades(n=300)
    clf, _ = train_meta_labeler("XAUUSD", df)
    bundle = {"model": clf, "threshold": 0.5, "feature_names": None}
    proba = predict_proba(
        bundle, combined_score=0.8, regime_label="Bull",
        direction="buy", hour_of_day=10, day_of_week=2,
    )
    assert 0.0 <= proba <= 1.0


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    """Save a trained labeler then load it back; predictions should match."""
    from src.ml import meta_labeler as ml_mod

    # Sprint 4: path template now carries both {symbol} and {primary}.
    monkeypatch.setattr(
        ml_mod, "MODEL_PATH_TEMPLATE",
        str(tmp_path / "meta_{symbol}_{primary}.pkl"),
    )
    monkeypatch.setattr(
        ml_mod, "LEGACY_MODEL_PATH_TEMPLATE",
        str(tmp_path / "meta_{symbol}.pkl"),
    )

    df = _synthetic_trades(n=300)
    clf, _ = ml_mod.train_meta_labeler("XAUUSD", df)
    path = ml_mod.save_meta_labeler(clf, "XAUUSD", threshold=0.55, primary="lstm")
    assert path.exists()

    loaded = ml_mod.load_meta_labeler("XAUUSD", primary="lstm")
    assert loaded is not None
    assert loaded["threshold"] == 0.55
    # Sprint 4: bundle now carries the schema hash + primary tag
    assert loaded.get("primary_kind") == "lstm"
    assert "feature_schema_hash" in loaded
    # Predictions on a fresh batch must agree
    X = ml_mod.build_feature_matrix(df.head(5))
    orig = clf.predict_proba(X)[:, 1]
    back = loaded["model"].predict_proba(X)[:, 1]
    assert np.allclose(orig, back)


def test_load_meta_labeler_missing_file_returns_none(tmp_path, monkeypatch):
    from src.ml import meta_labeler as ml_mod
    monkeypatch.setattr(
        ml_mod, "MODEL_PATH_TEMPLATE",
        str(tmp_path / "nope_{symbol}_{primary}.pkl"),
    )
    monkeypatch.setattr(
        ml_mod, "LEGACY_MODEL_PATH_TEMPLATE",
        str(tmp_path / "nope_{symbol}.pkl"),
    )
    assert ml_mod.load_meta_labeler("ZZZZZZ", primary="lstm") is None


def test_load_meta_labeler_falls_back_to_legacy_path(tmp_path, monkeypatch):
    """Sprint 4 back-compat: pre-existing meta_labeler_{symbol}.pkl bundles
    (without _primary suffix) should still load when no suffixed bundle
    exists. The schema-hash check in signal_combiner is what gates use,
    not the file-name path."""
    import joblib
    from src.ml import meta_labeler as ml_mod

    monkeypatch.setattr(
        ml_mod, "MODEL_PATH_TEMPLATE",
        str(tmp_path / "meta_{symbol}_{primary}.pkl"),
    )
    monkeypatch.setattr(
        ml_mod, "LEGACY_MODEL_PATH_TEMPLATE",
        str(tmp_path / "meta_{symbol}.pkl"),
    )
    # Drop a legacy-named bundle directly
    legacy_path = tmp_path / "meta_XAUUSD.pkl"
    joblib.dump({"model": "stub", "threshold": 0.42}, legacy_path)
    loaded = ml_mod.load_meta_labeler("XAUUSD", primary="lstm")
    assert loaded is not None
    assert loaded["threshold"] == 0.42
