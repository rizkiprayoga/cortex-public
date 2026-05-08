"""Unit tests for src/ml/drift.py — PSI + KS + training-distribution snapshot."""
from __future__ import annotations

import numpy as np
import pytest


# ────────────────────────────────────────────────────────────────────────────
# PSI
# ────────────────────────────────────────────────────────────────────────────

def test_psi_zero_for_identical_distributions():
    from src.ml.drift import psi

    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 1000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_small_for_same_distribution_different_samples():
    """Drawing two samples from the same N(0,1) should PSI very close to 0."""
    from src.ml.drift import psi

    rng = np.random.default_rng(42)
    train = rng.normal(0, 1, 5000)
    current = rng.normal(0, 1, 5000)
    # Typical PSI for matched distributions well under the 0.10 "no shift" threshold
    assert psi(train, current) < 0.05


def test_psi_large_for_mean_shift():
    """A 2-sigma mean shift should register clearly above the 0.25 threshold."""
    from src.ml.drift import psi

    rng = np.random.default_rng(42)
    train = rng.normal(0, 1, 5000)
    shifted = rng.normal(2, 1, 5000)   # mean shift
    assert psi(train, shifted) > 0.25


def test_psi_handles_zero_bins_gracefully():
    """PSI should not return inf/NaN when a bin is empty in current."""
    from src.ml.drift import psi

    rng = np.random.default_rng(42)
    train = rng.normal(0, 1, 1000)
    # Current skewed entirely to the right → leftmost bins are empty
    current = rng.normal(5, 0.1, 1000)
    val = psi(train, current)
    assert np.isfinite(val)
    assert val > 0.25   # definitely drifted


# ────────────────────────────────────────────────────────────────────────────
# KS statistic
# ────────────────────────────────────────────────────────────────────────────

def test_ks_zero_for_identical_arrays():
    from src.ml.drift import ks_statistic
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 500)
    assert ks_statistic(x, x) == pytest.approx(0.0, abs=1e-9)


def test_ks_detects_mean_shift():
    from src.ml.drift import ks_statistic
    rng = np.random.default_rng(7)
    train = rng.normal(0, 1, 2000)
    shifted = rng.normal(1.0, 1, 2000)
    # Classic rule of thumb: 1-sigma shift yields KS ≈ 0.3-0.4
    assert ks_statistic(train, shifted) > 0.2


# ────────────────────────────────────────────────────────────────────────────
# Training distribution save/load
# ────────────────────────────────────────────────────────────────────────────

def test_training_dist_save_and_load_round_trip(tmp_path):
    from src.ml.drift import save_training_distribution, load_training_distribution

    rng = np.random.default_rng(1)
    matrix = rng.normal(0, 1, size=(500, 4))
    feature_names = ("a", "b", "c", "d")

    path = save_training_distribution(
        tmp_path / "lstm_XAUUSD.training_dist.json",
        symbol="XAUUSD", timeframe="H4",
        feature_matrix=matrix, feature_names=feature_names,
    )
    assert path.exists()

    loaded = load_training_distribution(path)
    assert loaded["symbol"] == "XAUUSD"
    assert loaded["timeframe"] == "H4"
    assert loaded["n_samples"] == 500
    assert tuple(loaded["feature_names"]) == feature_names
    # Each feature must have the expected summary keys
    for fname in feature_names:
        stats = loaded["features"][fname]
        assert {"mean", "std", "q10", "q50", "q90", "samples"}.issubset(stats)
        assert len(stats["samples"]) > 0


def test_load_training_distribution_missing_file_returns_none(tmp_path):
    from src.ml.drift import load_training_distribution
    assert load_training_distribution(tmp_path / "nope.json") is None


# ────────────────────────────────────────────────────────────────────────────
# Combined drift-score shape
# ────────────────────────────────────────────────────────────────────────────

def test_compute_drift_returns_psi_and_ks_per_feature(tmp_path):
    from src.ml.drift import (
        save_training_distribution, load_training_distribution,
        compute_drift,
    )

    rng = np.random.default_rng(42)
    train = rng.normal(0, 1, size=(2000, 3))
    feature_names = ("x", "y", "z")
    path = save_training_distribution(
        tmp_path / "t.json",
        symbol="TEST", timeframe="H4",
        feature_matrix=train, feature_names=feature_names,
    )
    training_dist = load_training_distribution(path)

    # Current: 2 features match, 1 feature is mean-shifted
    current = rng.normal(0, 1, size=(500, 3))
    current[:, 1] += 2.0   # feature 'y' shifts by +2

    result = compute_drift(training_dist, current, feature_names)
    assert set(result["per_feature"].keys()) == set(feature_names)
    # 'y' should have the highest PSI of the three
    psis = {k: v["psi"] for k, v in result["per_feature"].items()}
    assert psis["y"] > psis["x"]
    assert psis["y"] > psis["z"]
    # Aggregate fields
    assert result["psi_max"] == max(psis.values())
    assert result["ks_max"] > 0
    assert result["n_current_samples"] == 500


# ────────────────────────────────────────────────────────────────────────────
# Schema migration + categorical auto-skip (post-2026-04-22 drift-fix)
# ────────────────────────────────────────────────────────────────────────────

def test_save_writes_current_schema_version(tmp_path):
    from src.ml.drift import (
        save_training_distribution, load_training_distribution,
        CURRENT_SCHEMA_VERSION,
    )

    rng = np.random.default_rng(0)
    matrix = rng.normal(0, 1, size=(100, 2))
    path = save_training_distribution(
        tmp_path / "t.json",
        symbol="T", timeframe="H4",
        feature_matrix=matrix, feature_names=("a", "b"),
    )
    loaded = load_training_distribution(path)
    assert loaded["schema_version"] == CURRENT_SCHEMA_VERSION


def test_load_rejects_legacy_v1_schema(tmp_path, caplog):
    """v1 snapshots stored z-scored samples with a batch-local scale.

    Drift metrics computed against them are unreliable, so the loader
    now returns None and logs the re-bootstrap instruction.
    """
    import json
    from src.ml.drift import load_training_distribution

    path = tmp_path / "v1.json"
    payload = {
        "schema_version": 1,
        "symbol": "T", "timeframe": "H4",
        "n_samples": 100,
        "feature_names": ["a"],
        "features": {"a": {"mean": 0, "std": 1, "q10": 0, "q50": 0, "q90": 0,
                           "samples": [0.0] * 50}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with caplog.at_level("WARNING"):
        assert load_training_distribution(path) is None
    assert any("schema v1" in rec.message for rec in caplog.records)


def test_compute_drift_skips_categorical_features(tmp_path):
    """One-hot / cyclic features have few unique training values; PSI/KS
    on them explode the moment the current window lands in a single
    category. Those features must be auto-skipped, and the skip list
    surfaced to the caller."""
    from src.ml.drift import (
        save_training_distribution, load_training_distribution,
        compute_drift,
    )

    rng = np.random.default_rng(7)
    # Column 0: continuous; column 1: binary one-hot; column 2: 12-value cyclic
    n = 2000
    train = np.column_stack([
        rng.normal(0, 1, n),
        rng.integers(0, 2, n).astype(float),
        np.cos(2 * np.pi * rng.integers(0, 12, n) / 12),
    ])
    feature_names = ("continuous", "is_london", "month_cos")
    path = save_training_distribution(
        tmp_path / "t.json",
        symbol="T", timeframe="H4",
        feature_matrix=train, feature_names=feature_names,
    )
    training_dist = load_training_distribution(path)

    # Current window: same continuous, but categorical columns collapse
    # into a single value (the drift-window pathology).
    m = 200
    current = np.column_stack([
        rng.normal(0, 1, m),
        np.ones(m),                 # always-London window
        np.full(m, np.cos(2 * np.pi * 4 / 12)),   # all April
    ])

    result = compute_drift(training_dist, current, feature_names)
    # Continuous feature stays in the per-feature map
    assert "continuous" in result["per_feature"]
    # Categorical-like features are auto-skipped, not scored
    assert "is_london" not in result["per_feature"]
    assert "month_cos" not in result["per_feature"]
    skipped_names = {name for name, _ in result["skipped"]}
    assert {"is_london", "month_cos"}.issubset(skipped_names)
    # The absurd PSI blowups are gone, and psi_max reflects the
    # continuous feature only
    assert result["psi_max"] < 0.1
    assert result["ks_max"] < 0.2


def test_compute_drift_reports_skipped_reasons(tmp_path):
    from src.ml.drift import (
        save_training_distribution, load_training_distribution,
        compute_drift,
    )
    rng = np.random.default_rng(9)
    train = rng.normal(0, 1, size=(500, 2))
    path = save_training_distribution(
        tmp_path / "t.json",
        symbol="T", timeframe="H4",
        feature_matrix=train, feature_names=("present", "also_present"),
    )
    training_dist = load_training_distribution(path)

    # Caller passes an extra feature name with no training samples
    current = rng.normal(0, 1, size=(100, 3))
    result = compute_drift(
        training_dist, current,
        feature_names=("present", "also_present", "missing"),
    )
    reasons = dict(result["skipped"])
    assert reasons.get("missing") == "no_training_samples"
