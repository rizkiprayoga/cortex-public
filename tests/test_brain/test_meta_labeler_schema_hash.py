"""
the model bake-off spec §3 invariant #12 — meta-labeler schema hash.

The hash is the contract: same hash means same feature schema, so
signal_combiner can refuse to load on mismatch (silent train/serve drift
would otherwise be invisible). These tests pin down the contract so
'just one extra GBM-specific feature' creep can't slip through.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest


def test_compute_schema_hash_deterministic():
    """Same input → same output across calls."""
    from src.brain.meta_labeler_features import compute_schema_hash

    h1 = compute_schema_hash(["a", "b", "c"])
    h2 = compute_schema_hash(["a", "b", "c"])
    assert h1 == h2


def test_compute_schema_hash_order_independent():
    """Permutations of the same set produce the same hash — sorting
    happens internally so callers don't have to think about column order
    at the API surface."""
    from src.brain.meta_labeler_features import compute_schema_hash

    assert compute_schema_hash(["a", "b", "c"]) == compute_schema_hash(["c", "a", "b"])


def test_compute_schema_hash_changes_on_different_schema():
    """Different feature sets must produce different hashes."""
    from src.brain.meta_labeler_features import compute_schema_hash

    h1 = compute_schema_hash(["a", "b", "c"])
    h3 = compute_schema_hash(["a", "b", "d"])
    assert h1 != h3


def test_compute_schema_hash_returns_short_hex():
    """16-char hex prefix (collision risk negligible for our schema count)."""
    from src.brain.meta_labeler_features import compute_schema_hash

    h = compute_schema_hash(["a", "b", "c"])
    assert isinstance(h, str)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_expected_schema_hash_matches_canonical_features():
    """The pre-computed EXPECTED_SCHEMA_HASH constant must match the
    hash of EXPECTED_FEATURE_NAMES — sanity check for the import."""
    from src.brain.meta_labeler_features import (
        EXPECTED_FEATURE_NAMES, EXPECTED_SCHEMA_HASH, compute_schema_hash,
    )

    assert compute_schema_hash(EXPECTED_FEATURE_NAMES) == EXPECTED_SCHEMA_HASH


def test_expected_feature_names_has_26_columns():
    """5 base + 17 fundamentals + 4 exec = 26 — schema lock (Phase 2B Opt 2)."""
    from src.brain.meta_labeler_features import (
        BASE_FEATURE_NAMES, FUNDAMENTAL_FEATURE_NAMES, EXEC_FEATURE_NAMES,
        EXPECTED_FEATURE_NAMES,
    )
    assert len(BASE_FEATURE_NAMES) == 5
    assert len(FUNDAMENTAL_FEATURE_NAMES) == 17
    assert len(EXEC_FEATURE_NAMES) == 4
    assert len(EXPECTED_FEATURE_NAMES) == 26


def test_legacy_22_schema_hash_in_accepted():
    """The pre-Phase-2B 22-feature schema hash stays in ACCEPTED so old
    bundles still load during the rollout window."""
    from src.brain.meta_labeler_features import (
        ACCEPTED_SCHEMA_HASHES, LEGACY_22_SCHEMA_HASH,
    )
    assert LEGACY_22_SCHEMA_HASH in ACCEPTED_SCHEMA_HASHES


def test_saved_artifact_includes_schema_hash(tmp_path, monkeypatch):
    """save_meta_labeler must bake the EXPECTED_SCHEMA_HASH into the bundle."""
    from src.brain.meta_labeler_features import EXPECTED_SCHEMA_HASH
    from src.ml import meta_labeler as ml_mod

    monkeypatch.setattr(
        ml_mod, "MODEL_PATH_TEMPLATE",
        str(tmp_path / "meta_{symbol}_{primary}.pkl"),
    )

    # Build a tiny but real LightGBM classifier so save/joblib round-trips.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "entry_time": [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=h)
                       for h in range(80)],
        "combined_score": rng.uniform(-1, 1, 80),
        "regime_label": rng.choice(["Bull", "Bear", "Neutral"], 80),
        "direction": rng.choice(["buy", "sell"], 80),
        "pnl": rng.choice([100.0, -50.0], 80),
    })
    clf, _ = ml_mod.train_meta_labeler("XAUUSD", df, val_fraction=0.2)
    path = ml_mod.save_meta_labeler(clf, "XAUUSD", threshold=0.5, primary="lstm")

    bundle = joblib.load(path)
    assert bundle.get("feature_schema_hash") == EXPECTED_SCHEMA_HASH
    assert bundle.get("primary_kind") == "lstm"
    assert tuple(bundle["feature_names"]) == ml_mod.EXPECTED_FEATURE_NAMES
