"""
the model bake-off data integrity gate — spec §3.

CI gate that BLOCKS bake-off result acceptance if any structural
invariant fails. If one of these tests fails, fix the data path before
running Sprint 6 — never accept a "result" produced by a broken pipeline.

Test names map 1:1 to invariant numbers from
docs/superpowers/specs/2026-04-25-phase-a-bakeoff-design.md §3.
The remaining 6 invariants (#4, #5, #7, #8, #9, #10) are enforced by
their own focused test files (e.g. test_train_serve_parity_gbm.py for
#8, test_meta_labeler_schema_hash.py for #12).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"


# ---- Invariant #1: True UTC + DB-direct OHLCV reads in both train scripts --
def test_inv01_train_gbm_db_direct():
    """train_gbm.py must use get_historical_db_only — never bare get_historical."""
    src = (SCRIPTS / "train_gbm.py").read_text(encoding="utf-8")
    assert "get_historical_db_only" in src, (
        "INV#1: train_gbm.py must reference get_historical_db_only — "
        "the only blessed MT5-free OHLCV reader."
    )
    # Bare get_historical( without _db_only is the MT5 API path.
    bare = re.findall(r"(?<!_db_only)\bget_historical\s*\(", src)
    assert not bare, (
        f"INV#1: train_gbm.py uses MT5 API path: {bare}. "
        f"Use feed.get_historical_db_only(...) instead."
    )


# ---- Invariant #2: Same training window enforced via CLI flags ----
def test_inv02_explicit_train_window_args():
    """Both LSTM and GBM scripts expose the same TVT CLI surface so the
    bake-off harness invokes them with identical windows (spec §3 #2)."""
    required_flags = ("--train-start", "--train-end",
                      "--val-start", "--val-end",
                      "--test-start", "--test-end")
    for script in ("train_deep_learning.py", "train_gbm.py"):
        src = (SCRIPTS / script).read_text(encoding="utf-8")
        missing = [f for f in required_flags if f not in src]
        assert not missing, (
            f"INV#2: {script} missing TVT CLI flags: {missing}"
        )


# ---- Invariant #3: TB-label parity across tracks ----
def test_inv03_tb_label_parity():
    """Both train scripts must compute TB labels via the SAME helper —
    FeatureEngineer.compute_triple_barrier_labels — so labels are
    bit-identical across tracks (spec §3 #3).

    Plan deviation: original wording said "must read existing
    triple_barrier_labels table". There is no such table in the
    codebase — TB labels are computed from H4 OHLCV inline, the
    same way for both LSTM and GBM. The actual parity guarantee is
    that both scripts use the SAME function.
    """
    for script in ("train_deep_learning.py", "train_gbm.py"):
        src = (SCRIPTS / script).read_text(encoding="utf-8")
        assert "compute_triple_barrier_labels" in src, (
            f"INV#3: {script} must call "
            f"FeatureEngineer.compute_triple_barrier_labels for TB-label "
            f"parity across tracks."
        )


# ---- Invariant #6: Lag/rolling lookahead safety in GBM features ----
def test_inv06_gbm_features_lookahead_safe():
    """src/brain/gbm/gbm_features.py must use .shift(k>=1) for lags and
    .rolling(W).agg().shift(1) for rolling stats (current bar excluded
    from its own statistic).
    """
    src = (PROJECT_ROOT / "src" / "brain" / "gbm" / "gbm_features.py").read_text("utf-8")

    # Forbidden: negative shift would leak future data.
    bad_shifts = re.findall(r"\.shift\(\s*-\s*\d+\s*\)", src)
    assert not bad_shifts, f"INV#6: negative shift found: {bad_shifts}"

    # Required: rolling().mean() / rolling().std() must be followed
    # by .shift(1) so the current bar isn't in its own window.
    rolling_aggs = re.findall(
        r"\.rolling\([^)]+\)\.(?:mean|std)\(\)(?:\.shift\(\s*1\s*\))?",
        src,
    )
    assert rolling_aggs, "INV#6: no rolling().mean()/std() calls found"
    bad = [r for r in rolling_aggs if not re.search(r"\.shift\(\s*1\s*\)$", r)]
    assert not bad, (
        f"INV#6: rolling stat missing .shift(1) — current bar leaks "
        f"into its own statistic. Offenders: {bad}"
    )


# ---- Invariant #11: Meta-labeler uses read_feature_store_safe only ----
def test_inv11_meta_labeler_safe_reads_only():
    """train_meta_labeler.py + src/ml/meta_labeler.py must NEVER call
    bare read_feature_store( — only read_feature_store_safe (which
    subtracts release_lag_hours per source). Spec §3 invariant #11.
    """
    targets = (
        SCRIPTS / "train_meta_labeler.py",
        PROJECT_ROOT / "src" / "ml" / "meta_labeler.py",
    )
    for path in targets:
        src = path.read_text(encoding="utf-8")
        bare = re.findall(r"(?<!_safe)\bread_feature_store\s*\(", src)
        assert not bare, (
            f"INV#11: {path.name} contains bare read_feature_store( call(s): "
            f"{bare}. Replace with read_feature_store_safe."
        )


# ---- Invariant #12: Meta-labeler bundle carries schema hash ----
def test_inv12_schema_hash_present():
    """src/ml/meta_labeler.py must compute and persist
    feature_schema_hash on every artifact (spec §3 #12)."""
    src = (PROJECT_ROOT / "src" / "ml" / "meta_labeler.py").read_text("utf-8")
    assert "feature_schema_hash" in src, (
        "INV#12: meta_labeler.py does not write feature_schema_hash — "
        "signal_combiner can't validate train/serve schema parity."
    )
    # And the constant must come from the single source of truth.
    assert "EXPECTED_SCHEMA_HASH" in src, (
        "INV#12: meta_labeler.py must import EXPECTED_SCHEMA_HASH from "
        "src.brain.meta_labeler_features (single source of truth)."
    )


# ---- Invariant #13: No magic-numbered hyperparams in train scripts ----
def test_inv13_tuning_spaces_referenced():
    """Both train scripts must reference config/tuning_spaces.yaml so
    Optuna search spaces and trial-0 defaults stay versioned (anchor 6)."""
    for script in ("train_deep_learning.py", "train_gbm.py"):
        src = (SCRIPTS / script).read_text("utf-8")
        assert "tuning_spaces.yaml" in src, (
            f"INV#13: {script} must reference config/tuning_spaces.yaml"
        )


# ---- Invariant #14: Test window NEVER touched during training ----
def test_inv14_test_window_excluded_from_optuna():
    """Both scripts must (a) accept --test-start CLI flag and (b) have
    a runtime guard that raises ValueError if training data extends
    into [test_start, ...]. Static check only — runtime check in the
    slow test_tuning_split_invariant.py.
    """
    for script in ("train_deep_learning.py", "train_gbm.py"):
        src = (SCRIPTS / script).read_text("utf-8")
        assert "--test-start" in src, f"INV#14: {script} missing --test-start flag"
        # Runtime guard: explicit raise on test-window leak. The phrase
        # "invariant #14" appears in both scripts' guard messages.
        assert (
            "invariant #14" in src.lower()
            or "test_start_ts" in src
        ), (
            f"INV#14: {script} missing the runtime test-window guard "
            f"(should raise ValueError if h4_ohlcv.index.max() >= test_start)."
        )


# Invariants #4 (HMM held constant), #5 (signal_combiner branch-free),
# #7 (regime as feature on prior bars), #8 (train/serve parity), #9 (DSR
# decision floor), #10 (model artifact dispatch by model_kind) are
# enforced by their own dedicated test files.
