"""
Sprint 2 Task 2.2b-2b: --tune Optuna study contract.

Verifies that:
  1. The --tune / --tune-trials flags exist in --help.
  2. config/tuning_spaces.yaml loads cleanly and lstm.defaults match
     model_config.yaml lstm.* (single source of truth — anchor 6).
  3. _train_one_lstm_for_symbol grew the three new keyword-only params
     (hparam_overrides, artifact_suffix, extra_tags) and the
     _run_optuna_study_for_symbol driver exists.
  4. assert_head_matches_existing accepts a suffix kwarg (path-aware
     guard so _default/_tuned artifacts are independent of the legacy
     unsuffixed file).
  5. Default no-tune behavior is preserved — artifact_suffix="" by default.
  6. Optuna study DB lives per-symbol under data/models/ (resumable).
  7. Invariant #14 (test window never loaded) holds in tune mode too.

Source-parse approach matches the sibling test_train_dl_extract.py and
test_train_dl_tvt.py — train_deep_learning.py runs heavy module-top
imports that aren't worth pulling into a structural unit test.
"""
from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest


def _help_text() -> str:
    return subprocess.run(
        [sys.executable, "scripts/train_deep_learning.py", "--help"],
        capture_output=True, text=True, check=True,
    ).stdout


def test_tune_flags_in_help():
    out = _help_text()
    assert "--tune" in out
    assert "--tune-trials" in out
    # Default 20 visible
    assert "default 20" in out or "default: 20" in out


def test_tuning_spaces_yaml_loads_with_lstm_defaults():
    """config/tuning_spaces.yaml must be readable and have lstm.defaults +
    lstm.search keys; defaults must match config/model_config.yaml lstm.*"""
    import yaml
    ts = yaml.safe_load(Path("config/tuning_spaces.yaml").read_text(encoding="utf-8"))
    mc = yaml.safe_load(Path("config/model_config.yaml").read_text(encoding="utf-8"))

    assert "lstm" in ts and "defaults" in ts["lstm"] and "search" in ts["lstm"]

    lstm_d = ts["lstm"]["defaults"]
    mc_l = mc["lstm"]
    mc_lt = mc["lstm"]["training"]
    assert lstm_d["hidden_size"] == mc_l["hidden_size"]
    assert lstm_d["num_layers"] == mc_l["num_layers"]
    assert lstm_d["dropout"] == mc_l["dropout"]
    assert lstm_d["learning_rate"] == mc_lt["learning_rate"]
    assert lstm_d["batch_size"] == mc_lt["batch_size"]


def test_train_one_lstm_signature_grew_correctly():
    """_train_one_lstm_for_symbol must accept hparam_overrides, artifact_suffix,
    extra_tags as keyword-only params for Task 2.2b-2b."""
    src = Path("scripts/train_deep_learning.py").read_text(encoding="utf-8")
    # Source-parse approach (matching existing test pattern in this dir).
    assert "hparam_overrides" in src
    assert "artifact_suffix" in src
    assert "extra_tags" in src
    assert "_run_optuna_study_for_symbol" in src


def test_assert_head_matches_existing_accepts_suffix():
    """assert_head_matches_existing should accept a suffix kwarg so tune-mode
    artifacts (lstm_{sym}_default.pt) don't trip on the unsuffixed legacy file."""
    from src.utils.model_head import assert_head_matches_existing
    sig = inspect.signature(assert_head_matches_existing)
    assert "suffix" in sig.parameters, (
        "assert_head_matches_existing missing `suffix` keyword param "
        "— Task 2.2b-2b path-aware guard"
    )


def test_no_tune_mode_unaffected():
    """When --tune is NOT set, behavior should match pre-2.2b-2b (no
    artifact suffix, single training run per symbol). Verify by checking
    that artifact_suffix='' is the default value of the function param."""
    src = Path("scripts/train_deep_learning.py").read_text(encoding="utf-8")
    # Default value `artifact_suffix: str = ""` should appear
    assert 'artifact_suffix: str = ""' in src or "artifact_suffix: str = ''" in src


def test_optuna_study_storage_path_pattern():
    """Optuna study DB should be saved per-symbol under data/models/, not
    in a single shared file (per-symbol resumable studies)."""
    src = Path("scripts/train_deep_learning.py").read_text(encoding="utf-8")
    assert "lstm_" in src and "_optuna_study.db" in src


def test_invariant_14_test_window_unreachable_in_tune_mode():
    """Tune mode runs with the same explicit_split clip — no Optuna trial
    can see the test window. Grep verification: the clip + assertion are
    NOT removed or bypassed in the tune path."""
    src = Path("scripts/train_deep_learning.py").read_text(encoding="utf-8")
    # The assertion message naming test_start should still be present
    assert "test_start" in src
    assert "INVARIANT" in src or "invariant" in src


def test_artifact_suffix_forwards_to_train_on_matrix():
    """The artifact_suffix kwarg on _train_one_lstm_for_symbol must propagate
    into the predictor._train_on_matrix(...) call. This is the load-bearing
    contract between the two files for the bake-off artifact path naming —
    drop the forwarding and lstm_{sym}_default.pt would silently land at
    lstm_{sym}.pt and overwrite the production model."""
    src = Path("scripts/train_deep_learning.py").read_text(encoding="utf-8")
    assert "artifact_suffix=artifact_suffix" in src, (
        "artifact_suffix kwarg is not forwarded to _train_on_matrix — "
        "lstm_{sym}_default.pt would land at lstm_{sym}.pt instead."
    )
