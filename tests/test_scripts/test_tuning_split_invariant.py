"""
the model bake-off spec §3 invariant #14 — runtime check that Optuna
trials never see test-window dates.

Pairs with the static-source check in test_bakeoff_data_integrity.py
(test_inv14_test_window_excluded_from_optuna). The static check
verifies the CLI flags + raise statement exist; this runtime check
actually trains a 1-trial study and inspects the saved
training_dist.json artifact to confirm latest_train < test_start.

Marked ``slow`` because a full training pass takes minutes; opt in
via ``pytest -m slow``. Default suite skips it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_optuna_trial_data_excludes_test_window():
    """Run a 1-trial GBM Optuna study and verify the saved training
    distribution proves invariant #14 was honored at runtime.

    The training_dist.json sidecar carries ``latest_train_ts`` — must
    fall strictly before the configured ``--test-start``.
    """
    test_start = "2025-05-01"
    proc = subprocess.run(
        [
            sys.executable, "scripts/train_gbm.py",
            "--symbols", "XAUUSD",
            "--tune", "--tune-trials", "1",
            "--train-start", "2021-01-01",
            "--train-end", "2024-06-30",
            "--val-start", "2024-07-01",
            "--val-end", "2025-04-30",
            "--test-start", test_start,
            "--test-end", "2026-04-30",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"Training failed (rc={proc.returncode}). "
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )

    dist_path = PROJECT_ROOT / "data" / "models" / "gbm_XAUUSD_default.training_dist.json"
    assert dist_path.exists(), (
        f"Training succeeded but {dist_path.name} was not produced — "
        f"check _save_artifact in scripts/train_gbm.py."
    )
    dist = json.loads(dist_path.read_text(encoding="utf-8"))
    latest = dist["latest_train_ts"]
    assert latest < test_start, (
        f"INV #14 RUNTIME VIOLATION: latest_train_ts={latest!r} reaches "
        f"into test window (test_start={test_start!r})."
    )
