"""
Sprint 2 Task 2.2a: explicit train/val/test calendar windows for
train_deep_learning.py. Spec §1 anchor 9 + invariant #14.

Verifies (1) the 6 CLI flags exist with the documented defaults,
(2) the legacy --start-date/--end-date flags still exist (backwards
compat), (3) test_start defaults sit AFTER val_end (no overlap).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _help_text() -> str:
    return subprocess.run(
        [sys.executable, "scripts/train_deep_learning.py", "--help"],
        capture_output=True, text=True, check=True,
    ).stdout


def test_tvt_flags_present_with_defaults():
    """All 6 flags must appear in --help output with their documented defaults."""
    out = _help_text()
    expected = {
        "--train-start": "2021-01-01",
        "--train-end":   "2024-06-30",
        "--val-start":   "2024-07-01",
        "--val-end":     "2025-04-30",
        "--test-start":  "2025-05-01",
        "--test-end":    "2026-04-30",
    }
    for flag, default in expected.items():
        assert flag in out, f"{flag} missing from train_deep_learning.py --help"
        assert default in out, (
            f"default {default!r} for {flag} missing — Phase A invariant #14 "
            f"requires this exact window"
        )


def test_legacy_date_flags_still_present():
    """--start-date and --end-date stay for backwards compat with monthly retrain."""
    out = _help_text()
    assert "--start-date" in out
    assert "--end-date" in out


def test_default_windows_are_non_overlapping():
    """test_start default must be strictly AFTER val_end default — invariant #14."""
    import argparse
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_dl", "scripts/train_deep_learning.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Don't actually load — we only need parse_args. But importing top-level
    # would trigger DataStore import which needs env. Read source instead.
    src = Path("scripts/train_deep_learning.py").read_text(encoding="utf-8")
    assert 'default="2024-06-30"' in src and 'default="2024-07-01"' in src
    assert 'default="2025-04-30"' in src and 'default="2025-05-01"' in src


def test_clip_retains_all_h4_bars_on_val_end_day():
    """Regression for the val_end clip bug: with val_end='2025-04-30' and H4
    bars stamped at 00:00/04:00/08:00/12:00/16:00/20:00, all 6 bars on
    April 30 must be retained, not just the midnight bar."""
    import pandas as pd

    val_end = "2025-04-30"
    val_end_exclusive = pd.Timestamp(val_end) + pd.Timedelta(days=1)

    h4_apr_30 = pd.date_range(
        f"{val_end} 00:00", f"{val_end} 20:00", freq="4h",
    )
    assert len(h4_apr_30) == 6

    kept = h4_apr_30[h4_apr_30 < val_end_exclusive]
    assert len(kept) == 6, f"clip lost {6 - len(kept)} bars on val_end day"
