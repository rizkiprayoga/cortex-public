"""Unit tests for run_bakeoff.py classification logic.

The Windows PyTorch teardown crash (rc=0xC0000409) is reproducible
on Python 3.13 + PyTorch 2.11. The orchestrator must override its
pass/fail summary using artifact-on-disk + success-marker checks so
the cosmetic crash doesn't false-flag a successful LSTM cell during
the 36-48h bake-off run.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from scripts.run_bakeoff import (
    SUCCESS_MARKER_LSTM,
    WINDOWS_TEARDOWN_RC,
    _classify,
    _lstm_extra_flags,
)


def _lstm_artifacts(symbol: str, models_dir: Path) -> list[Path]:
    return [
        models_dir / f"lstm_{symbol}_default.pt",
        models_dir / f"lstm_{symbol}_tuned.pt",
        models_dir / f"lstm_{symbol}_default.training_dist.json",
        models_dir / f"lstm_{symbol}_tuned.training_dist.json",
    ]


def _gbm_artifacts(symbol: str, models_dir: Path) -> list[Path]:
    return [
        models_dir / f"gbm_{symbol}_default.pkl",
        models_dir / f"gbm_{symbol}_tuned.pkl",
        models_dir / f"gbm_{symbol}_default.training_dist.json",
        models_dir / f"gbm_{symbol}_tuned.training_dist.json",
    ]


def _touch_artifacts(paths: list[Path], started_before: float) -> None:
    """Create artifacts with mtime >= started_before so the freshness
    check inside _classify treats them as products of the current run."""
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 16)
        # Ensure mtime is strictly after started_before by 1s.
        future = started_before + 1.0
        import os
        os.utime(p, (future, future))


@pytest.fixture
def fake_models_dir(tmp_path, monkeypatch):
    """Redirect MODELS_DIR to a tmp_path so _classify sees fresh paths."""
    fake = tmp_path / "models"
    fake.mkdir()
    import scripts.run_bakeoff as mod
    monkeypatch.setattr(mod, "MODELS_DIR", fake)
    return fake


def test_rc0_is_ok(fake_models_dir):
    """rc=0 short-circuits to ok regardless of artifacts."""
    result = {
        "script": "train_deep_learning.py",
        "symbol": "XAUUSD",
        "returncode": 0,
        "elapsed_s": 80.0,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    out = _classify(result, started_before=time.time() - 10)
    assert out["effective_status"] == "ok"


def test_lstm_teardown_with_marker_and_artifacts_is_ok_teardown(fake_models_dir):
    """The cosmetic Windows DLL-detach crash on LSTM trainer."""
    started = time.time() - 10
    _touch_artifacts(_lstm_artifacts("XAUUSD", fake_models_dir), started)
    result = {
        "script": "train_deep_learning.py",
        "symbol": "XAUUSD",
        "returncode": WINDOWS_TEARDOWN_RC,
        "elapsed_s": 80.0,
        "stdout_tail": f"...epoch=23\n{SUCCESS_MARKER_LSTM}\n",
        "stderr_tail": "",
    }
    out = _classify(result, started_before=started)
    assert out["effective_status"] == "ok_teardown"
    assert "Windows DLL teardown" in out["effective_reason"]


def test_lstm_teardown_without_marker_is_fail(fake_models_dir):
    """Missing success marker means training likely crashed mid-run."""
    started = time.time() - 10
    _touch_artifacts(_lstm_artifacts("XAUUSD", fake_models_dir), started)
    result = {
        "script": "train_deep_learning.py",
        "symbol": "XAUUSD",
        "returncode": WINDOWS_TEARDOWN_RC,
        "elapsed_s": 12.0,
        "stdout_tail": "...epoch=2\n",  # no marker
        "stderr_tail": "",
    }
    out = _classify(result, started_before=started)
    assert out["effective_status"] == "fail"


def test_lstm_teardown_without_artifacts_is_fail(fake_models_dir):
    """Marker without artifacts means partial save; do not whitewash."""
    started = time.time() - 10
    # No _touch_artifacts — directory is empty.
    result = {
        "script": "train_deep_learning.py",
        "symbol": "XAUUSD",
        "returncode": WINDOWS_TEARDOWN_RC,
        "elapsed_s": 80.0,
        "stdout_tail": SUCCESS_MARKER_LSTM,
        "stderr_tail": "",
    }
    out = _classify(result, started_before=started)
    assert out["effective_status"] == "fail"


def test_lstm_teardown_with_stale_artifacts_is_fail(fake_models_dir):
    """Artifacts predating the run shouldn't count — they're from a
    previous run and the current one actually failed early."""
    started = time.time()
    paths = _lstm_artifacts("XAUUSD", fake_models_dir)
    # Touch BEFORE started: simulate prior-run artifacts.
    import os
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 16)
        old = started - 3600
        os.utime(p, (old, old))
    result = {
        "script": "train_deep_learning.py",
        "symbol": "XAUUSD",
        "returncode": WINDOWS_TEARDOWN_RC,
        "elapsed_s": 5.0,
        "stdout_tail": SUCCESS_MARKER_LSTM,
        "stderr_tail": "",
    }
    out = _classify(result, started_before=started)
    assert out["effective_status"] == "fail"


def test_gbm_teardown_rc_is_fail(fake_models_dir):
    """The teardown override is LSTM-only. GBM with the same rc is
    a real failure (LightGBM doesn't have the PyTorch teardown bug)."""
    started = time.time() - 10
    _touch_artifacts(_gbm_artifacts("XAUUSD", fake_models_dir), started)
    result = {
        "script": "train_gbm.py",
        "symbol": "XAUUSD",
        "returncode": WINDOWS_TEARDOWN_RC,
        "elapsed_s": 5.0,
        "stdout_tail": "saved tuned artifact",
        "stderr_tail": "",
    }
    out = _classify(result, started_before=started)
    assert out["effective_status"] == "fail"


def test_lstm_extra_flags_softmax_symbol_gets_triple_barrier():
    """USDJPY + EURUSD trained with softmax head — train_deep_learning.py
    crashes with 'softmax head requires --triple-barrier' if the flag
    is missing. The orchestrator must add it so the bake-off doesn't
    silently fail those 2 of 4 LSTM cells halfway through a 48h run."""
    flags = _lstm_extra_flags("USDJPY")
    assert "--triple-barrier" in flags
    flags = _lstm_extra_flags("EURUSD")
    assert "--triple-barrier" in flags


def test_lstm_extra_flags_regression_symbol_omits_triple_barrier():
    """XAUUSD + USDCAD trained with regression head — passing
    --triple-barrier would change the training target away from what
    the live model expects. Must NOT be in flags."""
    flags = _lstm_extra_flags("XAUUSD")
    assert "--triple-barrier" not in flags
    flags = _lstm_extra_flags("USDCAD")
    assert "--triple-barrier" not in flags


def test_other_nonzero_rc_is_fail(fake_models_dir):
    """Any rc that isn't 0 and isn't WINDOWS_TEARDOWN_RC is a real
    fail, even with artifacts on disk (the artifacts may be stale)."""
    started = time.time() - 10
    _touch_artifacts(_lstm_artifacts("XAUUSD", fake_models_dir), started)
    result = {
        "script": "train_deep_learning.py",
        "symbol": "XAUUSD",
        "returncode": 1,
        "elapsed_s": 80.0,
        "stdout_tail": SUCCESS_MARKER_LSTM,
        "stderr_tail": "",
    }
    out = _classify(result, started_before=started)
    assert out["effective_status"] == "fail"
