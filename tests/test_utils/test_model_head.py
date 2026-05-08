"""
Tests for ``src/utils/model_head.py`` — per-symbol head resolution and
shape-preservation guard. These pin the invariant that a retrain cannot
silently flip a symbol's LSTM architecture (the May-1 scheduled-retrain
bug caught by the T-3 dry-run on 2026-04-18).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import torch

from src.utils.model_head import (
    HeadMismatchError,
    assert_head_matches_existing,
    resolve_softmax_for_symbol,
)


def _write_settings(tmp_path: Path, per_symbol: dict) -> Path:
    """Write a minimal settings.yaml with the given per-symbol block."""
    body = {"strategy": {"per_symbol_params": per_symbol}}
    import yaml
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def _write_lstm_pt(path: Path, out_dim: int, hidden: int = 64) -> None:
    """Fake a trained LSTM state-dict with just fc2.weight shape."""
    sd = {"fc2.weight": torch.zeros((out_dim, hidden))}
    torch.save(sd, path)


class TestResolveSoftmaxForSymbol:
    def test_config_softmax(self, tmp_path):
        sp = _write_settings(tmp_path, {"EURUSD": {"model_head": "softmax"}})
        assert resolve_softmax_for_symbol("EURUSD", settings_path=sp) is True

    def test_config_regression(self, tmp_path):
        sp = _write_settings(tmp_path, {"XAUUSD": {"model_head": "regression"}})
        assert resolve_softmax_for_symbol("XAUUSD", settings_path=sp) is False

    def test_cli_softmax_overrides_config(self, tmp_path):
        sp = _write_settings(tmp_path, {"XAUUSD": {"model_head": "regression"}})
        assert resolve_softmax_for_symbol(
            "XAUUSD", cli_override="softmax", settings_path=sp,
        ) is True

    def test_cli_regression_overrides_config(self, tmp_path):
        sp = _write_settings(tmp_path, {"ETHUSD": {"model_head": "softmax"}})
        assert resolve_softmax_for_symbol(
            "ETHUSD", cli_override="regression", settings_path=sp,
        ) is False

    def test_missing_config_raises(self, tmp_path):
        sp = _write_settings(tmp_path, {"XAUUSD": {}})  # no model_head
        with pytest.raises(ValueError, match="missing or invalid"):
            resolve_softmax_for_symbol("EURUSD", settings_path=sp)

    def test_invalid_head_value_raises(self, tmp_path):
        sp = _write_settings(tmp_path, {"XAUUSD": {"model_head": "linear"}})
        with pytest.raises(ValueError, match="got 'linear'"):
            resolve_softmax_for_symbol("XAUUSD", settings_path=sp)

    def test_bogus_cli_override_raises(self, tmp_path):
        sp = _write_settings(tmp_path, {"XAUUSD": {"model_head": "regression"}})
        with pytest.raises(ValueError, match="cli_override must be"):
            resolve_softmax_for_symbol(
                "XAUUSD", cli_override="bogus", settings_path=sp,
            )


class TestAssertHeadMatchesExisting:
    def test_no_existing_model_is_noop(self, tmp_path):
        # Empty models_dir — first-training case; must not raise.
        assert_head_matches_existing(
            "EURUSD", want_softmax=True, models_dir=tmp_path,
        )

    def test_matching_softmax_ok(self, tmp_path):
        _write_lstm_pt(tmp_path / "lstm_EURUSD.pt", out_dim=3)
        assert_head_matches_existing(
            "EURUSD", want_softmax=True, models_dir=tmp_path,
        )

    def test_matching_regression_ok(self, tmp_path):
        _write_lstm_pt(tmp_path / "lstm_XAUUSD.pt", out_dim=1)
        assert_head_matches_existing(
            "XAUUSD", want_softmax=False, models_dir=tmp_path,
        )

    def test_softmax_to_regression_mismatch_raises(self, tmp_path):
        # The May-1 bug: existing softmax(3), config says regression(1).
        _write_lstm_pt(tmp_path / "lstm_EURUSD.pt", out_dim=3)
        with pytest.raises(HeadMismatchError, match="softmax\\(3\\).*regression\\(1\\)"):
            assert_head_matches_existing(
                "EURUSD", want_softmax=False, models_dir=tmp_path,
            )

    def test_regression_to_softmax_mismatch_raises(self, tmp_path):
        _write_lstm_pt(tmp_path / "lstm_XAUUSD.pt", out_dim=1)
        with pytest.raises(HeadMismatchError, match="regression\\(1\\).*softmax\\(3\\)"):
            assert_head_matches_existing(
                "XAUUSD", want_softmax=True, models_dir=tmp_path,
            )

    def test_allow_change_bypasses_guard(self, tmp_path):
        _write_lstm_pt(tmp_path / "lstm_EURUSD.pt", out_dim=3)
        # Mismatch, but allow_change=True → must not raise.
        assert_head_matches_existing(
            "EURUSD",
            want_softmax=False,
            allow_change=True,
            models_dir=tmp_path,
        )

    def test_corrupt_model_does_not_raise(self, tmp_path):
        # A bit-flipped or truncated file shouldn't block retraining —
        # the guard is for shape integrity, not file integrity.
        (tmp_path / "lstm_EURUSD.pt").write_bytes(b"not-a-torch-file")
        assert_head_matches_existing(
            "EURUSD", want_softmax=True, models_dir=tmp_path,
        )


class TestLiveConfigIntegrity:
    """Pin the currently-shipping production per-symbol head config so an
    accidental settings.yaml edit breaks a test instead of breaking live
    trading on the next retrain."""

    _EXPECTED = {
        "XAUUSD": "regression",
        "EURUSD": "softmax",
        "USDJPY": "softmax",
        "USDCAD": "regression",
        "ETHUSD": "softmax",
    }

    def test_live_settings_per_symbol_heads(self):
        root = Path(__file__).parent.parent.parent
        import yaml
        cfg = yaml.safe_load(
            (root / "config" / "settings.yaml").read_text(encoding="utf-8")
        )
        params = cfg["strategy"]["per_symbol_params"]
        for sym, want in self._EXPECTED.items():
            assert params[sym]["model_head"] == want, (
                f"{sym}: expected {want}, got {params[sym].get('model_head')!r}"
            )
