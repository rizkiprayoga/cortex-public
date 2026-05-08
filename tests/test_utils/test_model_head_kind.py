"""
Tests for the model_kind axis (orthogonal to the existing model_head axis).
Spec §4.1 + anchor 7. Adds:

  - resolve_model_kind_for_symbol(symbol) -> "lstm" | "gbm"
        Reads strategy.per_symbol_params.<sym>.model_kind from settings.yaml,
        defaulting to "lstm" if unset (production default).

  - validate_kind_head(symbol, kind, head) -> None
        Enforces the only 3 valid (kind, head) combinations:
          (lstm, regression), (lstm, softmax), (gbm, classifier).
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _write_settings(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "settings.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_resolve_model_kind_default_is_lstm(tmp_path):
    from src.utils.model_head import resolve_model_kind_for_symbol

    cfg = _write_settings(tmp_path, """
strategy:
  per_symbol_params:
    EURUSD:
      model_head: softmax
""")
    assert resolve_model_kind_for_symbol("EURUSD", settings_path=cfg) == "lstm"


def test_resolve_model_kind_explicit_gbm(tmp_path):
    from src.utils.model_head import resolve_model_kind_for_symbol

    cfg = _write_settings(tmp_path, """
strategy:
  per_symbol_params:
    XAUUSD:
      model_kind: gbm
      model_head: classifier
""")
    assert resolve_model_kind_for_symbol("XAUUSD", settings_path=cfg) == "gbm"


def test_resolve_model_kind_explicit_lstm(tmp_path):
    from src.utils.model_head import resolve_model_kind_for_symbol

    cfg = _write_settings(tmp_path, """
strategy:
  per_symbol_params:
    USDJPY:
      model_kind: lstm
      model_head: softmax
""")
    assert resolve_model_kind_for_symbol("USDJPY", settings_path=cfg) == "lstm"


def test_resolve_model_kind_invalid_raises(tmp_path):
    from src.utils.model_head import resolve_model_kind_for_symbol

    cfg = _write_settings(tmp_path, """
strategy:
  per_symbol_params:
    EURUSD:
      model_kind: catboost
""")
    with pytest.raises(ValueError, match="model_kind"):
        resolve_model_kind_for_symbol("EURUSD", settings_path=cfg)


def test_resolve_model_kind_missing_symbol_defaults_to_lstm(tmp_path):
    """Symbol not in per_symbol_params at all → default to lstm."""
    from src.utils.model_head import resolve_model_kind_for_symbol

    cfg = _write_settings(tmp_path, """
strategy:
  per_symbol_params: {}
""")
    assert resolve_model_kind_for_symbol("EURUSD", settings_path=cfg) == "lstm"


def test_validate_kind_head_valid_combinations():
    from src.utils.model_head import validate_kind_head

    validate_kind_head("EURUSD", kind="lstm", head="regression")
    validate_kind_head("EURUSD", kind="lstm", head="softmax")
    validate_kind_head("XAUUSD", kind="gbm", head="classifier")


def test_validate_kind_head_invalid_combinations_raise():
    from src.utils.model_head import validate_kind_head

    with pytest.raises(ValueError, match="invalid.*combination"):
        validate_kind_head("XAUUSD", kind="gbm", head="regression")
    with pytest.raises(ValueError, match="invalid.*combination"):
        validate_kind_head("XAUUSD", kind="gbm", head="softmax")
    with pytest.raises(ValueError, match="invalid.*combination"):
        validate_kind_head("EURUSD", kind="lstm", head="classifier")
