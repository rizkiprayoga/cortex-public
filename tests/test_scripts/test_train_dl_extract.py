"""
Sprint 2 Task 2.2b-2a: pure refactor — extract _train_one_lstm_for_symbol
from main_async. Verify the function exists with the documented signature
so 2.2b-2b can grow it (hparam overrides, artifact suffix, return value).

Implementation note — source parse over live import: train_deep_learning.py
runs module-top imports (DataStore, MT5DataFeed, FeatureEngineer, dotenv
side-effects) that aren't worth pulling into a structural unit test. The
sibling test_train_dl_tvt.py uses the same source-parse pattern.
"""
from __future__ import annotations

import re
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "train_deep_learning.py"
)


def _read_source() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def test_train_one_lstm_for_symbol_exists():
    """The extracted function must be defined as `async def` at module level."""
    src = _read_source()
    # Match the signature header — async def + name + opening paren.
    assert re.search(
        r"^async\s+def\s+_train_one_lstm_for_symbol\s*\(",
        src,
        re.MULTILINE,
    ), (
        "extraction incomplete — `async def _train_one_lstm_for_symbol(...)` "
        "not found at module level in scripts/train_deep_learning.py"
    )


def test_train_one_lstm_for_symbol_signature():
    """Pinned signature so Task 2.2b-2b extends it deliberately, not by drift.

    Source-parse the function header and verify positional + keyword-only
    parameter names match the contract documented in the plan.
    """
    src = _read_source()
    match = re.search(
        r"async\s+def\s+_train_one_lstm_for_symbol\s*\((?P<sig>.*?)\)\s*->",
        src,
        re.DOTALL,
    )
    assert match, "could not parse function signature header"
    sig_text = match.group("sig")

    # Split params on commas at the top level — none of the params have
    # nested commas in this signature, so a naive split is fine.
    raw_params = [p.strip() for p in sig_text.split(",") if p.strip()]

    # Positional params come before the bare `*` separator; keyword-only
    # params come after.
    try:
        star_idx = raw_params.index("*")
    except ValueError:
        raise AssertionError(
            "expected a bare `*` separator to mark keyword-only params; "
            f"got params={raw_params}"
        )

    pos_raw = raw_params[:star_idx]
    kw_raw = raw_params[star_idx + 1:]

    def _name_of(param_text: str) -> str:
        # Strip annotation (": …") and default ("= …")
        return param_text.split(":", 1)[0].split("=", 1)[0].strip()

    pos_names = [_name_of(p) for p in pos_raw]
    kw_names = {_name_of(p) for p in kw_raw}

    assert pos_names == ["symbol", "args"], (
        f"unexpected positional params: {pos_names}"
    )

    expected_kw = {
        "feed", "engineer", "bars", "cli_head_override",
        "train_start_ts", "val_start_ts", "val_end_ts_exclusive",
        "test_start_ts",
        # Task 2.2b-2b additions:
        "hparam_overrides", "artifact_suffix", "extra_tags",
    }
    assert kw_names == expected_kw, (
        f"unexpected keyword-only params: have={kw_names}, want={expected_kw}"
    )
