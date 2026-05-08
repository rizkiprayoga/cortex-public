"""
the model bake-off spec §3 invariant #11 — meta-labeler must NEVER call
the bare ``read_feature_store`` (which can return future-dated rows
and triggers Postgres planner overhead). The only blessed read is
``read_feature_store_safe`` which subtracts ``release_lag_hours``.

Static-text scan of the relevant source files. CI gate.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

# Files where any read of feature_store MUST go through _safe.
_SAFE_ONLY_FILES = (
    _ROOT / "scripts" / "train_meta_labeler.py",
    _ROOT / "src" / "ml" / "meta_labeler.py",
)

# Match read_feature_store( NOT preceded by _safe (negative lookbehind).
# Also catches `from ... import read_feature_store as _reader` style aliases
# would NOT be caught — but those are an attack surface we accept; the
# review pass catches them.
_BARE_PATTERN = re.compile(r"(?<!_safe)\bread_feature_store\s*\(")


@pytest.mark.parametrize("path", _SAFE_ONLY_FILES)
def test_no_bare_read_feature_store(path: Path):
    """No bare ``read_feature_store(`` calls allowed in meta-labeler code."""
    assert path.exists(), f"Test target missing: {path}"
    src = path.read_text(encoding="utf-8")
    bare = _BARE_PATTERN.findall(src)
    assert not bare, (
        f"{path.name} contains bare read_feature_store( call(s) — meta-labeler "
        f"code MUST use read_feature_store_safe (spec §3 invariant #11). "
        f"Matches: {bare}. Replace with `read_feature_store_safe(store, sym, "
        f"feature_group, as_of=ts)`."
    )


def test_meta_labeler_module_imports_safe_reader():
    """src/ml/meta_labeler.py must import the _safe wrapper somewhere
    (otherwise the test_no_bare_read_feature_store check is vacuous)."""
    src = (_ROOT / "src" / "ml" / "meta_labeler.py").read_text(encoding="utf-8")
    assert "read_feature_store_safe" in src, (
        "src/ml/meta_labeler.py does not reference read_feature_store_safe — "
        "the lookahead-safe enrichment helper is the only blessed feature_store "
        "reader for the meta-labeler (spec §3 invariant #11)."
    )
