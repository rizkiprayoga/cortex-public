"""
test_train_scripts_mt5_free.py — regression guard against re-introducing
the shared-MT5-terminal hijack risk.

The 2026-04-25 incident polluted prod equity_history because dev's
``train_hmm.py`` called ``mt5.initialize()`` while the prod bot was running
on the shared Windows MT5 terminal binding. The fix is structural:
training and backtest scripts read OHLCV from DataStore only, never from
MT5.

These tests assert the structural invariant via static source-text scan:
none of the listed scripts may import ``MT5Connector`` or call
``mt5.initialize`` / ``mt5.login``. The DB-only path
(``MT5DataFeed.get_historical_db_only``) is the ONLY blessed OHLCV reader
for these scripts.

``scripts/backtest.py`` joined the list 2026-04-26 (T-9) — previously it
called ``MT5Connector().connect()`` + sync ``feed.get_historical()``,
which hijacked the shared terminal binding even with snapshot/restore
safety. Phase A Sprint 6 (the bake-off run) requires backtest.py be
MT5-free.

If a future PR re-introduces the dependency (e.g. someone re-adds a
"fast path" that hits MT5 directly), CI will catch it here before it
reaches the operator's terminal.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

# Scripts that must remain MT5-free. Add new training/backtest scripts here
# as they get refactored. Backfill scripts (e.g. backfill_ohlcv.py,
# backfill_ethusd.py) are legitimately MT5-bound and excluded by design;
# they use the snapshot/restore helpers in ``scripts/_mt5_safety.py``.
_MT5_FREE_SCRIPTS = (
    _ROOT / "scripts" / "train_hmm.py",
    _ROOT / "scripts" / "train_deep_learning.py",
    _ROOT / "scripts" / "backtest.py",
    _ROOT / "scripts" / "train_gbm.py",
)

# Patterns that indicate the script touches the shared MT5 terminal binding.
# Catches both ``import MetaTrader5 as mt5`` style and the wrapper class.
_FORBIDDEN_IMPORT_PATTERNS = (
    re.compile(r"^\s*from\s+src\.broker\.mt5_connector\s+import\s+MT5Connector",
               re.MULTILINE),
    re.compile(r"^\s*import\s+MetaTrader5", re.MULTILINE),
    re.compile(r"^\s*from\s+MetaTrader5\b", re.MULTILINE),
)

_FORBIDDEN_CALL_PATTERNS = (
    re.compile(r"\bMT5Connector\s*\("),
    re.compile(r"\bmt5\.initialize\s*\("),
    re.compile(r"\bmt5\.login\s*\("),
    # Any feed.get_historical(...) call on the sync MT5-bound path. The
    # DB-only path is feed.get_historical_db_only(...).
    re.compile(r"\bfeed\.get_historical\s*\("),
)


@pytest.mark.parametrize("script_path", _MT5_FREE_SCRIPTS)
class TestScriptIsMT5Free:
    """Static source scan — these scripts must not import or call MT5 APIs."""

    def test_no_mt5_imports(self, script_path: Path):
        assert script_path.exists(), f"Test target missing: {script_path}"
        text = script_path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_IMPORT_PATTERNS:
            match = pattern.search(text)
            assert match is None, (
                f"{script_path.name} imports MT5 ({pattern.pattern!r}). "
                "Training and backtest scripts must remain MT5-free to "
                "avoid the shared terminal hijack risk — see "
                "memory/feedback_dev_mt5_steals_prod_terminal.md. "
                "Use feed.get_historical_db_only() instead, with feed "
                "constructed as MT5DataFeed(connector=None, data_store=...)."
            )

    def test_no_mt5_api_calls(self, script_path: Path):
        text = script_path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_CALL_PATTERNS:
            match = pattern.search(text)
            assert match is None, (
                f"{script_path.name} contains MT5 API call matching "
                f"{pattern.pattern!r}. Training and backtest scripts must "
                "use feed.get_historical_db_only() (DB-only path) — never "
                "feed.get_historical() (sync, MT5-bound) or any direct "
                "mt5.* call. See "
                "memory/feedback_dev_mt5_steals_prod_terminal.md."
            )

    def test_uses_db_only_reader(self, script_path: Path):
        """Positive assertion: each script must reference the DB-only path."""
        text = script_path.read_text(encoding="utf-8")
        assert "get_historical_db_only" in text, (
            f"{script_path.name} doesn't use get_historical_db_only — how is "
            "it reading OHLCV without MT5?"
        )
