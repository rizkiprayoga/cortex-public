"""Tests for scripts/check_docs_consistency.py (Plan 2 — Doc-Drift Linter)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scripts.check_docs_consistency import (
    apply_transform,
    extract_claimed_numbers,
    flatten_yaml,
    run_check,
)


class TestFlatten:
    def test_nested_dicts_to_dotted_keys(self):
        data = {"a": {"b": {"c": 1}}, "top": 2}
        out = flatten_yaml(data)
        assert out == {"a.b.c": 1, "top": 2}

    def test_list_leaves_kept_as_is(self):
        data = {"symbols": ["XAUUSD", "EURUSD"]}
        out = flatten_yaml(data)
        # list is a leaf; lives at its parent dotted key
        assert out["symbols"] == ["XAUUSD", "EURUSD"]


class TestTransform:
    def test_identity(self):
        assert apply_transform(60.0, None, None) == 60.0

    def test_divide(self):
        assert apply_transform(60.0, "/", "4") == 15.0

    def test_multiply(self):
        assert apply_transform(1.25, "*", "2") == 2.5

    def test_add(self):
        assert apply_transform(0.45, "+", "0.1") == pytest.approx(0.55)

    def test_divide_by_zero_raises(self):
        with pytest.raises(ValueError):
            apply_transform(60, "/", "0")


class TestExtractClaimed:
    def test_markdown_bold_number(self):
        line = "The threshold is **0.45** <!-- doc-check: x -->"
        idx = line.index("doc-check")
        assert 0.45 in extract_claimed_numbers(line, idx)

    def test_python_inline_comment(self):
        line = "time_exit_bars: int = 20,  # doc-check: y"
        idx = line.index("doc-check")
        assert 20.0 in extract_claimed_numbers(line, idx)

    def test_percentage_stripped(self):
        line = "risk is 1.25% <!-- doc-check: z -->"
        idx = line.index("doc-check")
        assert 1.25 in extract_claimed_numbers(line, idx)

    def test_comma_thousands_stripped(self):
        line = "magic is 20,240,101 <!-- doc-check: x -->"
        idx = line.index("doc-check")
        assert 20240101.0 in extract_claimed_numbers(line, idx)

    def test_multiple_numbers_all_returned(self):
        # "15 H4 bars" — both 15 and 4 parsed; caller picks any match.
        line = "~15 H4 bars <!-- doc-check: x -->"
        idx = line.index("doc-check")
        nums = extract_claimed_numbers(line, idx)
        assert 15.0 in nums and 4.0 in nums

    def test_no_number_before_marker_returns_empty(self):
        line = "just text <!-- doc-check: x -->"
        idx = line.index("doc-check")
        assert extract_claimed_numbers(line, idx) == []


class TestRunCheck:
    @pytest.fixture
    def config_file(self, tmp_path: Path) -> Path:
        yml = tmp_path / "settings.yaml"
        yml.write_text(textwrap.dedent("""
            strategy:
              min_confidence: 0.55
              per_symbol_params:
                USDJPY:
                  time_exit_h1_bars: 60
                  tp_r_multiple: 2.0
            risk:
              max_daily_trades: 12
        """).strip(), encoding="utf-8")
        return yml

    def _write(self, tmp: Path, name: str, text: str) -> Path:
        p = tmp / name
        p.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
        return p

    def test_passing_marker(self, tmp_path: Path, config_file: Path):
        doc = self._write(tmp_path, "ok.md",
            "Threshold is **0.55** <!-- doc-check: strategy.min_confidence -->",
        )
        all_findings, failures = run_check(paths=[doc], config_path=config_file)
        assert len(all_findings) == 1
        assert failures == []

    def test_drifted_marker(self, tmp_path: Path, config_file: Path):
        doc = self._write(tmp_path, "bad.md",
            "Threshold is **0.45** <!-- doc-check: strategy.min_confidence -->",
        )
        _, failures = run_check(paths=[doc], config_path=config_file)
        assert len(failures) == 1
        assert failures[0].claimed == 0.45
        assert failures[0].expected == 0.55

    def test_transform_applied(self, tmp_path: Path, config_file: Path):
        # 60 H1 bars / 4 = 15 H4 bars — should pass.
        doc = self._write(tmp_path, "transform.md",
            "~15 H4 bars <!-- doc-check: strategy.per_symbol_params.USDJPY.time_exit_h1_bars / 4 -->",
        )
        _, failures = run_check(paths=[doc], config_path=config_file)
        assert failures == []

    def test_unknown_key_fails(self, tmp_path: Path, config_file: Path):
        doc = self._write(tmp_path, "ghost.md",
            "x is 1 <!-- doc-check: nonexistent.key -->",
        )
        _, failures = run_check(paths=[doc], config_path=config_file)
        assert len(failures) == 1

    def test_python_comment_marker(self, tmp_path: Path, config_file: Path):
        doc = self._write(tmp_path, "m.py",
            "time_exit_bars: int = 60  # doc-check: strategy.per_symbol_params.USDJPY.time_exit_h1_bars",
        )
        _, failures = run_check(paths=[doc], config_path=config_file)
        assert failures == []

    def test_multiple_markers_one_file(self, tmp_path: Path, config_file: Path):
        doc = self._write(tmp_path, "multi.md", """
            Risk daily: 12 trades <!-- doc-check: risk.max_daily_trades -->
            TP mult: 2.0 <!-- doc-check: strategy.per_symbol_params.USDJPY.tp_r_multiple -->
            WRONG: 9.9 <!-- doc-check: risk.max_daily_trades -->
        """)
        all_findings, failures = run_check(paths=[doc], config_path=config_file)
        assert len(all_findings) == 3
        assert len(failures) == 1
        assert failures[0].claimed == 9.9
