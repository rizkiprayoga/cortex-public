"""Unit tests for scripts/bakeoff_verdict.py — Sprint 7 reporter logic.

The expensive parts (running 16 backtests, ~3h) are NOT tested here —
that's covered by the operator-run end-to-end smoke. These tests cover
the pure-logic pieces that decide whether the verdict math is right.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.bakeoff_verdict import (
    CELLS,
    _compute_cell_metrics,
    _format_grid,
    _verdict_for_symbol,
)


def _write_fake_cell(out_dir: Path, symbol: str, pnls: list[float]) -> None:
    """Write the two CSVs that ``_compute_cell_metrics`` reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trades = pd.DataFrame({"pnl": pnls})
    trades.to_csv(out_dir / f"backtest_trades_{symbol}.csv", index=False)
    # Minimal equity curve with a fake drawdown column.
    equity = 10_000.0 + np.cumsum(pnls)
    peak = np.maximum.accumulate(np.concatenate([[10_000.0], equity]))[1:]
    dd_pct = (1 - equity / peak) * 100
    eq_df = pd.DataFrame({
        "equity": equity,
        "drawdown_pct": dd_pct,
    })
    eq_df.to_csv(out_dir / f"backtest_equity_{symbol}.csv", index=False)


def test_compute_cell_metrics_returns_none_when_csvs_missing(tmp_path):
    """No CSVs → no metrics. Sprint 7 reporter must NOT silently
    fabricate zeros for a failed cell."""
    out = _compute_cell_metrics(tmp_path, "XAUUSD")
    assert out is None


def test_compute_cell_metrics_returns_none_when_no_trades(tmp_path):
    """Empty trades CSV is the same as no signal — flag inconclusive."""
    pd.DataFrame({"pnl": []}).to_csv(
        tmp_path / "backtest_trades_XAUUSD.csv", index=False,
    )
    pd.DataFrame({"equity": [10000.0], "drawdown_pct": [0.0]}).to_csv(
        tmp_path / "backtest_equity_XAUUSD.csv", index=False,
    )
    out = _compute_cell_metrics(tmp_path, "XAUUSD")
    assert out is None


def test_compute_cell_metrics_pf_dd_correct(tmp_path):
    """Profit factor = gross_profit / gross_loss, max DD% from equity
    curve. Both must reflect the trades exactly."""
    # 7 wins of $100, 3 losses of $50 → PF = 700 / 150 = 4.667
    pnls = [100.0] * 7 + [-50.0] * 3
    _write_fake_cell(tmp_path, "XAUUSD", pnls)
    out = _compute_cell_metrics(tmp_path, "XAUUSD")
    # < 10 trades → DSR/stab clamped to 0 but PF/DD must still be right
    assert out is not None
    assert abs(out["pf"] - 700.0 / 150.0) < 1e-6
    assert out["trades"] == 10
    # The order is wins-first then losses, so equity peaks at 10700
    # then dips by 150 over the 3 losses → max DD ~ 150 / 10700 ≈ 1.40%
    assert 1.30 < out["dd"] < 1.50


def test_compute_cell_metrics_dsr_zero_below_10_trades(tmp_path):
    """DSR derivation needs ≥10 observations — fewer must zero-out
    rather than crash. The verdict gate then treats it as inconclusive."""
    pnls = [50.0, -25.0, 30.0]  # only 3 trades
    _write_fake_cell(tmp_path, "XAUUSD", pnls)
    out = _compute_cell_metrics(tmp_path, "XAUUSD")
    assert out is not None
    assert out["dsr"] == 0.0
    assert out["stability"] == 0.0


def test_compute_cell_metrics_dsr_nonzero_with_enough_trades(tmp_path):
    """≥10 trades + nonzero std → DSR + stability are computed."""
    rng = np.random.default_rng(0)
    pnls = list(rng.normal(loc=10.0, scale=50.0, size=30))
    _write_fake_cell(tmp_path, "XAUUSD", pnls)
    out = _compute_cell_metrics(tmp_path, "XAUUSD")
    assert out is not None
    assert out["trades"] == 30
    # DSR is in [0, 1] (it's a probability per López de Prado)
    assert 0.0 <= out["dsr"] <= 1.0
    # Stability is fraction in [0, 1]
    assert 0.0 <= out["stability"] <= 1.0


def test_verdict_for_symbol_requires_all_4_cells():
    """If any cell is missing, decide_winner_per_symbol can't run.
    Reporter must mark the symbol incomplete (not crash)."""
    incomplete = {
        "lstm_default": {"pf": 2.0, "dd": 5.0, "stability": 0.6, "dsr": 0.7},
        "lstm_tuned": {"pf": 2.5, "dd": 4.0, "stability": 0.65, "dsr": 0.8},
        # Missing both gbm cells.
    }
    assert _verdict_for_symbol(incomplete) is None


def test_verdict_for_symbol_passes_cells_to_gate():
    """Happy path: 4 cells present, decide_winner_per_symbol returns a
    dict with 'winner' + 'reasoning'."""
    cells = {
        "lstm_default": {"pf": 2.0, "dd": 5.0, "stability": 0.6, "dsr": 0.7},
        "lstm_tuned": {"pf": 2.5, "dd": 4.0, "stability": 0.65, "dsr": 0.8},
        "gbm_default": {"pf": 2.1, "dd": 5.5, "stability": 0.55, "dsr": 0.6},
        "gbm_tuned": {"pf": 2.3, "dd": 4.8, "stability": 0.6, "dsr": 0.7},
    }
    verdict = _verdict_for_symbol(cells)
    assert verdict is not None
    assert verdict["winner"] in ("lstm", "gbm")
    assert "reasoning" in verdict


def test_format_grid_handles_missing_cells():
    """Markdown formatter must show — for missing cells, real numbers
    for present ones. No KeyError, no silent zeros."""
    cells = {
        "lstm_default": {
            "trades": 100, "pf": 2.5, "dd": 4.2,
            "stability": 0.7, "dsr": 0.85, "net_pnl": 1234.0,
        },
    }
    md = _format_grid("XAUUSD", cells)
    assert "XAUUSD" in md
    assert "lstm_default" in md
    assert "2.50" in md
    # Missing cells render dashes, not crashes
    assert md.count("| — |") >= 3  # the 3 missing cells


def test_cells_constant_matches_decide_winner_keys():
    """Sanity: the CELLS list must produce the exact keys
    decide_winner_per_symbol expects, otherwise the verdict step
    silently skips the gate."""
    expected_keys = {"lstm_default", "lstm_tuned", "gbm_default", "gbm_tuned"}
    produced_keys = {f"{p}_{v}" for p, v in CELLS}
    assert produced_keys == expected_keys
