"""
the model bake-off decision gate per spec anchor 5.

decide_winner_per_symbol(cells) implements:
  - DSR > 0.5 floor: if neither tuned variant clears, inconclusive →
    keep LSTM (status quo). No architectural conclusion possible.
  - 2-of-3 gate on tuned variants (PF higher, DD lower, stability
    higher). Whoever wins 2+ takes the symbol.
  - Tied 1.5-1.5: simplicity tiebreaker → GBM wins (anchor 5: GBM is
    faster to retrain, SHAP-interpretable, no GPU).

The function lives in src/ml/gate.py (pure logic — testable in
isolation) and is re-exported from scripts/model_bench.py so the
orchestrator can use it without a circular dependency.
"""
from __future__ import annotations

import pytest


def test_dsr_floor_below_threshold_calls_inconclusive():
    """All 4 cells below DSR=0.5 → inconclusive → keep LSTM (status quo)."""
    from scripts.model_bench import decide_winner_per_symbol

    cells = {
        "lstm_default": {"pf": 1.5, "dd": 0.05, "stability": 0.6,  "dsr": 0.30},
        "lstm_tuned":   {"pf": 1.6, "dd": 0.04, "stability": 0.62, "dsr": 0.40},
        "gbm_default":  {"pf": 1.7, "dd": 0.05, "stability": 0.61, "dsr": 0.45},
        "gbm_tuned":    {"pf": 1.8, "dd": 0.04, "stability": 0.63, "dsr": 0.49},
    }
    decision = decide_winner_per_symbol(cells)
    assert decision["winner"] == "lstm"
    assert decision["inconclusive"] is True
    assert "DSR" in decision["reasoning"]


def test_simplicity_tiebreaker_gbm_wins_ties():
    """1.5-1.5 tie → simplicity tiebreaker selects GBM (anchor 5)."""
    from scripts.model_bench import decide_winner_per_symbol

    cells = {
        "lstm_default": {"pf": 1.5, "dd": 0.05, "stability": 0.6,  "dsr": 0.55},
        # Equal PF, equal stability, GBM has lower DD → GBM wins 1 (DD),
        # LSTM wins 0, ties 2 (PF, stability). Need to handle ties as
        # half-points to land at 1.5-1.5; the impl uses strict >.
        # To get a 1.5-1.5 outcome by counting strict-> we set GBM
        # ahead on PF and stability, behind on DD = 2-1 GBM win.
        # For an actual 1.5-1.5 we'd need exact ties; instead test
        # the tied-strict-count case → GBM wins via tiebreaker.
        "lstm_tuned":   {"pf": 2.0, "dd": 0.04, "stability": 0.65, "dsr": 0.60},
        "gbm_default":  {"pf": 1.6, "dd": 0.06, "stability": 0.55, "dsr": 0.55},
        "gbm_tuned":    {"pf": 2.0, "dd": 0.04, "stability": 0.65, "dsr": 0.62},
    }
    # All three metrics tie → 0-0 strict-count → tiebreaker → GBM.
    decision = decide_winner_per_symbol(cells)
    assert decision["winner"] == "gbm"
    assert decision["inconclusive"] is False
    assert "tiebreak" in decision["reasoning"].lower() or "simplicity" in decision["reasoning"].lower()


def test_clear_lstm_winner():
    """LSTM dominates 3-of-3, both clear DSR floor → LSTM wins clearly."""
    from scripts.model_bench import decide_winner_per_symbol

    cells = {
        "lstm_default": {"pf": 1.5, "dd": 0.05, "stability": 0.6,  "dsr": 0.55},
        "lstm_tuned":   {"pf": 3.0, "dd": 0.03, "stability": 0.70, "dsr": 0.70},
        "gbm_default":  {"pf": 1.6, "dd": 0.06, "stability": 0.55, "dsr": 0.55},
        "gbm_tuned":    {"pf": 2.0, "dd": 0.05, "stability": 0.60, "dsr": 0.55},
    }
    decision = decide_winner_per_symbol(cells)
    assert decision["winner"] == "lstm"
    assert decision["inconclusive"] is False


def test_clear_gbm_winner():
    """GBM dominates 3-of-3, both clear DSR floor → GBM wins clearly."""
    from scripts.model_bench import decide_winner_per_symbol

    cells = {
        "lstm_default": {"pf": 1.5, "dd": 0.05, "stability": 0.6,  "dsr": 0.55},
        "lstm_tuned":   {"pf": 1.8, "dd": 0.05, "stability": 0.60, "dsr": 0.55},
        "gbm_default":  {"pf": 1.7, "dd": 0.04, "stability": 0.62, "dsr": 0.60},
        "gbm_tuned":    {"pf": 2.5, "dd": 0.03, "stability": 0.72, "dsr": 0.68},
    }
    decision = decide_winner_per_symbol(cells)
    assert decision["winner"] == "gbm"
    assert decision["inconclusive"] is False


def test_dsr_floor_one_clears_other_doesnt_picks_clearer():
    """If only ONE tuned variant clears DSR≥0.5, that one wins
    (the other is below the trustworthiness floor)."""
    from scripts.model_bench import decide_winner_per_symbol

    cells = {
        "lstm_default": {"pf": 1.5, "dd": 0.05, "stability": 0.6,  "dsr": 0.30},
        "lstm_tuned":   {"pf": 1.6, "dd": 0.04, "stability": 0.62, "dsr": 0.40},  # below floor
        "gbm_default":  {"pf": 1.7, "dd": 0.05, "stability": 0.61, "dsr": 0.55},
        "gbm_tuned":    {"pf": 1.8, "dd": 0.04, "stability": 0.63, "dsr": 0.65},  # clears
    }
    decision = decide_winner_per_symbol(cells)
    assert decision["winner"] == "gbm"
    assert decision["inconclusive"] is False
