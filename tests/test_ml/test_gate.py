"""Unit tests for the model_bench decision gate.

The gate passes when the candidate beats the current on at least 2 of 3
criteria: portfolio PF, portfolio DD (lower is better), trade-count
stability (within ±20%).
"""
from __future__ import annotations


def _call(current_pf, current_dd, current_trades,
          candidate_pf, candidate_dd, candidate_trades,
          trade_tolerance=0.20):
    """Helper: call the gate with a minimal input shape."""
    from src.ml.gate import compute_gate_verdict
    return compute_gate_verdict(
        current_portfolio_pf=current_pf,
        current_portfolio_dd=current_dd,
        current_total_trades=current_trades,
        candidate_portfolio_pf=candidate_pf,
        candidate_portfolio_dd=candidate_dd,
        candidate_total_trades=candidate_trades,
        trade_stability_tolerance=trade_tolerance,
    )


def test_gate_passes_when_all_three_criteria_met():
    """Candidate strictly better on all three → PASS with 3/3."""
    result = _call(
        current_pf=2.89, current_dd=3.10, current_trades=1450,
        candidate_pf=3.20, candidate_dd=2.50, candidate_trades=1480,
    )
    assert result.passed is True
    assert result.criteria_met == 3
    assert result.pf_improved is True
    assert result.dd_improved is True
    assert result.trades_stable is True


def test_gate_passes_with_exactly_two_criteria():
    """Candidate better on PF + DD, trade count drifts > 20% → still PASS."""
    result = _call(
        current_pf=2.89, current_dd=3.10, current_trades=1000,
        candidate_pf=3.20, candidate_dd=2.50,
        candidate_trades=500,  # 50% drop — outside tolerance
    )
    assert result.passed is True
    assert result.criteria_met == 2
    assert result.pf_improved is True
    assert result.dd_improved is True
    assert result.trades_stable is False


def test_gate_fails_with_only_one_criterion():
    """Candidate better on PF only; DD worse + trade count collapses → FAIL."""
    result = _call(
        current_pf=2.89, current_dd=3.10, current_trades=1000,
        candidate_pf=3.50,       # better
        candidate_dd=5.00,       # worse
        candidate_trades=400,    # 60% drop — outside tolerance
    )
    assert result.passed is False
    assert result.criteria_met == 1
    assert result.pf_improved is True
    assert result.dd_improved is False
    assert result.trades_stable is False


def test_gate_ties_count_as_pass():
    """Candidate matches current exactly on PF + DD + trades → PASS.

    Rationale: ties mean 'no regression', which is what the gate is
    protecting against. A refactor that produces identical outputs
    should promote cleanly.
    """
    result = _call(
        current_pf=2.89, current_dd=3.10, current_trades=1000,
        candidate_pf=2.89, candidate_dd=3.10, candidate_trades=1000,
    )
    assert result.passed is True
    assert result.criteria_met == 3


def test_gate_respects_trade_stability_tolerance():
    """Trade-count change at exactly the tolerance boundary counts as stable."""
    # 20% tolerance, 1000 -> 1200 is exactly +20% (within)
    result = _call(
        current_pf=2.89, current_dd=3.10, current_trades=1000,
        candidate_pf=2.89, candidate_dd=3.10, candidate_trades=1200,
        trade_tolerance=0.20,
    )
    assert result.trades_stable is True

    # 1000 -> 1201 exceeds the tolerance (outside)
    result2 = _call(
        current_pf=2.89, current_dd=3.10, current_trades=1000,
        candidate_pf=2.89, candidate_dd=3.10, candidate_trades=1201,
        trade_tolerance=0.20,
    )
    assert result2.trades_stable is False


def test_gate_handles_zero_current_trades_edge_case():
    """If current trade count is zero, tolerance is meaningless.

    Treat trades_stable as True iff candidate trades is also zero;
    otherwise False. Prevents division-by-zero.
    """
    result_both_zero = _call(
        current_pf=0.0, current_dd=0.0, current_trades=0,
        candidate_pf=0.0, candidate_dd=0.0, candidate_trades=0,
    )
    assert result_both_zero.trades_stable is True

    result_one_zero = _call(
        current_pf=0.0, current_dd=0.0, current_trades=0,
        candidate_pf=2.0, candidate_dd=3.0, candidate_trades=100,
    )
    assert result_one_zero.trades_stable is False


def test_gate_verdict_details_are_human_readable():
    """Verdict must expose a ``rationale`` string list for operator context."""
    result = _call(
        current_pf=2.89, current_dd=3.10, current_trades=1000,
        candidate_pf=3.20, candidate_dd=3.50, candidate_trades=1100,
    )
    assert isinstance(result.rationale, list)
    assert len(result.rationale) == 3   # one line per criterion
    # Joining should be readable
    joined = "\n".join(result.rationale)
    assert "PF" in joined
    assert "DD" in joined
    assert "trade" in joined.lower()
