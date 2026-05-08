"""Unit tests for E-7 promotion-gate decision logic."""
from scripts.trend_mode_verdict import evaluate_gate, GateResult


class TestPromotionGate:
    def test_clear_pass(self):
        result = evaluate_gate(
            cagr_baseline=21.5, cagr_trend=27.0,        # +5.5pp
            max_dd_baseline=2.9, max_dd_trend=3.1,      # +0.2pp (within 1pp)
            pf_baseline=2.89, pf_trend=2.95,            # PF up
        )
        assert result.passed is True
        assert "PASS" in result.summary

    def test_fail_cagr_below_floor(self):
        result = evaluate_gate(
            cagr_baseline=21.5, cagr_trend=23.0,        # +1.5pp (below +3pp)
            max_dd_baseline=2.9, max_dd_trend=3.0,
            pf_baseline=2.89, pf_trend=2.90,
        )
        assert result.passed is False
        assert "CAGR" in result.summary

    def test_fail_dd_blowup(self):
        result = evaluate_gate(
            cagr_baseline=21.5, cagr_trend=27.0,
            max_dd_baseline=2.9, max_dd_trend=4.5,      # +1.6pp (above 1pp)
            pf_baseline=2.89, pf_trend=2.90,
        )
        assert result.passed is False
        assert "DD" in result.summary

    def test_fail_pf_drop_too_far(self):
        result = evaluate_gate(
            cagr_baseline=21.5, cagr_trend=27.0,
            max_dd_baseline=2.9, max_dd_trend=3.0,
            pf_baseline=2.89, pf_trend=2.70,            # -0.19 (below -0.10 floor)
        )
        assert result.passed is False
        assert "PF" in result.summary

    def test_borderline_pf_drop_within_floor(self):
        result = evaluate_gate(
            cagr_baseline=21.5, cagr_trend=27.0,
            max_dd_baseline=2.9, max_dd_trend=3.0,
            pf_baseline=2.89, pf_trend=2.81,            # -0.08 (within -0.10)
        )
        assert result.passed is True
