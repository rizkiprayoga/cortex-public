"""
Tests for StrategyOrchestrator — vol-rank → strategy class mapping.

Uses synthetic RegimeResult objects with known all_expected_vols arrays
so each test case deterministically lands in one of the three buckets.
"""

import numpy as np
import pytest

from src.brain.hmm_regime import RegimeResult
from src.brain.signal_combiner import SignalResult
from src.strategy.orchestrator import (
    HIGH_VOL_CUTOFF,
    LOW_VOL_CUTOFF,
    StrategyOrchestrator,
)
from src.strategy.low_vol_aggressive import LowVolAggressiveStrategy
from src.strategy.mid_vol_cautious import MidVolCautiousStrategy
from src.strategy.high_vol_defensive import HighVolDefensiveStrategy


def make_signal(
    regime_idx: int,
    expected_volatility: float,
    all_expected_vols: np.ndarray,
    direction: str = "buy",
) -> SignalResult:
    """Construct a minimal SignalResult for orchestrator tests."""
    labels = {0: "Crash", 1: "Bear", 2: "Neutral", 3: "Bull", 4: "Euphoria"}
    multipliers = {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}
    probs = np.zeros(5)
    probs[regime_idx] = 0.9
    probs[(regime_idx + 1) % 5] = 0.1
    regime = RegimeResult(
        symbol="XAUUSD",
        regime_index=regime_idx,
        regime_label=labels[regime_idx],
        state_probability=0.9,
        position_multiplier=multipliers[regime_idx],
        all_probabilities=probs,
        expected_volatility=float(expected_volatility),
        all_expected_vols=all_expected_vols,
    )
    return SignalResult(
        symbol="XAUUSD",
        should_trade=True,
        direction=direction,
        combined_score=0.7 if direction == "buy" else -0.7,
        regime=regime,
        lstm_prediction=0.01 if direction == "buy" else -0.01,
        confidence=0.9,
    )


class TestOrchestratorMapping:

    def setup_method(self):
        self.orch = StrategyOrchestrator()
        # 5 regimes with a clean linear vol spread so rank positions are
        # deterministic: lowest → rank 0.0, highest → rank 1.0.
        self.vols = np.array([0.005, 0.010, 0.015, 0.020, 0.025])

    def test_lowest_vol_regime_picks_low_vol_aggressive(self):
        """Regime at index 0 (lowest vol) → LowVolAggressiveStrategy."""
        signal = make_signal(
            regime_idx=3,  # label doesn't matter for the mapping
            expected_volatility=self.vols[0],
            all_expected_vols=self.vols,
        )
        strategy = self.orch.select_strategy(signal)
        assert isinstance(strategy, LowVolAggressiveStrategy)

    def test_second_lowest_vol_picks_low_vol_aggressive(self):
        """Rank 0.25 < LOW_VOL_CUTOFF → LowVolAggressive."""
        signal = make_signal(
            regime_idx=3,
            expected_volatility=self.vols[1],
            all_expected_vols=self.vols,
        )
        strategy = self.orch.select_strategy(signal)
        assert isinstance(strategy, LowVolAggressiveStrategy)
        # Sanity: 1 / (5-1) = 0.25 which is below the 0.33 cutoff.
        assert 0.25 <= LOW_VOL_CUTOFF

    def test_middle_vol_regime_picks_mid_vol_cautious(self):
        """Rank 0.5 → MidVolCautiousStrategy."""
        signal = make_signal(
            regime_idx=3,
            expected_volatility=self.vols[2],
            all_expected_vols=self.vols,
        )
        strategy = self.orch.select_strategy(signal)
        assert isinstance(strategy, MidVolCautiousStrategy)

    def test_second_highest_vol_picks_high_vol_defensive(self):
        """Rank 0.75 ≥ HIGH_VOL_CUTOFF → HighVolDefensive."""
        signal = make_signal(
            regime_idx=3,
            expected_volatility=self.vols[3],
            all_expected_vols=self.vols,
        )
        strategy = self.orch.select_strategy(signal)
        assert isinstance(strategy, HighVolDefensiveStrategy)
        assert 0.75 >= HIGH_VOL_CUTOFF

    def test_highest_vol_regime_picks_high_vol_defensive(self):
        """Rank 1.0 → HighVolDefensive."""
        signal = make_signal(
            regime_idx=3,
            expected_volatility=self.vols[4],
            all_expected_vols=self.vols,
        )
        strategy = self.orch.select_strategy(signal)
        assert isinstance(strategy, HighVolDefensiveStrategy)

    def test_missing_all_expected_vols_falls_back_to_mid(self):
        """When all_expected_vols is None → MidVolCautious (safe default)."""
        signal = make_signal(
            regime_idx=3,
            expected_volatility=0.02,
            all_expected_vols=None,  # type: ignore[arg-type]
        )
        strategy = self.orch.select_strategy(signal)
        assert isinstance(strategy, MidVolCautiousStrategy)

    def test_all_equal_vols_falls_back_to_mid(self):
        """When every regime has identical vol → MidVolCautious."""
        flat = np.array([0.015, 0.015, 0.015, 0.015, 0.015])
        signal = make_signal(
            regime_idx=3,
            expected_volatility=0.015,
            all_expected_vols=flat,
        )
        strategy = self.orch.select_strategy(signal)
        assert isinstance(strategy, MidVolCautiousStrategy)


class TestOrchestratorSelect:
    """Tests the end-to-end select(signal, context) path."""

    def setup_method(self):
        self.orch = StrategyOrchestrator()

    def test_select_returns_decision_with_stop_and_allocation(self):
        from src.strategy.base import MarketContext

        vols = np.array([0.005, 0.010, 0.015, 0.020, 0.025])
        signal = make_signal(
            regime_idx=3,
            expected_volatility=vols[0],  # lowest → LowVolAggressive
            all_expected_vols=vols,
        )
        context = MarketContext(
            symbol="XAUUSD",
            price=2000.0,
            atr=5.0,
            ema50=1990.0,
        )
        decision = self.orch.select(signal, context)
        assert decision.strategy_name == "LowVolAggressive"
        assert decision.direction == "buy"
        assert decision.allocation_pct == pytest.approx(0.95)
        assert decision.atr_trail_mult == pytest.approx(2.0)
        # Stop is the farther of (price − 3×ATR = 1985) and (EMA − 0.5×ATR = 1987.5).
        # LowVol uses min(atr_stop, ema_stop) for longs.
        # atr_stop = 2000 - 3*5 = 1985, ema_stop = 1990 - 0.5*5 = 1987.5
        # min(1985, 1987.5) = 1985
        assert decision.initial_stop_price == pytest.approx(1985.0)


class TestDrawdownClamp:
    """
    Wave 6 fix #20: when peak-to-equity drawdown exceeds 5%, the
    orchestrator clamps ``allocation_pct`` to 0.60 even for the
    LowVolAggressive class — keeps risk deployment sane during a
    drawdown so LowVolAggressive can't blast 95% alloc into a recovery
    attempt.
    """

    def setup_method(self):
        from src.strategy.base import MarketContext

        self.orch = StrategyOrchestrator()
        self.context = MarketContext(
            symbol="XAUUSD",
            price=2000.0,
            atr=5.0,
            ema50=1990.0,
        )
        self.vols = np.array([0.005, 0.010, 0.015, 0.020, 0.025])
        self.signal = make_signal(
            regime_idx=3,
            expected_volatility=self.vols[0],  # LowVolAggressive path
            all_expected_vols=self.vols,
        )

    def test_no_clamp_when_equity_args_omitted(self):
        """Old callers (no equity args) get the unclamped decision."""
        decision = self.orch.select(self.signal, self.context)
        assert decision.allocation_pct == pytest.approx(0.95)
        assert not any("dd_clamp" in r for r in decision.reasoning)

    def test_no_clamp_when_drawdown_below_threshold(self):
        """4% DD is below the 5% threshold → no clamp."""
        decision = self.orch.select(
            self.signal,
            self.context,
            current_equity=9_600.0,
            peak_equity=10_000.0,
        )
        assert decision.allocation_pct == pytest.approx(0.95)
        assert not any("dd_clamped" in r for r in decision.reasoning)

    def test_clamp_when_drawdown_exceeds_threshold(self):
        """7% DD → LowVolAggressive alloc clamps from 0.95 to 0.60."""
        decision = self.orch.select(
            self.signal,
            self.context,
            current_equity=9_300.0,
            peak_equity=10_000.0,
        )
        assert decision.allocation_pct == pytest.approx(0.60)
        assert any("dd_clamped" in r for r in decision.reasoning)

    def test_no_clamp_when_equity_above_peak(self):
        """New high → no drawdown → no clamp."""
        decision = self.orch.select(
            self.signal,
            self.context,
            current_equity=10_500.0,
            peak_equity=10_000.0,
        )
        assert decision.allocation_pct == pytest.approx(0.95)

    def test_clamp_no_op_when_alloc_already_below_cap(self):
        """
        HighVolDefensive uses 0.60 baseline already — the clamp must
        not *raise* allocation to 0.60, only lower. We assert the
        decision's alloc is still 0.60 (unchanged) and the reason
        string records that the clamp was checked.
        """
        signal = make_signal(
            regime_idx=3,
            expected_volatility=self.vols[4],  # highest → HighVolDefensive
            all_expected_vols=self.vols,
        )
        decision = self.orch.select(
            signal,
            self.context,
            current_equity=9_300.0,  # 7% DD
            peak_equity=10_000.0,
        )
        assert decision.allocation_pct == pytest.approx(0.60)
