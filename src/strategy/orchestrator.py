"""
orchestrator.py — Map an HMM regime to a concrete strategy instance.

The orchestrator is the glue between the Brain layer's ``RegimeResult``
and the strategy layer's ``BaseStrategy`` subclasses. It:

  1. Reads ``regime.expected_volatility`` and ``regime.all_expected_vols``
     and computes a **vol rank** in [0.0, 1.0] — where 0.0 means "current
     regime has the lowest volatility of any trained state" and 1.0 means
     "highest volatility."
  2. Maps the rank to one of three strategy instances via fixed cutoffs:

        rank ≤ 0.33           → LowVolAggressiveStrategy
        0.33 < rank < 0.67    → MidVolCautiousStrategy
        rank ≥ 0.67           → HighVolDefensiveStrategy

  3. If ``all_expected_vols`` is missing (typical during cold start before
     the HMM has run its first prediction) the orchestrator falls back to
     the mid-vol cautious strategy — a safe middle of the road.

Callers can use either ``select_strategy(signal)`` to get just the
strategy (useful in unit tests of the mapping) or
``select(signal, context)`` to get a full ``StrategyDecision`` in one
call (the path used by the main trading loop).
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from src.strategy.base import BaseStrategy, MarketContext, StrategyDecision
from src.strategy.high_vol_defensive import HighVolDefensiveStrategy
from src.strategy.low_vol_aggressive import LowVolAggressiveStrategy
from src.strategy.mid_vol_cautious import MidVolCautiousStrategy

if TYPE_CHECKING:
    from src.brain.hmm_regime import RegimeResult
    from src.brain.signal_combiner import SignalResult


# Vol-rank cutoffs for routing to LowVolAggressive / MidVolCautious / HighVolDefensive.
# PLACEHOLDERS — tuned values redacted from this public template.
LOW_VOL_CUTOFF: float = 0.0
HIGH_VOL_CUTOFF: float = 1.0

# Drawdown-aware allocation clamp: cap allocation_pct at ``DD_CLAMP_ALLOC``
# once peak-to-equity drawdown exceeds ``DD_CLAMP_THRESHOLD_PCT``.
# PLACEHOLDERS — tune below the weekly-hard circuit-breaker threshold.
DD_CLAMP_THRESHOLD_PCT: float = 0.0
DD_CLAMP_ALLOC: float = 0.0


class StrategyOrchestrator:
    """
    Volatility-rank router for the three concrete strategy classes.

    Stateless other than holding one instance of each strategy. Safe to
    construct once at startup and reuse for the lifetime of the process.
    """

    def __init__(
        self,
        low_vol_strategy: Optional[BaseStrategy] = None,
        mid_vol_strategy: Optional[BaseStrategy] = None,
        high_vol_strategy: Optional[BaseStrategy] = None,
    ):
        self.low_vol = low_vol_strategy or LowVolAggressiveStrategy()
        self.mid_vol = mid_vol_strategy or MidVolCautiousStrategy()
        self.high_vol = high_vol_strategy or HighVolDefensiveStrategy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_strategy(self, signal: "SignalResult") -> BaseStrategy:
        """
        Return the strategy class appropriate for the current regime's
        volatility rank. Does not build a StrategyDecision.
        """
        rank = self._vol_rank(signal.regime)
        if rank <= LOW_VOL_CUTOFF:
            return self.low_vol
        if rank >= HIGH_VOL_CUTOFF:
            return self.high_vol
        return self.mid_vol

    def select(
        self,
        signal: "SignalResult",
        context: MarketContext,
        current_equity: Optional[float] = None,
        peak_equity: Optional[float] = None,
    ) -> StrategyDecision:
        """
        Full path: pick the strategy and call ``decide()`` against the
        current market context. The returned ``StrategyDecision`` carries
        the strategy name, allocation_pct, initial_stop_price, and
        atr_trail_mult in one record.

        Wave 6 fix #20: if the caller passes both ``current_equity`` and
        ``peak_equity`` and the peak-to-equity drawdown exceeds
        ``DD_CLAMP_THRESHOLD_PCT`` (5%), the decision's ``allocation_pct``
        is clamped down to ``DD_CLAMP_ALLOC`` (0.60). This keeps the
        LowVolAggressive class from deploying 95% allocation into the
        middle of a drawdown — a failure mode the old code allowed because
        strategy selection only looked at regime volatility, not account
        state. Callers that don't pass equity args (unit tests of the
        selection logic) keep the old behavior unchanged.
        """
        strategy = self.select_strategy(signal)
        decision = strategy.decide(signal, context)
        decision = self._maybe_clamp_for_drawdown(
            decision, current_equity, peak_equity
        )
        return decision

    @staticmethod
    def _maybe_clamp_for_drawdown(
        decision: StrategyDecision,
        current_equity: Optional[float],
        peak_equity: Optional[float],
    ) -> StrategyDecision:
        """
        Apply Wave 6 fix #20 drawdown clamp if both equity args are
        available and the drawdown exceeds the threshold.

        Returns the original decision unchanged when the clamp doesn't
        apply. When it does apply, mutates the decision in place (and
        appends a reasoning line) rather than constructing a new record
        — StrategyDecision has no frozen flag so this is safe, and
        preserving identity keeps any caller that cached the reference
        from seeing a stale allocation_pct.
        """
        if current_equity is None or peak_equity is None:
            return decision
        if peak_equity <= 0.0 or current_equity <= 0.0:
            return decision
        if current_equity >= peak_equity:
            return decision
        dd_pct = (peak_equity - current_equity) / peak_equity * 100.0
        if dd_pct <= DD_CLAMP_THRESHOLD_PCT:
            return decision
        if decision.allocation_pct <= DD_CLAMP_ALLOC:
            # Already at/below the clamp — no-op, but still log the
            # reason so audit trails show the clamp was checked.
            decision.reasoning.append(
                f"dd_clamp_checked: dd={dd_pct:.1f}% > "
                f"{DD_CLAMP_THRESHOLD_PCT}%, alloc already "
                f"{decision.allocation_pct:.2f} ≤ {DD_CLAMP_ALLOC:.2f}"
            )
            return decision
        original = decision.allocation_pct
        decision.allocation_pct = DD_CLAMP_ALLOC
        decision.reasoning.append(
            f"dd_clamped: dd={dd_pct:.1f}% > {DD_CLAMP_THRESHOLD_PCT}%, "
            f"alloc clamped {original:.2f} → {DD_CLAMP_ALLOC:.2f}"
        )
        return decision

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _vol_rank(regime: "RegimeResult") -> float:
        """
        Compute the volatility rank of ``regime.expected_volatility``
        relative to ``regime.all_expected_vols``.

        Returns a float in [0.0, 1.0]:
            0.0 — current regime has the lowest vol of any trained state
            1.0 — highest
            0.5 — single-regime or all-equal (fallback)

        Ties resolve to the leftmost (lowest) position so repeated
        equal-vol regimes all map to the same rank.
        """
        all_vols = regime.all_expected_vols
        if all_vols is None:
            return 0.5
        all_vols = np.asarray(all_vols, dtype=np.float64)
        if all_vols.size < 2:
            return 0.5
        if float(all_vols.max() - all_vols.min()) < 1e-12:
            return 0.5
        sorted_vols = np.sort(all_vols)
        # searchsorted with side="left" gives the leftmost index where
        # current_vol would be inserted while keeping the array sorted.
        pos = int(np.searchsorted(sorted_vols, float(regime.expected_volatility)))
        # Clamp to [0, n-1]: a vol above max → last rank (1.0).
        pos = max(0, min(pos, sorted_vols.size - 1))
        return pos / (sorted_vols.size - 1)
