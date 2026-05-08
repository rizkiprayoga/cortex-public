"""
mid_vol_cautious.py — Middle-of-the-road strategy for mid-volatility regimes.

Used when the current HMM regime is in the middle third of the trained
volatility distribution. The stop is anchored tightly to the 50 EMA
(structural), and allocation depends on whether price is aligned with
the EMA in the signal direction — if yes, full aggressive size (95%); if
not, a conservative 60% because we're taking a signal against the
immediate structure.

Initial stop rule
-----------------
  buy:  stop = EMA50 − 0.5 × ATR
  sell: stop = EMA50 + 0.5 × ATR

Allocation rule
---------------
  buy  + price > EMA50  → 0.95   (aligned: aggressive)
  sell + price < EMA50  → 0.95   (aligned: aggressive)
  otherwise             → 0.60   (misaligned: cautious)

The ``allocation_pct`` class attribute serves as the fallback value and is
overridden on the instance in ``decide()`` before the base-class builds
the StrategyDecision. This keeps ``_compute_stop()``'s signature in line
with the other two strategies.
"""

from src.strategy.base import BaseStrategy, MarketContext, StrategyDecision
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.brain.signal_combiner import SignalResult


class MidVolCautiousStrategy(BaseStrategy):

    name: str = "MidVolCautious"
    # PLACEHOLDERS — tuned production values redacted from this public template.
    allocation_pct: float = 0.0       # default; overridden per-call when misaligned
    atr_trail_mult: float = 0.0
    MISALIGNED_ALLOCATION_PCT: float = 0.0
    _EMA_ATR_MULT: float = 0.0        # tune the ATR multiplier for the EMA-anchored stop

    def decide(
        self,
        signal: "SignalResult",
        context: MarketContext,
    ) -> StrategyDecision:
        # Decide alignment BEFORE delegating to base so we can adjust
        # allocation_pct for this one call. The base class reads
        # ``self.allocation_pct`` when building the StrategyDecision.
        aligned = (
            (signal.direction == "buy" and context.price > context.ema50)
            or (signal.direction == "sell" and context.price < context.ema50)
        )
        # Mutate the instance attribute for the duration of this call.
        # Safe because one orchestrator holds one strategy instance per
        # process, and ``decide()`` is never called concurrently on the
        # same instance (the trading loop is single-threaded).
        previous = self.allocation_pct
        self.allocation_pct = (0.0 if aligned else self.MISALIGNED_ALLOCATION_PCT)  # PLACEHOLDER aligned value
        try:
            decision = super().decide(signal, context)
            decision.reasoning.insert(
                0,
                f"mid_vol_alignment: price={context.price:.5f} "
                f"{'>' if context.price > context.ema50 else '<='} "
                f"ema50={context.ema50:.5f} — "
                f"{'aligned' if aligned else 'misaligned'}, "
                f"allocation={self.allocation_pct:.2f}",
            )
            return decision
        finally:
            self.allocation_pct = previous

    def _compute_stop(
        self,
        direction: str,
        context: MarketContext,
        reasoning: list[str],
    ) -> float:
        stop = (
            context.ema50 - self._EMA_ATR_MULT * context.atr
            if direction == "buy"
            else context.ema50 + self._EMA_ATR_MULT * context.atr
        )
        reasoning.append(
            f"mid_vol_stop: {direction} — EMA50 ± {self._EMA_ATR_MULT:.2f}×ATR = {stop:.5f}"
        )
        return float(stop)
