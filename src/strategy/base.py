"""
base.py — BaseStrategy ABC and shared data contracts for the strategy layer.

A strategy doesn't decide WHETHER to trade (SignalCombiner does) or HOW MUCH
(PositionSizer does). A strategy decides HOW AGGRESSIVELY to size and WHERE
to place the initial stop, given a signal that's already been approved by
the brain. The concrete strategies differ only in two parameters:

    allocation_pct  — fraction of the 1%-risk base lot to actually use
    atr_trail_mult  — the runner's trailing stop distance (× ATR)

and in the rule for computing the initial stop, which is the only place
where EMA50-anchoring and ATR width differ between strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.brain.signal_combiner import SignalResult


@dataclass
class MarketContext:
    """
    Minimal market snapshot a strategy needs to build its decision.

    Every field should reflect the bar that just closed (the same bar
    the SignalCombiner used to produce the signal). ``price`` is typically
    the close of that bar or the current bid/ask — callers choose.
    """

    symbol: str
    price: float        # Current / reference price
    atr: float          # ATR(14) on the strategy timeframe
    ema50: float        # 50-period EMA on the strategy timeframe


@dataclass
class StrategyDecision:
    """
    Immutable output of ``BaseStrategy.decide()``.

    Consumed by:
      - ``PositionSizer`` (reads ``allocation_pct`` and ``initial_stop_price``)
      - ``OrderManager`` (places the initial stop at ``initial_stop_price``)
      - ``ExitManager``  (reads ``atr_trail_mult`` for the runner's trail)
    """

    strategy_name: str
    direction: str              # "buy" or "sell" — echoes signal.direction
    allocation_pct: float       # 0.0 – 1.0 — multiplier into base 1% lot
    initial_stop_price: float   # absolute price where the stop sits
    atr_trail_mult: float       # runner ATR trail multiplier
    reasoning: list[str] = field(default_factory=list)


class BaseStrategy(ABC):
    """
    Abstract base. Concrete subclasses override ``_compute_stop()`` and set
    the two class attributes. ``decide()`` itself is shared — no subclass
    needs to reimplement the gate-check/direction-echo plumbing.
    """

    #: Human-readable name, used in logs and decision records.
    name: str = "base"
    # PLACEHOLDERS — tuned production values redacted from this public template.
    #: Fraction of the base risk lot this strategy deploys. 0.0 – 1.0.
    allocation_pct: float = 0.0
    #: ATR multiplier for the runner trailing stop.
    atr_trail_mult: float = 0.0
    #: Hard floor on initial-stop distance in ATR units. Protects against
    #: pathological stops from EMA-anchored rules when the EMA is very close
    #: to current price.
    min_stop_atr_mult: float = 0.0

    def decide(
        self,
        signal: "SignalResult",
        context: MarketContext,
    ) -> StrategyDecision:
        """
        Produce a StrategyDecision for an already-approved signal.

        The caller (typically ``StrategyOrchestrator``) must only invoke
        this when ``signal.should_trade`` is True and ``signal.direction``
        is a concrete "buy" or "sell" string.
        """
        if signal.direction not in ("buy", "sell"):
            raise ValueError(
                f"BaseStrategy.decide() requires a directional signal, "
                f"got direction={signal.direction!r}"
            )
        reasoning: list[str] = []
        initial_stop = self._compute_stop(signal.direction, context, reasoning)

        # --- Minimum-stop-distance floor ----------------------------------
        # Guarantee the stop is at least `min_stop_atr_mult * atr` away from
        # price, no matter what _compute_stop returned. Without this, the
        # position sizer's `risk_usd / (sl_distance × tick_value)` formula
        # explodes when sl_distance is near zero.
        if context.atr and context.atr > 0:
            min_distance = self.min_stop_atr_mult * context.atr
            if signal.direction == "buy":
                max_stop = context.price - min_distance
                if initial_stop > max_stop:
                    reasoning.append(
                        f"stop_floor: raised from {initial_stop:.5f} to "
                        f"{max_stop:.5f} ({self.min_stop_atr_mult:.1f}×ATR "
                        f"floor; ATR={context.atr:.5f})"
                    )
                    initial_stop = max_stop
            else:  # sell
                min_stop = context.price + min_distance
                if initial_stop < min_stop:
                    reasoning.append(
                        f"stop_floor: lowered from {initial_stop:.5f} to "
                        f"{min_stop:.5f} ({self.min_stop_atr_mult:.1f}×ATR "
                        f"floor; ATR={context.atr:.5f})"
                    )
                    initial_stop = min_stop

        reasoning.append(
            f"{self.name}: allocation_pct={self.allocation_pct:.2f}, "
            f"atr_trail_mult={self.atr_trail_mult:.2f}, "
            f"initial_stop={initial_stop:.5f}"
        )
        return StrategyDecision(
            strategy_name=self.name,
            direction=signal.direction,
            allocation_pct=self.allocation_pct,
            initial_stop_price=initial_stop,
            atr_trail_mult=self.atr_trail_mult,
            reasoning=reasoning,
        )

    @abstractmethod
    def _compute_stop(
        self,
        direction: str,
        context: MarketContext,
        reasoning: list[str],
    ) -> float:
        """
        Compute the initial stop price for the given direction.

        Subclasses append a one-line explanation of the rule they used to
        ``reasoning`` so the decision record remains readable.
        """
        ...
