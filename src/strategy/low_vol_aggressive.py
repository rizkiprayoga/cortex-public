"""
low_vol_aggressive.py — Aggressive strategy for low-volatility regimes.

Used when the current HMM regime sits in the lowest third of the
volatility distribution across all trained regimes. Low vol means:
  - stops can sit wide without giving back unreasonable amounts of equity
  - upside capture benefits from a generous runner trail
  - base lot size can run near full (95% of the 1%-risk base)

Initial stop rule
-----------------
The stop is placed at the *farther* of two anchors:
  (a) price − 3 × ATR          (wide ATR-only stop)
  (b) EMA50 − 0.5 × ATR        (structural stop below the 50 EMA)

For a "sell" direction the signs flip and we take the *nearer* of the two
symmetric upside anchors (via max instead of min).

Why "farther" for longs and "nearer" for shorts: we always want the stop
to give the trade as much room as the structure allows. For a long, that
means the *lower* of the two levels; for a short, the *higher* one.
"""

from src.strategy.base import BaseStrategy, MarketContext


class LowVolAggressiveStrategy(BaseStrategy):

    name: str = "LowVolAggressive"
    # PLACEHOLDERS — tuned production values redacted from this public template.
    allocation_pct: float = 0.0
    atr_trail_mult: float = 0.0

    # PLACEHOLDER — tune the ATR multipliers used in the stop formulas below.
    _ATR_PRICE_ANCHOR_MULT: float = 0.0
    _ATR_EMA_ANCHOR_MULT: float = 0.0

    def _compute_stop(
        self,
        direction: str,
        context: MarketContext,
        reasoning: list[str],
    ) -> float:
        atr_stop = (
            context.price - self._ATR_PRICE_ANCHOR_MULT * context.atr
            if direction == "buy"
            else context.price + self._ATR_PRICE_ANCHOR_MULT * context.atr
        )
        ema_stop = (
            context.ema50 - self._ATR_EMA_ANCHOR_MULT * context.atr
            if direction == "buy"
            else context.ema50 + self._ATR_EMA_ANCHOR_MULT * context.atr
        )
        if direction == "buy":
            # Whichever anchor sits LOWER is farther from price → more room.
            stop = min(atr_stop, ema_stop)
        else:
            # For shorts, the HIGHER anchor is farther from price.
            stop = max(atr_stop, ema_stop)
        reasoning.append(
            f"low_vol_stop: {direction} — atr_stop={atr_stop:.5f}, "
            f"ema_stop={ema_stop:.5f}, picked={stop:.5f}"
        )
        return float(stop)
