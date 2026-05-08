"""
high_vol_defensive.py — Defensive strategy for high-volatility regimes.

Used when the current HMM regime sits in the highest third of the
volatility distribution across all trained regimes. In high vol:
  - stops need to be wider in absolute terms to avoid noise stopouts
  - but sizing must come down so the wider stop doesn't translate into
    more than 1% risk (the risk budget is fixed, only the lot scales)
  - the runner trails tighter (1.5 × ATR) because reversals are faster

Initial stop rule
-----------------
  buy:  stop = EMA50 − 1.0 × ATR
  sell: stop = EMA50 + 1.0 × ATR
"""

from src.strategy.base import BaseStrategy, MarketContext


class HighVolDefensiveStrategy(BaseStrategy):

    name: str = "HighVolDefensive"
    # PLACEHOLDERS — tuned production values redacted from this public template.
    allocation_pct: float = 0.0
    atr_trail_mult: float = 0.0
    _EMA_ATR_MULT: float = 0.0       # tune the ATR multiplier for the EMA-anchored stop

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
            f"high_vol_stop: {direction} — EMA50 ± {self._EMA_ATR_MULT:.2f}×ATR = {stop:.5f}"
        )
        return float(stop)
