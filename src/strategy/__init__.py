"""
src.strategy — Strategy layer for the HMM + LSTM trading system.

The strategy layer sits between the Brain (which decides WHETHER and IN WHICH
DIRECTION to trade) and the Allocation layer (which decides HOW MUCH).

Its job is to pick a concrete *sizing/stop profile* based on the current
HMM regime's volatility, and to manage open positions via an R-based
partial-exit ladder.

Exports
-------
- ``BaseStrategy``       — abstract base class for volatility-ranked strategies
- ``StrategyDecision``   — immutable decision record returned by a strategy
- ``MarketContext``      — current market snapshot (price, ATR, EMA50) passed in
- ``LowVolAggressiveStrategy``
- ``MidVolCautiousStrategy``
- ``HighVolDefensiveStrategy``
- ``StrategyOrchestrator`` — vol-rank → strategy class mapper
- ``ExitManager``        — 3-tier partial exit + trailing stop + reversal exit
- ``OpenPosition``       — state carrier for positions tracked by ExitManager
- ``ExitAction``         — instruction record produced by ExitManager.check_exits
"""

from src.strategy.base import (
    BaseStrategy,
    MarketContext,
    StrategyDecision,
)
from src.strategy.low_vol_aggressive import LowVolAggressiveStrategy
from src.strategy.mid_vol_cautious import MidVolCautiousStrategy
from src.strategy.high_vol_defensive import HighVolDefensiveStrategy
from src.strategy.orchestrator import StrategyOrchestrator
from src.strategy.exit_manager import ExitAction, ExitManager, OpenPosition

__all__ = [
    "BaseStrategy",
    "MarketContext",
    "StrategyDecision",
    "LowVolAggressiveStrategy",
    "MidVolCautiousStrategy",
    "HighVolDefensiveStrategy",
    "StrategyOrchestrator",
    "ExitManager",
    "OpenPosition",
    "ExitAction",
]
