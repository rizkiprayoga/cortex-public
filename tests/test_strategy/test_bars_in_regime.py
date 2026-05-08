"""Unit tests for the bars_in_regime tracker helper used by E-7."""
import pytest

from src.strategy.trend_mode import RegimeBarTracker


class TestRegimeBarTracker:
    def test_increments_on_identical_regime(self):
        t = RegimeBarTracker()
        assert t.update("XAUUSD", regime_index=3) == 1
        assert t.update("XAUUSD", regime_index=3) == 2
        assert t.update("XAUUSD", regime_index=3) == 3

    def test_resets_to_one_on_flip(self):
        t = RegimeBarTracker()
        t.update("XAUUSD", regime_index=3)
        t.update("XAUUSD", regime_index=3)
        assert t.update("XAUUSD", regime_index=2) == 1  # flip Bull -> Neutral
        assert t.update("XAUUSD", regime_index=2) == 2

    def test_per_symbol_isolation(self):
        t = RegimeBarTracker()
        t.update("XAUUSD", regime_index=3)
        t.update("XAUUSD", regime_index=3)
        # EURUSD has its own counter, untouched by XAUUSD updates
        assert t.update("EURUSD", regime_index=1) == 1
        # XAUUSD's counter still intact
        assert t.update("XAUUSD", regime_index=3) == 3
