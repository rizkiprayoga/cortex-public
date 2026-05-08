"""
Tests for the ``_reconcile_tracked_positions`` helper in main.py.

Wave 4 fix: positions that survived a prior bot run (crash, Ctrl+C,
internet outage) must be re-adopted into the in-memory tracking dict
at startup — otherwise the 3-tier exit ladder stays dormant for them
until they hit their broker-side SL naturally. The broker's SL is
itself a safety net, but we want to keep partials, trailing, and the
reversal-flicker exit active for reconciled positions too.
"""

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub out MetaTrader5 if it's not importable in this environment — the
# main.py module imports several modules that eventually pull in MT5.
# We substitute a MagicMock so the helper's local `import MetaTrader5`
# lands on our mock.
if "MetaTrader5" not in sys.modules:
    sys.modules["MetaTrader5"] = MagicMock()

import main as main_module  # noqa: E402


def _fake_mt5_position(
    ticket: int,
    symbol: str,
    pos_type: int,
    price_open: float,
    sl: float,
    volume: float,
):
    """
    Build a minimal duck-typed position object. MT5's ``positions_get``
    returns a tuple of C-struct-like objects exposing fields as
    attributes — any object with the right attributes works for us.
    """
    p = types.SimpleNamespace()
    p.ticket = ticket
    p.symbol = symbol
    p.type = pos_type          # 0 = BUY, 1 = SELL
    p.price_open = price_open
    p.sl = sl
    p.volume = volume
    return p


class TestReconcileTrackedPositions:

    def test_reconciles_two_buy_and_sell(self):
        """Both buy and sell tickets land in tracked_positions with correct fields."""
        positions = (
            _fake_mt5_position(111, "XAUUSD", pos_type=0, price_open=2000.0, sl=1980.0, volume=0.10),
            _fake_mt5_position(222, "BTCUSD", pos_type=1, price_open=60000.0, sl=62000.0, volume=0.05),
        )

        # The helper does `import MetaTrader5 as mt5` locally — patch the
        # positions_get call on whatever MetaTrader5 module is in sys.modules.
        mt5_mod = sys.modules["MetaTrader5"]
        with patch.object(mt5_mod, "positions_get", return_value=positions, create=True):
            tracked: dict = {}
            count = main_module._reconcile_tracked_positions(tracked)

        assert count == 2
        assert set(tracked.keys()) == {111, 222}

        buy = tracked[111]
        assert buy.symbol == "XAUUSD"
        assert buy.direction == "buy"
        assert buy.entry_price == 2000.0
        assert buy.initial_stop == 1980.0
        assert buy.current_stop == 1980.0
        assert buy.volume == 0.10
        assert buy.initial_volume == 0.10
        assert buy.be_locked is False
        assert buy.tier_1_done is False  # property maps to be_locked
        assert buy.strategy_name == "reconciled"

        sell = tracked[222]
        assert sell.direction == "sell"
        assert sell.entry_price == 60000.0
        assert sell.initial_stop == 62000.0

    def test_skips_invalid_fields(self):
        """
        Positions with zero entry / SL / volume are skipped with a warning
        — we never want to build an OpenPosition with an initial_stop of 0
        because R-multiple math would divide by zero on the first tick.
        """
        positions = (
            _fake_mt5_position(111, "XAUUSD", 0, price_open=0.0, sl=1980.0, volume=0.10),
            _fake_mt5_position(222, "XAUUSD", 0, price_open=2000.0, sl=0.0,    volume=0.10),
            _fake_mt5_position(333, "XAUUSD", 0, price_open=2000.0, sl=1980.0, volume=0.0),
            _fake_mt5_position(444, "XAUUSD", 0, price_open=2000.0, sl=1980.0, volume=0.10),
        )

        mt5_mod = sys.modules["MetaTrader5"]
        with patch.object(mt5_mod, "positions_get", return_value=positions, create=True):
            tracked: dict = {}
            count = main_module._reconcile_tracked_positions(tracked)

        # Only the last one (444) is valid.
        assert count == 1
        assert set(tracked.keys()) == {444}

    def test_does_not_overwrite_existing_entries(self):
        """
        If a ticket is already tracked (e.g. the bot opened it mid-session
        and we're reconciling after a reconnect), the reconcile helper
        must leave the existing entry alone — overwriting would clobber
        tier_1_done / tier_2_done flags that only the in-memory loop knows.
        """
        # Build an existing position with tier_1_done=True to prove
        # the reconcile helper respects it.
        from src.strategy.exit_manager import OpenPosition

        existing = OpenPosition(
            symbol="XAUUSD",
            ticket=111,
            direction="buy",
            entry_price=2000.0,
            initial_stop=1980.0,
            current_stop=2000.0,  # moved to BE
            volume=0.067,          # already partialed
            initial_volume=0.10,
            atr_trail_mult=2.0,
            strategy_name="LowVolAggressive",
            be_locked=True,
        )
        tracked = {111: existing}

        positions = (
            _fake_mt5_position(111, "XAUUSD", 0, price_open=2000.0, sl=2000.0, volume=0.067),
            _fake_mt5_position(222, "XAUUSD", 0, price_open=2010.0, sl=1990.0, volume=0.05),
        )

        mt5_mod = sys.modules["MetaTrader5"]
        with patch.object(mt5_mod, "positions_get", return_value=positions, create=True):
            count = main_module._reconcile_tracked_positions(tracked)

        # Only the new ticket was added.
        assert count == 1
        assert 222 in tracked
        # Existing entry is untouched.
        assert tracked[111] is existing
        assert tracked[111].tier_1_done is True
        assert tracked[111].strategy_name == "LowVolAggressive"

    def test_empty_positions_returns_zero(self):
        """positions_get returning None or empty tuple → 0 added, no crash."""
        mt5_mod = sys.modules["MetaTrader5"]

        with patch.object(mt5_mod, "positions_get", return_value=None, create=True):
            tracked: dict = {}
            count = main_module._reconcile_tracked_positions(tracked)
            assert count == 0
            assert tracked == {}

        with patch.object(mt5_mod, "positions_get", return_value=(), create=True):
            tracked = {}
            count = main_module._reconcile_tracked_positions(tracked)
            assert count == 0
            assert tracked == {}

    def test_survives_mt5_exception(self):
        """If mt5.positions_get raises, the helper returns 0 and logs — never crashes startup."""
        mt5_mod = sys.modules["MetaTrader5"]
        with patch.object(
            mt5_mod,
            "positions_get",
            side_effect=RuntimeError("terminal dropped"),
            create=True,
        ):
            tracked: dict = {}
            count = main_module._reconcile_tracked_positions(tracked)
            assert count == 0
            assert tracked == {}
