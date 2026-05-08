"""
Tests for friction (slippage + commission) in scripts/backtest_full.py.

Pins the R-1 invariant: every backtest trade pays realistic friction so
downstream decision gates (M-*, E-*) compare PF against an honest
baseline. Without these tests, a future refactor could silently restore
the pre-R-1 frictionless math.
"""

from __future__ import annotations

import pytest

from scripts.backtest_full import (
    DEFAULT_FRICTION,
    _apply_entry_slippage,
    _apply_exit_slippage,
    _resolve_friction,
)


class TestResolveFriction:
    def test_known_symbol_uses_default(self):
        slip, comm, upl = _resolve_friction("XAUUSD", None)
        assert slip == DEFAULT_FRICTION["XAUUSD"]["slippage_price"]
        assert comm == DEFAULT_FRICTION["XAUUSD"]["commission_per_lot_per_side"]
        assert upl == DEFAULT_FRICTION["XAUUSD"]["units_per_lot"]

    def test_unknown_symbol_falls_back_to_zero(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            slip, comm, upl = _resolve_friction("UNKNOWN_SYM", None)
        assert slip == 0.0
        assert comm == 0.0
        # units_per_lot defaults to 1.0 so commission-per-unit is
        # well-defined even if we can't map to real lots.
        assert upl == 1.0
        assert any("No DEFAULT_FRICTION" in r.message for r in caplog.records)

    def test_explicit_override_wins(self):
        override = {"XAUUSD": {"slippage_price": 99.0, "commission_per_lot_per_side": 7.0, "units_per_lot": 50.0}}
        slip, comm, upl = _resolve_friction("XAUUSD", override)
        assert slip == 99.0
        assert comm == 7.0
        assert upl == 50.0

    def test_wildcard_override_applies_to_all(self):
        override = {"*": {"slippage_price": 1.5, "commission_per_lot_per_side": 3.0, "units_per_lot": 1.0}}
        slip, comm, upl = _resolve_friction("EURUSD", override)
        assert slip == 1.5
        assert comm == 3.0
        assert upl == 1.0

    def test_explicit_symbol_wins_over_wildcard(self):
        override = {
            "*": {"slippage_price": 99.0, "commission_per_lot_per_side": 99.0},
            "XAUUSD": {"slippage_price": 0.5, "commission_per_lot_per_side": 1.0, "units_per_lot": 100.0},
        }
        slip, comm, upl = _resolve_friction("XAUUSD", override)
        assert slip == 0.5
        assert comm == 1.0
        assert upl == 100.0

    def test_partial_override_keys_default_to_zero(self):
        # An override that sets only slippage must not inherit commission
        # from DEFAULT_FRICTION — it's an explicit override, so missing
        # keys are zero (prevents the "I forgot one number" silent bug).
        override = {"XAUUSD": {"slippage_price": 0.5}}
        slip, comm, upl = _resolve_friction("XAUUSD", override)
        assert slip == 0.5
        assert comm == 0.0
        assert upl == 1.0  # default when not specified

    def test_live_config_covers_every_production_symbol(self):
        """DEFAULT_FRICTION must have an entry for every production symbol.
        A new symbol added without a friction entry would silently run
        frictionless, inflating its backtest PF."""
        expected = {"XAUUSD", "EURUSD", "USDJPY", "USDCAD", "ETHUSD"}
        assert expected.issubset(DEFAULT_FRICTION.keys())
        # Every entry must specify units_per_lot — the 2026-04-18 EUR
        # commission-explosion bug happened because this field didn't
        # exist. Pin it so a future entry can't forget.
        for sym in expected:
            assert "units_per_lot" in DEFAULT_FRICTION[sym], (
                f"{sym} is missing units_per_lot — commission math "
                f"breaks without it"
            )
            assert DEFAULT_FRICTION[sym]["units_per_lot"] > 0


class TestApplySlippage:
    def test_entry_buy_pays_more(self):
        assert _apply_entry_slippage(2000.0, "buy", 0.15) == pytest.approx(2000.15)

    def test_entry_sell_receives_less(self):
        assert _apply_entry_slippage(2000.0, "sell", 0.15) == pytest.approx(1999.85)

    def test_exit_buy_closes_lower(self):
        # Closing a long position = selling; trader takes the lower price.
        assert _apply_exit_slippage(2000.0, "buy", 0.15) == pytest.approx(1999.85)

    def test_exit_sell_closes_higher(self):
        # Closing a short position = buying; trader pays the higher price.
        assert _apply_exit_slippage(2000.0, "sell", 0.15) == pytest.approx(2000.15)

    def test_zero_slippage_is_identity(self):
        assert _apply_entry_slippage(1.08500, "buy", 0.0) == 1.08500
        assert _apply_exit_slippage(1.08500, "sell", 0.0) == 1.08500

    def test_round_trip_cost_is_2x_slippage(self):
        # Entry at 2000 + 0.15, exit at 2000 - 0.15 with no move → -0.30 per unit.
        entry = _apply_entry_slippage(2000.0, "buy", 0.15)
        exit_ = _apply_exit_slippage(2000.0, "buy", 0.15)
        # PnL direction: (exit - entry) for a buy.
        assert (exit_ - entry) == pytest.approx(-0.30)


class TestFrictionNeverHelpsTheTrader:
    """Property test: friction is always a cost, never a benefit,
    regardless of direction or price."""

    @pytest.mark.parametrize("direction", ["buy", "sell"])
    @pytest.mark.parametrize("price", [0.5, 1.08500, 150.0, 2000.0, 2500.0])
    @pytest.mark.parametrize("slippage", [0.00005, 0.005, 0.15, 2.0])
    def test_round_trip_is_non_positive(self, direction, price, slippage):
        entry = _apply_entry_slippage(price, direction, slippage)
        # Simulate no-move exit (bar closes at the same price as opened).
        nominal_exit = price
        exit_ = _apply_exit_slippage(nominal_exit, direction, slippage)
        # Pre-friction PnL on a flat bar is 0. Post-friction PnL must be ≤ 0.
        if direction == "buy":
            pnl_per_unit = exit_ - entry
        else:
            pnl_per_unit = entry - exit_
        assert pnl_per_unit <= 0.0
