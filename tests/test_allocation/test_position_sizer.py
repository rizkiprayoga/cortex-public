"""
Tests for PositionSizer — 1% risk formula, strategy-aware sizing.
"""

import pytest

from src.allocation.position_sizer import PositionSizer, SizingResult, SymbolSpec


def xauusd_spec() -> SymbolSpec:
    # MT5 XAUUSD defaults: 100 oz/lot, step 0.01.
    return SymbolSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )


class TestBaseFormula:

    def setup_method(self):
        self.sizer = PositionSizer(max_risk_pct=1.0)

    def test_one_percent_risk_on_xauusd_longs(self):
        # equity 10_000, 1% → $100 risk. Price distance 10 × 100 oz = $1000/lot.
        # base_lot = 100 / 1000 = 0.1 lot. Allocation 1.0 → 0.1.
        result = self.sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0,
            stop_price=1990.0,
            equity=10_000.0,
            allocation_pct=1.0,
        )
        assert result.lot_size == pytest.approx(0.10)
        assert result.risk_amount_usd == pytest.approx(100.0)

    def test_allocation_pct_scales_lot(self):
        # 0.95 of base 0.1 lot → 0.095, floored to step 0.01 → 0.09.
        result = self.sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0,
            stop_price=1990.0,
            equity=10_000.0,
            allocation_pct=0.95,
        )
        assert result.lot_size == pytest.approx(0.09)

    def test_uncertainty_mode_halves_lot(self):
        result = self.sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0,
            stop_price=1990.0,
            equity=10_000.0,
            allocation_pct=1.0,
            uncertainty_mode=True,
        )
        # 0.1 base × 0.5 = 0.05
        assert result.lot_size == pytest.approx(0.05)

    def test_size_multiplier_applies(self):
        # External multiplier 0.5 (e.g. breaker soft halt) halves again.
        result = self.sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0,
            stop_price=1990.0,
            equity=10_000.0,
            allocation_pct=1.0,
            size_multiplier=0.5,
        )
        assert result.lot_size == pytest.approx(0.05)


class TestEdgeRejects:

    def setup_method(self):
        self.sizer = PositionSizer()

    def test_zero_equity_rejects(self):
        result = self.sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0,
            stop_price=1990.0,
            equity=0.0,
            allocation_pct=1.0,
        )
        assert result.lot_size == 0.0
        assert "equity" in result.reason

    def test_entry_equals_stop_rejects(self):
        result = self.sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0,
            stop_price=2000.0,
            equity=10_000.0,
            allocation_pct=1.0,
        )
        assert result.lot_size == 0.0
        assert "undefined risk" in result.reason

    def test_allocation_zero_rejects(self):
        result = self.sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0,
            stop_price=1990.0,
            equity=10_000.0,
            allocation_pct=0.0,
        )
        assert result.lot_size == 0.0

    def test_size_multiplier_zero_rejects(self):
        result = self.sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0,
            stop_price=1990.0,
            equity=10_000.0,
            allocation_pct=1.0,
            size_multiplier=0.0,
        )
        assert result.lot_size == 0.0

    def test_below_volume_min_rejects(self):
        # Tiny equity + wide stop → raw lot below 0.01 step → reject.
        result = self.sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0,
            stop_price=1000.0,   # $100k distance per lot
            equity=100.0,
            allocation_pct=1.0,
        )
        assert result.lot_size == 0.0
        assert "volume_min" in result.reason


class TestShortSideParity:

    def test_sell_with_stop_above_entry_sizes_identically(self):
        sizer = PositionSizer(max_risk_pct=1.0)
        long = sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0,
            stop_price=1990.0,
            equity=10_000.0,
            allocation_pct=1.0,
        )
        short = sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0,
            stop_price=2010.0,
            equity=10_000.0,
            allocation_pct=1.0,
        )
        assert long.lot_size == pytest.approx(short.lot_size)


class TestMinStackingWave6:
    """
    Wave 6 fix #16: multiple size-reduction gates stack via min(),
    not via product. The old code compounded the uncertainty half, the
    circuit-breaker multiplier, and the strategy allocation_pct and
    ended up at 0.24% effective risk in stress — exactly when we need
    to be deploying a reasonable recovery size, not shrinking to
    homeopathic doses.

    Under min-stacking, the effective multiplier equals the tightest
    active gate. The caller (PortfolioManager) is expected to pass an
    already-stacked ``size_multiplier = min(signal.size_discount,
    circuit_breaker.current_size_multiplier())`` — if that caller
    additionally sets ``uncertainty_mode=True`` (the legacy path) the
    sizer must fold the extra 0.5 via ``min()`` rather than
    compounding the halving.
    """

    def setup_method(self):
        self.sizer = PositionSizer(max_risk_pct=1.0)

    def test_uncertainty_with_half_multiplier_yields_half_not_quarter(self):
        """
        Legacy: uncertainty_mode=True AND size_multiplier=0.5 would
        have compounded to 0.25 (×0.5 × ×0.5) → 0.025 lot.
        New:     min(0.5, 0.5) = 0.5 → 0.05 lot.
        """
        result = self.sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0,
            stop_price=1990.0,
            equity=10_000.0,
            allocation_pct=1.0,
            uncertainty_mode=True,
            size_multiplier=0.5,
        )
        assert result.lot_size == pytest.approx(0.05)

    def test_uncertainty_with_strict_multiplier_defers_to_strict(self):
        """
        Tightest gate wins: uncertainty (0.5) + cb multiplier 0.25
        → effective 0.25, not 0.125.
        """
        result = self.sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0,
            stop_price=1990.0,
            equity=10_000.0,
            allocation_pct=1.0,
            uncertainty_mode=True,
            size_multiplier=0.25,
        )
        # base 0.1 lot × min(0.25, 0.5) = 0.025 → floored to 0.02.
        assert result.lot_size == pytest.approx(0.02)

    def test_reason_string_reports_effective_multiplier(self):
        """Reason string now carries the actual effective multiplier."""
        result = self.sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0,
            stop_price=1990.0,
            equity=10_000.0,
            allocation_pct=1.0,
            uncertainty_mode=True,
            size_multiplier=0.5,
        )
        assert "effective_mult=0.50" in result.reason


# -----------------------------------------------------------------------------
# Quote-currency correctness (P1d): sizer must yield correct lots for every
# forex structure, not just XXX/USD pairs. Tests cover both the MT5-tick-value
# path and the fallback (no tick data) path.
# -----------------------------------------------------------------------------

def eurusd_spec_fallback() -> SymbolSpec:
    """EUR/USD: quote currency already USD, fallback math works unchanged."""
    return SymbolSpec(
        symbol="EURUSD", contract_size=100_000.0,
        volume_min=0.01, volume_step=0.01,
        quote_currency="USD",
    )


def usdjpy_spec_fallback() -> SymbolSpec:
    """USD/JPY: quote currency JPY — the previously-buggy case."""
    return SymbolSpec(
        symbol="USDJPY", contract_size=100_000.0,
        volume_min=0.01, volume_step=0.01,
        quote_currency="JPY",
    )


def usdjpy_spec_tick() -> SymbolSpec:
    """USD/JPY with broker-authoritative tick fields (preferred path)."""
    # At price ~155, a 1-pip (0.01) move on 1.0 lot = 1000 JPY = $6.45 USD
    return SymbolSpec(
        symbol="USDJPY", contract_size=100_000.0,
        volume_min=0.01, volume_step=0.01,
        tick_value=6.45, tick_size=0.01, quote_currency="JPY",
    )


def usdcad_spec_fallback() -> SymbolSpec:
    return SymbolSpec(
        symbol="USDCAD", contract_size=100_000.0,
        volume_min=0.01, volume_step=0.01,
        quote_currency="CAD",
    )


class TestQuoteCurrencyCorrectness:
    """Regression against the USDJPY/USDCAD rejection bug.

    Before the fix: USDJPY at price 155 with a 40-pip SL produced
    lot=0.003 which failed the volume_min=0.01 gate, so EVERY USDJPY
    trade got silently skipped in live.

    After the fix: sizer correctly divides the quote-currency PnL by
    entry_price to get USD risk. Same equity/risk%/stop distance now
    yields a sane 0.48 lot on USDJPY.
    """

    def setup_method(self):
        self.sizer = PositionSizer(max_risk_pct=1.25)  # forex default

    # --- Fallback path (no tick_value populated) ---

    def test_eurusd_fallback_unchanged(self):
        """XXX/USD pairs must not regress."""
        # $125 risk, 40-pip SL (0.0040). PnL is already in USD.
        # Expected: 125 / (0.0040 * 100000) = 0.3125 → floor 0.31.
        r = self.sizer.calculate(
            symbol_spec=eurusd_spec_fallback(),
            entry_price=1.0830, stop_price=1.0790,
            equity=10_000.0, allocation_pct=1.0,
        )
        assert r.lot_size == pytest.approx(0.31, abs=0.01)

    def test_usdjpy_fallback_regression(self):
        """The exact scenario from the paper-trading log."""
        # $125 risk, 40-pip SL = 0.40 in price units, price ~155.
        # Before fix: 125 / (0.40 * 100000) = 0.003 → REJECTED
        # After fix:  125 * 155 / (0.40 * 100000) = 0.484 → floor 0.48
        r = self.sizer.calculate(
            symbol_spec=usdjpy_spec_fallback(),
            entry_price=155.00, stop_price=154.60,
            equity=10_000.0, allocation_pct=1.0,
        )
        assert r.lot_size > 0.01, f"USDJPY rejected: {r.reason}"
        assert r.lot_size == pytest.approx(0.48, abs=0.02)

    def test_usdcad_fallback(self):
        """USDCAD: price ~1.36, quote=CAD, same bug structure."""
        # $125 risk, 40-pip SL = 0.0040, price 1.3600.
        # 125 * 1.36 / (0.0040 * 100000) = 0.425 → floor 0.42
        r = self.sizer.calculate(
            symbol_spec=usdcad_spec_fallback(),
            entry_price=1.3600, stop_price=1.3560,
            equity=10_000.0, allocation_pct=1.0,
        )
        assert r.lot_size > 0.01
        assert r.lot_size == pytest.approx(0.42, abs=0.02)

    # --- MT5 tick-value path (preferred when broker data available) ---

    def test_usdjpy_tick_path_matches_fallback(self):
        """Tick-value path (broker-authoritative) should agree with the
        corrected fallback within rounding."""
        fallback = self.sizer.calculate(
            symbol_spec=usdjpy_spec_fallback(),
            entry_price=155.00, stop_price=154.60,
            equity=10_000.0, allocation_pct=1.0,
        )
        tick = self.sizer.calculate(
            symbol_spec=usdjpy_spec_tick(),
            entry_price=155.00, stop_price=154.60,
            equity=10_000.0, allocation_pct=1.0,
        )
        # Tick-value 6.45 comes from 155 exactly; fallback divides by 155.
        # Should match within one volume_step (0.01).
        assert abs(tick.lot_size - fallback.lot_size) <= 0.01

    def test_xauusd_unchanged_by_fix(self):
        """Gold: XAU/USD, quote=USD. Previously-correct math must stay correct."""
        # $150 risk (1.5%), SL $15 distance, contract 100 oz.
        # 150 / (15 * 100) = 0.10
        sizer = PositionSizer(max_risk_pct=1.5)
        r = sizer.calculate(
            symbol_spec=xauusd_spec(),
            entry_price=2000.0, stop_price=1985.0,
            equity=10_000.0, allocation_pct=1.0,
        )
        assert r.lot_size == pytest.approx(0.10, abs=0.01)
