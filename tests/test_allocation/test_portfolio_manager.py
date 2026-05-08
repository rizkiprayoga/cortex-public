"""
Tests for PortfolioManager — pyramiding BE-gate + concurrency caps.

The manager wraps PositionSizer and enforces:
    - max_concurrent_per_symbol (default 3)
    - max_concurrent_total (default 6)
    - pyramiding BE-gate (prior entries must all be risk-free)
    - free-margin reserve ≥ 20%
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from src.allocation.portfolio_manager import OpenPositionView, PortfolioManager
from src.allocation.position_sizer import PositionSizer, SymbolSpec
from src.broker.account_monitor import AccountSnapshot
from src.strategy.base import StrategyDecision


def xauusd_spec() -> SymbolSpec:
    return SymbolSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )


def make_account(
    equity: float = 10_000.0,
    margin: float = 500.0,
    free_margin: float = 9_500.0,
) -> AccountSnapshot:
    return AccountSnapshot(
        balance=equity,
        equity=equity,
        margin=margin,
        free_margin=free_margin,
        margin_level=equity / max(margin, 1e-9) * 100.0,
        floating_pnl=0.0,
        open_positions=0,
    )


def make_signal(direction: str = "buy", entry_price: float = 2000.0):
    return SimpleNamespace(
        should_trade=True,
        direction=direction,
        uncertainty_mode=False,
        entry_price=entry_price,
    )


def make_decision(direction: str = "buy", stop: float = 1990.0) -> StrategyDecision:
    return StrategyDecision(
        strategy_name="LowVolAggressive",
        direction=direction,
        allocation_pct=0.95,
        initial_stop_price=stop,
        atr_trail_mult=3.0,
    )


def make_pm(positions: list[OpenPositionView]) -> PortfolioManager:
    return PortfolioManager(
        sizer=PositionSizer(max_risk_pct=1.0),
        positions_provider=lambda: list(positions),
        symbol_spec_provider=lambda sym: xauusd_spec(),
    )


class TestPyramidingBEGate:
    """
    Wave 6 fix #18: the gate now reads ``tier_1_done`` directly off each
    prior position, not the current stop level. These tests pin the new
    semantic — a stop that happens to sit at entry is NOT sufficient; the
    exit ladder must have actually fired tier 1.
    """

    def test_first_entry_on_clean_symbol_sized_normally(self):
        pm = make_pm([])
        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), make_account(),
        )
        assert result.lot_size > 0.0

    def test_second_entry_blocked_when_prior_tier_1_not_done(self):
        # Prior position exists, tier_1_done flag is False → blocked
        # regardless of where the stop sits.
        prior = OpenPositionView(
            symbol="XAUUSD",
            direction="buy",
            entry_price=2000.0,
            current_stop=1990.0,
            tier_1_done=False,
        )
        pm = make_pm([prior])
        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), make_account(),
        )
        assert result.lot_size == 0.0
        assert result.reason == "pyramiding_blocked_prior_tier_1_not_done"

    def test_second_entry_allowed_when_prior_tier_1_done(self):
        prior = OpenPositionView(
            symbol="XAUUSD",
            direction="buy",
            entry_price=2000.0,
            current_stop=2000.0,
            tier_1_done=True,
        )
        pm = make_pm([prior])
        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), make_account(),
        )
        assert result.lot_size > 0.0

    def test_second_entry_blocked_when_stop_above_entry_but_tier_1_not_fired(self):
        # Wave 6 fix #18 spoof-prevention: a stop ABOVE entry is NOT
        # sufficient — a manual SL drag or reconciled position whose
        # broker-side stop happens to sit above entry must still block
        # pyramiding until tier 1 actually fires.
        prior = OpenPositionView(
            symbol="XAUUSD",
            direction="buy",
            entry_price=2000.0,
            current_stop=2010.0,       # above entry
            tier_1_done=False,          # ...but tier 1 never fired
        )
        pm = make_pm([prior])
        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), make_account(),
        )
        assert result.lot_size == 0.0
        assert result.reason == "pyramiding_blocked_prior_tier_1_not_done"

    def test_short_pyramid_mirrors_long_rule(self):
        prior = OpenPositionView(
            symbol="XAUUSD",
            direction="sell",
            entry_price=2000.0,
            current_stop=2010.0,
            tier_1_done=False,
        )
        pm = make_pm([prior])
        sell_signal = make_signal(direction="sell")
        sell_decision = make_decision(direction="sell", stop=2010.0)
        result = pm.calculate_lot_size(
            "XAUUSD", sell_signal, sell_decision, make_account(),
        )
        assert result.lot_size == 0.0
        assert result.reason == "pyramiding_blocked_prior_tier_1_not_done"

    def test_fourth_entry_on_same_symbol_rejected_by_cap(self):
        priors = [
            OpenPositionView("XAUUSD", "buy", 2000.0, 2020.0, tier_1_done=True),
            OpenPositionView("XAUUSD", "buy", 2005.0, 2015.0, tier_1_done=True),
            OpenPositionView("XAUUSD", "buy", 2010.0, 2010.0, tier_1_done=True),
        ]
        pm = make_pm(priors)
        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), make_account(),
        )
        assert result.lot_size == 0.0
        assert "max_concurrent_per_symbol" in result.reason


class TestGlobalCaps:

    def test_global_cap_blocks_new_entry(self):
        priors = [
            OpenPositionView("XAUUSD", "buy", 2000.0, 2000.0),
            OpenPositionView("XAUUSD", "buy", 2005.0, 2005.0),
            OpenPositionView("BTCUSD", "buy", 60000.0, 60000.0),
            OpenPositionView("BTCUSD", "buy", 61000.0, 61000.0),
            OpenPositionView("BTCUSD", "buy", 62000.0, 62000.0),
            OpenPositionView("EURUSD", "buy", 1.10, 1.10),
        ]
        pm = make_pm(priors)
        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), make_account(),
        )
        assert result.lot_size == 0.0
        assert "max_concurrent_total" in result.reason

    def test_free_margin_reserve_blocks_when_tight(self):
        # equity 10k, free_margin 1k → 10% < 20% reserve
        account = AccountSnapshot(
            balance=10_000.0,
            equity=10_000.0,
            margin=9_000.0,
            free_margin=1_000.0,
            margin_level=111.1,
            floating_pnl=0.0,
            open_positions=0,
        )
        pm = make_pm([])
        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), account,
        )
        assert result.lot_size == 0.0
        assert "free_margin_reserve" in result.reason


class TestSignalHygiene:

    def test_signal_should_trade_false_rejects(self):
        pm = make_pm([])
        signal = make_signal()
        signal.should_trade = False
        result = pm.calculate_lot_size(
            "XAUUSD", signal, make_decision(), make_account(),
        )
        assert result.lot_size == 0.0
        assert "should_trade=False" in result.reason

    def test_invalid_direction_rejects(self):
        pm = make_pm([])
        bad_decision = make_decision()
        bad_decision.direction = "long"
        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), bad_decision, make_account(),
        )
        assert result.lot_size == 0.0
        assert "direction" in result.reason


class TestTotalMarginCap:
    """
    Wave 6 fix #2 — risk_management.md promised a 15% portfolio-wide
    margin cap enforced in calculate_lot_size(), but the field was
    stored and never read. These tests pin the new behavior so the
    cap cannot silently drift back into being a phantom knob.
    """

    def test_total_margin_cap_blocks_when_account_near_cap(self):
        # equity 10k, margin 1.5k → 15% already used. Any new order
        # with any projected margin pushes past the 15% cap.
        account = AccountSnapshot(
            balance=10_000.0,
            equity=10_000.0,
            margin=1_500.0,
            free_margin=8_500.0,       # 85% free → passes reserve check
            margin_level=666.0,
            floating_pnl=0.0,
            open_positions=0,
        )
        pm = make_pm([])
        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), account,
        )
        assert result.lot_size == 0.0
        assert "max_used_margin_pct_total_exceeded" in result.reason

    def test_total_margin_cap_allows_when_headroom_available(self):
        # equity 10k, margin 500 → 5% used. Plenty of headroom under
        # the 15% cap — a normal order should size through.
        account = AccountSnapshot(
            balance=10_000.0,
            equity=10_000.0,
            margin=500.0,
            free_margin=9_500.0,
            margin_level=2000.0,
            floating_pnl=0.0,
            open_positions=0,
        )
        pm = make_pm([])
        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), account,
        )
        assert result.lot_size > 0.0

    def test_total_margin_cap_configurable(self):
        # Custom cap at 8% should reject what a 15% cap would allow.
        account = AccountSnapshot(
            balance=10_000.0,
            equity=10_000.0,
            margin=800.0,              # 8% already
            free_margin=9_200.0,
            margin_level=1250.0,
            floating_pnl=0.0,
            open_positions=0,
        )
        pm = PortfolioManager(
            sizer=PositionSizer(max_risk_pct=1.0),
            positions_provider=lambda: [],
            symbol_spec_provider=lambda sym: xauusd_spec(),
            max_used_margin_pct_total=8.0,
        )
        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), account,
        )
        assert result.lot_size == 0.0
        assert "max_used_margin_pct_total_exceeded" in result.reason


class TestDailyTradesCap:
    """
    Wave 6 fix #3 — risk.max_daily_trades was declared in settings.yaml
    and documented in risk_management.md, but nothing enforced it. The
    rolling 24h counter lives on PortfolioManager and increments only
    on successful sizings.
    """

    def test_hits_cap_after_max_trades(self):
        pm = PortfolioManager(
            sizer=PositionSizer(max_risk_pct=1.0),
            positions_provider=lambda: [],
            symbol_spec_provider=lambda sym: xauusd_spec(),
            max_daily_trades=3,   # small for fast test
        )
        # First 3 succeed...
        for _ in range(3):
            result = pm.calculate_lot_size(
                "XAUUSD", make_signal(), make_decision(), make_account(),
            )
            assert result.lot_size > 0.0
        # ...the 4th is blocked by the soft cap.
        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), make_account(),
        )
        assert result.lot_size == 0.0
        assert "max_daily_trades_reached" in result.reason
        assert "3/3" in result.reason

    def test_cap_resets_after_24h_window(self):
        from collections import deque
        from datetime import datetime, timedelta, timezone

        pm = PortfolioManager(
            sizer=PositionSizer(max_risk_pct=1.0),
            positions_provider=lambda: [],
            symbol_spec_provider=lambda sym: xauusd_spec(),
            max_daily_trades=2,
        )
        # Seed the deque with 2 old timestamps (> 24h old) so the
        # next call should drop them and succeed.
        ancient = datetime.now(tz=timezone.utc) - timedelta(hours=25)
        pm._recent_trade_ts = deque([ancient, ancient], maxlen=100)
        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), make_account(),
        )
        assert result.lot_size > 0.0
        # Both ancient stamps dropped, one new one added.
        assert len(pm._recent_trade_ts) == 1

    def test_rejection_does_not_count_toward_cap(self):
        pm = PortfolioManager(
            sizer=PositionSizer(max_risk_pct=1.0),
            positions_provider=lambda: [],
            symbol_spec_provider=lambda sym: xauusd_spec(),
            max_daily_trades=5,
        )
        # A rejected sizing (should_trade=False) should NOT bump the counter.
        bad = make_signal()
        bad.should_trade = False
        for _ in range(10):
            pm.calculate_lot_size(
                "XAUUSD", bad, make_decision(), make_account(),
            )
        assert len(pm._recent_trade_ts) == 0
        # Five valid sizings in a row now still all succeed.
        for _ in range(5):
            result = pm.calculate_lot_size(
                "XAUUSD", make_signal(), make_decision(), make_account(),
            )
            assert result.lot_size > 0.0


class TestMinStackingWave6:
    """
    Wave 6 fix #16: PortfolioManager must stack ``signal.size_discount``
    with the incoming ``size_multiplier`` via ``min()``. Under the legacy
    compound rule an uncertainty-flagged signal (discount 0.5) stacked
    with a breaker soft halt (0.5) landed at 0.25 effective risk — well
    below the 1% design target, exactly when a sane recovery-sized
    trade is what we want.
    """

    def test_signal_discount_min_stacks_with_cb_multiplier(self):
        pm = make_pm([])
        signal = make_signal()
        signal.size_discount = 0.5    # regime prob below min_confidence
        signal.uncertainty_mode = True  # legacy field; should be ignored

        # Breaker multiplier 0.5 → effective min(0.5, 0.5) = 0.5.
        result = pm.calculate_lot_size(
            "XAUUSD", signal, make_decision(), make_account(),
            size_multiplier=0.5,
        )
        # Without min-stacking the old product rule would give
        # base_lot × alloc × uncertainty × cb = 0.1 × 0.95 × 0.5 × 0.5 = 0.02375
        # → floored to 0.02 lot. With min-stacking we get
        # 0.1 × 0.95 × min(0.5, 0.5) = 0.0475 → floored to 0.04 lot.
        assert result.lot_size == pytest.approx(0.04)

    def test_signal_without_discount_defers_to_cb_multiplier(self):
        pm = make_pm([])
        signal = make_signal()
        # No size_discount field set → default 1.0 → cb multiplier wins.
        result = pm.calculate_lot_size(
            "XAUUSD", signal, make_decision(), make_account(),
            size_multiplier=0.5,
        )
        # 0.1 × 0.95 × min(1.0, 0.5) = 0.0475 → 0.04
        assert result.lot_size == pytest.approx(0.04)


class TestCorrelationBucketCap:
    """
    Wave 6 fix #17: when rolling ρ between XAUUSD and BTCUSD exceeds
    0.6 the two symbols merge into a single 3-slot risk bucket.
    """

    def _seed_correlated_history(self, pm: PortfolioManager, rho_target: float = 0.95):
        """
        Seed both correlation symbols with a strongly correlated series
        so the gate has enough samples and pairwise |ρ| > threshold.
        """
        import numpy as np
        rng = np.random.default_rng(42)
        n = 25
        base = np.cumsum(rng.standard_normal(n))
        # BTC = scaled XAU with small noise → ρ close to 1.
        noise = rng.standard_normal(n) * 0.05
        xau = 2000.0 + base
        btc = 60_000.0 + 30 * base + noise
        for x, b in zip(xau, btc):
            pm.update_daily_close("XAUUSD", float(x))
            pm.update_daily_close("BTCUSD", float(b))

    def _seed_uncorrelated_history(self, pm: PortfolioManager):
        """Independent random walks → |ρ| well below threshold."""
        import numpy as np
        rng = np.random.default_rng(1234)
        xau = 2000.0 + np.cumsum(rng.standard_normal(25))
        btc = 60_000.0 + np.cumsum(rng.standard_normal(25) * 100)
        for x, b in zip(xau, btc):
            pm.update_daily_close("XAUUSD", float(x))
            pm.update_daily_close("BTCUSD", float(b))

    def test_bucket_cap_blocks_fourth_entry(self):
        # 3 correlated-bucket positions already open (2 XAU + 1 BTC).
        # Priors use tier_1_done=True so the pyramiding gate itself
        # doesn't reject — we're isolating the correlation cap.
        priors = [
            OpenPositionView("XAUUSD", "buy", 2000.0, 2010.0, tier_1_done=True),
            OpenPositionView("XAUUSD", "buy", 2005.0, 2015.0, tier_1_done=True),
            OpenPositionView("BTCUSD", "buy", 60_000.0, 60_500.0, tier_1_done=True),
        ]
        pm = PortfolioManager(
            sizer=PositionSizer(max_risk_pct=1.0),
            positions_provider=lambda: list(priors),
            symbol_spec_provider=lambda sym: xauusd_spec(),
            max_concurrent_per_symbol=3,
        )
        self._seed_correlated_history(pm)

        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), make_account(),
        )
        assert result.lot_size == 0.0
        assert "correlation_cap_reached" in result.reason

    def test_bucket_cap_inactive_when_history_too_short(self):
        # Fewer than 20 samples → gate passes through regardless of ρ.
        priors = [
            OpenPositionView("XAUUSD", "buy", 2000.0, 2010.0, tier_1_done=True),
            OpenPositionView("XAUUSD", "buy", 2005.0, 2015.0, tier_1_done=True),
            OpenPositionView("BTCUSD", "buy", 60_000.0, 60_500.0, tier_1_done=True),
        ]
        pm = PortfolioManager(
            sizer=PositionSizer(max_risk_pct=1.0),
            positions_provider=lambda: list(priors),
            symbol_spec_provider=lambda sym: xauusd_spec(),
            max_concurrent_per_symbol=3,
        )
        # Feed only 5 samples to each.
        for i in range(5):
            pm.update_daily_close("XAUUSD", 2000.0 + i)
            pm.update_daily_close("BTCUSD", 60_000.0 + i * 10)

        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), make_account(),
        )
        # Not blocked by correlation — but per-symbol cap (3) still
        # applies because there are already 2 XAU positions.
        # So test the opposite: the rejection reason is NOT the
        # correlation message when the gate is inactive.
        assert "correlation_cap_reached" not in result.reason

    def test_bucket_cap_inactive_when_correlation_low(self):
        # 20+ samples but correlation is weak → cap should not trigger.
        # Use only 2 existing positions so per-symbol / total caps have
        # headroom and wouldn't block a third.
        priors = [
            OpenPositionView("XAUUSD", "buy", 2000.0, 2010.0, tier_1_done=True),
            OpenPositionView("BTCUSD", "buy", 60_000.0, 60_500.0, tier_1_done=True),
        ]
        pm = PortfolioManager(
            sizer=PositionSizer(max_risk_pct=1.0),
            positions_provider=lambda: list(priors),
            symbol_spec_provider=lambda sym: xauusd_spec(),
        )
        self._seed_uncorrelated_history(pm)

        result = pm.calculate_lot_size(
            "XAUUSD", make_signal(), make_decision(), make_account(),
        )
        # Trade should size through — the bucket gate is dormant because
        # the rolling ρ is below the 0.6 threshold.
        assert result.lot_size > 0.0

    def test_non_bucket_symbol_bypasses_gate(self):
        # EURUSD is not in CORRELATION_SYMBOLS → gate is a no-op for
        # that symbol even when XAU/BTC are tightly correlated.
        priors = [
            OpenPositionView("XAUUSD", "buy", 2000.0, 2010.0, tier_1_done=True),
            OpenPositionView("XAUUSD", "buy", 2005.0, 2015.0, tier_1_done=True),
            OpenPositionView("BTCUSD", "buy", 60_000.0, 60_500.0, tier_1_done=True),
        ]
        pm = PortfolioManager(
            sizer=PositionSizer(max_risk_pct=1.0),
            positions_provider=lambda: list(priors),
            # Provide a spec for EURUSD — reuse xauusd_spec shape; the
            # test only cares that the gate short-circuits for
            # non-bucket symbols, not about realistic EUR margin math.
            symbol_spec_provider=lambda sym: xauusd_spec(),
        )
        self._seed_correlated_history(pm)

        result = pm.calculate_lot_size(
            "EURUSD", make_signal(), make_decision(), make_account(),
        )
        # EURUSD isn't in the bucket — gate passes through, order sizes.
        assert result.lot_size > 0.0

    def test_update_daily_close_ignores_unknown_symbols(self):
        pm = make_pm([])
        # Not a bucket symbol → silent no-op.
        pm.update_daily_close("EURUSD", 1.10)
        assert "EURUSD" not in pm._price_history
        # Non-numeric close → logged + ignored, no exception.
        pm.update_daily_close("XAUUSD", "not-a-number")  # type: ignore[arg-type]
        assert len(pm._price_history["XAUUSD"]) == 0
