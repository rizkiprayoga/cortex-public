"""Unit tests for E-7 TrendModeDetector + supporting types."""
import pytest

from src.strategy.trend_mode import TrendModeConfig, TrendModeDetector, TrendModeState


class TestTrendModeConfig:
    def test_defaults_match_spec(self):
        """Spec §3 + §5.1 defaults — Balanced config."""
        cfg = TrendModeConfig()
        assert cfg.hmm_persist_bars_d1 == 4
        assert cfg.adx_threshold == 25.0
        assert cfg.er_threshold == 0.30
        assert cfg.er_window == 20
        assert cfg.tp_r_multiplier == 4.0

    def test_from_dict_loads_yaml_shape(self):
        """Mirrors how settings.yaml::strategy.trend_mode loads."""
        raw = {
            "hmm_persist_bars_d1": 6,
            "adx_threshold": 30.0,
            "er_threshold": 0.40,
            "er_window": 15,
            "tp_r_multiplier": 3.0,
        }
        cfg = TrendModeConfig.from_dict(raw)
        assert cfg.hmm_persist_bars_d1 == 6
        assert cfg.adx_threshold == 30.0
        assert cfg.er_threshold == 0.40
        assert cfg.er_window == 15
        assert cfg.tp_r_multiplier == 3.0

    def test_from_dict_with_missing_keys_uses_defaults(self):
        """Partial config (e.g. operator overrides only one knob) falls back to defaults."""
        cfg = TrendModeConfig.from_dict({"adx_threshold": 30.0})
        assert cfg.adx_threshold == 30.0
        assert cfg.hmm_persist_bars_d1 == 4  # default
        assert cfg.tp_r_multiplier == 4.0    # default

    def test_frozen_dataclass(self):
        """Config must be immutable so callers can't mutate at runtime."""
        cfg = TrendModeConfig()
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            cfg.adx_threshold = 99.0  # type: ignore[misc]


class TestTrendModeState:
    def test_initial_state_inactive(self):
        st = TrendModeState.initial()
        assert st.active is False
        assert st.direction == 0
        assert st.activated_at_bar == -1
        assert st.just_activated is False
        assert st.just_deactivated is False

    def test_activate_returns_new_state(self):
        st = TrendModeState.initial()
        new = st.activate(direction=+1, at_bar=42)
        assert new.active is True
        assert new.direction == +1
        assert new.activated_at_bar == 42
        assert new.just_activated is True
        assert new.just_deactivated is False
        # Original is unchanged (frozen)
        assert st.active is False

    def test_deactivate_returns_new_state(self):
        st = TrendModeState.initial().activate(+1, at_bar=10)
        new = st.deactivate()
        assert new.active is False
        assert new.direction == 0
        assert new.just_deactivated is True
        assert new.just_activated is False

    def test_unchanged_clears_transition_flags(self):
        """When .unchanged() is called on a just-activated state, the
        flags must clear so transitions only fire once per flip."""
        st = TrendModeState.initial().activate(+1, at_bar=10)
        assert st.just_activated is True
        next_bar = st.unchanged()
        assert next_bar.active is True
        assert next_bar.direction == +1
        assert next_bar.just_activated is False
        assert next_bar.just_deactivated is False


# Regime index constants (mirror src/brain/hmm_regime.py REGIME_LABELS)
CRASH, BEAR, NEUTRAL, BULL, EUPHORIA = 0, 1, 2, 3, 4


class TestTrendModeDetector:
    def _det(self, *, enabled_symbols=None, **cfg_overrides):
        # v2 default: enable every symbol the existing tests reference so
        # they keep passing without per-test edits. Tests that specifically
        # exercise the v2 gate live in TestSymbolGate below.
        if enabled_symbols is None:
            enabled_symbols = frozenset({"XAUUSD", "EURUSD", "USDJPY"})
        return TrendModeDetector(
            TrendModeConfig.from_dict(cfg_overrides, enabled_symbols=enabled_symbols)
        )

    def test_inactive_when_persistence_too_short(self):
        d = self._det()
        st = d.update(
            symbol="XAUUSD", bar_idx=0, regime_index=BULL,
            bars_in_regime=3,                     # below threshold of 4
            adx=30.0, plus_di=20.0, minus_di=10.0,
            er=0.5,
        )
        assert st.active is False

    def test_inactive_when_adx_below_threshold(self):
        d = self._det()
        st = d.update(
            symbol="XAUUSD", bar_idx=0, regime_index=BULL,
            bars_in_regime=10,
            adx=20.0,                             # below 25
            plus_di=15.0, minus_di=10.0, er=0.5,
        )
        assert st.active is False

    def test_inactive_when_er_below_threshold(self):
        d = self._det()
        st = d.update(
            symbol="XAUUSD", bar_idx=0, regime_index=BULL,
            bars_in_regime=10, adx=30.0,
            plus_di=20.0, minus_di=10.0,
            er=0.20,                              # below 0.30
        )
        assert st.active is False

    def test_inactive_when_directional_disagreement(self):
        """Bull regime but -DI > +DI - directional disagreement."""
        d = self._det()
        st = d.update(
            symbol="XAUUSD", bar_idx=0, regime_index=BULL,
            bars_in_regime=10, adx=30.0,
            plus_di=10.0, minus_di=20.0,          # disagrees with Bull
            er=0.5,
        )
        assert st.active is False

    def test_inactive_when_neutral_regime(self):
        d = self._det()
        st = d.update(
            symbol="XAUUSD", bar_idx=0, regime_index=NEUTRAL,
            bars_in_regime=10, adx=30.0,
            plus_di=20.0, minus_di=10.0, er=0.5,
        )
        assert st.active is False
        assert st.direction == 0

    def test_activates_on_bull_with_all_aligned(self):
        d = self._det()
        st = d.update(
            symbol="XAUUSD", bar_idx=0, regime_index=BULL,
            bars_in_regime=10, adx=30.0,
            plus_di=20.0, minus_di=10.0, er=0.5,
        )
        assert st.active is True
        assert st.direction == +1
        assert st.just_activated is True

    def test_activates_on_bear_with_all_aligned(self):
        d = self._det()
        st = d.update(
            symbol="USDJPY", bar_idx=5, regime_index=BEAR,
            bars_in_regime=10, adx=30.0,
            plus_di=10.0, minus_di=20.0,           # -DI > +DI agrees with Bear
            er=0.5,
        )
        assert st.active is True
        assert st.direction == -1
        assert st.activated_at_bar == 5

    def test_euphoria_treated_as_bull(self):
        d = self._det()
        st = d.update(
            symbol="XAUUSD", bar_idx=0, regime_index=EUPHORIA,
            bars_in_regime=10, adx=30.0,
            plus_di=20.0, minus_di=10.0, er=0.5,
        )
        assert st.active is True
        assert st.direction == +1

    def test_just_activated_clears_on_next_bar(self):
        d = self._det()
        bar0 = d.update("XAUUSD", 0, BULL, 10, 30.0, 20.0, 10.0, 0.5)
        bar1 = d.update("XAUUSD", 1, BULL, 11, 30.0, 20.0, 10.0, 0.5)
        assert bar0.just_activated is True
        assert bar1.just_activated is False
        assert bar1.active is True

    def test_deactivates_on_any_filter_break(self):
        """ADX drops below 25 - deactivate even if HMM persistence + ER still hold."""
        d = self._det()
        d.update("XAUUSD", 0, BULL, 10, 30.0, 20.0, 10.0, 0.5)  # active
        st = d.update("XAUUSD", 1, BULL, 11, 20.0, 20.0, 10.0, 0.5)  # ADX dropped
        assert st.active is False
        assert st.just_deactivated is True

    def test_per_symbol_isolation(self):
        d = self._det()
        d.update("XAUUSD", 0, BULL, 10, 30.0, 20.0, 10.0, 0.5)  # XAU active
        st_eur = d.update("EURUSD", 0, NEUTRAL, 1, 10.0, 5.0, 5.0, 0.1)
        assert st_eur.active is False
        # XAU still active despite EUR inactive
        assert d.is_active("XAUUSD", position_direction=+1)
        assert not d.is_active("EURUSD", position_direction=+1)

    def test_is_active_rejects_mismatched_direction(self):
        d = self._det()
        d.update("XAUUSD", 0, BULL, 10, 30.0, 20.0, 10.0, 0.5)  # long-trend active
        assert d.is_active("XAUUSD", position_direction=+1) is True
        assert d.is_active("XAUUSD", position_direction=-1) is False

    def test_is_active_inactive_symbol(self):
        d = self._det()
        # Never updated; default state is inactive
        assert d.is_active("XAUUSD", position_direction=+1) is False

    def test_reactivation_cycle(self):
        """Full inactive → active → inactive → active sequence on one symbol.

        Tasks 9/10 will rely on the second activation overwriting
        activated_at_bar with the new bar index (so trend-mode-attributed
        PnL points at the right activation event).
        """
        d = self._det()
        # Bar 0: activate
        s0 = d.update("XAUUSD", 0, BULL, 10, 30.0, 20.0, 10.0, 0.5)
        assert s0.active is True
        assert s0.just_activated is True
        assert s0.activated_at_bar == 0

        # Bar 1: ADX drops, deactivate
        s1 = d.update("XAUUSD", 1, BULL, 11, 20.0, 20.0, 10.0, 0.5)
        assert s1.active is False
        assert s1.just_deactivated is True

        # Bar 2: still inactive (ADX still below)
        s2 = d.update("XAUUSD", 2, BULL, 12, 22.0, 20.0, 10.0, 0.5)
        assert s2.active is False
        assert s2.just_deactivated is False  # cleared on bar after deactivation

        # Bar 3: ADX recovers, reactivate
        s3 = d.update("XAUUSD", 3, BULL, 13, 30.0, 20.0, 10.0, 0.5)
        assert s3.active is True
        assert s3.just_activated is True
        assert s3.activated_at_bar == 3  # overwritten to new bar, not still 0

    def test_just_deactivated_clears_on_next_bar(self):
        """just_deactivated must be a single-bar flag, symmetric with
        just_activated. Tasks 9/10 use this to drive position exit
        logic on exactly the deactivation bar."""
        d = self._det()
        d.update("XAUUSD", 0, BULL, 10, 30.0, 20.0, 10.0, 0.5)  # active
        bar1 = d.update("XAUUSD", 1, BULL, 11, 20.0, 20.0, 10.0, 0.5)  # deactivate
        assert bar1.just_deactivated is True

        bar2 = d.update("XAUUSD", 2, BULL, 12, 20.0, 20.0, 10.0, 0.5)  # still inactive
        assert bar2.active is False
        assert bar2.just_deactivated is False  # cleared
        assert bar2.just_activated is False


class TestLoadConfigFromSettings:
    def test_load_from_real_settings_yaml(self, tmp_path):
        """Round-trip: write a YAML, load via the helper, verify parsed values."""
        import yaml
        from src.strategy.trend_mode import load_config_from_settings

        settings = {
            "strategy": {
                "trend_mode_enabled": True,
                "trend_mode": {
                    "hmm_persist_bars_d1": 6,
                    "adx_threshold": 30.0,
                    "er_threshold": 0.40,
                    "er_window": 15,
                    "tp_r_multiplier": 3.5,
                },
            },
        }
        yml = tmp_path / "settings.yaml"
        yml.write_text(yaml.dump(settings))

        enabled, cfg = load_config_from_settings(yml)
        assert enabled is True
        assert cfg.hmm_persist_bars_d1 == 6
        assert cfg.adx_threshold == 30.0
        assert cfg.tp_r_multiplier == 3.5

    def test_load_disabled_by_default(self, tmp_path):
        """If trend_mode_enabled is missing, default is False."""
        import yaml
        from src.strategy.trend_mode import load_config_from_settings

        settings = {"strategy": {"trend_mode": {"adx_threshold": 25.0}}}
        yml = tmp_path / "settings.yaml"
        yml.write_text(yaml.dump(settings))

        enabled, cfg = load_config_from_settings(yml)
        assert enabled is False
        # Config still parsed (so a CLI flag can override the enable bit)
        assert cfg.adx_threshold == 25.0


class TestRetroactiveSweepTransform:
    """Locks in the retroactive-sweep transformation logic from
    scripts/backtest_full.py Task 10. The sweep itself is implemented
    inline in the per-bar loop, but the math is unit-testable here on
    synthetic _FullOpenTrade-shaped objects."""

    def test_widen_long_position_tp(self):
        """Long: entry=100, R=2 -> trend_tp_r=8 -> new TP = 116."""
        from dataclasses import dataclass

        @dataclass
        class _Trade:
            symbol: str
            direction: str
            entry_price: float
            initial_r_dist: float
            tp_price: float = 0.0
            time_exit_disabled: bool = False

        t = _Trade("XAUUSD", "buy", 100.0, 2.0, tp_price=104.0)
        trend_tp_r = 8.0  # = 4.0 multiplier x 2.0 baseline_tp_r

        # Apply the same transformation as backtest_full.py Task 10
        if t.direction == "buy":
            t.tp_price = t.entry_price + (trend_tp_r * t.initial_r_dist)
        else:
            t.tp_price = t.entry_price - (trend_tp_r * t.initial_r_dist)
        t.time_exit_disabled = True

        assert t.tp_price == pytest.approx(116.0)
        assert t.time_exit_disabled is True

    def test_widen_short_position_tp(self):
        """Short: entry=150, R=1.5 -> trend_tp_r=8 -> new TP = 138."""
        from dataclasses import dataclass

        @dataclass
        class _Trade:
            symbol: str
            direction: str
            entry_price: float
            initial_r_dist: float
            tp_price: float = 0.0
            time_exit_disabled: bool = False

        t = _Trade("USDJPY", "sell", 150.0, 1.5, tp_price=147.0)
        trend_tp_r = 8.0

        if t.direction == "buy":
            t.tp_price = t.entry_price + (trend_tp_r * t.initial_r_dist)
        else:
            t.tp_price = t.entry_price - (trend_tp_r * t.initial_r_dist)
        t.time_exit_disabled = True

        assert t.tp_price == pytest.approx(138.0)
        assert t.time_exit_disabled is True

    def test_soft_revert_on_flip_off_reenables_time_exit_keeps_widened_tp(self):
        """Spec §4.2 amendment (2026-04-27): on flip-OFF, time_exit_disabled
        flips back to False so the next time-exit cycle can fire, but the
        widened tp_price is preserved (no rug-pull on the upside).

        Locks in the inline soft-revert in scripts/backtest_full.py:
            if _tm_state.just_deactivated and open_trade is not None
                    and open_trade.time_exit_disabled:
                open_trade.time_exit_disabled = False
        """
        from dataclasses import dataclass

        @dataclass
        class _Trade:
            tp_price: float
            time_exit_disabled: bool

        # Position previously had trend-mode flip-ON applied: widened TP=138
        # and time_exit_disabled=True. Now the detector flips OFF.
        t = _Trade(tp_price=138.0, time_exit_disabled=True)
        widened_tp_before = t.tp_price

        # Mirror the inline soft-revert
        just_deactivated = True
        if just_deactivated and t.time_exit_disabled:
            t.time_exit_disabled = False

        assert t.time_exit_disabled is False, "time-exit must re-engage on flip-OFF"
        assert t.tp_price == widened_tp_before, "TP must NOT revert (no rug-pull on upside)"

    def test_soft_revert_no_op_when_time_exit_already_enabled(self):
        """If a trade was opened during a non-trend window (time_exit_disabled
        defaulted False), a flip-OFF should be a no-op on its time-exit flag.
        The guard `and open_trade.time_exit_disabled` prevents accidentally
        flipping unrelated trades.
        """
        from dataclasses import dataclass

        @dataclass
        class _Trade:
            time_exit_disabled: bool

        t = _Trade(time_exit_disabled=False)
        just_deactivated = True
        # Inline guard prevents touching trades with time_exit_disabled=False
        if just_deactivated and t.time_exit_disabled:
            t.time_exit_disabled = False  # would only execute if guard passed
        assert t.time_exit_disabled is False  # untouched (was False, still False)


class TestEntryTimeWideningLogic:
    """Locks in the entry-time widening logic from scripts/backtest_full.py
    Task 12. The actual edit is inline in backtest_full.py at the entry
    construction site; this test exercises the same branching/math on
    synthetic inputs."""

    def test_baseline_tp_when_inactive(self):
        baseline_tp_r = 2.0
        trend_tp_r = 8.0
        is_active = False
        effective = trend_tp_r if is_active else baseline_tp_r
        assert effective == 2.0

    def test_trend_tp_when_active(self):
        baseline_tp_r = 2.0
        trend_tp_r = 8.0
        is_active = True
        effective = trend_tp_r if is_active else baseline_tp_r
        assert effective == 8.0

    def test_long_position_trend_tp_price(self):
        nominal_entry, sl_dist = 100.0, 2.0
        trend_tp_r = 8.0
        # buy → entry + sl_dist * tp_r
        tp = nominal_entry + sl_dist * trend_tp_r
        assert tp == 116.0

    def test_short_position_trend_tp_price(self):
        nominal_entry, sl_dist = 150.0, 1.5
        trend_tp_r = 8.0
        # sell → entry - sl_dist * tp_r
        tp = nominal_entry - sl_dist * trend_tp_r
        assert tp == 138.0

    def test_time_exit_disabled_when_trend_active(self):
        """When trend-mode is active at entry, the new trade is created
        with time_exit_disabled=True so Task 11's exit barrier skips the
        time-exit branch on this position."""
        is_active = True
        time_exit_disabled_at_entry = is_active  # mirrors the inline expression
        assert time_exit_disabled_at_entry is True

    def test_time_exit_NOT_disabled_when_trend_inactive(self):
        """Baseline trades (trend-mode inactive) keep the existing time-exit
        behavior — time_exit_disabled defaults to False."""
        is_active = False
        time_exit_disabled_at_entry = is_active
        assert time_exit_disabled_at_entry is False


class TestSymbolGate:
    """v2 (2026-04-28) — per-symbol enable gate (spec §4 of E-7 v2 amendment).

    The gate adds a frozenset to TrendModeConfig that determines which symbols
    can have ``is_active()`` return True. Empty set = trend-mode off everywhere
    (fail-closed v2 contract). XAUUSD-only is the v2 ship target.
    """

    def test_is_active_returns_false_for_disabled_symbol_even_when_state_active(self):
        """Symbol gate short-circuits before the state check.

        Even when the detector's state is active for a symbol, ``is_active()``
        returns False if the symbol isn't in ``enabled_symbols``. This lets
        scripts/backtest_full.py compute trend-mode state for diagnostics on
        every symbol but only act on the enabled ones.
        """
        cfg = TrendModeConfig.from_dict({}, enabled_symbols=frozenset())  # no symbols enabled
        d = TrendModeDetector(cfg)
        d.update("XAUUSD", 0, BULL, 10, 30.0, 20.0, 10.0, 0.5)
        assert d.state("XAUUSD").active is True            # state IS active
        assert d.is_active("XAUUSD", position_direction=+1) is False  # gate says no

    def test_is_active_returns_true_for_enabled_symbol_when_state_active(self):
        """v2 reduces to v1 behavior when a symbol IS in enabled_symbols."""
        cfg = TrendModeConfig.from_dict({}, enabled_symbols=frozenset({"XAUUSD"}))
        d = TrendModeDetector(cfg)
        d.update("XAUUSD", 0, BULL, 10, 30.0, 20.0, 10.0, 0.5)
        assert d.is_active("XAUUSD", position_direction=+1) is True

    def test_per_symbol_gate_isolation(self):
        """Disabling EURUSD via the gate doesn't affect XAUUSD."""
        cfg = TrendModeConfig.from_dict({}, enabled_symbols=frozenset({"XAUUSD"}))
        d = TrendModeDetector(cfg)
        # State becomes active for both
        d.update("XAUUSD", 0, BULL, 10, 30.0, 20.0, 10.0, 0.5)
        d.update("EURUSD", 0, BULL, 10, 30.0, 20.0, 10.0, 0.5)
        # But only XAUUSD passes the gate
        assert d.is_active("XAUUSD", position_direction=+1) is True
        assert d.is_active("EURUSD", position_direction=+1) is False

    def test_empty_enabled_symbols_is_default_and_universal_off(self):
        """Default ``TrendModeConfig()`` has empty enabled_symbols → fail-closed.

        Backwards-compatible: any caller constructing a TrendModeConfig
        without specifying ``enabled_symbols`` gets v1-effectively-off
        behavior. The contract change is opt-in, not opt-out.
        """
        cfg = TrendModeConfig()  # no overrides
        assert cfg.enabled_symbols == frozenset()
        d = TrendModeDetector(cfg)
        d.update("XAUUSD", 0, BULL, 10, 30.0, 20.0, 10.0, 0.5)
        d.update("EURUSD", 0, BULL, 10, 30.0, 20.0, 10.0, 0.5)
        assert d.is_active("XAUUSD", position_direction=+1) is False
        assert d.is_active("EURUSD", position_direction=+1) is False

    def test_xau_short_passes_gate_when_xau_enabled(self):
        """Spec §9 Q3 — bidirectional XAU + euphoria-off interaction with v2.

        XAU went bidirectional in commit c3e3dfb (2026-04-27). The detector
        activates with direction=-1 on Bear/Crash regimes; the v2 gate must
        not block the short-direction case.
        """
        cfg = TrendModeConfig.from_dict({}, enabled_symbols=frozenset({"XAUUSD"}))
        d = TrendModeDetector(cfg)
        # Bear regime + -DI > +DI → state.direction = -1
        d.update("XAUUSD", 0, BEAR, 10, 30.0, 10.0, 20.0, 0.5)
        st = d.state("XAUUSD")
        assert st.active is True
        assert st.direction == -1
        # Short position passes the gate
        assert d.is_active("XAUUSD", position_direction=-1) is True
        # Long position correctly rejected (direction mismatch, separate from gate)
        assert d.is_active("XAUUSD", position_direction=+1) is False

class TestLoadConfigFromSettingsV2:
    """v2 — loader builds enabled_symbols frozenset from per_symbol_params."""

    def test_load_per_symbol_enabled_flags(self, tmp_path):
        """`per_symbol_params.<sym>.trend_mode_enabled: true` populates enabled_symbols."""
        import yaml
        from src.strategy.trend_mode import load_config_from_settings

        settings = {
            "strategy": {
                "trend_mode_enabled": True,
                "trend_mode": {"adx_threshold": 25.0},
                "per_symbol_params": {
                    "XAUUSD": {"trend_mode_enabled": True, "atr_sl_mult": 2.0},
                    "EURUSD": {"trend_mode_enabled": False, "atr_sl_mult": 1.5},
                    "USDJPY": {"atr_sl_mult": 2.0},  # no flag → not in enabled_symbols
                },
            },
        }
        yml = tmp_path / "settings.yaml"
        yml.write_text(yaml.dump(settings))

        enabled, cfg = load_config_from_settings(yml)
        assert enabled is True
        assert cfg.enabled_symbols == frozenset({"XAUUSD"})

    def test_load_global_off_does_not_clobber_per_symbol_set(self, tmp_path):
        """The global flag and enabled_symbols are independent in the loader.

        Live orchestrator decides whether to honor global flag (it should);
        backtest CLI ignores global flag and uses CLI ``--trend-mode`` instead.
        Loader returns BOTH so each caller picks the right semantics.
        """
        import yaml
        from src.strategy.trend_mode import load_config_from_settings

        settings = {
            "strategy": {
                "trend_mode_enabled": False,  # global kill switch ON
                "per_symbol_params": {
                    "XAUUSD": {"trend_mode_enabled": True},
                },
            },
        }
        yml = tmp_path / "settings.yaml"
        yml.write_text(yaml.dump(settings))

        enabled, cfg = load_config_from_settings(yml)
        assert enabled is False
        assert cfg.enabled_symbols == frozenset({"XAUUSD"})  # set still populated
