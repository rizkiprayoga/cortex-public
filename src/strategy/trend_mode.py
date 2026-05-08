"""E-7 trend-mode detector and supporting helpers.

This module is the single source of truth for trend-mode logic shared by
the backtest path and the (future v1.5) live path. See spec at
docs/superpowers/specs/2026-04-26-e7-trend-mode-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


class RegimeBarTracker:
    """Tracks consecutive bars at the same HMM regime_index, per symbol.

    HMM inference is stateless per call; this consumer-side tracker counts
    persistence so the detector can apply the spec's §3.1 ≥4 D1-bar rule.
    """

    def __init__(self) -> None:
        self._last: Dict[str, int] = {}      # symbol -> last regime_index
        self._count: Dict[str, int] = {}     # symbol -> bars in current regime

    def update(self, symbol: str, regime_index: int) -> int:
        """Record a new bar and return the resulting bars_in_regime count."""
        if self._last.get(symbol) == regime_index:
            self._count[symbol] = self._count.get(symbol, 0) + 1
        else:
            self._count[symbol] = 1
        self._last[symbol] = regime_index
        return self._count[symbol]

    def current(self, symbol: str) -> int:
        """Return the current bars_in_regime count without updating."""
        return self._count.get(symbol, 0)


@dataclass(frozen=True)
class TrendModeConfig:
    """Frozen config for E-7 trend-mode detector. Defaults = Balanced (spec §3).

    Loaded from `settings.yaml::strategy.trend_mode`. Single source of truth
    so backtest and live cannot drift.

    v2 (2026-04-28): per-symbol opt-in via `enabled_symbols` frozenset. Empty
    set (default) = trend-mode off everywhere — fail-closed. Populated by
    `load_config_from_settings` from `strategy.per_symbol_params.<sym>.
    trend_mode_enabled` flags.
    """
    # PLACEHOLDERS — tuned production values redacted from this public template.
    hmm_persist_bars_d1: int = 0
    adx_threshold: float = 0.0
    er_threshold: float = 0.0
    er_window: int = 0
    tp_r_multiplier: float = 0.0
    enabled_symbols: frozenset = frozenset()  # per-symbol opt-in gate

    @classmethod
    def from_dict(
        cls, raw: dict, enabled_symbols: frozenset | None = None,
    ) -> "TrendModeConfig":
        """Load from a YAML-shaped dict, falling back to defaults for missing keys.

        ``enabled_symbols`` is supplied separately because it lives under
        ``strategy.per_symbol_params.<sym>`` rather than
        ``strategy.trend_mode``. Pass None to keep the default (empty set =
        trend-mode off everywhere — fail-closed v2 contract).
        """
        defaults = cls()
        return cls(
            hmm_persist_bars_d1=int(raw.get("hmm_persist_bars_d1", defaults.hmm_persist_bars_d1)),
            adx_threshold=float(raw.get("adx_threshold", defaults.adx_threshold)),
            er_threshold=float(raw.get("er_threshold", defaults.er_threshold)),
            er_window=int(raw.get("er_window", defaults.er_window)),
            tp_r_multiplier=float(raw.get("tp_r_multiplier", defaults.tp_r_multiplier)),
            enabled_symbols=(
                enabled_symbols if enabled_symbols is not None else defaults.enabled_symbols
            ),
        )


@dataclass(frozen=True)
class TrendModeState:
    """Per-symbol trend-mode state. Immutable; each transition returns a new instance."""
    active: bool
    direction: int                 # +1 long, -1 short, 0 inactive
    activated_at_bar: int          # -1 if never activated
    just_activated: bool           # True for the single bar of activation
    just_deactivated: bool         # True for the single bar of deactivation

    @classmethod
    def initial(cls) -> "TrendModeState":
        return cls(
            active=False,
            direction=0,
            activated_at_bar=-1,
            just_activated=False,
            just_deactivated=False,
        )

    def activate(self, direction: int, at_bar: int) -> "TrendModeState":
        """Transition inactive→active. Returns a new state."""
        return TrendModeState(
            active=True,
            direction=direction,
            activated_at_bar=at_bar,
            just_activated=True,
            just_deactivated=False,
        )

    def deactivate(self) -> "TrendModeState":
        """Transition active→inactive. Returns a new state."""
        return TrendModeState(
            active=False,
            direction=0,
            activated_at_bar=self.activated_at_bar,  # preserve for diagnostics
            just_activated=False,
            just_deactivated=True,
        )

    def unchanged(self) -> "TrendModeState":
        """Same activation status as last bar; transition flags cleared."""
        return TrendModeState(
            active=self.active,
            direction=self.direction,
            activated_at_bar=self.activated_at_bar,
            just_activated=False,
            just_deactivated=False,
        )


# Regime index constants - mirror src/brain/hmm_regime.py REGIME_LABELS.
# 0=Crash, 1=Bear, 2=Neutral, 3=Bull, 4=Euphoria
_BULL_REGIMES = frozenset({3, 4})  # Bull, Euphoria
_BEAR_REGIMES = frozenset({0, 1})  # Crash, Bear


def _regime_dir(regime_index: int) -> int:
    """Map HMM regime_index -> trend direction. 0 = no trend (Neutral or unknown)."""
    if regime_index in _BULL_REGIMES:
        return +1
    if regime_index in _BEAR_REGIMES:
        return -1
    return 0


class TrendModeDetector:
    """E-7 trend-mode state machine, per symbol.

    Activation = AND of 4 conditions (spec §3.4):
        regime_dir != 0
        AND bars_in_regime >= cfg.hmm_persist_bars_d1
        AND adx > cfg.adx_threshold
        AND directional_agreement(+DI/-DI, regime_dir)
        AND er > cfg.er_threshold

    Deactivation = NOT Activation (any single conjunct fails).
    """

    def __init__(self, config: TrendModeConfig) -> None:
        self.config = config
        self._states: Dict[str, TrendModeState] = {}

    def _state_for(self, symbol: str) -> TrendModeState:
        return self._states.get(symbol, TrendModeState.initial())

    def update(
        self,
        symbol: str,
        bar_idx: int,
        regime_index: int,
        bars_in_regime: int,
        adx: float,
        plus_di: float,
        minus_di: float,
        er: float,
    ) -> TrendModeState:
        """Compute the new state for this bar and persist it. Returns new state."""
        cfg = self.config
        prev = self._state_for(symbol)
        regime_dir = _regime_dir(regime_index)

        # Directional agreement: +DI > -DI for long-trend, -DI > +DI for short-trend
        directional_ok = (
            (regime_dir == +1 and plus_di > minus_di)
            or (regime_dir == -1 and minus_di > plus_di)
        )

        should_activate = (
            regime_dir != 0
            and bars_in_regime >= cfg.hmm_persist_bars_d1
            and adx > cfg.adx_threshold
            and directional_ok
            and er > cfg.er_threshold
        )

        if should_activate and not prev.active:
            new_state = prev.activate(direction=regime_dir, at_bar=bar_idx)
        elif not should_activate and prev.active:
            new_state = prev.deactivate()
        else:
            new_state = prev.unchanged()

        self._states[symbol] = new_state
        return new_state

    def is_active(self, symbol: str, position_direction: int) -> bool:
        """True iff trend-mode is enabled for ``symbol`` AND state is active AND direction matches.

        v2 gate: short-circuits to False if ``symbol`` is not in
        ``self.config.enabled_symbols``. Empty enabled_symbols (default) =
        trend-mode off everywhere — preserves backwards compatibility for
        any caller that constructs a TrendModeConfig without specifying
        the field.
        """
        if symbol.upper() not in self.config.enabled_symbols:
            return False
        st = self._state_for(symbol)
        return st.active and st.direction == position_direction

    def state(self, symbol: str) -> TrendModeState:
        """Return the current state for `symbol` (initial-state if never updated)."""
        return self._state_for(symbol)

    def snapshot(self) -> Dict[str, dict]:
        """Diagnostic snapshot of all per-symbol states.

        Includes the per-bar transition flags (just_activated/just_deactivated)
        so consumers (Task 14 diagnostic JSON) can attribute trend-mode events
        to specific bars without re-walking the per-bar update sequence.
        """
        return {
            sym: {
                "active": s.active,
                "direction": s.direction,
                "activated_at_bar": s.activated_at_bar,
                "just_activated": s.just_activated,
                "just_deactivated": s.just_deactivated,
            }
            for sym, s in self._states.items()
        }


def load_config_from_settings(settings_path) -> tuple[bool, TrendModeConfig]:
    """Load (global_enabled_flag, config) from a settings.yaml file.

    The returned ``config.enabled_symbols`` is built from the per-symbol
    ``trend_mode_enabled`` flags under ``strategy.per_symbol_params.<sym>``.

    The first tuple element is the *global* kill-switch
    (``strategy.trend_mode_enabled``). The live orchestrator (v1.5+) uses
    the global flag as its enable gate; the backtest CLI uses its
    ``--trend-mode`` flag instead and ignores the global flag.

    Returns (False, defaults_with_empty_enabled_symbols) if the trend_mode
    block is absent — fail-closed v2 contract.
    """
    import yaml
    from pathlib import Path

    raw = yaml.safe_load(Path(settings_path).read_text())
    strategy = (raw or {}).get("strategy", {}) or {}
    global_enabled = bool(strategy.get("trend_mode_enabled", False))
    cfg_raw = strategy.get("trend_mode", {}) or {}

    # v2: build per-symbol opt-in frozenset from per_symbol_params.<sym>.trend_mode_enabled.
    # Both spellings (slashless / with-slash) are added so consumers don't need to
    # normalize before checking. Mirrors src/data_pipeline/fundamental/_currency_exposure.py.
    per_symbol = strategy.get("per_symbol_params", {}) or {}
    enabled: set[str] = set()
    for sym, params in per_symbol.items():
        if not isinstance(params, dict):
            continue
        if params.get("trend_mode_enabled", False):
            enabled.add(sym.upper())
    enabled_symbols = frozenset(enabled)

    cfg = TrendModeConfig.from_dict(cfg_raw, enabled_symbols=enabled_symbols)
    return global_enabled, cfg
