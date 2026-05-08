"""Tests for SignalCombiner."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.brain.signal_combiner import SignalCombiner, SignalResult
from src.brain.hmm_regime import RegimeResult


def make_regime(label: str, multiplier: float, prob: float = 0.85) -> RegimeResult:
    idx = {"Crash": 0, "Bear": 1, "Neutral": 2, "Bull": 3, "Euphoria": 4}[label]
    probs = np.zeros(5)
    probs[idx] = prob
    probs[(idx + 1) % 5] = 1.0 - prob
    return RegimeResult(
        symbol="XAUUSD",
        regime_index=idx,
        regime_label=label,
        state_probability=prob,
        position_multiplier=multiplier,
        all_probabilities=probs,
    )


class TestSignalCombiner:

    def setup_method(self):
        self.hmm = MagicMock()
        self.lstm = MagicMock()
        self.combiner = SignalCombiner(self.hmm, self.lstm, hmm_weight=0.5, lstm_weight=0.5, signal_threshold=0.5)
        self.features = np.random.randn(60, 10)

    def test_no_trade_in_crash_regime(self):
        """Crash regime should always produce should_trade=False."""
        self.hmm.predict.return_value = make_regime("Crash", 0.0)
        self.lstm.predict.return_value = 0.05   # Positive prediction
        result = self.combiner.get_signal("XAUUSD", self.features)
        assert isinstance(result, SignalResult)
        assert result.should_trade is False

    def test_meta_labeler_shadow_mode_logs_but_does_not_block(self, monkeypatch):
        """Shadow mode: the gate computes P(win) and appends a reasoning
        line with WOULD_{ALLOW,BLOCK} but MUST NOT set should_trade=False.
        """
        # Enable shadow, disable the active gate
        monkeypatch.setenv("CORTEX_META_LABELER", "0")
        monkeypatch.setenv("CORTEX_META_LABELER_SHADOW", "1")

        # Short-circuit flicker gate so the meta-labeler block is reached
        combiner = SignalCombiner(
            self.hmm, self.lstm,
            hmm_weight=0.5, lstm_weight=0.5, signal_threshold=0.5,
            flicker_bars_required=1, long_only_mode=False,
        )
        # Stub the gate to force a BLOCK decision (proba < threshold)
        combiner._meta_labeler_gate = MagicMock(return_value=(
            False, 0.20,
            "meta_labeler: P(win)=0.200 < 0.50 → BLOCK",
        ))

        # Build a signal that passes all upstream gates
        self.hmm.predict.return_value = make_regime("Bull", 1.0, prob=0.99)
        self.lstm.predict.return_value = 0.05

        result = combiner.get_signal("XAUUSD", self.features)

        # Shadow mode MUST NOT turn should_trade off
        assert result.should_trade is True, (
            "shadow mode must not block trades — only log what it "
            "WOULD have done. reasoning=" + " | ".join(result.reasoning)
        )
        # Reasoning must contain the shadow-mode WOULD_BLOCK tag
        joined = " | ".join(result.reasoning)
        assert "meta_labeler_shadow" in joined
        assert "WOULD_BLOCK" in joined

    def test_meta_labeler_active_mode_blocks_trade(self, monkeypatch):
        """When CORTEX_META_LABELER=1, a negative labeler decision flips
        should_trade to False."""
        monkeypatch.setenv("CORTEX_META_LABELER", "1")
        monkeypatch.delenv("CORTEX_META_LABELER_SHADOW", raising=False)

        combiner = SignalCombiner(
            self.hmm, self.lstm,
            hmm_weight=0.5, lstm_weight=0.5, signal_threshold=0.5,
            flicker_bars_required=1, long_only_mode=False,
        )
        combiner._meta_labeler_gate = MagicMock(return_value=(
            False, 0.20,
            "meta_labeler: P(win)=0.200 < 0.50 → BLOCK",
        ))
        self.hmm.predict.return_value = make_regime("Bull", 1.0, prob=0.99)
        self.lstm.predict.return_value = 0.05

        result = combiner.get_signal("XAUUSD", self.features)
        assert result.should_trade is False
        joined = " | ".join(result.reasoning)
        assert "meta_labeler:" in joined   # not the shadow variant
        assert "BLOCK" in joined
        assert "WOULD_" not in joined

    def test_meta_labeler_both_flags_active_wins(self, monkeypatch):
        """If both env flags are set, active gate takes precedence and
        blocks trades (shadow mode is ignored)."""
        monkeypatch.setenv("CORTEX_META_LABELER", "1")
        monkeypatch.setenv("CORTEX_META_LABELER_SHADOW", "1")

        combiner = SignalCombiner(
            self.hmm, self.lstm,
            hmm_weight=0.5, lstm_weight=0.5, signal_threshold=0.5,
        )
        assert combiner._meta_labeler_enabled is True
        assert combiner._meta_labeler_shadow is False   # suppressed

    def test_bull_regime_with_positive_lstm_gives_buy(self):
        """Bull regime + positive LSTM prediction should yield a buy signal."""
        self.hmm.predict.return_value = make_regime("Bull", 0.75)
        self.lstm.predict.return_value = 0.02   # Positive return predicted
        result = self.combiner.get_signal("XAUUSD", self.features)
        if result.should_trade:
            assert result.direction == "buy"

    def test_bear_regime_with_negative_lstm_gives_sell(self):
        """Bear regime + negative LSTM prediction should yield a sell signal."""
        self.hmm.predict.return_value = make_regime("Bear", 0.25)
        self.lstm.predict.return_value = -0.03
        result = self.combiner.get_signal("XAUUSD", self.features)
        if result.should_trade:
            assert result.direction == "sell"

    def test_combined_score_is_bounded(self):
        """combined_score should always be in [-1, 1]."""
        self.hmm.predict.return_value = make_regime("Bull", 0.75)
        self.lstm.predict.return_value = 0.10
        result = self.combiner.get_signal("XAUUSD", self.features)
        assert -1.0 <= result.combined_score <= 1.0

    def test_last_signal_is_cached_after_fuse(self):
        """
        Wave 5: SignalCombiner.last_signal must expose the most recent
        SignalResult so RiskMonitor.attach_signal_ref() can snapshot the
        active regime + lstm prediction into the circuit-breaker audit
        row at trip time. Before Wave 5 this attribute didn't exist and
        the attach_signal_ref lambda crashed silently.
        """
        # No signal yet before the first call.
        assert self.combiner.last_signal is None

        self.hmm.predict.return_value = make_regime("Bull", 0.75)
        self.lstm.predict.return_value = 0.02
        result = self.combiner.get_signal("XAUUSD", self.features)

        assert self.combiner.last_signal is result
        # RiskMonitor's audit reads these exact fields — pin the shape.
        assert self.combiner.last_signal.regime.regime_label == "Bull"
        assert hasattr(self.combiner.last_signal, "lstm_prediction")

    def test_last_signal_updates_on_each_call(self):
        """Each fusion call overwrites last_signal — stale context never lingers."""
        self.hmm.predict.return_value = make_regime("Bull", 0.75)
        self.lstm.predict.return_value = 0.02
        first = self.combiner.get_signal("XAUUSD", self.features)

        self.hmm.predict.return_value = make_regime("Bear", 0.25)
        self.lstm.predict.return_value = -0.03
        second = self.combiner.get_signal("XAUUSD", self.features)

        assert self.combiner.last_signal is second
        assert self.combiner.last_signal is not first
        assert self.combiner.last_signal.regime.regime_label == "Bear"

    def test_reset_state_clears_trading_state_preserves_display_cache(self):
        """
        ``reset_state()`` must drop the trading-safety state (flicker
        ring + per-symbol memo) so the first post-halt / post-switch
        bar has to wait ``flicker_bars_required`` fresh bars before an
        entry can fire (Wave 6 fix #10).

        It must PRESERVE the display cache (``last_signal``,
        ``last_signal_by_symbol``). Those represent market state
        (regime + combined score from OHLCV), not trading state —
        blanking them on account switch left the dashboard's regime /
        signal cards empty until the next H4 tick, a real UX bug with
        zero safety benefit (fixed 2026-04-18).
        """
        # Populate all four pieces of state via one signal call.
        self.hmm.predict.return_value = make_regime("Bull", 0.75)
        self.lstm.predict.return_value = 0.02
        self.combiner.get_signal("XAUUSD", self.features)
        self.combiner._last_signal_bar["XAUUSD"] = "2026-04-12T08:00:00"

        # Sanity: everything populated.
        assert "XAUUSD" in self.combiner._recent_dirs
        assert len(self.combiner._recent_dirs["XAUUSD"]) > 0
        assert "XAUUSD" in self.combiner._last_signal_bar
        assert self.combiner.last_signal is not None
        assert "XAUUSD" in self.combiner.last_signal_by_symbol

        # Snapshot the display cache BEFORE reset so we can assert
        # identity afterward (not just non-None — the object itself
        # must be preserved so the dashboard keeps rendering it).
        pre_last = self.combiner.last_signal
        pre_by_symbol = dict(self.combiner.last_signal_by_symbol)

        self.combiner.reset_state()

        # Trading state cleared:
        assert self.combiner._recent_dirs == {}
        assert self.combiner._last_signal_bar == {}
        # Display cache preserved:
        assert self.combiner.last_signal is pre_last
        assert self.combiner.last_signal_by_symbol == pre_by_symbol
