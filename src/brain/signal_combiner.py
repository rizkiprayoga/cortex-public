"""
signal_combiner.py — HMM + LSTM Signal Fusion

Combines the market regime from the HMM classifier with the
directional price prediction from the LSTM to produce a single
actionable trade signal.

Fusion formula:
    combined_score = hmm_weight * regime_score + lstm_weight * lstm_score

    regime_score  = position_multiplier (0.0 – 1.0) × direction_sign
    lstm_score    = predicted_return_zscore clipped to [-1, 1]

A trade is triggered when |combined_score| >= signal_threshold (config).

Weights and threshold are set in config/model_config.yaml.

Prediction persistence
----------------------
On each call, ``get_signal_async()`` persists BOTH the HMM regime
prediction AND the LSTM price_return prediction via
``DataStore.save_prediction()``. It also fires
``FeedbackLoop.run_h4_cycle()`` for the previous bar so that as soon
as the current bar's close is known, the outcome label for the prior
bar's prediction can be computed and used in the feedback loop.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from src.brain.hmm_regime import HMMRegimeClassifier, RegimeResult
from src.brain.deep_learning.lstm_model import LSTMPricePredictor

if TYPE_CHECKING:
    from src.data_pipeline.data_store import DataStore
    from src.data_pipeline.feedback_loop import FeedbackLoop

logger = logging.getLogger(__name__)


# Directional orientation of each HMM regime label. Used together with
# ``position_multiplier`` to form a signed regime contribution to the combined
# score. Crash/Bear point short, Bull/Euphoria point long, Neutral has no
# directional opinion.
REGIME_DIRECTION_SIGN: dict[int, int] = {
    0: -1,  # Crash
    1: -1,  # Bear
    2:  0,  # Neutral
    3: +1,  # Bull
    4: +1,  # Euphoria
}

# Regimes that satisfy the confluence check for a "buy" direction.
BULLISH_REGIMES: frozenset[int] = frozenset({3, 4})
# Regimes that satisfy the confluence check for a "sell" direction.
BEARISH_REGIMES: frozenset[int] = frozenset({0, 1})


@dataclass
class SignalResult:
    symbol: str
    should_trade: bool
    direction: Optional[str]       # "buy", "sell", or None
    combined_score: float          # [-1.0, 1.0] — strength + direction
    regime: RegimeResult
    lstm_prediction: float         # Raw predicted return from LSTM
    confidence: float              # 0.0 – 1.0
    bar_timestamp: Optional[str] = None  # ISO 8601 of the bar this signal refers to
    # --- strategy-layer integration -------------------------------------------
    uncertainty_mode: bool = False
    #   Set True when regime state_probability < min_confidence. Does NOT block
    #   the trade by itself — the downstream PositionSizer halves the lot size
    #   when this flag is on.
    size_discount: float = 1.0
    #   Wave 6 fix #16: explicit float gate that the main loop feeds into the
    #   ``min()`` stacking with the CircuitBreaker multiplier. Currently set to
    #   0.5 iff uncertainty_mode is True. Keeps multiple size-reduction gates
    #   from compounding (uncertainty × circuit-breaker × alloc_pct collapsed
    #   to 0.24% effective risk in stress scenarios under the old product
    #   rule). We defer to the tightest active guard instead.
    reasoning: list[str] = field(default_factory=list)
    #   Append-only human-readable trace of every gate the signal passed or
    #   failed (fusion, threshold, confluence, flickering, uncertainty,
    #   long_only). Consumed by logs and the dashboard; never parsed.


class SignalCombiner:
    """
    Fuses HMM regime and LSTM price prediction into a trade signal.

    Usage:
        combiner = SignalCombiner(
            hmm_classifier, lstm_predictor,
            data_store=store, feedback_loop=feedback_loop,
        )
        signal = await combiner.get_signal_async(
            "XAUUSD", feature_matrix,
            feature_sequence=lstm_sequence,
            bar_timestamp=current_bar_ts,
        )
        if signal.should_trade:
            place_order(signal.direction, lot_size)
    """

    def __init__(
        self,
        hmm: HMMRegimeClassifier,
        lstm: LSTMPricePredictor,
        # NOTE: Tuned production values for the constants below are REDACTED
        # in this public template. Defaults are 0.0 placeholders so the bot
        # refuses to fire trades until the operator supplies real values via
        # config/model_config.yaml + config/settings.yaml. Tune via grid sweep
        # or Optuna against your own backtest harness.
        hmm_weight: float = 0.0,           # PLACEHOLDER
        lstm_weight: float = 0.0,          # PLACEHOLDER
        signal_threshold: float = 0.0,     # PLACEHOLDER
        long_only_mode: bool = True,
        long_only_symbols: Optional[set[str]] = None,
        min_confidence: float = 0.0,       # PLACEHOLDER
        flicker_bars_required: int = 1,    # PLACEHOLDER (typical range 2-4)
        data_store: Optional["DataStore"] = None,
        feedback_loop: Optional["FeedbackLoop"] = None,
        fundamentals_fetcher: Optional[Callable[[str, "datetime"], dict]] = None,
        exec_features_fetcher: Optional[Callable[[str, "datetime"], dict]] = None,
    ):
        self.hmm = hmm
        self.lstm = lstm
        self.hmm_weight = hmm_weight
        self.lstm_weight = lstm_weight
        self.signal_threshold = signal_threshold
        self.long_only_mode = long_only_mode
        # Per-symbol long-only override: if set, only these symbols are long-only
        # regardless of long_only_mode. Enables BTC shorts in Crash/Bear.
        self.long_only_symbols = long_only_symbols
        self.min_confidence = min_confidence
        self.flicker_bars_required = flicker_bars_required
        self.data_store = data_store
        self.feedback_loop = feedback_loop
        # Track the previous bar per symbol so we can compute its outcome
        # once the next bar has closed.
        self._last_signal_bar: dict[str, str] = {}
        # Most recent SignalResult emitted (across any symbol). RiskMonitor
        # reads this via attach_signal_ref() to snapshot regime + lstm context
        # into the circuit-breaker audit log at trip time. Written on every
        # _fuse_signals() call; never consumed inside this class.
        self.last_signal: Optional[SignalResult] = None
        # Per-symbol last signal — preserves the most recent SignalResult
        # for each traded symbol so the dashboard can show all 4 regime
        # cards instead of just whichever symbol the trading loop touched
        # most recently. Updated alongside ``last_signal`` in _fuse_signals.
        self.last_signal_by_symbol: dict[str, SignalResult] = {}
        # Flickering ring buffer: last N directions per symbol. Used by the
        # confluence/flicker check to require N consecutive identical
        # directions before a trade is allowed.
        self._recent_dirs: dict[str, deque[Optional[str]]] = {}
        # Rolling window of recent LSTM raw predictions per symbol, used to
        # z-score normalize the next prediction.
        self._lstm_history: dict[str, deque[float]] = {}
        # M-1 meta-labeler (LightGBM) — OFF by default. Two modes:
        # - CORTEX_META_LABELER=1       → active gate (blocks trades)
        # - CORTEX_META_LABELER_SHADOW=1 → shadow mode (compute + log, no block)
        # Gate wins if both set. Shadow mode produces "meta_labeler_shadow:
        # P(win)=... WOULD_{ALLOW,BLOCK}" reasoning lines that flow into
        # signal_audit.csv so ``scripts/analyze_meta_labeler_shadow.py`` can
        # later compute what the active gate would have saved/cost.
        import os as _os
        self._meta_labeler_enabled = (
            _os.environ.get("CORTEX_META_LABELER", "").strip()
            in ("1", "true", "True")
        )
        self._meta_labeler_shadow = (
            not self._meta_labeler_enabled
            and _os.environ.get("CORTEX_META_LABELER_SHADOW", "").strip()
            in ("1", "true", "True")
        )
        self._meta_labeler_cache: dict[str, Optional[dict]] = {}
        self._meta_labeler_warned: set[str] = set()
        # Optional callable: ``(symbol, bar_ts) → dict`` of fundamental
        # features. When provided, _meta_labeler_gate fetches fundamentals
        # at signal time and passes them to predict_proba — closes the
        # train/serve gap that previously NaN-filled all 17 fundamental
        # slots at inference. Set by backtest_full when the prefetcher
        # is available; live wiring deferred until DataStore is on the
        # signal-time hot path.
        self._fundamentals_fetcher = fundamentals_fetcher
        # Phase 2B Option 2 (2026-04-27): execution-conditional features
        # fetcher — supplies the 3 rolling-vol features (rv_short_20,
        # rv_long_60, rv_ratio) at signal time. The 4th exec feature
        # (score_avg_20) is computed internally from a per-symbol score
        # deque maintained below — feedback-loop-free since it tracks
        # primary's pre-gate emissions, not gated outcomes.
        self._exec_features_fetcher = exec_features_fetcher
        self._signal_score_history: dict[str, deque[float]] = {}

    # -------------------------------------------------------------------------
    # State reset (called by RiskMonitor after a circuit-breaker halt)
    # -------------------------------------------------------------------------

    def reset_state(self) -> None:
        """
        Flush the per-symbol flickering ring and the "previous bar" memo.

        Wave 6 fix #10: after a circuit-breaker halt forces
        ``EmergencyClose.close_all()``, the pre-halt direction in the
        flickering ring is stale — first signal after the resume should
        NOT inherit those bars, because during the halt the market may
        have regimed-shifted and the old direction can fire on the very
        first post-halt bar if the ring still holds 3 matching entries.
        Clearing the ring forces the combiner to wait
        ``flicker_bars_required`` fresh bars before re-approving any entry.

        Same reasoning applies to account switches — the new account
        might hold different positions, so inherited direction history
        could trigger a bad first-bar trade.

        Preserves ``last_signal`` and ``last_signal_by_symbol`` —
        those feed the dashboard's regime/signal cards and represent
        current MARKET state, not trading state. Regimes and signals
        are computed from OHLCV features, which don't depend on the
        account. Clearing them on switch blanks the dashboard until
        the next H4 tick generates fresh signals (potentially hours)
        with zero trading-safety benefit.

        No-op on the LSTM rolling history — those normalization stats
        remain calibrated across halts on purpose.
        """
        self._recent_dirs.clear()
        self._last_signal_bar.clear()
        logger.info(
            "SignalCombiner: trading state reset (flickering ring, "
            "last_signal_bar cleared; display cache preserved)"
        )

    # -------------------------------------------------------------------------
    # Sync signal generation (used by backtests and sync tests)
    # -------------------------------------------------------------------------

    def get_signal(
        self,
        symbol: str,
        feature_matrix: np.ndarray,
        lstm_sequence: Optional[np.ndarray] = None,
        bar_timestamp: Optional[str] = None,
    ) -> SignalResult:
        """
        Compute the combined trade signal for a symbol (synchronous).

        Steps:
            1. Get regime from HMM (direction + position multiplier)
            2. Get predicted return from LSTM
            3. Delegate to ``_fuse_signals()`` which applies fusion + all gates
               (threshold, confluence, flickering, uncertainty, long-only).

        Args:
            symbol:         Trading symbol (e.g. "XAUUSD")
            feature_matrix: Feature window for HMM (n_bars × n_hmm_features)
            lstm_sequence:  Feature window for LSTM (n_bars × n_lstm_features).
                            If None, uses ``feature_matrix`` (backward compat).
            bar_timestamp:  ISO 8601 of the bar this signal refers to. Required
                            by the meta-labeler fundamentals_fetcher (Option B
                            inference path) to look up release-lag-safe
                            fundamentals at signal time. None is accepted for
                            back-compat with callers that don't yet pass it
                            (the meta-labeler then falls back to fundamentals=None).

        Returns:
            SignalResult with direction, combined_score, uncertainty_mode,
            and a full reasoning trace.

        Note:
            This method does NOT log predictions to the DB. Use
            ``get_signal_async()`` from the live trading loop to
            enable feedback-loop integration.
        """
        regime = self.hmm.predict(symbol, feature_matrix)
        lstm_input = lstm_sequence if lstm_sequence is not None else feature_matrix
        lstm_prediction = self.lstm.predict(symbol, lstm_input)
        return self._fuse_signals(
            symbol=symbol,
            regime=regime,
            lstm_prediction=float(lstm_prediction),
            bar_timestamp=bar_timestamp,
        )

    # -------------------------------------------------------------------------
    # Async signal generation (used by main trading loop)
    # -------------------------------------------------------------------------

    async def get_signal_async(
        self,
        symbol: str,
        feature_matrix: np.ndarray,
        feature_sequence: Optional[np.ndarray] = None,
        bar_timestamp: Optional[str] = None,
    ) -> SignalResult:
        """
        Compute the combined trade signal AND persist both HMM and LSTM
        predictions to ``model_predictions``. Also triggers the feedback
        loop for the previous bar (whose outcome is now known).

        Args:
            symbol:           Trading symbol
            feature_matrix:   D1 feature window for HMM (n_bars × n_features)
            feature_sequence: H4 feature sequence for LSTM (seq_len × n_features).
                              Defaults to ``feature_matrix`` if not provided.
            bar_timestamp:    ISO 8601 timestamp of the current bar.
        """
        # Step 1: fire feedback loop for the PREVIOUS bar (now that this bar
        # has closed, its outcome is known). Non-fatal if it fails.
        if self.feedback_loop is not None:
            prev_ts = self._last_signal_bar.get(symbol)
            if prev_ts:
                try:
                    await self.feedback_loop.run_h4_cycle(symbol, prev_ts)
                except Exception as e:
                    logger.warning(
                        f"[{symbol}] Feedback loop tick failed for {prev_ts}: {e}"
                    )

        # Step 2: HMM regime (predict + log)
        if bar_timestamp and self.data_store is not None:
            regime_result = await self.hmm.predict_and_log(
                symbol, feature_matrix, bar_timestamp
            )
        else:
            regime_result = self.hmm.predict(symbol, feature_matrix)

        # Step 3: LSTM price return (predict + log)
        lstm_input = feature_sequence if feature_sequence is not None else feature_matrix
        if bar_timestamp and self.data_store is not None:
            lstm_prediction = await self.lstm.predict_and_log(
                symbol, lstm_input, bar_timestamp,
                confidence=float(regime_result.state_probability),
            )
        else:
            lstm_prediction = self.lstm.predict(symbol, lstm_input)

        # Step 4: fuse the two signals into a single trade decision
        signal = self._fuse_signals(
            symbol=symbol,
            regime=regime_result,
            lstm_prediction=lstm_prediction,
            bar_timestamp=bar_timestamp,
        )

        # Step 5: remember this bar for the next feedback-loop tick
        if bar_timestamp:
            self._last_signal_bar[symbol] = bar_timestamp

        return signal

    # -------------------------------------------------------------------------
    # Fusion logic
    # -------------------------------------------------------------------------

    def _fuse_signals(
        self,
        symbol: str,
        regime: RegimeResult,
        lstm_prediction: float,
        bar_timestamp: Optional[str] = None,
    ) -> SignalResult:
        """
        Combine regime + LSTM predictions into a single SignalResult.

        Applies gates in this order (each gate appends a line to ``reasoning``):

            A. Raw fusion         — combined_score = w_h·regime + w_l·lstm
            B. Direction          — sign of combined_score
            C. Magnitude          — |combined_score| >= signal_threshold
            D. Flicker ring       — always updated before the flicker gate
            E. Confluence         — regime direction must agree with trade dir
            F. Flickering         — last N bar directions must all match
            G. Uncertainty flag   — sets uncertainty_mode (non-blocking)
            H. Long-only gate     — blocks sells when long_only_mode is on

        The LSTM contribution is NOT clipped to the buy side — sell
        predictions reach the combined score with their full signed value so
        the feedback loop sees calibrated predictions. The long-only
        enforcement happens at gate H, not in fusion.
        """
        reasoning: list[str] = []

        # ---- A. Raw fusion --------------------------------------------------
        regime_score = self._regime_score(regime)
        lstm_score = self._normalize_lstm_prediction(symbol, lstm_prediction)
        combined_score = self.hmm_weight * regime_score + self.lstm_weight * lstm_score
        combined_score = float(np.clip(combined_score, -1.0, 1.0))

        # ---- B. Direction from sign -----------------------------------------
        if combined_score > 0:
            direction: Optional[str] = "buy"
        elif combined_score < 0:
            direction = "sell"
        else:
            direction = None

        reasoning.append(
            f"fusion: regime_score={regime_score:+.3f} "
            f"lstm_score={lstm_score:+.3f} "
            f"combined={combined_score:+.3f} "
            f"direction={direction}"
        )

        # ---- C. Magnitude gate ----------------------------------------------
        # Per-symbol threshold overrides the default when set via
        # ``combiner.per_symbol_threshold = {"USDJPY": 0.55}``. Used by the
        # USDJPY grid-test and available at runtime if settings.yaml wires it.
        per_sym = getattr(self, "per_symbol_threshold", None) or {}
        threshold = per_sym.get(symbol.upper(), self.signal_threshold)
        should_trade = (
            direction is not None and abs(combined_score) >= threshold
        )
        if direction is not None and not should_trade:
            reasoning.append(
                f"below_threshold: |{combined_score:+.3f}| < {threshold}"
            )

        # ---- D. Flicker ring (always updated) -------------------------------
        dirs = self._recent_dirs.setdefault(
            symbol, deque(maxlen=self.flicker_bars_required)
        )
        dirs.append(direction)

        # ---- E. Confluence --------------------------------------------------
        # "Must not disagree" — Neutral is permissive (LSTM drives).
        # Only block when regime actively contradicts the direction.
        if should_trade:
            if direction == "buy" and regime.regime_index in BEARISH_REGIMES:
                should_trade = False
                reasoning.append(
                    f"confluence_fail: buy contradicts Crash/Bear, "
                    f"HMM={regime.regime_label}"
                )
            elif direction == "sell" and regime.regime_index in BULLISH_REGIMES:
                should_trade = False
                reasoning.append(
                    f"confluence_fail: sell contradicts Bull/Euphoria, "
                    f"HMM={regime.regime_label}"
                )
            # USD/JPY Euphoria intervention guard — BOJ historically
            # intervenes in the 155-160 zone. Default: block buys in Euphoria
            # for JPY pairs. Softened option (env CORTEX_USDJPY_EUPHORIA_MULT
            # or ``combiner.euphoria_size_mult``): allow the buy at reduced
            # size — captures carry-trade alpha at the cost of some
            # intervention risk. Set to 0.0 for full block, 1.0 for no guard,
            # 0.5 for half-size.
            elif (direction == "buy"
                  and regime.regime_index == 4  # Euphoria
                  and "JPY" in symbol.upper()):
                import os as _os
                mult = float(
                    getattr(self, "euphoria_size_mult",
                            _os.environ.get("CORTEX_USDJPY_EUPHORIA_MULT", "0.0"))
                )
                if mult <= 0.0:
                    should_trade = False
                    reasoning.append(
                        f"usdjpy_intervention_guard: buy in Euphoria regime "
                        f"blocked — BOJ intervention risk"
                    )
                else:
                    # Keep should_trade True, but attach a size multiplier
                    # that downstream sizing code reads via ``signal.size_mult``.
                    reasoning.append(
                        f"usdjpy_intervention_guard_soft: buy in Euphoria "
                        f"allowed at {mult:.0%} size (BoJ intervention risk)"
                    )
            # EURUSD Euphoria mean-reversion guard — Phase A.3/A.4 fix
            # Backtest data shows EURUSD Euphoria has WR=26-27% (vs 45-64%
            # in Bull) — model is buying tops. Portfolio A/B (2026-04-27)
            # confirmed removing this guard regresses EUR contribution from
            # $12,787 → $7,689 in the 10-pair sim. Guard stays on for EUR.
            #
            # XAUUSD was removed from this guard 2026-04-27 — Cell C A/B
            # showed XAU is bidirectional-trend-friendly post-2020 and the
            # euphoria block was leaving +$3,875 on the table at 5yr scope.
            #
            # Bypass for A/B experimentation: set
            #   CORTEX_DISABLE_EUPHORIA_GUARD=EURUSD
            # Comma-separated, case-insensitive. Mirrors the USDJPY
            # euphoria-guard bypass pattern.
            elif (direction == "buy"
                  and regime.regime_index == 4  # Euphoria
                  and symbol.upper() in ("EURUSD", "EUR/USD")):
                import os as _os
                _disabled = {
                    s.strip().upper()
                    for s in _os.environ.get(
                        "CORTEX_DISABLE_EUPHORIA_GUARD", "",
                    ).split(",")
                    if s.strip()
                }
                if symbol.upper() in _disabled:
                    reasoning.append(
                        f"euphoria_guard: bypassed via "
                        f"CORTEX_DISABLE_EUPHORIA_GUARD env (A/B mode)"
                    )
                else:
                    should_trade = False
                    reasoning.append(
                        f"euphoria_guard: buy in Euphoria blocked — "
                        f"{symbol} mean-reverts in extreme overbought conditions "
                        f"(historical WR 26-27% in this regime)"
                    )
            # USDCAD Neutral chop guard — Phase A.4 fix
            # USDCAD is oil-driven; Neutral regime = range-bound oil = chop.
            # Backtest data: Neutral WR=23.8% (-$1,258) vs Bull 50.9% (+$1,659).
            # Only trade USDCAD in clear directional regimes (Bull/Euphoria).
            elif (regime.regime_index == 2  # Neutral
                  and symbol.upper() in ("USDCAD", "USD/CAD")):
                should_trade = False
                reasoning.append(
                    f"usdcad_neutral_guard: USDCAD doesn't trade in Neutral — "
                    f"oil-driven pair needs clear directional regime "
                    f"(historical WR 23.8% in Neutral)"
                )

        # ---- F. Flickering --------------------------------------------------
        if should_trade:
            if len(dirs) < self.flicker_bars_required:
                should_trade = False
                reasoning.append(
                    f"flickering: only {len(dirs)}/{self.flicker_bars_required} "
                    "bars in history, waiting for stability"
                )
            elif not all(d == direction for d in dirs):
                should_trade = False
                reasoning.append(
                    f"flickering: direction unstable across last "
                    f"{self.flicker_bars_required} bars: {list(dirs)}"
                )

        # ---- G. Uncertainty flag (non-blocking) -----------------------------
        uncertainty_mode = float(regime.state_probability) < self.min_confidence
        if uncertainty_mode:
            reasoning.append(
                f"uncertainty_mode: regime_prob={regime.state_probability:.2f} "
                f"< {self.min_confidence}, sizing will be halved"
            )

        # ---- H. Long-only gate ----------------------------------------------
        # Per-symbol long-only gate: if long_only_symbols is set, only
        # those symbols block sells. Otherwise fall back to global flag.
        is_long_only = (
            (self.long_only_symbols is not None and symbol in self.long_only_symbols)
            or (self.long_only_symbols is None and self.long_only_mode)
        )
        if should_trade and is_long_only and direction == "sell":
            should_trade = False
            reasoning.append("long_only_mode: LSTM bearish, sitting out")

        # ---- I. Meta-labeler gate (M-1, opt-in) -----------------------------
        # Secondary LightGBM classifier filters out signals whose features
        # match historical *losers*. Active gate blocks; shadow mode only
        # logs what the gate WOULD have done.
        # Capture pre-meta state to drive the score-deque update below
        # (Phase 2B Opt 2): the deque should reflect "signals that passed all
        # UPSTREAM gates", which matches the training pool (backtest_trades
        # was generated without a meta-labeler).
        should_trade_pre_meta = should_trade
        if should_trade and (self._meta_labeler_enabled or self._meta_labeler_shadow):
            meta_decision = self._meta_labeler_gate(
                symbol=symbol,
                combined_score=combined_score,
                regime_label=regime.regime_label,
                direction=direction,
                bar_timestamp=bar_timestamp,
            )
            if meta_decision is not None:
                passed, proba, reason = meta_decision
                if self._meta_labeler_enabled:
                    if not passed:
                        should_trade = False
                    reasoning.append(reason)
                else:
                    # Shadow mode — rewrite reason as WOULD_{ALLOW,BLOCK}
                    reasoning.append(
                        f"meta_labeler_shadow: P(win)={proba:.3f} "
                        f"{'>=' if passed else '<'} threshold "
                        f"→ WOULD_{'ALLOW' if passed else 'BLOCK'}"
                    )

        # Phase 2B Option 2 (2026-04-27): update per-symbol score deque AFTER
        # _meta_labeler_gate has read it (so the current bar's score is not
        # part of its own exec__score_avg_20 — preserves shift(1) semantics
        # used at training). Track only signals that would have executed
        # under the no-meta-labeler counterfactual (= training distribution).
        if should_trade_pre_meta:
            history = self._signal_score_history.setdefault(
                symbol, deque(maxlen=20),
            )
            history.append(float(combined_score))

        if should_trade:
            reasoning.append(
                f"APPROVED: {direction} at score={combined_score:+.3f}"
            )

        signal = SignalResult(
            symbol=symbol,
            should_trade=should_trade,
            direction=direction,
            combined_score=combined_score,
            regime=regime,
            lstm_prediction=float(lstm_prediction),
            confidence=float(regime.state_probability),
            bar_timestamp=bar_timestamp,
            uncertainty_mode=uncertainty_mode,
            size_discount=0.5 if uncertainty_mode else 1.0,
            reasoning=reasoning,
        )
        self.last_signal = signal
        self.last_signal_by_symbol[signal.symbol] = signal
        return signal

    def _meta_labeler_gate(
        self,
        symbol: str,
        combined_score: float,
        regime_label: str,
        direction: str,
        bar_timestamp: str,
    ) -> Optional[tuple[bool, float, str]]:
        """Consult the per-symbol meta-labeler (M-1).

        Returns ``(passed, proba, reasoning)`` or None if the bundle is
        missing / inference failed — caller should treat None as "gate did
        not participate" (skip, don't block).
        """
        # Lazy-load, cache, warn once on miss. Use ``symbol not in cache``
        # as the "not yet resolved" signal instead of a string sentinel
        # (would violate the dict[str, Optional[dict]] type contract).
        # Spec §3 invariant #12: refuse to load a bundle whose
        # feature_schema_hash doesn't match the runtime contract — stops
        # silent train/serve drift if the meta-labeler was trained
        # against a different feature schema.
        if symbol not in self._meta_labeler_cache:
            try:
                from src.brain.meta_labeler_features import (
                    ACCEPTED_SCHEMA_HASHES,
                )
                from src.ml.meta_labeler import load_meta_labeler
                # Resolve which primary's meta-labeler to load. Symmetric to
                # the model_kind resolution used for the primary artifact.
                from src.utils.model_head import resolve_model_kind_for_symbol
                primary = resolve_model_kind_for_symbol(symbol)
                loaded = load_meta_labeler(symbol, primary=primary)
                if loaded is not None:
                    bundle_hash = loaded.get("feature_schema_hash")
                    if bundle_hash not in ACCEPTED_SCHEMA_HASHES:
                        logger.warning(
                            "[%s] meta-labeler refuses to load: schema_hash "
                            "mismatch (bundle=%r, accepted=%r). Was the "
                            "labeler trained against a different feature "
                            "contract? Re-run scripts/train_meta_labeler.py. "
                            "Gate disabled for this symbol (spec §3 #12).",
                            symbol, bundle_hash, sorted(ACCEPTED_SCHEMA_HASHES),
                        )
                        loaded = None
            except Exception as exc:
                logger.warning("[%s] meta-labeler load crashed: %s", symbol, exc)
                loaded = None
            self._meta_labeler_cache[symbol] = loaded
            if loaded is None and symbol not in self._meta_labeler_warned:
                logger.info(
                    "[%s] meta-labeler bundle missing; gate disabled for this symbol",
                    symbol,
                )
                self._meta_labeler_warned.add(symbol)
        bundle = self._meta_labeler_cache[symbol]

        if bundle is None:
            return None

        # Derive hour-of-day / day-of-week from bar_timestamp
        try:
            import pandas as _pd
            ts = _pd.to_datetime(bar_timestamp, utc=True)
            hour = int(ts.hour)
            dow = int(ts.dayofweek)
        except Exception:
            hour = -1
            dow = -1

        # Option B (2026-04-27): if a fundamentals fetcher was injected
        # (backtest path prefetches feature_store at run start and passes
        # a closure), resolve fundamentals for this bar and pass them to
        # predict_proba so the model gets the full 22-feature input.
        # Falls back to fundamentals=None when no fetcher is wired (live
        # hot path, smoke tests) — degraded but functional.
        fundamentals = None
        if self._fundamentals_fetcher is not None:
            try:
                fundamentals = self._fundamentals_fetcher(symbol, bar_timestamp)
            except Exception as exc:
                logger.debug(
                    "[%s] fundamentals fetcher failed at %s: %s — "
                    "falling back to NaN-filled inference",
                    symbol, bar_timestamp, exc,
                )
                fundamentals = None

        # Phase 2B Option 2 (2026-04-27): assemble execution-conditional
        # features. RV trio comes from the injected fetcher (backtest
        # precomputes from H4 OHLCV); score_avg_20 comes from the per-
        # symbol score deque updated AFTER this gate by _fuse_signals.
        # Both NaN-fall-through silently if not available — LightGBM
        # handles NaN, predictions degrade gracefully.
        exec_features: dict = {}
        if self._exec_features_fetcher is not None:
            try:
                rv_dict = self._exec_features_fetcher(symbol, bar_timestamp)
                if rv_dict:
                    exec_features.update(rv_dict)
            except Exception as exc:
                logger.debug(
                    "[%s] exec features fetcher failed at %s: %s",
                    symbol, bar_timestamp, exc,
                )
        score_history = self._signal_score_history.get(symbol)
        if score_history is not None and len(score_history) >= 5:
            arr = np.fromiter(score_history, dtype=np.float64)
            exec_features["exec__score_avg_20"] = float(arr.mean())

        try:
            from src.ml.meta_labeler import predict_proba
            proba = predict_proba(
                bundle,
                combined_score=combined_score,
                regime_label=regime_label,
                direction=direction,
                hour_of_day=hour,
                day_of_week=dow,
                fundamentals=fundamentals,
                exec_features=exec_features,
            )
        except Exception as exc:
            logger.warning("[%s] meta-labeler predict failed: %s", symbol, exc)
            return None

        threshold = float(bundle.get("threshold", 0.5))
        passed = proba >= threshold
        reasoning = (
            f"meta_labeler: P(win)={proba:.3f} "
            f"{'>=' if passed else '<'} {threshold:.2f} "
            f"→ {'ALLOW' if passed else 'BLOCK'}"
        )
        return passed, proba, reasoning

    def _regime_score(self, regime: RegimeResult) -> float:
        """
        Signed regime contribution to the combined score.

        regime_score = position_multiplier × direction_sign

        where direction_sign is -1 for Crash/Bear, 0 for Neutral, +1 for
        Bull/Euphoria. With symmetric MULTIPLIERS, Crash contributes -1.0
        (full-conviction short bias) and Euphoria contributes +1.0 — the
        bot trades both directions on all live pairs.
        """
        sign = REGIME_DIRECTION_SIGN.get(regime.regime_index, 0)
        return float(regime.position_multiplier) * sign

    def _normalize_lstm_prediction(
        self, symbol: str, raw_prediction: float
    ) -> float:
        """
        Normalize a raw LSTM return prediction to [-1, 1] using rolling
        statistics over the last 100 predictions for this symbol.

        Cold start (< 20 samples): fall back to ``tanh(raw / 0.005)`` which
        maps a ~0.5% return to ±0.76 and saturates smoothly for larger moves.

        Warm (≥ 20 samples): z-score vs. the rolling mean/std, then clip to
        [-1, 1]. This keeps the LSTM contribution calibrated even if model
        outputs drift over retrainings.
        """
        history = self._lstm_history.setdefault(symbol, deque(maxlen=100))
        history.append(float(raw_prediction))

        if len(history) < 20:
            return float(
                np.clip(np.tanh(float(raw_prediction) / 0.005), -1.0, 1.0)
            )

        arr = np.fromiter(history, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-12:
            return 0.0
        z = (float(raw_prediction) - mean) / std
        return float(np.clip(z, -1.0, 1.0))
