"""
portfolio_manager.py — Multi-Asset Portfolio Allocation + Pyramiding BE-Gate

Stands between SignalCombiner and PositionSizer. Enforces portfolio-wide
caps and the pyramiding "breakeven gate" rule before any new order
leaves the process.

Responsibilities
----------------
    1. Delegate lot calculation to ``PositionSizer``
    2. Enforce margin caps (per-symbol, total, free-margin reserve)
    3. Enforce concurrency caps
            max_concurrent_per_symbol   (default 3)
            max_concurrent_total        (default 6)
    4. Enforce the **pyramiding BE-gate**: a new entry on a symbol that
       already has open positions is allowed *only* if every prior
       position on that symbol is risk-free — i.e. its current stop is
       at-or-above entry for longs, or at-or-below entry for shorts.

This makes ``calculate_lot_size()`` the single authoritative place that
decides whether an entry is allowed and at what size. OrderManager
still runs its own ``sl_price is None`` rejection as a last-line
defense, but the *business rules* all live here.

Position queries are abstracted through an ``OpenPositionView``
protocol so the module can be unit-tested without MT5 at all.
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np

from src.allocation.position_sizer import PositionSizer, SizingResult, SymbolSpec
from src.broker.account_monitor import AccountSnapshot
from src.strategy.base import StrategyDecision

logger = logging.getLogger(__name__)

# Wave 6 fix #17: cross-asset correlation cap. XAUUSD and BTCUSD are
# the only instruments this bot trades today, and in the empirically
# observed macro regime they lock correlation during equity-risk-off
# moves (Fed tightening, liquidity crunches). Loading both up at the
# same time under the per-symbol gate gives us *2 × 1% = 2% aggregate
# risk to a single macro shock, on top of whatever the pyramiding gate
# is already allowing. When rolling Pearson ρ over the last 20+ daily
# closes exceeds ``CORRELATION_THRESHOLD``, the two symbols are merged
# into a single risk bucket capped at ``CORRELATED_BUCKET_MAX`` open
# legs. When correlation drops back below the threshold, the cap relaxes
# automatically — no persistent state to manage.
CORRELATION_SYMBOLS: tuple[str, ...] = ("XAUUSD", "BTCUSD")
CORRELATION_THRESHOLD: float = 0.6
CORRELATED_BUCKET_MAX: int = 3
CORRELATION_MIN_SAMPLES: int = 20
CORRELATION_HISTORY_LEN: int = 30


@dataclass
class OpenPositionView:
    """
    Minimal projection of an open MT5 position the portfolio manager
    needs for its gate checks.

    The caller populates this from ``mt5.positions_get()``; the sizer
    never touches MT5 directly so tests can inject fake histories.

    Wave 6 fix #18: the pyramiding gate now reads ``tier_1_done`` — a
    flag that can only be set by the exit ladder after it has actually
    fired tier 1 (partial close + stop-to-BE). Previously it inferred
    risk-free state from ``current_stop >= entry_price``, which could
    be spoofed by a manual SL adjustment or by a reconciled position
    whose broker-side stop happens to sit above entry (the Wave 4
    reconcile helper sets ``tier_1_done=False`` on every reconstructed
    leg, which is the safe default).
    """

    symbol: str
    direction: str          # "buy" or "sell"
    entry_price: float
    current_stop: float     # sl field — may equal entry once BE-move fired
    tier_1_done: bool = False   # True only after ExitManager fired tier 1


class PortfolioManager:
    """
    Gate + sizing coordinator for all new orders.

    Usage
    -----
        pm = PortfolioManager(
            sizer=PositionSizer(),
            positions_provider=lambda: account_monitor.get_open_positions(),
            symbol_spec_provider=lambda sym: broker_specs[sym],
            max_concurrent_per_symbol=3,
            max_concurrent_total=6,
            max_used_margin_pct_per_position=5.0,
            max_used_margin_pct_total=15.0,
            free_margin_reserve_pct=20.0,
        )
        result = pm.calculate_lot_size(
            "XAUUSD", signal, decision, account_snapshot,
        )
        if result.lot_size > 0:
            order_manager.place_order(..., lot_size=result.lot_size)
    """

    def __init__(
        self,
        sizer: Optional[PositionSizer] = None,
        positions_provider: Optional[Callable[[], list[OpenPositionView]]] = None,
        symbol_spec_provider: Optional[Callable[[str], SymbolSpec]] = None,
        max_concurrent_per_symbol: int = 3,
        max_concurrent_total: int = 6,
        max_used_margin_pct_per_position: float = 5.0,
        max_used_margin_pct_total: float = 15.0,
        free_margin_reserve_pct: float = 20.0,
        require_prior_at_breakeven_to_pyramid: bool = True,
        max_daily_trades: int = 12,
    ):
        self.sizer = sizer or PositionSizer()
        self._positions_provider = positions_provider or (lambda: [])
        self._symbol_spec_provider = symbol_spec_provider
        self.max_concurrent_per_symbol = max_concurrent_per_symbol
        self.max_concurrent_total = max_concurrent_total
        self.max_used_margin_pct_per_position = max_used_margin_pct_per_position
        self.max_used_margin_pct_total = max_used_margin_pct_total
        self.free_margin_reserve_pct = free_margin_reserve_pct
        self.require_prior_at_breakeven_to_pyramid = (
            require_prior_at_breakeven_to_pyramid
        )
        # Wave 6 fix #3: rolling 24h trade counter for the max_daily_trades
        # soft cap. Previously declared in settings.yaml and documented in
        # risk_management.md but never enforced. Now: on every successful
        # sizing we append the current UTC timestamp; at the top of every
        # calculate_lot_size() call we drop stamps older than 24h and
        # reject once the deque has reached max_daily_trades.
        self.max_daily_trades = max_daily_trades
        self._recent_trade_ts: deque[datetime] = deque(maxlen=max(100, max_daily_trades * 4))

        # Wave 6 fix #17: cross-asset correlation cap. Rolling closes are
        # fed in by the main loop via ``update_daily_close()`` once per
        # D1 bar close. The gate activates when we have enough samples on
        # both correlation symbols AND the pairwise |ρ| exceeds the
        # threshold. Prior to that, each symbol uses its own per-symbol
        # concurrency cap without any bucketed treatment.
        self._price_history: dict[str, deque[float]] = {
            sym: deque(maxlen=CORRELATION_HISTORY_LEN)
            for sym in CORRELATION_SYMBOLS
        }

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_lot_size(
        self,
        symbol: str,
        signal,                    # SignalResult
        decision: StrategyDecision,
        account_info: AccountSnapshot,
        size_multiplier: float = 1.0,
    ) -> SizingResult:
        """
        Decide the final lot size for the signal, enforcing all caps.

        Returns a SizingResult. A ``lot_size == 0.0`` result with a
        descriptive ``reason`` string means the order is rejected; the
        caller MUST NOT submit it.

        Args:
            symbol:            Trading symbol (for logging)
            signal:            SignalResult from SignalCombiner
            decision:          StrategyDecision from the active strategy
            account_info:      Current AccountSnapshot from AccountMonitor
            size_multiplier:   Optional external multiplier (e.g. from
                               CircuitBreaker.current_size_multiplier())
        """
        # Quick rejections on signal hygiene — SignalCombiner normally
        # catches these, but we double-check at the allocation boundary.
        if not getattr(signal, "should_trade", False):
            return self._reject_zero(
                symbol, decision.initial_stop_price,
                "signal.should_trade=False",
            )
        if decision.direction not in ("buy", "sell"):
            return self._reject_zero(
                symbol, decision.initial_stop_price,
                f"invalid decision.direction={decision.direction!r}",
            )

        # --- Daily trades soft cap (Wave 6 fix #3) -------------------
        # Drop stamps older than 24h, then reject if we've hit the cap.
        # The cap was declared in settings.yaml and documented in
        # risk_management.md before Wave 6 but was never actually
        # enforced anywhere in the codebase.
        now_utc = datetime.now(tz=timezone.utc)
        cutoff = now_utc - timedelta(hours=24)
        while self._recent_trade_ts and self._recent_trade_ts[0] < cutoff:
            self._recent_trade_ts.popleft()
        if len(self._recent_trade_ts) >= self.max_daily_trades:
            return self._reject_zero(
                symbol, decision.initial_stop_price,
                f"max_daily_trades_reached: "
                f"{len(self._recent_trade_ts)}/{self.max_daily_trades} in last 24h",
            )

        # --- Concurrency gates ----------------------------------------
        positions = self._positions_provider()
        total_open = len(positions)
        symbol_positions = [p for p in positions if p.symbol == symbol]
        symbol_open = len(symbol_positions)

        if total_open >= self.max_concurrent_total:
            return self._reject_zero(
                symbol, decision.initial_stop_price,
                f"max_concurrent_total={self.max_concurrent_total} reached",
            )
        if symbol_open >= self.max_concurrent_per_symbol:
            return self._reject_zero(
                symbol, decision.initial_stop_price,
                f"max_concurrent_per_symbol={self.max_concurrent_per_symbol} "
                f"reached",
            )

        # --- Cross-asset correlation cap (Wave 6 fix #17) -------------
        # XAUUSD and BTCUSD both respond to the same macro risk-off
        # shocks (Fed tightening, liquidity crunches). When rolling ρ
        # over the last 20+ daily closes exceeds 0.6 we treat them as a
        # single 3-slot risk bucket, so a single macro shock can't pull
        # 2% aggregate risk from us in one move.
        bucket_block = self._correlated_bucket_blocked(symbol, positions)
        if bucket_block is not None:
            return self._reject_zero(
                symbol, decision.initial_stop_price, bucket_block,
            )

        # --- Direction-conflict gate (bidirectional safety) ------------
        # Prevent holding buy AND sell on the same symbol simultaneously.
        # The bot must wait for reversal_hard_exit to close existing
        # positions before opening in the opposite direction.
        if symbol_open >= 1:
            conflicting = [
                p for p in symbol_positions
                if p.direction != decision.direction
            ]
            if conflicting:
                return self._reject_zero(
                    symbol, decision.initial_stop_price,
                    f"direction_conflict: {len(conflicting)} "
                    f"{conflicting[0].direction} position(s) still open — "
                    f"wait for reversal_hard_exit before opening "
                    f"{decision.direction}",
                )

        # --- Pyramiding BE-gate ---------------------------------------
        # Wave 6 fix #18: read tier_1_done on prior legs, not their stop
        # level. This prevents a manual SL adjustment or a reconciled
        # position from unlocking a new entry before the exit ladder has
        # actually fired tier 1 on the prior leg.
        if (
            symbol_open >= 1
            and self.require_prior_at_breakeven_to_pyramid
            and not self._all_prior_tier_1_done(symbol_positions)
        ):
            return self._reject_zero(
                symbol, decision.initial_stop_price,
                "pyramiding_blocked_prior_tier_1_not_done",
            )

        # --- Free-margin reserve check --------------------------------
        if not self._has_free_margin_reserve(account_info):
            return self._reject_zero(
                symbol, decision.initial_stop_price,
                f"free_margin_reserve<{self.free_margin_reserve_pct}%",
            )

        # --- Sizing ---------------------------------------------------
        symbol_spec = self._resolve_symbol_spec(symbol)
        if symbol_spec is None:
            return self._reject_zero(
                symbol, decision.initial_stop_price,
                f"no symbol_spec for {symbol}",
            )

        # Wave 6 fix #16: gates stack via min(), not product. The caller
        # already passes ``size_multiplier`` from the circuit breaker, and
        # the signal may carry its own ``size_discount`` (0.5 when regime
        # probability < min_confidence). Defer to the tightest gate — the
        # old behavior compounded all three (uncertainty × cb × alloc) and
        # could drive effective risk to 0.24% in stress, exactly when a
        # recovery-sized trade is what we want.
        signal_discount = float(getattr(signal, "size_discount", 1.0))
        effective_size_multiplier = min(signal_discount, float(size_multiplier))

        # Entry price approximation: the strategy's stop + the decision
        # direction together bracket the entry; the signal's regime or
        # the caller can override via signal.entry_price. Most common
        # case: the caller passes signal with an ``entry_price`` set.
        entry_price = (
            getattr(signal, "entry_price", None)
            or getattr(signal, "reference_price", None)
        )
        if entry_price is None:
            return self._reject_zero(
                symbol, decision.initial_stop_price,
                "signal lacks entry_price/reference_price",
            )

        # Fractional pyramid sizing: 100% → 50% → 25% per tier
        # Reduces risk on later pyramid legs for smoother equity curve.
        # symbol_positions is already filtered to this symbol above.
        pyramid_count = len(symbol_positions)
        pyramid_mult = {0: 1.0, 1: 0.50}.get(pyramid_count, 0.25)
        effective_allocation = decision.allocation_pct * pyramid_mult

        sizing = self.sizer.calculate(
            symbol_spec=symbol_spec,
            entry_price=entry_price,
            stop_price=decision.initial_stop_price,
            equity=account_info.equity,
            allocation_pct=effective_allocation,
            # Wave 6 fix #16: uncertainty is baked into size_discount
            # upstream; we pass uncertainty_mode=False to avoid the
            # legacy compounding path inside PositionSizer.
            uncertainty_mode=False,
            size_multiplier=effective_size_multiplier,
        )

        if sizing.lot_size <= 0.0:
            return sizing

        # --- Per-position margin cap check ----------------------------
        projected_margin_pct = self._projected_margin_pct(
            sizing, entry_price, account_info
        )
        if projected_margin_pct > self.max_used_margin_pct_per_position:
            logger.info(
                "%s: downsizing lot from %.4f to respect per-position "
                "margin cap %.1f%% (projected %.1f%%)",
                symbol,
                sizing.lot_size,
                self.max_used_margin_pct_per_position,
                projected_margin_pct,
            )
            scale = self.max_used_margin_pct_per_position / projected_margin_pct
            sizing = self.sizer.calculate(
                symbol_spec=symbol_spec,
                entry_price=entry_price,
                stop_price=decision.initial_stop_price,
                equity=account_info.equity,
                allocation_pct=decision.allocation_pct * scale,
                # Wave 6 fix #16: stay consistent with the first call —
                # uncertainty is already folded into effective_size_multiplier
                # via min(), so we pass uncertainty_mode=False here too.
                # Using the legacy compounded path on the rescale branch
                # would silently undo the min-stacking for any order that
                # happens to trip the per-position margin cap.
                uncertainty_mode=False,
                size_multiplier=effective_size_multiplier,
            )
            # Recompute projected margin after the rescale so the total
            # cap check below sees a current estimate.
            projected_margin_pct = self._projected_margin_pct(
                sizing, entry_price, account_info
            )

        if sizing.lot_size <= 0.0:
            return sizing

        # --- Total used-margin cap (Wave 6 fix #2) -------------------
        # The old code stored `max_used_margin_pct_total` as an
        # attribute but never read it — risk_management.md promised a
        # 15% portfolio-wide margin cap that wasn't actually enforced.
        # We now check: current used margin + projected margin of this
        # new order must not exceed the cap.
        current_total_margin_pct = (
            (account_info.margin / max(account_info.equity, 1e-9)) * 100.0
        )
        projected_total = current_total_margin_pct + projected_margin_pct
        if projected_total > self.max_used_margin_pct_total:
            return self._reject_zero(
                symbol, decision.initial_stop_price,
                f"max_used_margin_pct_total_exceeded: "
                f"projected {projected_total:.1f}% > "
                f"{self.max_used_margin_pct_total:.1f}%",
            )

        # Trade approved — record the timestamp for the daily-trades
        # rolling counter. Only append on the happy path; rejections
        # earlier in the function never reach here.
        self._recent_trade_ts.append(now_utc)

        return sizing

    def get_current_exposure(self, account: AccountSnapshot) -> dict[str, float]:
        """
        Return current used-margin percentage per symbol + total.

        {"XAUUSD": 3.5, "BTCUSD": 2.1, "total": 5.6}
        """
        positions = self._positions_provider()
        equity = max(account.equity, 1e-9)
        per_symbol: dict[str, float] = {}
        for pos in positions:
            # Margin data isn't on OpenPositionView by design — we
            # report the raw count-based exposure here and let the
            # live MT5 call do the precise per-position margin read.
            per_symbol.setdefault(pos.symbol, 0.0)
        total_used_pct = (account.margin / equity) * 100.0
        per_symbol["total"] = total_used_pct
        return per_symbol

    def has_capacity(self, symbol: str, account: AccountSnapshot) -> bool:
        """True if there is room in the portfolio for a new position."""
        positions = self._positions_provider()
        if len(positions) >= self.max_concurrent_total:
            return False
        if sum(1 for p in positions if p.symbol == symbol) >= self.max_concurrent_per_symbol:
            return False
        if self._correlated_bucket_blocked(symbol, positions) is not None:
            return False
        return self._has_free_margin_reserve(account)

    # ------------------------------------------------------------------
    def rollback_last_trade_attempt(self) -> bool:
        """
        Pop the most recent entry from the 24h trade-attempt deque.

        Called by main.py when the broker rejects an order that
        calculate_lot_size had already optimistically counted.
        Without this, repeated broker rejects burn the daily cap
        even though no position was ever opened.

        Returns True if an entry was popped, False if deque was empty
        (defensive — should never happen on the normal reject path).
        """
        with self._lock:
            if not self._recent_trade_ts:
                return False
            self._recent_trade_ts.pop()
            return True

    # ------------------------------------------------------------------
    # Hot-reload setters (Phase 10.2)
    # ------------------------------------------------------------------

    def set_max_daily_trades(self, value: int) -> None:
        """Update daily trades cap. Thread-safe."""
        if not (1 <= value <= 100):
            raise ValueError(f"max_daily_trades must be 1-100, got {value}")
        with self._lock:
            old = self.max_daily_trades
            self.max_daily_trades = value
        logger.info("PortfolioManager: max_daily_trades %d -> %d", old, value)

    def set_max_concurrent_per_symbol(self, value: int) -> None:
        """Update per-symbol concurrency cap. Thread-safe."""
        if not (1 <= value <= 20):
            raise ValueError(f"max_concurrent_per_symbol must be 1-20, got {value}")
        with self._lock:
            old = self.max_concurrent_per_symbol
            self.max_concurrent_per_symbol = value
        logger.info("PortfolioManager: max_concurrent_per_symbol %d -> %d", old, value)

    def set_max_concurrent_total(self, value: int) -> None:
        """Update total concurrency cap. Thread-safe."""
        if not (1 <= value <= 50):
            raise ValueError(f"max_concurrent_total must be 1-50, got {value}")
        with self._lock:
            old = self.max_concurrent_total
            self.max_concurrent_total = value
        logger.info("PortfolioManager: max_concurrent_total %d -> %d", old, value)

    def set_max_used_margin_pct_total(self, value: float) -> None:
        """Update total margin cap. Thread-safe."""
        if not (0.0 < value < 100.0):
            raise ValueError(f"max_used_margin_pct_total must be 0-100, got {value}")
        with self._lock:
            old = self.max_used_margin_pct_total
            self.max_used_margin_pct_total = value
        logger.info("PortfolioManager: max_used_margin_pct_total %.1f%% -> %.1f%%", old, value)

    def set_free_margin_reserve_pct(self, value: float) -> None:
        """Update free margin reserve. Thread-safe."""
        if not (0.0 < value < 100.0):
            raise ValueError(f"free_margin_reserve_pct must be 0-100, got {value}")
        with self._lock:
            old = self.free_margin_reserve_pct
            self.free_margin_reserve_pct = value
        logger.info("PortfolioManager: free_margin_reserve_pct %.1f%% -> %.1f%%", old, value)

    def update_daily_close(self, symbol: str, close: float) -> None:
        """
        Record a close for one of the correlation-bucket symbols.

        Called by the main trading loop once per tick with the current
        reference price. The name "daily_close" is aspirational — as
        long as the main loop calls this at a consistent cadence for
        both correlation symbols the resulting ρ is meaningful; when
        main.py gains a D1-aggregator in a later wave it can switch to
        feeding only on bar-close without any API change here.

        No-op for symbols outside ``CORRELATION_SYMBOLS`` so the caller
        can hand every symbol it tracks without a branch. The rolling
        window is capped at ``CORRELATION_HISTORY_LEN`` via the deque's
        maxlen.
        """
        history = self._price_history.get(symbol)
        if history is None:
            return
        try:
            history.append(float(close))
        except (TypeError, ValueError):
            logger.warning(
                "update_daily_close(%s): ignoring non-numeric close %r",
                symbol, close,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _all_prior_tier_1_done(
        self,
        prior_positions: list[OpenPositionView],
    ) -> bool:
        """
        Every prior position on the symbol must have already had the
        exit ladder fire tier 1 (partial close + stop-to-BE).

        Wave 6 fix #18 — we specifically do NOT infer this from the
        current stop level, because:
          1. A user manually dragging SL to BE in the MT5 terminal
             would spoof the old breakeven check and unlock pyramiding
             without the partial close having actually fired.
          2. A reconciled position (Wave 4 restart path) carries the
             broker's current SL but starts life with ``tier_1_done=False``
             — safer to require a fresh tier 1 fire before any new
             leg gets stacked on top.

        Unexpected direction strings block the pyramid as a defensive
        default — a malformed position row should never let new risk on.
        """
        for pos in prior_positions:
            if pos.direction not in ("buy", "sell"):
                logger.warning(
                    "Unexpected direction %r on %s — blocking pyramid",
                    pos.direction,
                    pos.symbol,
                )
                return False
            if not pos.tier_1_done:
                return False
        return True

    def _correlated_bucket_blocked(
        self,
        symbol: str,
        positions: list[OpenPositionView],
    ) -> Optional[str]:
        """
        Wave 6 fix #17 cross-asset correlation gate.

        Returns a rejection reason string when the correlated-symbol
        bucket is full, or ``None`` if the trade is allowed.

        The gate only fires when:
          1. The incoming symbol is one of ``CORRELATION_SYMBOLS``.
          2. BOTH correlation symbols have accumulated at least
             ``CORRELATION_MIN_SAMPLES`` daily closes.
          3. The rolling Pearson ρ between them exceeds
             ``CORRELATION_THRESHOLD`` in absolute value.

        When all three hold, we count open positions on the bucket
        (regardless of symbol) and block the new entry if the count has
        reached ``CORRELATED_BUCKET_MAX``. Cold-start, low-correlation
        periods, and non-bucket symbols all pass through untouched.
        """
        if symbol not in CORRELATION_SYMBOLS:
            return None

        histories = [self._price_history.get(sym) for sym in CORRELATION_SYMBOLS]
        if any(h is None or len(h) < CORRELATION_MIN_SAMPLES for h in histories):
            return None

        # Align both series on their common tail so a ragged fill still
        # produces a valid correlation — the deques are updated
        # independently per-symbol so lengths can drift by a bar.
        n = min(len(histories[0]), len(histories[1]))
        a = np.fromiter(list(histories[0])[-n:], dtype=np.float64)
        b = np.fromiter(list(histories[1])[-n:], dtype=np.float64)
        if a.std() < 1e-12 or b.std() < 1e-12:
            return None
        rho = float(np.corrcoef(a, b)[0, 1])
        if abs(rho) < CORRELATION_THRESHOLD:
            return None

        bucket_open = sum(
            1 for p in positions if p.symbol in CORRELATION_SYMBOLS
        )
        if bucket_open >= CORRELATED_BUCKET_MAX:
            return (
                f"correlation_cap_reached: rho={rho:+.2f}, "
                f"bucket_open={bucket_open}/{CORRELATED_BUCKET_MAX}"
            )
        return None

    def _has_free_margin_reserve(self, account: AccountSnapshot) -> bool:
        """
        True if free_margin / equity >= free_margin_reserve_pct.

        Equity is the denominator (not margin capacity) because at
        init time margin==0 and a ratio over margin would be
        undefined; free / equity captures the same intuition while
        staying well-defined.
        """
        if account.equity <= 0.0:
            return False
        free_pct = (account.free_margin / account.equity) * 100.0
        return free_pct >= self.free_margin_reserve_pct

    def _projected_margin_pct(
        self,
        sizing: SizingResult,
        entry_price: float,
        account: AccountSnapshot,
    ) -> float:
        """
        Rough projection of the new position's margin use as a %
        of equity. Uses notional / equity * leverage-ish estimate;
        the live broker check via mt5.order_check() is the source of
        truth — this is only a local guardrail.
        """
        if account.equity <= 0.0:
            return 100.0
        notional = sizing.lot_size * entry_price
        # For leveraged instruments (XAUUSD, BTCUSD) margin ~ notional / 100
        # on a typical 1:100 broker. We use 100 as the conservative
        # estimate; the real mt5.order_calc_margin() call downstream
        # will override this before the order is actually placed.
        estimated_margin = notional / 100.0
        return (estimated_margin / account.equity) * 100.0

    def _resolve_symbol_spec(self, symbol: str) -> Optional[SymbolSpec]:
        if self._symbol_spec_provider is None:
            return None
        try:
            return self._symbol_spec_provider(symbol)
        except Exception as exc:
            logger.warning("symbol_spec_provider(%s) raised: %s", symbol, exc)
            return None

    @staticmethod
    def _reject_zero(
        symbol: str,
        stop_price: float,
        reason: str,
    ) -> SizingResult:
        logger.info("PortfolioManager rejected %s: %s", symbol, reason)
        return SizingResult(
            symbol=symbol,
            lot_size=0.0,
            risk_amount_usd=0.0,
            entry_price=0.0,
            stop_price=stop_price,
            reason=reason,
        )
