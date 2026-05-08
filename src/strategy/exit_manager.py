"""
exit_manager.py — Triple Barrier exit system + reversal hard-exit.

Walks the list of open positions every tick and emits a list of
``ExitAction`` records telling the broker layer what to do. The broker
then translates those into ``OrderManager.close_position(..., volume)``
and ``OrderManager.modify_sl_tp(...)`` calls.

Triple Barrier semantics (replaces the old 3-tier partial close ladder)
----------------------------------------------------------------------
Evidence: 567,000 backtests show combined exits (partial close + trail)
underperform simpler exits. Target exits outperform "let profits run."

Let R = abs(entry_price - initial_stop) — computed once at entry.

  Upper barrier (Take-Profit):
      Full close at ``tp_r_multiple`` (default +2.5R)

  Lower barrier (Stop-Loss):
      Full close when price hits initial_stop (managed by broker SL)

  Vertical barrier (Time Exit):
      Full close after ``time_exit_bars`` H1 bars have elapsed since
      entry. Counter is wall-clock-derived (``now - opened_at``) so it
      is independent of main-loop tick cadence and survives restarts.
      Per-position ``OpenPosition.time_exit_bars`` takes precedence over
      ``ExitManager.time_exit_bars`` so each symbol gets its own limit.
      Per-symbol values live in settings.yaml (H1 bars). Sprint 3 tuned
      values (2026-05-01) for the trading universe pairs — see joint sweep results:
        XAUUSD 60 (~2.5d)   # doc-check: strategy.per_symbol_params.XAUUSD.time_exit_h1_bars
        EURUSD 60 (~2.5d)   # doc-check: strategy.per_symbol_params.EURUSD.time_exit_h1_bars
        USDJPY 40 (~1.7d)   # doc-check: strategy.per_symbol_params.USDJPY.time_exit_h1_bars
        USDCAD 40 (~1.7d)   # doc-check: strategy.per_symbol_params.USDCAD.time_exit_h1_bars
        ETHUSD 100 (~4d)    # doc-check: strategy.per_symbol_params.ETHUSD.time_exit_h1_bars

  Breakeven lock:
      At ``be_trigger_r`` (default +1R), move stop to entry price.
      This is the ONE combined exit that works per the evidence.

Hard-exit override
------------------
If the M15 combined signal has been in the *opposite* direction for
``reversal_bars_required`` (default 4) consecutive bars, close the entire
remaining volume regardless. Symmetric to the entry flickering check.

One action per position per tick
--------------------------------
The ``check_exits()`` method emits at most one action per position per
call so the broker can round-trip each instruction cleanly.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Symbols that trade 24/7 (crypto). For everything else we assume
# forex/metals hours: Sat 00:00 UTC → Mon 00:00 UTC is closed.
_CONTINUOUS_SYMBOLS: frozenset[str] = frozenset({"ETHUSD", "BTCUSD"})


def _weekend_hours(start: datetime, end: datetime) -> int:
    """
    Hours within [start, end) that fall in the Sat 00:00 → Mon 00:00 UTC
    window. Used to subtract weekend time from the H1 bar counter so
    time_exit fires after N *trading* hours (matching backtest bar counts)
    instead of wall-clock hours.
    """
    if end <= start:
        return 0
    start_midnight = start.replace(hour=0, minute=0, second=0, microsecond=0)
    monday = start_midnight - timedelta(days=start_midnight.weekday())
    sat = monday + timedelta(days=5)
    total = 0.0
    while sat < end:
        wknd_start = sat
        wknd_end = sat + timedelta(days=2)
        overlap_start = max(wknd_start, start)
        overlap_end = min(wknd_end, end)
        if overlap_end > overlap_start:
            total += (overlap_end - overlap_start).total_seconds()
        sat += timedelta(days=7)
    return int(total // 3600)


# Canonical exit reason codes — any new ExitAction.reason prefix must be
# mapped here. Anything unmapped falls through to "unknown" so the trade
# journal never silently drops information.
REASON_CODES = {
    "take_profit",          # Upper barrier hit (R >= tp_r_multiple)
    "stop_loss",             # Lower barrier (server-side SL or BE-stop) hit
    "time_exit",             # Vertical barrier (bars_held >= time_exit_bars)
    "reversal_hard_exit",    # N consecutive opposite M15 signals
    "manual",                # Dashboard/operator close
    "breaker_emergency",     # RiskMonitor / CircuitBreaker veto
    "unknown",
}


def classify_reason(reason: str | None) -> str:
    """
    Map an ExitAction.reason string (or ad-hoc broker-layer text) to a
    canonical close_reason_code. Prefix match — first token up to ':'.

    Returns one of REASON_CODES, always; never raises. Unmapped input
    returns "unknown".
    """
    if not reason:
        return "unknown"
    head = reason.split(":", 1)[0].strip().lower()
    if head in REASON_CODES:
        return head
    # Common aliases from the broker reconcile path
    if head in ("sl", "stop", "stop-loss", "stoploss"):
        return "stop_loss"
    if head in ("tp", "takeprofit", "take-profit"):
        return "take_profit"
    if "emergency" in head or "breaker" in head:
        return "breaker_emergency"
    if "manual" in head or "dashboard" in head:
        return "manual"
    return "unknown"


@dataclass
class OpenPosition:
    """
    State carrier for a position managed by ExitManager.

    In production, this is rebuilt each tick from ``mt5.positions_get()``
    plus a side-car store for the BE lock and tick counter. For testing
    the ``check_exits()`` method, construct OpenPositions directly.
    """

    symbol: str
    ticket: int
    direction: str          # "buy" or "sell"
    entry_price: float
    initial_stop: float
    current_stop: float
    volume: float           # current position volume
    initial_volume: float   # locked at entry time
    atr_trail_mult: float   # from the strategy that opened this position
    strategy_name: str = ""
    # Triple Barrier state
    be_locked: bool = False         # Breakeven stop has been set
    bars_held: int = 0              # H1 bars elapsed since entry (derived from opened_at)
    tp_price: float = 0.0           # Take-profit price (set at entry)
    # Per-position Triple Barrier thresholds. Seeded at entry from
    # settings.yaml::strategy.per_symbol_params.<symbol>.{time_exit_h1_bars,
    # tp_r_multiple, be_trigger_r}. When unset (0/0.0), ExitManager falls
    # back to its class-level defaults. Sprint 3 audit fix (2026-05-01):
    # tp_r_multiple and be_trigger_r are now per-pair like time_exit_bars,
    # closing the live/backtest divergence on Triple Barrier params.
    time_exit_bars: int = 0
    tp_r_multiple: float = 0.0      # 0 = use ExitManager class default
    be_trigger_r: float = 0.0       # 0 = use ExitManager class default
    # Legacy compatibility — tier_1_done maps to be_locked for pyramiding gate
    @property
    def tier_1_done(self) -> bool:
        return self.be_locked
    # Reversal tracking
    opened_at: Optional[datetime] = None


class TierStateStore:
    """
    Side-car JSON store for per-ticket exit state.

    Persists BE lock status and bars_held across restarts so the exit
    manager picks up where it left off. Without persistence, a restart
    would reset bars_held to 0 (delaying time exits) and re-trigger
    the BE lock (safe no-op, but noisy).

    Schema:
        {
            "<ticket>": {
                "be_locked":      bool,
                "bars_held":      int,
                "initial_stop_R": float,
                "initial_volume": float,
                "tp_price":       float,
                "opened_at":      "<iso8601>"
            },
            ...
        }
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                self._data = loaded
                logger.info(
                    "TierStateStore: loaded %d tickets from %s",
                    len(self._data),
                    self.path,
                )
            else:
                logger.warning(
                    "TierStateStore: %s is not a JSON object — starting empty",
                    self.path,
                )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "TierStateStore: failed to load %s: %s — starting empty",
                self.path,
                exc,
            )
            self._data = {}

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, sort_keys=True)
            tmp.replace(self.path)
        except OSError as exc:
            logger.warning(
                "TierStateStore: failed to write %s: %s",
                self.path,
                exc,
            )

    def get(self, ticket: int) -> Optional[dict]:
        return self._data.get(str(ticket))

    def upsert(self, ticket: int, **kwargs) -> None:
        """
        Merge ``kwargs`` into the record for ``ticket`` and flush.
        Missing records are created.
        """
        record = self._data.setdefault(str(ticket), {})
        record.update(kwargs)
        self._flush()

    def delete(self, ticket: int) -> None:
        if str(ticket) in self._data:
            del self._data[str(ticket)]
            self._flush()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, ticket: int) -> bool:
        return str(ticket) in self._data


@dataclass
class ExitAction:
    """
    Instruction record produced by ExitManager.check_exits().

    The broker layer consumes these and issues MT5 calls. The exit
    manager itself never touches MT5 — it's pure logic that can be
    exhaustively simulated in tests.
    """

    ticket: int
    action: str                 # "partial_close" | "modify_stop" | "full_close"
    close_volume: Optional[float] = None
    new_stop: Optional[float] = None
    reason: str = ""


class ExitManager:
    """
    Triple Barrier exit manager.

    Replaces the old 3-tier partial close ladder with evidence-based
    simpler exits: fixed TP + SL + time expiry + BE lock.
    """

    def __init__(
        self,
        # PLACEHOLDERS — tuned production values redacted from this public template.
        tp_r_multiple: float = 0.0,
        be_trigger_r: float = 0.0,
        time_exit_bars: int = 0,
        reversal_bars_required: int = 0,
        tier_state_store: Optional[TierStateStore] = None,
        # Legacy compatibility — accepted but ignored
        partial_tiers: tuple[float, ...] = (),
        partial_fractions: tuple[float, ...] = (),
    ):
        self.tp_r_multiple = tp_r_multiple
        self.be_trigger_r = be_trigger_r
        self.time_exit_bars = time_exit_bars
        self.reversal_bars_required = reversal_bars_required
        self.tier_state_store = tier_state_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_exits(
        self,
        positions: list[OpenPosition],
        current_prices: dict[str, float],
        current_atrs: dict[str, float],
        recent_signals: Optional[dict[str, list[str]]] = None,
        now: Optional[datetime] = None,
    ) -> list[ExitAction]:
        """
        Walk positions and emit exit instructions for this tick.

        Triple Barrier priority (per position):
        1. Reversal hard-exit (4 opposite signals)
        2. Time exit (bars_held >= time_exit_bars)
        3. Take-profit (R >= tp_r_multiple)
        4. Breakeven lock (R >= be_trigger_r, move stop to entry)
        5. Stop-loss is handled by the broker (MT5 server-side SL)
        """
        actions: list[ExitAction] = []

        # ---- Reversal hard-exit (newest leg per symbol) ------------------
        reversal_closed: set[int] = set()
        by_symbol: dict[str, list[OpenPosition]] = {}
        for pos in positions:
            by_symbol.setdefault(pos.symbol, []).append(pos)

        for symbol, legs in by_symbol.items():
            if not legs:
                continue
            if not self._reversal_triggered(legs[0], recent_signals):
                continue
            newest = max(
                legs,
                key=lambda p: (
                    p.opened_at if p.opened_at is not None
                    else datetime.min.replace(tzinfo=timezone.utc)
                ),
            )
            reversal_closed.add(newest.ticket)
            actions.append(
                ExitAction(
                    ticket=newest.ticket,
                    action="full_close",
                    close_volume=newest.volume,
                    reason=(
                        f"reversal_hard_exit: {len(legs)} legs on "
                        f"{symbol}, closing newest (ticket={newest.ticket}) "
                        f"after {self.reversal_bars_required} consecutive "
                        f"{self._opposite(newest.direction)} signals"
                    ),
                )
            )

        # ---- Triple Barrier walk -----------------------------------------
        for pos in positions:
            if pos.ticket in reversal_closed:
                continue

            price = current_prices.get(pos.symbol)
            atr = current_atrs.get(pos.symbol)
            if price is None or atr is None or atr <= 0.0:
                continue

            r = self._risk_unit(pos)
            if r <= 0.0:
                continue

            r_multiple = self._r_multiple(pos, price, r)

            # Derive H1 bars_held from wall-clock since entry. Using a
            # time-derived counter (rather than per-tick increment) makes
            # the threshold independent of main-loop tick rate and of
            # restarts — the old scheme silently ran at M15 cadence.
            pos.bars_held = self._h1_bars_elapsed(pos, now=now)

            # Per-position threshold wins; class default is the fallback
            # for legacy callers / tests that don't set it.
            limit = pos.time_exit_bars if pos.time_exit_bars > 0 else self.time_exit_bars

            # ---- Vertical barrier: Time exit -----------------------------
            if pos.bars_held >= limit:
                # Invariant: bars_held should be within ~6h of the limit
                # when time-exit fires live. Large overshoots are legit
                # for positions recovered after a long bot-down window;
                # demoted to WARN to avoid Telegram spam, still logs
                # context for post-hoc analysis.
                from src.safety.invariants import Severity, check as _inv_check
                _inv_check(
                    "strategy.time_exit_bar_count",
                    pos.bars_held <= limit + 6,
                    severity=Severity.WARN,
                    symbol=pos.symbol,
                    context={
                        "ticket": pos.ticket,
                        "bars_held": pos.bars_held,
                        "limit": limit,
                    },
                    message=(
                        f"time_exit fired at bars_held={pos.bars_held} "
                        f"for limit={limit} (possibly recovered position)"
                    ),
                )
                actions.append(
                    ExitAction(
                        ticket=pos.ticket,
                        action="full_close",
                        close_volume=pos.volume,
                        reason=(
                            f"time_exit: {pos.bars_held} H1 bars held "
                            f"≥ {limit} limit, R={r_multiple:.2f}"
                        ),
                    )
                )
                if self.tier_state_store is not None:
                    self.tier_state_store.upsert(
                        pos.ticket, bars_held=pos.bars_held,
                    )
                continue

            # ---- Upper barrier: Take-profit at TP R-multiple -------------
            # Per-pair tp_r_multiple from yaml (seeded on OpenPosition at entry);
            # falls back to ExitManager class default when unset.
            tp_r = pos.tp_r_multiple if pos.tp_r_multiple > 0 else self.tp_r_multiple
            if r_multiple >= tp_r:
                actions.append(
                    ExitAction(
                        ticket=pos.ticket,
                        action="full_close",
                        close_volume=pos.volume,
                        reason=(
                            f"take_profit: R={r_multiple:.2f} "
                            f"≥ {tp_r:.1f} TP barrier"
                        ),
                    )
                )
                continue

            # ---- BE lock: Move stop to entry at BE trigger ---------------
            # Per-pair be_trigger_r from yaml; falls back to class default.
            be_r = pos.be_trigger_r if pos.be_trigger_r > 0 else self.be_trigger_r
            if not pos.be_locked and r_multiple >= be_r:
                new_stop = pos.entry_price
                actions.append(
                    ExitAction(
                        ticket=pos.ticket,
                        action="modify_stop",
                        new_stop=new_stop,
                        reason=(
                            f"be_lock: R={r_multiple:.2f} "
                            f"≥ {be_r:.1f}, "
                            "moving stop to breakeven"
                        ),
                    )
                )
                pos.be_locked = True
                pos.current_stop = new_stop
                if self.tier_state_store is not None:
                    self.tier_state_store.upsert(
                        pos.ticket, be_locked=True,
                        bars_held=pos.bars_held,
                    )
                continue

            # ---- Persist bars_held periodically --------------------------
            if self.tier_state_store is not None and pos.bars_held % 5 == 0:
                self.tier_state_store.upsert(
                    pos.ticket, bars_held=pos.bars_held,
                )

        return actions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _h1_bars_elapsed(pos: OpenPosition, now: Optional[datetime] = None) -> int:
        """
        Number of full H1 *trading* bars elapsed since ``pos.opened_at``.

        For forex/metals symbols the weekend window (Sat 00:00 → Mon 00:00
        UTC) is excluded so the counter matches the backtest's bar-indexed
        `bars_held` — a position held over the weekend no longer burns 48
        phantom hours toward the time-exit limit. Crypto symbols
        (ETHUSD/BTCUSD) trade 24/7 and keep wall-clock behaviour.

        Returns 0 when ``opened_at`` is missing (legacy positions) so the
        time-exit never fires spuriously — the operator can close manually.
        """
        if pos.opened_at is None:
            return 0
        ref = now or datetime.now(tz=timezone.utc)
        opened = pos.opened_at
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        delta_hours = int((ref - opened).total_seconds() // 3600)
        if pos.symbol in _CONTINUOUS_SYMBOLS:
            return max(0, delta_hours)
        return max(0, delta_hours - _weekend_hours(opened, ref))

    @staticmethod
    def _risk_unit(pos: OpenPosition) -> float:
        return abs(pos.entry_price - pos.initial_stop)

    @staticmethod
    def _r_multiple(pos: OpenPosition, price: float, r: float) -> float:
        if pos.direction == "buy":
            return (price - pos.entry_price) / r
        return (pos.entry_price - price) / r

    @staticmethod
    def _opposite(direction: str) -> str:
        return "sell" if direction == "buy" else "buy"

    def _reversal_triggered(
        self,
        pos: OpenPosition,
        recent_signals: Optional[dict[str, list[str]]],
    ) -> bool:
        if recent_signals is None:
            return False
        history = recent_signals.get(pos.symbol)
        if not history or len(history) < self.reversal_bars_required:
            return False
        tail = history[-self.reversal_bars_required:]
        opposite = self._opposite(pos.direction)
        return all(d == opposite for d in tail)
