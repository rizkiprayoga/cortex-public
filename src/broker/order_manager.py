"""
order_manager.py — MT5 Order Execution with SL-required validation

Handles placing, modifying, and closing orders on MetaTrader 5. Every
new order is validated with ``mt5.order_check()`` before being sent,
and any request that lacks a stop-loss is rejected at the module
boundary — this is the last-line defense behind PortfolioManager's
business-rule gates.

Pyramiding note
---------------
The duplicate-order guard used to live here as a blunt "reject any
(symbol, direction) within 4h" filter. With the pyramiding rules now
living in ``PortfolioManager.calculate_lot_size()``, the OrderManager
no longer blocks same-direction orders — but it still tracks them in
a ``_recent_orders`` deque so we can emit telemetry and catch runaway
scenarios. The deque is advisory only.

Key MT5 functions
-----------------
    mt5.symbol_info()        — tick size, digits, filling mode
    mt5.symbol_info_tick()   — current bid/ask
    mt5.order_check()        — validate order feasibility
    mt5.order_send()         — submit order to broker
    mt5.positions_get()      — retrieve open positions
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:  # pragma: no cover — allows unit tests w/o MT5
    mt5 = None  # type: ignore

from src.broker.mt5_connector import MT5Connector

logger = logging.getLogger(__name__)


class OrderDirection(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderRequest:
    symbol: str
    direction: OrderDirection
    lot_size: float
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    sl_points: Optional[int] = None
    tp_points: Optional[int] = None
    comment: str = "HMM_DL_Bot"
    magic: int = 234_000


@dataclass
class OrderResult:
    success: bool
    ticket: Optional[int] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    retcode: Optional[int] = None
    # --- Execution quality snapshot (E-3 Phase 1) -----------------------
    # Populated by place_order() on every send attempt; main.py persists
    # into `execution_events` when CORTEX_LOG_FILL_QUALITY=1.
    symbol: Optional[str] = None
    direction: Optional[str] = None            # "buy" | "sell"
    requested_price: Optional[float] = None    # from our request dict
    fill_price: Optional[float] = None         # send_result.price (None on reject)
    spread_at_send: Optional[float] = None     # tick.ask - tick.bid at build time
    volume_requested: Optional[float] = None
    volume_filled: Optional[float] = None      # send_result.volume (None on reject)


@dataclass
class _RecentOrder:
    symbol: str
    direction: OrderDirection
    timestamp: datetime


class OrderManager:
    """
    Places and manages orders on MetaTrader 5.

    Thread-unsafe; intended to run on the main trading loop. The
    ``_recent_orders`` deque is advisory telemetry — PortfolioManager
    is the authoritative source for "should this entry be placed".
    """

    RECENT_ORDERS_MAX = 50

    # Retcodes that indicate a transient network / broker-server
    # connectivity problem where retrying with a short delay is
    # appropriate. Real rejections (Invalid stops, No money, Market
    # closed) fall through to the caller on the first attempt.
    # Built lazily because mt5 may be None during unit tests.
    @staticmethod
    def _connection_retcodes() -> set[int]:
        if mt5 is None:
            return set()
        return {
            getattr(mt5, "TRADE_RETCODE_CONNECTION", -1),
            getattr(mt5, "TRADE_RETCODE_TIMEOUT", -1),
            getattr(mt5, "TRADE_RETCODE_REQUOTE", -1),
        } - {-1}

    def __init__(
        self,
        connector: MT5Connector,
        retry_attempts: int = 3,
        retry_backoff_sec: float = 20.0,
    ):
        self.connector = connector
        self._recent_orders: deque[_RecentOrder] = deque(maxlen=self.RECENT_ORDERS_MAX)
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_backoff_sec = max(0.0, float(retry_backoff_sec))

    def _send_with_retry(self, mt5_request):
        """
        Wrap ``mt5.order_send`` with retry on transient connection errors.

        Returns the last ``send_result`` (possibly None). Each retry
        logs a WARNING so the user can see the attempt count in the
        trading bot log.
        """
        conn_codes = self._connection_retcodes()
        last_result = None
        for attempt in range(1, self.retry_attempts + 1):
            last_result = mt5.order_send(mt5_request)
            transient = (
                last_result is None
                or getattr(last_result, "retcode", None) in conn_codes
            )
            if not transient:
                return last_result
            if attempt < self.retry_attempts:
                rc = getattr(last_result, "retcode", None) if last_result else None
                logger.warning(
                    "order_send transient failure (retcode=%s) — retry %d/%d after %.0fs",
                    rc, attempt + 1, self.retry_attempts, self.retry_backoff_sec,
                )
                time.sleep(self.retry_backoff_sec)
        return last_result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        signal,                # SignalResult
        lot_size: float,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
    ) -> OrderResult:
        """
        Build and submit a market order.

        Rejects the request early (without touching MT5) if the
        stop-loss price is missing. A trade without a stop is the one
        failure mode we refuse to tolerate — the safety model assumes
        every position has a hard bound, full stop.
        """
        if sl_price is None:
            return OrderResult(
                success=False,
                error_message="sl_price is required — refuse to place unbounded order",
            )
        if lot_size <= 0.0:
            return OrderResult(
                success=False,
                error_message=f"lot_size must be > 0 (got {lot_size})",
            )

        direction = self._signal_direction(signal)
        if direction is None:
            return OrderResult(
                success=False,
                error_message=f"invalid signal.direction={getattr(signal, 'direction', None)!r}",
            )

        # Invariants: belt-and-suspenders — fires before we touch MT5.
        # Long-only rule is enforced elsewhere (signal_combiner); this
        # guard surfaces any bypass.
        from src.safety.invariants import Severity as _InvSev, check as _inv_check
        _inv_check(
            "order.stop_loss_present",
            sl_price is not None and sl_price > 0.0,
            severity=_InvSev.WARN,
            symbol=symbol,
            context={"sl": sl_price, "lot": lot_size},
        )
        # Per-symbol long-only check: only ETH is long-only.
        # Forex pairs trade bidirectionally since 2026-04-18; XAU flipped to
        # bidirectional 2026-04-27 (Cell C A/B verdict, see CLAUDE.md). Source
        # of truth is settings.yaml::strategy.long_only_symbols — kept here as
        # a hardcoded fallback because the broker layer doesn't see config.
        _long_only_syms = {"ETHUSD"}
        if symbol in _long_only_syms:
            _inv_check(
                "order.long_only_honored",
                direction == OrderDirection.BUY,
                severity=_InvSev.ALERT,
                symbol=symbol,
                context={"direction": str(direction)},
                message=f"short order for long-only symbol {symbol}",
            )
        # SL side-of-entry check requires current price; done post-build.

        request = OrderRequest(
            symbol=symbol,
            direction=direction,
            lot_size=lot_size,
            sl_price=sl_price,
            tp_price=tp_price,
        )
        mt5_request = self._build_request(request)
        if mt5_request is None:
            return OrderResult(
                success=False,
                error_message=f"could not build MT5 request for {symbol}",
                symbol=symbol,
                direction=direction.value,
                volume_requested=lot_size,
            )

        check_ok, check_rc, check_comment = self._validate_request(mt5_request)
        if not check_ok:
            return OrderResult(
                success=False,
                retcode=check_rc,
                error_code=check_rc,
                error_message=check_comment or "mt5.order_check rejected the request",
                symbol=symbol,
                direction=direction.value,
                requested_price=mt5_request.get("price"),
                volume_requested=lot_size,
            )

        if mt5 is None:  # pragma: no cover
            return OrderResult(success=False, error_message="MetaTrader5 unavailable")

        # Snapshot spread at send time — tiny cost (one tick read), gives
        # us the bid/ask gap paired with requested_price for R-1b analysis.
        spread_at_send: Optional[float] = None
        try:
            _tick = mt5.symbol_info_tick(symbol)
            if _tick is not None:
                spread_at_send = float(_tick.ask) - float(_tick.bid)
        except Exception:
            spread_at_send = None

        send_result = self._send_with_retry(mt5_request)
        if send_result is None:
            return OrderResult(
                success=False,
                error_message="mt5.order_send returned None after retries",
                symbol=symbol,
                direction=direction.value,
                requested_price=mt5_request.get("price"),
                spread_at_send=spread_at_send,
                volume_requested=lot_size,
            )

        success = getattr(send_result, "retcode", None) == mt5.TRADE_RETCODE_DONE
        ticket = int(getattr(send_result, "order", 0)) or None

        if success:
            self._recent_orders.append(
                _RecentOrder(
                    symbol=symbol,
                    direction=direction,
                    timestamp=datetime.now(tz=timezone.utc),
                )
            )

        # Fill snapshot — only meaningful on success; MT5 sets price/volume
        # on the send_result even for some partial-reject retcodes, but we
        # trust them only when retcode=DONE.
        fill_price: Optional[float] = None
        volume_filled: Optional[float] = None
        if success:
            _fp = getattr(send_result, "price", None)
            _vf = getattr(send_result, "volume", None)
            fill_price = float(_fp) if _fp is not None else None
            volume_filled = float(_vf) if _vf is not None else None

        return OrderResult(
            success=success,
            ticket=ticket if success else None,
            retcode=getattr(send_result, "retcode", None),
            error_code=None if success else getattr(send_result, "retcode", None),
            error_message=None if success else getattr(send_result, "comment", None),
            symbol=symbol,
            direction=direction.value,
            requested_price=mt5_request.get("price"),
            fill_price=fill_price,
            spread_at_send=spread_at_send,
            volume_requested=lot_size,
            volume_filled=volume_filled,
        )

    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
    ) -> OrderResult:
        """
        Close a specific open position by ticket number.

        When ``volume`` is None, closes the full position volume (the
        original behavior). When a specific volume is passed, closes
        exactly that amount — used by ExitManager's 3-tier partial
        exit ladder (+1R / +2R / runner). The partial volume is
        floored to ``symbol_info.volume_step`` before submission; if
        that flooring collapses it to zero (partial smaller than the
        broker's minimum lot), the call is a no-op and returns a
        non-success result with an explanatory error_message so the
        caller can log and move on without killing the runner.
        """
        if mt5 is None:  # pragma: no cover
            return OrderResult(success=False, error_message="MetaTrader5 unavailable")
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(
                success=False,
                error_message=f"no open position for ticket {ticket}",
            )
        pos = positions[0]

        # Resolve the effective close volume. Floor to the symbol's
        # volume_step so we never submit a fractional amount the broker
        # will refuse. Full-close (volume=None) still uses the raw
        # position volume — the broker already tracks that exactly and
        # flooring would be incorrect at final close.
        if volume is None:
            effective_volume = float(pos.volume)
        else:
            symbol_info = mt5.symbol_info(pos.symbol)
            step = float(getattr(symbol_info, "volume_step", 0.01)) if symbol_info else 0.01
            if step <= 0.0:
                step = 0.01
            # Floor to nearest step; cap at pos.volume so we never ask
            # the broker to close more than the ticket holds.
            raw = min(float(volume), float(pos.volume))
            effective_volume = math.floor(raw / step) * step
            if effective_volume < step or effective_volume <= 0.0:
                logger.warning(
                    "close_position: partial volume %.6f on ticket %d floors to 0 "
                    "(step=%.4f, pos.volume=%.4f) — skipping partial, runner preserved",
                    float(volume),
                    ticket,
                    step,
                    float(pos.volume),
                )
                return OrderResult(
                    success=False,
                    ticket=ticket,
                    error_message=(
                        f"partial volume {float(volume):.6f} below volume_step "
                        f"{step:.4f} — skipped"
                    ),
                )

        close_type = (
            mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return OrderResult(
                success=False,
                error_message=f"no tick data for {pos.symbol}",
            )
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": float(effective_volume),
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": pos.magic or 234_000,
            "comment": "HMM_DL_Bot close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        send_result = mt5.order_send(request)
        if send_result is None:
            return OrderResult(success=False, error_message="order_send returned None")
        success = send_result.retcode == mt5.TRADE_RETCODE_DONE
        return OrderResult(
            success=success,
            ticket=ticket,
            retcode=send_result.retcode,
            error_message=None if success else send_result.comment,
        )

    def close_all_positions(
        self,
        symbol: Optional[str] = None,
    ) -> list[OrderResult]:
        """Close every open position, optionally filtered by symbol."""
        if mt5 is None:  # pragma: no cover
            return []
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if not positions:
            return []
        return [self.close_position(int(p.ticket)) for p in positions]

    def modify_sl_tp(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> OrderResult:
        """Modify SL/TP on an existing position."""
        if mt5 is None:  # pragma: no cover
            return OrderResult(success=False, error_message="MetaTrader5 unavailable")
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(
                success=False,
                error_message=f"no open position for ticket {ticket}",
            )
        pos = positions[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": pos.symbol,
            "sl": float(sl) if sl is not None else float(pos.sl),
            "tp": float(tp) if tp is not None else float(pos.tp),
        }
        send_result = mt5.order_send(request)
        if send_result is None:
            return OrderResult(success=False, error_message="order_send returned None")
        success = send_result.retcode == mt5.TRADE_RETCODE_DONE
        return OrderResult(
            success=success,
            ticket=ticket,
            retcode=send_result.retcode,
            error_message=None if success else send_result.comment,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _signal_direction(signal) -> Optional[OrderDirection]:
        direction = getattr(signal, "direction", None)
        if direction == "buy":
            return OrderDirection.BUY
        if direction == "sell":
            return OrderDirection.SELL
        return None

    def _build_request(self, order: OrderRequest) -> Optional[dict]:
        if mt5 is None:  # pragma: no cover
            return None
        symbol_info = mt5.symbol_info(order.symbol)
        if symbol_info is None:
            logger.warning("symbol_info(%s) returned None", order.symbol)
            return None
        tick = mt5.symbol_info_tick(order.symbol)
        if tick is None:
            logger.warning("symbol_info_tick(%s) returned None", order.symbol)
            return None
        if order.direction == OrderDirection.BUY:
            trade_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            trade_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        # --- SL / TP precision + broker min-distance enforcement ----------
        # MT5 rejects `retcode=10016 Invalid stops` when:
        #   (a) SL has more decimals than `symbol_info.digits` (price
        #       precision mismatch), or
        #   (b) SL sits inside `symbol_info.trade_stops_level * point`
        #       of current price (broker's minimum-distance cushion).
        # USDCAD hit this every cycle because its digits=5 and the
        # LSTM/strategy SL was raw-float without rounding, and ATR stops
        # sometimes landed inside the broker's stops_level.
        digits = int(getattr(symbol_info, "digits", 5))
        point = float(getattr(symbol_info, "point", 10 ** -digits))
        stops_level = int(getattr(symbol_info, "trade_stops_level", 0))
        min_dist = max(float(stops_level) * point, 0.0)

        sl_val = float(order.sl_price) if order.sl_price is not None else 0.0
        tp_val = float(order.tp_price) if order.tp_price is not None else 0.0

        if sl_val > 0.0:
            sl_val = round(sl_val, digits)
            if order.direction == OrderDirection.BUY:
                # For longs, SL must be below price by at least min_dist.
                max_allowed_sl = round(price - min_dist, digits)
                if sl_val > max_allowed_sl:
                    logger.warning(
                        "[%s] SL %.*f inside broker stops_level (price=%.*f "
                        "min_dist=%.*f) — widening to %.*f",
                        order.symbol, digits, sl_val, digits, price,
                        digits, min_dist, digits, max_allowed_sl,
                    )
                    sl_val = max_allowed_sl
            else:
                # For shorts, SL must be above price by at least min_dist.
                min_allowed_sl = round(price + min_dist, digits)
                if sl_val < min_allowed_sl:
                    logger.warning(
                        "[%s] SL %.*f inside broker stops_level (price=%.*f "
                        "min_dist=%.*f) — widening to %.*f",
                        order.symbol, digits, sl_val, digits, price,
                        digits, min_dist, digits, min_allowed_sl,
                    )
                    sl_val = min_allowed_sl

        if tp_val > 0.0:
            tp_val = round(tp_val, digits)
            # TP must be on the profit side with ≥ min_dist cushion too.
            if order.direction == OrderDirection.BUY:
                min_allowed_tp = round(price + min_dist, digits)
                if tp_val < min_allowed_tp:
                    tp_val = min_allowed_tp
            else:
                max_allowed_tp = round(price - min_dist, digits)
                if tp_val > max_allowed_tp:
                    tp_val = max_allowed_tp

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": float(order.lot_size),
            "type": trade_type,
            "price": round(price, digits),
            "sl": sl_val,
            "tp": tp_val,
            "deviation": 20,
            "magic": order.magic,
            "comment": order.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        return request

    def _validate_request(self, request: dict) -> tuple[bool, Optional[int], Optional[str]]:
        """
        Run mt5.order_check() against the request.

        Returns ``(ok, retcode, comment)``. On success ``retcode``/``comment``
        are None. On failure the broker's actual retcode + comment are
        surfaced so the caller can put them in ``OrderResult.retcode`` /
        ``error_message`` (otherwise the reject path logs "retcode=None"
        which is useless for debugging).
        """
        if mt5 is None:  # pragma: no cover
            return False, None, "MetaTrader5 unavailable"
        check = mt5.order_check(request)
        if check is None:
            logger.warning("order_check returned None for %s", request.get("symbol"))
            return False, None, "order_check returned None"
        if check.retcode != 0:
            comment = getattr(check, "comment", "") or ""
            logger.warning(
                "order_check rejected %s: retcode=%d comment=%s",
                request.get("symbol"),
                check.retcode,
                comment,
            )
            return False, int(check.retcode), comment
        return True, None, None
