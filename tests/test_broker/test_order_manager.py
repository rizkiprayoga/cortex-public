"""
Tests for OrderManager — Wave 6 fix #1 + #9: partial-close volume + floor.

Before Wave 6, ``close_position(ticket)`` had no ``volume`` parameter and
always submitted ``pos.volume`` to the broker — which silently liquidated
the whole runner the first time the exit ladder fired its tier-1 partial.
The 3-tier exit ladder looked fine in unit tests because the assertions
only inspected the ``ExitAction`` dataclass, not the broker interaction.

These tests pin the new behavior: when ``volume`` is passed, the request
carries exactly that amount (floored to ``symbol_info.volume_step``); if
the flooring collapses the partial to zero we skip the order entirely and
return a no-op so the runner stays intact.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# MetaTrader5 is Windows-only and optional — stub it out if missing so
# ``src.broker.order_manager`` imports cleanly. We re-patch the module
# attribute inside each test with a fresh MagicMock.
if "MetaTrader5" not in sys.modules:
    sys.modules["MetaTrader5"] = MagicMock()

from src.broker import order_manager as order_manager_module
from src.broker.order_manager import OrderManager


def _fake_position(ticket: int, symbol: str, volume: float, pos_type: int = 0):
    p = types.SimpleNamespace()
    p.ticket = ticket
    p.symbol = symbol
    p.volume = volume
    p.type = pos_type
    p.magic = 234_000
    return p


def _fake_symbol_info(volume_step: float = 0.01):
    si = types.SimpleNamespace()
    si.volume_step = volume_step
    return si


def _fake_tick(bid: float = 1999.0, ask: float = 2001.0):
    t = types.SimpleNamespace()
    t.bid = bid
    t.ask = ask
    return t


def _fake_send_result(retcode_done: int = 10009):
    r = types.SimpleNamespace()
    r.retcode = retcode_done
    r.comment = "ok"
    return r


class TestClosePositionPartialVolume:
    """Wave 6 fix #1 + #9 — partial-close volume flows through to broker."""

    def test_partial_volume_is_passed_to_order_send(self):
        """
        close_position(ticket=111, volume=0.05) on a pos with volume=0.15
        should submit an order_send request carrying volume=0.05, floored
        to the symbol's volume_step (0.01 in this case, so no actual change).
        """
        fake_mt5 = MagicMock()
        fake_mt5.POSITION_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_SELL = 1
        fake_mt5.TRADE_ACTION_DEAL = 1
        fake_mt5.ORDER_TIME_GTC = 0
        fake_mt5.ORDER_FILLING_IOC = 1
        fake_mt5.TRADE_RETCODE_DONE = 10009

        fake_mt5.positions_get.return_value = (
            _fake_position(111, "XAUUSD", volume=0.15, pos_type=0),
        )
        fake_mt5.symbol_info.return_value = _fake_symbol_info(volume_step=0.01)
        fake_mt5.symbol_info_tick.return_value = _fake_tick()
        fake_mt5.order_send.return_value = _fake_send_result(10009)

        with patch.object(order_manager_module, "mt5", fake_mt5):
            om = OrderManager(connector=MagicMock())
            result = om.close_position(111, volume=0.05)

        assert result.success is True
        fake_mt5.order_send.assert_called_once()
        submitted = fake_mt5.order_send.call_args[0][0]
        assert submitted["volume"] == pytest.approx(0.05)
        assert submitted["position"] == 111
        assert submitted["symbol"] == "XAUUSD"

    def test_partial_volume_floored_to_step(self):
        """
        Request 0.037 volume on a step=0.01 symbol → floor to 0.03
        (math.floor(0.037 / 0.01) * 0.01 = 0.03).
        """
        fake_mt5 = MagicMock()
        fake_mt5.POSITION_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_SELL = 1
        fake_mt5.TRADE_ACTION_DEAL = 1
        fake_mt5.ORDER_TIME_GTC = 0
        fake_mt5.ORDER_FILLING_IOC = 1
        fake_mt5.TRADE_RETCODE_DONE = 10009

        fake_mt5.positions_get.return_value = (
            _fake_position(222, "XAUUSD", volume=0.15, pos_type=0),
        )
        fake_mt5.symbol_info.return_value = _fake_symbol_info(volume_step=0.01)
        fake_mt5.symbol_info_tick.return_value = _fake_tick()
        fake_mt5.order_send.return_value = _fake_send_result(10009)

        with patch.object(order_manager_module, "mt5", fake_mt5):
            om = OrderManager(connector=MagicMock())
            result = om.close_position(222, volume=0.037)

        assert result.success is True
        submitted = fake_mt5.order_send.call_args[0][0]
        assert submitted["volume"] == pytest.approx(0.03, abs=1e-9)

    def test_partial_volume_capped_at_position_volume(self):
        """
        Asking for more than pos.volume should be capped at pos.volume
        — never ask the broker to close more than the ticket holds.
        """
        fake_mt5 = MagicMock()
        fake_mt5.POSITION_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_SELL = 1
        fake_mt5.TRADE_ACTION_DEAL = 1
        fake_mt5.ORDER_TIME_GTC = 0
        fake_mt5.ORDER_FILLING_IOC = 1
        fake_mt5.TRADE_RETCODE_DONE = 10009

        fake_mt5.positions_get.return_value = (
            _fake_position(333, "XAUUSD", volume=0.10, pos_type=0),
        )
        fake_mt5.symbol_info.return_value = _fake_symbol_info(volume_step=0.01)
        fake_mt5.symbol_info_tick.return_value = _fake_tick()
        fake_mt5.order_send.return_value = _fake_send_result(10009)

        with patch.object(order_manager_module, "mt5", fake_mt5):
            om = OrderManager(connector=MagicMock())
            result = om.close_position(333, volume=0.50)

        assert result.success is True
        submitted = fake_mt5.order_send.call_args[0][0]
        assert submitted["volume"] == pytest.approx(0.10, abs=1e-9)

    def test_partial_below_step_is_skipped(self):
        """
        Partial volume smaller than volume_step floors to 0 → we refuse
        to submit, log a WARNING, and return a non-success OrderResult
        so the runner stays intact. The important invariant: the broker
        is NEVER called with volume=0.
        """
        fake_mt5 = MagicMock()
        fake_mt5.POSITION_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_SELL = 1
        fake_mt5.TRADE_ACTION_DEAL = 1
        fake_mt5.ORDER_TIME_GTC = 0
        fake_mt5.ORDER_FILLING_IOC = 1
        fake_mt5.TRADE_RETCODE_DONE = 10009

        fake_mt5.positions_get.return_value = (
            _fake_position(444, "XAUUSD", volume=0.15, pos_type=0),
        )
        fake_mt5.symbol_info.return_value = _fake_symbol_info(volume_step=0.01)
        fake_mt5.symbol_info_tick.return_value = _fake_tick()

        with patch.object(order_manager_module, "mt5", fake_mt5):
            om = OrderManager(connector=MagicMock())
            result = om.close_position(444, volume=0.005)

        assert result.success is False
        assert result.ticket == 444
        assert "below volume_step" in (result.error_message or "")
        fake_mt5.order_send.assert_not_called()

    def test_full_close_ignores_volume_step_flooring(self):
        """
        Backward-compatible path: close_position(ticket) with no volume
        argument still sends the full pos.volume — we do NOT floor it,
        because the broker tracks it exactly and flooring could leave a
        dust-sized residual position behind on final close.
        """
        fake_mt5 = MagicMock()
        fake_mt5.POSITION_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_SELL = 1
        fake_mt5.TRADE_ACTION_DEAL = 1
        fake_mt5.ORDER_TIME_GTC = 0
        fake_mt5.ORDER_FILLING_IOC = 1
        fake_mt5.TRADE_RETCODE_DONE = 10009

        fake_mt5.positions_get.return_value = (
            _fake_position(555, "XAUUSD", volume=0.157, pos_type=0),
        )
        fake_mt5.symbol_info.return_value = _fake_symbol_info(volume_step=0.01)
        fake_mt5.symbol_info_tick.return_value = _fake_tick()
        fake_mt5.order_send.return_value = _fake_send_result(10009)

        with patch.object(order_manager_module, "mt5", fake_mt5):
            om = OrderManager(connector=MagicMock())
            result = om.close_position(555)

        assert result.success is True
        submitted = fake_mt5.order_send.call_args[0][0]
        # Un-floored — exact broker value.
        assert submitted["volume"] == pytest.approx(0.157, abs=1e-9)


# -----------------------------------------------------------------------
# Connection-loss retry (Phase P follow-up)
# -----------------------------------------------------------------------

class TestSendWithRetry:
    """_send_with_retry retries only on transient connection retcodes."""

    def _fake_signal(self, direction: str = "buy", symbol: str = "XAUUSD"):
        sig = types.SimpleNamespace()
        sig.direction = direction
        sig.symbol = symbol
        sig.should_trade = True
        sig.confidence = 0.8
        sig.combined_score = 0.5
        return sig

    def _wire_mt5(self, retcodes_sequence):
        """Return a fake mt5 object whose order_send yields the given retcodes."""
        fake_mt5 = MagicMock()
        fake_mt5.POSITION_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_SELL = 1
        fake_mt5.TRADE_ACTION_DEAL = 1
        fake_mt5.ORDER_TIME_GTC = 0
        fake_mt5.ORDER_FILLING_IOC = 1
        fake_mt5.TRADE_RETCODE_DONE = 10009
        fake_mt5.TRADE_RETCODE_CONNECTION = 10031
        fake_mt5.TRADE_RETCODE_TIMEOUT = 10008
        fake_mt5.TRADE_RETCODE_REQUOTE = 10004

        # symbol_info + tick + order_check all succeed
        si = types.SimpleNamespace(
            volume_step=0.01, volume_min=0.01, volume_max=100.0,
            digits=2, point=0.01, trade_tick_value=1.0, trade_tick_size=0.01,
            filling_mode=1, currency_quote="USD",
        )
        fake_mt5.symbol_info.return_value = si
        fake_mt5.symbol_info_tick.return_value = _fake_tick()
        fake_mt5.order_check.return_value = types.SimpleNamespace(retcode=0)

        # Sequence of send results
        send_results = []
        for rc in retcodes_sequence:
            if rc is None:
                send_results.append(None)
            else:
                send_results.append(types.SimpleNamespace(
                    retcode=rc,
                    order=99 if rc == fake_mt5.TRADE_RETCODE_DONE else 0,
                    comment="ok" if rc == fake_mt5.TRADE_RETCODE_DONE else "err",
                ))
        fake_mt5.order_send.side_effect = send_results
        return fake_mt5

    def test_retries_on_connection_reject_then_succeeds(self):
        """None then connection then DONE → 3 sends, sleep twice, success."""
        fake_mt5 = self._wire_mt5([None, 10031, 10009])

        with patch.object(order_manager_module, "mt5", fake_mt5), \
             patch.object(order_manager_module.time, "sleep") as mock_sleep:
            om = OrderManager(connector=MagicMock(), retry_attempts=3, retry_backoff_sec=20.0)
            result = om.place_order(
                symbol="XAUUSD",
                signal=self._fake_signal(),
                lot_size=0.1,
                sl_price=1990.0,
            )

        assert result.success is True
        assert fake_mt5.order_send.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(20.0)

    def test_no_retry_on_non_connection_reject(self):
        """Invalid-stops retcode (10016) fails fast — one call, no sleep."""
        fake_mt5 = self._wire_mt5([10016])

        with patch.object(order_manager_module, "mt5", fake_mt5), \
             patch.object(order_manager_module.time, "sleep") as mock_sleep:
            om = OrderManager(connector=MagicMock(), retry_attempts=3, retry_backoff_sec=20.0)
            result = om.place_order(
                symbol="XAUUSD",
                signal=self._fake_signal(),
                lot_size=0.1,
                sl_price=1990.0,
            )

        assert result.success is False
        assert fake_mt5.order_send.call_count == 1
        mock_sleep.assert_not_called()

    def test_retries_exhausted_returns_last_reject(self):
        """4× connection → 3 attempts, 2 sleeps, final result is the reject."""
        fake_mt5 = self._wire_mt5([10031, 10031, 10031, 10031])

        with patch.object(order_manager_module, "mt5", fake_mt5), \
             patch.object(order_manager_module.time, "sleep") as mock_sleep:
            om = OrderManager(connector=MagicMock(), retry_attempts=3, retry_backoff_sec=20.0)
            result = om.place_order(
                symbol="XAUUSD",
                signal=self._fake_signal(),
                lot_size=0.1,
                sl_price=1990.0,
            )

        assert result.success is False
        assert result.retcode == 10031
        assert fake_mt5.order_send.call_count == 3
        assert mock_sleep.call_count == 2


# -----------------------------------------------------------------------
# SL rounding + broker stops_level widening (Phase 1 of post-CW work)
# -----------------------------------------------------------------------

class TestStopsLevelAndPrecision:
    """_build_request must round SL to digits AND widen SL out of stops_level."""

    def _fake_sym_info(self, digits=5, point=0.00001, stops_level=0, volume_min=0.01):
        si = types.SimpleNamespace()
        si.digits = digits
        si.point = point
        si.trade_stops_level = stops_level
        si.volume_step = 0.01
        si.volume_min = volume_min
        si.volume_max = 100.0
        si.filling_mode = 1
        si.trade_tick_value = 1.0
        si.trade_tick_size = point
        si.currency_quote = "USD"
        return si

    def test_sl_rounded_to_digits(self):
        """A BUY with sl_price=1.234567891 on a 5-digit symbol must round to 1.23457."""
        fake_mt5 = MagicMock()
        fake_mt5.ORDER_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_SELL = 1
        fake_mt5.TRADE_ACTION_DEAL = 1
        fake_mt5.ORDER_TIME_GTC = 0
        fake_mt5.ORDER_FILLING_IOC = 1
        fake_mt5.symbol_info.return_value = self._fake_sym_info(digits=5, point=0.00001, stops_level=0)
        fake_mt5.symbol_info_tick.return_value = _fake_tick(bid=1.50000, ask=1.50010)

        with patch.object(order_manager_module, "mt5", fake_mt5):
            om = OrderManager(connector=MagicMock())
            req = om._build_request(types.SimpleNamespace(
                symbol="USDCAD", direction=order_manager_module.OrderDirection.BUY,
                lot_size=0.1, sl_price=1.234567891, tp_price=None,
                magic=234_000, comment="test",
            ))

        assert req is not None
        # SL should be rounded to 5 digits: 1.23457
        assert round(req["sl"], 5) == pytest.approx(1.23457, abs=1e-9)

    def test_sl_widened_when_inside_stops_level(self):
        """BUY @ 1.50000 with SL=1.49998 and stops_level=10 points → SL widened to 1.49990."""
        fake_mt5 = MagicMock()
        fake_mt5.ORDER_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_SELL = 1
        fake_mt5.TRADE_ACTION_DEAL = 1
        fake_mt5.ORDER_TIME_GTC = 0
        fake_mt5.ORDER_FILLING_IOC = 1
        fake_mt5.symbol_info.return_value = self._fake_sym_info(
            digits=5, point=0.00001, stops_level=10,  # stops_level = 10 points = 0.00010
        )
        fake_mt5.symbol_info_tick.return_value = _fake_tick(bid=1.49990, ask=1.50000)

        with patch.object(order_manager_module, "mt5", fake_mt5):
            om = OrderManager(connector=MagicMock())
            req = om._build_request(types.SimpleNamespace(
                symbol="USDCAD", direction=order_manager_module.OrderDirection.BUY,
                lot_size=0.1, sl_price=1.49998, tp_price=None,
                magic=234_000, comment="test",
            ))

        assert req is not None
        # SL 1.49998 is only 2 points below ask 1.50000, inside 10-point cushion
        # Should widen to 1.50000 - 10*0.00001 = 1.49990
        assert round(req["sl"], 5) == pytest.approx(1.49990, abs=1e-9)

    def test_sell_sl_widened_when_inside_stops_level(self):
        """SELL @ 1.50000 with SL=1.50002 and stops_level=10 pts → widened to 1.50010."""
        fake_mt5 = MagicMock()
        fake_mt5.ORDER_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_SELL = 1
        fake_mt5.TRADE_ACTION_DEAL = 1
        fake_mt5.ORDER_TIME_GTC = 0
        fake_mt5.ORDER_FILLING_IOC = 1
        fake_mt5.symbol_info.return_value = self._fake_sym_info(
            digits=5, point=0.00001, stops_level=10,
        )
        fake_mt5.symbol_info_tick.return_value = _fake_tick(bid=1.50000, ask=1.50010)

        with patch.object(order_manager_module, "mt5", fake_mt5):
            om = OrderManager(connector=MagicMock())
            req = om._build_request(types.SimpleNamespace(
                symbol="USDCAD", direction=order_manager_module.OrderDirection.SELL,
                lot_size=0.1, sl_price=1.50002, tp_price=None,
                magic=234_000, comment="test",
            ))

        assert req is not None
        # SELL uses bid=1.50000 as reference price. SL 1.50002 is inside 10-pt cushion.
        # Widened to 1.50000 + 10*0.00001 = 1.50010
        assert round(req["sl"], 5) == pytest.approx(1.50010, abs=1e-9)

    def test_no_widening_when_sl_safely_outside(self):
        """SL already >20 points below ask should pass through unchanged (only rounded)."""
        fake_mt5 = MagicMock()
        fake_mt5.ORDER_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_SELL = 1
        fake_mt5.TRADE_ACTION_DEAL = 1
        fake_mt5.ORDER_TIME_GTC = 0
        fake_mt5.ORDER_FILLING_IOC = 1
        fake_mt5.symbol_info.return_value = self._fake_sym_info(
            digits=5, point=0.00001, stops_level=10,
        )
        fake_mt5.symbol_info_tick.return_value = _fake_tick(bid=1.49990, ask=1.50000)

        with patch.object(order_manager_module, "mt5", fake_mt5):
            om = OrderManager(connector=MagicMock())
            req = om._build_request(types.SimpleNamespace(
                symbol="USDCAD", direction=order_manager_module.OrderDirection.BUY,
                lot_size=0.1, sl_price=1.49000,  # 100 points below ask — plenty of room
                tp_price=None, magic=234_000, comment="test",
            ))

        assert req is not None
        assert round(req["sl"], 5) == pytest.approx(1.49000, abs=1e-9)


# -----------------------------------------------------------------------
# Execution-quality snapshot (E-3 Phase 1)
# -----------------------------------------------------------------------

class TestExecutionQualitySnapshot:
    """OrderResult carries requested/fill/spread snapshot for R-1b logging."""

    def _fake_signal(self, direction: str = "buy", symbol: str = "XAUUSD"):
        sig = types.SimpleNamespace()
        sig.direction = direction
        sig.symbol = symbol
        sig.should_trade = True
        sig.confidence = 0.8
        sig.combined_score = 0.5
        return sig

    def _wire_mt5_with_fill(self, fill_price=2001.5, fill_volume=0.1, retcode=10009):
        fake_mt5 = MagicMock()
        fake_mt5.POSITION_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_BUY = 0
        fake_mt5.ORDER_TYPE_SELL = 1
        fake_mt5.TRADE_ACTION_DEAL = 1
        fake_mt5.ORDER_TIME_GTC = 0
        fake_mt5.ORDER_FILLING_IOC = 1
        fake_mt5.TRADE_RETCODE_DONE = 10009
        fake_mt5.TRADE_RETCODE_CONNECTION = 10031
        fake_mt5.TRADE_RETCODE_TIMEOUT = 10008
        fake_mt5.TRADE_RETCODE_REQUOTE = 10004

        si = types.SimpleNamespace(
            volume_step=0.01, volume_min=0.01, volume_max=100.0,
            digits=2, point=0.01, trade_stops_level=0,
            trade_tick_value=1.0, trade_tick_size=0.01,
            filling_mode=1, currency_quote="USD",
        )
        fake_mt5.symbol_info.return_value = si
        fake_mt5.symbol_info_tick.return_value = _fake_tick(bid=2001.0, ask=2001.3)
        fake_mt5.order_check.return_value = types.SimpleNamespace(retcode=0)
        fake_mt5.order_send.return_value = types.SimpleNamespace(
            retcode=retcode,
            order=12345 if retcode == 10009 else 0,
            price=fill_price,
            volume=fill_volume,
            comment="ok" if retcode == 10009 else "err",
        )
        return fake_mt5

    def test_success_populates_all_snapshot_fields(self):
        """Happy-path fill surfaces requested, fill, slippage inputs, spread."""
        fake_mt5 = self._wire_mt5_with_fill(fill_price=2001.5, fill_volume=0.1)

        with patch.object(order_manager_module, "mt5", fake_mt5):
            om = OrderManager(connector=MagicMock())
            result = om.place_order(
                symbol="XAUUSD",
                signal=self._fake_signal(),
                lot_size=0.1,
                sl_price=1990.0,
            )

        assert result.success is True
        assert result.symbol == "XAUUSD"
        assert result.direction == "buy"
        assert result.requested_price == pytest.approx(2001.3, abs=1e-6)  # ask
        assert result.fill_price == pytest.approx(2001.5, abs=1e-6)
        assert result.spread_at_send == pytest.approx(0.3, abs=1e-6)      # 2001.3 - 2001.0
        assert result.volume_requested == pytest.approx(0.1)
        assert result.volume_filled == pytest.approx(0.1)

    def test_reject_leaves_fill_fields_none(self):
        """Broker reject: requested + spread captured, fill fields None."""
        fake_mt5 = self._wire_mt5_with_fill(retcode=10016)  # Invalid stops

        with patch.object(order_manager_module, "mt5", fake_mt5):
            om = OrderManager(connector=MagicMock())
            result = om.place_order(
                symbol="XAUUSD",
                signal=self._fake_signal(),
                lot_size=0.1,
                sl_price=1990.0,
            )

        assert result.success is False
        assert result.symbol == "XAUUSD"
        assert result.direction == "buy"
        assert result.requested_price == pytest.approx(2001.3, abs=1e-6)
        assert result.spread_at_send == pytest.approx(0.3, abs=1e-6)
        assert result.fill_price is None
        assert result.volume_filled is None
        assert result.volume_requested == pytest.approx(0.1)

    def test_sell_direction_records_bid_as_requested(self):
        """SELL uses bid for price; snapshot reflects that."""
        fake_mt5 = self._wire_mt5_with_fill(fill_price=2000.9, fill_volume=0.1)

        with patch.object(order_manager_module, "mt5", fake_mt5):
            om = OrderManager(connector=MagicMock())
            result = om.place_order(
                symbol="EURUSD",
                signal=self._fake_signal(direction="sell", symbol="EURUSD"),
                lot_size=0.1,
                sl_price=2010.0,  # above price for short
            )

        assert result.success is True
        assert result.direction == "sell"
        assert result.requested_price == pytest.approx(2001.0, abs=1e-6)  # bid
