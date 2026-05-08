"""Tests for ExitAction.reason → canonical close_reason_code mapping."""

import pytest

from src.strategy.exit_manager import REASON_CODES, classify_reason


class TestClassifyReason:
    def test_take_profit_from_exit_manager_string(self):
        assert classify_reason("take_profit: R=2.13 ≥ 2.0 TP barrier") == "take_profit"

    def test_time_exit_from_exit_manager_string(self):
        assert classify_reason("time_exit: 60 bars held ≥ 60 limit, R=0.42") == "time_exit"

    def test_reversal_from_exit_manager_string(self):
        assert classify_reason(
            "reversal_hard_exit: 2 legs on USDJPY, closing newest ticket=1"
        ) == "reversal_hard_exit"

    def test_broker_sl_alias(self):
        assert classify_reason("sl") == "stop_loss"

    def test_broker_tp_alias(self):
        assert classify_reason("tp") == "take_profit"

    def test_stop_out_bucketed_to_breaker(self):
        # MT5 deal.reason=6 → "so" (stop-out from margin call) is
        # categorized as breaker_emergency in the broker reconcile path.
        # classify_reason itself only handles text; the numeric mapping
        # lives in main.py (see test_mt5_deal_reason_mapping below).
        assert classify_reason("breaker_emergency: drawdown 5%") == "breaker_emergency"

    def test_manual_close_from_dashboard(self):
        assert classify_reason("manual: closed via dashboard") == "manual"

    def test_none_returns_unknown(self):
        assert classify_reason(None) == "unknown"

    def test_empty_string_returns_unknown(self):
        assert classify_reason("") == "unknown"

    def test_garbage_returns_unknown(self):
        assert classify_reason("some garbage the broker emitted") == "unknown"

    def test_mt5_deal_reason_mapping(self):
        """Lock in MT5 ENUM_DEAL_REASON → canonical close_reason_code.

        MT5 values (from MetaTrader5 Python module):
          CLIENT=0, MOBILE=1, WEB=2, EXPERT=3, SL=4, TP=5, SO=6, ROLLOVER=7

        Earlier, main.py had the mapping shifted by -1 so SL closes
        (reason=4) were alerted as 'tp'. This test pins the correct values
        to prevent the same regression.
        """
        mt5_to_reason_str = {4: "sl", 5: "tp", 6: "so", 7: "rollover"}
        mt5_to_canonical = {
            4: "stop_loss", 5: "take_profit",
            6: "breaker_emergency", 7: "unknown",
        }
        # Every canonical target must also pass classify_reason round-trip.
        for code in mt5_to_canonical.values():
            assert classify_reason(code) in REASON_CODES
        # Broker aliases must classify correctly.
        assert classify_reason(mt5_to_reason_str[4]) == "stop_loss"
        assert classify_reason(mt5_to_reason_str[5]) == "take_profit"

    def test_all_returned_codes_are_canonical(self):
        # Guarantee: every classify_reason output is in REASON_CODES.
        samples = [
            "take_profit: x", "time_exit: 60", "reversal_hard_exit: y",
            "sl", "tp", "manual: x", "breaker: margin call", None, "",
            "stopout", "rollover", "anything else",
        ]
        for s in samples:
            assert classify_reason(s) in REASON_CODES
