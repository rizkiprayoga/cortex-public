"""Tests for AlertManager."""

import os

import pytest
from unittest.mock import MagicMock, patch

from src.alerts.manager import AlertManager


def _make_manager(tg_enabled=True, email_enabled=True):
    """Create an AlertManager with mock notifiers.

    Strips the email-scope env flags so tests see default broadcast
    behavior regardless of what the local .env sets (prevents cross-run
    pollution when the operator has EMAIL_WEEKLY_ONLY=1 or
    EMAIL_DIGEST_ONLY=1 live).
    """
    for k in ("EMAIL_WEEKLY_ONLY", "EMAIL_DIGEST_ONLY"):
        os.environ.pop(k, None)

    tg = MagicMock()
    tg.enabled = tg_enabled
    tg.send.return_value = True
    tg.test_connection.return_value = True

    em = MagicMock()
    em.enabled = email_enabled
    em.send.return_value = True
    em.test_connection.return_value = True

    mgr = AlertManager(telegram=tg, email=em)
    return mgr, tg, em


class TestAlertManager:
    """Unit tests for AlertManager."""

    def test_enabled_both(self):
        mgr, _, _ = _make_manager(True, True)
        assert mgr.enabled is True

    def test_enabled_telegram_only(self):
        mgr, _, _ = _make_manager(True, False)
        assert mgr.enabled is True

    def test_enabled_email_only(self):
        mgr, _, _ = _make_manager(False, True)
        assert mgr.enabled is True

    def test_disabled_none(self):
        mgr, _, _ = _make_manager(False, False)
        assert mgr.enabled is False

    def test_breaker_trip_dispatches_both(self):
        mgr, tg, em = _make_manager()
        mgr.notify_breaker_trip(
            active_breakers=["daily_hard"],
            daily_dd_pct=3.5,
            weekly_dd_pct=1.2,
            peak_dd_pct=0.8,
            equity=9500.0,
        )
        tg.send.assert_called_once()
        em.send.assert_called_once()
        # Telegram message contains key info
        tg_msg = tg.send.call_args[0][0]
        assert "CIRCUIT BREAKER" in tg_msg
        assert "daily_hard" in tg_msg
        assert "3.50%" in tg_msg

    def test_emergency_close_dispatches(self):
        mgr, tg, em = _make_manager()
        mgr.notify_emergency_close(
            closed_tickets=[100, 101],
            failed_tickets=[102],
        )
        tg.send.assert_called_once()
        tg_msg = tg.send.call_args[0][0]
        assert "EMERGENCY CLOSE" in tg_msg
        assert "MANUAL INTERVENTION" in tg_msg

    def test_trade_entry_dispatches(self):
        mgr, tg, em = _make_manager()
        mgr.notify_trade_entry(
            symbol="XAUUSD",
            direction="buy",
            lot_size=0.05,
            entry_price=2350.50,
            stop_loss=2320.00,
            ticket=12345,
            strategy="LowVolAggressive",
        )
        tg.send.assert_called_once()
        tg_msg = tg.send.call_args[0][0]
        assert "NEW TRADE OPENED" in tg_msg
        assert "XAUUSD" in tg_msg
        assert "BUY" in tg_msg

    def test_trade_close_dispatches(self):
        mgr, tg, em = _make_manager()
        mgr.notify_trade_close(
            symbol="BTCUSD",
            direction="buy",
            entry_price=60000,
            exit_price=62000,
            lot_size=0.01,
            pnl=200.0,
            ticket=99999,
            reason="tier_2",
        )
        tg.send.assert_called_once()
        tg_msg = tg.send.call_args[0][0]
        assert "TRADE CLOSED" in tg_msg
        assert "+$200.00" in tg_msg

    def test_trade_close_negative_pnl(self):
        mgr, tg, em = _make_manager()
        mgr.notify_trade_close(
            symbol="XAUUSD",
            direction="sell",
            entry_price=2300,
            exit_price=2350,
            lot_size=0.1,
            pnl=-500.0,
            ticket=11111,
            reason="stop_loss",
        )
        tg_msg = tg.send.call_args[0][0]
        assert "-$500.00" in tg_msg

    def test_daily_summary_dispatches(self):
        mgr, tg, em = _make_manager()
        mgr.notify_daily_summary(
            equity=15000.0,
            daily_pnl=350.0,
            open_positions=2,
            trades_today=5,
            win_rate=0.6,
            breaker_status="clear",
        )
        tg.send.assert_called_once()
        em.send.assert_called_once()
        tg_msg = tg.send.call_args[0][0]
        assert "DAILY SUMMARY" in tg_msg
        assert "$15,000.00" in tg_msg
        assert "+$350.00" in tg_msg
        assert "60.0%" in tg_msg

    def test_system_event_dispatches(self):
        mgr, tg, em = _make_manager()
        mgr.notify_system("Bot Started", "Symbols: XAUUSD, BTCUSD")
        tg.send.assert_called_once()
        tg_msg = tg.send.call_args[0][0]
        assert "Bot Started" in tg_msg

    def test_test_all(self):
        mgr, tg, em = _make_manager()
        results = mgr.test_all()
        assert results == {"telegram": True, "email": True}

    def test_test_all_disabled(self):
        mgr, tg, em = _make_manager(False, False)
        results = mgr.test_all()
        assert results == {"telegram": False, "email": False}

    def test_telegram_exception_does_not_propagate(self):
        """AlertManager must never crash the trading loop."""
        mgr, tg, em = _make_manager()
        tg.send.side_effect = Exception("network dead")

        # Should not raise
        mgr.notify_system("Test")
        tg.send.assert_called_once()
        # Email still gets called
        em.send.assert_called_once()

    def test_email_exception_does_not_propagate(self):
        mgr, tg, em = _make_manager()
        em.send.side_effect = Exception("smtp broken")

        # Should not raise
        mgr.notify_system("Test")
