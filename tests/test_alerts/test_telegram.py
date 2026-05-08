"""Tests for TelegramNotifier."""

import pytest
from unittest.mock import patch, MagicMock

from src.alerts.telegram import TelegramNotifier


class TestTelegramNotifier:
    """Unit tests for TelegramNotifier."""

    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}, clear=False)
    def test_disabled_when_no_token(self):
        notifier = TelegramNotifier(bot_token="", chat_id="123")
        assert not notifier.enabled
        assert notifier.send("hello") is False

    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}, clear=False)
    def test_disabled_when_no_chat_id(self):
        notifier = TelegramNotifier(bot_token="abc:123", chat_id="")
        assert not notifier.enabled

    def test_disabled_when_placeholder_token(self):
        notifier = TelegramNotifier(
            bot_token="your_telegram_bot_token",
            chat_id="your_telegram_chat_id",
        )
        assert not notifier.enabled

    def test_enabled_when_configured(self):
        notifier = TelegramNotifier(bot_token="123:ABC", chat_id="456")
        assert notifier.enabled

    @patch("src.alerts.telegram.requests.post")
    def test_send_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        mock_post.return_value = mock_resp

        notifier = TelegramNotifier(bot_token="123:ABC", chat_id="456")
        result = notifier.send("Test message")

        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["chat_id"] == "456"
        assert call_kwargs[1]["json"]["text"] == "Test message"

    @patch("src.alerts.telegram.requests.post")
    def test_send_api_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        mock_post.return_value = mock_resp

        notifier = TelegramNotifier(bot_token="123:ABC", chat_id="456")
        result = notifier.send("Test message")
        assert result is False

    @patch("src.alerts.telegram.requests.post")
    def test_send_network_error(self, mock_post):
        import requests
        mock_post.side_effect = requests.ConnectionError("no network")

        notifier = TelegramNotifier(bot_token="123:ABC", chat_id="456")
        result = notifier.send("Test message")
        assert result is False

    @patch("src.alerts.telegram.requests.post")
    def test_test_connection(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        mock_post.return_value = mock_resp

        notifier = TelegramNotifier(bot_token="123:ABC", chat_id="456")
        assert notifier.test_connection() is True
