"""
telegram.py — Telegram Bot Alert Sender

Sends alert messages to a Telegram chat via the Bot API.
Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in env.

Uses the ``requests`` library (already a project dependency) for
synchronous HTTP calls. Messages are sent with MarkdownV2 formatting.
Failures are logged but never raised — the trading loop must never
crash because of a notification failure.
"""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Telegram Bot API base URL
_API_BASE = "https://api.telegram.org/bot{token}"

# Timeout for HTTP requests (seconds)
_TIMEOUT = 10


class TelegramNotifier:
    """
    Sends plain-text messages to a single Telegram chat.

    Usage
    -----
        notifier = TelegramNotifier()        # reads env vars
        notifier.send("Hello from Cortex!")
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self.bot_token and self.chat_id
                             and self.bot_token != "your_telegram_bot_token"
                             and self.chat_id != "your_telegram_chat_id")
        if self._enabled:
            logger.info("TelegramNotifier enabled (chat_id=%s)", self.chat_id)
        else:
            logger.info("TelegramNotifier disabled — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a text message to the configured chat.

        Args:
            text: Message content (supports HTML tags: <b>, <i>, <code>).
            parse_mode: Telegram parse mode (default "HTML").

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        if not self._enabled:
            return False

        url = f"{_API_BASE.format(token=self.bot_token)}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        try:
            resp = requests.post(url, json=payload, timeout=_TIMEOUT)
            if resp.status_code == 200 and resp.json().get("ok"):
                return True
            logger.warning(
                "Telegram sendMessage failed: %d %s",
                resp.status_code,
                resp.text[:200],
            )
            return False
        except requests.RequestException as exc:
            logger.warning("Telegram request failed: %s", exc)
            return False

    def test_connection(self) -> bool:
        """Send a test message to verify the bot is configured correctly."""
        return self.send("Cortex Trading Bot connected.")
