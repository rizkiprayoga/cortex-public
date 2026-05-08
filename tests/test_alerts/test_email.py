"""Tests for EmailNotifier."""

import pytest
from unittest.mock import patch, MagicMock

from src.alerts.email import EmailNotifier


class TestEmailNotifier:
    """Unit tests for EmailNotifier."""

    @patch.dict("os.environ", {"ALERT_EMAIL_PASSWORD": ""}, clear=False)
    def test_disabled_when_no_password(self):
        notifier = EmailNotifier(
            email_from="a@b.com", email_to="c@d.com", password=""
        )
        assert not notifier.enabled
        assert notifier.send("Subject", "<p>body</p>") is False

    def test_disabled_when_placeholder(self):
        notifier = EmailNotifier(
            email_from="a@b.com",
            email_to="c@d.com",
            password="your_app_password",
        )
        assert not notifier.enabled

    def test_enabled_when_configured(self):
        notifier = EmailNotifier(
            email_from="a@b.com", email_to="c@d.com", password="real_pass"
        )
        assert notifier.enabled

    @patch("src.alerts.email.smtplib.SMTP")
    def test_send_success(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        notifier = EmailNotifier(
            email_from="a@b.com", email_to="c@d.com", password="pass123"
        )
        result = notifier.send("Test Subject", "<p>Hello</p>")

        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("a@b.com", "pass123")
        mock_server.sendmail.assert_called_once()

    @patch("src.alerts.email.smtplib.SMTP")
    def test_send_smtp_error(self, mock_smtp_cls):
        import smtplib
        mock_server = MagicMock()
        mock_server.login.side_effect = smtplib.SMTPAuthenticationError(
            535, b"Bad credentials"
        )
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        notifier = EmailNotifier(
            email_from="a@b.com", email_to="c@d.com", password="wrong"
        )
        result = notifier.send("Test", "<p>body</p>")
        assert result is False

    @patch("src.alerts.email.smtplib.SMTP")
    def test_send_connection_error(self, mock_smtp_cls):
        mock_smtp_cls.side_effect = ConnectionRefusedError("refused")

        notifier = EmailNotifier(
            email_from="a@b.com", email_to="c@d.com", password="pass"
        )
        result = notifier.send("Test", "<p>body</p>")
        assert result is False

    @patch("src.alerts.email.smtplib.SMTP")
    def test_subject_prefix(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        notifier = EmailNotifier(
            email_from="a@b.com", email_to="c@d.com", password="pass"
        )
        notifier.send("My Subject", "<p>body</p>")

        # Check the sent message contains [Cortex] prefix
        call_args = mock_server.sendmail.call_args
        msg_text = call_args[0][2]  # third positional arg is the message string
        assert "[Cortex] My Subject" in msg_text
