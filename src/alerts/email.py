"""
email.py — Email Alert Sender (SMTP / Gmail)

Sends alert emails via SMTP. Designed for Gmail with App Passwords
but works with any SMTP server.

Requires ALERT_EMAIL_FROM, ALERT_EMAIL_TO, and ALERT_EMAIL_PASSWORD
in env. Failures are logged but never raised.
"""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

logger = logging.getLogger(__name__)

# Gmail SMTP defaults
_DEFAULT_SMTP_HOST = "smtp.gmail.com"
_DEFAULT_SMTP_PORT = 587
_TIMEOUT = 15


class EmailNotifier:
    """
    Sends alert emails via SMTP with TLS.

    Usage
    -----
        notifier = EmailNotifier()           # reads env vars
        notifier.send("Circuit Breaker Trip", "<b>Daily hard breaker tripped</b>")
    """

    def __init__(
        self,
        email_from: Optional[str] = None,
        email_to: Optional[str] = None,
        password: Optional[str] = None,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
    ):
        self.email_from = email_from or os.getenv("ALERT_EMAIL_FROM", "")
        self.email_to = email_to or os.getenv("ALERT_EMAIL_TO", "")
        self.password = password or os.getenv("ALERT_EMAIL_PASSWORD", "")
        self.smtp_host = smtp_host or os.getenv("ALERT_SMTP_HOST", _DEFAULT_SMTP_HOST)
        self.smtp_port = smtp_port or int(os.getenv("ALERT_SMTP_PORT", str(_DEFAULT_SMTP_PORT)))

        self._enabled = bool(
            self.email_from and self.email_to and self.password
            and self.password != "your_app_password"
        )
        if self._enabled:
            logger.info("EmailNotifier enabled (%s -> %s)", self.email_from, self.email_to)
        else:
            logger.info("EmailNotifier disabled — ALERT_EMAIL_* not fully configured")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def send(self, subject: str, body_html: str) -> bool:
        """
        Send an HTML email.

        Args:
            subject: Email subject line.
            body_html: HTML body content.

        Returns:
            True if the email was sent successfully, False otherwise.
        """
        if not self._enabled:
            return False

        msg = MIMEMultipart("alternative")
        msg["From"] = self.email_from
        msg["To"] = self.email_to
        msg["Subject"] = f"[Cortex] {subject}"

        # Plain-text fallback (strip HTML tags)
        import re
        plain = re.sub(r"<[^>]+>", "", body_html)
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=_TIMEOUT) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.email_from, self.password)
                server.sendmail(self.email_from, self.email_to, msg.as_string())
            logger.info("Email sent: %s", subject)
            return True
        except smtplib.SMTPException as exc:
            logger.warning("Email send failed: %s", exc)
            return False
        except Exception as exc:
            logger.warning("Email connection failed: %s", exc)
            return False

    def test_connection(self) -> bool:
        """Send a test email to verify SMTP credentials."""
        return self.send(
            "Test Connection",
            "<p>Cortex Trading Bot email alerts connected successfully.</p>",
        )
