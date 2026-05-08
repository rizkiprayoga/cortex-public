"""
mt5_connector.py — MetaTrader 5 Connection Manager

Handles initializing, authenticating, and maintaining the connection
to the MetaTrader 5 terminal. All other broker modules depend on this.

Key MT5 functions used:
    mt5.initialize()   — Start MT5 terminal
    mt5.login()        — Authenticate with broker credentials
    mt5.shutdown()     — Close the connection
    mt5.terminal_info()— Verify terminal is reachable
"""

import os
import time
import logging
from typing import Optional

import MetaTrader5 as mt5
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_BACKOFF_CAP_SECONDS = 60.0


def _safe_terminal_info():
    """Call ``mt5.terminal_info()`` swallowing any exception."""
    try:
        return mt5.terminal_info()
    except Exception:
        return None


def _safe_last_error():
    """Call ``mt5.last_error()`` swallowing any exception — for logging only."""
    try:
        return mt5.last_error()
    except Exception:
        return "unknown"


class MT5Connector:
    """Manages the lifecycle of the MetaTrader 5 terminal connection."""

    def __init__(self, max_retries: int = 5, retry_delay: int = 10):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connected = False
        # Credentials from the last successful connect/connect_with_creds.
        # ``reconnect()`` prefers these over env vars so a transient drop
        # after an account switch doesn't silently revert MT5 to MT5_LOGIN.
        # Regression caught 2026-04-18: operator switched to a demo slot,
        # connection flaked, reconnect() reverted to the default account,
        # dashboard kept showing "demo2 active" while the bot was actually
        # on the default login → polluted equity_history with mismatched
        # tags.
        self._last_creds: Optional[dict] = None

    def connect(self) -> bool:
        """
        Initialize MT5 terminal and authenticate.

        Reads MT5_LOGIN, MT5_PASSWORD, MT5_SERVER (and optional MT5_PATH)
        from environment. Retries up to ``max_retries`` times with a
        fixed ``retry_delay`` between attempts. Uses the two-step
        ``mt5.initialize()`` → ``mt5.login()`` flow so that terminal
        attach failures and credential failures can be diagnosed
        separately.

        Returns:
            True on success. Raises RuntimeError after exhausting retries.
        """
        creds = self._read_credentials()
        for attempt in range(1, self.max_retries + 1):
            if self._try_initialize_and_login(creds):
                self._connected = True
                self._last_creds = creds
                info = _safe_terminal_info()
                logger.info(
                    "MT5 connected (build=%s, broker=%s)",
                    getattr(info, "build", "?"),
                    getattr(info, "company", "?"),
                )
                return True

            last_err = _safe_last_error()
            logger.warning(
                "MT5 connect attempt %d/%d failed: %s",
                attempt, self.max_retries, last_err,
            )
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        self._connected = False
        raise RuntimeError(
            f"MT5 connect failed after {self.max_retries} attempts"
        )

    def connect_with_creds(
        self, login: int, password: str, server: str,
    ) -> bool:
        """
        Connect to MT5 with explicit credentials (for account switching).

        Disconnects the current session first, then reconnects with
        the provided credentials. Does NOT change environment variables.

        Returns:
            True on success. Raises RuntimeError on failure.
        """
        self.disconnect()
        creds = {"login": login, "password": password, "server": server}
        for attempt in range(1, self.max_retries + 1):
            if self._try_initialize_and_login(creds):
                self._connected = True
                self._last_creds = creds
                info = _safe_terminal_info()
                logger.info(
                    "MT5 reconnected to account %d (build=%s, broker=%s)",
                    login,
                    getattr(info, "build", "?"),
                    getattr(info, "company", "?"),
                )
                return True

            last_err = _safe_last_error()
            logger.warning(
                "MT5 reconnect attempt %d/%d failed: %s",
                attempt, self.max_retries, last_err,
            )
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        self._connected = False
        raise RuntimeError(
            f"MT5 reconnect to account {login} failed after "
            f"{self.max_retries} attempts"
        )

    def connect_attach_only(self) -> bool:
        """
        Attach to a running MT5 terminal **without** issuing ``mt5.login()``.

        Use this for read-only market-data tooling (OHLCV backfill, symbol
        specs, tick history) that only needs a broker-authenticated session
        — not a specific trading account. The terminal keeps whatever
        account is currently logged in, so running this from a second
        process while the live bot is attached does NOT repoint the
        terminal to ``MT5_LOGIN``.

        Returns:
            True on success. Raises RuntimeError after exhausting retries.
        """
        for attempt in range(1, self.max_retries + 1):
            if self._try_initialize_and_login(creds=None):
                self._connected = True
                # Deliberately leave _last_creds as None — this handle does
                # not own an account, so reconnect() should not try to
                # restore one from it.
                info = _safe_terminal_info()
                acct = mt5.account_info()
                logger.info(
                    "MT5 attached read-only (build=%s, broker=%s, current_login=%s)",
                    getattr(info, "build", "?"),
                    getattr(info, "company", "?"),
                    getattr(acct, "login", "?") if acct else "?",
                )
                return True

            last_err = _safe_last_error()
            logger.warning(
                "MT5 attach attempt %d/%d failed: %s",
                attempt, self.max_retries, last_err,
            )
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        self._connected = False
        raise RuntimeError(
            f"MT5 attach failed after {self.max_retries} attempts"
        )

    def disconnect(self) -> None:
        """Gracefully shut down the MT5 terminal connection."""
        try:
            mt5.shutdown()
        except Exception as exc:  # pragma: no cover
            logger.warning("mt5.shutdown() raised: %s", exc)
        finally:
            self._connected = False
            logger.info("MT5 disconnected")

    def is_connected(self) -> bool:
        """
        Return True if :meth:`connect` / :meth:`reconnect` last succeeded and
        :meth:`disconnect` has not been called since. This is a cheap state
        query — it does *not* probe the terminal. Freshness is enforced at
        the RiskMonitor layer: repeated account-read failures escalate to
        :meth:`reconnect`, which flips this flag back to False on failure.
        """
        return self._connected

    def reconnect(self) -> bool:
        """
        Attempt to restore a dropped connection.

        Uses exponential backoff starting at ``retry_delay`` seconds and
        doubling up to a 60 s cap, for at most ``max_retries`` attempts.
        Unlike :meth:`connect`, this method does **not** raise on failure —
        it returns ``False`` so callers (e.g. :class:`RiskMonitor`) can
        decide how to escalate.

        Returns:
            True on successful reconnect, False after exhausting retries.
        """
        # Release any stale handle before retrying.
        try:
            mt5.shutdown()
        except Exception:
            pass
        self._connected = False

        # Prefer the last successfully-used credentials. Falling back to
        # env vars silently reverted switched accounts — see __init__ docstring.
        creds = self._last_creds or self._read_credentials()
        delay = float(self.retry_delay)

        for attempt in range(1, self.max_retries + 1):
            if self._try_initialize_and_login(creds):
                self._connected = True
                # _last_creds stays pointing at the same account — we
                # just restored the session for it. Only connect() or
                # connect_with_creds() can change the active account.
                logger.info(
                    "MT5 reconnected on attempt %d/%d to login=%s",
                    attempt, self.max_retries,
                    (creds or {}).get("login", "env"),
                )
                return True

            last_err = _safe_last_error()
            logger.warning(
                "MT5 reconnect attempt %d/%d failed: %s",
                attempt, self.max_retries, last_err,
            )
            if attempt < self.max_retries:
                time.sleep(min(delay, _BACKOFF_CAP_SECONDS))
                delay = min(delay * 2.0, _BACKOFF_CAP_SECONDS)

        logger.error(
            "MT5 reconnect gave up after %d attempts — caller must escalate",
            self.max_retries,
        )
        return False

    def get_terminal_info(self) -> dict:
        """Return MT5 terminal metadata (version, build, connected broker)."""
        try:
            info = mt5.terminal_info()
        except Exception as exc:
            logger.error("terminal_info() failed: %s", exc)
            return {}
        if info is None:
            return {}
        try:
            return info._asdict()
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_credentials(self) -> Optional[dict]:
        """
        Parse MT5_LOGIN / MT5_PASSWORD / MT5_SERVER from the environment.

        Returns a dict suitable for ``mt5.login(**creds)``, or ``None`` if
        any field is missing — in which case ``connect()`` will attach to
        whatever credentials the terminal is already holding (dev flow).
        """
        login_raw = os.getenv("MT5_LOGIN", "").strip()
        password = os.getenv("MT5_PASSWORD", "").strip()
        server = os.getenv("MT5_SERVER", "").strip()
        if not (login_raw and password and server):
            return None
        try:
            login_int = int(login_raw)
        except ValueError:
            logger.warning("MT5_LOGIN=%r is not an int — ignoring credentials", login_raw)
            return None
        return {"login": login_int, "password": password, "server": server}

    def _try_initialize_and_login(self, creds: Optional[dict]) -> bool:
        """
        One attempt at the two-step connect flow.

        Returns True only if ``mt5.initialize()`` succeeds AND (when
        credentials are provided) ``mt5.login()`` also succeeds. Any
        raised exception is caught and logged so the retry loop can
        continue cleanly.
        """
        init_kwargs: dict = {}
        path = os.getenv("MT5_PATH")
        if path:
            init_kwargs["path"] = path
        try:
            if not mt5.initialize(**init_kwargs):
                return False
        except Exception as exc:
            logger.error("mt5.initialize() raised: %s", exc)
            return False

        if creds is None:
            return True

        try:
            if not mt5.login(**creds):
                return False
        except Exception as exc:
            logger.error("mt5.login() raised: %s", exc)
            return False

        # Post-login invariant: the MetaTrader5 Python lib can return True
        # from mt5.login() while the terminal session quietly stays on a
        # previous account. Verify account_info().login matches what we
        # asked for; otherwise the bot would silently trade on the wrong
        # account. (Incident: 2026-05-02 prod connected to dev account.)
        try:
            acct = mt5.account_info()
        except Exception as exc:
            logger.error("account_info() after login raised: %s", exc)
            return False
        if acct is None:
            logger.error("mt5.login() returned True but account_info() is None")
            return False
        try:
            actual_login = int(acct.login)
        except (TypeError, ValueError):
            logger.error("account_info().login is not an int: %r", acct.login)
            return False
        if actual_login != int(creds["login"]):
            logger.error(
                "BROKER ACCOUNT MISMATCH after login: requested=%d actual=%d. "
                "Refusing to mark connection healthy.",
                int(creds["login"]), actual_login,
            )
            return False
        return True
