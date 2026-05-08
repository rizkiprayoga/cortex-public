"""Tests for MT5Connector (mocked MT5 terminal)."""

import pytest
from unittest.mock import patch, MagicMock, call

from src.broker.mt5_connector import MT5Connector


def _acct(login):
    """Return a MagicMock simulating ``mt5.account_info()`` with the given login.

    The post-login invariant in ``_try_initialize_and_login`` reads
    ``acct.login`` so any test that exercises ``connect()`` /
    ``connect_with_creds()`` / ``reconnect()`` must mock ``mt5.account_info``
    to return an object whose ``.login`` matches the requested login —
    otherwise the connector refuses to mark the connection healthy.
    (Regression guard for the 2026-05-02 silent-account-mismatch incident.)
    """
    a = MagicMock()
    a.login = login
    return a


class TestMT5Connector:

    @patch("MetaTrader5.account_info", return_value=_acct(123))
    @patch("MetaTrader5.initialize", return_value=True)
    @patch("MetaTrader5.login", return_value=True)
    @patch.dict("os.environ", {"MT5_LOGIN": "123", "MT5_PASSWORD": "pass", "MT5_SERVER": "Demo"})
    def test_connect_succeeds(self, mock_login, mock_init, mock_acct):
        """connect() should return True when MT5 init and login succeed."""
        connector = MT5Connector()
        result = connector.connect()
        assert result is True
        assert connector.is_connected() is True

    @patch("MetaTrader5.initialize", return_value=False)
    @patch.dict("os.environ", {"MT5_LOGIN": "123", "MT5_PASSWORD": "pass", "MT5_SERVER": "Demo"})
    def test_connect_raises_on_init_failure(self, mock_init):
        """connect() should raise RuntimeError if MT5 fails to initialize."""
        connector = MT5Connector(max_retries=1, retry_delay=0)
        with pytest.raises(RuntimeError):
            connector.connect()

    @patch("MetaTrader5.initialize", return_value=True)
    @patch("MetaTrader5.login", return_value=False)
    @patch.dict("os.environ", {"MT5_LOGIN": "123", "MT5_PASSWORD": "wrong", "MT5_SERVER": "Demo"})
    def test_connect_raises_on_login_failure(self, mock_login, mock_init):
        """connect() should raise RuntimeError if login fails."""
        connector = MT5Connector(max_retries=1, retry_delay=0)
        with pytest.raises(RuntimeError):
            connector.connect()

    @patch("MetaTrader5.account_info", return_value=_acct(123))
    @patch("MetaTrader5.initialize", return_value=True)
    @patch("MetaTrader5.login", return_value=True)
    @patch("MetaTrader5.shutdown")
    @patch.dict("os.environ", {"MT5_LOGIN": "123", "MT5_PASSWORD": "pass", "MT5_SERVER": "Demo"})
    def test_disconnect_calls_shutdown(self, mock_shutdown, mock_login, mock_init, mock_acct):
        """disconnect() should call mt5.shutdown() and set connected=False."""
        connector = MT5Connector()
        connector.connect()
        connector.disconnect()
        mock_shutdown.assert_called_once()
        assert connector.is_connected() is False


class TestMT5ConnectorReconnect:
    """
    Wave 4: verify the exponential-backoff reconnect loop. The live
    risk path depends on this — RiskMonitor escalates to close_all()
    if reconnect returns False after repeated account-read failures.
    """

    @patch("MetaTrader5.account_info", return_value=_acct(123))
    @patch("src.broker.mt5_connector.time.sleep")
    @patch("MetaTrader5.shutdown")
    @patch("MetaTrader5.login", return_value=True)
    @patch.dict("os.environ", {"MT5_LOGIN": "123", "MT5_PASSWORD": "pass", "MT5_SERVER": "Demo"})
    def test_reconnect_succeeds_after_transient_failures(
        self, mock_login, mock_shutdown, mock_sleep, mock_acct,
    ):
        """
        Simulate initialize failing twice then succeeding. reconnect()
        should return True and the sleeps in between should follow the
        doubling backoff schedule (capped at 60s).
        """
        # initialize returns False, False, True — succeeds on attempt 3
        with patch(
            "MetaTrader5.initialize",
            side_effect=[False, False, True],
        ):
            connector = MT5Connector(max_retries=5, retry_delay=2)
            result = connector.reconnect()

        assert result is True
        assert connector.is_connected() is True
        # Two failed attempts → two sleeps. Backoff sequence: 2s, 4s.
        actual_sleeps = [c.args[0] for c in mock_sleep.call_args_list]
        assert actual_sleeps == [2.0, 4.0]

    @patch("src.broker.mt5_connector.time.sleep")
    @patch("MetaTrader5.shutdown")
    @patch("MetaTrader5.initialize", return_value=False)
    @patch.dict("os.environ", {"MT5_LOGIN": "123", "MT5_PASSWORD": "pass", "MT5_SERVER": "Demo"})
    def test_reconnect_returns_false_on_exhaustion(
        self, mock_init, mock_shutdown, mock_sleep
    ):
        """
        If initialize never succeeds, reconnect() must return False
        (not raise) — RiskMonitor branches on the bool to decide
        whether to fire EmergencyClose.
        """
        connector = MT5Connector(max_retries=3, retry_delay=1)
        result = connector.reconnect()
        assert result is False
        assert connector.is_connected() is False

    @patch("src.broker.mt5_connector.time.sleep")
    @patch("MetaTrader5.shutdown")
    @patch("MetaTrader5.initialize", return_value=False)
    @patch.dict("os.environ", {"MT5_LOGIN": "123", "MT5_PASSWORD": "pass", "MT5_SERVER": "Demo"})
    def test_reconnect_backoff_is_capped_at_60s(
        self, mock_init, mock_shutdown, mock_sleep
    ):
        """
        Starting at 40s retry_delay, the backoff should double to 80s
        but get capped at 60s before the next sleep — so the observed
        sleeps are 40s, 60s, 60s, 60s across 5 attempts.
        """
        connector = MT5Connector(max_retries=5, retry_delay=40)
        result = connector.reconnect()
        assert result is False
        actual_sleeps = [c.args[0] for c in mock_sleep.call_args_list]
        # 4 sleeps between 5 attempts. First is the raw retry_delay,
        # subsequent ones are clamped at _BACKOFF_CAP_SECONDS (60s).
        assert actual_sleeps == [40.0, 60.0, 60.0, 60.0]


class TestMT5ConnectorLastCredsSticky:
    """
    Regression guard for the 2026-04-18 silent-account-revert bug:
    after ``connect_with_creds(demo2)``, a subsequent ``reconnect()``
    must log in to demo2, NOT MT5_LOGIN from the environment.
    Without the _last_creds memory, any transient MT5 drop silently
    reverted to the default account while LiveState still said demo2,
    polluting equity_history with mismatched account tags.
    """

    @patch("MetaTrader5.account_info", side_effect=[_acct(111), _acct(222), _acct(222)])
    @patch("MetaTrader5.shutdown")
    @patch("MetaTrader5.initialize", return_value=True)
    @patch("MetaTrader5.login", return_value=True)
    @patch.dict("os.environ", {
        "MT5_LOGIN": "111", "MT5_PASSWORD": "default_pw", "MT5_SERVER": "DefaultSrv",
    })
    def test_reconnect_uses_last_creds_not_env(
        self, mock_login, mock_init, mock_shutdown, mock_acct,
    ):
        connector = MT5Connector(max_retries=1, retry_delay=0)
        connector.connect()                     # sticks creds for login=111
        connector.connect_with_creds(222, "demo2_pw", "Demo2Srv")  # switch

        mock_login.reset_mock()
        result = connector.reconnect()
        assert result is True
        # The reconnect's login call MUST use demo2 creds, not env (111).
        call_kwargs = mock_login.call_args.kwargs
        assert call_kwargs["login"] == 222
        assert call_kwargs["password"] == "demo2_pw"
        assert call_kwargs["server"] == "Demo2Srv"

    @patch("MetaTrader5.account_info", return_value=_acct(111))
    @patch("MetaTrader5.shutdown")
    @patch("MetaTrader5.initialize", return_value=True)
    @patch("MetaTrader5.login", return_value=True)
    @patch.dict("os.environ", {
        "MT5_LOGIN": "111", "MT5_PASSWORD": "default_pw", "MT5_SERVER": "DefaultSrv",
    })
    def test_reconnect_falls_back_to_env_when_never_connected(
        self, mock_login, mock_init, mock_shutdown, mock_acct,
    ):
        # Fresh connector that has never called connect()/connect_with_creds
        # should still be able to reconnect (e.g. during startup races)
        # by reading env vars as the fallback source of truth.
        connector = MT5Connector(max_retries=1, retry_delay=0)
        result = connector.reconnect()
        assert result is True
        call_kwargs = mock_login.call_args.kwargs
        assert call_kwargs["login"] == 111

    @patch("MetaTrader5.account_info", side_effect=[_acct(111), _acct(222)])
    @patch("MetaTrader5.shutdown")
    @patch("MetaTrader5.initialize", return_value=True)
    @patch("MetaTrader5.login", return_value=True)
    @patch.dict("os.environ", {
        "MT5_LOGIN": "111", "MT5_PASSWORD": "default_pw", "MT5_SERVER": "DefaultSrv",
    })
    def test_connect_with_creds_updates_last_creds(
        self, mock_login, mock_init, mock_shutdown, mock_acct,
    ):
        connector = MT5Connector(max_retries=1, retry_delay=0)
        connector.connect()
        assert connector._last_creds == {
            "login": 111, "password": "default_pw", "server": "DefaultSrv",
        }
        connector.connect_with_creds(222, "demo2_pw", "Demo2Srv")
        assert connector._last_creds == {
            "login": 222, "password": "demo2_pw", "server": "Demo2Srv",
        }


class TestMT5ConnectorAccountInfoInvariant:
    """
    Regression guard for the 2026-05-02 silent-account-mismatch incident:
    ``mt5.login()`` returned True while the terminal session quietly stayed
    on a previous account, and the connector logged "MT5 reconnected to
    account X" without checking what account was actually live. The fix
    in ``_try_initialize_and_login`` reads ``mt5.account_info()`` after
    every successful ``mt5.login()`` and refuses to mark the connection
    healthy if ``account_info().login != requested_login``. These tests
    pin that invariant so any future refactor that drops the check fails
    loudly in CI instead of silently in prod.

    Memory: feedback_account_mismatch_silent_login_2026-05-02.md.
    """

    @patch("MetaTrader5.account_info", return_value=_acct(123))
    @patch("MetaTrader5.initialize", return_value=True)
    @patch("MetaTrader5.login", return_value=True)
    @patch.dict("os.environ", {"MT5_LOGIN": "123", "MT5_PASSWORD": "pass", "MT5_SERVER": "Demo"})
    def test_matching_account_info_succeeds(self, mock_login, mock_init, mock_acct):
        """Happy path: account_info().login == requested → connection healthy."""
        connector = MT5Connector(max_retries=1, retry_delay=0)
        assert connector.connect() is True
        assert connector.is_connected() is True

    @patch("MetaTrader5.account_info", return_value=_acct(999))
    @patch("MetaTrader5.initialize", return_value=True)
    @patch("MetaTrader5.login", return_value=True)
    @patch.dict("os.environ", {"MT5_LOGIN": "123", "MT5_PASSWORD": "pass", "MT5_SERVER": "Demo"})
    def test_mismatched_account_info_refuses_connection(
        self, mock_login, mock_init, mock_acct, caplog,
    ):
        """The 2026-05-02 incident path: login claims success but the
        terminal session is actually on a different account. The connector
        MUST NOT mark itself connected — it must raise after retries and
        log ``BROKER ACCOUNT MISMATCH`` so the operator can act."""
        connector = MT5Connector(max_retries=1, retry_delay=0)
        with pytest.raises(RuntimeError):
            connector.connect()
        assert connector.is_connected() is False
        assert any(
            "BROKER ACCOUNT MISMATCH" in r.message for r in caplog.records
        ), "Expected explicit BROKER ACCOUNT MISMATCH log line"

    @patch("MetaTrader5.account_info", return_value=None)
    @patch("MetaTrader5.initialize", return_value=True)
    @patch("MetaTrader5.login", return_value=True)
    @patch.dict("os.environ", {"MT5_LOGIN": "123", "MT5_PASSWORD": "pass", "MT5_SERVER": "Demo"})
    def test_none_account_info_refuses_connection(
        self, mock_login, mock_init, mock_acct,
    ):
        """When account_info() returns None (terminal half-open / race
        during init), refuse the connection rather than trade blind."""
        connector = MT5Connector(max_retries=1, retry_delay=0)
        with pytest.raises(RuntimeError):
            connector.connect()

    @patch("MetaTrader5.account_info", side_effect=RuntimeError("boom"))
    @patch("MetaTrader5.initialize", return_value=True)
    @patch("MetaTrader5.login", return_value=True)
    @patch.dict("os.environ", {"MT5_LOGIN": "123", "MT5_PASSWORD": "pass", "MT5_SERVER": "Demo"})
    def test_account_info_raising_refuses_connection(
        self, mock_login, mock_init, mock_acct,
    ):
        """If account_info() itself raises, treat as failed verification.
        The bot must never trade against a half-initialized terminal."""
        connector = MT5Connector(max_retries=1, retry_delay=0)
        with pytest.raises(RuntimeError):
            connector.connect()

    @patch("MetaTrader5.account_info")
    @patch("MetaTrader5.initialize", return_value=True)
    @patch("MetaTrader5.login", return_value=True)
    @patch.dict("os.environ", {"MT5_LOGIN": "123", "MT5_PASSWORD": "pass", "MT5_SERVER": "Demo"})
    def test_non_int_login_in_account_info_refuses_connection(
        self, mock_login, mock_init, mock_acct,
    ):
        """If account_info().login is non-int (e.g. broker SDK quirk),
        verification is impossible — refuse the connection."""
        bad_acct = MagicMock()
        bad_acct.login = "not-an-int"
        mock_acct.return_value = bad_acct

        connector = MT5Connector(max_retries=1, retry_delay=0)
        with pytest.raises(RuntimeError):
            connector.connect()

    @patch("MetaTrader5.account_info", side_effect=[_acct(111), _acct(999)])
    @patch("MetaTrader5.shutdown")
    @patch("MetaTrader5.initialize", return_value=True)
    @patch("MetaTrader5.login", return_value=True)
    @patch.dict("os.environ", {
        "MT5_LOGIN": "111", "MT5_PASSWORD": "default_pw", "MT5_SERVER": "DefaultSrv",
    })
    def test_connect_with_creds_mismatch_raises(
        self, mock_login, mock_init, mock_shutdown, mock_acct,
    ):
        """Replays the exact 2026-05-02 incident shape: connect() succeeds
        on the default 111 account, then a slot-restore call to 222 returns
        True from mt5.login() but the terminal session stays on 111
        (modeled here by account_info still reporting 111-then-999 — both
        non-222 values that should trip the invariant). connect_with_creds
        MUST raise rather than mark the connection healthy."""
        connector = MT5Connector(max_retries=1, retry_delay=0)
        connector.connect()  # initial OK on 111
        with pytest.raises(RuntimeError):
            connector.connect_with_creds(222, "demo2_pw", "Demo2Srv")
        assert connector.is_connected() is False
