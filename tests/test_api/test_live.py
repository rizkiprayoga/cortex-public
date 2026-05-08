"""
test_live.py — Live state endpoint tests.

Tests:
    - GET /api/live/state returns correct shape
    - GET /api/live/positions returns tracked positions
    - GET /api/live/signals/{symbol} returns signal
    - GET /api/live/breaker returns breaker state
    - All live endpoints require authentication
"""

import pytest


class TestLiveState:
    def test_requires_auth(self, client):
        resp = client.get("/api/live/state")
        assert resp.status_code == 401

    def test_returns_state(self, client, auth_headers):
        resp = client.get("/api/live/state", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "account" in data
        assert "breaker" in data
        assert "peak_equity" in data
        assert "bot_status" in data
        assert "positions_count" in data
        assert "signals" in data

    def test_account_data(self, client, auth_headers):
        resp = client.get("/api/live/state", headers=auth_headers)
        data = resp.json()
        account = data["account"]
        assert account["balance"] == 10000.0
        assert account["equity"] == 10245.0
        assert account["floating_pnl"] == 245.0

    def test_peak_equity(self, client, auth_headers):
        resp = client.get("/api/live/state", headers=auth_headers)
        data = resp.json()
        assert data["peak_equity"] == 10280.0

    def test_bot_status_default_running(self, client, auth_headers):
        resp = client.get("/api/live/state", headers=auth_headers)
        data = resp.json()
        assert data["bot_status"] == "running"

    def test_positions_count(self, client, auth_headers):
        resp = client.get("/api/live/state", headers=auth_headers)
        data = resp.json()
        assert data["positions_count"] == 1

    def test_signal_included(self, client, auth_headers):
        resp = client.get("/api/live/state", headers=auth_headers)
        data = resp.json()
        assert "XAUUSD" in data["signals"]
        sig = data["signals"]["XAUUSD"]
        assert sig["direction"] == "buy"
        assert sig["combined_score"] == pytest.approx(0.73)


class TestPositions:
    def test_requires_auth(self, client):
        resp = client.get("/api/live/positions")
        assert resp.status_code == 401

    def test_returns_positions(self, client, auth_headers):
        resp = client.get("/api/live/positions", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        pos = data[0]
        assert pos["ticket"] == 23451
        assert pos["symbol"] == "XAUUSD"
        assert pos["direction"] == "buy"
        assert pos["entry_price"] == 2045.20
        assert pos["strategy_name"] == "MidCautious"
        assert pos["tier_1_done"] is True
        assert pos["tier_2_done"] is False

    def test_dashboard_extras_present(self, client, auth_headers):
        """Dashboard cards (Variant B) need risk_dollars + time_exit fields.

        risk_dollars is None in unit tests because MetaTrader5 is unavailable
        — the route's order_calc_profit call is wrapped in try/except. The
        test guards the *shape* of the field (key present, nullable) so a
        future schema rename or accidental drop fails loudly here.

        time_exit_bars + time_exit_remaining_sec ARE populated because they
        derive purely from in-process state (OpenPosition.time_exit_bars
        seeded from yaml + ExitManager._h1_bars_elapsed wall-clock math).
        """
        resp = client.get("/api/live/positions", headers=auth_headers)
        assert resp.status_code == 200
        pos = resp.json()[0]
        # Schema contract — keys must exist even when null
        assert "risk_dollars" in pos
        assert "time_exit_bars" in pos
        assert "time_exit_remaining_sec" in pos
        # time_exit_bars echoes the OpenPosition value (80 in the fixture)
        assert pos["time_exit_bars"] == 80
        # Countdown is non-negative when time_exit_bars > 0 and opened_at
        # is set. We don't pin an exact value because it's wall-clock-derived
        # against datetime.now(); just enforce it's an int >= 0 and bounded
        # above by the time_exit window in seconds.
        remaining = pos["time_exit_remaining_sec"]
        assert remaining is None or (
            isinstance(remaining, int) and 0 <= remaining <= 80 * 3600
        )


class TestSignals:
    def test_requires_auth(self, client):
        resp = client.get("/api/live/signals/XAUUSD")
        assert resp.status_code == 401

    def test_returns_signal(self, client, auth_headers):
        resp = client.get("/api/live/signals/XAUUSD", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "XAUUSD"
        assert data["should_trade"] is True
        assert data["direction"] == "buy"
        assert data["confidence"] == pytest.approx(0.89)
        assert data["regime"]["regime_label"] == "Bull"
        assert len(data["regime"]["all_probabilities"]) == 5

    def test_unknown_symbol_returns_warming_up_stub(self, client, auth_headers):
        """After Phase 2 dashboard fix: unknown symbol returns a clean
        warming-up stub at 200 instead of 404, so the Signals detail page
        renders an informative card rather than an error."""
        resp = client.get("/api/live/signals/UNKNOWN", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "UNKNOWN"
        assert data["should_trade"] is False
        assert data["direction"] is None
        assert data["combined_score"] == 0.0
        assert any("warming up" in r.lower() for r in data["reasoning"])


class TestBreaker:
    def test_requires_auth(self, client):
        resp = client.get("/api/live/breaker")
        assert resp.status_code == 401

    def test_returns_clean_breaker(self, client, auth_headers):
        resp = client.get("/api/live/breaker", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["multiplier"] == 1.0
        assert data["requires_flat"] is False
        assert data["active_breakers"] == []


class TestCandles:
    """GET /api/live/candles/{symbol} (Phase F2 — live charts)."""

    def test_requires_auth(self, client):
        resp = client.get("/api/live/candles/XAUUSD")
        assert resp.status_code == 401

    def test_default_returns_bars(self, client, auth_headers):
        resp = client.get("/api/live/candles/XAUUSD", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "XAUUSD"
        assert data["timeframe"] == "H1"
        assert len(data["bars"]) == 5  # conftest seeds 5 bars
        first = data["bars"][0]
        for f in ("time", "open", "high", "low", "close", "volume"):
            assert f in first

    def test_symbol_normalized_upper(self, client, auth_headers):
        resp = client.get("/api/live/candles/xauusd", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["symbol"] == "XAUUSD"

    def test_timeframe_accepts_allowed_values(self, client, auth_headers):
        for tf in ("M15", "H1", "H4", "D1", "W1"):
            resp = client.get(
                f"/api/live/candles/XAUUSD?timeframe={tf}",
                headers=auth_headers,
            )
            assert resp.status_code == 200, tf
            assert resp.json()["timeframe"] == tf

    def test_invalid_timeframe_rejected(self, client, auth_headers):
        resp = client.get(
            "/api/live/candles/XAUUSD?timeframe=M1",
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "Unsupported timeframe" in resp.json()["detail"]

    def test_limit_caps_to_max(self, client, auth_headers):
        resp = client.get(
            "/api/live/candles/XAUUSD?limit=9999",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert len(resp.json()["bars"]) <= 1000

    def test_limit_negative_rejected(self, client, auth_headers):
        resp = client.get(
            "/api/live/candles/XAUUSD?limit=0",
            headers=auth_headers,
        )
        assert resp.status_code == 400
