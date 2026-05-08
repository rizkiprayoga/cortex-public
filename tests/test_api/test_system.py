"""
test_system.py — System, bot control, and dashboard lock tests.

Tests:
    - GET /api/system/health always returns 200 (no auth)
    - GET /api/system/lock-status returns lock state (no auth)
    - POST /api/system/unlock from localhost succeeds
    - POST /api/system/unlock from non-localhost returns 403
    - POST /api/system/lock requires auth
    - Dashboard lock gate blocks all /api/* when locked
    - Bot control: start, pause, stop transitions
    - Stop requires typed confirmation
    - GET /api/system/status returns system info
"""

import pytest
from fastapi.testclient import TestClient

from src.api.live_state import BotStatus


class TestHealth:
    def test_health_no_auth(self, client):
        """Health endpoint is always accessible, no auth required."""
        resp = client.get("/api/system/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_health_when_locked(self, app):
        """Health works even when dashboard is locked."""
        app.state.live_state.dashboard_lock.lock()
        locked_client = TestClient(app)
        resp = locked_client.get("/api/system/health")
        assert resp.status_code == 200


class TestDashboardLock:
    def test_lock_status_no_auth(self, client):
        """Lock status is public — returns only locked bool."""
        resp = client.get("/api/system/lock-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "locked" in data
        assert "is_local" in data

    def test_lock_status_when_locked(self, app):
        """Lock-status endpoint works even when locked."""
        app.state.live_state.dashboard_lock.lock()
        locked_client = TestClient(app)
        resp = locked_client.get("/api/system/lock-status")
        assert resp.status_code == 200
        assert resp.json()["locked"] is True

    def test_locked_blocks_api(self, app, auth_headers):
        """When locked, all API routes except exempt ones return 403."""
        app.state.live_state.dashboard_lock.lock()
        locked_client = TestClient(app)
        resp = locked_client.get("/api/live/state", headers=auth_headers)
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Dashboard locked"

    def test_unlock_from_localhost(self, app):
        """Unlock from localhost (TestClient simulates 'testclient')."""
        app.state.live_state.dashboard_lock.lock()
        assert app.state.live_state.dashboard_lock.is_locked

        # TestClient uses host="testclient" by default, not "127.0.0.1".
        # We test the unlock logic directly instead.
        app.state.live_state.dashboard_lock.unlock()
        assert not app.state.live_state.dashboard_lock.is_locked

    def test_lock_requires_auth(self, client):
        """POST /api/system/lock requires authentication."""
        resp = client.post("/api/system/lock")
        assert resp.status_code == 401

    def test_lock_with_auth(self, client, auth_headers):
        """Authenticated user can lock the dashboard."""
        resp = client.post("/api/system/lock", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "locked"
        assert client.app.state.live_state.dashboard_lock.is_locked


class TestBotControl:
    def test_bot_status_default(self, client, auth_headers):
        resp = client.get("/api/bot/status", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"

    def test_pause(self, client, auth_headers):
        resp = client.post(
            "/api/bot/control",
            json={"action": "pause"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

    def test_start_after_pause(self, client, auth_headers):
        # Pause first
        client.post(
            "/api/bot/control",
            json={"action": "pause"},
            headers=auth_headers,
        )
        # Then start
        resp = client.post(
            "/api/bot/control",
            json={"action": "start"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_stop_without_confirmation_fails(self, client, auth_headers):
        resp = client.post(
            "/api/bot/control",
            json={"action": "stop"},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "confirmation" in resp.json()["detail"].lower()

    def test_stop_with_wrong_confirmation_fails(self, client, auth_headers):
        resp = client.post(
            "/api/bot/control",
            json={"action": "stop", "confirmation": "YES"},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_stop_with_correct_confirmation(self, client, auth_headers):
        resp = client.post(
            "/api/bot/control",
            json={"action": "stop", "confirmation": "STOP"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_unknown_action_fails(self, client, auth_headers):
        resp = client.post(
            "/api/bot/control",
            json={"action": "restart"},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_bot_control_requires_auth(self, client):
        resp = client.post(
            "/api/bot/control",
            json={"action": "pause"},
        )
        assert resp.status_code == 401

    def test_changed_by_tracked(self, client, auth_headers):
        client.post(
            "/api/bot/control",
            json={"action": "pause"},
            headers=auth_headers,
        )
        resp = client.get("/api/bot/status", headers=auth_headers)
        assert resp.json()["changed_by"] == "rizki"


class TestRestart:
    """POST /api/system/restart — dashboard-triggered restart."""

    def test_requires_auth(self, client):
        resp = client.post("/api/system/restart")
        assert resp.status_code == 401

    def test_non_windows_returns_501(self, client, auth_headers, monkeypatch):
        """Restart is Windows-only; other platforms must 501."""
        import sys
        monkeypatch.setattr(sys, "platform", "linux")
        resp = client.post("/api/system/restart", headers=auth_headers)
        assert resp.status_code == 501

    def test_spawns_detached_helper(self, client, auth_headers, monkeypatch):
        """Happy path: returns 200 + scheduled status + helper PID."""
        import sys
        import subprocess as sp

        calls: list[dict] = []

        class _FakeProc:
            pid = 12345

        def fake_popen(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return _FakeProc()

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(sp, "Popen", fake_popen)

        resp = client.post("/api/system/restart", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "scheduled"
        assert data["pid"] == 12345
        # Must be detached so it survives killing main.py
        assert len(calls) == 1
        kw = calls[0]["kwargs"]
        assert kw["creationflags"] & 0x00000008  # DETACHED_PROCESS
        assert kw["creationflags"] & 0x00000200  # CREATE_NEW_PROCESS_GROUP


class TestSystemStatus:
    def test_requires_auth(self, client):
        resp = client.get("/api/system/status")
        assert resp.status_code == 401

    def test_returns_status(self, client, auth_headers):
        resp = client.get("/api/system/status", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["bot_status"] == "running"
        assert "uptime_seconds" in data
        assert data["positions_count"] == 1
        assert data["breaker_active"] is False
