"""
test_auth.py — JWT authentication endpoint tests.

Tests:
    - Login with correct password → JWT returned
    - Login with wrong password → 401
    - Protected route without token → 401
    - Protected route with expired token → 401
    - GET /api/auth/me with valid token → username
    - Rate limiting blocks after too many failures
"""

from datetime import timedelta

import pytest


class TestLogin:
    def test_login_success(self, client):
        resp = client.post(
            "/api/auth/login",
            json={"username": "rizki", "password": "testpass123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client):
        resp = client.post(
            "/api/auth/login",
            json={"username": "rizki", "password": "wrongpass"},
        )
        assert resp.status_code == 401
        assert "Invalid credentials" in resp.json()["detail"]

    def test_login_empty_password(self, client):
        resp = client.post(
            "/api/auth/login",
            json={"username": "rizki", "password": ""},
        )
        assert resp.status_code == 401


class TestProtectedRoutes:
    def test_no_token_returns_401(self, client):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401

    def test_invalid_token_returns_401(self, client):
        resp = client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert resp.status_code == 401

    def test_expired_token_returns_401(self, client):
        from src.api.auth import create_access_token
        token = create_access_token("rizki", expires_delta=timedelta(seconds=-1))
        resp = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401

    def test_valid_token_returns_user(self, client, auth_headers):
        resp = client.get("/api/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["username"] == "rizki"


class TestRateLimit:
    def test_rate_limit_after_many_failures(self, client):
        """After MAX_CONSECUTIVE_FAILURES, the IP should be blocked."""
        # Clear any prior rate-limit state for this test
        from src.api.routes.auth import _attempts, _consecutive_failures, _blocked_until, _rate_lock
        with _rate_lock:
            _attempts.clear()
            _consecutive_failures.clear()
            _blocked_until.clear()

        # Fire 10 bad logins
        for _ in range(10):
            client.post(
                "/api/auth/login",
                json={"username": "rizki", "password": "bad"},
            )

        # 11th should be blocked
        resp = client.post(
            "/api/auth/login",
            json={"username": "rizki", "password": "bad"},
        )
        assert resp.status_code == 429

        # Clean up for other tests
        with _rate_lock:
            _attempts.clear()
            _consecutive_failures.clear()
            _blocked_until.clear()
