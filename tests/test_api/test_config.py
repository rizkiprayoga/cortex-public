"""
Tests for config endpoints (GET/POST /api/config/risk).
"""

import pytest
from unittest.mock import MagicMock

from src.api.routes.config import CONFIRM_TOKEN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _setup_portfolio_attrs(fake_live_state):
    """Give the portfolio mock the attributes the config route reads."""
    pm = fake_live_state.portfolio
    pm.max_used_margin_pct_total = 15.0
    pm.free_margin_reserve_pct = 20.0
    pm.max_concurrent_per_symbol = 3
    pm.max_concurrent_total = 6
    pm.max_daily_trades = 12

    # Setter mocks
    pm.set_max_daily_trades = MagicMock()
    pm.set_max_concurrent_per_symbol = MagicMock()
    pm.set_max_concurrent_total = MagicMock()
    pm.set_max_used_margin_pct_total = MagicMock()
    pm.set_free_margin_reserve_pct = MagicMock()


@pytest.fixture(autouse=True)
def _setup_cb_attrs(fake_live_state):
    """Give the circuit_breaker mock the real threshold attributes."""
    cb = fake_live_state.circuit_breaker
    cb.max_daily_loss_soft_pct = 2.0
    cb.max_daily_loss_hard_pct = 3.0
    cb.max_weekly_loss_soft_pct = 5.0
    cb.max_weekly_loss_hard_pct = 7.0
    cb.max_peak_drawdown_pct = 10.0

    # Setter mocks
    cb.set_daily_soft = MagicMock()
    cb.set_daily_hard = MagicMock()
    cb.set_weekly_soft = MagicMock()
    cb.set_weekly_hard = MagicMock()
    cb.set_peak = MagicMock()


# ---------------------------------------------------------------------------
# GET /api/config/risk
# ---------------------------------------------------------------------------

class TestGetRiskConfig:
    def test_get_risk_config(self, client, auth_headers):
        resp = client.get("/api/config/risk", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["max_daily_loss_soft_pct"] == 2.0
        assert body["max_peak_drawdown_pct"] == 10.0
        assert body["max_daily_trades"] == 12

    def test_get_risk_config_unauthenticated(self, client):
        resp = client.get("/api/config/risk")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/config/risk
# ---------------------------------------------------------------------------

class TestUpdateRiskConfig:
    def test_update_soft_breaker(self, client, auth_headers, fake_live_state):
        """Soft breaker changes don't require confirmation."""
        resp = client.post(
            "/api/config/risk",
            json={"max_daily_loss_soft_pct": 2.5},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        fake_live_state.circuit_breaker.set_daily_soft.assert_called_once_with(2.5)

    def test_update_hard_breaker_without_confirmation(self, client, auth_headers):
        """Hard-halt changes without confirmation → 400."""
        resp = client.post(
            "/api/config/risk",
            json={"max_daily_loss_hard_pct": 4.0},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "confirmation" in resp.json()["detail"].lower()

    def test_update_hard_breaker_with_confirmation(
        self, client, auth_headers, fake_live_state,
    ):
        """Hard-halt changes with correct confirmation → 200."""
        resp = client.post(
            "/api/config/risk",
            json={
                "max_daily_loss_hard_pct": 4.0,
                "confirmation": CONFIRM_TOKEN,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        fake_live_state.circuit_breaker.set_daily_hard.assert_called_once_with(4.0)

    def test_update_peak_requires_confirmation(self, client, auth_headers):
        """Peak drawdown is a hard-halt knob."""
        resp = client.post(
            "/api/config/risk",
            json={"max_peak_drawdown_pct": 12.0},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_update_portfolio_params(self, client, auth_headers, fake_live_state):
        """Portfolio allocation params update without confirmation."""
        resp = client.post(
            "/api/config/risk",
            json={"max_daily_trades": 15, "max_concurrent_total": 8},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        fake_live_state.portfolio.set_max_daily_trades.assert_called_once_with(15)
        fake_live_state.portfolio.set_max_concurrent_total.assert_called_once_with(8)

    def test_update_empty_body(self, client, auth_headers):
        """No fields to update → 400."""
        resp = client.post(
            "/api/config/risk",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "no fields" in resp.json()["detail"].lower()

    def test_update_invalid_value(self, client, auth_headers, fake_live_state):
        """Setter raises ValueError → 400."""
        fake_live_state.circuit_breaker.set_daily_soft.side_effect = ValueError(
            "daily_soft must be 0 < pct < 100, got -1"
        )
        resp = client.post(
            "/api/config/risk",
            json={"max_daily_loss_soft_pct": -1},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_config_store_persistence(self, client, auth_headers, fake_live_state):
        """ConfigStore.write_risk_section is called when config_store is set."""
        mock_store = MagicMock()
        mock_store.read_risk_section.return_value = {"max_daily_loss_soft_pct": 2.0}
        fake_live_state.config_store = mock_store

        resp = client.post(
            "/api/config/risk",
            json={"max_daily_loss_soft_pct": 2.5},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        mock_store.write_risk_section.assert_called_once()
