"""
Tests for /api/backtest/* routes — specifically the F3 drill-down detail
endpoint (/runs/{run_id}/detail). The list/submit endpoints already have
indirect coverage via the frontend hooks + integration tests; this file
focuses on the detail payload and error paths.
"""
import pytest
from unittest.mock import AsyncMock


class TestBacktestDetail:
    """GET /api/backtest/runs/{run_id}/detail (Phase F3)."""

    def test_requires_auth(self, client):
        resp = client.get("/api/backtest/runs/test-run-id/detail")
        assert resp.status_code == 401

    def test_returns_full_payload(self, client, auth_headers):
        resp = client.get(
            "/api/backtest/runs/test-run-id/detail",
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()

        # Summary section is the normal BacktestRunSummary
        summary = data["summary"]
        assert summary["id"] == "test-run-id"
        assert summary["symbol"] == "XAUUSD"
        assert summary["status"] == "done"
        assert summary["net_pnl"] == 500.0

        # Equity curve (seeded with 3 points by conftest)
        eq = data["equity_curve"]
        assert isinstance(eq, list)
        assert len(eq) == 3
        assert eq[0]["bar_timestamp"] == "2025-01-01T00:00:00"
        assert eq[0]["equity"] == 10000.0
        assert eq[2]["drawdown_pct"] == pytest.approx(0.2)

        # Trades (seeded with 1 trade by conftest)
        trades = data["trades"]
        assert len(trades) == 1
        t = trades[0]
        assert t["symbol"] == "XAUUSD"
        assert t["direction"] == "buy"
        assert t["pnl"] == 50.0
        assert t["r_multiple"] == pytest.approx(1.5)
        assert t["exit_reason"] == "tp"
        assert t["regime_label"] == "Bull"
        assert t["combined_score"] == pytest.approx(0.72)

    def test_unknown_run_404(self, client, auth_headers, fake_live_state):
        # Override the mock to return None for this call
        fake_live_state.data_store.get_backtest_run = AsyncMock(return_value=None)
        resp = client.get(
            "/api/backtest/runs/does-not-exist/detail",
            headers=auth_headers,
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_empty_equity_and_trades_ok(
        self, client, auth_headers, fake_live_state,
    ):
        """A just-finished run with no equity rows should still return 200."""
        fake_live_state.data_store.get_backtest_equity = AsyncMock(return_value=[])
        fake_live_state.data_store.get_backtest_trades = AsyncMock(return_value=[])
        resp = client.get(
            "/api/backtest/runs/test-run-id/detail",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["equity_curve"] == []
        assert data["trades"] == []
