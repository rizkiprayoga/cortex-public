"""
Shared fixtures for API tests.

Creates a fake LiveState with synthetic data and builds a
FastAPI TestClient around it. No MT5, Postgres, or ML libs needed.

All trading-system dataclasses are replaced with simple SimpleNamespace
objects that have the same attributes, avoiding heavy transitive imports
(hmmlearn, torch, etc.) that require C++ build tools.
"""

import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import build_app
from src.api.live_state import BotControl, DashboardLock, LiveState


# ---------------------------------------------------------------------------
# Env vars for auth (set once for the test session)
# ---------------------------------------------------------------------------

TEST_JWT_SECRET = "test-secret-key-for-jwt-signing-not-for-production"


@pytest.fixture(autouse=True)
def _set_auth_env(monkeypatch):
    """Set auth env vars for every test and clear rate limiter state."""
    monkeypatch.setenv("DASHBOARD_JWT_SECRET", TEST_JWT_SECRET)
    # Audit H5: tests use username "rizki"
    monkeypatch.setenv("DASHBOARD_USERNAME", "rizki")
    import bcrypt
    real_hash = bcrypt.hashpw(b"testpass123", bcrypt.gensalt()).decode("utf-8")
    monkeypatch.setenv("DASHBOARD_PW_HASH", real_hash)

    # Clear rate limiter state so tests don't interfere with each other
    from src.api.routes.auth import _attempts, _consecutive_failures, _blocked_until, _rate_lock
    with _rate_lock:
        _attempts.clear()
        _consecutive_failures.clear()
        _blocked_until.clear()

    # Clear module-level caches so response caching doesn't leak
    # between tests. Each test expects a cold cache (mocks return
    # different values per test). Add new module caches here as they
    # ship.
    from src.api.routes.history import _metrics_cache_clear
    _metrics_cache_clear()


# ---------------------------------------------------------------------------
# Fake data builders (SimpleNamespace instead of real dataclasses)
# ---------------------------------------------------------------------------

def _make_regime_result(symbol: str = "XAUUSD"):
    """Mimics RegimeResult from src.brain.hmm_regime."""
    return SimpleNamespace(
        symbol=symbol,
        regime_index=3,
        regime_label="Bull",
        state_probability=0.87,
        position_multiplier=0.75,
        all_probabilities=np.array([0.01, 0.03, 0.06, 0.87, 0.03]),
        expected_volatility=0.012,
        all_expected_vols=np.array([0.025, 0.018, 0.012, 0.010, 0.019]),
    )


def _make_signal_result(symbol: str = "XAUUSD"):
    """Mimics SignalResult from src.brain.signal_combiner."""
    return SimpleNamespace(
        symbol=symbol,
        should_trade=True,
        direction="buy",
        combined_score=0.73,
        regime=_make_regime_result(symbol),
        lstm_prediction=0.024,
        confidence=0.89,
        bar_timestamp="2026-04-12T14:30:00",
        uncertainty_mode=False,
        size_discount=1.0,
        reasoning=["HMM: Bull 0.87", "LSTM: +0.024", "Flicker: 4/4 stable"],
    )


def _make_open_position(ticket: int = 23451):
    """Mimics OpenPosition from src.strategy.exit_manager."""
    return SimpleNamespace(
        symbol="XAUUSD",
        ticket=ticket,
        direction="buy",
        entry_price=2045.20,
        initial_stop=2030.00,
        current_stop=2038.70,
        volume=0.10,
        initial_volume=0.10,
        atr_trail_mult=2.0,
        strategy_name="MidCautious",
        tier_1_done=True,
        tier_2_done=False,
        max_price=None,
        min_price=None,
        opened_at=datetime(2026, 4, 12, 14, 2, 10, tzinfo=timezone.utc),
        # Per-pair Triple Barrier params surfaced by the API for the
        # dashboard countdown. 80 H1 bars matches XAUUSD's pre-Sprint-3
        # value; either fixture value is fine.
        time_exit_bars=80,
    )


def _make_fake_circuit_breaker():
    """Mimics CircuitBreaker with clean state (no active breakers)."""
    cb = MagicMock()
    cb.current_size_multiplier.return_value = 1.0
    cb.requires_flat.return_value = False
    cb.active_breakers.return_value = []
    cb.is_halted.return_value = False
    # New Phase-2 API: BreakerResponse comes from the last snapshot when
    # available. Return None so the fallback path runs (zeros + "warming up"
    # reason) — clean test state.
    cb.get_last_snapshot.return_value = None
    return cb


def _make_account_snapshot():
    """Mimics AccountSnapshot from src.broker.account_monitor."""
    return SimpleNamespace(
        balance=10000.0,
        equity=10245.0,
        margin=500.0,
        free_margin=9745.0,
        margin_level=2049.0,
        floating_pnl=245.0,
        open_positions=2,
    )


@pytest.fixture
def fake_live_state():
    """A LiveState with all mocked dependencies and synthetic data."""
    # Account monitor
    account_monitor = MagicMock()
    account_monitor.get_info.return_value = _make_account_snapshot()

    # Circuit breaker — use the mock
    circuit_breaker = _make_fake_circuit_breaker()

    # Risk monitor
    risk_monitor = MagicMock()
    risk_monitor.get_peak_equity.return_value = 10280.0

    # Combiner — both legacy `last_signal` (single) and the per-symbol cache
    # so the /api/live/signals/{symbol} handler finds the test signal via
    # the primary code path (last_signal_by_symbol[symbol]).
    combiner = MagicMock()
    combiner.last_signal = _make_signal_result()
    combiner.last_signal_by_symbol = {"XAUUSD": _make_signal_result()}

    # Positions
    tracked_positions = {
        23451: _make_open_position(23451),
    }

    # Order manager, orchestrator, portfolio, data store — all mocks
    order_manager = MagicMock()
    orchestrator = MagicMock()
    portfolio = MagicMock()
    data_store = MagicMock()

    # Async mock defaults for DataStore methods used by new routes
    data_store.create_backtest_run = AsyncMock()
    data_store.get_backtest_run = AsyncMock(return_value={
        "id": "test-run-id", "status": "done", "symbol": "XAUUSD",
        "timeframe": "H4", "start_date": "2025-01-01", "end_date": "2025-06-01",
        "created_at": "2026-04-12T00:00:00", "finished_at": "2026-04-12T00:01:00",
        "total_trades": 10, "win_rate": 0.6, "net_pnl": 500.0,
        "max_drawdown_pct": 3.5, "sharpe_ratio": 1.2, "profit_factor": 1.8,
        "error_message": None,
    })
    data_store.list_backtest_runs = AsyncMock(return_value=[{
        "id": "test-run-id", "status": "done", "symbol": "XAUUSD",
        "timeframe": "H4", "start_date": "2025-01-01", "end_date": "2025-06-01",
        "created_at": "2026-04-12T00:00:00", "finished_at": "2026-04-12T00:01:00",
        "total_trades": 10, "win_rate": 0.6, "net_pnl": 500.0,
        "max_drawdown_pct": 3.5, "sharpe_ratio": 1.2, "profit_factor": 1.8,
    }])

    # OHLCV stub for /api/live/candles/{symbol} (F2 live charts)
    import pandas as _pd
    _dummy_idx = _pd.date_range("2026-04-13 00:00", periods=5, freq="h")
    _dummy_df = _pd.DataFrame({
        "open":   [3200.0, 3205.0, 3210.0, 3208.0, 3212.0],
        "high":   [3206.0, 3212.0, 3214.0, 3214.0, 3216.0],
        "low":    [3198.0, 3203.0, 3206.0, 3205.0, 3210.0],
        "close":  [3205.0, 3210.0, 3208.0, 3212.0, 3214.0],
        "volume": [120.0,  130.0,  118.0,  140.0,  150.0],
    }, index=_dummy_idx)
    data_store.get_ohlcv_range = AsyncMock(return_value=_dummy_df)

    # Backtest detail stubs for /api/backtest/runs/{run_id}/detail (F3)
    data_store.get_backtest_equity = AsyncMock(return_value=[
        {"bar_timestamp": "2025-01-01T00:00:00", "equity": 10000.0, "drawdown_pct": 0.0},
        {"bar_timestamp": "2025-01-02T00:00:00", "equity": 10050.0, "drawdown_pct": 0.0},
        {"bar_timestamp": "2025-01-03T00:00:00", "equity": 10030.0, "drawdown_pct": 0.2},
    ])
    data_store.get_backtest_trades = AsyncMock(return_value=[
        {
            "symbol": "XAUUSD", "direction": "buy",
            "entry_time": "2025-01-02T10:00:00", "exit_time": "2025-01-02T14:00:00",
            "entry_price": 3000.0, "exit_price": 3015.0,
            "pnl": 50.0, "r_multiple": 1.5,
            "exit_reason": "tp", "strategy_name": "LowVolAggressive",
            "regime_label": "Bull", "combined_score": 0.72,
        },
    ])

    return LiveState(
        tracked_positions=tracked_positions,
        combiner=combiner,
        circuit_breaker=circuit_breaker,
        account_monitor=account_monitor,
        risk_monitor=risk_monitor,
        order_manager=order_manager,
        orchestrator=orchestrator,
        portfolio=portfolio,
        data_store=data_store,
        bot_control=BotControl(),
        dashboard_lock=DashboardLock(idle_timeout_seconds=1800),
        current_account_id=123456,  # Set so H9 account filter returns data
    )


@pytest.fixture
def app(fake_live_state):
    """FastAPI app with fake LiveState."""
    return build_app(fake_live_state)


@pytest.fixture
def client(app):
    """TestClient with the dashboard UNLOCKED for convenience."""
    app.state.live_state.dashboard_lock.unlock()
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    """Get a valid JWT token via login and return auth headers."""
    resp = client.post(
        "/api/auth/login",
        json={"username": "rizki", "password": "testpass123"},
    )
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
