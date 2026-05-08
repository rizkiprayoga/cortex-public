"""
Tests for history endpoints (GET /api/history/*).
"""

import pandas as pd
import pytest
from unittest.mock import AsyncMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_data_store(fake_live_state):
    """Wire up async mocks for the DataStore methods used by history routes."""
    ds = fake_live_state.data_store

    # get_trades_paginated
    ds.get_trades_paginated = AsyncMock(return_value=([
        {
            "id": 1,
            "timestamp_open": "2026-04-10T09:00:00",
            "timestamp_close": "2026-04-10T12:00:00",
            "symbol": "XAUUSD",
            "direction": "buy",
            "lot_size": 0.1,
            "entry_price": 2040.0,
            "exit_price": 2060.0,
            "pnl_usd": 200.0,
            "regime_at_entry": "Bull",
            "combined_score": 0.75,
            "ticket": 11111,
        },
        {
            "id": 2,
            "timestamp_open": "2026-04-11T10:00:00",
            "timestamp_close": "2026-04-11T14:00:00",
            "symbol": "XAUUSD",
            "direction": "buy",
            "lot_size": 0.1,
            "entry_price": 2065.0,
            "exit_price": 2050.0,
            "pnl_usd": -150.0,
            "regime_at_entry": "Neutral",
            "combined_score": 0.55,
            "ticket": 11112,
        },
    ], 2))

    # get_equity_history
    df = pd.DataFrame({
        "balance": [10000.0, 10200.0, 10050.0],
        "equity": [10000.0, 10200.0, 10050.0],
        "floating_pnl": [0.0, 200.0, 50.0],
    }, index=pd.to_datetime([
        "2026-04-10T00:00:00",
        "2026-04-11T00:00:00",
        "2026-04-12T00:00:00",
    ]))
    df.index.name = "timestamp"
    ds.get_equity_history = AsyncMock(return_value=df)

    # get_signals_paginated
    ds.get_signals_paginated = AsyncMock(return_value=([
        {
            "id": 1,
            "timestamp": "2026-04-12T14:30:00",
            "symbol": "XAUUSD",
            "regime": "Bull",
            "regime_probability": 0.87,
            "lstm_prediction": 0.024,
            "combined_score": 0.73,
            "should_trade": True,
            "direction": "buy",
        },
    ], 1))

    # get_rolling_metrics
    ds.get_rolling_metrics = AsyncMock(return_value={
        "directional_accuracy": 0.62,
        "mse": 0.0015,
        "mae": 0.028,
        "n_predictions": 100,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTradesEndpoint:
    def test_get_trades_success(self, client, auth_headers):
        resp = client.get("/api/history/trades", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert body["page"] == 1
        assert body["page_size"] == 50
        assert len(body["trades"]) == 2
        assert body["trades"][0]["symbol"] == "XAUUSD"

    def test_get_trades_with_filters(self, client, auth_headers):
        resp = client.get(
            "/api/history/trades?symbol=XAUUSD&page=1&page_size=10",
            headers=auth_headers,
        )
        assert resp.status_code == 200

    def test_get_trades_unauthenticated(self, client):
        resp = client.get("/api/history/trades")
        assert resp.status_code == 401


class TestEquityEndpoint:
    def test_get_equity_success(self, client, auth_headers):
        resp = client.get("/api/history/equity", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert len(body["points"]) == 3
        assert body["points"][0]["equity"] == 10000.0

    def test_get_equity_with_limit(self, client, auth_headers):
        resp = client.get(
            "/api/history/equity?limit=100", headers=auth_headers,
        )
        assert resp.status_code == 200


class TestSignalsEndpoint:
    def test_get_signals_success(self, client, auth_headers):
        resp = client.get("/api/history/signals", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert len(body["signals"]) == 1
        assert body["signals"][0]["direction"] == "buy"


class TestAccuracyEndpoint:
    def test_get_accuracy_success(self, client, auth_headers):
        resp = client.get("/api/history/accuracy", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["symbol"] == "XAUUSD"
        assert body["directional_accuracy"] == 0.62
        assert body["n_predictions"] == 100

    def test_get_accuracy_custom_params(self, client, auth_headers):
        resp = client.get(
            "/api/history/accuracy?symbol=BTCUSD&window=100",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["symbol"] == "BTCUSD"


class TestMetricsEndpoint:
    def test_get_metrics_success(self, client, auth_headers):
        resp = client.get("/api/history/metrics", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_trades"] == 2
        assert body["net_pnl"] == 50.0  # 200 + (-150)
        assert body["win_rate"] == 0.5
        assert body["profit_factor"] == pytest.approx(200.0 / 150.0, rel=1e-3)

    def test_get_metrics_empty(self, client, auth_headers, fake_live_state):
        """Empty trade list returns zeroed metrics."""
        fake_live_state.data_store.get_trades_paginated = AsyncMock(
            return_value=([], 0)
        )
        resp = client.get("/api/history/metrics", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_trades"] == 0
        assert body["net_pnl"] == 0.0

    def test_get_metrics_response_is_cached(
        self, client, auth_headers, fake_live_state,
    ):
        """P-1 stage 2d: metrics response is cached for 30s per
        (account, symbol) key. Second call within TTL must return the
        same response WITHOUT re-hitting the underlying trades query —
        this is what absorbs the repeat calls on History page load
        (~2-3s saved per extra call per the P-1 capture)."""
        call_count = {"n": 0}

        async def _count(*args, **kwargs):
            call_count["n"] += 1
            return ([
                {"pnl_usd": 200.0, "timestamp_close": "2026-04-10T10:00:00+00:00"},
            ], 1)

        fake_live_state.data_store.get_trades_paginated = _count

        r1 = client.get("/api/history/metrics", headers=auth_headers)
        r2 = client.get("/api/history/metrics", headers=auth_headers)
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json() == r2.json()
        # Only ONE round-trip to the trades query — the second served
        # from cache.
        assert call_count["n"] == 1

    def test_get_metrics_cache_keyed_by_symbol(
        self, client, auth_headers, fake_live_state,
    ):
        """Different `symbol` query params must not collide in cache —
        otherwise a /metrics?symbol=XAUUSD could serve
        /metrics?symbol=EURUSD's result."""
        call_count = {"n": 0}

        async def _count(symbol=None, **kwargs):
            call_count["n"] += 1
            # Return a distinctive PnL per symbol so collisions surface.
            pnl = 100.0 if symbol == "XAUUSD" else 200.0
            return ([
                {"pnl_usd": pnl, "timestamp_close": "2026-04-10T10:00:00+00:00"},
            ], 1)

        fake_live_state.data_store.get_trades_paginated = _count

        r_xau = client.get("/api/history/metrics?symbol=XAUUSD", headers=auth_headers)
        r_eur = client.get("/api/history/metrics?symbol=EURUSD", headers=auth_headers)
        assert r_xau.json()["net_pnl"] == 100.0
        assert r_eur.json()["net_pnl"] == 200.0
        # Each distinct symbol key should produce its own DB call.
        assert call_count["n"] == 2

    def test_metrics_cache_expires_after_ttl(
        self, client, auth_headers, fake_live_state, monkeypatch,
    ):
        """After the TTL elapses, the next call must re-hit the DB.
        Without this guarantee a future refactor that accidentally
        disabled expiry would never fail a test. Uses a 0-second TTL
        so expiry is immediate."""
        from src.api.routes import history as history_module
        monkeypatch.setattr(history_module, "_HISTORY_CACHE_TTL_SEC", 0)

        call_count = {"n": 0}

        async def _count(*args, **kwargs):
            call_count["n"] += 1
            return ([
                {"pnl_usd": 100.0, "timestamp_close": "2026-04-10T10:00:00+00:00"},
            ], 1)

        fake_live_state.data_store.get_trades_paginated = _count

        client.get("/api/history/metrics", headers=auth_headers)
        client.get("/api/history/metrics", headers=auth_headers)
        # TTL=0 means every call is expired → every call hits the DB.
        assert call_count["n"] == 2


class TestTradesCache:
    """P-1 stage 2d follow-up: /trades shares the History page-load burst
    with /metrics. Same 30s cache keyed per (account, symbol, since,
    until, page, page_size)."""

    def test_repeat_trades_call_is_cached(
        self, client, auth_headers, fake_live_state,
    ):
        call_count = {"n": 0}

        async def _count(**kwargs):
            call_count["n"] += 1
            return ([], 0)

        fake_live_state.data_store.get_trades_paginated = _count
        client.get("/api/history/trades", headers=auth_headers)
        client.get("/api/history/trades", headers=auth_headers)
        assert call_count["n"] == 1

    def test_trades_cache_keyed_by_page(
        self, client, auth_headers, fake_live_state,
    ):
        """Pagination pages are distinct cache entries — serving page
        2's rows for page 1 would corrupt the UI table."""
        call_count = {"n": 0}

        async def _count(offset=0, **kwargs):
            call_count["n"] += 1
            return ([], offset)  # offset reflects page

        fake_live_state.data_store.get_trades_paginated = _count
        client.get("/api/history/trades?page=1", headers=auth_headers)
        client.get("/api/history/trades?page=2", headers=auth_headers)
        assert call_count["n"] == 2


class TestSignalAudit:
    """GET /api/history/signal-audit — CSV-backed signal reasoning feed.

    Patches the module-level _SIGNAL_AUDIT_CSV constant to point to a
    temp file so the tests don't depend on live bot output.
    """

    CSV_HEADER = (
        "timestamp,symbol,regime,regime_prob,lstm_prediction,combined_score,"
        "direction,should_trade,executed,news_blackout,nearest_cb,"
        "nearest_hours,block_reason,cb_multiplier,reasoning"
    )

    def _seed_csv(self, tmp_path, rows):
        p = tmp_path / "signal_audit.csv"
        p.write_text(
            self.CSV_HEADER + "\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        return p

    def _patch_csv_path(self, monkeypatch, path):
        from src.api.routes import history as route
        monkeypatch.setattr(route, "_SIGNAL_AUDIT_CSV", path)

    def test_requires_auth(self, client):
        resp = client.get("/api/history/signal-audit")
        assert resp.status_code == 401

    def test_missing_file_returns_empty(
        self, client, auth_headers, monkeypatch, tmp_path,
    ):
        self._patch_csv_path(monkeypatch, tmp_path / "nope.csv")
        resp = client.get("/api/history/signal-audit", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"] == []
        assert body["total"] == 0

    def test_newest_first_and_fields_parsed(
        self, client, auth_headers, monkeypatch, tmp_path,
    ):
        self._patch_csv_path(monkeypatch, self._seed_csv(tmp_path, [
            '2026-04-13T10:00:00+00:00,XAUUSD,Bull,0.9,0.6,0.58,buy,True,False,False,,,combiner_rejected,1.0,"fusion"',
            '2026-04-13T10:15:00+00:00,USDJPY,Neutral,0.4,1.0,0.6,buy,True,True,False,,,,1.0,"APPROVED"',
        ]))
        resp = client.get("/api/history/signal-audit", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert body["items"][0]["symbol"] == "USDJPY"
        assert body["items"][0]["executed"] is True
        assert body["items"][0]["combined_score"] == pytest.approx(0.6)
        assert body["items"][1]["symbol"] == "XAUUSD"
        assert body["items"][1]["block_reason"] == "combiner_rejected"

    def test_filter_by_symbol(
        self, client, auth_headers, monkeypatch, tmp_path,
    ):
        self._patch_csv_path(monkeypatch, self._seed_csv(tmp_path, [
            '2026-04-13T10:00:00+00:00,XAUUSD,Bull,0.9,0.6,0.58,buy,True,False,False,,,combiner_rejected,1.0,',
            '2026-04-13T10:15:00+00:00,USDJPY,Neutral,0.4,1.0,0.6,buy,True,True,False,,,,1.0,',
        ]))
        resp = client.get(
            "/api/history/signal-audit?symbol=XAUUSD",
            headers=auth_headers,
        )
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["symbol"] == "XAUUSD"

    def test_filter_by_executed_flag(
        self, client, auth_headers, monkeypatch, tmp_path,
    ):
        self._patch_csv_path(monkeypatch, self._seed_csv(tmp_path, [
            '2026-04-13T10:00:00+00:00,XAUUSD,Bull,0.9,0.6,0.58,buy,True,False,False,,,combiner_rejected,1.0,',
            '2026-04-13T10:15:00+00:00,USDJPY,Neutral,0.4,1.0,0.6,buy,True,True,False,,,,1.0,',
        ]))
        resp = client.get(
            "/api/history/signal-audit?executed=false",
            headers=auth_headers,
        )
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["symbol"] == "XAUUSD"
        resp = client.get(
            "/api/history/signal-audit?executed=true",
            headers=auth_headers,
        )
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["symbol"] == "USDJPY"

    def test_filter_by_block_reason_substring(
        self, client, auth_headers, monkeypatch, tmp_path,
    ):
        self._patch_csv_path(monkeypatch, self._seed_csv(tmp_path, [
            '2026-04-13T10:00:00+00:00,XAUUSD,Bull,0.9,0.6,0.58,buy,True,False,False,,,sizing:lot=0 below volume_min,1.0,',
            '2026-04-13T10:15:00+00:00,USDJPY,Neutral,0.4,1.0,0.6,buy,True,False,False,,,combiner_rejected,1.0,',
        ]))
        resp = client.get(
            "/api/history/signal-audit?block_reason=sizing",
            headers=auth_headers,
        )
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["symbol"] == "XAUUSD"

    def test_pagination(
        self, client, auth_headers, monkeypatch, tmp_path,
    ):
        rows = [
            f'2026-04-13T10:{i:02d}:00+00:00,XAUUSD,Bull,0.9,0.6,0.58,buy,True,False,False,,,x,1.0,'
            for i in range(25)
        ]
        self._patch_csv_path(monkeypatch, self._seed_csv(tmp_path, rows))
        resp = client.get(
            "/api/history/signal-audit?page=1&page_size=10",
            headers=auth_headers,
        )
        body = resp.json()
        assert body["total"] == 25
        assert body["page"] == 1
        assert body["page_size"] == 10
        assert len(body["items"]) == 10
        resp = client.get(
            "/api/history/signal-audit?page=3&page_size=10",
            headers=auth_headers,
        )
        assert len(resp.json()["items"]) == 5
