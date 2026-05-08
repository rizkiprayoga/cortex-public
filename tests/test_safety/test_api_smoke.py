"""Tests for src/safety/api_smoke.py route enumeration + failure handling."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.safety import api_smoke
from src.safety.api_smoke import SmokeResult, _enumerate_get_paths


def _fake_route(path: str, methods: set[str]):
    r = MagicMock()
    r.path = path
    r.methods = methods
    return r


def test_enumerate_skips_path_params_and_docs():
    app = MagicMock()
    app.routes = [
        _fake_route("/api/live/state", {"GET"}),
        _fake_route("/api/news/blackouts", {"GET"}),
        _fake_route("/api/history/trades/{ticket}", {"GET"}),   # param → skip
        _fake_route("/docs", {"GET"}),                           # docs → skip
        _fake_route("/api/auth/login", {"POST"}),                # not GET
        _fake_route("/api/system/restart", {"POST", "GET"}),     # skip list
    ]
    paths = _enumerate_get_paths(app)
    assert "/api/live/state" in paths
    assert "/api/news/blackouts" in paths
    assert "/api/history/trades/{ticket}" not in paths
    assert "/docs" not in paths
    assert "/api/auth/login" not in paths
    assert "/api/system/restart" not in paths


def test_run_smoke_fires_invariant_on_500(monkeypatch, tmp_path):
    """A 500 from any route must fire api.route_healthy ALERT."""
    from src.safety import invariants
    # Isolated registry so our assertion doesn't pollute real logs.
    reg = invariants.InvariantRegistry(
        telegram_send=None,
        jsonl_path=tmp_path / "inv.jsonl",
        halt_flag=tmp_path / "HALT.flag",
    )
    monkeypatch.setattr(invariants, "_REGISTRY", reg)

    class _FakeResp:
        def __init__(self, code, text=""):
            self.status_code = code
            self.text = text

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw): return _FakeResp(200, '{"access_token":"x"}')
        async def get(self, path, headers=None):
            return _FakeResp(500, "boom") if "news" in path else _FakeResp(200)

    monkeypatch.setattr(api_smoke.httpx, "AsyncClient", _FakeClient)
    # Force login to return a successful outcome (the fake post returns JSON shape).
    outcome = api_smoke.LoginOutcome(token="tok", configured=True, login_status=200)
    monkeypatch.setattr(api_smoke, "_login", AsyncMock(return_value=outcome))

    results = asyncio.run(api_smoke.run_smoke(
        paths=["/api/live/state", "/api/news/blackouts", "/api/news/events"],
    ))
    assert len(results) == 3
    assert [r.ok for r in results] == [True, False, False]

    recent = reg.recent()
    failing_paths = [
        f.context.get("path") for f in recent
        if f.invariant == "api.route_healthy"
    ]
    assert "/api/news/blackouts" in failing_paths
    assert "/api/news/events" in failing_paths
    assert "/api/live/state" not in failing_paths


def test_sse_path_probed_via_stream_not_get(monkeypatch, tmp_path):
    """SSE routes must be probed with ``client.stream()`` (headers-only),
    never ``client.get()`` (body never ends → hang → job stacking)."""
    from src.safety import invariants
    reg = invariants.InvariantRegistry(
        telegram_send=None,
        jsonl_path=tmp_path / "inv.jsonl",
        halt_flag=tmp_path / "HALT.flag",
    )
    monkeypatch.setattr(invariants, "_REGISTRY", reg)

    stream_called = {"n": 0}
    get_called_on_sse = {"n": 0}

    class _FakeStreamCtx:
        def __init__(self, status):
            self.status_code = status
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class _FakeResp:
        def __init__(self, code, text=""):
            self.status_code = code
            self.text = text

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw): return _FakeResp(200, '{"access_token":"x"}')
        async def get(self, path, headers=None):
            if path in api_smoke.SSE_PATHS:
                get_called_on_sse["n"] += 1
                # Simulate the real hang: never returns.
                await asyncio.sleep(3600)
            return _FakeResp(200)
        def stream(self, method, path, headers=None, timeout=None):
            assert method == "GET"
            stream_called["n"] += 1
            return _FakeStreamCtx(200)

    monkeypatch.setattr(api_smoke.httpx, "AsyncClient", _FakeClient)
    outcome = api_smoke.LoginOutcome(token="tok", configured=True, login_status=200)
    monkeypatch.setattr(api_smoke, "_login", AsyncMock(return_value=outcome))

    results = asyncio.run(api_smoke.run_smoke(
        paths=["/api/live/state", "/api/live/stream"],
    ))
    by_path = {r.path: r for r in results}
    assert by_path["/api/live/stream"].ok is True
    assert by_path["/api/live/stream"].status == 200
    assert stream_called["n"] == 1
    assert get_called_on_sse["n"] == 0


def test_sse_path_fires_invariant_on_5xx_headers(monkeypatch, tmp_path):
    """If the SSE route returns 500 headers, api.route_healthy still fires."""
    from src.safety import invariants
    reg = invariants.InvariantRegistry(
        telegram_send=None,
        jsonl_path=tmp_path / "inv.jsonl",
        halt_flag=tmp_path / "HALT.flag",
    )
    monkeypatch.setattr(invariants, "_REGISTRY", reg)

    class _FakeStreamCtx:
        def __init__(self, status):
            self.status_code = status
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class _FakeResp:
        def __init__(self, code, text=""):
            self.status_code = code
            self.text = text

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw): return _FakeResp(200, '{"access_token":"x"}')
        async def get(self, path, headers=None): return _FakeResp(200)
        def stream(self, method, path, headers=None, timeout=None):
            return _FakeStreamCtx(500)

    monkeypatch.setattr(api_smoke.httpx, "AsyncClient", _FakeClient)
    outcome = api_smoke.LoginOutcome(token="tok", configured=True, login_status=200)
    monkeypatch.setattr(api_smoke, "_login", AsyncMock(return_value=outcome))

    results = asyncio.run(api_smoke.run_smoke(paths=["/api/live/stream"]))
    assert results[0].ok is False
    assert results[0].status == 500
    failing = [f for f in reg.recent() if f.invariant == "api.route_healthy"]
    assert any(f.context.get("path") == "/api/live/stream" for f in failing)


def test_run_smoke_handles_network_failure(monkeypatch, tmp_path):
    """Connection errors become ok=False with error message, don't raise."""
    from src.safety import invariants
    reg = invariants.InvariantRegistry(
        telegram_send=None,
        jsonl_path=tmp_path / "inv.jsonl",
        halt_flag=tmp_path / "HALT.flag",
    )
    monkeypatch.setattr(invariants, "_REGISTRY", reg)

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw):
            raise ConnectionError("refused")
        async def get(self, *a, **kw):
            raise ConnectionError("refused")

    monkeypatch.setattr(api_smoke.httpx, "AsyncClient", _FakeClient)
    results = asyncio.run(api_smoke.run_smoke(paths=["/api/live/state"]))
    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].status == 0
    assert "refused" in (results[0].error or "")
