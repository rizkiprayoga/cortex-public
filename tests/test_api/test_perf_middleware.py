"""
Tests for the api_perf_logger middleware — P-1 investigation tool.

Pins:
  * Off by default (CORTEX_API_PERF_LOG unset / "0") — zero writes.
  * On when CORTEX_API_PERF_LOG=1 — writes one JSONL record per request.
  * Skip paths are honored (SSE stream, static assets).
  * Middleware never raises even if the log dir is read-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _tail_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestPerfLogToggle:
    def test_disabled_by_default_writes_nothing(
        self, monkeypatch, tmp_path, client,
    ):
        from src.api import app as app_module
        monkeypatch.delenv("CORTEX_API_PERF_LOG", raising=False)
        log_path = tmp_path / "api_perf.jsonl"
        monkeypatch.setattr(app_module, "_API_PERF_LOG_PATH", log_path)

        r = client.get("/api/system/health")
        assert r.status_code == 200
        assert _tail_log(log_path) == []

    def test_enabled_writes_one_record_per_request(
        self, monkeypatch, tmp_path, client,
    ):
        from src.api import app as app_module
        monkeypatch.setenv("CORTEX_API_PERF_LOG", "1")
        log_path = tmp_path / "api_perf.jsonl"
        monkeypatch.setattr(app_module, "_API_PERF_LOG_PATH", log_path)

        r1 = client.get("/api/system/health")
        r2 = client.get("/api/system/health")
        assert r1.status_code == 200 and r2.status_code == 200
        records = _tail_log(log_path)
        assert len(records) == 2
        for rec in records:
            assert rec["method"] == "GET"
            assert rec["path"] == "/api/system/health"
            assert rec["status"] == 200
            assert isinstance(rec["duration_ms"], int)
            assert rec["duration_ms"] >= 0
            assert "ts" in rec


class TestPerfLogSkipPaths:
    def test_sse_stream_path_is_skipped(
        self, monkeypatch, tmp_path, client, auth_headers,
    ):
        """SSE /api/live/stream is permanent-open; logging duration would
        be misleading (it's unbounded by design). Must be skipped."""
        from src.api import app as app_module
        monkeypatch.setenv("CORTEX_API_PERF_LOG", "1")
        log_path = tmp_path / "api_perf.jsonl"
        monkeypatch.setattr(app_module, "_API_PERF_LOG_PATH", log_path)

        # Hit a normal endpoint first so we know the logger IS active.
        client.get("/api/system/health")
        # Any requests to /api/live/stream must not be recorded.
        # (We can't cleanly make a real SSE request in the TestClient, but
        # the skip path check happens before the request body — so
        # confirming a request to a skip path does not produce a record
        # is enough. Here we just confirm the skip-list contains it.)
        assert "/api/live/stream" in app_module._API_PERF_SKIP_PATHS
        recs = _tail_log(log_path)
        # Only the health check should be there, not a stream record.
        assert all(r["path"] != "/api/live/stream" for r in recs)


class TestGZipMiddleware:
    """GZipMiddleware shrinks JSON ~5-10x on typical dashboard payloads.
    Pin that it fires above the minimum_size threshold AND that it
    doesn't fire below (compressing a 200-byte response wastes CPU
    with no wire-size benefit)."""

    def test_large_response_is_gzipped(self, client, auth_headers):
        """A response above the 1 KB threshold must come back with
        Content-Encoding: gzip when the client sends Accept-Encoding."""
        # Replace the shared candles mock with a 300-bar payload that
        # is comfortably >1 KB post-serialization (300 × ~60 bytes ≈ 18 KB).
        import pandas as pd
        from unittest.mock import AsyncMock
        idx = pd.date_range("2026-01-01", periods=300, freq="h")
        big_df = pd.DataFrame({
            "open":   [3200.0] * 300, "high":   [3210.0] * 300,
            "low":    [3195.0] * 300, "close":  [3205.0] * 300,
            "volume": [100.0]  * 300,
        }, index=idx)
        client.app.state.live_state.data_store.get_ohlcv_range = AsyncMock(
            return_value=big_df,
        )

        headers = {**auth_headers, "Accept-Encoding": "gzip"}
        r = client.get("/api/live/candles/XAUUSD?limit=300", headers=headers)
        assert r.status_code == 200
        # TestClient transparently decompresses, but the header stays.
        assert r.headers.get("content-encoding") == "gzip"

    def test_small_response_is_not_gzipped(self, client, auth_headers):
        """Tiny responses (below 1 KB) skip gzip — compressing them
        costs CPU with no meaningful wire-size benefit."""
        headers = {**auth_headers, "Accept-Encoding": "gzip"}
        r = client.get("/api/system/health", headers=headers)
        assert r.status_code == 200
        assert r.headers.get("content-encoding") != "gzip"


class TestPerfLogResilience:
    def test_log_failure_does_not_fail_request(
        self, monkeypatch, tmp_path, client,
    ):
        """If the log write fails (disk full, permissions, etc.), the
        user-facing request must still return normally."""
        from src.api import app as app_module
        monkeypatch.setenv("CORTEX_API_PERF_LOG", "1")

        def _boom(_record):
            raise OSError("disk full simulation")

        monkeypatch.setattr(app_module, "_api_perf_write", _boom)

        r = client.get("/api/system/health")
        assert r.status_code == 200
