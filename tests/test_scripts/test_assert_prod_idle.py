"""Tests for scripts/_assert_prod_idle.py — the dev-side prod-idle guard."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from scripts._assert_prod_idle import (
    _OVERRIDE_ENV_VAR,
    ProdLiveError,
    assert_prod_idle,
)


class TestAssertProdIdle:
    def test_passes_when_no_heartbeat_file(self, tmp_path):
        """If prod isn't installed at this path, no risk — proceed."""
        nonexistent = tmp_path / "no_such_heartbeat.json"
        # Should not raise
        assert_prod_idle(heartbeat_path=nonexistent)

    def test_passes_when_heartbeat_is_stale(self, tmp_path):
        hb = tmp_path / "bot_heartbeat.json"
        hb.write_text('{"timestamp_utc": "2024-01-01T00:00:00+00:00"}')
        # Set mtime to 5 minutes ago
        old = time.time() - 300
        os.utime(hb, (old, old))
        # 120s threshold, mtime is 300s old → stale → proceed
        assert_prod_idle(heartbeat_path=hb, stale_after_seconds=120.0)

    def test_blocks_when_heartbeat_is_fresh(self, tmp_path):
        hb = tmp_path / "bot_heartbeat.json"
        hb.write_text('{"timestamp_utc": "2024-01-01T00:00:00+00:00"}')
        # mtime is "now" (just-written) → fresh → block
        with pytest.raises(ProdLiveError, match="Prod bot heartbeat"):
            assert_prod_idle(heartbeat_path=hb, stale_after_seconds=120.0)

    def test_block_message_mentions_3_recovery_options(self, tmp_path):
        """Operator-friendly: block error must teach how to proceed safely."""
        hb = tmp_path / "bot_heartbeat.json"
        hb.write_text("{}")
        with pytest.raises(ProdLiveError) as exc_info:
            assert_prod_idle(heartbeat_path=hb, stale_after_seconds=120.0)
        msg = str(exc_info.value)
        # Three options surfaced in the message
        assert "Stop prod bot" in msg
        assert "MT5-free" in msg
        assert _OVERRIDE_ENV_VAR in msg

    def test_override_env_var_bypasses_check(self, tmp_path, monkeypatch):
        """Operator escape hatch: explicit env var allows fresh-heartbeat run."""
        hb = tmp_path / "bot_heartbeat.json"
        hb.write_text("{}")
        monkeypatch.setenv(_OVERRIDE_ENV_VAR, "1")
        # Even though heartbeat is fresh, override should let it pass
        assert_prod_idle(heartbeat_path=hb, stale_after_seconds=120.0)

    def test_override_only_accepts_literal_1(self, tmp_path, monkeypatch):
        """Wrong values (true, yes, ...) don't trigger override — strict
        opt-in to avoid accidental shell-quoting bypass."""
        hb = tmp_path / "bot_heartbeat.json"
        hb.write_text("{}")
        for falsy in ("0", "true", "yes", "1 ", " 1"):
            monkeypatch.setenv(_OVERRIDE_ENV_VAR, falsy)
            with pytest.raises(ProdLiveError):
                assert_prod_idle(heartbeat_path=hb, stale_after_seconds=120.0)

    def test_default_threshold_is_two_minutes(self, tmp_path):
        """Default 120s — bot heartbeats every ~60s, so 120s is the right
        threshold to catch a still-running prod."""
        hb = tmp_path / "bot_heartbeat.json"
        hb.write_text("{}")
        # mtime 90s ago — fresher than 120s, should block
        old = time.time() - 90
        os.utime(hb, (old, old))
        with pytest.raises(ProdLiveError):
            assert_prod_idle(heartbeat_path=hb)
