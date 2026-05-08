"""
Root conftest for the Cortex test suite.

Primary job: isolate the invariants registry from production telemetry.

Code under test (e.g. ExitManager, OrderManager) calls the module-level
``src.safety.invariants.check()`` which resolves a process-global
``_REGISTRY``. Its default ``JSONL_PATH`` is ``data/logs/invariants.jsonl``
— the live operator log. Without this fixture every pytest run writes
violations to the prod file (and, if any invariant is promoted to ALERT
or CRITICAL, could fire real Telegrams or drop TRADING_HALTED.flag).

The autouse fixture below rebinds ``_REGISTRY`` to a per-test tmp path
before each test and restores the prior instance after. Combined with
the defensive guard in ``InvariantRegistry.__init__`` (which refuses the
prod path when ``PYTEST_CURRENT_TEST`` is set), prod telemetry is safe
by default.
"""

from __future__ import annotations

import pytest

from src.safety import invariants as _invariants


@pytest.fixture(autouse=True)
def _isolate_invariant_registry(tmp_path, monkeypatch):
    jsonl = tmp_path / "invariants.jsonl"
    halt = tmp_path / "TRADING_HALTED.flag"
    test_registry = _invariants.InvariantRegistry(
        telegram_send=None,
        jsonl_path=jsonl,
        halt_flag=halt,
    )
    monkeypatch.setattr(_invariants, "_REGISTRY", test_registry)
    yield test_registry
