"""
Static scoping check for ``main.py`` scheduler closures.

APScheduler jobs that reference names unbound in their enclosing scope
raise ``NameError`` silently at cron time and are swallowed by the
try/except around each closure's body. The result is a job that logs a
warning but never reaches its ``alert_manager.notify_*()`` call, so
downstream Telegram/email traffic quietly stops.

This happened on 2026-04-18 and 2026-04-19: ``_send_daily_summary``
read bare ``positions_lock`` but only ``live_state.positions_lock`` was
bound in ``main()``. The daily summary dropped silently every night
until discovered in a sanity sweep.

The test walks ``main.py``'s symbol table and asserts that every free
reference inside each scheduler closure resolves to a binding reachable
at call time — locals of the closure, locals of ``async def main()``,
module-level imports/assignments, or Python builtins.

Fails with a readable diff listing which closure and which name is
unbound. Prevents this class of bug from recurring in any future
scheduler closure added to ``main()``.
"""
from __future__ import annotations

import builtins
import symtable
from pathlib import Path

import pytest

MAIN_PY = Path(__file__).parents[2] / "main.py"

SCHEDULER_CLOSURES = (
    "_send_daily_summary",
    "_send_weekly_summary",
    "_monthly_full_retrain",
    "_pf_drift_job",
    "_drift_check_job",
)

# Module-level dunders Python binds implicitly at import time — not in
# dir(builtins) but reachable from any scope in the module.
_MODULE_DUNDERS = frozenset({
    "__name__", "__file__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__cached__", "__annotations__",
})


def _collect_bound(sym_table: symtable.SymbolTable) -> set[str]:
    """Names that will resolve inside this scope at runtime."""
    bound: set[str] = set()
    for sym in sym_table.get_symbols():
        if (
            sym.is_assigned()
            or sym.is_parameter()
            or sym.is_imported()
            or sym.is_local()
        ):
            bound.add(sym.get_name())
    return bound


def _find_child(parent: symtable.SymbolTable, name: str) -> symtable.SymbolTable | None:
    for child in parent.get_children():
        if child.get_name() == name:
            return child
    return None


@pytest.mark.parametrize("closure_name", SCHEDULER_CLOSURES)
def test_scheduler_closure_has_no_unbound_names(closure_name: str) -> None:
    src = MAIN_PY.read_text(encoding="utf-8")
    module_st = symtable.symtable(src, str(MAIN_PY), "exec")

    main_st = _find_child(module_st, "main")
    assert main_st is not None, "async def main() not found in main.py"

    closure_st = _find_child(main_st, closure_name)
    if closure_st is None:
        pytest.skip(f"{closure_name} not present in main.py")

    reachable = (
        _collect_bound(closure_st)
        | _collect_bound(main_st)
        | _collect_bound(module_st)
        | set(dir(builtins))
        | _MODULE_DUNDERS
    )

    # Names the closure references but does not bind itself.
    referenced = {
        s.get_name()
        for s in closure_st.get_symbols()
        if s.is_referenced() and not s.is_local() and not s.is_parameter()
    }
    unresolved = sorted(referenced - reachable)

    assert not unresolved, (
        f"{closure_name} references names bound in no reachable scope "
        f"(closure/main()/module/builtins): {unresolved}. "
        f"APScheduler will swallow the NameError and the job runs silently."
    )
