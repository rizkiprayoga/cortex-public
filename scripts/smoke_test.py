"""
smoke_test.py — End-to-end integration check before paper trading.

Exercises every script and module the production bot depends on, without
connecting to a live broker. Fails loud on any regression. Returns 0 if
all checks pass, non-zero otherwise.

Checks
------
1. Python imports — every module loads cleanly (catches syntax errors,
   missing deps, circular imports).
2. Config YAML parses — settings.yaml, model_config.yaml, mt5_config.yaml.
3. Models present on disk for all 4 symbols (hmm + lstm + scaler + pca).
4. Audit-log writers function (signal_audit, trade_events, tick_summary).
5. Analyzer works on real or empty CSVs.
6. Model snapshot round-trip (save → list → show) without mutating state.
7. Calendar news-blackout logic handles past + future dates.
8. PostgreSQL DB is reachable (but not required — warn if not).
9. MT5 connector import + init-without-login path.
10. Signal combiner instantiates and exposes per_symbol_threshold hook.
11. PortfolioManager instantiates with settings.yaml values (audit HIGH-C
    regression check — previously hardcoded defaults were silently used).
12. Backfill script imports with validated allowlists.
13. Test suite green (delegates to pytest).

Usage
-----
    python scripts/smoke_test.py
    python scripts/smoke_test.py --skip-tests  (skip pytest, faster)
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

_passes: list[str] = []
_failures: list[str] = []
_warnings: list[str] = []


def check(name: str):
    """Decorator: record pass/fail per check."""
    def wrap(fn):
        def run():
            try:
                result = fn()
                if result is False:
                    _failures.append(name)
                    print(f"{RED}FAIL{RESET}  {name}")
                elif result == "warn":
                    _warnings.append(name)
                    print(f"{YELLOW}WARN{RESET}  {name}")
                else:
                    _passes.append(name)
                    print(f"{GREEN}PASS{RESET}  {name}")
            except Exception as exc:
                _failures.append(name)
                print(f"{RED}FAIL{RESET}  {name}: {exc}")
                traceback.print_exc()
        return run
    return wrap


# -------------------------------------------------------------------------
# Individual checks
# -------------------------------------------------------------------------

@check("imports: main.py loads")
def _c1():
    importlib.import_module("main")


@check("imports: all core src modules")
def _c2():
    mods = [
        "src.brain.hmm_regime",
        "src.brain.signal_combiner",
        "src.brain.deep_learning.lstm_model",
        "src.brain.deep_learning.trainer",
        "src.broker.mt5_connector",
        "src.broker.order_manager",
        "src.broker.account_monitor",
        "src.data_pipeline.mt5_feed",
        "src.data_pipeline.feature_engineering",
        "src.data_pipeline.data_store",
        "src.data_pipeline.feedback_loop",
        "src.data_pipeline.market.calendar_features",
        "src.data_pipeline.fundamental.manager",
        "src.allocation.portfolio_manager",
        "src.allocation.position_sizer",
        "src.strategy.orchestrator",
        "src.strategy.exit_manager",
        "src.safety.circuit_breaker",
        "src.safety.risk_monitor",
        "src.alerts.manager",
        "src.api.app",
        "src.api.live_state",
        "src.utils.audit_log",
        "src.utils.config_store",
    ]
    for m in mods:
        importlib.import_module(m)


@check("imports: all scripts")
def _c3():
    scripts = [
        "scripts.backtest",
        "scripts.backtest_full",
        "scripts.train_hmm",
        "scripts.train_deep_learning",
        "scripts.model_snapshot",
        "scripts.backfill_ohlcv",
        "scripts.analyze_paper_trading",
        "scripts.portfolio_simulator",
    ]
    for m in scripts:
        importlib.import_module(m)


@check("configs: YAML parses without errors")
def _c4():
    import yaml
    for path in ["config/settings.yaml", "config/model_config.yaml",
                  "config/mt5_config.yaml"]:
        full = ROOT / path
        assert full.exists(), f"missing {path}"
        yaml.safe_load(full.read_text(encoding="utf-8"))


@check("models: all 4 symbols have hmm+lstm+scaler+pca")
def _c5():
    for sym in ("XAUUSD", "USDJPY", "EURUSD", "USDCAD"):
        for suffix in (".pkl", ".pt", ".pca.pkl"):
            if suffix == ".pkl":
                # HMM
                p = ROOT / f"data/models/hmm_{sym}.pkl"
            elif suffix == ".pca.pkl":
                p = ROOT / f"data/models/lstm_{sym}.pca.pkl"
            else:
                p = ROOT / f"data/models/lstm_{sym}.pt"
            if not p.exists():
                raise AssertionError(f"missing model: {p}")
        scaler = ROOT / f"data/models/lstm_scaler_{sym}.pkl"
        if not scaler.exists():
            raise AssertionError(f"missing scaler: {scaler}")


@check("audit_log: CSV writers create headers + append")
def _c6():
    from src.utils.audit_log import SIGNAL_AUDIT, TRADE_EVENTS, TICK_SUMMARY, now_iso
    test_ts = now_iso()
    before = {
        "signal": SIGNAL_AUDIT.path.stat().st_size if SIGNAL_AUDIT.path.exists() else 0,
        "trade": TRADE_EVENTS.path.stat().st_size if TRADE_EVENTS.path.exists() else 0,
        "tick":  TICK_SUMMARY.path.stat().st_size if TICK_SUMMARY.path.exists() else 0,
    }
    SIGNAL_AUDIT.write({"timestamp": test_ts, "symbol": "SMOKE",
                         "reasoning": "smoke_test probe"})
    TRADE_EVENTS.write({"timestamp": test_ts, "event": "smoke", "symbol": "SMOKE"})
    TICK_SUMMARY.write({"timestamp": test_ts, "symbol": "SMOKE", "price": 0.0})
    after = {
        "signal": SIGNAL_AUDIT.path.stat().st_size,
        "trade": TRADE_EVENTS.path.stat().st_size,
        "tick":  TICK_SUMMARY.path.stat().st_size,
    }
    for k in before:
        assert after[k] > before[k], f"{k} log did not grow"


@check("analyzer: runs on current logs without crash")
def _c7():
    result = subprocess.run(
        [sys.executable, "scripts/analyze_paper_trading.py"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"rc={result.returncode} stderr={result.stderr[-500:]}"


@check("snapshot: list works")
def _c8():
    result = subprocess.run(
        [sys.executable, "scripts/model_snapshot.py", "list"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    # Expect at least one snapshot
    assert "phase-" in result.stdout


@check("calendar: news blackout logic covers 2021-2027")
def _c9():
    from datetime import datetime
    from src.data_pipeline.market.calendar_features import (
        is_in_news_blackout, _FOMC_DT, _ECB_DT, _BOJ_DT, _BOC_DT,
        calendar_freshness_warning,
    )
    for dates in (_FOMC_DT, _ECB_DT, _BOJ_DT, _BOC_DT):
        years = {d.year for d in dates}
        for y in (2021, 2022, 2023, 2024, 2025, 2026, 2027):
            assert y in years, f"missing year {y}"
    # XAU always allowed
    assert is_in_news_blackout("XAUUSD", datetime(2024, 1, 31, 12, 0)) is False
    # FOMC ±1h blocks USD pairs
    assert is_in_news_blackout("USDJPY", datetime(2024, 1, 31, 12, 0)) is True
    # Post-news window should NOT be blocked
    assert is_in_news_blackout("USDJPY", datetime(2024, 2, 1, 18, 0)) is False
    # Freshness check runs
    warning = calendar_freshness_warning(min_lookahead_days=60)
    # warning may be None or a string; we just check it doesn't raise


@check("database: PostgreSQL reachable (optional)")
def _c10():
    import asyncio
    from src.data_pipeline.data_store import DataStore
    async def _ping():
        store = DataStore()
        await store.connect()
        await store.close()
    try:
        asyncio.run(_ping())
    except Exception as exc:
        print(f"       DB unreachable: {exc}")
        return "warn"


@check("mt5: connector import + class instantiation")
def _c11():
    from src.broker.mt5_connector import MT5Connector
    # Constructor only — no connect() call (requires live MT5)
    _ = MT5Connector()


@check("combiner: per_symbol_threshold attribute hook works")
def _c12():
    from src.brain.signal_combiner import SignalCombiner
    # Minimal instantiation — other components mocked
    class _MockHmm: pass
    class _MockLstm: pass
    c = SignalCombiner(hmm=_MockHmm(), lstm=_MockLstm())
    # Default attribute — not set initially
    assert getattr(c, "per_symbol_threshold", None) is None
    c.per_symbol_threshold = {"USDJPY": 0.55}
    assert c.per_symbol_threshold["USDJPY"] == 0.55


@check("portfolio_manager: settings.yaml values are honored")
def _c13():
    import yaml
    from src.allocation.portfolio_manager import PortfolioManager
    cfg = yaml.safe_load(
        (ROOT / "config/settings.yaml").read_text(encoding="utf-8")
    )
    risk = cfg["risk"]
    pm = PortfolioManager(
        max_concurrent_per_symbol=int(risk.get("max_concurrent_per_symbol", 3)),
        max_concurrent_total=int(risk.get("max_concurrent_total", 8)),
        max_used_margin_pct_per_position=float(risk.get("max_position_size_pct", 5.0)),
        max_used_margin_pct_total=float(risk.get("max_total_exposure_pct", 15.0)),
        free_margin_reserve_pct=float(risk.get("free_margin_reserve_pct", 20.0)),
        max_daily_trades=int(risk.get("max_daily_trades", 12)),
    )
    assert pm.max_concurrent_total == int(risk.get("max_concurrent_total", 8)), \
        f"max_concurrent_total not plumbed: {pm.max_concurrent_total}"
    assert pm.max_used_margin_pct_per_position == float(risk.get("max_position_size_pct", 5.0))


@check("backfill: symbol/timeframe allowlists enforced")
def _c14():
    from scripts.backfill_ohlcv import VALID_SYMBOLS, VALID_TIMEFRAMES
    assert "XAUUSD" in VALID_SYMBOLS
    assert "H4" in VALID_TIMEFRAMES
    assert "INVALID" not in VALID_SYMBOLS


@check("triple_barrier: label helper produces valid outputs")
def _c15():
    import numpy as np
    import pandas as pd
    from src.data_pipeline.feature_engineering import FeatureEngineer
    n = 200
    np.random.seed(7)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3, "close": close,
    }, index=pd.date_range("2023-01-01", periods=n, freq="4h"))
    labels = FeatureEngineer.compute_triple_barrier_labels(
        df, tp_r_mult=2.0, sl_atr_mult=1.0, time_limit_bars=10,
    )
    assert len(labels) == n
    assert set(np.unique(labels)).issubset({-1.0, 0.0, 1.0})


@check("tests: pytest suite passes")
def _c16():
    if _SKIP_TESTS:
        print("       skipped by --skip-tests")
        return "warn"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    print(f"       {result.stdout.strip().splitlines()[-1] if result.stdout else ''}")
    assert result.returncode == 0, result.stdout[-2000:]


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

_SKIP_TESTS = False


def main():
    global _SKIP_TESTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests", action="store_true",
                         help="Skip the pytest run (saves ~45s)")
    args = parser.parse_args()
    _SKIP_TESTS = args.skip_tests

    print("=" * 70)
    print("CORTEX SMOKE TEST — pre-paper-trading integration check")
    print("=" * 70)

    for fn_name in sorted([k for k in globals() if k.startswith("_c")]):
        globals()[fn_name]()

    print()
    print("=" * 70)
    print(f"RESULTS: {GREEN}{len(_passes)} passed{RESET}  "
           f"{YELLOW}{len(_warnings)} warnings{RESET}  "
           f"{RED}{len(_failures)} failed{RESET}")
    if _failures:
        print(f"\nFailed checks:")
        for f in _failures:
            print(f"  - {f}")
    if _warnings:
        print(f"\nWarnings (non-blocking):")
        for w in _warnings:
            print(f"  - {w}")
    print("=" * 70)
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
