"""
probe_mt5_symbol_specs.py — pull authoritative MT5 symbol metadata.

Phase 1C tool. Connects to MT5 (using the same env-driven init pattern
as ``MT5Connector``), calls ``mt5.symbol_info()`` for each requested
trading pair, and emits ready-to-paste YAML for ``config/mt5_config.yaml``.

Why: broker-specific specs (digits, point, lot bounds, contract size,
tick value) vary subtly across brokers and across symbol naming. We
trust the broker's own response over hardcoded conventions.

Usage:

    # Probe the 6 Phase 1 expansion pairs against dev MT5
    python -m scripts.probe_mt5_symbol_specs

    # Custom symbol list
    python -m scripts.probe_mt5_symbol_specs --symbols GBPUSD,AUDUSD

    # Print as plain table instead of YAML
    python -m scripts.probe_mt5_symbol_specs --format table

Constraint: only one Python process can hold an MT5 terminal connection
at a time. If the bot is currently running against this MT5 install,
the probe will fail with a clear error suggesting which process to
pause. The probe never trades — read-only access.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


# Phase 1 expansion pairs — keep in sync with project_phase1_status.md.
DEFAULT_SYMBOLS = ["GBPUSD", "AUDUSD", "EURGBP", "EURJPY", "GBPJPY", "AUDNZD"]


def _connect():
    """
    Initialize MT5 the same way ``MT5Connector._try_initialize_and_login``
    does. Returns the ``mt5`` module on success, raises RuntimeError on
    failure with operator-friendly diagnostics.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise RuntimeError(
            "MetaTrader5 package not installed in this venv. "
            "Run: pip install MetaTrader5"
        )

    init_kwargs: dict = {}
    path = os.getenv("MT5_PATH")
    if path:
        init_kwargs["path"] = path

    try:
        ok = mt5.initialize(**init_kwargs)
    except Exception as exc:
        raise RuntimeError(f"mt5.initialize() raised: {exc}")
    if not ok:
        last_err = mt5.last_error()
        raise RuntimeError(
            f"mt5.initialize() returned False. last_error={last_err}. "
            f"This usually means another Python process is already holding "
            f"the connection (e.g. main.py / the trading bot). Check: "
            f"Get-CimInstance Win32_Process | Where-Object "
            f"{{ $_.CommandLine -match 'main\\.py' }}"
        )

    # Optional login — read-only symbol_info works without a logged-in
    # account on most brokers, but some (incl. the broker demo) require it.
    login_raw = os.getenv("MT5_LOGIN", "").strip()
    password  = os.getenv("MT5_PASSWORD", "").strip()
    server    = os.getenv("MT5_SERVER", "").strip()
    if login_raw and password and server:
        try:
            if not mt5.login(login=int(login_raw), password=password, server=server):
                logger.warning(
                    "mt5.login failed (last_error=%s) — symbol_info may still "
                    "work for visible symbols",
                    mt5.last_error(),
                )
        except Exception as exc:
            logger.warning("mt5.login raised: %s — continuing without login", exc)
    return mt5


def _probe(mt5, symbol: str) -> dict | None:
    """
    Call ``symbol_info`` for one symbol. Returns a dict of fields we need
    for ``mt5_config.yaml``, or ``None`` if the symbol is unknown.

    Adds the symbol to Market Watch via ``symbol_select`` first — the broker
    hides crosses by default and ``symbol_info`` returns ``None`` for
    invisible symbols.
    """
    if not mt5.symbol_select(symbol, True):
        logger.warning("Could not add %s to Market Watch", symbol)
        return None
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.warning("symbol_info(%s) returned None — symbol unknown to broker", symbol)
        return None
    tick = mt5.symbol_info_tick(symbol)
    return {
        "symbol":         info.name,
        "digits":         info.digits,
        "point":          info.point,
        "min_lot":        info.volume_min,
        "max_lot":        info.volume_max,
        "lot_step":       info.volume_step,
        "contract_size":  info.trade_contract_size,
        "tick_value":     getattr(info, "trade_tick_value", None),
        "tick_size":      getattr(info, "trade_tick_size", None),
        "spread_points":  info.spread,        # current spread in points
        "current_bid":    tick.bid if tick else None,
        "current_ask":    tick.ask if tick else None,
        "currency_base":  info.currency_base,
        "currency_profit": info.currency_profit,
    }


def _emit_yaml(rows: list[dict]) -> str:
    """Format probe results as YAML mt5_config.yaml symbol blocks."""
    out = []
    for r in rows:
        # Default sl/tp_points: heuristic from the existing entries
        # (forex uses 300/750). Operator can adjust per symbol.
        sl_points = 500 if "JPY" in r["symbol"] else 300
        tp_points = sl_points * 2 + 250  # roughly 2.5R headroom
        out.append(
            f"  {r['symbol']}:\n"
            f"    digits: {r['digits']}\n"
            f"    point: {r['point']}\n"
            f"    min_lot: {r['min_lot']}\n"
            f"    max_lot: {r['max_lot']}\n"
            f"    lot_step: {r['lot_step']}\n"
            f"    sl_points: {sl_points}      # default — adjusted live by ATR\n"
            f"    tp_points: {tp_points}     # ~2.5R\n"
        )
    return "\n".join(out)


def _emit_table(rows: list[dict]) -> str:
    """Format probe results as a fixed-width text table."""
    if not rows:
        return "(no rows)"
    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    sep = "  ".join("-" * widths[h] for h in headers)
    body = "\n".join(
        "  ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers)
        for r in rows
    )
    return f"{line}\n{sep}\n{body}"


def _main(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv
    load_dotenv()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    try:
        mt5 = _connect()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    rows = []
    try:
        for sym in symbols:
            info = _probe(mt5, sym)
            if info:
                rows.append(info)
    finally:
        mt5.shutdown()

    if not rows:
        print("No symbols probed successfully.", file=sys.stderr)
        return 1

    if args.format == "yaml":
        print("# Paste the block below under `symbols:` in config/mt5_config.yaml")
        print(_emit_yaml(rows))
    else:
        print(_emit_table(rows))
    return 0


def main() -> None:
    # Belt-and-braces: refuse to run if prod bot is live, since this script
    # calls mt5.initialize() and would repoint the shared terminal. See
    # memory/feedback_dev_mt5_steals_prod_terminal.md.
    from scripts._assert_prod_idle import assert_prod_idle
    assert_prod_idle()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Probe MT5 for authoritative symbol specs (digits, lot bounds, etc.)",
    )
    parser.add_argument(
        "--symbols", default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated symbols (default: 6 Phase 1 expansion pairs)",
    )
    parser.add_argument(
        "--format", choices=["yaml", "table"], default="yaml",
        help="Output format (default: yaml — ready to paste into mt5_config.yaml)",
    )
    sys.exit(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
