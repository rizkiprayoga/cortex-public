"""
analyze_paper_trading.py — Post-session health check for paper-trading logs.

Reads the three CSV audit streams plus the DB tables and produces a report
covering:

  1. Session stats  — uptime, tick count, symbols observed
  2. Signal funnel  — signals generated vs rejected (and why)
  3. Trade outcomes — count, WR, PF, per-symbol P/L, avg holding time
  4. Equity curve   — balance/equity/drawdown summary
  5. Red flags      — non-obvious problems (stuck regime, no trades for X days,
                       LSTM drift, repeated broker rejects, missing schedules)

Usage
-----
    # Analyze all data in data/logs/
    python scripts/analyze_paper_trading.py

    # Limit to the last N days
    python scripts/analyze_paper_trading.py --since 7

    # Focus on a specific symbol
    python scripts/analyze_paper_trading.py --symbol XAUUSD

The output is human-readable (prints to stdout) and designed for quick
operator glance or Claude-assisted diagnosis. A JSON dump is also saved
to data/logs/analysis_latest.json for programmatic consumption.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "data" / "logs"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return df
    except Exception as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return pd.DataFrame()


def _filter_since(df: pd.DataFrame, since_dt: datetime | None) -> pd.DataFrame:
    if df.empty or since_dt is None or "timestamp" not in df.columns:
        return df
    return df[df["timestamp"] >= since_dt]


# =========================================================================
# Report sections
# =========================================================================

def section_session(signals: pd.DataFrame, ticks: pd.DataFrame) -> dict:
    out = {"section": "session", "warnings": []}
    if signals.empty and ticks.empty:
        out["warnings"].append("NO DATA — no signal or tick rows found in CSVs.")
        return out

    df = ticks if not ticks.empty else signals
    out["first_event"] = str(df["timestamp"].min())
    out["last_event"] = str(df["timestamp"].max())
    span = df["timestamp"].max() - df["timestamp"].min()
    out["span_hours"] = float(span.total_seconds() / 3600)
    out["tick_rows"] = int(len(ticks))
    out["signal_rows"] = int(len(signals))
    out["symbols_seen"] = sorted(df["symbol"].dropna().unique().tolist()) if "symbol" in df.columns else []

    # Heartbeat check — expect 1 tick_summary per symbol per 15min
    expected_per_sym_per_hour = 4
    expected_total = max(1, int(out["span_hours"])) * expected_per_sym_per_hour * max(1, len(out["symbols_seen"]))
    if not ticks.empty and len(ticks) < expected_total * 0.5:
        out["warnings"].append(
            f"Tick heartbeat sparse: {len(ticks)} rows vs expected ~{expected_total}. "
            f"Bot may have been stopped/paused during the session."
        )
    return out


def section_signal_funnel(signals: pd.DataFrame, symbol_filter: str | None) -> dict:
    out = {"section": "signal_funnel", "warnings": []}
    if signals.empty:
        out["warnings"].append("No signal rows — bot may not have generated any signals.")
        return out

    df = signals if not symbol_filter else signals[signals["symbol"] == symbol_filter]
    if df.empty:
        out["warnings"].append(f"No signals for symbol {symbol_filter}")
        return out

    total = len(df)
    combiner_pass = int((df.get("should_trade", pd.Series([False] * len(df))) == True).sum())
    executed = int((df.get("executed", pd.Series([False] * len(df))) == True).sum())
    blocked = total - executed

    reason_counts = {}
    if "block_reason" in df.columns:
        reasons = df["block_reason"].fillna("")
        reasons = reasons[reasons != ""].value_counts().head(15)
        reason_counts = {str(k): int(v) for k, v in reasons.items()}

    per_symbol = {}
    if "symbol" in df.columns:
        for sym in df["symbol"].dropna().unique():
            sub = df[df["symbol"] == sym]
            per_symbol[str(sym)] = {
                "total":    int(len(sub)),
                "combiner_pass": int((sub.get("should_trade", False) == True).sum()),
                "executed": int((sub.get("executed", False) == True).sum()),
            }

    out["total_signals"] = total
    out["combiner_pass"] = combiner_pass
    out["executed"] = executed
    out["block_rate"] = round(blocked / total, 3) if total else 0.0
    out["top_block_reasons"] = reason_counts
    out["per_symbol"] = per_symbol

    # Red flag: nothing executed
    if executed == 0 and total > 10:
        out["warnings"].append(
            "ZERO EXECUTIONS across all signals — check block_reason column."
        )
    # Red flag: combiner passes but always blocked
    if combiner_pass > 10 and executed / max(combiner_pass, 1) < 0.3:
        out["warnings"].append(
            f"Combiner passed {combiner_pass} times but only {executed} executed. "
            f"Downstream gates (sizing, news, broker) rejecting most."
        )
    return out


def section_trade_outcomes(events: pd.DataFrame, symbol_filter: str | None) -> dict:
    out = {"section": "trade_outcomes", "warnings": []}
    if events.empty:
        out["warnings"].append("No trade_events rows — no trades opened or closed.")
        return out

    df = events if not symbol_filter else events[events["symbol"] == symbol_filter]
    entries = df[df["event"] == "entry"] if "event" in df.columns else pd.DataFrame()
    exits = df[df["event"] == "exit"] if "event" in df.columns else pd.DataFrame()

    out["entries"] = int(len(entries))
    out["exits"] = int(len(exits))
    out["open_positions"] = out["entries"] - out["exits"]

    # Per-symbol PnL summary (from exits)
    per_symbol = {}
    if not exits.empty and "pnl_usd" in exits.columns:
        exits = exits.copy()
        exits["pnl_usd"] = pd.to_numeric(exits["pnl_usd"], errors="coerce").fillna(0.0)
        for sym in exits["symbol"].dropna().unique():
            sub = exits[exits["symbol"] == sym]
            wins = sub[sub["pnl_usd"] > 0]
            losses = sub[sub["pnl_usd"] < 0]
            gross_win = float(wins["pnl_usd"].sum())
            gross_loss = abs(float(losses["pnl_usd"].sum()))
            pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
            per_symbol[str(sym)] = {
                "exits":      int(len(sub)),
                "net_pnl":    round(float(sub["pnl_usd"].sum()), 2),
                "win_rate":   round(len(wins) / max(len(sub), 1), 3),
                "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
                "avg_win":    round(float(wins["pnl_usd"].mean()), 2) if len(wins) else 0.0,
                "avg_loss":   round(float(losses["pnl_usd"].mean()), 2) if len(losses) else 0.0,
            }

        overall_win = float(exits[exits["pnl_usd"] > 0]["pnl_usd"].sum())
        overall_loss = abs(float(exits[exits["pnl_usd"] < 0]["pnl_usd"].sum()))
        out["net_pnl"] = round(float(exits["pnl_usd"].sum()), 2)
        out["overall_pf"] = round(overall_win / overall_loss, 2) if overall_loss else "inf"

    # Exit reason distribution
    if "exit_reason" in exits.columns:
        reasons = exits["exit_reason"].fillna("").value_counts().head(10)
        out["exit_reasons"] = {str(k): int(v) for k, v in reasons.items()}

    out["per_symbol"] = per_symbol
    return out


def section_equity(ticks: pd.DataFrame) -> dict:
    out = {"section": "equity", "warnings": []}
    if ticks.empty or "equity" not in ticks.columns:
        return out
    eq = pd.to_numeric(ticks["equity"], errors="coerce").dropna()
    if eq.empty:
        return out
    running_peak = eq.cummax()
    dd_pct = (eq - running_peak) / running_peak * 100
    out["start_equity"] = float(eq.iloc[0])
    out["end_equity"] = float(eq.iloc[-1])
    out["peak_equity"] = float(running_peak.max())
    out["max_drawdown_pct"] = round(float(abs(dd_pct.min())), 2)
    out["net_change_pct"] = round(
        100.0 * (eq.iloc[-1] - eq.iloc[0]) / max(eq.iloc[0], 1e-9), 2
    )
    if out["max_drawdown_pct"] > 8.0:
        out["warnings"].append(
            f"DRAWDOWN {out['max_drawdown_pct']}% > 8% — near peak breaker (10%)"
        )
    return out


def section_red_flags(signals: pd.DataFrame, events: pd.DataFrame,
                      ticks: pd.DataFrame) -> dict:
    """Non-obvious problems worth surfacing."""
    out = {"section": "red_flags", "warnings": []}

    # Regime stuck at one state
    if not signals.empty and "regime" in signals.columns:
        regime_counts = signals["regime"].value_counts()
        if len(regime_counts) == 1:
            out["warnings"].append(
                f"Regime never changed from '{regime_counts.index[0]}' — "
                f"HMM may be overconfident or malformed."
            )

    # Same LSTM prediction repeated (model frozen?)
    if not signals.empty and "lstm_prediction" in signals.columns:
        preds = pd.to_numeric(signals["lstm_prediction"], errors="coerce").dropna()
        if len(preds) > 20 and preds.std() < 1e-6:
            out["warnings"].append(
                "LSTM prediction has ~zero variance — model may be stuck / not loading correctly."
            )

    # Repeated broker rejects
    if not signals.empty and "block_reason" in signals.columns:
        rejects = signals[signals["block_reason"].fillna("").str.startswith("broker_reject")]
        if len(rejects) > 5:
            out["warnings"].append(
                f"{len(rejects)} broker rejects — check spreads, margin, symbol trading status."
            )

    # Circuit breaker trip
    if not ticks.empty and "breaker_active" in ticks.columns:
        breakers = ticks["breaker_active"].fillna("").astype(str)
        tripped = breakers[(breakers != "") & (breakers != "none")]
        if not tripped.empty:
            top = tripped.value_counts().head(3)
            lines = [f"{k}:{v}" for k, v in top.items()]
            out["warnings"].append(
                f"Circuit breaker active rows: {len(tripped)} — {'; '.join(lines)}"
            )

    # Long gap since last signal
    if not signals.empty:
        last = signals["timestamp"].max()
        now = datetime.now(tz=timezone.utc)
        gap_hours = (now - last).total_seconds() / 3600
        if gap_hours > 2:
            out["warnings"].append(
                f"Last signal {gap_hours:.1f}h ago — bot may be stopped or not processing bars."
            )

    return out


# =========================================================================
# Main
# =========================================================================

def _print_section(sec: dict) -> None:
    name = sec.pop("section", "?")
    warnings = sec.pop("warnings", [])
    print()
    print(f"== {name.upper()} ==")
    for k, v in sec.items():
        if isinstance(v, (dict, list)):
            print(f"  {k}: {json.dumps(v, default=str, indent=2)}")
        else:
            print(f"  {k}: {v}")
    if warnings:
        print("  --- warnings ---")
        for w in warnings:
            print(f"   ! {w}")
    # Re-insert for JSON output
    sec["section"] = name
    sec["warnings"] = warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=int, default=None,
                        help="Only analyze last N days (default: all)")
    parser.add_argument("--symbol", default=None,
                        help="Filter to a single symbol")
    parser.add_argument("--out", default=str(LOG_DIR / "analysis_latest.json"))
    args = parser.parse_args()

    since_dt = None
    if args.since:
        since_dt = datetime.now(tz=timezone.utc) - timedelta(days=args.since)

    signals = _filter_since(_read_csv(LOG_DIR / "signal_audit.csv"), since_dt)
    events = _filter_since(_read_csv(LOG_DIR / "trade_events.csv"), since_dt)
    ticks = _filter_since(_read_csv(LOG_DIR / "tick_summary.csv"), since_dt)

    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "since": since_dt.isoformat() if since_dt else "all",
        "symbol_filter": args.symbol or "all",
    }
    sections = [
        section_session(signals, ticks),
        section_signal_funnel(signals, args.symbol),
        section_trade_outcomes(events, args.symbol),
        section_equity(ticks),
        section_red_flags(signals, events, ticks),
    ]

    print("=" * 70)
    print(f"CORTEX PAPER-TRADING ANALYSIS — {report['generated_at']}")
    print(f"Window: {report['since']}   Symbol: {report['symbol_filter']}")
    print("=" * 70)

    all_warnings = []
    for sec in sections:
        all_warnings.extend(sec.get("warnings", []))
        _print_section(sec)

    print()
    print("== SUMMARY ==")
    print(f"  total warnings: {len(all_warnings)}")
    if all_warnings:
        print("  Review the warnings above before continuing.")
    else:
        print("  No red flags detected.")
    print("=" * 70)

    report["sections"] = sections
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
