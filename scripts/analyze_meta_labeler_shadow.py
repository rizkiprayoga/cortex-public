"""
analyze_meta_labeler_shadow.py — M-1 shadow-mode verdict (C-option helper).

After running the bot with ``CORTEX_META_LABELER_SHADOW=1`` for a few weeks,
this script reconciles the WOULD_BLOCK / WOULD_ALLOW decisions from
``data/logs/signal_audit.csv`` against actual trade outcomes from the
``trades`` table to answer: *if we had flipped the gate on live, would it
have helped?*

Usage:
    python scripts/analyze_meta_labeler_shadow.py
    python scripts/analyze_meta_labeler_shadow.py --days 30
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("analyze_meta_labeler_shadow")


_SHADOW_RE = re.compile(
    r"meta_labeler_shadow: P\(win\)=(?P<proba>[\d.]+)\s+(?:>=|<)\s+"
    r"threshold\s+\S+\s+WOULD_(?P<decision>ALLOW|BLOCK)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=30,
                   help="Analyze the last N days of signals (default 30).")
    p.add_argument("--audit-csv", type=Path,
                   default=Path("data/logs/signal_audit.csv"))
    return p.parse_args()


def load_shadow_signals(audit_path: Path, since: datetime) -> pd.DataFrame:
    if not audit_path.exists():
        raise SystemExit(f"signal_audit.csv not found at {audit_path}")

    df = pd.read_csv(audit_path, on_bad_lines="skip")
    if "reasoning" not in df.columns or "timestamp" not in df.columns:
        raise SystemExit("unexpected signal_audit schema")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["timestamp"] >= pd.Timestamp(since)]

    def _extract(reasoning: object) -> tuple[float | None, str | None]:
        if not isinstance(reasoning, str):
            return None, None
        m = _SHADOW_RE.search(reasoning)
        if m is None:
            return None, None
        return float(m.group("proba")), m.group("decision")

    parsed = df["reasoning"].apply(_extract)
    df["shadow_proba"] = [p for p, _ in parsed]
    df["shadow_decision"] = [d for _, d in parsed]
    return df[df["shadow_decision"].notna()].reset_index(drop=True)


async def load_trades_since(since: datetime) -> pd.DataFrame:
    import asyncpg

    dsn = os.environ["POSTGRES_DSN"].replace(
        "postgresql+asyncpg://", "postgresql://",
    )
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT symbol, direction, timestamp_open, timestamp_close,
                   pnl_usd, close_reason_code
            FROM trades
            WHERE timestamp_open >= $1
            """,
            since.isoformat(),
        )
    finally:
        await conn.close()
    return pd.DataFrame(rows, columns=[
        "symbol", "direction", "timestamp_open", "timestamp_close",
        "pnl_usd", "close_reason_code",
    ])


def match_signals_to_trades(
    shadow_df: pd.DataFrame, trades_df: pd.DataFrame,
    match_window_s: int = 900,
) -> pd.DataFrame:
    """Join signals to trades by (symbol, direction, time<=15m) tolerance."""
    if trades_df.empty or shadow_df.empty:
        return pd.DataFrame()

    trades_df = trades_df.copy()
    trades_df["timestamp_open"] = pd.to_datetime(
        trades_df["timestamp_open"], utc=True, errors="coerce",
    )
    shadow_df = shadow_df[shadow_df["direction"].isin({"buy", "sell"})].copy()

    matched_rows: list[dict] = []
    for _, s in shadow_df.iterrows():
        same = trades_df[
            (trades_df["symbol"] == s["symbol"])
            & (trades_df["direction"] == s["direction"])
            & (trades_df["timestamp_open"] >= s["timestamp"])
            & (trades_df["timestamp_open"]
               <= s["timestamp"] + pd.Timedelta(seconds=match_window_s))
        ].sort_values("timestamp_open")
        if same.empty:
            continue
        t = same.iloc[0]
        matched_rows.append({
            "symbol": s["symbol"],
            "direction": s["direction"],
            "signal_ts": s["timestamp"],
            "trade_ts": t["timestamp_open"],
            "shadow_decision": s["shadow_decision"],
            "shadow_proba": s["shadow_proba"],
            "pnl_usd": float(t["pnl_usd"]) if pd.notna(t["pnl_usd"]) else None,
            "close_reason": t["close_reason_code"],
        })
    return pd.DataFrame(matched_rows)


def summarize(matched: pd.DataFrame) -> None:
    if matched.empty:
        print("\nNo shadow-mode signals matched to executed trades.")
        print("Either shadow mode hasn't been running long enough, the bot")
        print("hasn't traded, or CORTEX_META_LABELER_SHADOW=1 isn't set.")
        return

    closed = matched[matched["pnl_usd"].notna()].copy()
    if closed.empty:
        print("\nMatched signals exist but no closed trades yet — wait for more data.")
        return
    closed["is_win"] = closed["pnl_usd"] > 0

    print("\n" + "=" * 78)
    print("META-LABELER SHADOW-MODE VERDICT")
    print("=" * 78)
    print(f"  Signal-trade matched rows:   {len(matched)}")
    print(f"  Closed trades:               {len(closed)}")
    print()

    total_block_pnl = 0.0
    for sym, grp in closed.groupby("symbol"):
        block = grp[grp["shadow_decision"] == "BLOCK"]
        allow = grp[grp["shadow_decision"] == "ALLOW"]
        block_wins = int(block["is_win"].sum())
        block_losses = int((~block["is_win"]).sum())
        allow_wins = int(allow["is_win"].sum())
        allow_losses = int((~allow["is_win"]).sum())
        block_pnl = float(block["pnl_usd"].sum())
        allow_pnl = float(allow["pnl_usd"].sum())

        print(f"  {sym:<8}  n={len(grp):>4d}")
        print(f"    shadow=ALLOW  n={len(allow):>3d}  "
              f"{allow_wins}W/{allow_losses}L  net ${allow_pnl:>+10,.2f}")
        print(f"    shadow=BLOCK  n={len(block):>3d}  "
              f"{block_wins}W/{block_losses}L  net ${block_pnl:>+10,.2f}")
        verdict = ("SAVES MONEY" if block_pnl < 0
                   else "COSTS MONEY" if block_pnl > 0
                   else "NEUTRAL")
        print(f"    -> flipping gate on would have:  {verdict} "
              f"(${-block_pnl:>+10,.2f})")
        print()
        total_block_pnl += block_pnl

    print("  " + "-" * 70)
    verb = "saved" if total_block_pnl < 0 else "cost"
    print(f"  PORTFOLIO: flipping CORTEX_META_LABELER=1 would have "
          f"{verb} ${abs(total_block_pnl):,.2f} across this window.")
    print()


def main() -> int:
    args = parse_args()
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    logger.info("Analyzing shadow-mode signals since %s", since.isoformat())

    shadow_df = load_shadow_signals(args.audit_csv, since)
    logger.info("Shadow-mode signals parsed: %d", len(shadow_df))
    if shadow_df.empty:
        print("\nNo shadow-mode reasoning lines found in signal_audit.csv.")
        print("Is CORTEX_META_LABELER_SHADOW=1 set + bot restarted?")
        return 0

    trades_df = asyncio.run(load_trades_since(since))
    logger.info("Trades in window: %d", len(trades_df))

    matched = match_signals_to_trades(shadow_df, trades_df)
    summarize(matched)
    return 0


if __name__ == "__main__":
    sys.exit(main())
