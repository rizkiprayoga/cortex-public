"""
analyze_news_impact.py — Post-hoc: did Tier 2/3 events actually hurt trades?

Reads:
  - config/economic_calendar.yaml  (all tracked events)
  - data/logs/invariants.jsonl     (trade.near_economic_event findings)
  - Postgres trades table          (PnL, close_reason_code, bars_held)

Outputs a per-event-type table with:
  n_closes_near_event  — trades closing within ±4h of an event of this type
  avg_pnl              — mean PnL of those closes
  win_rate             — fraction with pnl_usd > 0
  median_pnl           — median PnL
  adverse_closes       — stop_loss or time_exit within the window

Use this to promote Tier 2 → Tier 1 (add to blackout) when a specific
event type shows a consistently adverse PnL signature.

Usage:
    python scripts/analyze_news_impact.py              # last 30 days
    python scripts/analyze_news_impact.py --days 90    # last quarter
    python scripts/analyze_news_impact.py --tier 2     # only Tier 2 events
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



def _load_invariant_findings(path: Path, since: datetime) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("invariant") != "trade.near_economic_event":
            continue
        try:
            ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if ts < since:
            continue
        out.append(row)
    return out


def _summarize(findings: list[dict], tier_filter: Optional[int]) -> None:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for f in findings:
        ctx = f.get("context") or {}
        if tier_filter is not None and ctx.get("tier") != tier_filter:
            continue
        buckets[ctx.get("event", "?")].append(ctx)

    if not buckets:
        print("No matching close-near-event findings in the window.")
        return

    print(f"{'Event':38s} {'Tier':>4s} {'N':>4s} {'Avg PnL':>10s} "
          f"{'Median':>10s} {'Win%':>6s} {'Adv%':>6s}")
    print("-" * 84)
    for name in sorted(buckets, key=lambda n: -len(buckets[n])):
        rows = buckets[name]
        pnls = [float(r.get("pnl_usd", 0.0)) for r in rows]
        adverse = sum(
            1 for r in rows
            if str(r.get("reason", "")).lower().startswith(("sl", "stop", "time"))
        )
        tier = rows[0].get("tier", "?")
        n = len(rows)
        wr = sum(1 for p in pnls if p > 0) / n * 100
        adv = adverse / n * 100
        print(f"{name:38s} {tier:>4} {n:>4d} {mean(pnls):>+10.2f} "
              f"{median(pnls):>+10.2f} {wr:>5.1f}% {adv:>5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--tier", type=int, choices=(1, 2, 3), default=None)
    ap.add_argument(
        "--jsonl", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "logs" / "invariants.jsonl",
    )
    args = ap.parse_args()

    since = datetime.now(tz=timezone.utc) - timedelta(days=args.days)
    findings = _load_invariant_findings(args.jsonl, since)
    print(f"Loaded {len(findings)} close-near-event findings since {since.isoformat()}")
    _summarize(findings, args.tier)


if __name__ == "__main__":
    main()
