"""
analyze_api_perf.py — summarize data/logs/api_perf.jsonl.

Reads the JSONL written by ``api_perf_logger`` in ``src/api/app.py``
and prints three rankings useful for P-1 triage:

  * Top endpoints by **total time consumed** (p95 × count). Cheap
    endpoints called often can dominate user-perceived latency even
    when each individual call is fast, so total time is a better
    triage metric than p95 alone.
  * Top endpoints by **p95 latency**. Used to decide whether a single
    slow endpoint is the bottleneck.
  * Top endpoints by **total bytes**. Candidates for gzip / payload
    shrinkage.

Usage
-----
    # Default: data/logs/api_perf.jsonl, last 24h of records
    python scripts/analyze_api_perf.py

    # Specific window
    python scripts/analyze_api_perf.py --since "2026-04-18T10:00:00Z"

    # Top 5 instead of 10
    python scripts/analyze_api_perf.py --top 5
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


DEFAULT_LOG = Path("data/logs/api_perf.jsonl")


def _parse_since(since: Optional[str]) -> Optional[datetime]:
    if not since:
        return None
    return datetime.fromisoformat(since.replace("Z", "+00:00"))


def _load(path: Path, since: Optional[datetime]) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no log at {path} — enable CORTEX_API_PERF_LOG=1 and restart bot")
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since is not None:
                try:
                    ts = datetime.fromisoformat(r["ts"])
                    if ts < since:
                        continue
                except (KeyError, ValueError):
                    continue
            out.append(r)
    return out


def _key(record: dict) -> str:
    """Group by method + path (not including query string — we want
    'GET /api/history/equity' not 'GET /api/history/equity?limit=500'
    since query params vary)."""
    return f"{record.get('method','?')} {record.get('path','?')}"


def _summarize(records: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[_key(r)].append(r)
    out: dict[str, dict] = {}
    for k, rs in groups.items():
        durs = [int(r.get("duration_ms") or 0) for r in rs]
        bytes_ = [int(r.get("resp_bytes") or 0) for r in rs if r.get("resp_bytes") is not None]
        out[k] = {
            "count":        len(rs),
            "p50_ms":       int(statistics.median(durs)) if durs else 0,
            "p95_ms":       int(_p95(durs)) if durs else 0,
            "max_ms":       max(durs, default=0),
            "total_ms":     sum(durs),
            "mean_bytes":   int(statistics.mean(bytes_)) if bytes_ else 0,
            "total_bytes":  sum(bytes_),
            "error_count":  sum(1 for r in rs if (r.get("status") or 0) >= 500),
        }
    return out


def _p95(values: list[int]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return s[idx]


def _print_table(title: str, rows: list[tuple[str, dict]], value_key: str, unit: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'endpoint':<50} {'n':>6} {'p50':>6} {'p95':>6} {'max':>7} {value_key:>10}")
    for k, v in rows:
        print(
            f"{k:<50} {v['count']:>6} {v['p50_ms']:>5}ms {v['p95_ms']:>5}ms "
            f"{v['max_ms']:>6}ms {v[value_key]:>9}{unit}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", default=str(DEFAULT_LOG))
    p.add_argument("--since", default=None,
                   help="ISO-8601 cutoff; default: last 24h from now")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    since = _parse_since(args.since)
    if since is None:
        since = datetime.now(tz=timezone.utc) - timedelta(hours=24)

    records = _load(Path(args.log), since)
    print(f"Loaded {len(records)} records since {since.isoformat()}")
    if not records:
        return

    stats = _summarize(records)
    errors = sum(v["error_count"] for v in stats.values())
    total_time_s = sum(v["total_ms"] for v in stats.values()) / 1000.0
    print(f"  endpoints={len(stats)}  total wall-time={total_time_s:.1f}s  5xx={errors}")

    by_total_time = sorted(stats.items(), key=lambda kv: kv[1]["total_ms"], reverse=True)
    _print_table(
        f"Top {args.top} by TOTAL TIME CONSUMED (p95 × count proxy — triage metric)",
        by_total_time[: args.top], "total_ms", "ms",
    )

    by_p95 = sorted(stats.items(), key=lambda kv: kv[1]["p95_ms"], reverse=True)
    _print_table(
        f"Top {args.top} by p95 LATENCY",
        by_p95[: args.top], "p95_ms", "ms",
    )

    by_bytes = sorted(stats.items(), key=lambda kv: kv[1]["total_bytes"], reverse=True)
    _print_table(
        f"Top {args.top} by TOTAL BYTES SHIPPED",
        by_bytes[: args.top], "total_bytes", "B",
    )


if __name__ == "__main__":
    main()
