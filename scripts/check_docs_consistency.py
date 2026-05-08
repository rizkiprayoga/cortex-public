"""
check_docs_consistency.py — Doc-Drift Linter (Plan 2)

Prevents numeric claims in docs / CLAUDE.md / source docstrings from
silently drifting out of sync with config/settings.yaml. Every tagged
claim is verified against the canonical config value; any mismatch
exits non-zero with a human-readable report.

Why this exists
---------------
Claims like "60 H1 bars", "1.25% risk per trade", or "signal threshold
0.45" are scattered across CLAUDE.md, docs/*.md, and Python docstrings.
When settings.yaml changes, the docs rot. This script is the enforcement
layer so that drift is impossible to ship.

Marker syntax
-------------
In Markdown (HTML comment — invisible when rendered):

    The signal threshold is **0.45** <!-- doc-check: strategy.min_confidence -->

In Python / YAML / TS comments:

    # Default 20 H4 bars  # doc-check: strategy.per_symbol_params.USDJPY.time_exit_h1_bars / 4

The optional `± * /` arithmetic transform (one operator, one constant)
is for unit-converted claims (e.g. 60 H1 bars ÷ 4 = 15 H4 bars). If the
numeric token adjacent to the marker equals the config value after the
transform, the claim passes.

Usage
-----
    python scripts/check_docs_consistency.py                 # full scan, exit 1 on drift
    python scripts/check_docs_consistency.py --verbose       # show passing rows too
    python scripts/check_docs_consistency.py --paths a.md b  # restrict scope
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"

# Files/globs scanned by default. Tests use the --paths override.
DEFAULT_SCAN_GLOBS = [
    "CLAUDE.md",
    "README.md",
    "docs/**/*.md",
    "src/**/*.py",
    "scripts/**/*.py",
    "main.py",
    "frontend/src/**/*.ts",
    "frontend/src/**/*.tsx",
]

# A doc-check marker. Captures:
#   group 1: dotted config key (letters, digits, underscore, dot)
#   group 2: optional operator (+ - * /)
#   group 3: optional numeric constant
MARKER_RE = re.compile(
    r"doc-check:\s*([A-Za-z0-9_.]+)(?:\s*([+\-*/])\s*(-?\d+(?:\.\d+)?))?"
)

# Numeric token — allows optional leading sign, decimals, %.
# We strip a trailing %, commas, and unit suffixes like "s" / "x" before
# comparing. Matches "1.25%", "60", "-0.5", "2,000".
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

TOLERANCE = 1e-6


@dataclass
class Finding:
    path: Path
    line_no: int
    key: str
    claimed: float | None
    expected: float
    transform: str | None  # e.g. "/4" or None
    ok: bool
    raw_line: str

    def summary(self) -> str:
        loc = f"{self.path.relative_to(PROJECT_ROOT)}:{self.line_no}"
        expr = self.key + (f" {self.transform}" if self.transform else "")
        if self.ok:
            return f"[OK]   {loc:<60} {expr} = {self.claimed} (config: {self.expected})"
        claimed = "<no number>" if self.claimed is None else str(self.claimed)
        return (
            f"[FAIL] {loc:<60} {expr} = {claimed} "
            f"(config: {self.expected}) <-- DRIFT"
        )


def flatten_yaml(data: Any, prefix: str = "") -> dict[str, Any]:
    """Depth-first flatten dict-of-dicts into dotted keys.

    Lists are left as-is and keyed by their parent path. Only scalar
    leaves end up in the result with their dotted keys, because the
    marker syntax only supports scalar comparisons.
    """
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            subkey = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_yaml(v, subkey))
    else:
        out[prefix] = data
    return out


def load_config_values(path: Path = CONFIG_FILE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return flatten_yaml(yaml.safe_load(fh))


def apply_transform(value: float, op: str | None, operand: str | None) -> float:
    if op is None or operand is None:
        return value
    x = float(operand)
    if op == "+":
        return value + x
    if op == "-":
        return value - x
    if op == "*":
        return value * x
    if op == "/":
        if x == 0:
            raise ValueError("division by zero in marker transform")
        return value / x
    raise ValueError(f"unsupported transform operator: {op!r}")


def extract_claimed_numbers(line: str, marker_start: int) -> list[float]:
    """Return all numeric tokens on the line *excluding* those inside
    the marker itself (i.e. before `marker_start`).

    Rationale: locating "the" claimed number is brittle — consider
    "15 H4 bars" where "15" is the claim and "4" is unit shorthand.
    Instead of guessing, we accept the claim as correct if any number
    on the line matches the expected value. In practice claims are
    unique enough on their own line that this stays precise.
    """
    before = line[:marker_start]
    out: list[float] = []
    for m in NUM_RE.finditer(before):
        raw = m.group(0).replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def scan_file(path: Path, config: dict[str, Any]) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    findings: list[Finding] = []
    # Pragma-driven skip region: everything between `lint-off` and
    # `lint-on` is ignored. Used to embed example markers in docs
    # (like the Conventions block in CLAUDE.md) without triggering.
    lint_off = False
    for line_no, line in enumerate(text.splitlines(), start=1):
        if "doc-check-lint-off" in line:
            lint_off = True
            continue
        if "doc-check-lint-on" in line:
            lint_off = False
            continue
        if lint_off:
            continue
        for m in MARKER_RE.finditer(line):
            key, op, operand = m.group(1), m.group(2), m.group(3)
            if key not in config:
                findings.append(Finding(
                    path=path, line_no=line_no, key=key,
                    claimed=None, expected=float("nan"),
                    transform=None, ok=False,
                    raw_line=line.rstrip(),
                ))
                continue
            raw = config[key]
            try:
                expected_raw = float(raw)
            except (TypeError, ValueError):
                # Non-numeric config leaf — skip with a warning-style
                # finding. This catches markers pointing at string
                # values like `timeframes.signal: "H4"`.
                findings.append(Finding(
                    path=path, line_no=line_no, key=key,
                    claimed=None, expected=float("nan"),
                    transform=None, ok=False,
                    raw_line=line.rstrip(),
                ))
                continue
            expected = apply_transform(expected_raw, op, operand)
            claimed_nums = extract_claimed_numbers(line, m.start())
            match = next(
                (n for n in claimed_nums if abs(n - expected) <= TOLERANCE),
                None,
            )
            ok = match is not None
            transform = f"{op} {operand}" if op else None
            # For reporting, prefer the matched number; otherwise show
            # whatever was closest so the drift is legible.
            if ok:
                claimed: float | None = match
            elif claimed_nums:
                claimed = min(claimed_nums, key=lambda n: abs(n - expected))
            else:
                claimed = None
            findings.append(Finding(
                path=path, line_no=line_no, key=key,
                claimed=claimed, expected=expected,
                transform=transform, ok=ok, raw_line=line.rstrip(),
            ))
    return findings


SELF_PATH = Path(__file__).resolve()


def iter_scan_paths(globs: Iterable[str]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in globs:
        for p in PROJECT_ROOT.glob(pattern):
            # Skip the linter itself — its docstring contains example
            # markers that would otherwise be parsed as real ones.
            if p.is_file() and p not in seen and p.resolve() != SELF_PATH:
                seen.add(p)
                out.append(p)
    return out


def run_check(
    paths: Iterable[Path] | None = None,
    config_path: Path = CONFIG_FILE,
) -> tuple[list[Finding], list[Finding]]:
    """Return (all_findings, failures). Pure function for testing."""
    config = load_config_values(config_path)
    scan_paths = list(paths) if paths is not None else iter_scan_paths(DEFAULT_SCAN_GLOBS)
    all_findings: list[Finding] = []
    for p in scan_paths:
        all_findings.extend(scan_file(p, config))
    failures = [f for f in all_findings if not f.ok]
    return all_findings, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify every doc-check marker in docs/src matches config/settings.yaml.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show passing rows too, not just failures.",
    )
    parser.add_argument(
        "--paths", nargs="*", default=None,
        help="Explicit file paths to scan (overrides default globs).",
    )
    args = parser.parse_args()

    scan = [Path(p) for p in args.paths] if args.paths else None
    all_findings, failures = run_check(paths=scan)

    if args.verbose:
        for f in all_findings:
            print(f.summary())
    else:
        for f in failures:
            print(f.summary())

    total = len(all_findings)
    bad = len(failures)
    good = total - bad

    if total == 0:
        print("doc-check: no markers found (nothing to verify).")
        return 0

    print(f"\ndoc-check: {good}/{total} markers OK, {bad} drift.")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
