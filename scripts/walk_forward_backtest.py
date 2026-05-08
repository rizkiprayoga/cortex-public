"""
walk_forward_backtest.py — True walk-forward OOS validation.

For each target year, retrains HMM+LSTM on data ending the prior
year-end, then backtests the target year. This gives a genuine
out-of-sample result — the model never saw the test year's data.

Saves a single summary CSV at data/walk_forward_results.csv.

Usage:
    python scripts/walk_forward_backtest.py --years 2021 2022 2023 2024
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
SYMBOLS = ["XAUUSD", "USDJPY", "EURUSD", "USDCAD"]
PYTHON = sys.executable


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a subprocess, capture output, raise on failure."""
    logger.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def train_through(end_date: str) -> None:
    """Retrain HMM + LSTM+PCA+TB on data ending at end_date."""
    run([PYTHON, "scripts/train_hmm.py",
         "--symbols", *SYMBOLS,
         "--bars", "5000",
         "--end-date", end_date,
         "--no-snapshot"])
    run([PYTHON, "scripts/train_deep_learning.py",
         "--symbols", *SYMBOLS,
         "--bars", "0",
         "--pca-components", "25",
         "--triple-barrier",
         "--end-date", end_date,
         "--no-snapshot"])


def backtest_year(symbol: str, year: int) -> dict:
    """Run full backtest and parse summary metrics."""
    # Remove halt flag between runs
    flag = ROOT / "data/logs/TRADING_HALTED.flag"
    if flag.exists():
        flag.unlink()

    cmd = [PYTHON, "scripts/backtest.py",
           "--symbols", symbol,
           "--start", f"{year}-01-01",
           "--end", f"{year}-12-31",
           "--mode", "full"]
    result = run(cmd)

    # Parse metrics from stdout (logger output goes to stderr)
    text = result.stdout + result.stderr
    metrics = {}
    for line in text.splitlines():
        for key in ["total_trades", "win_rate", "net_pnl",
                    "max_drawdown_pct", "profit_factor", "sharpe_ratio"]:
            marker = f"  {key}:"
            if marker in line:
                val = line.split(marker, 1)[1].strip()
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = val
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int,
                        default=[2021, 2022, 2023, 2024])
    parser.add_argument("--output",
                        default="data/walk_forward_results.json")
    args = parser.parse_args()

    results = []
    for year in args.years:
        end_date = f"{year - 1}-12-31"
        logger.info("=" * 70)
        logger.info("WALK-FORWARD: Year %s (train ends %s)", year, end_date)
        logger.info("=" * 70)

        # Retrain models
        train_through(end_date)

        # Backtest each symbol on the target year
        for symbol in SYMBOLS:
            logger.info("-- Backtesting %s %s --", symbol, year)
            metrics = backtest_year(symbol, year)
            row = {
                "year": year,
                "symbol": symbol,
                "trained_through": end_date,
                **metrics,
            }
            results.append(row)
            logger.info("  %s %s: trades=%s pf=%s pnl=%s",
                         year, symbol,
                         row.get("total_trades"),
                         row.get("profit_factor"),
                         row.get("net_pnl"))

    # Save JSON
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", out)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'YEAR':<6}{'SYMBOL':<10}{'TRADES':>8}{'WR':>8}{'PNL':>12}"
          f"{'PF':>8}{'MAXDD%':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['year']:<6}{r['symbol']:<10}"
              f"{int(r.get('total_trades', 0)):>8}"
              f"{r.get('win_rate', 0):>8.2%}"
              f"{r.get('net_pnl', 0):>12.2f}"
              f"{r.get('profit_factor', 0):>8.2f}"
              f"{r.get('max_drawdown_pct', 0):>10.2f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
