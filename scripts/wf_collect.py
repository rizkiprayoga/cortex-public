"""Collect walk-forward backtest metrics from existing CSV outputs."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "data" / "backtest_results"


def summarize_trades(trades_csv: Path, initial_equity: float = 10000.0) -> dict:
    """Parse a backtest_trades_{symbol}.csv and compute summary stats."""
    if not trades_csv.exists():
        return {"error": f"missing: {trades_csv}"}
    df = pd.read_csv(trades_csv)
    if df.empty or "pnl" not in df.columns:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
        }
    pnl = df["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    total_wins = wins.sum()
    total_losses = losses.sum()

    profit_factor = (total_wins / abs(total_losses)) if abs(total_losses) > 0 else float("inf")

    # Equity curve from PnL
    equity = initial_equity + pnl.cumsum()
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak * 100.0
    max_dd = abs(drawdown.min()) if len(drawdown) else 0.0

    return {
        "total_trades": int(len(df)),
        "win_rate": float((pnl > 0).sum() / len(pnl)),
        "net_pnl": float(pnl.sum()),
        "profit_factor": float(profit_factor),
        "max_drawdown_pct": float(max_dd),
        "total_wins_usd": float(total_wins),
        "total_losses_usd": float(total_losses),
        "avg_win_usd": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_usd": float(losses.mean()) if len(losses) else 0.0,
    }


if __name__ == "__main__":
    year = sys.argv[1]
    symbol = sys.argv[2]
    src = RESULTS_DIR / f"backtest_trades_{symbol}.csv"
    m = summarize_trades(src)
    m["year"] = int(year)
    m["symbol"] = symbol
    print(json.dumps(m))
