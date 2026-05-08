"""
analyze_backtest.py — Post-Backtest Analysis and Reporting

Reads completed backtest trade/equity CSVs and prints comprehensive analysis:
    1. Performance summary (Sharpe, Sortino, Calmar, max DD, profit factor, win rate)
    2. Regime analysis (win rate & avg PnL by regime)
    3. Strategy analysis (win rate & avg PnL by strategy)
    4. Monthly returns table
    5. Top drawdown periods
    6. Parameter recommendations

Read-only — never modifies parameters. Prints to console.

Usage:
    python scripts/analyze_backtest.py --symbol XAUUSD
    python scripts/analyze_backtest.py --symbol BTCUSD --trades data/logs/backtest_trades_BTCUSD.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(symbol: str, data_dir: str = "data/logs",
              trades_file: str = None, equity_file: str = None):
    """Load trades and equity CSVs for a symbol."""
    base = Path(data_dir)

    tf = Path(trades_file) if trades_file else base / f"backtest_trades_{symbol}.csv"
    ef = Path(equity_file) if equity_file else base / f"backtest_equity_{symbol}.csv"

    if not tf.exists():
        print(f"ERROR: Trades file not found: {tf}")
        sys.exit(1)
    if not ef.exists():
        print(f"ERROR: Equity file not found: {ef}")
        sys.exit(1)

    trades = pd.read_csv(tf)
    equity = pd.read_csv(ef)

    return trades, equity


def performance_summary(trades: pd.DataFrame, equity: pd.DataFrame) -> dict:
    """Compute comprehensive performance metrics."""
    n = len(trades)
    if n == 0:
        return {"total_trades": 0}

    pnls = trades["pnl"].values
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate = len(wins) / n
    net_pnl = float(np.sum(pnls))
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Equity-based metrics
    equities = equity["equity"].values
    max_dd = float(equity["drawdown_pct"].max())

    # Returns
    if len(equities) > 1:
        returns = np.diff(equities) / equities[:-1]
        returns = returns[np.isfinite(returns)]
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))

        # Sharpe (annualized assuming H4 bars: ~6 bars/day, ~252 trading days)
        bars_per_year = 6 * 252
        sharpe = (mean_ret / std_ret * np.sqrt(bars_per_year)) if std_ret > 0 else 0.0

        # Sortino
        downside = returns[returns < 0]
        downside_std = float(np.std(downside)) if len(downside) > 0 else 0.0
        sortino = (mean_ret / downside_std * np.sqrt(bars_per_year)) if downside_std > 0 else 0.0

        # Calmar
        annual_return = mean_ret * bars_per_year
        calmar = (annual_return / (max_dd / 100.0)) if max_dd > 0 else 0.0
    else:
        sharpe = sortino = calmar = 0.0

    # Expectancy
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        "total_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "net_pnl": net_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "expectancy": expectancy,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "initial_equity": float(equities[0]) if len(equities) > 0 else 0.0,
        "final_equity": float(equities[-1]) if len(equities) > 0 else 0.0,
    }


def regime_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """Win rate and avg PnL by regime."""
    if "regime_label" not in trades.columns or trades["regime_label"].isna().all():
        return pd.DataFrame()

    grouped = trades.groupby("regime_label").agg(
        count=("pnl", "size"),
        wins=("pnl", lambda x: (x > 0).sum()),
        avg_pnl=("pnl", "mean"),
        total_pnl=("pnl", "sum"),
        avg_r=("r_multiple", "mean"),
    )
    grouped["win_rate"] = grouped["wins"] / grouped["count"]
    return grouped.sort_values("count", ascending=False)


def strategy_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """Win rate and avg PnL by strategy."""
    if "strategy_name" not in trades.columns or trades["strategy_name"].isna().all():
        return pd.DataFrame()

    grouped = trades.groupby("strategy_name").agg(
        count=("pnl", "size"),
        wins=("pnl", lambda x: (x > 0).sum()),
        avg_pnl=("pnl", "mean"),
        total_pnl=("pnl", "sum"),
        avg_r=("r_multiple", "mean"),
    )
    grouped["win_rate"] = grouped["wins"] / grouped["count"]
    return grouped.sort_values("count", ascending=False)


def exit_reason_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """Breakdown by exit reason."""
    if "exit_reason" not in trades.columns:
        return pd.DataFrame()

    grouped = trades.groupby("exit_reason").agg(
        count=("pnl", "size"),
        avg_pnl=("pnl", "mean"),
        total_pnl=("pnl", "sum"),
        avg_r=("r_multiple", "mean"),
    )
    return grouped.sort_values("count", ascending=False)


def monthly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    """Monthly PnL returns table."""
    eq = equity.copy()
    eq["bar_timestamp"] = pd.to_datetime(eq["bar_timestamp"])
    eq.set_index("bar_timestamp", inplace=True)

    monthly = eq["equity"].resample("ME").last()
    monthly_returns = monthly.pct_change().dropna() * 100

    # Reshape to year × month
    df = monthly_returns.to_frame("return_pct")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(values="return_pct", index="year", columns="month", aggfunc="sum")
    pivot.columns = [f"M{m:02d}" for m in pivot.columns]
    pivot["Annual"] = pivot.sum(axis=1)
    return pivot


def top_drawdowns(equity: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Find the top N drawdown periods."""
    equities = equity["equity"].values
    timestamps = equity["bar_timestamp"].values

    peak = equities[0]
    dd_start = 0
    drawdowns = []
    in_drawdown = False
    current_dd_start = 0

    for i in range(len(equities)):
        if equities[i] >= peak:
            if in_drawdown:
                # Drawdown ended
                dd_depth = (peak - min(equities[current_dd_start:i+1])) / peak * 100
                dd_trough_idx = current_dd_start + np.argmin(equities[current_dd_start:i+1])
                drawdowns.append({
                    "start": str(timestamps[current_dd_start]),
                    "trough": str(timestamps[dd_trough_idx]),
                    "end": str(timestamps[i]),
                    "depth_pct": round(dd_depth, 2),
                    "bars": i - current_dd_start,
                })
                in_drawdown = False
            peak = equities[i]
        else:
            if not in_drawdown:
                current_dd_start = i
                in_drawdown = True

    # If still in drawdown at end
    if in_drawdown:
        dd_depth = (peak - min(equities[current_dd_start:])) / peak * 100
        dd_trough_idx = current_dd_start + np.argmin(equities[current_dd_start:])
        drawdowns.append({
            "start": str(timestamps[current_dd_start]),
            "trough": str(timestamps[dd_trough_idx]),
            "end": "ongoing",
            "depth_pct": round(dd_depth, 2),
            "bars": len(equities) - current_dd_start,
        })

    drawdowns.sort(key=lambda x: x["depth_pct"], reverse=True)
    return drawdowns[:top_n]


def parameter_recommendations(perf: dict, trades: pd.DataFrame) -> list[str]:
    """Flag obvious issues and suggest parameter adjustments."""
    recs = []

    n = perf.get("total_trades", 0)
    if n == 0:
        recs.append("NO TRADES GENERATED — signal threshold may be too tight or models may not be loaded")
        return recs

    if n < 20:
        recs.append(f"Only {n} trades in backtest period — signal threshold may be too tight. "
                     "Consider lowering min_confidence or signal_threshold.")

    wr = perf.get("win_rate", 0)
    if wr < 0.35:
        recs.append(f"Win rate {wr:.1%} is low — entry quality may be poor. "
                     "Consider increasing min_confidence or flicker_bars_required.")
    elif wr > 0.70:
        recs.append(f"Win rate {wr:.1%} is unusually high — check for look-ahead bias or overfitting.")

    pf = perf.get("profit_factor", 0)
    if 0 < pf < 1.0:
        recs.append(f"Profit factor {pf:.2f} < 1.0 — strategy is net losing. "
                     "Review exit parameters (ATR trail multiplier, tier R-multiples).")

    md = perf.get("max_drawdown_pct", 0)
    if md > 15:
        recs.append(f"Max drawdown {md:.1f}% exceeds 15% threshold. "
                     "Consider tightening circuit breaker thresholds or reducing allocation_pct.")

    sharpe = perf.get("sharpe_ratio", 0)
    if 0 < sharpe < 0.5:
        recs.append(f"Sharpe ratio {sharpe:.2f} is marginal. "
                     "Returns may not justify risk. Review strategy parameters.")

    # Check trade duration distribution
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        try:
            entry_ts = pd.to_datetime(trades["entry_time"])
            exit_ts = pd.to_datetime(trades["exit_time"])
            durations = (exit_ts - entry_ts).dt.total_seconds() / 3600  # hours
            median_hrs = durations.median()
            if median_hrs < 4:
                recs.append(f"Median trade duration {median_hrs:.1f}h is very short — "
                            "possible overtrading on noise.")
            elif median_hrs > 500:
                recs.append(f"Median trade duration {median_hrs:.0f}h ({median_hrs/24:.0f} days) — "
                            "trailing stop may be too wide.")
        except Exception:
            pass

    # Long-only check
    if "direction" in trades.columns:
        dirs = trades["direction"].value_counts()
        if "sell" in dirs.index and dirs.get("sell", 0) > 0:
            recs.append("WARNING: Short trades detected — long_only_mode may not be enforced.")

    if not recs:
        recs.append("No obvious issues detected. Strategy parameters appear reasonable.")

    return recs


def print_report(symbol: str, trades: pd.DataFrame, equity: pd.DataFrame):
    """Print full analysis report to console."""
    print(f"\n{'='*70}")
    print(f"  BACKTEST ANALYSIS: {symbol}")
    print(f"{'='*70}")

    # 1. Performance summary
    perf = performance_summary(trades, equity)
    print(f"\n--- Performance Summary ---")
    if perf.get("total_trades", 0) == 0:
        print("  No trades generated.")
        return

    print(f"  Period:         {equity['bar_timestamp'].iloc[0][:10]} to "
          f"{equity['bar_timestamp'].iloc[-1][:10]}")
    print(f"  Total Bars:     {len(equity):,}")
    print(f"  Total Trades:   {perf['total_trades']}")
    print(f"  Wins/Losses:    {perf['wins']} / {perf['losses']}")
    print(f"  Win Rate:       {perf['win_rate']:.1%}")
    print(f"  Net PnL:        ${perf['net_pnl']:,.2f}")
    print(f"  Profit Factor:  {perf['profit_factor']:.2f}")
    print(f"  Payoff Ratio:   {perf['payoff_ratio']:.2f}")
    print(f"  Avg Win:        ${perf['avg_win']:,.2f}")
    print(f"  Avg Loss:       ${perf['avg_loss']:,.2f}")
    print(f"  Expectancy:     ${perf['expectancy']:,.2f}")
    print(f"  Max Drawdown:   {perf['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio:   {perf['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:  {perf['sortino_ratio']:.3f}")
    print(f"  Calmar Ratio:   {perf['calmar_ratio']:.3f}")
    print(f"  Initial Equity: ${perf['initial_equity']:,.2f}")
    print(f"  Final Equity:   ${perf['final_equity']:,.2f}")
    ret_pct = ((perf['final_equity'] - perf['initial_equity'])
               / perf['initial_equity'] * 100) if perf['initial_equity'] > 0 else 0
    print(f"  Total Return:   {ret_pct:+.2f}%")

    # 2. Regime analysis
    ra = regime_analysis(trades)
    if not ra.empty:
        print(f"\n--- Regime Analysis ---")
        for regime, row in ra.iterrows():
            print(f"  {regime:12s}  {int(row['count']):4d} trades  "
                  f"WR={row['win_rate']:.1%}  "
                  f"Avg PnL=${row['avg_pnl']:+.2f}  "
                  f"Total=${row['total_pnl']:+,.2f}  "
                  f"Avg R={row['avg_r']:+.2f}")

    # 3. Strategy analysis
    sa = strategy_analysis(trades)
    if not sa.empty:
        print(f"\n--- Strategy Analysis ---")
        for strat, row in sa.iterrows():
            print(f"  {strat:22s}  {int(row['count']):4d} trades  "
                  f"WR={row['win_rate']:.1%}  "
                  f"Avg PnL=${row['avg_pnl']:+.2f}  "
                  f"Total=${row['total_pnl']:+,.2f}  "
                  f"Avg R={row['avg_r']:+.2f}")

    # 4. Exit reason analysis
    ea = exit_reason_analysis(trades)
    if not ea.empty:
        print(f"\n--- Exit Reason Breakdown ---")
        for reason, row in ea.iterrows():
            print(f"  {reason:18s}  {int(row['count']):4d} trades  "
                  f"Avg PnL=${row['avg_pnl']:+.2f}  "
                  f"Total=${row['total_pnl']:+,.2f}")

    # 5. Monthly returns
    try:
        mr = monthly_returns(equity)
        if not mr.empty:
            print(f"\n--- Monthly Returns (%) ---")
            print(mr.to_string(float_format=lambda x: f"{x:+.2f}"))
    except Exception as exc:
        logger.debug("Monthly returns error: %s", exc)

    # 6. Top drawdowns
    dds = top_drawdowns(equity)
    if dds:
        print(f"\n--- Top {len(dds)} Drawdown Periods ---")
        for j, dd in enumerate(dds, 1):
            print(f"  #{j}: {dd['depth_pct']:.2f}%  "
                  f"Start: {dd['start'][:10]}  "
                  f"Trough: {dd['trough'][:10]}  "
                  f"End: {dd['end'][:10] if dd['end'] != 'ongoing' else 'ongoing'}  "
                  f"({dd['bars']} bars)")

    # 7. Parameter recommendations
    recs = parameter_recommendations(perf, trades)
    print(f"\n--- Recommendations ---")
    for rec in recs:
        print(f"  • {rec}")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze backtest results")
    parser.add_argument("--symbol", required=True, help="Symbol to analyze")
    parser.add_argument("--data-dir", default="data/logs",
                        help="Directory with backtest CSVs")
    parser.add_argument("--trades", default=None,
                        help="Path to trades CSV (overrides default)")
    parser.add_argument("--equity", default=None,
                        help="Path to equity CSV (overrides default)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    trades, equity = load_data(args.symbol, args.data_dir, args.trades, args.equity)
    print_report(args.symbol, trades, equity)


if __name__ == "__main__":
    main()
