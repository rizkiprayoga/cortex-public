"""
backtest.py — Historical Strategy Backtesting

Simulates the full trading strategy on historical data without
placing any real orders. Uses the same pipeline as live trading:
    HMM regime → LSTM prediction → signal combiner → position sizer

Output:
    - Equity curve CSV (data/logs/backtest_equity.csv)
    - Trade list CSV   (data/logs/backtest_trades.csv)
    - Console summary: Sharpe, max drawdown, win rate, profit factor

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --symbols XAUUSD --start 2022-01-01 --end 2024-01-01

Backtesting assumptions:
    - Entry at next bar's open price (no look-ahead bias)
    - ATR-based stop loss (2× ATR from entry)
    - 3-tier exit: tier1 at 1R, tier2 at 2R, trailing stop remainder
    - No slippage modeled
    - No partial fills — full lot size executed
    - Deterministic: same input → same output (no randomness)
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure project root is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Walk-forward simulation engine (pure function, no external deps)
# ---------------------------------------------------------------------------

ATR_PERIOD = 14
ATR_SL_MULT = 2.0
TIER1_R = 1.0
TIER2_R = 2.0
TIER1_FRAC = 0.33
TIER2_FRAC = 0.33
RISK_PER_TRADE_PCT = 1.0  # risk 1% of equity per trade


@dataclass
class _OpenTrade:
    """In-flight trade during simulation."""
    symbol: str
    direction: str        # "buy" or "sell"
    entry_bar: int        # bar index of entry
    entry_time: str
    entry_price: float
    stop_loss: float
    atr: float
    volume: float         # remaining volume
    initial_volume: float
    tier1_done: bool = False
    tier2_done: bool = False
    peak_price: float = 0.0  # for trailing stop
    partial_pnl: float = 0.0  # accumulated PnL from tier partial closes


def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 period: int = ATR_PERIOD) -> np.ndarray:
    """Compute ATR series. Returns NaN for first `period` bars."""
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _simple_signal(closes: np.ndarray, i: int, lookback: int = 20) -> Optional[str]:
    """
    Simple momentum signal for backtesting.

    Uses a 20-bar moving average crossover:
    - Price above MA and rising → "buy"
    - Price below MA and falling → "sell"
    - Otherwise → None (no trade)
    """
    if i < lookback:
        return None
    ma = np.mean(closes[i - lookback + 1: i + 1])
    if closes[i] > ma and closes[i] > closes[i - 1]:
        return "buy"
    elif closes[i] < ma and closes[i] < closes[i - 1]:
        return "sell"
    return None


def run_backtest(
    symbol: str,
    ohlcv: pd.DataFrame,
    initial_equity: float = 10000.0,
    mode: str = "simple",
    d1_ohlcv: pd.DataFrame = None,
    w1_ohlcv: pd.DataFrame = None,
    h1_ohlcv: pd.DataFrame = None,
    **kwargs,
) -> tuple[list[dict], list[dict]]:
    """
    Walk-forward simulation over historical bars.

    Pure function — deterministic, no randomness, no external state.
    Entry at next-bar-open after signal. ATR-based stops with 3-tier exit.

    Args:
        symbol:         Trading symbol
        ohlcv:          DataFrame with columns [open, high, low, close, volume]
                        indexed by bar_timestamp (datetime)
        initial_equity: Starting account balance
        mode:           "simple" (MA crossover) or "full" (HMM+LSTM pipeline)
        d1_ohlcv:       Optional D1 data for full mode (multi-TF + regime)
        w1_ohlcv:       Optional W1 data for full mode (multi-TF features)

    Returns:
        (equity_curve, trades) — both as lists of dicts.
    """
    if mode == "full":
        from scripts.backtest_full import run_backtest_full
        # Forward any sweep-override kwargs + the model bake-off cell selectors
        _sweep_kw = {}
        for _k in ("hmm_weight_override", "signal_threshold_override",
                    "long_only_symbols_override", "friction_override",
                    "primary", "variant", "trend_mode"):
            if _k in kwargs:
                _sweep_kw[_k] = kwargs[_k]
        return run_backtest_full(symbol, ohlcv, initial_equity,
                                d1_ohlcv=d1_ohlcv, w1_ohlcv=w1_ohlcv,
                                h1_ohlcv=h1_ohlcv, **_sweep_kw)

    if len(ohlcv) < ATR_PERIOD + 2:
        return [], []

    opens = ohlcv["open"].values.astype(float)
    highs = ohlcv["high"].values.astype(float)
    lows = ohlcv["low"].values.astype(float)
    closes = ohlcv["close"].values.astype(float)
    timestamps = [str(ts) for ts in ohlcv.index]

    atr_series = _compute_atr(highs, lows, closes)
    n = len(closes)

    equity = initial_equity
    peak_equity = initial_equity
    open_trade: Optional[_OpenTrade] = None
    pending_signal: Optional[str] = None

    equity_curve: list[dict] = []
    completed_trades: list[dict] = []

    for i in range(n):
        # --- Execute pending entry at this bar's open ---
        if pending_signal is not None and open_trade is None:
            atr_val = atr_series[i - 1] if i > 0 and not np.isnan(atr_series[i - 1]) else None
            if atr_val is not None and atr_val > 0:
                entry_price = opens[i]
                sl_dist = atr_val * ATR_SL_MULT
                risk_amount = equity * RISK_PER_TRADE_PCT / 100.0

                if pending_signal == "buy":
                    stop_loss = entry_price - sl_dist
                else:
                    stop_loss = entry_price + sl_dist

                volume = risk_amount / sl_dist if sl_dist > 0 else 0.0
                if volume > 0:
                    open_trade = _OpenTrade(
                        symbol=symbol,
                        direction=pending_signal,
                        entry_bar=i,
                        entry_time=timestamps[i],
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        atr=atr_val,
                        volume=volume,
                        initial_volume=volume,
                        peak_price=entry_price,
                    )
            pending_signal = None

        # --- Check exits for open trade ---
        if open_trade is not None:
            t = open_trade
            is_buy = t.direction == "buy"
            bar_high = highs[i]
            bar_low = lows[i]
            exit_price = None
            exit_reason = None

            # Update peak for trailing stop
            if is_buy:
                t.peak_price = max(t.peak_price, bar_high)
            else:
                t.peak_price = min(t.peak_price, bar_low) if t.peak_price > 0 else bar_low

            # Stop loss hit
            if is_buy and bar_low <= t.stop_loss:
                exit_price = t.stop_loss
                exit_reason = "sl"
            elif not is_buy and bar_high >= t.stop_loss:
                exit_price = t.stop_loss
                exit_reason = "sl"

            # Tier exits (check before SL in case both happen on same bar)
            r_dist = abs(t.entry_price - t.stop_loss)
            if r_dist > 0 and exit_reason is None:
                if is_buy:
                    current_r = (bar_high - t.entry_price) / r_dist
                else:
                    current_r = (t.entry_price - bar_low) / r_dist

                if not t.tier1_done and current_r >= TIER1_R:
                    t.tier1_done = True
                    partial_vol = t.initial_volume * TIER1_FRAC
                    # Book partial profit: close 33% at +1R
                    t.partial_pnl += partial_vol * r_dist * TIER1_R
                    t.volume -= partial_vol
                    # Move stop to breakeven
                    t.stop_loss = t.entry_price

                if not t.tier2_done and current_r >= TIER2_R:
                    t.tier2_done = True
                    partial_vol = t.initial_volume * TIER2_FRAC
                    # Book partial profit: close 33% at +2R
                    t.partial_pnl += partial_vol * r_dist * TIER2_R
                    t.volume -= partial_vol

                # Trailing stop after tier2
                if t.tier2_done:
                    trail_dist = t.atr * ATR_SL_MULT
                    if is_buy:
                        trail_stop = t.peak_price - trail_dist
                        t.stop_loss = max(t.stop_loss, trail_stop)
                        if bar_low <= t.stop_loss:
                            exit_price = t.stop_loss
                            exit_reason = "trail"
                    else:
                        trail_stop = t.peak_price + trail_dist
                        t.stop_loss = min(t.stop_loss, trail_stop)
                        if bar_high >= t.stop_loss:
                            exit_price = t.stop_loss
                            exit_reason = "trail"

            # Full exit
            if exit_price is not None:
                # Remaining position PnL + accumulated partial close profits
                if is_buy:
                    remaining_pnl = (exit_price - t.entry_price) * t.volume
                else:
                    remaining_pnl = (t.entry_price - exit_price) * t.volume

                pnl = remaining_pnl + t.partial_pnl
                risk_amount = r_dist * t.initial_volume if r_dist > 0 else 1.0
                r_multiple = pnl / risk_amount if risk_amount > 0 else 0.0
                equity += pnl

                completed_trades.append({
                    "symbol": symbol,
                    "direction": t.direction,
                    "entry_time": t.entry_time,
                    "exit_time": timestamps[i],
                    "entry_price": round(t.entry_price, 5),
                    "exit_price": round(exit_price, 5),
                    "pnl": round(pnl, 2),
                    "r_multiple": round(r_multiple, 4),
                    "exit_reason": exit_reason,
                })
                open_trade = None

        # --- Generate signal for next bar ---
        if open_trade is None and not np.isnan(atr_series[i]) if i < n else False:
            sig = _simple_signal(closes, i)
            if sig is not None:
                pending_signal = sig

        # Also generate signal when no open trade (fix condition above)
        if open_trade is None and i < n and not np.isnan(atr_series[i]):
            sig = _simple_signal(closes, i)
            if sig is not None:
                pending_signal = sig

        # --- Record equity ---
        peak_equity = max(peak_equity, equity)
        dd_pct = ((peak_equity - equity) / peak_equity * 100) if peak_equity > 0 else 0.0
        equity_curve.append({
            "bar_timestamp": timestamps[i],
            "equity": round(equity, 2),
            "drawdown_pct": round(dd_pct, 4),
        })

    return equity_curve, completed_trades


def compute_summary(
    equity_curve: list[dict],
    trades: list[dict],
) -> dict:
    """Compute summary statistics from backtest results."""
    n = len(trades)
    if n == 0:
        return {
            "total_trades": 0, "win_rate": 0.0, "net_pnl": 0.0,
            "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "profit_factor": 0.0,
            "calmar_ratio": 0.0,
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate = len(wins) / n if n > 0 else 0.0
    net_pnl = sum(pnls)
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    max_dd = max((e["drawdown_pct"] for e in equity_curve), default=0.0)

    # Daily Sharpe from equity curve
    if len(equity_curve) > 1:
        equities = np.array([e["equity"] for e in equity_curve])
        returns = np.diff(equities) / equities[:-1]
        sharpe = float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0

    # Calmar = CAGR / |Max DD%|. Both expressed as percentages so the
    # ratio is unit-free. Undefined when DD < 0.5% (one drawdown spike
    # dominates and blows the ratio up); we clamp to 0.0 in that case —
    # the UI can display "—" when Calmar reads zero if it wants to be
    # explicit. See BACKLOG A-3 for the rationale.
    calmar = 0.0
    if len(equity_curve) >= 2 and max_dd >= 0.5:
        try:
            ts_key = "bar_timestamp" if "bar_timestamp" in equity_curve[0] else "timestamp"
            first_ts = pd.Timestamp(equity_curve[0][ts_key])
            last_ts = pd.Timestamp(equity_curve[-1][ts_key])
            years = (last_ts - first_ts).total_seconds() / (365.25 * 86400)
            first_eq = float(equity_curve[0]["equity"])
            last_eq = float(equity_curve[-1]["equity"])
            if years > 0 and first_eq > 0 and last_eq > 0:
                cagr_pct = ((last_eq / first_eq) ** (1.0 / years) - 1.0) * 100.0
                calmar = cagr_pct / max_dd
        except Exception:
            calmar = 0.0

    return {
        "total_trades": n,
        "win_rate": round(win_rate, 4),
        "net_pnl": round(net_pnl, 2),
        "max_drawdown_pct": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 6),
        "profit_factor": round(profit_factor, 4),
        "calmar_ratio": round(calmar, 3),
    }


async def run_backtest_async(
    run_id: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    initial_equity: float,
    data_store,
    mode: str = "simple",
) -> None:
    """
    API entry point: fetch data from DB, run simulation, persist results.

    Called from BackgroundTasks in the backtest route.
    """
    try:
        await data_store.update_backtest_run(run_id, {"status": "running"})

        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        ohlcv = await data_store.get_ohlcv_range(symbol, timeframe, start_dt, end_dt)

        if ohlcv.empty or len(ohlcv) < ATR_PERIOD + 2:
            await data_store.update_backtest_run(run_id, {
                "status": "failed",
                "error_message": f"Insufficient data: {len(ohlcv)} bars (need {ATR_PERIOD + 2}+)",
                "finished_at": datetime.utcnow().isoformat(),
            })
            return

        equity_curve, trades = run_backtest(symbol, ohlcv, initial_equity, mode=mode)
        summary = compute_summary(equity_curve, trades)

        # Persist equity curve
        eq_rows = [{"run_id": run_id, **e} for e in equity_curve]
        await data_store.bulk_insert_backtest_equity(eq_rows)

        # Persist trades
        tr_rows = [{"run_id": run_id, **t} for t in trades]
        await data_store.bulk_insert_backtest_trades(tr_rows)

        # Update run with summary
        await data_store.update_backtest_run(run_id, {
            "status": "done",
            "finished_at": datetime.utcnow().isoformat(),
            **summary,
        })
        logger.info("Backtest %s completed: %d trades, PnL=%.2f",
                     run_id, summary["total_trades"], summary["net_pnl"])

    except Exception as exc:
        logger.exception("Backtest %s failed", run_id)
        try:
            await data_store.update_backtest_run(run_id, {
                "status": "failed",
                "error_message": str(exc),
                "finished_at": datetime.utcnow().isoformat(),
            })
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI entry point (requires MT5, heavy ML imports)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest the trading strategy")
    parser.add_argument("--symbols", nargs="+", default=["XAUUSD", "BTCUSD"])
    parser.add_argument("--start", default="2022-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--output-dir", default="data/logs")
    parser.add_argument("--mode", choices=["simple", "full"], default="full",
                        help="simple = MA crossover, full = HMM+LSTM pipeline")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip persisting to backtest_runs/trades/equity DB tables.")
    parser.add_argument(
        "--primary", choices=["lstm", "gbm"], default="lstm",
        help="Primary predictor for full-mode backtest. 'lstm' = production "
             "path (default). 'gbm' = the model bake-off GBM-as-primary cell."
    )
    parser.add_argument(
        "--variant", choices=["prod", "default", "tuned"], default="prod",
        help="Artifact variant for full-mode backtest. 'prod' = unsuffixed "
             "live artifact. 'default'/'tuned' = the model bake-off "
             "{kind}_{symbol}_{variant}.{ext}. Has no effect in --mode simple."
    )
    parser.add_argument("--no-friction", action="store_true",
                        help="Run frictionless (no slippage, no commission). "
                             "Used to A/B-compare against the realistic "
                             "defaults in DEFAULT_FRICTION.")
    parser.add_argument("--trend-mode", action="store_true",
                        help="Enable E-7 trend-mode logic (widens TP by "
                             "tp_r_multiplier and disables time-exit when "
                             "the 3-filter AND-gate fires: HMM persistence + "
                             "ADX(14) > 25 with directional agreement + "
                             "Kaufman ER(20) > 0.30). Off by default; use to "
                             "A/B against the baseline. Spec: "
                             "docs/superpowers/specs/2026-04-26-e7-trend-mode-design.md")
    return parser.parse_args()


async def _persist_cli_run(
    symbol: str, timeframe: str, start_date: str, end_date: str,
    mode: str, equity_curve: list[dict], trades: list[dict], summary: dict,
) -> str:
    """
    Persist a CLI backtest run to the DB (backtest_runs + equity + trades).
    Returns the new run_id. Uses a short-lived DataStore connection.
    """
    import uuid as _uuid
    from src.data_pipeline.data_store import DataStore

    ds = DataStore()
    await ds.connect()
    try:
        run_id = str(_uuid.uuid4())
        # calmar_ratio is computed but not persisted (no DB column yet —
        # derive on read from net_pnl / max_drawdown_pct / date span).
        db_summary = {k: v for k, v in summary.items() if k != "calmar_ratio"}
        await ds.create_backtest_run({
            "id": run_id,
            "status": "done",
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "run_mode": mode,
            "finished_at": end_date,
            **db_summary,
        })
        await ds.bulk_insert_backtest_equity(
            [{"run_id": run_id, **e} for e in equity_curve]
        )
        # R-1 friction work added a 'commission' column to each trade row
        # that the BacktestTrade ORM doesn't accept yet. Strip it here —
        # same workaround as scripts/ingest_backtest_csvs.py. Safe to
        # remove once the schema gets a commission column.
        # E-7 Task 15 added 'trend_pnl_delta' + 'was_in_trend_mode_at_close'
        # for the trend-mode A/B diagnostic CSVs. ORM doesn't have those
        # columns either; strip the same way until the schema catches up.
        _UNSUPPORTED_COLS = {"commission", "trend_pnl_delta", "was_in_trend_mode_at_close"}
        await ds.bulk_insert_backtest_trades(
            [{"run_id": run_id, **{k: v for k, v in t.items() if k not in _UNSUPPORTED_COLS}}
             for t in trades]
        )
        return run_id
    finally:
        await ds.close()


async def _main_async(args) -> None:
    """Run the CLI backtest pipeline using DB-only OHLCV reads.

    Refactored 2026-04-26 (T-9) to be MT5-free. Previously connected via
    the MT5 connector + sync historical reader, which repointed the shared
    MT5 terminal binding to ``MT5_LOGIN``. Even with the snapshot/restore
    safety net, the in-flight window polluted prod's heartbeat. The
    structural fix mirrors the training-script pattern: construct
    ``MT5DataFeed(connector=None, data_store=...)`` and read via
    ``feed.get_historical_db_only(...)``. See
    ``memory/feedback_dev_mt5_steals_prod_terminal.md``.
    """
    from src.data_pipeline.data_store import DataStore
    from src.data_pipeline.mt5_feed import MT5DataFeed

    data_store = DataStore()
    await data_store.connect()
    logger.info("DataStore connected — OHLCV reads via DB (MT5-free)")

    feed = MT5DataFeed(connector=None, data_store=data_store)

    try:
        for symbol in args.symbols:
            logger.info("Running %s backtest for %s: %s → %s",
                         args.mode.upper(), symbol, args.start, args.end)

            start_dt = datetime.strptime(args.start, "%Y-%m-%d")
            end_dt = datetime.strptime(args.end, "%Y-%m-%d")
            ohlcv = await feed.get_historical_db_only(
                symbol, "H4", start_date=start_dt,
            )
            # get_historical_db_only has no end_date param — clip after fetch.
            if ohlcv is not None and not ohlcv.empty:
                ohlcv = ohlcv[ohlcv.index <= end_dt]

            # Fetch D1/W1/H1 for full mode (multi-TF + regime + H1 execution)
            d1_ohlcv = None
            w1_ohlcv = None
            h1_ohlcv = None
            if args.mode == "full":
                d1_ohlcv = await feed.get_historical_db_only(symbol, "D1", bars=5000)
                w1_ohlcv = await feed.get_historical_db_only(symbol, "W1", bars=1000)
                h1_ohlcv = await feed.get_historical_db_only(
                    symbol, "H1", start_date=start_dt,
                )
                # Honor --end for D1/W1/H1 too
                if d1_ohlcv is not None and not d1_ohlcv.empty:
                    d1_ohlcv = d1_ohlcv[d1_ohlcv.index <= end_dt]
                if w1_ohlcv is not None and not w1_ohlcv.empty:
                    w1_ohlcv = w1_ohlcv[w1_ohlcv.index <= end_dt]
                if h1_ohlcv is not None and not h1_ohlcv.empty:
                    h1_ohlcv = h1_ohlcv[h1_ohlcv.index <= end_dt]

            _bt_kw = {}
            if args.no_friction:
                _bt_kw["friction_override"] = {
                    "*": {"slippage_price": 0.0, "commission_per_lot_per_side": 0.0},
                }
            # the model bake-off cell selection — only meaningful in --mode full.
            if args.mode == "full":
                _bt_kw["primary"] = args.primary
                _bt_kw["variant"] = args.variant
                # E-7 trend-mode flag (Phase 2B v1)
                _bt_kw["trend_mode"] = args.trend_mode
            equity_curve, trades = run_backtest(symbol, ohlcv, args.initial_equity,
                                                 mode=args.mode,
                                                 d1_ohlcv=d1_ohlcv,
                                                 w1_ohlcv=w1_ohlcv,
                                                 h1_ohlcv=h1_ohlcv,
                                                 **_bt_kw)
            summary = compute_summary(equity_curve, trades)

            # Save results — CSV (always) + DB (unless --no-db)
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(equity_curve).to_csv(out / f"backtest_equity_{symbol}.csv", index=False)
            pd.DataFrame(trades).to_csv(out / f"backtest_trades_{symbol}.csv", index=False)

            if not args.no_db:
                try:
                    run_id = await _persist_cli_run(
                        symbol=symbol,
                        timeframe="H4",
                        start_date=args.start,
                        end_date=args.end,
                        mode=args.mode,
                        equity_curve=equity_curve,
                        trades=trades,
                        summary=summary,
                    )
                    logger.info("  persisted to DB: run_id=%s", run_id)
                except Exception as exc:
                    logger.warning("DB persistence failed (CSV already saved): %s", exc)

            logger.info("\n=== %s Backtest Results ===", symbol)
            for k, v in summary.items():
                logger.info("  %s: %s", k, v)
    finally:
        await data_store.close()


def main():
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args = parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
