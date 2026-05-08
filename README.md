# Cortex

A personal project: an algorithmic trading bot for MetaTrader 5, combining HMM regime detection with LSTM price prediction.

I'm a data analyst with a statistics background. This started as a way to apply ML and statistics to something I cared about, and grew into a more involved system as I kept hitting real problems and trying to fix them. Most of the code was written with heavy AI-assistant support (Claude Code).

This public repo is a **redacted version**. Tuned strategy parameters are replaced with placeholder zeros so the bot won't actually trade anything as-is. The structure of the system is still visible — that's what I want to share.

---

## What's in the box

The system has the usual pieces of a small trading stack:

- **Data pipeline** — pulls OHLCV from MT5, caches it in PostgreSQL, computes ~120 features per bar (technical indicators + multi-timeframe + macro data from FRED / Stooq / yfinance / COT / ECB)
- **ML models** — a Hidden Markov Model that classifies the current regime (Crash / Bear / Neutral / Bull / Euphoria) on daily bars, and an LSTM that predicts whether a hypothetical entry would hit take-profit before stop-loss within N bars (Triple-Barrier labels)
- **Signal combination** — weighted blend of the two model outputs, gated by a few rules (direction stability, confidence, regime confluence, news blackout)
- **Strategy router** — picks one of three strategy classes based on volatility rank, each with its own stop-loss formula
- **Risk layers** — position sizer, portfolio manager (max concurrent positions, pyramiding rules), and an independent safety thread with multi-level circuit breakers
- **Web dashboard** — FastAPI backend + React frontend showing signals, positions, equity curve, model state, drift monitoring
- **Operational bits** — nightly Postgres backups, drift monitoring, MLflow experiment tracking, a watchdog that pings me on Telegram if the bot goes silent

It's been running on a personal demo account.

---

## Architecture (text version)

```text
                  MetaTrader 5
                       │
                       ▼
   data_pipeline (mt5_feed, feature_engineering, feature_store)
                       │
                       ▼
          ┌────────────────────────┐
          │  brain                 │
          │   ├── HMM regime       │
          │   ├── LSTM (TB+PCA)    │
          │   └── signal_combiner  │
          └───────────┬────────────┘
                      ▼
          ┌────────────────────────┐    ┌─────────────────────┐
          │  strategy              │    │  safety             │
          │   ├── orchestrator     │◄───┤   (circuit breakers,│
          │   ├── vol-tier classes │    │    drift monitor,   │
          │   └── exit_manager     │    │    risk monitor)    │
          └───────────┬────────────┘    └─────────────────────┘
                      ▼
          ┌────────────────────────┐
          │  allocation            │
          │   ├── position_sizer   │
          │   └── portfolio_mgr    │
          └───────────┬────────────┘
                      ▼
                 broker (MT5)

   FastAPI (port 8787) ↔ React frontend
```

---

## What's redacted

Replaced with placeholder zeros / blanks throughout the code:

- All numeric strategy parameters (fusion weights, signal thresholds, regime multipliers, ATR stop multipliers, vol-rank cutoffs, breaker trip levels)
- Per-pair tuned configuration
- Backtest results and the actual symbols I trade
- News blackout schedule
- Decision history and notes

`data/`, `.env`, model `.pt`/`.pkl` files are all excluded by `.gitignore`. The configs in `config/` are `.example` templates only.

The placeholder zeros mean the bot will refuse to trade until someone supplies real values. That's intentional.

---

## Tech stack

- Python 3.13, PyTorch, scikit-learn, hmmlearn, LightGBM, MLflow, FastAPI, asyncpg, Pydantic
- PostgreSQL 14+
- React 19 + Vite + TypeScript + Tailwind v4 + lightweight-charts
- MetaTrader 5 (Windows-only)
- pytest

---

## Trying it out

```bash
python -m venv venv
source venv/Scripts/activate          # Git Bash / WSL
pip install -r requirements.txt

# Copy yaml templates and fill in your own values
cp config/settings.yaml.example config/settings.yaml
cp config/model_config.yaml.example config/model_config.yaml
# ... etc

# Set up .env from .env.example

# Run tests
pytest tests/ -v

# Build frontend
cd frontend && npm install && npm run build
```

The placeholder configs won't fire trades. Tuning is on you.

---

## A few things I learned along the way

Mostly recording these for myself, but they might be useful:

- **Drift detection isn't as clean as the textbooks suggest.** PSI > 2.0 happens in real markets during regime shifts, and "auto-retrain on PSI > 0.5" creates retrain loops. Some kind of upper ceiling is needed.
- **MetaTrader 5 timezone handling is a gotcha.** The MT5 Python API returns broker-local time (EET/EEST), not UTC, but it looks like UTC if you don't double-check. I had to retrain everything once I caught that.
- **`mode` is a reserved word in PostgreSQL.** Don't name a column `mode`. I did. Renaming it was annoying.
- **Process supervision matters.** A bot that goes silent when it crashes is more common than I thought. An external watchdog (separate Scheduled Task that reads a heartbeat file and pings me on Telegram) caught real outages that in-process error handling missed.

---

## License

No license — all rights reserved. The repo is here so people can look at it, not reuse it.

---

## Contact

[github.com/rizkiprayoga](https://github.com/rizkiprayoga)
