# Cortex — Algorithmic Trading System

> An end-to-end, autonomous trading system on MetaTrader 5 — built from scratch by a single engineer over ~2 years.
>
> **This is the public engineering portfolio version.** Tuned strategy parameters and proprietary research artifacts are redacted; the architecture, infrastructure, and engineering practice are not.

---

## What this demonstrates

This project shows production-grade engineering work across:

- **Machine learning** — HMM regime classification + LSTM price prediction with Triple-Barrier labels, mixed regression / softmax model heads per-symbol, MLflow registry, Combinatorial Purged Cross-Validation, drift monitoring with PSI/KS thresholds, Deflated Sharpe + Sharpe stability
- **Real-time systems** — async Python trading loop with independent risk-monitoring thread, signed circuit-breakers (HMAC), broker reconnection logic, per-bar feature engineering at sub-second latency
- **Data engineering** — partitioned PostgreSQL feature store (5 external sources × 11 symbols × 25 yr depth ≈ 245k rows), DB-first caching, lookahead-safe joins with per-source release-lag, broker-time → true-UTC normalization
- **Web stack** — FastAPI backend with JWT auth, React 19 + Vite + Tailwind v4 frontend, lightweight-charts integration, SSE streaming, ~25 routes
- **DevOps** — nightly DB backups (GFS retention), Windows Scheduled Tasks + NSSM service, external bot-liveness watchdog, Husky pre-commit hooks, 1000+ test suite with TDD discipline
- **Observability** — JSONL invariants logging, drift-trigger retrain, API smoke checks, PF-drift monitor, dashboard health card, Telegram + email alerting with severity routing

The trading system is currently running in production on a personal account.

---

## High-level architecture

```
                  ┌────────────────────────┐
                  │      MetaTrader 5      │
                  └───────────┬────────────┘
                              │ OHLCV + ticks + orders
                              ▼
   ┌──────────────────────────────────────────────────┐
   │  data_pipeline                                    │
   │   ├── mt5_feed (DB-first cache)                   │
   │   ├── feature_engineering (~120 features/bar)     │
   │   ├── feature_store (5 external sources, FRED /   │
   │   │   Stooq / yfinance / COT / ECB)               │
   │   └── data_store (PostgreSQL async ORM)           │
   └────────────┬─────────────────────────────────────┘
                ▼
   ┌────────────────────────┐  ┌────────────────────────┐
   │  brain                 │  │  ml                    │
   │   ├── HMM regime       │  │   ├── drift_check      │
   │   ├── LSTM (TB+PCA)    │  │   ├── meta_labeler     │
   │   └── signal_combiner  │  │   ├── registry (MLflow)│
   └───────────┬────────────┘  │   └── gate (decision)  │
               ▼               └────────────────────────┘
   ┌────────────────────────┐  ┌────────────────────────┐
   │  strategy              │  │  safety                │
   │   ├── orchestrator     │  │   ├── invariants       │
   │   ├── vol-tier classes │◄─┤   ├── circuit_breaker  │
   │   └── exit_manager     │  │   ├── risk_monitor     │
   └───────────┬────────────┘  │   └── emergency_close  │
               ▼               └────────────────────────┘
   ┌────────────────────────┐
   │  allocation            │
   │   ├── position_sizer   │
   │   └── portfolio_mgr    │
   └───────────┬────────────┘
               ▼
   ┌────────────────────────┐    ┌──────────────────────┐
   │  broker                │    │  api (FastAPI)       │
   │   └── order_manager    │    │   + frontend (React) │
   └────────────────────────┘    └──────────────────────┘
```

Key design rules:

- **Safety thread is independent** of brain → absolute veto authority
- **Stop-loss required on every order** → OrderManager rejects orders without one
- **DB-first caching** → MT5DataFeed checks PostgreSQL before MT5 API
- **True UTC everywhere** → broker-server time (EET/EEST) converted at ingest
- **Account-segmented data** → trades/signals/equity tagged with `mt5_account` for multi-account support
- **HMAC-signed circuit-breaker state** → restart cannot silently clear a trip
- **`mode` is reserved in PostgreSQL** → never use it as a column name (collides with `mode()` aggregate)

---

## What's redacted

This is a public template. The following are intentionally REDACTED and replaced with placeholder values:

- **All numeric strategy parameters** — fusion weights, signal thresholds, regime multipliers, ATR stop multipliers, vol-rank cutoffs, per-pair tuned `(time_exit, tp_r)` config, drawdown clamps, breaker trip levels
- **Per-pair tuned configuration** — symbol-specific `model_head` choices, `tp_r_multiple`, `time_exit_h1_bars`, `be_trigger_r`
- **Backtest results** — specific PF, DD, CAGR numbers; sprint-by-sprint A/B verdicts
- **Trading universe** — actual list of pairs traded
- **News blackout schedule** — central-bank meeting dates per pair
- **Strategy decision history** — sprint roadmaps, audit notes, daily checkup logs
- **Live trade artifacts** — every value under `data/` (already excluded by `.gitignore`)
- **Operational state** — broker name, account numbers, MT5 paths

The structure of every file is preserved; the values aren't. A reader can see how the system is built, but not what makes it work.

---

## Tech stack

- **Python 3.13** (PyTorch 2.11 + CUDA 12.8, scikit-learn, hmmlearn, LightGBM, MLflow, FastAPI, asyncpg, Pydantic, loguru)
- **PostgreSQL 14+** (asyncpg, partitioned tables, JSONB, custom indexes for account-segmented queries)
- **React 19 + Vite 6 + TypeScript 5 + Tailwind v4** (lightweight-charts, recharts, react-router 7)
- **MetaTrader 5** (Windows-only broker integration via official Python package)
- **Pytest** (1000+ tests; TDD discipline; pytest-asyncio for async coverage)
- **MLflow** (file-based tracking backend, experiments, dataset fingerprinting)
- **Optuna** (hyperparameter tuning, SQLite study storage)
- **Husky** (planned — pre-commit frontend auto-build)

---

## Run with placeholder values

The code is functional but **trades nothing** with the placeholder values shipped here. To run a sanity build:

```bash
# 1. Create venv + install
python -m venv venv
source venv/Scripts/activate          # Git Bash / WSL
pip install -r requirements.txt

# 2. Create config/*.yaml from .example templates
cp config/settings.yaml.example config/settings.yaml
cp config/model_config.yaml.example config/model_config.yaml
cp config/mt5_config.yaml.example config/mt5_config.yaml
cp config/tuning_spaces.yaml.example config/tuning_spaces.yaml

# 3. Fill in tuned values in those yaml files (REDACTED — you'd tune these
#    against your own backtest harness; the placeholder zeros block trades)

# 4. Set up .env from .env.example (POSTGRES_*, MT5_*, FRED_API_KEY,
#    DASHBOARD_*, TELEGRAM_*, ALERT_EMAIL_*)

# 5. Run the test suite
pytest tests/ -v

# 6. Build the frontend
cd frontend && npm install && npm run build
```

The full operator runbook (preflight + first-run training + monthly retrain) lives in the private repo.

---

## Selected components worth a closer look

If you're skimming for engineering quality, these files are good places to start:

- [`src/safety/invariants.py`](src/safety/invariants.py) — JSONL-logged invariants with severity routing, dedup window, and registry pattern
- [`src/safety/circuit_breaker.py`](src/safety/circuit_breaker.py) — multi-level breakers with HMAC-signed halt flag (restart cannot clear silently)
- [`src/data_pipeline/mt5_feed.py`](src/data_pipeline/mt5_feed.py) — DB-first OHLCV caching + broker-time → UTC normalization
- [`src/data_pipeline/feature_engineering.py`](src/data_pipeline/feature_engineering.py) — ~120-feature engineering pipeline with lookahead-safe `feature_store` joins
- [`src/brain/deep_learning/lstm_model.py`](src/brain/deep_learning/lstm_model.py) — Triple-Barrier label generation + per-symbol PCA + mixed regression/softmax heads
- [`src/api/routes/`](src/api/routes/) — FastAPI route design across live, history, models, system, accounts
- [`src/ml/drift_check.py`](src/ml/drift_check.py) — daily PSI/KS check with auto-retrain trigger + cooldown logic
- [`src/utils/model_head.py`](src/utils/model_head.py) — per-symbol model contract enforcement (head-shape mismatch → load refused)
- [`frontend/src/screens/`](frontend/src/screens/) — React 19 dashboard screens (Overview, Backtest, Models, History, Signals, System, Config)
- [`tests/`](tests/) — 1000+ tests including async coverage, schema parity tests, drift fixture tests, MT5-free backtest tests

---

## What I learned building this

A handful of lessons that don't usually show up in tutorials:

1. **Drift monitoring is a different problem than people write about.** PSI > 2.0 happens in real markets during regime shifts; the textbook "auto-retrain on PSI > 0.5" trigger creates an infinite-retrain loop. You need a ceiling above which you suppress the trigger and surface a different alert.
2. **Backtest engines drift from live engines silently.** Live and backtest both call the same `signal_combiner.combine()` — but the *gates* applied to the result historically diverged. We built schema-parity tests + a doc-drift linter to catch this.
3. **Model validation is the actual hard part.** We have 5+ different validation approaches (single train/test split, walk-forward, CPCV with purging, Deflated Sharpe, OOS regime stratification) and they tell different stories. None is sufficient on its own.
4. **MetaTrader 5 timezone handling is a footgun.** The MT5 Python API returns broker-local time (EET/EEST) but everything looks like UTC if you don't double-check. Caused a multi-hour drift in our HMM features that retroactively required retraining all 11 models.
5. **PostgreSQL `mode` is a reserved aggregate.** Don't name a column `mode`. We did. We renamed it `run_mode`. Painful.
6. **Process supervision matters more than I expected.** Solo bots that go down silently are common. The external bot-liveness watchdog (separate Scheduled Task that reads heartbeat → Telegrams on stale) caught real outages that in-process error handling missed.

---

## Why this isn't open source

The trading strategy embedded here is what generates the alpha — and there's only so much alpha in retail FX. Public code with tuned values would compress what edge there is by attracting copycats trading the same instruments.

The engineering practice (the *how*), on the other hand, is general-purpose and worth sharing. That's what this repo demonstrates.

---

## License

No license — all rights reserved. Source visible to viewers but not licensed for reuse. Contact the author for any usage question.

---

## Contact

Author: Rizki Prayoga · [github.com/rizkiprayoga](https://github.com/rizkiprayoga)
