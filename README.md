# Cortex

An algorithmic trading bot for **MetaTrader 5**, combining **HMM regime detection**, **LSTM price prediction with Triple-Barrier labels**, **per-symbol model heads** (regression / 3-class softmax), and **multi-layer risk management**.

This public repo is a redacted version — tuned strategy parameters are replaced with placeholder zeros so the bot won't trade anything as-is. The system structure is still visible.

---

## What's in the box

- **Data pipeline** — pulls OHLCV from MT5, caches it in PostgreSQL, computes ~120 features per bar (technical indicators + multi-timeframe + macro data from FRED / Stooq / yfinance / COT / ECB)
- **ML models** — a Hidden Markov Model that classifies the current regime (Crash / Bear / Neutral / Bull / Euphoria) on daily bars, and an LSTM that predicts whether a hypothetical entry would hit take-profit before stop-loss within N bars (Triple-Barrier labels)
- **Signal combination** — weighted blend of the two model outputs, gated by a few rules (direction stability, confidence, regime confluence, news blackout)
- **Strategy router** — picks one of three strategy classes based on volatility rank, each with its own stop-loss formula
- **Risk layers** — position sizer, portfolio manager (max concurrent positions, pyramiding rules), and an independent safety thread with multi-level circuit breakers
- **Web dashboard** — FastAPI backend + React frontend showing signals, positions, equity curve, model state, drift monitoring
- **Operational bits** — nightly Postgres backups, drift monitoring, MLflow experiment tracking, an external watchdog that pings Telegram if the bot's heartbeat goes stale

---

## Architecture

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

## Trading strategy

### Signal fusion

The signal combiner blends two model outputs:

```
combined_score = hmm_weight × regime_score + lstm_weight × lstm_score
```

(weights are placeholders in the public configs; see `config/model_config.yaml.example`)

| Component | Role |
|---|---|
| HMM regime | 5-state Gaussian HMM on daily bars: Crash (-1), Bear, Neutral, Bull, Euphoria (+1). Provides a regime label and a probability. |
| LSTM (Triple Barrier) | Predicts `{-1, 0, +1}` — SL-hits-first / time-out / TP-hits-first. **Per-symbol model heads:** some symbols use 1-output regression, others use 3-class softmax (`P(+1) − P(−1)`). |

A trade fires only if:

1. `abs(combined_score)` exceeds a uniform threshold *(magnitude gate)*
2. The last N bars all agree on direction *(flicker / stability gate)*
3. Per-symbol direction is allowed *(long-only gating for chosen symbols)*
4. Regime doesn't contradict direction *(confluence gate — "must not disagree")*
5. Not in a per-symbol Euphoria block where one is configured
6. No active news blackout for any of this pair's affected central banks
7. Direction-conflict guard passes — no opposite-direction simultaneous holds on the same symbol

### Triple-Barrier LSTM target

A meaningful departure from standard next-bar regression. Instead of predicting log return, the LSTM learns the question the strategy actually asks:

```
Walk forward from each bar up to time_limit bars:
  high crosses TP first  →  label = +1
  low  crosses SL first  →  label = -1
  neither hits in time   →  label =  0
```

The model is trained on the same answer the strategy needs at decision time — no representation gap.

### Exit strategy — Triple Barrier + extras

Each position has **5 independent exits**. First trigger wins:

1. **SL** — vol-tier-aware ATR multiple. Each strategy class (LowVolAggressive / MidVolCautious / HighVolDefensive) has its own stop formula.
2. **TP** — fixed at `tp_r × sl_distance` per the per-pair tuning in `config/settings.yaml`
3. **Time exit** — close after `time_exit_h1_bars` H1 bars regardless of P/L
4. **BE lock** — at `+be_trigger_r` R multiples, move SL to entry (risk-free past that point)
5. **Reversal hard-exit** — close newest leg after N consecutive opposite-direction signals

### News blackout

A pre-news + spike zone `[T-blackout_pre_hours, T+blackout_post_hours]` blocks new entries. Per-pair routing maps each pair to the central banks that affect it (FOMC + ECB + BoJ + BoC + BoE + RBA + RBNZ + SNB depending on the pair's currencies). Schedule lives in `config/economic_calendar.yaml`.

---

## Risk management — three independent layers

Each layer can override the one below it.

### Layer 1 — Position sizing (`src/allocation/position_sizer.py`)

Fixed-fractional sizing with a regime-aware multiplier:

```
risk_usd  = equity × regime_multiplier × per_symbol_risk_pct
lot_size  = risk_usd / (sl_distance × tick_value / tick_size)
```

`regime_multiplier` is read from `HMMRegimeClassifier.MULTIPLIERS`. Two valid shapes (depending on whether the pair is bidirectional or long-only):

- asymmetric long-only ramp  (Bull / Euphoria favored)
- symmetric bidirectional    (mirror around Neutral)

Production multipliers are redacted in this template.

### Layer 2 — Portfolio manager (`src/allocation/portfolio_manager.py`)

- Max positions per symbol (pyramiding) with fractional sizing 100% → 50% → 25%
- Max total concurrent positions across all symbols
- BE-gate on pyramiding — additional entries only after the prior leg reaches `+be_trigger_r`
- Margin caps: total used-margin cap, free-margin reserve floor
- Daily trade cap (sliding 24h window)
- Direction-conflict guard — no opposite-direction simultaneous holds on the same symbol

### Layer 3 — Safety (`src/safety/`)

Independent 30-second polling thread. Never consults the brain; absolute veto:

| Breaker | Trigger | Action |
|:---:|:---|:---|
| Daily soft | Equity below threshold intraday | Halve lot sizes |
| Daily hard | Equity below threshold intraday | Flat all + halt until next UTC day |
| Weekly soft | Equity below threshold on the week | Halve lot sizes |
| Weekly hard | Equity below threshold on the week | Flat all + halt until Monday |
| Peak drawdown | Down threshold from all-time peak | Flat all + halt until manual reset |
| Consecutive SL | N SL hits in a row | Halt for X hours then auto-resume |

Breaker state is **HMAC-signed** and persisted to `data/logs/TRADING_HALTED.flag` — a restart cannot silently clear a trip.

Additional safety infrastructure:

- **Drift monitor** (`src/ml/drift_check.py`) — daily PSI/KS check per symbol against the training-distribution snapshot. Three severity levels (WARN / ALERT / auto-trigger off-cycle retrain) plus an absurd-PSI ceiling that suppresses runaway alerts during real regime shifts.
- **API smoke checks** every 60s on critical endpoints, with route-health invariants that page the operator when latency or status code regresses.
- **PF-drift monitor** — live realized PF vs backtest PF; alerts when the two diverge.
- **Bot-liveness watchdog** — external Scheduled Task reads heartbeat file; pings Telegram if mtime is stale 5+ min. Catches process crashes that bypass Python's `finally` blocks.

---

## Model engineering

A few things that aren't standard for a small bot:

- **Phase A bake-off** — head-to-head LSTM vs GBM with a 2-of-3 decision gate (PF, DD, stability) and a Deflated Sharpe floor. Settled per-symbol model choices.
- **MLflow registry** at `data/mlflow/` — every train run auto-logs params, metrics, artifacts, and dataset fingerprint. Launch the viewer with `mlflow ui --backend-store-uri file:./data/mlflow --port 5000`.
- **Combinatorial Purged Cross-Validation** (`scripts/backtest_cpcv.py`) — N=6, k=2 → 15 folds with a hybrid retrain on the largest contiguous pre-test block per fold. Gives an honest read of OOS-vs-in-sample decay vs walk-forward.
- **Deflated Sharpe + Sharpe stability** computed and surfaced on the dashboard's Backtest detail drawer.
- **Drift-trigger retrain** — auto-fires when PSI breaches the threshold; 24h cooldown.
- **Monthly subprocess retrain** — fires on a schedule. Snapshots current models first, rolls back on training failure.
- **Mixed model heads per-symbol** — `LSTMPricePredictor.predict()` auto-detects regression vs softmax via `fc2.weight.shape[0]`. `src/utils/model_head.py` enforces the head-type contract; refuses any head-shape mismatch with the existing on-disk model.
- **Triple-Barrier label generation + per-symbol PCA** in `src/data_pipeline/feature_engineering.py`.
- **Meta-labeler** (`src/ml/meta_labeler.py`) — secondary classifier on a 22-feature schema (5 base + 17 fundamentals from `feature_store`). Operates in shadow mode by default; can be flipped to active gate via env var.

---

## What's redacted

Replaced with placeholder zeros / blanks throughout the code:

- All numeric strategy parameters (fusion weights, signal thresholds, regime multipliers, ATR stop multipliers, vol-rank cutoffs, breaker trip levels)
- Per-pair tuned configuration (model_head choices, tp_r_multiple, time_exit_h1_bars, be_trigger_r)
- Backtest results and the symbols actually traded
- News blackout schedule
- Decision history and notes

`data/`, `.env`, model `.pt`/`.pkl` files are all excluded by `.gitignore`. The configs in `config/` are `.example` templates only.

The placeholder zeros cause the bot to refuse trade execution until tuned values are supplied.

---

## Tech stack

- Python 3.13, PyTorch, scikit-learn, hmmlearn, LightGBM, MLflow, FastAPI, asyncpg, Pydantic
- PostgreSQL 14+
- React 19 + Vite + TypeScript + Tailwind v4 + lightweight-charts
- MetaTrader 5 (Windows-only)
- pytest

---

## Conventions

- async/await for all I/O, loguru for logging, Pydantic for API, dataclasses internal
- All prices in broker native units — no pip conversion
- Build output goes to `src/api/static/dist/`; FastAPI serves it directly
- Backtest + Models screens are lazy-loaded via `React.lazy` to keep first-paint bundle small
- **True UTC everywhere** — broker-server time (EET/EEST) is converted to UTC at ingest via `_broker_ts_to_utc`. Database `bar_timestamp` and `deal.time` are true UTC; dashboard candles align with TradingView.
- **Account-segmented data** — trades / signals / equity all tagged with `mt5_account`. Dashboard filters by current slot. DB indexes: `ix_trades_account_ts`, `ix_signals_account_ts`, `ix_equity_account_ts`.
- **`mode` is reserved in PostgreSQL** — collides with the `mode()` ordered-set aggregate. Never name a column `mode`. The `BacktestRun.run_mode` rename is the precedent.

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

The placeholder configs will refuse to fire trades.

---

## Notes

A few small but non-obvious things from building this:

- **PSI > 2.0 happens** in real markets during regime shifts. A textbook "auto-retrain on PSI > 0.5" trigger creates infinite-retrain loops. An absurd-PSI ceiling that suppresses the trigger and surfaces a different alert is necessary.
- **MetaTrader 5 timezone handling is a footgun.** The MT5 Python API returns broker-local time (EET/EEST) but everything looks like UTC if you don't double-check. Caused multi-hour drift in features.
- **Backtest engines drift from live engines silently.** Live and backtest both call the same `signal_combiner.combine()` — but the gates applied to the result historically diverged. Schema-parity tests + a doc-drift linter catch this.
- **Process supervision matters more than expected.** Bots that go silent when they crash are common. The external bot-liveness watchdog (separate Scheduled Task that reads heartbeat → Telegrams on stale) catches real outages that in-process error handling misses.

---

## License

No license — all rights reserved.
