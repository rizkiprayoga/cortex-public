# Model registry (MLflow)

Every retrain of `train_hmm.py` or `train_deep_learning.py` logs a run to a local MLflow file-backed registry at `data/mlflow/` (git-ignored).

## Viewing runs

### Web UI (recommended)

```bash
source venv/Scripts/activate
mlflow ui --backend-store-uri file:./data/mlflow --port 5000
```

Open <http://127.0.0.1:5000> — browse experiments `hmm_regime` and `lstm_price`, filter/sort by any param or metric, download artifacts.

### CLI

```bash
# List recent runs in lstm_price experiment
mlflow runs list --experiment-name lstm_price
```

### Python

```python
import mlflow
mlflow.set_tracking_uri("file:./data/mlflow")
client = mlflow.MlflowClient()
for run in client.search_runs(
    experiment_ids=[client.get_experiment_by_name("lstm_price").experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=10,
):
    print(run.info.run_id, run.data.tags.get("symbol"),
          run.data.metrics.get("best_val_loss"))
```

## What gets logged

### HMM runs (`experiment: hmm_regime`)

- **Params:** symbol, n_components, n_init, bars_requested, bars_actual, end_date, dataset_fingerprint
- **Metrics:** regime_0_weight ... regime_N_weight (stationary distribution), n_training_bars
- **Tags:** symbol, model_head=hmm, training_script
- **Artifacts:** `hmm_{symbol}.pkl`, `dataset/dataset_fingerprint.json`

### LSTM runs (`experiment: lstm_price`)

- **Params:** symbol, bars, epochs, batch_size, lr, patience, pca_components, triple_barrier, class_weight, softmax, no_regime, no_multi_tf, end_date, dataset_fingerprint
- **Metrics:** best_val_loss, directional_accuracy, epochs_trained, n_training_bars; per-epoch train_loss and val_loss curves
- **Tags:** symbol, model_head=(regression|softmax), training_script
- **Artifacts:** `lstm_{symbol}.pt`, `lstm_{symbol}.pca.pkl`, `lstm_scaler_{symbol}.pkl`, `dataset/dataset_fingerprint.json`

## Dataset fingerprint

SHA-256 over `(symbol, timeframe, first_bar_ts, last_bar_ts, close_prices)`. Two runs with the same fingerprint saw identical training data; any change (window shift, new bars, symbol, timeframe) produces a different hash. Useful for answering "was this model trained on the same data as that one?" without diffing huge CSVs.

## Coexistence with `scripts/model_snapshot.py`

Model snapshots and MLflow serve different purposes:

- **Snapshots** (`data/model_snapshots/<label>/`) are **rollback points** — restore quickly if a retrain goes sideways.
- **MLflow runs** (`data/mlflow/`) are **tracking records** — compare configs, metrics, and provenance across retrains.

Both are written on every training invocation. Neither replaces the other.

## Tracking URI override

Set `MLFLOW_TRACKING_URI` env var to redirect to a non-default location, including a remote MLflow server if you ever run one:

```bash
export MLFLOW_TRACKING_URI="file:/path/to/external/mlflow"
# or
export MLFLOW_TRACKING_URI="http://mlflow-server.local:5000"
```

## Comparing two runs (model_bench)

`scripts/model_bench.py` (T-8 piece 2) compares two training invocations side-by-side and applies a 2-of-3 decision gate for promoting a candidate model.

### Identifying runs by timestamp

When `train_deep_learning.py` logs a run, it sets `run_name={symbol}-{YYYYMMDDThhmmss}`. All symbols from the same invocation share the timestamp suffix, so you can reference a whole invocation with one value. Find timestamps in the MLflow UI (the "Run Name" column) or via `mlflow runs list --experiment-name lstm_price`.

### Running a bench

```bash
source venv/Scripts/activate
python scripts/model_bench.py \
    --current 20260419T151316 \
    --candidate 20260501T030000 \
    --symbols XAUUSD EURUSD USDJPY USDCAD ETHUSD
```

Optional:

- `--skip-portfolio-sim` — per-symbol backtests only, gate falls back to summed metrics. Faster for iteration.
- `--report-dir <path>` — override the markdown-report output dir (default `data/logs/model_bench/`).
- `CORTEX_BENCH_START=YYYY-MM-DD CORTEX_BENCH_END=YYYY-MM-DD` — override the backtest window (default: 2022-01-01 → 2024-01-01).

### What the bench does

1. Resolves each timestamp to a set of `(symbol → run_id)` from MLflow.
2. Snapshots current `data/models/` via `scripts/model_snapshot.py` for safe restore.
3. For each side (current, then candidate):
    - Downloads that run's LSTM artifacts.
    - Copies them into `data/models/` (HMM not swapped — regime detection stays at production).
    - Runs `scripts/backtest.py --mode full` per symbol.
    - Runs `scripts/portfolio_simulator.py` (unless `--skip-portfolio-sim`).
4. Restores the snapshot.
5. Applies the 2-of-3 decision gate.
6. Prints a side-by-side table + verdict; writes a markdown report.

### Decision gate (2-of-3)

- **Portfolio PF** — candidate must be ≥ current (higher is better).
- **Portfolio DD %** — candidate must be ≤ current (lower is better).
- **Trade-count stability** — candidate must be within ±20% of current.

Ties count as "no regression" and satisfy the criterion. Gate passes when ≥2 of 3 are met.

### Safety

- Snapshot before any swap (`scripts/model_snapshot.py cmd_save`).
- Restore on exit via both `try/finally` and `atexit` — even Ctrl-C leaves your live models intact.
- HMM artifacts are never touched.
- The bot should be **stopped during a bench** because `scripts/backtest.py` and the live bot both talk to MT5 as a single terminal session.

### Exit codes

- `0` — bench complete, gate verdict PASS
- `2` — bench complete, gate verdict FAIL
- non-zero other — bench aborted (run resolution failed, subprocess error)

## Future-proofing: SQLite backend

MLflow 3.11 (Feb 2026) deprecated the file-based backend in favor of SQLite or a full database. The file backend still works (a `FutureWarning` is emitted at runtime) but is slated for removal in a future major version.

To migrate when ready:

```bash
# Stop any running MLflow UI
# Back up current runs (one-time):
mv data/mlflow data/mlflow.filestore-backup

# Switch to SQLite:
export MLFLOW_TRACKING_URI="sqlite:///data/mlflow.db"
```

See [MLflow's migration guide](https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store) for moving existing runs across backends. For Cortex, given the low run volume (~5-10 runs per monthly retrain), a clean start after an archive is fine; no migration pressure.
