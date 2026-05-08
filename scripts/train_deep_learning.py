"""
train_deep_learning.py — Standalone LSTM Training Script

Trains and saves an LSTM price predictor for each symbol using
multi-timeframe H4/D1/W1 data + HMM regime features from MT5.

Usage:
    python scripts/train_deep_learning.py
    python scripts/train_deep_learning.py --symbols XAUUSD --bars 0 --epochs 150

Steps:
    1. Connect to MT5
    2. Fetch H4 + D1 + W1 historical OHLCV for each symbol
    3. Compute multi-TF features (H4 primary + D1/W1 context)
    4. Inject HMM regime as 6 extra LSTM features (one-hot + probability)
    5. Add calendar features + zero-fill fundamental placeholders
    6. Create sliding-window sequences (length=60 bars)
    7. Split into train/val/test
    8. Train LSTM with early stopping
    9. Print test set metrics (RMSE, directional accuracy)
    10. Save model weights + feature scaler + manifest to data/models/
    11. Disconnect from MT5

Requires: HMM model already trained (run train_hmm.py first).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.mt5_feed import MT5DataFeed
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.brain.deep_learning.lstm_model import LSTMPricePredictor
from src.utils.model_head import (
    HeadMismatchError,
    assert_head_matches_existing,
    resolve_softmax_for_symbol,
)
from src.brain.hmm_regime import HMMRegimeClassifier

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM price predictor")
    parser.add_argument("--symbols", nargs="+",
                        default=["XAUUSD", "EURUSD", "USDJPY", "USDCAD"])
    parser.add_argument("--bars", type=int, default=0,
                        help="H4 bars of history (0 = all available)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--no-regime", action="store_true",
                        help="Skip HMM regime feature injection")
    parser.add_argument("--no-multi-tf", action="store_true",
                        help="Skip multi-timeframe features (H4 only)")
    parser.add_argument("--no-snapshot", action="store_true",
                        help="Skip auto-snapshot before training")
    parser.add_argument("--snapshot-label", default=None,
                        help="Override auto-snapshot label")
    parser.add_argument("--pca-components", type=int, default=None,
                        help="Apply PCA reduction to N components (e.g. 25). "
                             "Default: no PCA. Recommended: 25-30.")
    parser.add_argument("--end-date", default=None,
                        help="Truncate training data at this date (YYYY-MM-DD). "
                             "Used for walk-forward OOS validation.")
    parser.add_argument("--start-date", default=None,
                        help="Drop training data BEFORE this date (YYYY-MM-DD). "
                             "Combined with --end-date lets CPCV specify a "
                             "closed window.")
    parser.add_argument("--triple-barrier", action="store_true",
                        help="Use Triple Barrier labels ({-1, 0, +1} based on "
                             "whether TP hits first, SL hits first, or time "
                             "runs out) instead of next-bar log return.")
    parser.add_argument("--class-weight", action="store_true",
                        help="Apply inverse-class-frequency sample weights to "
                             "Triple Barrier training. Prevents the LSTM from "
                             "collapsing to the majority class when the {-1,0,+1} "
                             "distribution is imbalanced (observed: EURUSD live "
                             "output stuck at -0.7, XAU at +0.78 before this "
                             "flag was added). Weights capped at [0.5, 3.0] to "
                             "avoid a rare class dominating.")
    parser.add_argument("--no-softmax", action="store_true",
                        help="Force regression (1-output) head for every "
                             "symbol in this run. Overrides settings.yaml "
                             "per-symbol model_head. Paired with --softmax: "
                             "latest flag wins.")
    parser.add_argument("--allow-head-change", action="store_true",
                        help="Permit retraining a symbol with a head shape "
                             "different from its existing on-disk model. "
                             "Default: refuse the mismatch (guards against "
                             "silent architecture flips like the May-1 bug).")
    parser.add_argument("--softmax", action="store_true",
                        help="3-class softmax head with cross-entropy loss over "
                             "{-1, 0, +1} Triple-Barrier labels (instead of the "
                             "1-output regression + MSE default). Inference returns "
                             "P(+1) - P(-1) so the [-1, +1] scale stays compatible "
                             "with the combined_score arithmetic. Requires "
                             "--triple-barrier.")
    parser.add_argument(
        "--no-focal-loss", action="store_true",
        help="Disable focal loss for softmax heads (default: enabled, gamma=2.0).",
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Focal loss gamma (down-weighting strength). Ignored if "
             "--no-focal-loss.",
    )

    # --- Phase A: explicit train/val/test windows (invariant #14) ---
    parser.add_argument(
        "--train-start", default="2021-01-01",
        help="Training window start (UTC). Phase A default 2021-01-01."
    )
    parser.add_argument(
        "--train-end", default="2024-06-30",
        help="Training window end (UTC). Phase A default 2024-06-30."
    )
    parser.add_argument(
        "--val-start", default="2024-07-01",
        help="Validation window start (UTC). Phase A default 2024-07-01."
    )
    parser.add_argument(
        "--val-end", default="2025-04-30",
        help="Validation window end (UTC). Phase A default 2025-04-30."
    )
    parser.add_argument(
        "--test-start", default="2025-05-01",
        help="Test window start (UTC). Phase A default 2025-05-01. "
             "NEVER loaded during training — invariant #14."
    )
    parser.add_argument(
        "--test-end", default="2026-04-30",
        help="Test window end (UTC). Phase A default 2026-04-30."
    )

    # --- Phase A: Optuna hyperparameter tuning (spec §4.1 + anchor 6) ---
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna study per symbol. Trial 0 = literature defaults; "
             "trials 1..N-1 sample from config/tuning_spaces.yaml. Saves both "
             "lstm_{symbol}_default.pt and lstm_{symbol}_tuned.pt artifacts."
    )
    parser.add_argument(
        "--tune-trials", type=int, default=20,
        help="Number of Optuna trials when --tune is set (default 20)."
    )
    return parser.parse_args()


def _auto_snapshot_models(label: str = None) -> str:
    """
    Auto-snapshot current models before retraining.

    Returns the snapshot label (so caller can log it). Failures are
    non-fatal — training continues even if snapshot fails.
    """
    from datetime import datetime
    if label is None:
        label = f"auto-pretrain-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    try:
        from scripts.model_snapshot import cmd_save
        cmd_save(label, note="Auto-snapshot before retraining")
        logger.info("Pre-training snapshot saved: %s", label)
        logger.info("To restore: python scripts/model_snapshot.py restore %s",
                     label)
    except Exception as exc:
        logger.warning("Auto-snapshot failed (continuing anyway): %s", exc)
    return label


async def _train_one_lstm_for_symbol(
    symbol: str,
    args: argparse.Namespace,
    *,
    feed: "MT5DataFeed",
    engineer: "FeatureEngineer",
    bars: int,
    cli_head_override: Optional[str],
    train_start_ts: "pd.Timestamp",
    val_start_ts: "pd.Timestamp",
    val_end_ts_exclusive: "pd.Timestamp",
    test_start_ts: "pd.Timestamp",
    hparam_overrides: Optional[dict] = None,
    artifact_suffix: str = "",
    extra_tags: Optional[dict] = None,
) -> dict:
    """Train one symbol's LSTM end-to-end. Side-effecting:
    saves model artifacts under data/models/, writes the training-
    distribution JSON, and logs an MLflow run.

    Per-symbol resolution (head, head guard, OHLCV fetch, feature
    engineering, regime injection, matrix build, TB targets, class-
    weighting, training, MLflow logging, drift snapshot) all live
    here. The orchestration (DataStore connect, the symbol loop,
    snapshot, teardown) stays in main_async.

    Task 2.2b-2b — Optuna tuning support:
    - ``hparam_overrides`` (Optional[dict]) — keys ``hidden_size``,
      ``num_layers``, ``dropout``, ``learning_rate``, ``batch_size`` are
      forwarded to ``LSTMPricePredictor._train_on_matrix`` overrides.
      When None the constructor / config defaults are used.
    - ``artifact_suffix`` ("" / "_default" / "_tuned" / "_trial_N") —
      threaded through to ``_train_on_matrix`` so saved files become
      ``lstm_{symbol}{suffix}.{pt,pca.pkl}`` /
      ``lstm_scaler_{symbol}{suffix}.pkl`` /
      ``lstm_{symbol}{suffix}.training_dist.json``. The head guard at the
      top of this function uses the same suffix so bake-off artifacts
      are independent of the legacy unsuffixed production file.
    - ``extra_tags`` (Optional[dict]) — merged into the MLflow tag set
      so trial vs final runs (``phase=optuna_trial`` /
      ``phase=optuna_final``) are queryable in the registry.

    Returns:
        dict with at least ``val_loss``, ``directional_accuracy``,
        ``model_path``, ``n_training_bars`` so an Optuna objective can
        minimize ``val_loss`` and the caller can locate the saved file.
    """
    zero_fill = engineer.get_zero_fill_feature_names(symbol)
    try:
        use_softmax = resolve_softmax_for_symbol(symbol, cli_head_override)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        # Path-aware guard: when artifact_suffix is set, check
        # lstm_{symbol}{suffix}.pt instead of the legacy unsuffixed file
        # so bake-off runs don't trip on the production model.
        assert_head_matches_existing(
            symbol, use_softmax, allow_change=args.allow_head_change,
            suffix=artifact_suffix,
        )
    except HeadMismatchError as exc:
        raise SystemExit(str(exc)) from exc
    if args.allow_head_change:
        logger.warning(
            "[%s] --allow-head-change is set; any head-shape mismatch "
            "with existing model will be overwritten.", symbol,
        )
    head_label = "softmax(3)" if use_softmax else "regression(1)"
    logger.info(
        "Training LSTM for %s (%s H4 bars, head=%s)...",
        symbol, "all" if args.bars == 0 else args.bars, head_label,
    )

    # --- 1. Fetch multi-TF OHLCV ---
    h4_ohlcv = await feed.get_historical_db_only(symbol, "H4", bars=bars)
    if h4_ohlcv is None or len(h4_ohlcv) < 300:
        logger.warning("Not enough H4 bars for %s: %s",
                       symbol, len(h4_ohlcv) if h4_ohlcv is not None else 0)
        return
    # Optional truncation for walk-forward OOS validation
    if args.end_date:
        import pandas as pd
        cutoff = pd.Timestamp(args.end_date)
        h4_ohlcv = h4_ohlcv[h4_ohlcv.index <= cutoff]
        logger.info("  Truncated H4 to %d bars (<= %s)",
                     len(h4_ohlcv), args.end_date)
    if args.start_date:
        import pandas as pd
        floor = pd.Timestamp(args.start_date)
        h4_ohlcv = h4_ohlcv[h4_ohlcv.index >= floor]
        logger.info("  Truncated H4 to %d bars (>= %s)",
                     len(h4_ohlcv), args.start_date)

    # Phase A invariant #14 — clip to [train_start, val_end] so the
    # test window never enters memory. Defense-in-depth assertion below
    # confirms the post-clip max bar timestamp falls strictly before
    # test_start.
    h4_ohlcv = h4_ohlcv[
        (h4_ohlcv.index >= train_start_ts)
        & (h4_ohlcv.index < val_end_ts_exclusive)
    ]
    if len(h4_ohlcv) > 0:
        # Explicit raise (not `assert`) so this lookahead-safety guard
        # can't be silently disabled when Python runs with -O.
        if h4_ohlcv.index.max() >= test_start_ts:
            raise ValueError(
                f"[invariant #14] {symbol} training/val data extends into "
                f"test window: H4 bar at {h4_ohlcv.index.max()} >= "
                f"test_start={test_start_ts}. Check --train-end / "
                f"--val-end / --test-start CLI args for {symbol}."
            )
    logger.info(
        "  Fetched %d H4 bars for %s (Phase A clip [%s, %s])",
        len(h4_ohlcv), symbol, args.train_start, args.val_end,
    )

    # --- 2. Compute features ---
    # Pre-initialize so both vars are bound on every codepath; the regime
    # injection block below reads d1_ohlcv via short-circuit and a future
    # reader could otherwise miss the conditional binding.
    d1_ohlcv = None
    w1_ohlcv = None
    if args.no_multi_tf:
        # H4 only (original behavior)
        features_df = engineer.transform_with_externals(
            h4_ohlcv, symbol, zero_fill_cols=zero_fill,
        )
    else:
        # Multi-TF: H4 primary + D1 + W1 context
        d1_ohlcv = await feed.get_historical_db_only(symbol, "D1", bars=bars // 6)
        w1_ohlcv = await feed.get_historical_db_only(symbol, "W1", bars=bars // 30)
        # Apply same truncation to D1/W1 for walk-forward
        if args.end_date:
            import pandas as pd
            cutoff = pd.Timestamp(args.end_date)
            if d1_ohlcv is not None:
                d1_ohlcv = d1_ohlcv[d1_ohlcv.index <= cutoff]
            if w1_ohlcv is not None:
                w1_ohlcv = w1_ohlcv[w1_ohlcv.index <= cutoff]
        if args.start_date:
            import pandas as pd
            floor = pd.Timestamp(args.start_date)
            if d1_ohlcv is not None:
                d1_ohlcv = d1_ohlcv[d1_ohlcv.index >= floor]
            if w1_ohlcv is not None:
                w1_ohlcv = w1_ohlcv[w1_ohlcv.index >= floor]

        # Phase A invariant #14 — same clip as H4.
        if d1_ohlcv is not None:
            d1_ohlcv = d1_ohlcv[
                (d1_ohlcv.index >= train_start_ts)
                & (d1_ohlcv.index < val_end_ts_exclusive)
            ]
        if w1_ohlcv is not None:
            w1_ohlcv = w1_ohlcv[
                (w1_ohlcv.index >= train_start_ts)
                & (w1_ohlcv.index < val_end_ts_exclusive)
            ]
        logger.info("  Fetched D1=%d, W1=%d bars",
                     len(d1_ohlcv) if d1_ohlcv is not None else 0,
                     len(w1_ohlcv) if w1_ohlcv is not None else 0)

        ohlcv_by_tf = {"H4": h4_ohlcv}
        if d1_ohlcv is not None and len(d1_ohlcv) > 50:
            ohlcv_by_tf["D1"] = d1_ohlcv
        if w1_ohlcv is not None and len(w1_ohlcv) > 10:
            ohlcv_by_tf["W1"] = w1_ohlcv

        # Multi-TF alignment + cross-asset + calendar externals injection.
        # (The historical macro/yield/curve/COT readers were removed from
        # the LSTM input path by the Phase A revert — they're consumed by
        # the meta-labeler in Sprint 4.) We're inside main_async
        # (asyncio.run already running) so we MUST use the _async variant;
        # the sync wrapper would raise "asyncio.run() cannot be called
        # from a running event loop".
        features_df = await engineer.transform_multi_timeframe_with_externals_async(
            ohlcv_by_tf, symbol, primary_tf="H4", zero_fill_cols=zero_fill,
        )

    # --- 3. Inject HMM regime features ---
    if not args.no_regime:
        hmm = HMMRegimeClassifier()
        if hmm.load(symbol):
            d1_for_regime = (
                d1_ohlcv
                if not args.no_multi_tf and d1_ohlcv is not None
                else await feed.get_historical_db_only(
                    symbol, "D1", bars=bars // 6,
                )
            )
            features_df = engineer.inject_regime_features(
                features_df, hmm, symbol, d1_for_regime,
            )
        else:
            logger.warning("  HMM not found for %s — skipping regime injection."
                           " Run train_hmm.py first.", symbol)
            # Add neutral regime features as placeholders
            for i in range(5):
                features_df[f"regime_{i}"] = 0.2
            features_df["regime_probability"] = 0.2

    # --- 4. Build manifest and matrix ---
    feature_manifest = engineer.get_feature_columns(features_df)
    feature_matrix = engineer.to_matrix(features_df)
    logger.info("  Feature matrix shape: %s (%d features)",
                 feature_matrix.shape, len(feature_manifest))

    # Phase A (Task 2.2b-1): derive the train/val cut from the calendar
    # boundary so the trainer's split matches the CLI windows rather
    # than a hard-coded 70/15/15 ratio. The OHLCV was already clipped
    # to [train_start, val_end_exclusive] in 2.2a, so the matrix only
    # contains train + val rows. The cut is "first row whose feature
    # timestamp >= val_start". val_start_ts is computed once in
    # main_async and passed in as a kwarg.
    n_train = int((features_df.index < val_start_ts).sum())
    n_total = len(feature_matrix)
    # Fallback to proportional 80/20 split when --val-start is outside the
    # data window. This happens in CPCV per-fold training where the
    # training sub-window can end well before the global Phase A val_start
    # default. The fallback preserves the original Phase A "calendar split"
    # behavior for the common full-history training case while making
    # CPCV (and other sub-window training scenarios) work without
    # requiring explicit --val-start/--val-end on every invocation.
    if n_train == 0 or n_train == n_total:
        proportional_n_train = max(1, int(n_total * 0.8))
        if n_train == 0:
            reason = (f"val_start={args.val_start} is at/before the first "
                      f"feature row {features_df.index[0] if len(features_df) else 'EMPTY'}")
        else:
            reason = (f"val_start={args.val_start} is past the last feature "
                      f"row {features_df.index[-1] if len(features_df) else 'EMPTY'}")
        logger.warning(
            "[%s] calendar split outside data window (%s); falling back to "
            "proportional 80/20 (n_train=%d/%d). Set --val-start within data "
            "range to use calendar split.",
            symbol, reason, proportional_n_train, n_total,
        )
        n_train = proportional_n_train
    explicit_split = (n_train, n_total)  # test slice intentionally empty
    logger.info(
        "  [%s] Calendar split: train=%d val=%d (val_start=%s)",
        symbol, n_train, n_total - n_train, args.val_start,
    )

    # --- 4b. Triple Barrier targets (Phase B.3) ---
    tb_targets = None
    if args.triple_barrier:
        # Per-symbol TB params (match settings.yaml per_symbol_params)
        tb_params = {
            "XAUUSD": {"tp_r": 2.5, "sl_atr": 2.0, "time_h1": 80},
            "EURUSD": {"tp_r": 2.0, "sl_atr": 1.5, "time_h1": 60},
            "USDJPY": {"tp_r": 2.0, "sl_atr": 2.0, "time_h1": 60},
            "USDCAD": {"tp_r": 2.0, "sl_atr": 1.8, "time_h1": 60},
            # Crypto: wider stops + longer holds (matches settings.yaml
            # per_symbol_params for ETHUSD).
            "ETHUSD": {"tp_r": 2.5, "sl_atr": 2.5, "time_h1": 100},
        }
        p = tb_params.get(symbol.upper(), tb_params["EURUSD"])
        time_h4 = max(1, p["time_h1"] // 4)  # convert H1 → H4
        # Align OHLCV to the feature_df index (drops warmup bars).
        # ATR is computed *inside* compute_triple_barrier_labels in
        # absolute price units — do NOT pass the feature-matrix atr_14
        # column (it's normalized as atr/close, a fraction).
        h4_aligned = h4_ohlcv.reindex(features_df.index)
        tb_targets = engineer.compute_triple_barrier_labels(
            h4_aligned, atr=None,
            tp_r_mult=p["tp_r"],
            sl_atr_mult=p["sl_atr"],
            time_limit_bars=time_h4,
        )
        logger.info("  TB labels for %s: tp=%.1fR sl=%.1fATR time=%dH4bars",
                     symbol, p["tp_r"], p["sl_atr"], time_h4)

    # --- 4c. Class-weighted sample weights (Phase: class-weighting 2026-04-14) ---
    # TB labels are {-1, 0, +1}. Observed distribution: -1 dominates (~45-48%)
    # which causes the LSTM to minimize MSE loss by predicting the class-prior
    # rather than learning bar-level discrimination. ModelTrainer.fit already
    # accepts sample_weights and threads them through weighted_mse_loss — we
    # just compute inverse-class-frequency weights and pass them in.
    sample_weights = None
    if args.class_weight and tb_targets is not None:
        import numpy as _np
        label_idx = (tb_targets + 1).astype(int)  # map {-1,0,+1} → {0,1,2}
        counts = _np.bincount(label_idx, minlength=3).astype(float)
        total = counts.sum()
        # Inverse frequency, normalized so mean weight = 1 if classes were
        # balanced. Capped at [0.5, 3.0] so no class over- or under-counts
        # by more than 3×.
        inv = total / (3.0 * _np.maximum(counts, 1.0))
        inv = _np.clip(inv, 0.5, 3.0)
        sample_weights = inv[label_idx].astype(_np.float32)
        logger.info(
            "  [%s] class weights: -1=%.2f (n=%d), 0=%.2f (n=%d), +1=%.2f (n=%d)",
            symbol, inv[0], int(counts[0]), inv[1], int(counts[1]),
            inv[2], int(counts[2]),
        )

    # --- 5. Train LSTM ---
    if use_softmax and not args.triple_barrier:
        raise SystemExit(
            f"[{symbol}] softmax head requires --triple-barrier "
            "(softmax classifies the {-1, 0, +1} TB labels directly)."
        )
    predictor = LSTMPricePredictor()

    # MLflow run per (symbol, training invocation) — T-8
    from datetime import datetime, timezone
    import mlflow
    from src.ml.registry import start_run, dataset_fingerprint

    suffix_tag = artifact_suffix or "live"
    run_name = (
        f"{symbol}-{suffix_tag}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    )
    fp = dataset_fingerprint(
        symbol=symbol,
        timeframe="H4",
        first_bar_ts=str(h4_ohlcv.index[0]),
        last_bar_ts=str(h4_ohlcv.index[-1]),
        closes=h4_ohlcv["close"].values,
    )
    model_head = "softmax" if use_softmax else "regression"
    base_tags: dict = {
        "symbol": symbol,
        "model_head": model_head,
        "training_script": "train_deep_learning.py",
        "artifact_suffix": artifact_suffix or "(none)",
    }
    if extra_tags:
        base_tags.update(extra_tags)
    with start_run(
        experiment="lstm_price",
        run_name=run_name,
        tags=base_tags,
    ):
        # Resolve effective hyperparameters: overrides win, then args.
        ov = hparam_overrides or {}
        eff_hidden = ov.get("hidden_size")
        eff_layers = ov.get("num_layers")
        eff_dropout = ov.get("dropout")
        eff_lr = float(ov.get("learning_rate", args.lr))
        eff_batch = int(ov.get("batch_size", args.batch_size))

        mlflow.log_params({
            "symbol": symbol,
            "bars_requested": args.bars,
            "bars_actual": len(h4_ohlcv),
            "epochs": args.epochs,
            "batch_size": eff_batch,
            "lr": eff_lr,
            "patience": args.patience,
            "pca_components": args.pca_components or 0,
            "triple_barrier": bool(args.triple_barrier),
            "class_weight": bool(args.class_weight),
            "softmax": bool(use_softmax),
            "use_focal_loss": bool(use_softmax and (not args.no_focal_loss)),
            "focal_gamma": float(args.focal_gamma),
            "no_regime": bool(args.no_regime),
            "no_multi_tf": bool(args.no_multi_tf),
            "end_date": args.end_date or "",
            "dataset_fingerprint": fp,
            # Log architecture overrides so the run is fully reproducible
            # from the registry. Empty string = "use constructor default".
            "hp_hidden_size": "" if eff_hidden is None else int(eff_hidden),
            "hp_num_layers": "" if eff_layers is None else int(eff_layers),
            "hp_dropout": "" if eff_dropout is None else float(eff_dropout),
            "artifact_suffix": artifact_suffix or "",
        })

        use_focal_loss = use_softmax and (not args.no_focal_loss)
        result = predictor._train_on_matrix(
            symbol, feature_matrix, feature_manifest=feature_manifest,
            pca_components=args.pca_components,
            targets_override=tb_targets,
            sample_weights=sample_weights,
            softmax=use_softmax,
            use_focal_loss=use_focal_loss,
            focal_gamma=args.focal_gamma,
            explicit_split=explicit_split,
            artifact_suffix=artifact_suffix,
            hidden_size_override=eff_hidden,
            num_layers_override=eff_layers,
            dropout_override=eff_dropout,
            learning_rate_override=eff_lr,
            batch_size_override=eff_batch,
        )

        # A-8: snapshot the training feature distribution alongside the
        # saved model so daily drift monitoring can compare the live
        # feature vectors against the data this model was trained on.
        # Save RAW values (pre-``to_matrix`` z-score), because drift is
        # computed on raw scales; ``to_matrix`` normalizes per-batch so
        # its output scale is not comparable across different batches.
        try:
            from src.ml.drift import save_training_distribution
            dist_path = (
                Path("data/models")
                / f"lstm_{symbol}{artifact_suffix}.training_dist.json"
            )
            raw_df = features_df[feature_manifest]
            raw_matrix = raw_df.to_numpy(dtype=float, copy=True)
            raw_matrix = np.nan_to_num(
                raw_matrix, nan=0.0, posinf=0.0, neginf=0.0,
            )
            save_training_distribution(
                dist_path,
                symbol=symbol, timeframe="H4",
                feature_matrix=raw_matrix,
                feature_names=tuple(feature_manifest or []),
            )
            mlflow.log_artifact(str(dist_path), artifact_path="dataset")
        except Exception as _exc:
            logger.warning("[%s] failed to save training distribution: %s",
                           symbol, _exc)

        # Metrics
        if result is not None:
            if hasattr(result, "best_val_loss"):
                mlflow.log_metric("best_val_loss", float(result.best_val_loss))
            if hasattr(result, "directional_accuracy"):
                mlflow.log_metric(
                    "directional_accuracy",
                    float(result.directional_accuracy),
                )
            if hasattr(result, "epochs_trained"):
                mlflow.log_metric(
                    "epochs_trained", float(result.epochs_trained),
                )
            # Log per-epoch curves if available
            if hasattr(result, "train_losses") and result.train_losses:
                for i, v in enumerate(result.train_losses):
                    mlflow.log_metric("train_loss", float(v), step=i)
            if hasattr(result, "val_losses") and result.val_losses:
                for i, v in enumerate(result.val_losses):
                    mlflow.log_metric("val_loss", float(v), step=i)
        mlflow.log_metric("n_training_bars", float(len(h4_ohlcv)))

        for ext in ("pt", "pca.pkl"):
            p = Path("data/models") / f"lstm_{symbol}{artifact_suffix}.{ext}"
            if p.exists():
                mlflow.log_artifact(str(p))
        scaler_p = (
            Path("data/models") / f"lstm_scaler_{symbol}{artifact_suffix}.pkl"
        )
        if scaler_p.exists():
            mlflow.log_artifact(str(scaler_p))

        # Dataset fingerprint sidecar
        import json, tempfile
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8",
        ) as tmp:
            json.dump({
                "symbol": symbol,
                "timeframe": "H4",
                "first_bar": str(h4_ohlcv.index[0]),
                "last_bar": str(h4_ohlcv.index[-1]),
                "bars": len(h4_ohlcv),
                "fingerprint_sha256": fp,
            }, tmp, indent=2)
            tmp_fp = tmp.name
        try:
            mlflow.log_artifact(tmp_fp, artifact_path="dataset")
        finally:
            Path(tmp_fp).unlink(missing_ok=True)

    logger.info("  LSTM training complete for %s", symbol)

    # Return structured metrics so the Optuna objective can minimize
    # val_loss and the caller can locate the saved model file.
    val_loss = (
        float(result.best_val_loss)
        if (result is not None and hasattr(result, "best_val_loss"))
        else float("nan")
    )
    directional_accuracy = (
        float(result.directional_accuracy)
        if (result is not None and hasattr(result, "directional_accuracy"))
        else float("nan")
    )
    model_path = Path("data/models") / f"lstm_{symbol}{artifact_suffix}.pt"
    return {
        "val_loss": val_loss,
        "directional_accuracy": directional_accuracy,
        "model_path": str(model_path),
        "n_training_bars": int(len(h4_ohlcv)),
    }


async def _run_optuna_study_for_symbol(
    symbol: str,
    args: argparse.Namespace,
    *,
    feed: "MT5DataFeed",
    engineer: "FeatureEngineer",
    bars: int,
    cli_head_override: Optional[str],
    train_start_ts: "pd.Timestamp",
    val_start_ts: "pd.Timestamp",
    val_end_ts_exclusive: "pd.Timestamp",
    test_start_ts: "pd.Timestamp",
) -> None:
    """Drive an Optuna study for one symbol — the model bake-off (spec §4.1,
    anchor 6).

    Strategy:
    1. Load ``config/tuning_spaces.yaml`` (lstm.defaults + lstm.search).
    2. Open a per-symbol Optuna study with SQLite storage at
       ``data/models/lstm_{symbol}_optuna_study.db`` so it's resumable.
       study.direction = "minimize" on val_loss.
    3. For each trial:
       - Trial 0: hard-pin literature defaults (singletons through
         suggest_categorical so the study records them as the canonical
         baseline).
       - Trials 1..N-1: sample from the search space.
       Use Optuna's ``ask`` / ``tell`` API instead of ``study.optimize``
       so we can ``await`` the async training step from inside the
       running event loop without bridging through a thread pool.
    4. After all trials, train two finals at top level (no nesting) so
       they're cleanly queryable in the registry:
         - ``lstm_{symbol}_default.pt`` from defaults (anchor 6 verbatim)
         - ``lstm_{symbol}_tuned.pt`` from study.best_params
    5. Clean up the trial-N intermediate artifacts so disk usage stays
       bounded (each trial drops 4 files; 20 trials × 4 = 80 / symbol
       without cleanup).

    Note:
        On a resumed study (trial 0 already exists in the SQLite DB),
        new trials sample from the search space — defaults are NOT
        re-pinned. To re-pin defaults (e.g. after editing
        ``config/tuning_spaces.yaml``), delete
        ``data/models/lstm_{symbol}_optuna_study.db`` and re-run.
    """
    import yaml
    import optuna
    from optuna.trial import TrialState

    spaces_path = Path("config/tuning_spaces.yaml")
    spaces_full = yaml.safe_load(spaces_path.read_text(encoding="utf-8"))
    spaces = spaces_full.get("lstm", {})
    if "defaults" not in spaces or "search" not in spaces:
        raise SystemExit(
            f"[{symbol}] tuning_spaces.yaml missing lstm.defaults or "
            f"lstm.search keys (path={spaces_path})"
        )

    storage_path = Path("data/models") / f"lstm_{symbol}_optuna_study.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_uri = f"sqlite:///{storage_path.as_posix()}"

    study = optuna.create_study(
        direction="minimize",
        study_name=f"lstm_{symbol}_phase_a",
        storage=storage_uri,
        load_if_exists=True,
    )

    # Trial 0 = literature defaults is a one-shot contract: pin the
    # baseline only when the study is fresh. With load_if_exists=True a
    # resumed study returns higher trial.number values from study.ask(),
    # so naively keying off `trial.number == 0` would never pin
    # defaults on resume AND would silently skip the defaults branch on
    # the first new trial. Operator can clear the DB to re-pin.
    existing_finished = [t for t in study.trials if t.state.is_finished()]
    needs_defaults_pin = len(existing_finished) == 0
    if needs_defaults_pin:
        # Canonical Optuna pattern for "pin a known-good config as trial 0":
        # enqueue defaults BEFORE any study.ask() so they pop off the queue
        # on the first ask, but suggest_* still records the standard
        # search-space distribution. The naive alternative
        # (suggest_categorical(k, [v]) on trial 0, suggest_categorical(k,
        # search_list) on trial 1+) hits Optuna's
        # `CategoricalDistribution does not support dynamic value space`
        # error because the per-param distribution is locked after the
        # first call. enqueue_trial sidesteps this entirely.
        # (Same fix as scripts/train_gbm.py — caught by the GBM smoke
        # tune on 2026-04-26, before Task 2.3 hit it on the LSTM side.)
        study.enqueue_trial(dict(spaces["defaults"]))
    else:
        logger.info(
            "[%s] Resuming Optuna study with %d existing trials — defaults "
            "already pinned in trial 0; new trials sample from search space.",
            symbol, len(existing_finished),
        )

    common_kwargs = dict(
        feed=feed, engineer=engineer, bars=bars,
        cli_head_override=cli_head_override,
        train_start_ts=train_start_ts,
        val_start_ts=val_start_ts,
        val_end_ts_exclusive=val_end_ts_exclusive,
        test_start_ts=test_start_ts,
    )

    n_trials = int(args.tune_trials)
    logger.info(
        "[%s] starting Optuna study (n_trials=%d, storage=%s)",
        symbol, n_trials, storage_path,
    )

    trial_numbers_this_run: list[int] = []
    try:
        for _ in range(n_trials):
            trial = study.ask()
            trial_numbers_this_run.append(trial.number)
            try:
                # Same suggest_* path for every trial — when the
                # enqueue_trial queue is non-empty (trial 0 of a fresh
                # study), suggest_* returns the enqueued value. After
                # that, the sampler takes over.
                params = {}
                for k, spec in spaces["search"].items():
                    if isinstance(spec, list):
                        params[k] = trial.suggest_categorical(k, spec)
                    elif isinstance(spec, dict):
                        low = spec["low"]
                        high = spec["high"]
                        if spec.get("log"):
                            params[k] = trial.suggest_float(
                                k, low, high, log=True,
                            )
                        else:
                            params[k] = trial.suggest_float(k, low, high)
                    else:
                        # ValueError (not SystemExit) so the per-trial
                        # except below catches it, marks the trial FAIL,
                        # and the loop continues. SystemExit inherits
                        # BaseException and would leave an orphan
                        # RUNNING trial in the SQLite study DB.
                        raise ValueError(
                            f"[{symbol}] unsupported search spec for "
                            f"{k!r}: {spec!r}"
                        )

                result = await _train_one_lstm_for_symbol(
                    symbol, args,
                    hparam_overrides=params,
                    artifact_suffix=f"_trial_{trial.number}",
                    extra_tags={
                        "phase": "optuna_trial",
                        "optuna_trial_number": str(trial.number),
                    },
                    **common_kwargs,
                )
                val_loss = float(result["val_loss"])
            except Exception as exc:
                # tell(FAIL) lives in except; tell(val_loss) lives in else.
                # If we put tell(val_loss) inside the try, a rare double-tell
                # (e.g. SQLite contention) would land in the except handler
                # and call tell(FAIL) on an already-COMPLETE trial → Optuna
                # raises RuntimeError on double-tell and kills remaining
                # trials. Splitting them keeps the calls mutually exclusive.
                logger.exception(
                    "[%s] trial %d failed: %s", symbol, trial.number, exc,
                )
                study.tell(trial, state=TrialState.FAIL)
            else:
                study.tell(trial, val_loss)
                logger.info(
                    "[%s] trial %d val_loss=%.6f params=%s",
                    symbol, trial.number, val_loss, params,
                )

        # After trials: train two finals at top level (no MLflow nesting).
        default_params = dict(spaces["defaults"])
        try:
            tuned_params = dict(study.best_params)
        except ValueError:
            logger.warning(
                "[%s] no completed trials — falling back to defaults for tuned",
                symbol,
            )
            tuned_params = default_params
        for label, params in (("default", default_params),
                              ("tuned", tuned_params)):
            logger.info("[%s] training %s final with params=%s",
                        symbol, label, params)
            res = await _train_one_lstm_for_symbol(
                symbol, args,
                hparam_overrides=params,
                artifact_suffix=f"_{label}",
                extra_tags={"phase": "optuna_final", "label": label},
                **common_kwargs,
            )
            logger.info(
                "[%s] saved %s artifact: val_loss=%.6f model=%s",
                symbol, label, res["val_loss"], res["model_path"],
            )
    finally:
        # Cleanup trial intermediates so disk usage stays bounded
        # regardless of whether the study completed or errored.
        # Iterate the trials we actually created this run — on a resumed
        # study trial.number starts above 0, so range(n_trials) would miss
        # the new files entirely.
        models_dir = Path("data/models")
        removed = 0
        for trial_num in trial_numbers_this_run:
            base = f"lstm_{symbol}_trial_{trial_num}"
            candidates = [
                models_dir / f"{base}.pt",
                models_dir / f"{base}.pca.pkl",
                models_dir / f"{base}.training_dist.json",
                models_dir / f"lstm_scaler_{symbol}_trial_{trial_num}.pkl",
            ]
            for p in candidates:
                existed = p.exists()
                try:
                    p.unlink(missing_ok=True)
                    if existed:
                        removed += 1
                except OSError as exc:
                    logger.warning(
                        "[%s] failed to remove %s: %s", symbol, p, exc,
                    )
        if removed:
            logger.info(
                "[%s] cleaned up %d trial-intermediate artifacts",
                symbol, removed,
            )


async def main_async():
    args = parse_args()

    # Auto-snapshot existing models before retraining (rollback safety)
    if not args.no_snapshot:
        _auto_snapshot_models(args.snapshot_label)

    # DB-only OHLCV reads — no MT5 contact, eliminates the shared-terminal
    # hijack risk that polluted prod equity_history on 2026-04-25.
    # See feedback_dev_mt5_steals_prod_terminal.md.
    #
    # Same DataStore feeds (a) OHLCV reads via get_historical_db_only and
    # (b) feature_store reads inside FeatureEngineer.transform_with_externals
    # (Phase 2A wiring). Both paths require it; missing DataStore is fatal.
    from src.data_pipeline.data_store import DataStore
    data_store = DataStore()
    await data_store.connect()
    logger.info(
        "DataStore connected — OHLCV reads via DB; transform_with_externals "
        "will inject historical externals from feature_store."
    )

    feed = MT5DataFeed(connector=None, data_store=data_store)
    engineer = FeatureEngineer(data_store=data_store)

    bars = args.bars if args.bars > 0 else 99999  # 0 = all available

    # Resolve the CLI head override once: latest of --softmax / --no-softmax
    # wins. argparse doesn't express "one-of", so we compare the flag
    # namespace directly and treat "neither set" as "read from config".
    if args.softmax and args.no_softmax:
        raise SystemExit("--softmax and --no-softmax are mutually exclusive.")
    cli_head_override: str | None = (
        "softmax" if args.softmax else "regression" if args.no_softmax else None
    )

    # Phase A train/val/test windows resolved once. The H4/D1/W1 OHLCV
    # slices are clipped to [train_start, val_end] upstream so the test
    # window is NEVER loaded into memory during training (invariant #14).
    #
    # val_end_ts_exclusive = (val_end + 1 day) so the clip uses a strict
    # `<` upper bound. pd.Timestamp("2025-04-30") evaluates to 00:00:00,
    # so an inclusive `<= val_end_ts` filter would silently drop the
    # 04:00/08:00/12:00/16:00/20:00 H4 bars on the last day. The exclusive
    # bound (= test_start_ts at the Phase A defaults) keeps the clip
    # numerically consistent with the strict-< test_start assertion below.
    import pandas as pd
    train_start_ts = pd.Timestamp(args.train_start)
    val_start_ts = pd.Timestamp(args.val_start)
    val_end_ts_exclusive = pd.Timestamp(args.val_end) + pd.Timedelta(days=1)
    test_start_ts = pd.Timestamp(args.test_start)

    for symbol in args.symbols:
        if args.tune:
            await _run_optuna_study_for_symbol(
                symbol, args,
                feed=feed,
                engineer=engineer,
                bars=bars,
                cli_head_override=cli_head_override,
                train_start_ts=train_start_ts,
                val_start_ts=val_start_ts,
                val_end_ts_exclusive=val_end_ts_exclusive,
                test_start_ts=test_start_ts,
            )
        else:
            await _train_one_lstm_for_symbol(
                symbol, args,
                feed=feed,
                engineer=engineer,
                bars=bars,
                cli_head_override=cli_head_override,
                train_start_ts=train_start_ts,
                val_start_ts=val_start_ts,
                val_end_ts_exclusive=val_end_ts_exclusive,
                test_start_ts=test_start_ts,
            )

    logger.info("All LSTM training complete.")


def main():
    """Sync wrapper — entry point for the script."""
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
