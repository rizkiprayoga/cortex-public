"""
Daily drift-check job (A-8).

Wired into ``main.py``'s scheduler at 01:00 UTC. For each live symbol:

1. Loads the per-symbol training-feature-distribution snapshot saved
   alongside the LSTM model.
2. Fetches the most recent 30 days of H4 bars + runs the full feature
   pipeline to build a "current" feature matrix.
3. Computes PSI + KS against the training distribution.
4. Persists the result into ``drift_scores`` (one row per symbol per run).
5. Raises invariants at tiered thresholds (0.25 WARN, 0.35 ALERT).
6. Optionally auto-triggers an off-cycle LSTM retrain when PSI exceeds
   the retrain threshold (default 0.5), rate-limited to once per 24h.

The thresholds follow the industry-standard PSI interpretation:
    <0.10   no shift
    0.10-0.25  slight shift
    0.25-0.35  significant shift  (WARN)
    0.35-0.50  major shift        (ALERT → Telegram)
    >0.50   severe shift          (auto-trigger retrain)
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_WARN = 0.25
DEFAULT_ALERT = 0.35
DEFAULT_RETRAIN = 0.50
DEFAULT_CURRENT_WINDOW_DAYS = 30

# PSI values above this almost always indicate a broken baseline reference
# (calendar-feature wraparound, distribution-schema mismatch, post-promotion
# bootstrap, etc.) rather than real concept drift. Above this ceiling we:
#   - Suppress auto-retrain (operator must inspect first), and
#   - Downgrade ALERT severity to WARN (no Telegram spam).
# Both behaviors are enforced consistently so the operator isn't paged on
# something they shouldn't act on. See docs/notes/2026-05-02-post-promotion-audit.md.
ABSURD_PSI_CEILING = 2.0


@dataclass
class DriftCheckSummary:
    symbol: str
    psi_max: float
    ks_max: float
    n_current_samples: int
    worst_feature: Optional[str]
    warn_breached: bool
    alert_breached: bool
    retrain_triggered: bool
    error: Optional[str] = None


def _compute_current_feature_matrix(
    symbol: str, feed, engineer, window_days: int,
):
    """Build the current feature matrix for drift comparison.

    Mirrors the H4 + HMM-regime feature pipeline used by the LSTM at
    training time. Returns (feature_matrix, feature_manifest) or
    (None, None) on failure.
    """
    import pandas as pd

    # Recent H4 bars — fetch enough for feature warmup (SMA200 needs 200 bars,
    # price_percentile_200 needs another 200). Fetch ~400 bars minimum, then
    # filter the computed feature frame down to the window at the end.
    # Previous version sliced H4 BEFORE feature computation, leaving only the
    # window's ~125 bars — not enough for SMA200 warmup, so transform() dropped
    # every row and the feature matrix came back empty (N, 0). This manifested
    # as "index 0 is out of bounds for axis 1 with size 0" in compute_drift.
    fetch_bars = max(window_days * 6 * 2, 500)
    h4 = feed.get_historical(symbol, "H4", bars=fetch_bars)
    if h4 is None or len(h4) < 250:
        logger.warning("[%s] drift: not enough H4 data (%s bars, need ≥250 for warmup)",
                       symbol, len(h4) if h4 is not None else 0)
        return None, None

    d1 = feed.get_historical(symbol, "D1", bars=window_days + 60)
    w1 = feed.get_historical(symbol, "W1", bars=window_days // 5 + 20)

    ohlcv_by_tf = {"H4": h4}
    if d1 is not None and len(d1) > 10:
        ohlcv_by_tf["D1"] = d1
    if w1 is not None and len(w1) > 3:
        ohlcv_by_tf["W1"] = w1

    features_df = engineer.transform_multi_timeframe(ohlcv_by_tf, primary_tf="H4")

    try:
        from src.data_pipeline.market.calendar_features import CalendarFeatureBuilder
        cal = CalendarFeatureBuilder()
        cal_df = cal.get_historical_calendar_features(features_df.index)
        features_df = features_df.join(cal_df, how="left")
    except Exception as _exc:
        logger.debug("Calendar features unavailable for drift-check: %s", _exc)

    zero_fill = engineer.get_zero_fill_feature_names(symbol)
    for col in zero_fill:
        if col not in features_df.columns:
            features_df[col] = 0.0
    features_df = features_df.fillna(0.0)

    # HMM regime injection — matches training
    from src.brain.hmm_regime import HMMRegimeClassifier
    hmm = HMMRegimeClassifier()
    if hmm.load(symbol) and d1 is not None and len(d1) > 30:
        features_df = engineer.inject_regime_features(features_df, hmm, symbol, d1)

    # NOW filter to the drift-comparison window, after warmup-sensitive
    # features are fully computed.
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    features_df = features_df[
        features_df.index >= pd.Timestamp(cutoff.replace(tzinfo=None))
    ]
    if len(features_df) < 50:
        logger.warning(
            "[%s] drift: after window filter only %d feature rows remain",
            symbol, len(features_df),
        )
        return None, None

    # Drift is computed on RAW feature values, not the z-scored output of
    # ``to_matrix()``. ``to_matrix`` normalizes each batch against its own
    # mean/std, so a 30-day drift window and a multi-year training batch
    # get different scale references — PSI/KS between them is meaningless.
    # Training callers must likewise pass raw values into
    # ``save_training_distribution`` (see scripts/train_deep_learning.py,
    # scripts/bootstrap_training_distributions.py).
    feature_manifest = engineer.get_feature_columns(features_df)
    sorted_df = features_df[feature_manifest]
    raw_matrix = sorted_df.to_numpy(dtype=float, copy=True)
    raw_matrix = np.nan_to_num(raw_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return raw_matrix, feature_manifest


def check_symbol_drift(
    symbol: str, feed, engineer,
    window_days: int = DEFAULT_CURRENT_WINDOW_DAYS,
    warn: float = DEFAULT_WARN, alert: float = DEFAULT_ALERT,
    retrain: float = DEFAULT_RETRAIN,
) -> DriftCheckSummary:
    """Compute drift for one symbol and return the summary."""
    from src.ml.drift import (
        load_training_distribution, compute_drift,
    )

    dist_path = Path("data/models") / f"lstm_{symbol}.training_dist.json"
    training_dist = load_training_distribution(dist_path)
    if training_dist is None:
        return DriftCheckSummary(
            symbol=symbol, psi_max=0.0, ks_max=0.0, n_current_samples=0,
            worst_feature=None, warn_breached=False, alert_breached=False,
            retrain_triggered=False,
            error=f"training distribution missing: {dist_path}",
        )

    try:
        matrix, feature_names = _compute_current_feature_matrix(
            symbol, feed, engineer, window_days,
        )
    except Exception as exc:
        logger.exception("[%s] drift: feature pipeline failed", symbol)
        return DriftCheckSummary(
            symbol=symbol, psi_max=0.0, ks_max=0.0, n_current_samples=0,
            worst_feature=None, warn_breached=False, alert_breached=False,
            retrain_triggered=False, error=f"feature pipeline: {exc}",
        )
    if matrix is None:
        return DriftCheckSummary(
            symbol=symbol, psi_max=0.0, ks_max=0.0, n_current_samples=0,
            worst_feature=None, warn_breached=False, alert_breached=False,
            retrain_triggered=False, error="current feature matrix empty",
        )

    # Defensive: a zero-column matrix sneaks past the `matrix is None`
    # check but crashes compute_drift at `matrix[:, i]`. Seen in the wild
    # when the H4 warmup window was too short so every feature row was
    # NaN-dropped (fixed 2026-04-21). Guard the downstream call either way.
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        logger.warning(
            "[%s] drift: feature matrix degenerate (shape=%s); skipping",
            symbol, getattr(matrix, "shape", None),
        )
        return DriftCheckSummary(
            symbol=symbol, psi_max=0.0, ks_max=0.0, n_current_samples=0,
            worst_feature=None, warn_breached=False, alert_breached=False,
            retrain_triggered=False,
            error=f"feature matrix degenerate shape={getattr(matrix, 'shape', None)}",
        )

    try:
        drift = compute_drift(training_dist, matrix, feature_names)
    except Exception as exc:
        logger.exception("[%s] drift: compute_drift failed", symbol)
        return DriftCheckSummary(
            symbol=symbol, psi_max=0.0, ks_max=0.0, n_current_samples=0,
            worst_feature=None, warn_breached=False, alert_breached=False,
            retrain_triggered=False, error=f"compute_drift: {exc}",
        )

    psi_max = drift["psi_max"]
    ks_max = drift["ks_max"]
    # Worst feature by PSI
    per = drift["per_feature"]
    worst = max(per.items(), key=lambda kv: kv[1]["psi"])[0] if per else None

    return DriftCheckSummary(
        symbol=symbol,
        psi_max=float(psi_max), ks_max=float(ks_max),
        n_current_samples=int(drift["n_current_samples"]),
        worst_feature=worst,
        warn_breached=psi_max >= warn,
        alert_breached=psi_max >= alert,
        retrain_triggered=False,   # caller flips this when retrain actually fires
    )


_REPO_ROOT = Path(__file__).resolve().parents[2]   # src/ml/ -> repo root


def maybe_trigger_retrain(
    symbol: str, psi_max: float, retrain_threshold: float,
    last_trigger_path: Optional[Path] = None,
    cooldown_hours: int = 24,
) -> bool:
    """Fire an off-cycle LSTM retrain for one symbol, rate-limited.

    Returns True iff the retrain subprocess was actually launched.
    Writes a small JSON file to track the last trigger time per symbol.

    Safety rails:
      - ``CORTEX_DRIFT_AUTO_RETRAIN=0`` kill-switch in env disables auto-
        retrain entirely (monitor still writes scores and alerts). Default
        enabled; set to 0 when drift thresholds need operator tuning.
      - Absurd-PSI cap: values above 2.0 almost certainly reflect a data
        or feature-pipeline issue (binning artifacts, distribution schema
        mismatch) rather than real concept drift. Skip auto-retrain and
        let the operator inspect via Telegram alert.
    """
    if psi_max < retrain_threshold:
        return False

    # Kill-switch for operator tuning periods
    if os.environ.get("CORTEX_DRIFT_AUTO_RETRAIN", "1").strip() not in ("1", "true", "True"):
        logger.info(
            "[%s] drift: CORTEX_DRIFT_AUTO_RETRAIN disabled — skipping retrain (psi=%.3f)",
            symbol, psi_max,
        )
        return False

    # High-PSI ceiling: suppress auto-retrain when the shift is large enough
    # that an operator should look at WHY before blindly retraining. Under
    # schema v2 this typically means a genuine regime shift (e.g. volatility
    # doubling) rather than a pipeline bug — but retraining on that without
    # review risks overfitting to a transient state.
    if psi_max > ABSURD_PSI_CEILING:
        logger.warning(
            "[%s] drift: psi=%.3f exceeds review ceiling %.1f — suppressing "
            "auto-retrain. Likely a real regime shift or pipeline artifact; "
            "inspect per-feature PSI via scripts/verify_drift_fix.py before retraining.",
            symbol, psi_max, ABSURD_PSI_CEILING,
        )
        return False

    import json
    if last_trigger_path is None:
        # Anchor to repo root so this works regardless of cwd at call time
        last_trigger_path = _REPO_ROOT / "data" / "state" / "drift_last_retrain.json"
    now = datetime.now(timezone.utc)
    last_trigger_path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, str] = {}
    if last_trigger_path.exists():
        try:
            state = json.loads(last_trigger_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("drift cooldown state unreadable: %s", exc)
            state = {}

    last_iso = state.get(symbol)
    if last_iso:
        try:
            last_dt = datetime.fromisoformat(last_iso)
            # Defensive: if stored value was written by older code without a
            # timezone suffix, attach UTC so the subtraction doesn't raise.
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            if now - last_dt < timedelta(hours=cooldown_hours):
                logger.info(
                    "[%s] drift: retrain suppressed (last trigger %s, cooldown %dh)",
                    symbol, last_iso, cooldown_hours,
                )
                return False
        except Exception as exc:
            logger.debug("drift cooldown parse failed: %s", exc)

    cmd = [
        sys.executable, "scripts/train_deep_learning.py",
        "--symbols", symbol, "--triple-barrier", "--pca-components", "25",
        "--no-snapshot",
    ]
    logger.warning(
        "[%s] drift: psi_max=%.3f >= %.2f — triggering off-cycle retrain",
        symbol, psi_max, retrain_threshold,
    )
    try:
        # Anchor cwd to repo root (relative script path would otherwise
        # silently FileNotFoundError). Redirect stdout/stderr to DEVNULL
        # so the fire-and-forget child doesn't inherit the bot's fds.
        subprocess.Popen(
            cmd, cwd=str(_REPO_ROOT),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        logger.error("[%s] retrain subprocess failed to launch: %s", symbol, exc)
        return False

    state[symbol] = now.isoformat()
    last_trigger_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return True


async def run_daily_drift_check(
    feed, engineer, data_store, alert_manager=None,
    symbols: tuple[str, ...] = ("XAUUSD", "EURUSD", "USDJPY", "USDCAD", "ETHUSD"),
    warn: float = DEFAULT_WARN, alert: float = DEFAULT_ALERT,
    retrain: float = DEFAULT_RETRAIN,
    window_days: int = DEFAULT_CURRENT_WINDOW_DAYS,
) -> list[DriftCheckSummary]:
    """Top-level job callable — runs all symbols, persists, alerts, triggers.

    Async because ``data_store.save_drift_score`` is an async SQLAlchemy
    call that must be awaited on the bot's main event loop. Caller in
    ``main.py`` registers an async wrapper with APScheduler so the
    framework runs this via ``run_coroutine_threadsafe`` on the right loop.
    Matches the pattern used by ``_pf_drift_job``.
    """
    from src.safety import invariants

    summaries: list[DriftCheckSummary] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for symbol in symbols:
        summary = check_symbol_drift(
            symbol, feed, engineer,
            window_days=window_days, warn=warn, alert=alert, retrain=retrain,
        )

        # Auto-trigger retrain if needed (sets retrain_triggered on summary)
        if summary.error is None and summary.psi_max >= retrain:
            fired = maybe_trigger_retrain(symbol, summary.psi_max, retrain)
            summary = DriftCheckSummary(**{**summary.__dict__, "retrain_triggered": fired})

        summaries.append(summary)

        # Persist to drift_scores table
        try:
            await data_store.save_drift_score({
                "timestamp": now_iso,
                "symbol": symbol,
                "psi_max": summary.psi_max,
                "ks_max": summary.ks_max,
                "n_current_samples": summary.n_current_samples,
                "threshold_warn_breached": summary.warn_breached,
                "threshold_alert_breached": summary.alert_breached,
                "retrain_triggered": summary.retrain_triggered,
                "worst_feature": summary.worst_feature,
                "notes": summary.error or "",
            })
        except Exception as exc:
            logger.warning("[%s] failed to persist drift_score: %s", symbol, exc)

        # Fire invariant at the highest breached severity. ALERT severity
        # routes to Telegram via the invariant registry; WARN does not.
        # When PSI exceeds ABSURD_PSI_CEILING the baseline is almost certainly
        # broken (operator can't act on it — auto-retrain is already suppressed
        # for the same reason) so downgrade to WARN to avoid daily Telegram
        # spam. See ABSURD_PSI_CEILING docstring above.
        severity = None
        if summary.alert_breached:
            if summary.psi_max > ABSURD_PSI_CEILING:
                severity = invariants.Severity.WARN
            else:
                severity = invariants.Severity.ALERT
        elif summary.warn_breached:
            severity = invariants.Severity.WARN
        if severity is not None:
            invariants.check(
                name="model.feature_drift",
                condition=False,
                severity=severity,
                symbol=symbol,
                message=(
                    f"PSI {summary.psi_max:.3f} (≥ {alert if summary.alert_breached else warn}) "
                    f"worst feature: {summary.worst_feature}"
                ),
                context={
                    "psi_max": summary.psi_max,
                    "ks_max": summary.ks_max,
                    "worst_feature": summary.worst_feature,
                    "retrain_triggered": summary.retrain_triggered,
                },
            )

        # Telegram alert only on ALERT level (WARN stays in invariants log)
        if alert_manager is not None and summary.alert_breached:
            try:
                alert_manager.notify_system(
                    event=f"feature_drift_alert_{symbol}",
                    details=(
                        f"PSI {summary.psi_max:.3f} (threshold {alert}) · "
                        f"worst feature {summary.worst_feature} · "
                        f"retrain triggered: {summary.retrain_triggered}"
                    ),
                )
            except Exception as exc:
                logger.warning("[%s] drift alert notification failed: %s", symbol, exc)

        log_level = logger.warning if summary.warn_breached else logger.info
        log_level(
            "[%s] drift: psi_max=%.3f ks_max=%.3f n=%d worst=%s "
            "warn=%s alert=%s retrain=%s err=%s",
            symbol, summary.psi_max, summary.ks_max, summary.n_current_samples,
            summary.worst_feature, summary.warn_breached, summary.alert_breached,
            summary.retrain_triggered, summary.error,
        )

    return summaries
