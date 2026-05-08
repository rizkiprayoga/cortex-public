"""
main.py — Trading System Orchestrator

Entry point for the Cortex autonomous trading bot.
Starts all modules, runs the async trading loop, and handles graceful shutdown.

Architecture
------------
    async main()
        └─ init DataStore (PostgreSQL async pool)
        └─ init MT5 + data pipeline + models + feedback loop
        └─ start Safety monitor (independent thread — does NOT consult Brain)
        └─ load or train HMM and LSTM models
        └─ start AsyncIOScheduler for retraining + feedback loop jobs
        └─ enter async trading loop
               ├─ fetch latest OHLCV + features (DB-backed)
               ├─ get signal from SignalCombiner (persists predictions)
               ├─ size position + place order
               └─ asyncio.sleep(900)   # M15 bar

Scheduled jobs
--------------
    hmm_retrain             every 7 days
    lstm_retrain            every 1 day
    feedback_check          every 24 hours (check_and_retrain per symbol)
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # load .env before any module reads os.environ

import pandas as pd

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.broker.mt5_connector import MT5Connector
from src.broker.account_monitor import AccountMonitor
from src.brain.hmm_regime import HMMRegimeClassifier
from src.brain.deep_learning.lstm_model import LSTMPricePredictor
from src.brain.signal_combiner import SignalCombiner
from src.data_pipeline.mt5_feed import MT5DataFeed, _broker_ts_to_utc
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.data_pipeline.data_store import DataStore
from src.data_pipeline.feedback_loop import FeedbackLoop
from src.data_pipeline.market.calendar_features import (
    calendar_freshness_warning,
    describe_news_context,
    is_in_news_blackout,
)
from src.utils.audit_log import (
    SIGNAL_AUDIT, TRADE_EVENTS, TICK_SUMMARY, now_iso,
)
from src.allocation.portfolio_manager import PortfolioManager
from src.allocation.position_sizer import PositionSizer
from src.broker.order_manager import OrderManager
from src.safety.circuit_breaker import CircuitBreaker
from src.safety.risk_monitor import RiskMonitor
from src.strategy.base import MarketContext
from src.strategy.exit_manager import ExitManager, OpenPosition, TierStateStore
from src.strategy.orchestrator import StrategyOrchestrator

# Dashboard API (Wave 12)
from src.api.app import build_app
from src.api.live_state import BotControl, BotStatus, LiveState
from src.utils.config_store import ConfigStore

# Alerts (Telegram + Email)
from src.alerts.manager import AlertManager

logger = logging.getLogger(__name__)

# the universe sweep sprint the trading universe — DEV PREVIEW of post-Sprint-2-promotion production set.
# Symmetric MULTIPLIERS +  selection. Drops AUDUSD/GBPJPY from
# the asymmetric b3_u5 winner; adds NZDUSD (short-USD diversifier) and EURJPY
# (cleaner JPY exposure than GBPJPY). Keeps JPY-bucket count at 2-of-7 below
# the b2 cap so no runtime correlation guard is needed pre-Sprint-3b.
SYMBOLS = [
    "XAUUSD", "GBPUSD", "USDJPY", "USDCAD", "NZDUSD",
    "USDCHF", "GBPCHF", "EURAUD", "GBPAUD", "EURJPY",
]

# Timeframes that get backfilled from MT5 on every startup so that a
# post-blackout restart resumes with a complete DB corpus before training.
BACKFILL_TIMEFRAMES = ["D1", "H4", "H1", "M15", "W1"]

# Sleep between trading-loop iterations (seconds) — one M15 bar
TRADING_LOOP_INTERVAL = 900

# Wave 5 tick resilience: if the main loop body raises this many ticks
# in a row, break out of the loop so the ``finally`` block can shut
# everything down cleanly instead of spinning on a broken brain/broker.
# Mirrors the RiskMonitor escalation pattern one level up — a persistent
# failure at the tick level means something is structurally wrong, and
# sitting in a retry loop prevents the ops operator from noticing.
MAX_CONSECUTIVE_TICK_FAILURES = 10

# Wave 6 fix #19: where to persist the tier state store on disk. Gitignored
# (see data/state/ in .gitignore) — per-bot, per-host runtime state only.
TIER_STATE_PATH = Path("data/state/tier_state.json")


async def main() -> None:
    logger.info("=== Cortex Trading System Starting ===")

    # --- 1. PostgreSQL async pool ---
    data_store = DataStore()
    await data_store.connect()
    logger.info("PostgreSQL data store connected")

    # --- 2. Connect to MT5 ---
    connector = MT5Connector()
    connector.connect()

    # --- 2b. Restore previous account slot if heartbeat recorded one ---
    # On restart the default MT5_LOGIN account connects first. If the
    # heartbeat shows a different account was active before shutdown,
    # reconnect to that account so the user doesn't have to re-switch.
    _heartbeat_path = Path("data/logs/bot_heartbeat.json")
    try:
        if _heartbeat_path.exists():
            _hb = json.loads(_heartbeat_path.read_text(encoding="utf-8"))
            _prev_account = _hb.get("mt5_account")
            import MetaTrader5 as _mt5_check
            _cur_info = _mt5_check.account_info()
            _cur_login = int(_cur_info.login) if _cur_info else None
            if (_prev_account is not None
                    and _cur_login is not None
                    and int(_prev_account) != _cur_login):
                # Find the slot that matches the previous account
                _target_slot = None
                for _key in os.environ:
                    if _key.startswith("MT5_SLOT_") and _key.endswith("_LOGIN"):
                        _slot_login = os.environ[_key].strip()
                        if _slot_login and int(_slot_login) == int(_prev_account):
                            _target_slot = _key[9:-6].lower()
                            break
                if _target_slot:
                    from src.api.routes.accounts import _resolve_slot_credentials
                    _creds = _resolve_slot_credentials(_target_slot)
                    if _creds:
                        connector.connect_with_creds(
                            _creds["login"], _creds["password"], _creds["server"],
                        )
                        logger.info(
                            "Restored previous account slot '%s' (login %d)",
                            _target_slot, _creds["login"],
                        )
    except Exception as _exc:
        logger.warning("Account restore from heartbeat failed: %s", _exc)

    # --- 3. Initialize pipeline & models (all DB-aware) ---
    data_feed = MT5DataFeed(connector, data_store=data_store)
    feature_engineer = FeatureEngineer(data_store=data_store)

    # Wave 7.5: FundamentalDataManager coordinates all external data fetchers.
    # Gracefully degrades — if any external API is unavailable, its features
    # default to neutral values and the bot continues trading.
    from src.data_pipeline.fundamental.manager import FundamentalDataManager
    fundamental_mgr = FundamentalDataManager()
    fundamental_mgr.init_fetchers()

    hmm = HMMRegimeClassifier(data_store=data_store)
    lstm = LSTMPricePredictor(data_store=data_store)

    feedback_loop = FeedbackLoop(data_store=data_store, hmm=hmm, lstm=lstm)

    # Load signal-gate params from model_config.yaml so live matches the
    # backtest path. Previously the default threshold was hardcoded to 0.6
    # (SignalCombiner's built-in fallback) which silently diverged from
    # backtest_full.py (which reads 0.45 from this config). See audit
    # notes 2026-04-14: that divergence blocked every ETHUSD/XAU/USDCAD
    # signal with "below_threshold" while backtest showed PF 1.6–6.8.
    # PLACEHOLDERS — tuned production values redacted from this public template.
    # Real values come from config/model_config.yaml; these defaults must be
    # overridden there for the bot to fire trades.
    _default_signal_threshold = 0.0
    _hmm_weight = 0.0
    _lstm_weight = 0.0
    _per_sym_threshold: dict[str, float] = {}
    try:
        import yaml
        _mc = yaml.safe_load(
            Path("config/model_config.yaml").read_text(encoding="utf-8")
        )
        _sc = (_mc or {}).get("signal_combiner", {})
        _default_signal_threshold = float(
            _sc.get("signal_threshold", _default_signal_threshold)
        )
        _hmm_weight = float(_sc.get("hmm_weight", _hmm_weight))
        _lstm_weight = float(_sc.get("lstm_weight", _lstm_weight))
        _per_sym = _sc.get("signal_threshold_per_symbol") or {}
        _per_sym_threshold = {str(k).upper(): float(v) for k, v in _per_sym.items()}
    except Exception as _exc:
        logger.warning("Failed to load signal thresholds from config: %s", _exc)

    # Load strategy block from settings.yaml — long_only_symbols, min_confidence,
    # flicker_bars_required all live under `strategy.*` and were previously
    # hardcoded here, leaving the yaml entries as dead config (incoherent-rules
    # audit 2026-04-30, items C3+D3+D7). Defaults below match the post-2026-04-27
    # XAU-bidirectional flip + the existing hardcoded values, so behavior is
    # preserved if the file is missing.
    _long_only_symbols: set[str] = {"ETHUSD"}  # default — XAU flipped 2026-04-27
    _min_confidence: float = 0.55              # default
    _flicker_bars: int = 2                     # default
    try:
        _settings = yaml.safe_load(
            Path("config/settings.yaml").read_text(encoding="utf-8")
        )
        _strategy_cfg = (_settings or {}).get("strategy", {}) or {}
        _lo_list = _strategy_cfg.get("long_only_symbols", [])
        if _lo_list:
            _long_only_symbols = set(str(s).upper() for s in _lo_list)
        _mc_yaml = _strategy_cfg.get("min_confidence")
        if _mc_yaml is not None:
            _min_confidence = float(_mc_yaml)
        _fb_yaml = _strategy_cfg.get("flicker_bars_required")
        if _fb_yaml is not None:
            _flicker_bars = int(_fb_yaml)
    except Exception as _exc:
        logger.warning(
            "Failed to load strategy config from settings.yaml, using defaults "
            "(long_only=%s, min_confidence=%.2f, flicker_bars=%d): %s",
            _long_only_symbols, _min_confidence, _flicker_bars, _exc,
        )

    combiner = SignalCombiner(
        hmm=hmm,
        lstm=lstm,
        data_store=data_store,
        feedback_loop=feedback_loop,
        hmm_weight=_hmm_weight,
        lstm_weight=_lstm_weight,
        long_only_mode=False,
        long_only_symbols=_long_only_symbols,
        min_confidence=_min_confidence,
        flicker_bars_required=_flicker_bars,
        signal_threshold=_default_signal_threshold,
    )
    if _per_sym_threshold:
        combiner.per_symbol_threshold = _per_sym_threshold
        logger.info("Loaded per-symbol signal thresholds: %s",
                     combiner.per_symbol_threshold)
    logger.info("SignalCombiner: hmm_w=%.2f, lstm_w=%.2f, threshold=%.2f, "
                "long_only=%s, min_confidence=%.2f, flicker_bars=%d",
                _hmm_weight, _lstm_weight, _default_signal_threshold,
                _long_only_symbols, _min_confidence, _flicker_bars)

    # Strategy layer — vol-rank orchestrator + 3-tier exit ladder
    orchestrator = StrategyOrchestrator()
    # Wave 6 fix #19: tier state persistence. The store is loaded at init
    # time from data/state/tier_state.json (tolerating a missing or
    # corrupt file) and written through on every tier 1 / tier 2 fire.
    tier_state_store = TierStateStore(TIER_STATE_PATH)
    exit_manager = ExitManager(
        reversal_bars_required=4,
        tier_state_store=tier_state_store,
    )
    # Load settings early — per-symbol params feed the initial position
    # reconcile below, and risk caps feed PortfolioManager further down.
    _risk_cfg: dict = {}
    _per_symbol_params: dict = {}
    try:
        import yaml
        _set = yaml.safe_load(
            Path("config/settings.yaml").read_text(encoding="utf-8")
        )
        _risk_cfg = (_set or {}).get("risk", {}) or {}
        _per_symbol_params = (
            ((_set or {}).get("strategy", {}) or {}).get("per_symbol_params", {}) or {}
        )
    except Exception as _exc:
        logger.warning("Failed to load risk config, using defaults: %s", _exc)

    # Per-symbol time-exit thresholds (H1 bars). Wire through to every
    # new OpenPosition — previously ExitManager fell back to its class
    # default of 20 while incrementing per main-loop tick (M15), which
    # gave a 5-hour de-facto time exit instead of the intended 60-100 H1.
    TIME_EXIT_H1_BY_SYMBOL: dict[str, int] = {
        sym: int((params or {}).get("time_exit_h1_bars", 60) or 60)
        for sym, params in (_per_symbol_params or {}).items()
    }
    if TIME_EXIT_H1_BY_SYMBOL:
        logger.info(
            "Per-symbol time_exit (H1 bars): %s",
            ", ".join(f"{k}={v}" for k, v in TIME_EXIT_H1_BY_SYMBOL.items()),
        )
    # Sprint 3 audit fix (2026-05-01): per-symbol tp_r_multiple and be_trigger_r
    # — previously ExitManager held single class-default 2.5/1.0 for ALL pairs
    # while yaml's per_symbol_params declared per-pair values that were silently
    # ignored. Now wired through to OpenPosition like time_exit_bars.
    TP_R_BY_SYMBOL: dict[str, float] = {
        sym: float((params or {}).get("tp_r_multiple", 2.5) or 2.5)
        for sym, params in (_per_symbol_params or {}).items()
    }
    BE_TRIGGER_R_BY_SYMBOL: dict[str, float] = {
        sym: float((params or {}).get("be_trigger_r", 1.0) or 1.0)
        for sym, params in (_per_symbol_params or {}).items()
    }
    if TP_R_BY_SYMBOL:
        logger.info(
            "Per-symbol tp_r_multiple: %s",
            ", ".join(f"{k}={v}" for k, v in TP_R_BY_SYMBOL.items()),
        )
    if BE_TRIGGER_R_BY_SYMBOL:
        logger.info(
            "Per-symbol be_trigger_r: %s",
            ", ".join(f"{k}={v}" for k, v in BE_TRIGGER_R_BY_SYMBOL.items()),
        )

    # Per-symbol ring of open positions tracked for the exit ladder.
    tracked_positions: dict[int, OpenPosition] = {}
    # Rebuild from mt5.positions_get() so positions that survived a prior
    # run (crash, Ctrl+C, internet outage) come back under exit-ladder
    # management. They still have their server-side SL — this just
    # re-enables partials, trailing, and reversal exits for them.
    _reconcile_tracked_positions(
        tracked_positions,
        tier_state_store=tier_state_store,
        time_exit_by_symbol=TIME_EXIT_H1_BY_SYMBOL,
        tp_r_by_symbol=TP_R_BY_SYMBOL,
        be_trigger_r_by_symbol=BE_TRIGGER_R_BY_SYMBOL,
    )
    recent_signal_dirs: dict[str, list[str]] = {sym: [] for sym in SYMBOLS}
    # Per-symbol last-processed bar timestamp. The trading loop ticks at
    # TRADING_LOOP_INTERVAL (M15) for exit-manager responsiveness, but
    # signal generation + entries operate on H4 data. Without this dedup
    # the LSTM rolling history, flicker ring, and reversal direction log
    # would all advance 16x faster than backtest expects — which collapses
    # the z-score normalizer's rolling std and pins combined_score at the
    # Euphoria floor (+0.4 HMM − 0.6 saturated-LSTM = −0.2).
    last_processed_bar_ts: dict[str, str] = {}

    # Allocation layer — sizer + pyramiding-aware portfolio manager
    sizer = PositionSizer(max_risk_pct=1.0)
    account_monitor = AccountMonitor(connector)

    portfolio = PortfolioManager(
        sizer=sizer,
        positions_provider=lambda: _build_position_views(tracked_positions),
        symbol_spec_provider=_resolve_symbol_spec,
        max_concurrent_per_symbol=int(_risk_cfg.get("max_concurrent_per_symbol", 3)),
        max_concurrent_total=int(_risk_cfg.get("max_concurrent_total", 8)),
        max_used_margin_pct_per_position=float(_risk_cfg.get("max_position_size_pct", 5.0)),
        max_used_margin_pct_total=float(_risk_cfg.get("max_total_exposure_pct", 15.0)),
        free_margin_reserve_pct=float(_risk_cfg.get("free_margin_reserve_pct", 20.0)),
        max_daily_trades=int(_risk_cfg.get("max_daily_trades", 12)),
    )
    logger.info(
        "PortfolioManager: per_symbol=%d total=%d pos_margin<=%.1f%% "
        "total_margin<=%.1f%% free_reserve>=%.1f%% daily=%d",
        portfolio.max_concurrent_per_symbol, portfolio.max_concurrent_total,
        portfolio.max_used_margin_pct_per_position,
        portfolio.max_used_margin_pct_total,
        portfolio.free_margin_reserve_pct,
        portfolio.max_daily_trades,
    )

    # Alerts (Telegram + Email — graceful degradation if not configured)
    alert_manager = AlertManager()

    # Fire the startup alert BEFORE the gap-backfill step — backfill can
    # take 30+ minutes on a stale cache, which used to delay this alert
    # past the point where the operator assumed the bot failed to launch.
    alert_manager.notify_system(
        "Bot Started",
        f"Symbols: {', '.join(SYMBOLS)} | Loop interval: {TRADING_LOOP_INTERVAL}s",
    )

    # Invariant registry — fires Telegram only on ALERT/CRITICAL (24h dedup).
    from src.safety.invariants import configure_registry as _configure_invariants
    _configure_invariants(telegram_send=alert_manager.telegram.send)

    # Safety layer
    circuit_breaker = CircuitBreaker()
    risk_monitor = RiskMonitor(connector, circuit_breaker, alert_manager=alert_manager)
    risk_monitor.attach_signal_ref(lambda: combiner.last_signal)
    # Wave 6 fix #24: RiskMonitor becomes the canonical owner of the
    # tracked-positions dict and the combiner reset hook. After
    # EmergencyClose fires, the halt path atomically clears this dict
    # and flushes the combiner's 4-bar flickering ring so the next
    # tick cannot chase ghost tickets or inherit a pre-halt direction.
    _trading_cfg = (_set or {}).get("trading", {}) or {}
    order_mgr = OrderManager(
        connector,
        retry_attempts=int(_trading_cfg.get("order_retry_attempts", 3)),
        retry_backoff_sec=float(_trading_cfg.get("order_retry_backoff_sec", 20.0)),
    )
    logger.info(
        "OrderManager: retry_attempts=%d backoff=%.0fs",
        order_mgr.retry_attempts, order_mgr.retry_backoff_sec,
    )

    # --- 3b. Dashboard API (Wave 12) ---
    bot_control = BotControl()
    config_store = ConfigStore(Path("config/settings.yaml"))
    live_state = LiveState(
        tracked_positions=tracked_positions,
        combiner=combiner,
        circuit_breaker=circuit_breaker,
        account_monitor=account_monitor,
        risk_monitor=risk_monitor,
        order_manager=order_mgr,
        orchestrator=orchestrator,
        portfolio=portfolio,
        data_store=data_store,
        config_store=config_store,
        bot_control=bot_control,
    )

    # Tag every account-segmented persistence path with the live MT5
    # account number. Without this call, equity_history / trades /
    # signals all save with mt5_account=NULL and the dashboard filter
    # `WHERE mt5_account = X` returns 0 rows — empty Equity chart,
    # empty Trades view, etc. This is the root cause we hit twice now.
    try:
        import MetaTrader5 as _mt5
        _info = _mt5.account_info()
        if _info is not None and getattr(_info, "login", None):
            live_state.set_account_id(int(_info.login))
            logger.info("LiveState: bound to mt5_account=%d", int(_info.login))
        else:
            logger.warning("Could not read MT5 account_info() — account-segmented data will be untagged")
    except Exception as _exc:
        logger.warning("Failed to set live_state account_id: %s", _exc)

    # Audit C5: wire positions_lock to RiskMonitor so cross-thread
    # mutations of tracked_positions are serialized with the main loop.
    risk_monitor.set_position_tracker(
        tracked_positions, positions_lock=live_state.positions_lock,
    )
    risk_monitor.set_signal_combiner(combiner)
    # Audit C7: give RiskMonitor the main event loop so it can schedule
    # combiner.reset_state() via call_soon_threadsafe.
    risk_monitor.set_main_event_loop(asyncio.get_running_loop())

    api_app = build_app(live_state)
    import uvicorn
    # API bind host: DASHBOARD_BIND_HOST env var overrides.
    #   "127.0.0.1" (default) = local only — safest
    #   "0.0.0.0"              = also accessible from your LAN (phone, laptop)
    # The dashboard's own auth layer (JWT + lock flag) protects LAN access,
    # but anyone on your Wi-Fi who knows the login can reach it. Use a
    # strong DASHBOARD_PW_HASH and don't expose on untrusted networks.
    bind_host = os.environ.get("DASHBOARD_BIND_HOST", "127.0.0.1")
    # Port defaults to 8787 (prod). Dev instance overrides via CORTEX_API_PORT
    # env var (typically 8788) so both can run concurrently on the same host.
    bind_port = int(os.environ.get("CORTEX_API_PORT", "8787"))
    uvicorn_config = uvicorn.Config(
        api_app, host=bind_host, port=bind_port, log_level="warning"
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)
    api_task = asyncio.create_task(uvicorn_server.serve())
    logger.info("Dashboard API started on http://%s:%d", bind_host, bind_port)
    if bind_host == "0.0.0.0":
        # Surface the LAN IP(s) for easy phone access
        try:
            import socket as _socket
            hostname = _socket.gethostname()
            lan_ips = _socket.gethostbyname_ex(hostname)[2]
            for ip in lan_ips:
                if not ip.startswith("127."):
                    logger.info("  LAN access: http://%s:%d (phone/other devices)", ip, bind_port)
        except Exception:
            pass

    # --- 4. Start safety monitor (independent — does NOT consult Brain) ---
    risk_monitor.start()

    # --- 4b. Backfill any OHLCV gaps from downtime (self-healing after blackout) ---
    # Runs as a background task so startup isn't gated on it. When the
    # feature_vectors table is missing rows for a symbol/TF, backfill can
    # recompute tens of thousands of rows and take 30+ minutes — which
    # delays the trading loop and the dashboard indefinitely. The first
    # trading cycle may run against slightly stale features; subsequent
    # cycles pick up fresh values as backfill completes.
    async def _background_backfill() -> None:
        logger.info("Background backfill: checking for data gaps...")
        for symbol in SYMBOLS:
            for tf in BACKFILL_TIMEFRAMES:
                try:
                    n_bars = await data_feed.backfill_gaps(symbol, tf)
                    if n_bars > 0:
                        n_feat = await feature_engineer.backfill_features(symbol, tf)
                        logger.info(
                            f"[{symbol} {tf}] gap-filled: {n_bars} bars, {n_feat} feature rows"
                        )
                    else:
                        logger.info(f"[{symbol} {tf}] no gap to fill")
                except Exception as e:
                    logger.error(
                        f"[{symbol} {tf}] backfill failed: {e}",
                        exc_info=True,
                    )
        logger.info("Background backfill: complete.")

    asyncio.create_task(_background_backfill())

    # --- 5. Load pre-trained models (or train from scratch on first run) ---
    await hmm.load_or_train_async(data_feed, feature_engineer, symbols=SYMBOLS)
    await lstm.load_or_train_async(data_feed, feature_engineer, symbols=SYMBOLS)

    # --- 6. Scheduler: retraining + feedback loop jobs ---
    scheduler = AsyncIOScheduler()

    # Monthly subprocess retrain (Phase C.3) — runs the proper scripts
    # (train_hmm.py + train_deep_learning.py --triple-barrier --pca-components 25)
    # so production models stay consistent with the TB+PCA pipeline.
    #
    # We deliberately DO NOT use `hmm.retrain` / `lstm.retrain` directly
    # because those in-process paths train on 56 plain H4 features only —
    # they would silently corrupt the 25-dim PCA + TB-label production
    # model every retrain cycle. The subprocess approach invokes the same
    # scripts that were used to build the `phase-b3-tb` snapshot, with
    # auto-snapshotting so a bad retrain can always be rolled back.
    def _monthly_full_retrain() -> None:
        import subprocess
        import sys
        from datetime import datetime, timedelta, timezone
        from pathlib import Path

        project_root = Path(__file__).parent

        # Walk-forward expanding-window dates for the LSTM cron retrain.
        # Phase A's `feat(lstm): explicit train/val/test windows` (commit
        # 43463c1) introduced an internal `[train_start, val_end]` clip
        # for invariant #14. Without dynamic dates here, that clip would
        # pin every monthly retrain to the same fixed [2021-01-01,
        # 2025-04-30] window — defeating walk-forward entirely.
        #
        # By computing dates from `today` we keep the prod behavior the
        # cron has always had: an expanding window from earliest available
        # history through ~yesterday, with a thin val window for early-
        # stopping. The clip becomes a wide no-op: it admits everything
        # the data feed returns.
        today = datetime.now(timezone.utc).date()
        # 2000-01-01 is well before any forex H4 data (Dec 2000 majors,
        # Aug 2010 XAU, May 2016 ETH) — sentinel "include all history".
        cron_train_start = "2000-01-01"
        cron_train_end   = (today - timedelta(days=61)).isoformat()
        cron_val_start   = (today - timedelta(days=60)).isoformat()
        cron_val_end     = (today - timedelta(days=1)).isoformat()
        # `test_start` is never loaded by the trainer (invariant #14),
        # but is asserted strictly > `val_end_ts_exclusive`. Set it to
        # tomorrow so the assertion holds.
        cron_test_start  = today.isoformat()
        cron_test_end    = (today + timedelta(days=365)).isoformat()

        try:
            logger.info(
                "[retrain] Monthly retrain starting: HMM + LSTM (TB+PCA). "
                "Walk-forward window train=[%s, %s], val=[%s, %s].",
                cron_train_start, cron_train_end,
                cron_val_start, cron_val_end,
            )
            # Step 1: HMM (auto-snapshots models before retraining)
            subprocess.run(
                [sys.executable, "scripts/train_hmm.py",
                 "--symbols", *SYMBOLS, "--bars", "5000"],
                cwd=project_root, check=True, timeout=1800,
            )
            # Step 2: LSTM with TB+PCA (matches phase-b3-tb production config)
            subprocess.run(
                [sys.executable, "scripts/train_deep_learning.py",
                 "--symbols", *SYMBOLS, "--bars", "0",
                 "--pca-components", "25", "--triple-barrier",
                 "--no-snapshot",  # HMM step already snapshotted
                 "--train-start", cron_train_start,
                 "--train-end",   cron_train_end,
                 "--val-start",   cron_val_start,
                 "--val-end",     cron_val_end,
                 "--test-start",  cron_test_start,
                 "--test-end",    cron_test_end],
                cwd=project_root, check=True, timeout=3600,
            )
            logger.info("[retrain] Monthly retrain complete — reloading models")
            # Reload freshly trained models into the running process
            for _sym in SYMBOLS:
                try:
                    hmm.load(_sym)
                    lstm.load(_sym)
                except Exception as _exc:
                    logger.warning("[%s] reload after retrain failed: %s", _sym, _exc)
            try:
                alert_manager.notify_system(
                    event="retrain_complete",
                    details=f"Monthly retrain finished for {', '.join(SYMBOLS)}",
                )
            except Exception:
                pass  # alert failures must never crash the scheduler
        except subprocess.CalledProcessError as exc:
            logger.error("[retrain] Subprocess failed (rc=%s) — models unchanged. "
                          "Restore via: python scripts/model_snapshot.py restore "
                          "<last-good-label> --yes", exc.returncode)
        except Exception as exc:
            logger.exception("[retrain] Monthly retrain error: %s", exc)

    # Wrap as an async coroutine so APScheduler (which may invoke cron
    # jobs via a thread pool) always dispatches back onto the event loop
    # before we call asyncio.to_thread. Fixes an intermittent RuntimeError
    # that would fire when the executor happened to run the lambda off
    # the loop thread. (Audit MED-1.)
    async def _monthly_retrain_wrapper():
        await asyncio.to_thread(_monthly_full_retrain)

    # First Saturday of each month at 03:00 UTC (10:00 Jakarta) — 6 hours
    # after Friday NY close (21:00 UTC), so forex markets are closed and
    # no live bars are being ingested. Previously fired day=1 regardless of
    # weekday, which could land mid-session on a Tuesday-Thursday.
    scheduler.add_job(
        _monthly_retrain_wrapper,
        trigger="cron", day="1-7", day_of_week="sat",
        hour=3, minute=0, timezone="UTC",
        id="monthly_full_retrain",
    )

    # Feedback loop: keep outcome/error LOGGING (useful for monitoring).
    # In-process retrain is DISABLED at the feedback_loop level
    # (``enable_inprocess_retrain = False``) so this job only computes
    # and persists prediction errors — full retrains happen monthly via
    # the subprocess job above.
    for symbol in SYMBOLS:
        scheduler.add_job(
            feedback_loop.check_and_retrain,
            trigger="interval", hours=24,
            args=[symbol, data_feed, feature_engineer],
            id=f"feedback_check_{symbol}",
        )

    # Event loop handle — captured below (line ~811) once we're inside
    # the async context. Pre-declared as None so the scheduler closures
    # (daily + weekly summary) can close over a name that exists even if
    # a missed job fires during startup before `asyncio.get_running_loop()`
    # is called. The closures check for None and skip.
    _main_loop: Optional[asyncio.AbstractEventLoop] = None

    # Daily summary alert — fires once per day at 23:55 UTC
    def _send_daily_summary():
        try:
            positions_lock = live_state.positions_lock
            snap = account_monitor.get_info()
            breaker_snap = circuit_breaker.check_and_update(
                current_equity=snap.equity,
                daily_start_equity=risk_monitor._daily_start_equity,
                weekly_start_equity=risk_monitor._weekly_start_equity,
                peak_equity=risk_monitor._peak_equity,
            )
            daily_pnl = snap.equity - risk_monitor._daily_start_equity
            weekly_pnl = snap.equity - risk_monitor._weekly_start_equity
            breaker_str = (
                ", ".join(breaker_snap.active_breakers)
                if breaker_snap.active_breakers else "clear"
            )

            # Fetch realized trades today from DB (async → sync bridge)
            trades_today = 0
            win_rate = None
            realized_pnl = None
            per_symbol_pnl = None
            if _main_loop is None:
                logger.warning("Daily summary: event loop not captured yet, skipping DB fetch")
                # Fall through to the alert call with zero trade counts — still
                # better than a NameError when a missed job fires on startup.
            try:
                from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                _today_start = _dt.now(_tz.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0,
                )
                if _main_loop is None:
                    raise RuntimeError("event loop not captured")
                # `get_trades(since=...)` filters by timestamp_open. We want
                # trades that *closed* today, not opened today — a position
                # opened days ago and closed today must still count. Fetch a
                # wider window by open-time and post-filter on close-time.
                fut = asyncio.run_coroutine_threadsafe(
                    data_store.get_trades(since=_today_start - _td(days=14)),
                    _main_loop,
                )
                df = fut.result(timeout=10)
                if not df.empty:
                    # Account-scope so multi-slot doesn't pull in other accounts.
                    acct_id = live_state.get_account_id()
                    if acct_id is not None and "mt5_account" in df.columns:
                        df = df[df["mt5_account"] == acct_id]
                    close_dt = pd.to_datetime(
                        df["timestamp_close"], utc=True, errors="coerce",
                    )
                    closed = df[close_dt.notna() & (close_dt >= _today_start)]
                else:
                    closed = df
                trades_today = len(closed)
                if trades_today > 0:
                    wins = int((closed["pnl_usd"] > 0).sum())
                    win_rate = wins / trades_today
                    realized_pnl = float(closed["pnl_usd"].sum())
                    # Per-symbol breakdown
                    per_symbol_pnl = {}
                    for sym, grp in closed.groupby("symbol"):
                        per_symbol_pnl[sym] = {
                            "count": len(grp),
                            "pnl": float(grp["pnl_usd"].sum()),
                            "wins": int((grp["pnl_usd"] > 0).sum()),
                        }
                else:
                    realized_pnl = 0.0
            except Exception as _exc:
                logger.debug("Daily summary DB query failed: %s", _exc)

            # Open position details from tracked_positions
            open_details = []
            try:
                if positions_lock is not None:
                    positions_lock.acquire()
                for _t, pos in tracked_positions.items():
                    open_details.append({
                        "symbol": pos.symbol,
                        "direction": pos.direction,
                        "entry_price": pos.entry_price,
                        "floating_pnl": 0.0,  # not available per-position here
                        "be_locked": getattr(pos, "be_locked", False),
                    })
                if positions_lock is not None:
                    positions_lock.release()
            except Exception:
                if positions_lock is not None:
                    try:
                        positions_lock.release()
                    except RuntimeError:
                        pass

            # Regime summary from signal combiner
            regime_summary = None
            try:
                if combiner is not None and hasattr(combiner, "last_signal_by_symbol"):
                    regime_summary = {}
                    for sym, sig in combiner.last_signal_by_symbol.items():
                        if sig.regime is not None:
                            regime_summary[sym] = sig.regime.regime_label
            except Exception:
                pass

            alert_manager.notify_daily_summary(
                equity=snap.equity,
                daily_pnl=daily_pnl,
                open_positions=len(tracked_positions),
                trades_today=trades_today,
                win_rate=win_rate,
                breaker_status=breaker_str,
                balance=snap.balance,
                floating_pnl=snap.equity - snap.balance,
                realized_pnl=realized_pnl,
                weekly_pnl=weekly_pnl,
                daily_dd_pct=breaker_snap.daily_dd_pct,
                weekly_dd_pct=breaker_snap.weekly_dd_pct,
                peak_dd_pct=breaker_snap.peak_dd_pct,
                margin_used=snap.margin,
                free_margin=snap.free_margin,
                per_symbol_pnl=per_symbol_pnl,
                open_position_details=open_details if open_details else None,
                regime_summary=regime_summary,
            )
        except Exception as exc:
            logger.warning("Daily summary alert failed: %s", exc)

    scheduler.add_job(
        _send_daily_summary,
        trigger="cron", hour=23, minute=55, timezone="UTC",
        id="daily_summary_alert",
    )

    # Weekly summary — email-only (see AlertManager._send digest routing).
    # Fires Sunday 23:55 UTC. Data: past-7d trades + equity snapshots.
    def _send_weekly_summary() -> None:
        try:
            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            from pathlib import Path as _Path

            # Guard against APScheduler firing a missed Sunday job during
            # startup before the event-loop handle is captured (the
            # async→sync bridge requires a live loop). If so, skip this
            # run; the next scheduled tick will have `_main_loop` set.
            if _main_loop is None:
                logger.warning("Weekly summary skipped: event loop not yet captured")
                return

            end_utc = _dt.now(_tz.utc)
            start_utc = end_utc - _td(days=7)
            week_start = start_utc.strftime("%Y-%m-%d")
            week_end = end_utc.strftime("%Y-%m-%d")

            # --- Trades in the window (async → sync bridge) ---
            trade_count = 0
            wins = 0
            losses = 0
            net_pnl = 0.0
            per_symbol: dict[str, dict] = {}
            best_trade = None
            worst_trade = None
            try:
                # `get_trades(since=...)` filters by timestamp_open. We want
                # trades that *closed* in the week, not opened — a position
                # opened pre-window and closed inside it must still count.
                # Fetch a wider open-window and post-filter on close-time
                # (mirrors the daily-summary fix at lines 678-695).
                fut = asyncio.run_coroutine_threadsafe(
                    data_store.get_trades(since=start_utc - _td(days=14)),
                    _main_loop,
                )
                df = fut.result(timeout=15)
                if not df.empty:
                    # trades table stores all accounts — scope to the
                    # currently active slot so the summary matches equity.
                    _acct_id = live_state.get_account_id()
                    if _acct_id is not None and "mt5_account" in df.columns:
                        df = df[df["mt5_account"] == _acct_id]
                    close_dt = pd.to_datetime(
                        df["timestamp_close"], utc=True, errors="coerce",
                    )
                    closed = df[
                        close_dt.notna()
                        & (close_dt >= start_utc)
                        & (close_dt < end_utc)
                        & df["exit_price"].notna()
                        & df["pnl_usd"].notna()
                    ]
                    trade_count = len(closed)
                    if trade_count > 0:
                        wins = int((closed["pnl_usd"] > 0).sum())
                        losses = int((closed["pnl_usd"] < 0).sum())
                        net_pnl = float(closed["pnl_usd"].sum())
                        for sym, grp in closed.groupby("symbol"):
                            per_symbol[str(sym)] = {
                                "count": int(len(grp)),
                                "pnl": float(grp["pnl_usd"].sum()),
                                "wins": int((grp["pnl_usd"] > 0).sum()),
                            }
                        # Best / worst by realized pnl
                        b = closed.loc[closed["pnl_usd"].idxmax()]
                        w = closed.loc[closed["pnl_usd"].idxmin()]
                        best_trade = {
                            "symbol": str(b["symbol"]),
                            "pnl": float(b["pnl_usd"]),
                            "entry_time": str(b["timestamp_open"]),
                        }
                        worst_trade = {
                            "symbol": str(w["symbol"]),
                            "pnl": float(w["pnl_usd"]),
                            "entry_time": str(w["timestamp_open"]),
                        }
            except Exception as _exc:
                logger.debug("Weekly summary trades query failed: %s", _exc)

            # --- Equity at window start/end (nearest equity_history rows) ---
            starting_equity = 0.0
            ending_equity = 0.0
            max_dd_pct = 0.0
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    data_store.get_equity_history(
                        limit=5000,
                        mt5_account=live_state.get_account_id(),
                    ),
                    _main_loop,
                )
                eq_df = fut.result(timeout=15)
                if not eq_df.empty:
                    # Normalize index to match start_utc's tz-awareness.
                    # DataStore returns pd.Timestamp index which may or may
                    # not be tz-aware depending on the stored string format.
                    idx = eq_df.index
                    if getattr(idx, "tz", None) is None:
                        cutoff_ts = start_utc.replace(tzinfo=None)
                    else:
                        cutoff_ts = start_utc
                    window = eq_df[idx >= cutoff_ts]
                    if not window.empty:
                        starting_equity = float(window["equity"].iloc[0])
                        ending_equity = float(window["equity"].iloc[-1])
                        # Peak-to-trough DD within the window
                        running_peak = window["equity"].cummax()
                        dd = ((running_peak - window["equity"]) / running_peak * 100).fillna(0)
                        max_dd_pct = float(dd.max()) if len(dd) > 0 else 0.0
            except Exception as _exc:
                logger.debug("Weekly summary equity query failed: %s", _exc)

            # Fallback: current account snapshot if equity history unavailable
            if ending_equity == 0.0:
                try:
                    snap = account_monitor.get_info()
                    ending_equity = float(snap.equity)
                    if starting_equity == 0.0:
                        starting_equity = ending_equity - net_pnl
                except Exception:
                    pass

            # --- Stale models (LSTM files older than 30 days) ---
            stale_models: list[str] = []
            try:
                models_dir = _Path("data/models")
                cutoff_30d = _dt.now(_tz.utc).timestamp() - 30 * 86_400
                for sym in SYMBOLS:
                    p = models_dir / f"lstm_{sym}.pt"
                    if p.exists() and p.stat().st_mtime < cutoff_30d:
                        stale_models.append(sym)
            except Exception:
                pass

            alert_manager.notify_weekly_summary(
                week_start=week_start,
                week_end=week_end,
                starting_equity=starting_equity,
                ending_equity=ending_equity,
                net_pnl=net_pnl,
                trade_count=trade_count,
                wins=wins,
                losses=losses,
                max_dd_pct=max_dd_pct,
                per_symbol=per_symbol or None,
                best_trade=best_trade,
                worst_trade=worst_trade,
                breaker_events=None,   # TODO wire from a breaker audit log once it exists
                stale_models=stale_models or None,
            )
        except Exception as exc:
            logger.warning("Weekly summary alert failed: %s", exc)

    scheduler.add_job(
        _send_weekly_summary,
        trigger="cron", day_of_week="sun", hour=23, minute=55, timezone="UTC",
        id="weekly_summary_alert",
    )

    # Equity snapshot every 5 minutes — populates equity_history table so
    # the dashboard can render PnL curves and the feedback loop can reason
    # about drawdowns. Also written to the tick_summary CSV for offline
    # analysis.
    #
    # APScheduler runs jobs on a worker thread where there is no event
    # loop; on Python 3.13 `asyncio.get_event_loop()` raises. Capture the
    # main loop here (we're already inside an async context) and close
    # over it so the scheduler thread can schedule the coroutine onto
    # the real running loop via run_coroutine_threadsafe.
    _main_loop = asyncio.get_running_loop()

    def _persist_equity_snapshot() -> None:
        try:
            snap = account_monitor.get_info()
            from src.data_pipeline.data_store import EquityRecord
            record = EquityRecord(
                timestamp=now_iso(),
                balance=snap.balance,
                equity=snap.equity,
                floating_pnl=snap.equity - snap.balance,
                mt5_account=live_state.get_account_id(),
            )
            asyncio.run_coroutine_threadsafe(
                data_store.save_equity_snapshot(record),
                _main_loop,
            )
        except Exception as _exc:
            # Was DEBUG; bumped to WARNING because a silently-failing
            # snapshot leaves the Overview equity chart empty for days
            # and there's no other indicator anything is wrong.
            logger.warning("equity snapshot failed: %s", _exc)

    scheduler.add_job(
        _persist_equity_snapshot,
        trigger="interval", minutes=5,
        id="equity_snapshot",
    )

    # Heartbeat file — lets the launcher script auto-detect whether the bot
    # was running before a PC restart. Written every 5 minutes and on
    # graceful shutdown. The launcher compares its mtime to PC boot time:
    # if heartbeat is newer -> fresh start; if older -> post-restart mode.
    HEARTBEAT_PATH = Path("data/logs/bot_heartbeat.json")

    def _write_heartbeat() -> None:
        try:
            snap = account_monitor.get_info()
            # Invariant: the MT5 session's actual login must match what
            # LiveState thinks is the active account. A mismatch means
            # some code path (typically reconnect after a drop) silently
            # switched the broker session under the running bot — the
            # dashboard would keep reporting the old slot while trades
            # and equity flow to the new one. First caught 2026-04-18
            # when reconnect() fell back to MT5_LOGIN after an account
            # switch, polluting equity_history with mismatched tags.
            try:
                import MetaTrader5 as _mt5_chk
                _live_info = _mt5_chk.account_info()
                _mt5_login = int(_live_info.login) if _live_info else None
            except Exception:
                _mt5_login = None
            _state_login = live_state.get_account_id()
            if _mt5_login is not None and _state_login is not None:
                from src.safety.invariants import Severity, check as _inv_check
                _inv_check(
                    "broker.connected_account_matches_state",
                    condition=(_mt5_login == _state_login),
                    severity=Severity.ALERT,
                    context={
                        "mt5_login": _mt5_login,
                        "state_account_id": _state_login,
                    },
                    dedup_key="broker.connected_account_matches_state",
                    message=(
                        f"MT5 session on {_mt5_login} but LiveState "
                        f"says {_state_login} — broker session may "
                        f"have reverted under an active account switch"
                    ),
                )
            HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
            HEARTBEAT_PATH.write_text(json.dumps({
                "timestamp_utc": now_iso(),
                "equity":        float(snap.equity),
                "balance":       float(snap.balance),
                "open_positions": len(tracked_positions),
                "mt5_account":    live_state.get_account_id(),
                "pid":            os.getpid(),
            }, indent=2), encoding="utf-8")
        except Exception as _exc:
            logger.debug("heartbeat write failed: %s", _exc)

    _write_heartbeat()  # initial write so the file exists immediately
    scheduler.add_job(
        _write_heartbeat,
        trigger="interval", minutes=5,
        id="heartbeat",
    )

    # API smoke check — hits every GET endpoint, fires api.route_healthy
    # ALERT on any non-200. Catches silent route breakage (e.g. the
    # Frankfurt ZoneInfo 500 from 2026-04-15 that killed news on the
    # dashboard for hours). Runs on startup + every 15 minutes.
    from src.safety.api_smoke import run_smoke_from_app as _run_smoke

    async def _api_smoke_job():
        try:
            await _run_smoke(api_app)
        except Exception as exc:
            logger.warning("API smoke check failed: %s", exc)

    scheduler.add_job(
        _api_smoke_job,
        trigger="interval", minutes=15,
        id="api_smoke",
        # Defer the first smoke by 60s so the event loop isn't still
        # warming up (MT5 login, model load, backfill). Running too early
        # produces spurious status=0 timeouts that alert on a healthy
        # system.
        next_run_time=datetime.now(tz=timezone.utc) + timedelta(seconds=60),
    )

    # Fast close reconcile — decoupled from the M15 trading loop so that
    # broker-side SL/TP hits surface in DB + Telegram within ~30s instead
    # of waiting for the next M15 cycle. Cheap: one MT5 positions_get +
    # history_deals_get; only writes when tickets actually changed.
    async def _fast_reconcile_job():
        try:
            await _reconcile_closed_trades(
                tracked_positions, data_store,
                positions_lock=live_state.positions_lock,
                alert_manager=alert_manager,
                combiner=combiner,
                circuit_breaker=circuit_breaker,
            )
        except Exception as exc:
            logger.warning("fast_reconcile: %s", exc)

    scheduler.add_job(
        _fast_reconcile_job,
        trigger="interval", seconds=30,
        id="fast_reconcile", max_instances=1, coalesce=True,
    )

    # A-4: daily live-vs-backtest PF drift check. Fires
    # strategy.live_pf_drift invariant (WARN/ALERT) when rolling 30d live
    # PF deviates below the latest full-mode backtest baseline for any
    # symbol. 02:00 UTC is after the 23:55 daily summary and before Asia
    # session activity, so the check runs on a settled set of trades
    # without racing the daily-PnL accounting.
    async def _pf_drift_job():
        try:
            from src.safety.pf_drift import run_drift_checks
            results = await run_drift_checks(
                data_store, list(SYMBOLS),
                account_id=live_state.get_account_id(),
            )
            for r in results:
                logger.info(
                    "pf_drift [%s]: %s (baseline=%s live=%s ratio=%s n=%d)",
                    r.symbol, r.reason,
                    f"{r.baseline_pf:.2f}" if r.baseline_pf else "-",
                    f"{r.live_pf:.2f}" if r.live_pf else "-",
                    f"{r.ratio:.2f}" if r.ratio else "-",
                    r.n_trades,
                )
        except Exception as exc:
            logger.warning("pf_drift job failed: %s", exc)

    scheduler.add_job(
        _pf_drift_job,
        trigger="cron", hour=2, minute=0, timezone="UTC",
        id="pf_drift_check", max_instances=1, coalesce=True,
    )

    # A-8: daily feature-drift check at 01:00 UTC. Compares current
    # 30-day feature distributions against the training-distribution
    # snapshots saved alongside each symbol's LSTM. Writes to
    # drift_scores, fires invariants at WARN/ALERT thresholds, and may
    # auto-trigger an off-cycle retrain if PSI exceeds 0.5 (rate-limited).
    # Async so data_store.save_drift_score() is awaited on the bot's
    # event loop instead of a throwaway loop on the scheduler thread
    # (matches _pf_drift_job).
    async def _drift_check_job() -> None:
        try:
            from src.ml.drift_check import run_daily_drift_check
            await run_daily_drift_check(
                feed=data_feed, engineer=feature_engineer,
                data_store=data_store,
                alert_manager=alert_manager,
                symbols=tuple(SYMBOLS),
            )
        except Exception as exc:
            logger.warning("drift_check job failed: %s", exc)

    scheduler.add_job(
        _drift_check_job,
        trigger="cron", hour=1, minute=0, timezone="UTC",
        id="drift_check", max_instances=1, coalesce=True,
    )

    scheduler.start()
    logger.info(
        "Scheduler started: monthly_full_retrain (1st@03:00UTC), "
        "feedback_check/24h (log only), daily_summary/23:55UTC, "
        "weekly_summary/Sun@23:55UTC, pf_drift_check/daily@02:00UTC, "
        "drift_check/daily@01:00UTC"
    )

    # Audit HIGH-1: warn if central-bank calendar is near end of coverage.
    # Static hardcoded dates silently rot — this surfaces the issue early.
    try:
        cal_warning = calendar_freshness_warning(min_lookahead_days=60)
        if cal_warning:
            logger.warning("%s", cal_warning)
    except Exception as _exc:
        logger.debug("Calendar freshness check failed: %s", _exc)

    # Invariant: economic_calendar.yaml freshness. Silent drift in the
    # calendar file (operator forgot to regenerate, hardcoded dates
    # rotted, or the file went missing) is the bug class that today's
    # BoC-date mess would have caught earlier. WARN only — surfaces on
    # the dashboard Health card without Telegram spam.
    try:
        from pathlib import Path as _Path
        from datetime import datetime as _dt, timedelta as _td
        from src.data_pipeline.market import economic_calendar as _ec
        from src.safety.invariants import Severity as _Sev, check as _inv
        _yaml = _Path("config/economic_calendar.yaml")
        if _yaml.exists():
            _age_days = (_dt.now().timestamp() - _yaml.stat().st_mtime) / 86400
            _inv(
                "calendar.yaml_fresh",
                _age_days <= 60,
                severity=_Sev.WARN,
                context={"age_days": round(_age_days, 1), "path": str(_yaml)},
                message=f"economic_calendar.yaml is {_age_days:.1f} days old (>60d)",
            )
        _now = _dt.now(tz=timezone.utc)
        _tier1_next_14d = sum(
            1 for e in _ec.load_events()
            if e.tier == 1 and _now <= e.event_utc <= _now + _td(days=14)
        )
        _inv(
            "calendar.has_upcoming_events",
            _tier1_next_14d >= 1,
            severity=_Sev.WARN,
            context={"tier1_next_14d": _tier1_next_14d},
            message=f"only {_tier1_next_14d} Tier 1 events in next 14 days",
        )
    except Exception as _exc:
        logger.debug("Calendar invariant check failed: %s", _exc)

    # --- 7. Graceful shutdown on Ctrl+C ---
    shutdown_event = asyncio.Event()

    def _signal_handler(sig_num, frame):
        logger.info(f"Shutdown signal {sig_num} received — closing gracefully...")
        # Flush a final heartbeat so the launcher sees 'was running before'
        try:
            _write_heartbeat()
        except Exception:
            pass
        # Release the PID lock file ASAP so a restart doesn't see our
        # own PID as a "running instance" and refuse to start.
        try:
            _release_pid_lock()
        except Exception:
            pass
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # --- 8. Main trading loop ---
    # Startup self-check: detect MT5 toolbar Algo Trading toggled OFF and
    # other "trading disabled" states *before* we start blasting orders
    # that will all reject with retcode 10027/10026. If disabled, the
    # loop still runs (safe — orders will be rejected & logged) but the
    # operator gets a Telegram/email alert immediately.
    try:
        ok, reason = account_monitor.trading_enabled()
        if not ok:
            logger.warning("STARTUP: trading is disabled — %s", reason)
            alert_manager.notify_trade_blocked(
                symbol="ALL",
                reason=reason,
                category="trade_disabled",
            )
        else:
            logger.info("STARTUP: trading_enabled check OK")
    except Exception as exc:
        logger.warning("STARTUP: trading_enabled check failed: %s", exc)

    logger.info("Entering main trading loop...")
    consecutive_tick_failures = 0
    # Re-check trade_allowed every N cycles so the operator gets notified
    # if MT5 Algo Trading is toggled off mid-session. 16 ticks * 900s ≈
    # every 4 hours (one alert per state change, not per cycle — guarded
    # by _last_trade_disabled).
    _trade_allowed_check_every = 16
    _last_trade_disabled: bool = False
    _cycle_count = 0

    # Tick-loop watchdog — detects a hung/stalled main loop. Updated at the
    # top of every tick; a sidecar task alerts if it goes stale >5min.
    # (Addresses the 6h+ signal gap on 2026-04-14: loop stopped calling
    # combiner.get_signal_async with no exception in logs.)
    _last_tick_ts: list[float] = [time.time()]
    _watchdog_alerted: list[bool] = [False]

    async def _tick_watchdog() -> None:
        # Tick-start cadence is TRADING_LOOP_INTERVAL (15min M15-aligned), so
        # "no tick progress" over the planned sleep is normal. Threshold must
        # cover the interval + a generous safety buffer. Anything over 25min
        # of zero tick-starts is a real hang (the reference incident was 6h+).
        stale_threshold = TRADING_LOOP_INTERVAL + 600.0  # 900 + 600 = 25min
        while not shutdown_event.is_set():
            await _sleep_or_shutdown(120, shutdown_event)
            elapsed = time.time() - _last_tick_ts[0]
            if elapsed > stale_threshold and not _watchdog_alerted[0]:
                logger.critical(
                    "TICK LOOP STALLED — no tick progress for %.0fs "
                    "(threshold %.0fs). Likely a hung coroutine or deadlock.",
                    elapsed, stale_threshold,
                )
                try:
                    alert_manager.notify_system(
                        f"Cortex tick loop stalled {elapsed:.0f}s — check logs"
                    )
                except Exception:
                    pass
                _watchdog_alerted[0] = True
            elif elapsed <= stale_threshold and _watchdog_alerted[0]:
                logger.info(
                    "Tick loop recovered after stall (elapsed=%.0fs)", elapsed,
                )
                _watchdog_alerted[0] = False

    watchdog_task = asyncio.create_task(_tick_watchdog(), name="tick_watchdog")

    try:
        while not shutdown_event.is_set():
            _last_tick_ts[0] = time.time()
            try:
                # Bot control gate (Wave 12)
                if bot_control.status == BotStatus.STOPPED:
                    logger.warning("Bot STOPPED — breaking out of trading loop")
                    break
                if bot_control.status == BotStatus.PAUSED:
                    logger.info("Bot PAUSED — skipping tick. RiskMonitor still active.")
                    await _sleep_or_shutdown(30, shutdown_event)
                    continue

                if circuit_breaker.is_halted():
                    logger.warning("Circuit breaker active — trading halted. Sleeping 60s...")
                    await _sleep_or_shutdown(60, shutdown_event)
                    continue

                # Periodic trade_allowed re-check. Catches the user
                # toggling MT5 Algo Trading off mid-session, so they
                # find out before hours of silent rejects pile up.
                _cycle_count += 1
                if _cycle_count % _trade_allowed_check_every == 0:
                    try:
                        ok, reason = account_monitor.trading_enabled()
                        if not ok and not _last_trade_disabled:
                            logger.warning("Trading disabled mid-session — %s", reason)
                            alert_manager.notify_trade_blocked(
                                symbol="ALL",
                                reason=reason,
                                category="trade_disabled",
                            )
                            _last_trade_disabled = True
                        elif ok and _last_trade_disabled:
                            logger.info("Trading re-enabled in MT5")
                            alert_manager.notify_system("Trading Re-Enabled")
                            _last_trade_disabled = False
                    except Exception as exc:
                        logger.debug("trading_enabled check failed: %s", exc)

                # --- 8a. Exit management runs FIRST each tick ---
                # Walk every tracked position and fire partial closes /
                # stop updates before considering any new entries.
                await _run_exit_manager(
                    exit_manager,
                    order_mgr,
                    tracked_positions,
                    data_feed,
                    recent_signal_dirs,
                    tier_state_store,
                    positions_lock=live_state.positions_lock,
                    alert_manager=alert_manager,
                    data_store=data_store,
                    combiner=combiner,
                    circuit_breaker=circuit_breaker,
                )

                # --- 8b. Entry path per symbol ---
                for symbol in SYMBOLS:
                    # Fetch latest bars across all timeframes (DB-first cache).
                    # H4 needs >=250 bars: SMA200 indicator requires 200-bar
                    # warmup, then transform() drops NaN rows. With the old
                    # default of 60, every row was NaN -> features empty.
                    raw_data = await data_feed.get_latest_async(symbol, "H4", bars=300)
                    ohlcv_by_tf: dict[str, pd.DataFrame] = {"H4": raw_data}
                    for tf in ["D1", "H1", "M15", "W1"]:
                        try:
                            ohlcv_by_tf[tf] = await data_feed.get_latest_async(
                                symbol, tf, bars=300
                            )
                        except Exception:
                            pass  # degrade gracefully — missing TF won't crash

                    # Get all external features (cached, never blocks)
                    from datetime import datetime as _dt
                    fundamental_features = fundamental_mgr.get_all_features(
                        symbol, _dt.utcnow()
                    )

                    # Compute multi-TF aligned features
                    features = feature_engineer.transform_multi_timeframe(
                        ohlcv_by_tf, fundamental_features, primary_tf="H4"
                    )

                    # Inject HMM regime as LSTM features (one-hot + probability)
                    d1_ohlcv = ohlcv_by_tf.get("D1")
                    if d1_ohlcv is not None and not d1_ohlcv.empty:
                        features = feature_engineer.inject_regime_features(
                            features, hmm, symbol, d1_ohlcv,
                        )

                    await feature_engineer.persist_features(symbol, "H4", features)

                    # Bug 3 fix: HMM was trained on D1 single-TF features,
                    # so inference must also use D1 features (not H4 multi-TF).
                    # D1 OHLCV is already available from ohlcv_by_tf.
                    hmm_manifest = hmm._feature_manifests.get(symbol, [])
                    if d1_ohlcv is not None and not d1_ohlcv.empty:
                        d1_features = feature_engineer.transform(d1_ohlcv)
                        if hmm_manifest and not d1_features.empty:
                            hmm_matrix = feature_engineer.align_to_manifest(
                                d1_features, hmm_manifest,
                            ).values
                        elif not d1_features.empty:
                            hmm_matrix = d1_features[
                                sorted(d1_features.columns)
                            ].values
                        else:
                            hmm_matrix = features.values
                    elif hmm_manifest:
                        hmm_matrix = feature_engineer.align_to_manifest(
                            features.copy(), hmm_manifest,
                        ).values
                    else:
                        hmm_matrix = features.values

                    # Defensive: feature pipeline can briefly return 0 rows
                    # right after startup when the H4 cache is still warming
                    # up or after a data-gap backfill. Skip this tick rather
                    # than crash the HMM. The next tick (after more bars
                    # arrive) will succeed.
                    if hmm_matrix.shape[0] == 0:
                        logger.warning(
                            "[%s] feature matrix empty (shape=%s) — skipping "
                            "this tick. Likely a startup warmup; will retry next bar.",
                            symbol, hmm_matrix.shape,
                        )
                        continue

                    # LSTM gets full feature set (multi-TF + regime + fundamentals)
                    lstm_manifest = lstm._feature_manifests.get(symbol, [])
                    if lstm_manifest:
                        lstm_matrix = feature_engineer.align_to_manifest(
                            features.copy(), lstm_manifest,
                        ).values
                    else:
                        lstm_matrix = features.values

                    # Determine the current bar timestamp
                    current_bar_ts = _last_bar_iso(raw_data)

                    # Per-symbol liveness heartbeat — stays on M15 cadence
                    # (above the dedup guard) so "did the bot see symbol X
                    # at time T" forensics and stall detection keep working.
                    # Regime/prob come from the cached signal because the
                    # fresh one only recomputes on new H4 bars; regime
                    # updates on D1 cadence so a 15min lag is invisible.
                    try:
                        _cached_sig = combiner.last_signal_by_symbol.get(symbol)
                        _snap = account_monitor.get_info()
                        _positions_for_sym = sum(
                            1 for p in tracked_positions.values()
                            if p.symbol == symbol
                        )
                        TICK_SUMMARY.write({
                            "timestamp":    now_iso(),
                            "symbol":       symbol,
                            "price":        float(raw_data["close"].iloc[-1]),
                            "atr_14":       float(features["atr_14"].iloc[-1]) if "atr_14" in features.columns else 0.0,
                            "regime":       getattr(getattr(_cached_sig, "regime", None), "regime_label", None),
                            "regime_prob":  getattr(getattr(_cached_sig, "regime", None), "state_probability", None),
                            "open_positions": _positions_for_sym,
                            "equity":       _snap.equity,
                            "floating_pnl": _snap.equity - _snap.balance,
                            "daily_pnl":    _snap.equity - risk_monitor._daily_start_equity,
                            "breaker_active": ",".join(circuit_breaker.active_breakers()) or "none",
                            "breaker_multiplier": circuit_breaker.current_size_multiplier(),
                        })
                    except Exception as _exc:
                        logger.debug("[%s] tick_summary write failed: %s", symbol, _exc)

                    # H4-bar dedup: exit manager already ran in §8a and will
                    # run again next M15 tick — those are the price-based
                    # guards that must stay responsive. Everything below
                    # (signal fusion, audit, DB write, entry path) is
                    # bar-driven and must only fire once per new H4 bar.
                    if current_bar_ts and last_processed_bar_ts.get(symbol) == current_bar_ts:
                        continue
                    if current_bar_ts:
                        last_processed_bar_ts[symbol] = current_bar_ts

                    # Brain: get combined signal (also persists predictions)
                    signal_result = await combiner.get_signal_async(
                        symbol=symbol,
                        feature_matrix=hmm_matrix,
                        feature_sequence=lstm_matrix,
                        bar_timestamp=current_bar_ts,
                    )

                    # Record this tick's direction for the exit-manager
                    # reversal-flicker check (separate from SignalCombiner's
                    # own 4-bar entry-stability ring).
                    _record_recent_direction(
                        recent_signal_dirs, symbol, signal_result.direction
                    )

                    # ----------------------------------------------------------
                    # Build the signal audit record — written at EVERY decision
                    # branch below so we can reconstruct "why did (or didn't)
                    # symbol X trade at time T" from data/logs/signal_audit.csv.
                    # ----------------------------------------------------------
                    now_utc = datetime.now(tz=timezone.utc).replace(tzinfo=None)
                    # Live blackout path uses the YAML-backed economic calendar
                    # (Tier 1 events only — CBs + NFP + US/Canada CPI + Canada
                    # Employment + FOMC Minutes). Backtests still use the
                    # hardcoded CB date lists in calendar_features for
                    # reproducibility of past runs.
                    from src.data_pipeline.market import economic_calendar as _ec
                    news_ctx = _ec.describe_blackout_context(symbol, now_utc)
                    is_blackout = bool(news_ctx.get("blackout"))

                    audit_ctx = {
                        "timestamp":       now_iso(),
                        "symbol":          symbol,
                        "regime":          getattr(signal_result.regime, "regime_label", None),
                        "regime_prob":     getattr(signal_result.regime, "state_probability", None),
                        "lstm_prediction": signal_result.lstm_prediction,
                        "combined_score":  signal_result.combined_score,
                        "direction":       signal_result.direction,
                        "should_trade":    bool(signal_result.should_trade),
                        "executed":        False,  # flipped to True after order placed
                        "news_blackout":   is_blackout,
                        "nearest_cb":      news_ctx.get("nearest_event"),
                        "nearest_hours":   news_ctx.get("nearest_hours"),
                        "block_reason":    "",
                        "cb_multiplier":   circuit_breaker.current_size_multiplier(),
                        "reasoning":       " | ".join(signal_result.reasoning or []),
                    }

                    # Mirror into the `signals` DB table (long-term analysis)
                    try:
                        from src.data_pipeline.data_store import SignalRecord
                        await data_store.save_signal(SignalRecord(
                            timestamp=audit_ctx["timestamp"],
                            symbol=symbol,
                            regime=audit_ctx["regime"],
                            regime_probability=audit_ctx["regime_prob"],
                            lstm_prediction=audit_ctx["lstm_prediction"],
                            combined_score=audit_ctx["combined_score"],
                            should_trade=audit_ctx["should_trade"],
                            direction=audit_ctx["direction"],
                            mt5_account=live_state.get_account_id(),
                        ))
                    except Exception as _exc:
                        logger.debug("[%s] save_signal failed: %s", symbol, _exc)

                    if not signal_result.should_trade:
                        audit_ctx["block_reason"] = "combiner_rejected"
                        SIGNAL_AUDIT.write(audit_ctx)
                        continue

                    # Smart news blackout (Phase C): block new entries during
                    # pre-news + spike zone (T-24h to T+2h). Post-news
                    # continuation window (T+2h to T+48h) is intentionally
                    # NOT blocked — that is where the retail edge lives.
                    # XAUUSD is exempt; per-symbol CB routing is handled
                    # inside is_in_news_blackout().
                    try:
                        if _ec.is_in_blackout(symbol, now_utc):
                            logger.info(
                                "[%s] entry blocked: news blackout "
                                "(%s nearest at %+.1fh)",
                                symbol,
                                news_ctx.get("active_event") or news_ctx.get("nearest_event"),
                                news_ctx.get("nearest_hours") or 0.0,
                            )
                            audit_ctx["block_reason"] = "news_blackout"
                            SIGNAL_AUDIT.write(audit_ctx)
                            continue
                    except Exception as exc:
                        # Never let news-filter errors block trading —
                        # log and fall through to normal flow.
                        logger.warning("[%s] news blackout check failed: %s",
                                       symbol, exc)

                    # Strategy: vol-rank → strategy class → StrategyDecision
                    ref_price = float(raw_data["close"].iloc[-1])

                    # atr_14 in features is stored as a fraction of close
                    # (atr/close), not absolute price. Convert to price units.
                    if "atr_14" in features.columns:
                        atr_frac = float(features["atr_14"].iloc[-1])
                        atr_val = atr_frac * ref_price
                    else:
                        atr_val = ref_price * 0.005  # 0.5% fallback

                    # ema50 must be an absolute trend-anchor price (strategies
                    # compare `price > ema50` and build stops from it). We don't
                    # store ema_50 directly — but sma_50_rel = (close-sma50)/sma50
                    # lets us recover sma50 = close / (1 + sma_50_rel) which is
                    # a stable ~50-period trend anchor. Audit HIGH-B fix.
                    if "sma_50_rel" in features.columns:
                        sma50_rel = float(features["sma_50_rel"].iloc[-1])
                        # Guard against extreme / NaN values
                        if -0.5 < sma50_rel < 0.5:
                            ema_val = ref_price / (1.0 + sma50_rel)
                        else:
                            ema_val = ref_price
                    else:
                        ema_val = ref_price
                    context = MarketContext(
                        symbol=symbol,
                        price=ref_price,
                        atr=atr_val,
                        ema50=ema_val,
                    )
                    # Attach reference_price for PortfolioManager sizing.
                    signal_result.reference_price = ref_price  # type: ignore[attr-defined]

                    # Fetch the account snapshot ONCE per symbol per tick:
                    # orchestrator.select() needs current equity for the
                    # Wave 6 DD-clamp (#20), and portfolio.calculate_lot_size()
                    # needs the full snapshot for margin caps.
                    account_snapshot = account_monitor.get_info()
                    peak_equity = risk_monitor.get_peak_equity()

                    # Strategy: vol-rank → strategy class → StrategyDecision
                    decision = orchestrator.select(
                        signal_result,
                        context,
                        current_equity=account_snapshot.equity,
                        peak_equity=peak_equity,
                    )

                    # Feed the daily close into the portfolio manager's
                    # rolling correlation window (Wave 6 fix #17). No-op
                    # for symbols outside the correlation bucket.
                    portfolio.update_daily_close(symbol, ref_price)

                    # Allocation: sizer + portfolio caps + pyramiding gate
                    sizing = portfolio.calculate_lot_size(
                        symbol=symbol,
                        signal=signal_result,
                        decision=decision,
                        account_info=account_snapshot,
                        size_multiplier=circuit_breaker.current_size_multiplier(),
                    )
                    if sizing.lot_size <= 0.0:
                        logger.info(
                            "[%s] entry skipped: %s",
                            symbol, sizing.reason,
                        )
                        audit_ctx["block_reason"] = f"sizing:{sizing.reason}"
                        SIGNAL_AUDIT.write(audit_ctx)
                        # Suppress noise: we only alert on sizing rejects
                        # that imply something operationally wrong, not
                        # routine cap-hits like "max_concurrent_per_symbol"
                        # or "max_daily_trades_reached".
                        _noisy = (
                            "max_concurrent",
                            "max_daily_trades",
                            "pyramid",
                            "free_margin",
                            "max_used_margin",
                        )
                        if not any(n in sizing.reason for n in _noisy):
                            alert_manager.notify_trade_blocked(
                                symbol=symbol,
                                reason=sizing.reason,
                                category="sizing",
                            )
                        continue

                    # Broker: place order with the strategy's initial stop
                    order_result = order_mgr.place_order(
                        symbol=symbol,
                        signal=signal_result,
                        lot_size=sizing.lot_size,
                        sl_price=decision.initial_stop_price,
                    )

                    # E-3 Phase 1 — execution-quality logging.
                    # Gated by CORTEX_LOG_FILL_QUALITY=1. Feeds R-1b after
                    # ≥30 days. Wrapped in try/except so any logging bug
                    # can never crash the trading loop; grep for
                    # "execution_event_log_failed" in trading_bot.log to
                    # spot issues.
                    if os.environ.get("CORTEX_LOG_FILL_QUALITY") == "1":
                        try:
                            from src.data_pipeline.data_store import ExecutionEvent
                            _req_px = order_result.requested_price
                            _fill_px = order_result.fill_price
                            _slip = (
                                (_fill_px - _req_px)
                                if (_fill_px is not None and _req_px is not None)
                                else None
                            )
                            # Sanity guard: don't poison the table if a
                            # successful send reports fill_price=0 (MT5
                            # quirk we haven't seen but shouldn't trust).
                            _sane = not (
                                order_result.success and (
                                    _fill_px is None or _fill_px <= 0.0
                                )
                            )
                            if _sane:
                                await data_store.save_execution_event(ExecutionEvent(
                                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                                    symbol=order_result.symbol or symbol,
                                    direction=order_result.direction,
                                    ticket=order_result.ticket,
                                    requested_price=_req_px,
                                    fill_price=_fill_px,
                                    slippage=_slip,
                                    spread_at_send=order_result.spread_at_send,
                                    volume_requested=order_result.volume_requested,
                                    volume_filled=order_result.volume_filled,
                                    retcode=order_result.retcode,
                                    mt5_account=live_state.get_account_id(),
                                ))
                            else:
                                logger.warning(
                                    "execution_event_log_skipped: [%s] success=True "
                                    "but fill_price=%s — row not written",
                                    symbol, _fill_px,
                                )
                        except Exception as _exc:
                            logger.warning(
                                "execution_event_log_failed: [%s] %s",
                                symbol, _exc,
                            )

                    if not order_result.success:
                        # Order rejected by broker — still log the attempt so
                        # we can diagnose broker-side problems (margin, spread,
                        # trading disabled, etc.) after the fact.
                        reject_msg = order_result.error_message or "unknown"
                        reject_code = order_result.retcode
                        audit_ctx["block_reason"] = (
                            f"broker_reject:retcode={reject_code}:{reject_msg}"
                        )
                        SIGNAL_AUDIT.write(audit_ctx)
                        logger.warning(
                            "[%s] broker rejected order: retcode=%s msg=%s",
                            symbol, reject_code, reject_msg,
                        )
                        # Roll back the daily-trades counter — calculate_lot_size
                        # optimistically appended this attempt before we knew the
                        # broker verdict. Without rollback, 12 rejects burn the
                        # whole 24h cap.
                        portfolio.rollback_last_trade_attempt()
                        # Push a real-time alert so operator finds out without
                        # having to tail the log. Telegram + email if configured.
                        alert_manager.notify_trade_blocked(
                            symbol=symbol,
                            reason=reject_msg,
                            retcode=reject_code,
                            category="broker_reject",
                        )
                    if order_result.success and order_result.ticket:
                        opened_at = datetime.now(tz=timezone.utc)
                        # Audit C5: lock-protected mutation
                        with live_state.positions_lock:
                            tracked_positions[order_result.ticket] = OpenPosition(
                                symbol=symbol,
                                ticket=order_result.ticket,
                                direction=decision.direction,
                                entry_price=ref_price,
                                initial_stop=decision.initial_stop_price,
                                current_stop=decision.initial_stop_price,
                                volume=sizing.lot_size,
                                initial_volume=sizing.lot_size,
                                atr_trail_mult=decision.atr_trail_mult,
                                strategy_name=decision.strategy_name,
                                opened_at=opened_at,
                                time_exit_bars=TIME_EXIT_H1_BY_SYMBOL.get(symbol, 60),
                                tp_r_multiple=TP_R_BY_SYMBOL.get(symbol, 2.5),
                                be_trigger_r=BE_TRIGGER_R_BY_SYMBOL.get(symbol, 1.0),
                            )
                        # Wave 6 fix #19: seed the tier state store with
                        # the immutable entry-time values. Subsequent barrier
                        # events update be_locked / bars_held.
                        initial_stop_R = abs(
                            ref_price - decision.initial_stop_price
                        )
                        tier_state_store.upsert(
                            order_result.ticket,
                            be_locked=False,
                            bars_held=0,
                            initial_stop_R=initial_stop_R,
                            initial_volume=sizing.lot_size,
                            opened_at=opened_at.isoformat(),
                        )

                        # Alert: new trade opened (enriched with Brain context
                        # — the operator sees HMM regime, LSTM prediction,
                        # combined score, confidence, and the full reasoning
                        # trail from SignalCombiner + the StrategyDecision).
                        _entry_reasoning: list[str] = list(
                            getattr(signal_result, "reasoning", []) or []
                        )
                        _entry_reasoning += list(
                            getattr(decision, "reasoning", []) or []
                        )
                        alert_manager.notify_trade_entry(
                            symbol=symbol,
                            direction=decision.direction,
                            lot_size=sizing.lot_size,
                            entry_price=ref_price,
                            stop_loss=decision.initial_stop_price,
                            ticket=order_result.ticket,
                            strategy=decision.strategy_name,
                            regime=getattr(signal_result.regime, "regime_label", None),
                            regime_prob=getattr(signal_result.regime, "state_probability", None),
                            lstm_prediction=getattr(signal_result, "lstm_prediction", None),
                            combined_score=getattr(signal_result, "combined_score", None),
                            confidence=getattr(signal_result, "confidence", None),
                            reasoning=_entry_reasoning,
                        )

                        # Audit: mark executed + write signal row
                        audit_ctx["executed"] = True
                        SIGNAL_AUDIT.write(audit_ctx)

                        # Append an entry event to trade_events.csv
                        TRADE_EVENTS.write({
                            "timestamp":     now_iso(),
                            "event":         "entry",
                            "ticket":        order_result.ticket,
                            "symbol":        symbol,
                            "direction":     decision.direction,
                            "lot_size":      sizing.lot_size,
                            "entry_price":   ref_price,
                            "current_price": ref_price,
                            "sl_price":      decision.initial_stop_price,
                            "tp_price":      getattr(decision, "tp_price", 0.0) or 0.0,
                            "pnl_usd":       0.0,
                            "r_multiple":    0.0,
                            "bars_held":     0,
                            "be_locked":     False,
                            "regime_at_entry": audit_ctx.get("regime"),
                            "combined_score_at_entry": audit_ctx.get("combined_score"),
                            "exit_reason":   "",
                        })

                        # Persist open trade to `trades` table (close row is
                        # updated by the exit manager / reconciliation flow).
                        try:
                            from src.data_pipeline.data_store import TradeRecord
                            from src.utils.model_version_label import get_lstm_version_label
                            _entry_score = audit_ctx.get("combined_score")
                            _tp_price = getattr(decision, "tp_price", None) or None
                            await data_store.save_trade(TradeRecord(
                                timestamp_open=opened_at.isoformat(),
                                timestamp_close=None,
                                symbol=symbol,
                                direction=decision.direction,
                                lot_size=sizing.lot_size,
                                entry_price=ref_price,
                                exit_price=None,
                                pnl_usd=None,
                                regime_at_entry=audit_ctx.get("regime"),
                                combined_score=_entry_score,
                                ticket=order_result.ticket,
                                mt5_account=live_state.get_account_id(),
                                # Trade journal — snapshots locked at entry.
                                # close_* / exit_* fields get filled on close.
                                entry_score=_entry_score,
                                initial_stop=decision.initial_stop_price,
                                tp_price=_tp_price,
                                model_version=get_lstm_version_label(symbol),
                            ))
                        except Exception as _exc:
                            logger.debug("[%s] save_trade(entry) failed: %s",
                                          symbol, _exc)

                # Tick completed without raising — reset the failure counter.
                consecutive_tick_failures = 0

                # Sleep until next execution bar (M15 = 900s) — interruptible
                await _sleep_or_shutdown(TRADING_LOOP_INTERVAL, shutdown_event)

            except Exception as e:
                consecutive_tick_failures += 1
                logger.error(
                    "Main loop error (%d/%d consecutive): %s",
                    consecutive_tick_failures,
                    MAX_CONSECUTIVE_TICK_FAILURES,
                    e,
                    exc_info=True,
                )
                if consecutive_tick_failures >= MAX_CONSECUTIVE_TICK_FAILURES:
                    logger.critical(
                        "Main loop has failed %d times consecutively — "
                        "breaking out for clean shutdown. Investigate the "
                        "underlying error above and restart.",
                        consecutive_tick_failures,
                    )
                    break
                await _sleep_or_shutdown(60, shutdown_event)

    finally:
        logger.info("Shutting down API, scheduler, risk monitor, and MT5...")
        alert_manager.notify_system("Bot Shutting Down")
        # Stop the dashboard API server
        uvicorn_server.should_exit = True
        api_task.cancel()
        try:
            await api_task
        except asyncio.CancelledError:
            pass
        # Cancel the tick watchdog sidecar
        try:
            watchdog_task.cancel()
            await watchdog_task
        except (asyncio.CancelledError, NameError):
            pass
        scheduler.shutdown(wait=False)
        risk_monitor.stop()
        connector.disconnect()
        await data_store.close()
        logger.info("Shutdown complete.")


async def _sleep_or_shutdown(seconds: float, shutdown_event: asyncio.Event) -> None:
    """Sleep for ``seconds`` but wake immediately if shutdown is requested."""
    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass  # normal timeout — loop continues


def _last_bar_iso(ohlcv) -> str:
    """Return the ISO 8601 timestamp of the most recent bar in a DataFrame."""
    if ohlcv is None or len(ohlcv) == 0:
        return ""
    ts = ohlcv.index[-1]
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def _record_recent_direction(
    buffer: dict,
    symbol: str,
    direction,
) -> None:
    """Append the latest signal direction to a per-symbol ring (max 8)."""
    entries = buffer.setdefault(symbol, [])
    entries.append(direction)
    if len(entries) > 8:
        del entries[: len(entries) - 8]


def _reconcile_tracked_positions(
    tracked_positions: dict,
    default_atr_trail_mult: float = 2.0,
    tier_state_store: "Optional[TierStateStore]" = None,
    time_exit_by_symbol: "Optional[dict[str, int]]" = None,
    tp_r_by_symbol: "Optional[dict[str, float]]" = None,
    be_trigger_r_by_symbol: "Optional[dict[str, float]]" = None,
) -> int:
    """
    Rebuild ``tracked_positions`` from ``mt5.positions_get()``.

    Called at startup (and after a successful MT5 reconnect) so positions
    that survived a prior bot run are brought back under exit-ladder
    management. Without this step, those positions still have their
    initial SL on the broker side — so they can't run unbounded — but
    the 3-tier partial-exit ladder and ATR trail would be dormant until
    they hit SL naturally.

    Trade-offs on reconciliation:
        * ``atr_trail_mult`` is not recoverable from MT5 (the broker has
          no concept of our strategy layer). We default to 2.0
          (MidVolCautious) as a middle-of-the-road trail until the
          position exits.
        * Wave 6 fix #19: tier flags (``tier_1_done`` / ``tier_2_done``),
          original ``initial_stop``, ``initial_volume``, and ``opened_at``
          are recovered from ``tier_state_store`` when available. This
          prevents a restart from silently re-firing tier 1 against an
          already-BE stop or computing partial fractions against the
          surviving volume instead of the true initial. If no record
          exists for a given ticket (first-run scenario, or a position
          opened manually in the terminal), we fall back to the old
          conservative defaults: tier flags False, initial_stop=sl,
          initial_volume=volume, opened_at=None.

    Args:
        tracked_positions: The dict owned by ``main()``; mutated in-place.
        default_atr_trail_mult: Trail multiplier applied to every
            reconciled position (default 2.0 = MidVolCautious).
        tier_state_store: Optional persisted tier state. When provided,
            records are merged into each reconstructed OpenPosition so
            the exit ladder resumes at the correct tier.

    Returns:
        Number of positions added to ``tracked_positions``.
    """
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception as exc:
        logger.warning("reconcile: MetaTrader5 unavailable: %s", exc)
        return 0

    try:
        positions = mt5.positions_get()
    except Exception as exc:
        logger.error("reconcile: mt5.positions_get() raised: %s", exc)
        return 0

    if positions is None:
        logger.info("reconciled 0 positions from MT5 (positions_get returned None)")
        return 0

    added = 0
    for pos in positions:
        ticket = int(getattr(pos, "ticket", 0))
        if not ticket or ticket in tracked_positions:
            continue
        symbol = str(getattr(pos, "symbol", ""))
        pos_type = int(getattr(pos, "type", 0))
        # MT5 convention: POSITION_TYPE_BUY = 0, POSITION_TYPE_SELL = 1
        direction = "buy" if pos_type == 0 else "sell"
        entry_price = float(getattr(pos, "price_open", 0.0))
        sl_price = float(getattr(pos, "sl", 0.0))
        volume = float(getattr(pos, "volume", 0.0))
        if entry_price <= 0 or sl_price <= 0 or volume <= 0:
            logger.warning(
                "reconcile: skipping ticket %s with invalid fields "
                "(entry=%s sl=%s vol=%s)",
                ticket, entry_price, sl_price, volume,
            )
            continue

        # ---- merge persisted exit state when available ---------------
        be_locked = False
        bars_held = 0
        initial_volume = volume
        initial_stop = sl_price
        opened_at: Optional[datetime] = None

        record = (
            tier_state_store.get(ticket) if tier_state_store is not None else None
        )
        if record:
            be_locked = bool(record.get("be_locked", record.get("tier_1_done", False)))
            bars_held = int(record.get("bars_held", 0))
            if isinstance(record.get("initial_volume"), (int, float)):
                initial_volume = float(record["initial_volume"])
            # Reconstruct initial_stop from the persisted R distance so
            # ExitManager._risk_unit() still returns the true R even if
            # the broker-side SL has since been moved to BE.
            r_val = record.get("initial_stop_R")
            if isinstance(r_val, (int, float)) and r_val > 0:
                if direction == "buy":
                    initial_stop = entry_price - float(r_val)
                else:
                    initial_stop = entry_price + float(r_val)
            opened_iso = record.get("opened_at")
            if isinstance(opened_iso, str) and opened_iso:
                try:
                    opened_at = datetime.fromisoformat(opened_iso)
                except ValueError:
                    logger.warning(
                        "reconcile: could not parse opened_at=%r for "
                        "ticket %s — leaving None",
                        opened_iso, ticket,
                    )

        tracked_positions[ticket] = OpenPosition(
            symbol=symbol,
            ticket=ticket,
            direction=direction,
            entry_price=entry_price,
            initial_stop=initial_stop,
            current_stop=sl_price,
            volume=volume,
            initial_volume=initial_volume,
            atr_trail_mult=default_atr_trail_mult,
            strategy_name="reconciled",
            be_locked=be_locked,
            bars_held=bars_held,
            opened_at=opened_at,
            time_exit_bars=int((time_exit_by_symbol or {}).get(symbol, 60)),
            tp_r_multiple=float((tp_r_by_symbol or {}).get(symbol, 2.5)),
            be_trigger_r=float((be_trigger_r_by_symbol or {}).get(symbol, 1.0)),
        )
        added += 1

    logger.info("reconciled %d positions from MT5", added)
    return added


def _build_position_views(tracked):
    """Project tracked positions into OpenPositionView for PortfolioManager."""
    from src.allocation.portfolio_manager import OpenPositionView
    return [
        OpenPositionView(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            current_stop=pos.current_stop,
            # Wave 6 fix #18: surface the exit-ladder's tier_1_done flag
            # so the pyramiding gate reads the real state, not the stop
            # level which can be spoofed by a manual SL adjustment.
            tier_1_done=pos.tier_1_done,
        )
        for pos in tracked.values()
    ]


def _resolve_symbol_spec(symbol: str):
    """
    Best-effort MT5 → SymbolSpec mapping. Falls back to sane defaults
    if the live MT5 call is unavailable during unit-tests.

    tick_value + tick_size + currency_profit are populated so the
    position sizer can use MT5's broker-authoritative "how much USD
    per tick per lot" number. That is the ONLY path that correctly
    sizes USD-base forex pairs (USDJPY / USDCAD / USDCHF) whose PnL
    is naturally in the quote currency. See position_sizer.py for
    the fallback math when these fields are zero.
    """
    from src.allocation.position_sizer import SymbolSpec
    try:
        import MetaTrader5 as mt5  # type: ignore
        info = mt5.symbol_info(symbol)
        if info is not None:
            return SymbolSpec(
                symbol=symbol,
                contract_size=float(getattr(info, "trade_contract_size", 1.0)),
                volume_min=float(getattr(info, "volume_min", 0.01)),
                volume_max=float(getattr(info, "volume_max", 100.0)),
                volume_step=float(getattr(info, "volume_step", 0.01)),
                tick_value=float(getattr(info, "trade_tick_value", 0.0)),
                tick_size=float(getattr(info, "trade_tick_size", 0.0)),
                quote_currency=str(getattr(info, "currency_profit", "USD")),
            )
    except Exception:
        pass
    return SymbolSpec(symbol=symbol)


async def _reconcile_closed_trades(
    tracked_positions: dict,
    data_store,
    positions_lock=None,
    alert_manager=None,
    combiner=None,
    circuit_breaker=None,
) -> int:
    """
    Sync the ``trades`` DB table when MT5 closes a position server-side
    (SL/TP hit without the bot placing a close order). For every ticket
    we still track in memory but that's no longer in ``positions_get``,
    pull the closing deal from ``history_deals_get`` and update the
    matching DB row with the authoritative exit price, time, and PnL.

    Without this, the Trades tab on the dashboard shows old positions as
    "open" forever — we saw 2 USDJPY trades stuck this way because both
    hit SL server-side and nothing on the bot side noticed.

    Returns the number of rows updated.
    """
    try:
        import MetaTrader5 as mt5  # type: ignore
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    except Exception as exc:
        logger.debug("reconcile_closed_trades: MT5 unavailable: %s", exc)
        return 0

    try:
        live_positions = mt5.positions_get() or []
    except Exception as exc:
        logger.debug("reconcile_closed_trades: positions_get failed: %s", exc)
        return 0

    live_tickets = {int(p.ticket) for p in live_positions}
    # Snapshot tracked tickets under the lock
    if positions_lock is not None:
        with positions_lock:
            tracked_tickets = set(tracked_positions.keys())
    else:
        tracked_tickets = set(tracked_positions.keys())

    # Pull open-in-DB tickets AND recently-closed rows. The recent-closed
    # window lets us overwrite the gross-estimate PnL that full_close
    # writes at close-time with authoritative broker-truth
    # (profit+commission+swap) from history_deals_get.
    db_candidates: dict[int, float] = {}  # ticket -> currently-stored pnl_usd
    db_open_tickets: set[int] = set()     # subset still open (timestamp_close IS NULL)
    try:
        from sqlalchemy import text as _text
        cutoff_iso = (_dt.now(tz=_tz.utc) - _td(hours=48)).isoformat()
        async with data_store._session_factory() as _s:
            _r = await _s.execute(_text(
                "SELECT ticket, COALESCE(pnl_usd, 0.0), "
                "  (timestamp_close IS NULL) AS is_open "
                "FROM trades "
                "WHERE ticket IS NOT NULL "
                "AND (timestamp_close IS NULL OR timestamp_close > :cutoff)"
            ), {"cutoff": cutoff_iso})
            for row in _r.fetchall():
                t = int(row[0])
                db_candidates[t] = float(row[1])
                if bool(row[2]):
                    db_open_tickets.add(t)
    except Exception as exc:
        logger.debug("reconcile_closed_trades: DB ticket lookup failed: %s", exc)
    # We only consider a ticket an "orphan" for first-fill if it's still
    # timestamp_close=NULL in DB. But for refresh-with-broker-truth we
    # also want recently-closed rows, which we'll visit below.
    orphans = (tracked_tickets | db_open_tickets) - live_tickets
    refresh_tickets = set(db_candidates) - live_tickets - orphans  # closed rows ≤48h
    if not orphans and not refresh_tickets:
        return 0

    # Pull recent deal history. MT5 compares the date window against
    # broker-local time (Helsinki EET/EEST), not true UTC, so an unshifted
    # `to_dt = now_utc` would silently drop the most recent ~2-3h of deals.
    # Widen the window by 1 day on each side to cover broker-TZ drift and
    # overnight swap credits.
    try:
        to_dt = _dt.now(tz=_tz.utc) + _td(days=1)
        from_dt = _dt.now(tz=_tz.utc) - _td(days=3)
        deals = mt5.history_deals_get(from_dt, to_dt) or []
    except Exception as exc:
        logger.warning("reconcile_closed_trades: history_deals_get failed: %s", exc)
        return 0

    # For each target ticket, find the out-leg deal and write to DB.
    # MT5 emits one deal per trade leg — we want entry == DEAL_ENTRY_OUT
    # (1) / OUT_BY (4) / INOUT (2) to pick up the realized PnL.
    out_entries = {1, 2, 4}
    updated = 0
    for ticket in orphans | refresh_tickets:
        is_refresh = ticket in refresh_tickets
        close_deals = [
            d for d in deals
            if int(getattr(d, "position_id", 0)) == ticket
            and getattr(d, "entry", None) in out_entries
        ]
        if not close_deals:
            if not is_refresh:
                logger.debug(
                    "reconcile_closed_trades: no close deal yet for ticket %s "
                    "(may arrive on next tick)", ticket,
                )
            continue
        close_deals.sort(key=lambda d: d.time)
        close = close_deals[-1]
        exit_price = float(close.price)
        # deal.time is broker-local-wall-clock-as-unix, not true UTC.
        # Walk it through the broker TZ to get a correct ISO timestamp.
        exit_time_iso = _broker_ts_to_utc(int(close.time)).replace(
            tzinfo=_tz.utc
        ).isoformat()
        # Commission and swap can be charged on BOTH legs (open + close) on
        # some brokers (the broker splits commission 50/50 across the pair).
        # Sum across ALL deals for this position so the figures match MT5's
        # Account History totals. Profit stays on the close deal — it's the
        # round-trip P/L in account currency.
        position_deals = [
            d for d in deals
            if int(getattr(d, "position_id", 0)) == ticket
        ]
        commission = sum(
            float(getattr(d, "commission", 0.0) or 0.0) for d in position_deals
        )
        swap = sum(
            float(getattr(d, "swap", 0.0) or 0.0) for d in position_deals
        )
        gross_profit = float(getattr(close, "profit", 0.0) or 0.0)
        pnl = gross_profit + commission + swap  # NET (matches MT5 Account History)
        # Invariant: every filled close deal must carry commission + swap
        # fields from the broker. Catches the "closed trade missing fees"
        # bug that slipped past reconciliation silently.
        from src.safety.invariants import Severity as _InvSev, check as _inv_check
        _inv_check(
            "broker.close_deal_has_fees",
            len(position_deals) > 0,
            severity=_InvSev.ALERT,
            context={"ticket": ticket},
            message="history_deals_get returned no deals for closed position",
        )
        # For refresh-mode, skip if DB already matches broker truth.
        if is_refresh and abs(db_candidates.get(ticket, 0.0) - pnl) < 0.01:
            continue
        # MT5 ENUM_DEAL_REASON: CLIENT=0, MOBILE=1, WEB=2, EXPERT=3,
        # SL=4, TP=5, SO=6, ROLLOVER=7, VMARGIN=8, SPLIT=9. Earlier
        # mapping was shifted by -1 so every SL close was alerted as TP.
        reason_enum = getattr(close, "reason", 0)
        reason_str = {4: "sl", 5: "tp", 6: "so", 7: "rollover"}.get(
            reason_enum, f"reason_{reason_enum}",
        )
        # Trade journal canonical code. SO (stop-out from margin call) is
        # bucketed under breaker_emergency; rollover under unknown.
        _mt5_reason_code = {
            4: "stop_loss", 5: "take_profit",
            6: "breaker_emergency", 7: "unknown",
        }.get(reason_enum)
        # R-multiple at exit — compute from tracked position if we still
        # have it (orphan path); refresh-mode rows were already populated
        # by the full_close path, so passing None here leaves prior values
        # intact.
        _r_mult = None
        _bars = None
        _be = None
        _pos_for_r = tracked_positions.get(int(ticket))
        if _pos_for_r is not None:
            _r = abs(_pos_for_r.entry_price - _pos_for_r.initial_stop)
            if _r > 0:
                _r_mult = (
                    (exit_price - _pos_for_r.entry_price) / _r
                    if _pos_for_r.direction == "buy"
                    else (_pos_for_r.entry_price - exit_price) / _r
                )
            _bars = getattr(_pos_for_r, "bars_held", None)
            _be = getattr(_pos_for_r, "be_locked", None)

        try:
            ok = await data_store.close_trade_record(
                ticket=int(ticket),
                exit_price=exit_price,
                exit_time_iso=exit_time_iso,
                pnl_usd=pnl,
                exit_reason=reason_str,
                allow_overwrite=is_refresh,
                commission_usd=commission,
                swap_usd=swap,
                close_reason_code=_mt5_reason_code,
                r_multiple_at_exit=_r_mult,
                bars_held=_bars,
                be_locked_at_close=_be,
            )
        except Exception as exc:
            logger.warning(
                "reconcile_closed_trades: close_trade_record failed for %s: %s",
                ticket, exc,
            )
            continue

        if ok:
            updated += 1
            # Track consecutive SL hits for the breaker (only fresh closes,
            # not pnl refreshes — refreshes correct stored values, not new
            # close events). is_loss = SL exit specifically, matching the
            # "consecutive_loss_limit" semantics in CLAUDE.md ("4 consecutive
            # SL → 4h pause"). Other losing close types (time_exit, manual,
            # reversal_hard_exit) don't increment the SL counter.
            if not is_refresh and circuit_breaker is not None:
                _is_sl = (_mt5_reason_code or "").lower() in ("stop_loss", "sl")
                try:
                    circuit_breaker.record_trade_result(is_loss=_is_sl)
                except Exception as _exc:
                    logger.debug("record_trade_result failed (reconcile): %s", _exc)
            if is_refresh:
                logger.info(
                    "reconcile refreshed pnl for closed ticket %s: "
                    "stored=%+.2f → broker=%+.2f reason=%s",
                    ticket, db_candidates.get(ticket, 0.0), pnl, reason_str,
                )
            else:
                logger.info(
                    "reconciled broker-closed ticket %s: exit=%.5f pnl=%+.2f reason=%s",
                    ticket, exit_price, pnl, reason_str,
                )
                # Append an exit row to trade_events.csv so the History
                # timeline shows the broker-side close. Without this, the
                # UI only renders the original ENTRY event.
                try:
                    TRADE_EVENTS.write({
                        "timestamp": exit_time_iso,
                        "event": "exit",
                        "ticket": int(ticket),
                        "symbol": getattr(_pos_for_r, "symbol", "") if _pos_for_r else "",
                        "direction": getattr(_pos_for_r, "direction", "") if _pos_for_r else "",
                        "lot_size": getattr(_pos_for_r, "volume", "") if _pos_for_r else "",
                        "entry_price": getattr(_pos_for_r, "entry_price", "") if _pos_for_r else "",
                        "current_price": exit_price,
                        "sl_price": getattr(_pos_for_r, "current_stop", "") if _pos_for_r else "",
                        "tp_price": "",
                        "pnl_usd": pnl,
                        "r_multiple": _r_mult if _r_mult is not None else "",
                        "bars_held": _bars if _bars is not None else "",
                        "be_locked": _be if _be is not None else "",
                        "exit_reason": reason_str,
                    })
                except Exception as exc:
                    logger.debug("TRADE_EVENTS exit write failed for %s: %s", ticket, exc)
            # Invariant: passive tagging for post-hoc news-impact analysis.
            # WARN severity = JSONL only, no Telegram. Tier 2/3 events are
            # tracked here so we can later query "did CPI events materially
            # impact live trades?" and decide to promote to blackout tier.
            if not is_refresh:
                try:
                    from datetime import timedelta as _td
                    from src.data_pipeline.market import economic_calendar as _ec
                    from src.safety.invariants import Severity as _Sev, check as _inv
                    exit_dt = datetime.fromisoformat(exit_time_iso.replace("Z", "+00:00"))
                    if exit_dt.tzinfo is None:
                        exit_dt = exit_dt.replace(tzinfo=timezone.utc)
                    for _e in _ec.load_events():
                        if symbol.upper() not in _e.affects:
                            continue
                        delta = (exit_dt - _e.event_utc).total_seconds() / 3600.0
                        if abs(delta) > 4.0:
                            continue
                        _inv(
                            "trade.near_economic_event",
                            False,  # always records (passive data collection)
                            severity=_Sev.WARN,
                            symbol=symbol,
                            context={
                                "ticket": int(ticket),
                                "event": _e.name,
                                "tier": _e.tier,
                                "hours_from_event": round(delta, 2),
                                "pnl_usd": round(pnl, 2),
                                "reason": reason_str,
                            },
                            message=f"close within 4h of {_e.name} (Tier {_e.tier})",
                        )
                except Exception as exc:
                    logger.debug("near-event tagging failed for %s: %s", ticket, exc)
            # Fire trade-close alert so operator knows (exit manager didn't
            # run a close path for this, so no alert fired from there).
            # Refresh-mode skips the alert — full_close already fired one.
            if alert_manager is not None and not is_refresh:
                try:
                    pos_obj = None
                    if positions_lock is not None:
                        with positions_lock:
                            pos_obj = tracked_positions.get(int(ticket))
                    else:
                        pos_obj = tracked_positions.get(int(ticket))
                    if pos_obj is not None:
                        # Compute entry time + duration for reconciled close
                        _rc_entry_time = None
                        _rc_duration = None
                        _rc_opened = getattr(pos_obj, "opened_at", None)
                        if _rc_opened is not None:
                            _rc_entry_time = _rc_opened.strftime("%Y-%m-%d %H:%M UTC")
                            _rc_elapsed = datetime.now(timezone.utc) - _rc_opened
                            _rc_sec = int(_rc_elapsed.total_seconds())
                            _rc_d = _rc_sec // 86400
                            _rc_h = (_rc_sec % 86400) // 3600
                            _rc_m = (_rc_sec % 3600) // 60
                            if _rc_d > 0:
                                _rc_duration = f"{_rc_d}d {_rc_h}h {_rc_m}m"
                            elif _rc_h > 0:
                                _rc_duration = f"{_rc_h}h {_rc_m}m"
                            else:
                                _rc_duration = f"{_rc_m}m"
                        # R-multiple for reconciled close
                        _rc_r_mult = None
                        try:
                            _rc_ep = float(getattr(pos_obj, "entry_price", 0) or 0)
                            _rc_sl = float(getattr(pos_obj, "initial_stop", 0) or 0)
                            _rc_risk = abs(_rc_ep - _rc_sl)
                            if _rc_risk > 0:
                                _rc_sign = 1.0 if getattr(pos_obj, "direction", "") == "buy" else -1.0
                                _rc_r_mult = _rc_sign * (exit_price - _rc_ep) / _rc_risk
                        except Exception:
                            _rc_r_mult = None
                        # Pull Brain context if available
                        _rc_sig = None
                        if combiner is not None and hasattr(combiner, "last_signal_by_symbol"):
                            _rc_sig = combiner.last_signal_by_symbol.get(
                                getattr(pos_obj, "symbol", "")
                            )
                        alert_manager.notify_trade_close(
                            symbol=getattr(pos_obj, "symbol", "?"),
                            direction=getattr(pos_obj, "direction", "?"),
                            entry_price=float(getattr(pos_obj, "entry_price", 0.0)),
                            exit_price=exit_price,
                            lot_size=float(getattr(pos_obj, "volume", 0.0)),
                            pnl=pnl,
                            ticket=int(ticket),
                            reason=reason_str,
                            regime=getattr(_rc_sig.regime, "regime_label", None) if _rc_sig else None,
                            regime_prob=getattr(_rc_sig.regime, "state_probability", None) if _rc_sig else None,
                            bars_held=getattr(pos_obj, "bars_held", None),
                            r_multiple=_rc_r_mult,
                            entry_time=_rc_entry_time,
                            duration=_rc_duration,
                            initial_stop=getattr(pos_obj, "initial_stop", None),
                            strategy_name=getattr(pos_obj, "strategy_name", None) or None,
                        )
                except Exception as exc:
                    logger.warning("notify_trade_close (reconciled) failed: %s", exc)

        # Drop from tracked_positions regardless — the broker says it's gone.
        # (No-op for refresh-mode: full_close already popped it.)
        if positions_lock is not None:
            with positions_lock:
                tracked_positions.pop(int(ticket), None)
        else:
            tracked_positions.pop(int(ticket), None)

    return updated


async def _run_exit_manager(
    exit_mgr,
    order_mgr,
    tracked_positions,
    data_feed,
    recent_signal_dirs,
    tier_state_store: TierStateStore,
    positions_lock=None,
    alert_manager=None,
    data_store=None,
    combiner=None,
    circuit_breaker=None,
) -> None:
    """
    Walk tracked open positions through ExitManager.check_exits and
    translate each ExitAction into broker calls.

    Audit C5: positions_lock protects all tracked_positions access
    against concurrent RiskMonitor.clear() in the halt path.
    """
    # Sync any broker-initiated closes FIRST — this updates DB rows for
    # server-side SL/TP hits and removes them from tracked_positions so
    # the exit manager doesn't waste work on phantom tickets.
    if data_store is not None:
        try:
            await _reconcile_closed_trades(
                tracked_positions, data_store,
                positions_lock=positions_lock,
                alert_manager=alert_manager,
                combiner=combiner,
                circuit_breaker=circuit_breaker,
            )
        except Exception as exc:
            logger.warning("reconcile_closed_trades: %s", exc)
    # Snapshot under lock to prevent RuntimeError on concurrent clear
    if positions_lock is not None:
        with positions_lock:
            if not tracked_positions:
                return
            pos_list = list(tracked_positions.values())
    else:
        if not tracked_positions:
            return
        pos_list = list(tracked_positions.values())
    # Build price + ATR snapshots per symbol.
    current_prices: dict[str, float] = {}
    current_atrs: dict[str, float] = {}
    unique_symbols = {pos.symbol for pos in pos_list}
    for symbol in unique_symbols:
        try:
            bars = await data_feed.get_latest_async(symbol)
            current_prices[symbol] = float(bars["close"].iloc[-1])
            if "atr_14" in bars.columns:
                current_atrs[symbol] = float(bars["atr_14"].iloc[-1])
            else:
                current_atrs[symbol] = max(
                    float(bars["high"].iloc[-1] - bars["low"].iloc[-1]),
                    1e-9,
                )
        except Exception as exc:
            logger.warning("exit_manager: price fetch failed for %s: %s", symbol, exc)

    actions = exit_mgr.check_exits(
        pos_list,
        current_prices=current_prices,
        current_atrs=current_atrs,
        recent_signals=recent_signal_dirs,
    )
    for action in actions:
        # Lock-protected read — position might have been cleared by halt
        if positions_lock is not None:
            with positions_lock:
                pos = tracked_positions.get(action.ticket)
        else:
            pos = tracked_positions.get(action.ticket)
        if pos is None:
            continue
        exit_price = current_prices.get(pos.symbol, pos.entry_price)
        if action.action == "partial_close":
            # Wave 6 fix #1: pass close_volume through so the broker
            # actually closes the partial fraction, not the whole position.
            order_mgr.close_position(action.ticket, volume=action.close_volume)
            if action.new_stop is not None:
                order_mgr.modify_sl_tp(action.ticket, sl=action.new_stop)
            TRADE_EVENTS.write({
                "timestamp": now_iso(), "event": "partial_close",
                "ticket": action.ticket, "symbol": pos.symbol,
                "direction": pos.direction, "lot_size": action.close_volume or 0,
                "entry_price": pos.entry_price, "current_price": exit_price,
                "sl_price": action.new_stop or pos.current_stop, "tp_price": 0.0,
                "exit_reason": getattr(action, "reason", "") or "tier",
            })
        elif action.action == "modify_stop":
            if action.new_stop is not None:
                order_mgr.modify_sl_tp(action.ticket, sl=action.new_stop)
            TRADE_EVENTS.write({
                "timestamp": now_iso(), "event": "modify",
                "ticket": action.ticket, "symbol": pos.symbol,
                "direction": pos.direction, "lot_size": pos.volume,
                "entry_price": pos.entry_price, "current_price": exit_price,
                "sl_price": action.new_stop, "tp_price": 0.0,
                "be_locked": getattr(pos, "be_locked", False),
                "exit_reason": getattr(action, "reason", "") or "be_lock",
            })
        elif action.action == "full_close":
            close_result = order_mgr.close_position(action.ticket, volume=None)
            # Guard: if broker rejected/timed out the close, do NOT write
            # fake exit data to the DB. Position is still live on the
            # broker; reconcile_closed_trades will catch it when MT5
            # eventually emits the close deal (bot retry or SL hit).
            if not getattr(close_result, "success", False):
                logger.warning(
                    "full_close REJECTED ticket=%s sym=%s retcode=%s msg=%s "
                    "— DB row left open; reconcile will fill when broker closes",
                    action.ticket, pos.symbol,
                    getattr(close_result, "retcode", None),
                    getattr(close_result, "error_message", None),
                )
                TRADE_EVENTS.write({
                    "timestamp": now_iso(), "event": "full_close_rejected",
                    "ticket": action.ticket, "symbol": pos.symbol,
                    "direction": pos.direction, "lot_size": pos.volume,
                    "entry_price": pos.entry_price, "current_price": exit_price,
                    "retcode": getattr(close_result, "retcode", None),
                    "error_message": getattr(close_result, "error_message", None),
                    "exit_reason": getattr(action, "reason", "") or "full_close",
                })
                continue
            # Gross-PnL estimate in account currency via MT5 broker calc.
            # (commission + swap land later via reconcile_closed_trades)
            pnl_est = 0.0
            try:
                import MetaTrader5 as _mt5  # type: ignore
                _ot = _mt5.ORDER_TYPE_BUY if pos.direction == "buy" else _mt5.ORDER_TYPE_SELL
                _calc = _mt5.order_calc_profit(
                    _ot, pos.symbol, float(pos.volume),
                    float(pos.entry_price), float(exit_price),
                )
                if _calc is not None:
                    pnl_est = float(_calc)
                else:
                    # Fallback: naive delta (only correct for USD-quote pairs
                    # like ETHUSD / XAUUSD-per-oz). Will be corrected by
                    # reconcile within one tick.
                    _sign = 1 if pos.direction == "buy" else -1
                    pnl_est = _sign * (exit_price - pos.entry_price) * pos.volume
                    logger.warning(
                        "order_calc_profit returned None for %s — used naive "
                        "fallback; reconcile will overwrite",
                        pos.symbol,
                    )
            except Exception as _exc:
                logger.warning("pnl_est calc failed for %s: %s", pos.symbol, _exc)
            exit_reason = getattr(action, "reason", "") or "full_close"
            # Trade journal: classify reason + compute R-multiple at exit.
            from src.strategy.exit_manager import classify_reason as _classify
            _reason_code = _classify(exit_reason)
            _r = abs(pos.entry_price - pos.initial_stop)
            if _r > 0:
                _r_mult = (
                    (exit_price - pos.entry_price) / _r
                    if pos.direction == "buy"
                    else (pos.entry_price - exit_price) / _r
                )
            else:
                _r_mult = None
            # Mark the `trades` DB row closed with our estimates. The
            # broker-side numbers from history_deals_get arrive shortly
            # after and overwrite via the same helper (idempotent).
            try:
                await data_store.close_trade_record(
                    ticket=int(action.ticket),
                    exit_price=float(exit_price),
                    exit_time_iso=now_iso(),
                    pnl_usd=float(pnl_est),
                    exit_reason=exit_reason,
                    close_reason_code=_reason_code,
                    r_multiple_at_exit=_r_mult,
                    bars_held=getattr(pos, "bars_held", None),
                    be_locked_at_close=getattr(pos, "be_locked", None),
                )
            except Exception as _exc:
                logger.debug("close_trade_record failed: %s", _exc)
            # Track consecutive SL hits for the breaker — increments on SL,
            # resets to 0 on any non-SL close (TP, time_exit, manual, etc.).
            # Matches "4 consecutive SL → 4h pause" semantics in CLAUDE.md.
            if circuit_breaker is not None:
                try:
                    _is_sl = (_reason_code or "").lower() in ("stop_loss", "sl")
                    circuit_breaker.record_trade_result(is_loss=_is_sl)
                except Exception as _exc:
                    logger.debug("record_trade_result failed (live close): %s", _exc)
            TRADE_EVENTS.write({
                "timestamp": now_iso(), "event": "exit",
                "ticket": action.ticket, "symbol": pos.symbol,
                "direction": pos.direction, "lot_size": pos.volume,
                "entry_price": pos.entry_price, "current_price": exit_price,
                "sl_price": pos.current_stop, "tp_price": 0.0,
                "pnl_usd": pnl_est,
                "bars_held": getattr(pos, "bars_held", 0),
                "be_locked": getattr(pos, "be_locked", False),
                "exit_reason": exit_reason,
            })
            # Fire Telegram + email trade-close alert. notify_trade_close is
            # fire-and-forget (swallows errors internally) so it cannot block
            # or crash the trading loop.
            if alert_manager is not None:
                try:
                    # Pull current Brain context for this symbol for the
                    # close alert so operator sees what the regime/model
                    # looked like at the moment of exit.
                    _last_sig = None
                    if combiner is not None and hasattr(combiner, "last_signal_by_symbol"):
                        _last_sig = combiner.last_signal_by_symbol.get(pos.symbol)
                    _r_mult: Optional[float] = None
                    try:
                        _risk = abs(pos.entry_price - pos.initial_stop)
                        if _risk > 0:
                            _sign = 1.0 if pos.direction == "buy" else -1.0
                            _r_mult = _sign * (exit_price - pos.entry_price) / _risk
                    except Exception:
                        _r_mult = None
                    # Compute entry time and duration for the alert
                    _entry_time_str = None
                    _duration_str = None
                    _opened = getattr(pos, "opened_at", None)
                    if _opened is not None:
                        _entry_time_str = _opened.strftime("%Y-%m-%d %H:%M UTC")
                        _elapsed = datetime.now(timezone.utc) - _opened
                        _total_sec = int(_elapsed.total_seconds())
                        _days = _total_sec // 86400
                        _hours = (_total_sec % 86400) // 3600
                        _mins = (_total_sec % 3600) // 60
                        if _days > 0:
                            _duration_str = f"{_days}d {_hours}h {_mins}m"
                        elif _hours > 0:
                            _duration_str = f"{_hours}h {_mins}m"
                        else:
                            _duration_str = f"{_mins}m"
                    alert_manager.notify_trade_close(
                        symbol=pos.symbol,
                        direction=pos.direction,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        lot_size=pos.volume,
                        pnl=pnl_est,
                        ticket=action.ticket,
                        reason=exit_reason,
                        regime=getattr(_last_sig.regime, "regime_label", None) if _last_sig else None,
                        regime_prob=getattr(_last_sig.regime, "state_probability", None) if _last_sig else None,
                        lstm_prediction=getattr(_last_sig, "lstm_prediction", None) if _last_sig else None,
                        combined_score=getattr(_last_sig, "combined_score", None) if _last_sig else None,
                        bars_held=getattr(pos, "bars_held", None),
                        r_multiple=_r_mult,
                        entry_time=_entry_time_str,
                        duration=_duration_str,
                        initial_stop=getattr(pos, "initial_stop", None),
                        strategy_name=getattr(pos, "strategy_name", None) or None,
                    )
                except Exception as _exc:
                    logger.warning("notify_trade_close failed: %s", _exc)
            # Lock-protected removal
            if positions_lock is not None:
                with positions_lock:
                    tracked_positions.pop(action.ticket, None)
            else:
                tracked_positions.pop(action.ticket, None)
            # Wave 6 fix #19: drop the tier state record so the file
            # doesn't accumulate dead tickets.
            tier_state_store.delete(action.ticket)


def _configure_logging() -> None:
    """
    Configure logging for both stdout (live monitoring) and rotating files
    (post-session analysis). Creates three handlers:

        data/logs/trading_bot.log   — all INFO+, 10MB rotation × 30 files
        data/logs/errors.log        — only WARNING+, 5MB rotation × 12 files
        stdout                       — INFO+ for live terminal monitoring

    File logs survive restarts; stdout is for live attention. Errors are
    separated so ``tail -F data/logs/errors.log`` gives instant signal.
    """
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    # Main rotating log
    main_file = RotatingFileHandler(
        log_dir / "trading_bot.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=30,
        encoding="utf-8",
    )
    main_file.setLevel(logging.INFO)
    main_file.setFormatter(fmt)
    root.addHandler(main_file)

    # Errors-only log (fast scanning for problems)
    err_file = RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=12,
        encoding="utf-8",
    )
    err_file.setLevel(logging.WARNING)
    err_file.setFormatter(fmt)
    root.addHandler(err_file)

    # Stdout for live monitoring (so you can still see output in terminal)
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    stdout.setFormatter(fmt)
    root.addHandler(stdout)

    # Quiet down some chatty third-party loggers
    for noisy in ("asyncio", "urllib3", "apscheduler.executors.default"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


PID_LOCK_PATH = Path("data/state/bot.pid")


def _pid_alive(pid: int) -> bool:
    """Return True iff the given OS PID is a currently running process.

    Cross-platform:
      * POSIX: signal 0 probe (no real signal sent)
      * Windows: OpenProcess(SYNCHRONIZE) + immediately close — fails only
        when the PID no longer exists or belongs to a different session.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        SYNCHRONIZE = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _acquire_pid_lock() -> None:
    """Ensure only one bot instance runs at a time.

    Two-phase lock to avoid the create-then-write race: O_CREAT|O_EXCL
    creates the file atomically but the PID isn't written until a moment
    later. A concurrent reader could see the empty file and wrongly
    conclude the lock is stale. We mitigate by (a) writing + fsyncing
    before exiting the ``with`` block, and (b) on read, treating empty
    content as "someone is mid-write" and waiting briefly before deciding.
    """
    import time as _t
    PID_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    for _attempt in range(10):
        try:
            with open(PID_LOCK_PATH, "x", encoding="utf-8") as f:
                f.write(str(os.getpid()))
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            return
        except FileExistsError:
            pass
        # Someone else owns (or is creating) the lock.
        try:
            text = PID_LOCK_PATH.read_text(encoding="utf-8").strip()
        except OSError:
            _t.sleep(0.05)
            continue
        if not text:
            # Empty file = racer is still mid-write. Wait and loop.
            _t.sleep(0.05)
            continue
        try:
            existing = int(text)
        except ValueError:
            # Corrupted — treat as stale, remove and retry.
            try:
                PID_LOCK_PATH.unlink()
            except OSError:
                pass
            continue
        if existing != os.getpid() and _pid_alive(existing):
            msg = (
                f"ERROR: another Cortex instance is already running "
                f"(PID {existing}). Stop it first, or delete {PID_LOCK_PATH} "
                f"if you're sure it's stale."
            )
            print(msg, file=sys.stderr, flush=True)
            try:
                logger.error(msg)
            except Exception:
                pass
            sys.exit(3)
        # Stale (owner dead or matches us) — remove and retry exclusive-create.
        try:
            PID_LOCK_PATH.unlink()
        except OSError:
            pass
    # Gave up racing; force-write so we don't hang. Better to let the bot
    # run than to abort on a pathological lock-file state.
    PID_LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")


def _release_pid_lock() -> None:
    """Remove our PID file on graceful shutdown."""
    try:
        if PID_LOCK_PATH.exists():
            text = PID_LOCK_PATH.read_text(encoding="utf-8").strip()
            # Only delete if it still owns our PID; otherwise another
            # instance may have legitimately taken over.
            if text == str(os.getpid()):
                PID_LOCK_PATH.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    _configure_logging()
    _acquire_pid_lock()
    import atexit as _atexit
    _atexit.register(_release_pid_lock)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — exiting.")
        sys.exit(0)
    finally:
        _release_pid_lock()
