"""
schemas.py — Pydantic response/request models for the FastAPI dashboard.

These mirror the existing dataclasses (SignalResult, RegimeResult,
BreakerSnapshot, AccountSnapshot, OpenPosition) as Pydantic models
so FastAPI can serialize them directly.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------- Request models ----------

class LoginRequest(BaseModel):
    username: str
    password: str


class BotControlRequest(BaseModel):
    action: str  # "start", "pause", "stop"
    confirmation: Optional[str] = None  # required for "stop": must be "STOP"


class AccountSwitchRequest(BaseModel):
    """
    Switch MT5 account by slot name (Security Audit H1).

    Credentials are stored in environment variables, never transmitted
    over the API. Configure slots in .env:
        MT5_SLOT_DEMO_LOGIN, MT5_SLOT_DEMO_PASSWORD, MT5_SLOT_DEMO_SERVER
        MT5_SLOT_LIVE_LOGIN, MT5_SLOT_LIVE_PASSWORD, MT5_SLOT_LIVE_SERVER
    """
    slot: str  # e.g. "demo", "live" — maps to MT5_SLOT_<UPPER>_* env vars


class AccountRegisterRequest(BaseModel):
    """Register a new MT5 account slot. Credentials sent once (localhost only)."""
    slot_name: str       # alphanumeric + underscore, e.g. "live2"
    login: int
    password: str
    server: str
    auto_switch: bool = False


class AccountInfoResponse(BaseModel):
    account_id: Optional[int] = None
    server: Optional[str] = None
    is_demo: bool = False
    balance: float = 0.0
    equity: float = 0.0


class AccountSlotInfo(BaseModel):
    """One available account slot."""
    slot: str
    login: int
    server: str
    is_current: bool = False


class AccountSlotsResponse(BaseModel):
    """All configured account slots."""
    slots: list[AccountSlotInfo]
    current_slot: Optional[str] = None


# ---------- Response models ----------

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    username: str


class RegimeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    regime_index: int
    regime_label: str
    state_probability: float
    position_multiplier: float
    all_probabilities: list[float]
    expected_volatility: float = 0.0
    all_expected_vols: Optional[list[float]] = None


class SignalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    should_trade: bool
    direction: Optional[str]
    combined_score: float
    regime: RegimeResponse
    lstm_prediction: float
    confidence: float
    bar_timestamp: Optional[str] = None
    uncertainty_mode: bool = False
    size_discount: float = 1.0
    reasoning: list[str] = []


class BreakerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    multiplier: float
    requires_flat: bool
    active_breakers: list[str]
    daily_dd_pct: float
    weekly_dd_pct: float
    peak_dd_pct: float
    reason: str
    consecutive_losses: int = 0
    consecutive_loss_limit: int = 4


class AccountResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    floating_pnl: float
    open_positions: int


class PositionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    ticket: int
    direction: str
    entry_price: float
    initial_stop: float
    current_stop: float
    take_profit: Optional[float] = None
    volume: float
    initial_volume: float
    atr_trail_mult: float
    strategy_name: str = ""
    tier_1_done: bool = False
    tier_2_done: bool = False
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    opened_at: Optional[datetime] = None
    current_price: Optional[float] = None
    floating_pnl: Optional[float] = None
    # 1R risk in account currency, computed via order_calc_profit at the
    # initial stop. Used by the dashboard to derive $ tickmarks under the
    # R-multiple progress bar (-1R / +1R / +2R / +3R → $-89 / $+89 / etc.).
    risk_dollars: Optional[float] = None
    # Per-pair time_exit_h1_bars from settings.yaml, surfaced for the
    # dashboard countdown. None when unset (legacy positions or operator
    # disabled time-exit, e.g. via E-7 trend-mode).
    time_exit_bars: Optional[int] = None
    # Trading-hour-aware seconds remaining until time-exit fires. Mirrors
    # ExitManager._h1_bars_elapsed for forex/metals (excludes weekend) and
    # wall-clock for crypto. None when time_exit_bars is unset.
    time_exit_remaining_sec: Optional[int] = None


class BotStatusResponse(BaseModel):
    status: str  # "running", "paused", "stopped"
    last_change: datetime
    changed_by: str


class LockStatusResponse(BaseModel):
    locked: bool
    is_local: bool = False


class RestartResponse(BaseModel):
    status: str  # "scheduled"
    message: str
    pid: Optional[int] = None  # PID of the detached restart helper


class LiveStateResponse(BaseModel):
    """Top-level card data for the Overview screen."""
    account: Optional[AccountResponse] = None
    breaker: Optional[BreakerResponse] = None
    peak_equity: float = 0.0
    bot_status: str = "running"
    positions_count: int = 0
    signals: dict[str, Optional[SignalResponse]] = {}


class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: datetime


class SystemStatusResponse(BaseModel):
    bot_status: str
    uptime_seconds: float
    last_tick: Optional[str] = None
    api_port: int = 8787
    positions_count: int = 0
    breaker_active: bool = False
    # Health-monitor fields (Phase D dashboard health card)
    heartbeat_age_seconds: Optional[float] = None
    heartbeat_equity: Optional[float] = None
    heartbeat_open_positions: Optional[int] = None
    dashboard_locked: bool = False
    recent_errors: list[str] = Field(default_factory=list)  # last 10 min, filtered
    log_tail: list[str] = Field(default_factory=list)        # last ~10 lines


# ---------- Live OHLCV (Phase F2 — live charts) ----------


class OHLCVBarResponse(BaseModel):
    """One OHLCV candle, ready for a charting library."""
    time: str            # ISO-8601 UTC, e.g. "2026-04-13T17:00:00"
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class CandlesResponse(BaseModel):
    """Response for GET /api/live/candles/{symbol}."""
    symbol: str
    timeframe: str
    bars: list[OHLCVBarResponse]


# ---------- History models (Phase 10.2) ----------

class TradeHistoryItem(BaseModel):
    id: int
    timestamp_open: Optional[str] = None
    timestamp_close: Optional[str] = None
    symbol: str
    direction: Optional[str] = None
    lot_size: Optional[float] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    pnl_usd: Optional[float] = None  # NET (profit + commission + swap)
    commission_usd: Optional[float] = None
    swap_usd: Optional[float] = None
    regime_at_entry: Optional[str] = None
    combined_score: Optional[float] = None
    ticket: Optional[int] = None
    # Trade journal (Plan 1) — populated by ExitManager + reconcile paths
    close_reason: Optional[str] = None
    close_reason_code: Optional[str] = None
    r_multiple_at_exit: Optional[float] = None
    bars_held: Optional[int] = None
    entry_score: Optional[float] = None
    exit_score: Optional[float] = None
    regime_at_exit: Optional[str] = None
    initial_stop: Optional[float] = None
    tp_price: Optional[float] = None
    be_locked_at_close: Optional[bool] = None


class TradeHistoryResponse(BaseModel):
    trades: list[TradeHistoryItem]
    total: int
    page: int
    page_size: int


class BalanceOperation(BaseModel):
    """One deposit / withdrawal / credit event from MT5 history."""
    time: str                # ISO-8601 UTC
    type: str                # "deposit" | "withdrawal" | "credit"
    amount: float
    comment: str = ""
    ticket: int = 0


class BalanceOperationsResponse(BaseModel):
    operations: list[BalanceOperation]
    count: int


# ---------- Models dashboard (Phase P follow-up) ----------

class ModelSummaryRow(BaseModel):
    """One row per live symbol on the Models screen."""
    symbol: str
    lstm_version: Optional[int] = None
    hmm_version: Optional[int] = None
    lstm_trained_at: Optional[str] = None
    hmm_trained_at:  Optional[str] = None
    lstm_val_loss: Optional[float] = None
    lstm_train_dir_acc: Optional[float] = None  # from training time
    live_dir_acc: Optional[float] = None        # rolling feedback-loop metric
    live_mae: Optional[float] = None
    n_predictions: int = 0
    lstm_file_mtime: Optional[str] = None  # fs source of truth for retrain
    next_retrain_due: Optional[str] = None  # 1st of next month @ 03:00 UTC
    # A-8: most recent daily drift check, if any. `drift_status` is one of
    # "ok" / "warn" / "alert" / None (no data yet).
    drift_psi_max: Optional[float] = None
    drift_ks_max: Optional[float] = None
    drift_checked_at: Optional[str] = None
    drift_status: Optional[str] = None
    drift_worst_feature: Optional[str] = None


class ModelSummaryResponse(BaseModel):
    symbols: list[ModelSummaryRow]


class AccuracyPoint(BaseModel):
    date: str
    directional_accuracy: float
    mae: float
    n: int


class AccuracyTimeSeriesResponse(BaseModel):
    symbol: str
    points: list[AccuracyPoint]


class ModelVersionEntry(BaseModel):
    model_name: str
    version: int
    trained_at: Optional[str] = None
    val_loss: Optional[float] = None
    directional_accuracy: Optional[float] = None
    trained_data_start: Optional[str] = None
    trained_data_end: Optional[str] = None


class ModelVersionHistoryResponse(BaseModel):
    model_name: str
    versions: list[ModelVersionEntry]


class EquityPoint(BaseModel):
    timestamp: str
    equity: float
    balance: Optional[float] = None
    floating_pnl: Optional[float] = None


class EquityCurveResponse(BaseModel):
    points: list[EquityPoint]
    count: int


class SignalLogItem(BaseModel):
    id: int
    timestamp: str
    symbol: str
    regime: Optional[str] = None
    regime_probability: Optional[float] = None
    lstm_prediction: Optional[float] = None
    combined_score: Optional[float] = None
    should_trade: Optional[bool] = None
    direction: Optional[str] = None


class SignalLogResponse(BaseModel):
    signals: list[SignalLogItem]
    total: int
    page: int
    page_size: int


class SignalAuditItem(BaseModel):
    """One row from the signal_audit.csv file.

    Superset of SignalLogItem — adds the fields that only exist in the
    CSV audit stream: executed flag, block_reason, news-blackout context,
    circuit-breaker multiplier at the time of the signal, and the full
    gate-by-gate reasoning string from SignalCombiner._fuse_signals.
    """
    timestamp: str
    symbol: str
    regime: Optional[str] = None
    regime_prob: Optional[float] = None
    lstm_prediction: Optional[float] = None
    combined_score: Optional[float] = None
    direction: Optional[str] = None
    should_trade: bool = False
    executed: bool = False
    news_blackout: bool = False
    nearest_cb: Optional[str] = None
    nearest_hours: Optional[float] = None
    block_reason: Optional[str] = None
    cb_multiplier: Optional[float] = None
    reasoning: Optional[str] = None


class SignalAuditResponse(BaseModel):
    """Paginated view of signal_audit.csv."""
    items: list[SignalAuditItem]
    total: int
    page: int
    page_size: int


class TradeEventItem(BaseModel):
    """One row from trade_events.csv — entry, modify, partial_close, exit."""
    timestamp: str
    event: str  # entry | modify | partial_close | exit | full_close_rejected | smoke
    ticket: Optional[int] = None
    symbol: Optional[str] = None
    direction: Optional[str] = None
    lot_size: Optional[float] = None
    entry_price: Optional[float] = None
    current_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    pnl_usd: Optional[float] = None
    r_multiple: Optional[float] = None
    bars_held: Optional[int] = None
    be_locked: Optional[bool] = None
    regime_at_entry: Optional[str] = None
    combined_score_at_entry: Optional[float] = None
    exit_reason: Optional[str] = None


class TradeTimelineResponse(BaseModel):
    """Stitched lifecycle for a single ticket — events + the signals that
    fired around the open time (for rationale context)."""
    ticket: int
    events: list[TradeEventItem]
    signals: list[SignalAuditItem]


class ModelAccuracyResponse(BaseModel):
    symbol: str
    window: int
    directional_accuracy: float
    mse: float
    mae: float
    n_predictions: int


class TradingMetricsResponse(BaseModel):
    win_rate: float
    profit_factor: float
    sharpe_daily: float
    max_drawdown_pct: float
    net_pnl: float
    total_r: float
    total_trades: int
    # Rolling 90-day Calmar = annualized_return_over_window / |max_dd_pct_in_window|.
    # Undefined (0.0 → UI shows "—") when DD < 0.5% OR fewer than ~10 trades
    # in the window (low-N noise dominates). Computed on read; no DB field.
    calmar_90d: float = 0.0


# ---------- Config models (Phase 10.2) ----------

class RiskConfigResponse(BaseModel):
    max_daily_loss_soft_pct: float
    max_daily_loss_hard_pct: float
    max_weekly_loss_soft_pct: float
    max_weekly_loss_hard_pct: float
    max_peak_drawdown_pct: float
    max_position_size_pct: float
    max_total_exposure_pct: float
    free_margin_reserve_pct: float
    max_concurrent_per_symbol: int
    max_concurrent_total: int
    max_daily_trades: int


class RiskConfigUpdateRequest(BaseModel):
    """All fields optional — only supplied fields are updated."""
    max_daily_loss_soft_pct: Optional[float] = None
    max_daily_loss_hard_pct: Optional[float] = None
    max_weekly_loss_soft_pct: Optional[float] = None
    max_weekly_loss_hard_pct: Optional[float] = None
    max_peak_drawdown_pct: Optional[float] = None
    max_position_size_pct: Optional[float] = None
    max_total_exposure_pct: Optional[float] = None
    free_margin_reserve_pct: Optional[float] = None
    max_concurrent_per_symbol: Optional[int] = None
    max_concurrent_total: Optional[int] = None
    max_daily_trades: Optional[int] = None
    confirmation: Optional[str] = None  # "CONFIRM_HARD_HALT_CHANGE" for hard knobs


# ---------- Backtest models (Phase 10.2) ----------

class BacktestSubmitRequest(BaseModel):
    symbol: str
    timeframe: str = "H4"
    start_date: str      # ISO 8601
    end_date: str         # ISO 8601
    initial_equity: float = 10000.0
    mode: str = "simple"  # "simple" or "full"


class BacktestSubmitResponse(BaseModel):
    run_id: str
    status: str = "pending"


class BacktestRunSummary(BaseModel):
    id: str
    status: str
    symbol: str
    timeframe: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    created_at: Optional[str] = None
    finished_at: Optional[str] = None
    total_trades: int = 0
    win_rate: float = 0.0
    net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    # Calmar = CAGR / |Max DD%|. Computed on read (see backtest route) —
    # no DB column; derived from net_pnl + max_drawdown_pct + date span.
    # 0.0 when DD < 0.5% (undefined-guard) or when inputs missing.
    calmar_ratio: float = 0.0
    # A-7: overfitting diagnostics. Populated on the detail endpoint only
    # (requires the trade series). None on the runs list.
    deflated_sharpe: Optional[float] = None   # P(SR > null expected max) ∈ [0, 1]
    sharpe_stability: Optional[float] = None  # fraction of sub-windows Sharpe>0
    mode: str = "simple"
    # Model architecture used for this run (best-effort: filled at run creation
    # from data/models/lstm_<SYMBOL>.pt mtime + version registry. None for old
    # rows that pre-date this field).
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_trained_at: Optional[str] = None


class BacktestStatusResponse(BaseModel):
    run: BacktestRunSummary
    error_message: Optional[str] = None


class BacktestRunsResponse(BaseModel):
    runs: list[BacktestRunSummary]
    count: int


# ---------- Backtest detail drawer (Phase F3) ----------


class BacktestEquityPoint(BaseModel):
    """One point on the per-run equity curve."""
    bar_timestamp: str
    equity: float
    drawdown_pct: float = 0.0


class BacktestTradeRow(BaseModel):
    """One closed trade inside a backtest run."""
    symbol: str
    direction: str
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    r_multiple: float = 0.0
    exit_reason: str = ""
    strategy_name: Optional[str] = None
    regime_label: Optional[str] = None
    combined_score: Optional[float] = None


class BacktestDetailResponse(BaseModel):
    """Full drill-down payload for a single backtest run."""
    summary: BacktestRunSummary
    equity_curve: list[BacktestEquityPoint] = Field(default_factory=list)
    trades: list[BacktestTradeRow] = Field(default_factory=list)
