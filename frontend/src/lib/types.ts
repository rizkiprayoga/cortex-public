/* Mirror of FastAPI Pydantic schemas */

export interface AccountSwitchRequest {
  slot: string;
}

export interface AccountRegisterRequest {
  slot_name: string;
  login: number;
  password: string;
  server: string;
  auto_switch?: boolean;
}

export interface AccountInfo {
  account_id: number | null;
  server: string | null;
  is_demo: boolean;
  balance: number;
  equity: number;
}

export interface AccountSlotInfo {
  slot: string;
  login: number;
  server: string;
  is_current: boolean;
}

export interface AccountSlotsResponse {
  slots: AccountSlotInfo[];
  current_slot: string | null;
}

export interface AccountData {
  balance: number;
  equity: number;
  margin: number;
  free_margin: number;
  margin_level: number;
  floating_pnl: number;
  open_positions: number;
}

export interface RegimeData {
  symbol: string;
  regime_index: number;
  regime_label: string;
  state_probability: number;
  position_multiplier: number;
  all_probabilities: number[];
  expected_volatility: number;
  all_expected_vols: number[] | null;
}

export interface SignalData {
  symbol: string;
  should_trade: boolean;
  direction: string | null;
  combined_score: number;
  regime: RegimeData;
  lstm_prediction: number;
  confidence: number;
  bar_timestamp: string | null;
  uncertainty_mode: boolean;
  size_discount: number;
  reasoning: string[];
}

export interface BreakerData {
  multiplier: number;
  requires_flat: boolean;
  active_breakers: string[];
  daily_dd_pct: number;
  weekly_dd_pct: number;
  peak_dd_pct: number;
  reason: string;
  consecutive_losses: number;
  consecutive_loss_limit: number;
}

export interface PositionData {
  symbol: string;
  ticket: number;
  direction: string;
  entry_price: number;
  initial_stop: number;
  current_stop: number;
  take_profit?: number | null;
  volume: number;
  initial_volume: number;
  atr_trail_mult: number;
  strategy_name: string;
  tier_1_done: boolean;
  tier_2_done: boolean;
  max_price: number | null;
  min_price: number | null;
  opened_at: string | null;
  current_price: number | null;
  floating_pnl: number | null;
  risk_dollars: number | null;
  time_exit_bars: number | null;
  time_exit_remaining_sec: number | null;
}

export interface LiveStateData {
  account: AccountData | null;
  breaker: BreakerData | null;
  peak_equity: number;
  bot_status: string;
  positions_count: number;
  signals: Record<string, SignalData>;
}

export interface BotStatusData {
  status: string;
  last_change: string;
  changed_by: string;
}

export interface OHLCVBar {
  time: string; // ISO-8601 UTC
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export type ChartTimeframe = "M15" | "H1" | "H4" | "D1" | "W1";

export interface CandlesResponse {
  symbol: string;
  timeframe: string;
  bars: OHLCVBar[];
}

/* ── Backtest drill-down (Phase F3) ── */

export interface BacktestEquityPoint {
  bar_timestamp: string;
  equity: number;
  drawdown_pct: number;
}

export interface BacktestTradeRow {
  symbol: string;
  direction: string;
  entry_time: string | null;
  exit_time: string | null;
  entry_price: number;
  exit_price: number;
  pnl: number;
  r_multiple: number;
  exit_reason: string;
  strategy_name: string | null;
  regime_label: string | null;
  combined_score: number | null;
}

export interface BacktestDetailResponse {
  summary: BacktestRunSummary;
  equity_curve: BacktestEquityPoint[];
  trades: BacktestTradeRow[];
}

export interface SystemStatusData {
  bot_status: string;
  uptime_seconds: number;
  last_tick: string | null;
  api_port: number;
  positions_count: number;
  breaker_active: boolean;
  // Health monitor (Phase D)
  heartbeat_age_seconds: number | null;
  heartbeat_equity: number | null;
  heartbeat_open_positions: number | null;
  dashboard_locked: boolean;
  recent_errors: string[];
  log_tail: string[];
}

export interface LockStatusData {
  locked: boolean;
  is_local: boolean;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export type BotAction = "start" | "pause" | "stop";

/* ── History (Phase 10.2) ── */

export interface TradeHistoryItem {
  id: number;
  timestamp_open: string | null;
  timestamp_close: string | null;
  symbol: string;
  direction: string | null;
  lot_size: number | null;
  entry_price: number | null;
  exit_price: number | null;
  pnl_usd: number | null;
  commission_usd: number | null;
  swap_usd: number | null;
  regime_at_entry: string | null;
  combined_score: number | null;
  ticket: number | null;
  // Trade journal (Plan 1)
  close_reason: string | null;
  close_reason_code: string | null;
  r_multiple_at_exit: number | null;
  bars_held: number | null;
  entry_score: number | null;
  exit_score: number | null;
  regime_at_exit: string | null;
  initial_stop: number | null;
  tp_price: number | null;
  be_locked_at_close: boolean | null;
}

export interface TradeHistoryResponse {
  trades: TradeHistoryItem[];
  total: number;
  page: number;
  page_size: number;
}

/* Trade timeline (Plan A) — per-ticket events stitched with signals. */
export interface TradeEventItem {
  timestamp: string;
  event: string; // entry | modify | partial_close | exit | full_close_rejected
  ticket: number | null;
  symbol: string | null;
  direction: string | null;
  lot_size: number | null;
  entry_price: number | null;
  current_price: number | null;
  sl_price: number | null;
  tp_price: number | null;
  pnl_usd: number | null;
  r_multiple: number | null;
  bars_held: number | null;
  be_locked: boolean | null;
  regime_at_entry: string | null;
  combined_score_at_entry: number | null;
  exit_reason: string | null;
}

export interface TradeTimelineResponse {
  ticket: number;
  events: TradeEventItem[];
  signals: SignalAuditItem[];
}

export interface EquityPoint {
  timestamp: string;
  equity: number;
  balance: number | null;
  floating_pnl: number | null;
}

export interface EquityCurveResponse {
  points: EquityPoint[];
  count: number;
}

export interface BalanceOperation {
  time: string; // ISO-8601 UTC
  type: string; // "deposit" | "withdrawal" | "credit"
  amount: number;
  comment: string;
  ticket: number;
}

export interface BalanceOperationsResponse {
  operations: BalanceOperation[];
  count: number;
}

/* ── Signal audit (Phase 3 — Signals Log screen) ── */

export interface SignalAuditItem {
  timestamp: string;
  symbol: string;
  regime: string | null;
  regime_prob: number | null;
  lstm_prediction: number | null;
  combined_score: number | null;
  direction: string | null;
  should_trade: boolean;
  executed: boolean;
  news_blackout: boolean;
  nearest_cb: string | null;
  nearest_hours: number | null;
  block_reason: string | null;
  cb_multiplier: number | null;
  reasoning: string | null;
}

export interface SignalAuditResponse {
  items: SignalAuditItem[];
  total: number;
  page: number;
  page_size: number;
}

/* ── Model performance (Phase P follow-up) ── */

export interface ModelSummaryRow {
  symbol: string;
  lstm_version: number | null;
  hmm_version: number | null;
  lstm_trained_at: string | null;
  hmm_trained_at: string | null;
  lstm_val_loss: number | null;
  lstm_train_dir_acc: number | null;
  live_dir_acc: number | null;
  live_mae: number | null;
  n_predictions: number;
  lstm_file_mtime: string | null;
  next_retrain_due: string | null;
  // A-8: latest daily drift-check snapshot
  drift_psi_max: number | null;
  drift_ks_max: number | null;
  drift_checked_at: string | null;
  drift_status: "ok" | "warn" | "alert" | null;
  drift_worst_feature: string | null;
}

export interface ModelSummaryResponse {
  symbols: ModelSummaryRow[];
}

export interface AccuracyPoint {
  date: string;
  directional_accuracy: number;
  mae: number;
  n: number;
}

export interface AccuracyTimeSeriesResponse {
  symbol: string;
  points: AccuracyPoint[];
}

export interface ModelVersionEntry {
  model_name: string;
  version: number;
  trained_at: string | null;
  val_loss: number | null;
  directional_accuracy: number | null;
  trained_data_start: string | null;
  trained_data_end: string | null;
}

export interface ModelVersionHistoryResponse {
  model_name: string;
  versions: ModelVersionEntry[];
}

export interface TradingMetrics {
  win_rate: number;
  profit_factor: number;
  sharpe_daily: number;
  max_drawdown_pct: number;
  net_pnl: number;
  total_r: number;
  total_trades: number;
  /** Rolling 90-day Calmar (CAGR / |Max DD%|). 0.0 when undefined. */
  calmar_90d: number;
}

/* ── Config (Phase 10.2) ── */

export interface RiskConfig {
  max_daily_loss_soft_pct: number;
  max_daily_loss_hard_pct: number;
  max_weekly_loss_soft_pct: number;
  max_weekly_loss_hard_pct: number;
  max_peak_drawdown_pct: number;
  max_position_size_pct: number;
  max_total_exposure_pct: number;
  free_margin_reserve_pct: number;
  max_concurrent_per_symbol: number;
  max_concurrent_total: number;
  max_daily_trades: number;
}

export interface RiskConfigUpdate {
  [key: string]: number | string | undefined;
  confirmation?: string;
}

/* ── Backtest (Phase 10.2) ── */

export interface BacktestRunSummary {
  id: string;
  status: string;
  symbol: string;
  timeframe: string;
  start_date: string | null;
  end_date: string | null;
  created_at: string | null;
  finished_at: string | null;
  total_trades: number;
  win_rate: number;
  net_pnl: number;
  max_drawdown_pct: number;
  sharpe_ratio: number;
  profit_factor: number;
  calmar_ratio: number;
  mode?: string;
  model_name?: string | null;
  model_version?: string | null;
  model_trained_at?: string | null;
  // A-7: overfitting diagnostics (populated on detail endpoint only)
  deflated_sharpe?: number | null;
  sharpe_stability?: number | null;
}

export interface BacktestStatusResponse {
  run: BacktestRunSummary;
  error_message: string | null;
}

export interface BacktestRunsResponse {
  runs: BacktestRunSummary[];
  count: number;
}

export interface BacktestSubmitResponse {
  run_id: string;
  status: string;
}

