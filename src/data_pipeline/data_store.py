"""
data_store.py — PostgreSQL Async Data Layer (SQLAlchemy 2.x)

Stores and retrieves ALL system data via an async PostgreSQL connection:

    ORIGINAL TABLES (3):
        trades          — completed trade records
        signals         — every signal generated (traded or not)
        equity_history  — account equity snapshots

    NEW TABLES (6, added v2.0):
        ohlcv_bars          — all fetched MT5 OHLCV bars (persistent cache)
        engineered_features — all computed feature vectors per bar
        model_versions      — training metadata per retrain event
        model_predictions   — every HMM + LSTM prediction logged
        actual_outcomes     — ground-truth labels (next-bar actual returns)
        prediction_errors   — prediction vs actual (for feedback loop)

Connection: postgresql+asyncpg://... (from POSTGRES_DSN env var)
Driver:     asyncpg (non-blocking, ~2x faster than psycopg2)
ORM:        SQLAlchemy 2.x AsyncSession with async_sessionmaker
Inserts:    Bulk upsert via INSERT ... ON CONFLICT DO NOTHING
"""

import logging
import os
from datetime import datetime
from typing import Literal, Optional


# Bulk insert chunk size — capped by Postgres's 32767-parameter wire-protocol
# limit. 5 columns per row × 6500 rows ≈ 32500 parameters, comfortably under.
# Stooq US 10Y series (back to 1871) produces 50k+ rows for one symbol —
# without chunking the bulk insert would fail with "too many query arguments".
FEATURE_STORE_BULK_CHUNK = 6500

import numpy as np
import pandas as pd
from sqlalchemy import (
    Boolean, Column, Float, ForeignKey, Index, Integer, PrimaryKeyConstraint,
    String, UniqueConstraint, func, select, text
)
from sqlalchemy.dialects.postgresql import insert as pg_insert, JSONB, TIMESTAMP as PG_TIMESTAMP
from sqlalchemy.ext.asyncio import (
    async_sessionmaker, create_async_engine
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM Base & Engine
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


def build_engine(dsn: Optional[str] = None, **kwargs):
    """
    Create async SQLAlchemy engine.

    Args:
        dsn:    PostgreSQL DSN string. Defaults to POSTGRES_DSN env var.
                Format: postgresql+asyncpg://user:pass@host:port/dbname
        **kwargs: Overrides for engine settings (pool_size, echo, etc.)
    """
    dsn = dsn or os.environ["POSTGRES_DSN"]
    defaults = dict(
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_pre_ping=True,
        echo=False,
    )
    defaults.update(kwargs)
    return create_async_engine(dsn, **defaults)


# ---------------------------------------------------------------------------
# ORM Models — Original 3 Tables
# ---------------------------------------------------------------------------

class TradeRecord(Base):
    """Completed trade record."""
    __tablename__ = "trades"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    timestamp_open   = Column(String)        # ISO 8601 string (timezone-aware)
    timestamp_close  = Column(String)
    symbol           = Column(String(10), nullable=False)
    direction        = Column(String(4))     # "buy" or "sell"
    lot_size         = Column(Float)
    entry_price      = Column(Float)
    exit_price       = Column(Float)
    pnl_usd          = Column(Float)  # NET = profit + commission + swap
    commission_usd   = Column(Float, nullable=True)
    swap_usd         = Column(Float, nullable=True)
    regime_at_entry  = Column(String(10))
    combined_score   = Column(Float)
    ticket           = Column(Integer)
    mt5_account      = Column(Integer, nullable=True)
    # --- Trade journal (Plan 1: close reason + R-multiple + snapshots) ---
    close_reason        = Column(String(200), nullable=True)  # full ExitAction.reason string
    close_reason_code   = Column(String(30), nullable=True)   # enum: take_profit|stop_loss|time_exit|reversal_hard_exit|manual|breaker_emergency|unknown|inferred:*
    r_multiple_at_exit  = Column(Float, nullable=True)        # signed (exit - entry) / initial_R
    bars_held           = Column(Integer, nullable=True)      # ExitManager tick counter at close
    entry_score         = Column(Float, nullable=True)        # combined_score snapshot at order send
    exit_score          = Column(Float, nullable=True)        # combined_score snapshot at close
    regime_at_exit      = Column(String(10), nullable=True)
    initial_stop        = Column(Float, nullable=True)        # required to compute R post-hoc
    tp_price            = Column(Float, nullable=True)        # planned TP at entry
    be_locked_at_close  = Column(Boolean, nullable=True)
    # Predictor traceability — `lstm_<SYMBOL>@<YYYY-MM-DD>` derived from the
    # loaded LSTM .pt file mtime at trade-open. Lets you trace any live trade
    # back to a specific retrain. NULL on pre-stamping rows; backfilled by
    # scripts/backfill_trade_model_versions.py from MLflow run history.
    model_version       = Column(String(64), nullable=True)

    __table_args__ = (
        Index("ix_trades_account_ts", "mt5_account", "timestamp_close"),
        Index("ix_trades_model_version", "model_version"),
    )


class SignalRecord(Base):
    """Signal event — every bar, traded or not."""
    __tablename__ = "signals"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    timestamp          = Column(String, nullable=False)
    symbol             = Column(String(10), nullable=False)
    regime             = Column(String(10))
    regime_probability = Column(Float)
    lstm_prediction    = Column(Float)
    combined_score     = Column(Float)
    should_trade       = Column(Boolean)
    direction          = Column(String(4))
    mt5_account        = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_signals_account_ts", "mt5_account", "timestamp"),
    )


class EquityRecord(Base):
    """Account equity snapshot."""
    __tablename__ = "equity_history"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    timestamp    = Column(String, nullable=False)
    balance      = Column(Float)
    equity       = Column(Float)
    floating_pnl = Column(Float)
    mt5_account  = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_equity_account_ts", "mt5_account", "timestamp"),
    )


class DriftScoreRecord(Base):
    """A-8: daily feature-drift score per symbol.

    Populated by the 01:00 UTC drift-check job in main.py. One row per
    (symbol, daily check). Dashboard reads the most recent row per symbol.
    """
    __tablename__ = "drift_scores"

    id                       = Column(Integer, primary_key=True, autoincrement=True)
    timestamp                = Column(String, nullable=False)
    symbol                   = Column(String(10), nullable=False)
    psi_max                  = Column(Float)
    ks_max                   = Column(Float)
    n_current_samples        = Column(Integer)
    threshold_warn_breached  = Column(Boolean, default=False)
    threshold_alert_breached = Column(Boolean, default=False)
    retrain_triggered        = Column(Boolean, default=False)
    worst_feature            = Column(String(80), nullable=True)
    notes                    = Column(String(500), nullable=True)

    __table_args__ = (
        Index("ix_drift_symbol_ts", "symbol", "timestamp"),
    )


# ---------------------------------------------------------------------------
# ORM Models — New Persistence Tables (v2.0)
# ---------------------------------------------------------------------------

class OHLCVBar(Base):
    """
    All fetched MT5 OHLCV bars.
    Acts as a persistent cache so MT5 is not re-queried for historical data.
    Unique on (symbol, timeframe, bar_timestamp).
    """
    __tablename__ = "ohlcv_bars"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "bar_timestamp",
                         name="uq_ohlcv_symbol_tf_ts"),
        Index("ix_ohlcv_symbol_tf_ts", "symbol", "timeframe", "bar_timestamp"),
    )

    id            = Column(Integer, primary_key=True, autoincrement=True)
    symbol        = Column(String(10), nullable=False)
    timeframe     = Column(String(5), nullable=False)   # D1, H4, M15, etc.
    bar_timestamp = Column(String, nullable=False)       # ISO 8601
    open          = Column(Float)
    high          = Column(Float)
    low           = Column(Float)
    close         = Column(Float)
    volume        = Column(Integer)
    created_at    = Column(String, default=lambda: datetime.utcnow().isoformat())


class EngineeredFeature(Base):
    """
    Computed feature vectors stored per bar (EAV layout: one row per feature).
    Allows adding new features without schema migration.
    Unique on (symbol, timeframe, bar_timestamp, feature_name) for idempotent upserts.
    Indexed on (symbol, timeframe, bar_timestamp) for fast range queries.
    """
    __tablename__ = "engineered_features"
    __table_args__ = (
        UniqueConstraint(
            "symbol", "timeframe", "bar_timestamp", "feature_name",
            name="uq_feat_symbol_tf_ts_name",
        ),
        Index("ix_feat_symbol_tf_ts", "symbol", "timeframe", "bar_timestamp"),
    )

    id            = Column(Integer, primary_key=True, autoincrement=True)
    symbol        = Column(String(10), nullable=False)
    timeframe     = Column(String(5), nullable=False)
    bar_timestamp = Column(String, nullable=False)
    feature_name  = Column(String(50), nullable=False)   # rsi_14, macd, etc.
    feature_value = Column(Float)
    created_at    = Column(String, default=lambda: datetime.utcnow().isoformat())


class FeatureVector(Base):
    """
    Computed feature vectors stored per bar as a single JSONB blob.
    Replaces the EAV layout (EngineeredFeature) for scalability:
    one row per bar instead of one row per feature per bar.

    With 170 features across 5 timeframes, EAV would produce ~6M rows/year
    per symbol. JSONB produces ~35K rows/year per symbol.

    Unique on (symbol, timeframe, bar_timestamp) for idempotent upserts.
    """
    __tablename__ = "feature_vectors"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "bar_timestamp",
                         name="uq_fv_symbol_tf_ts"),
        Index("ix_fv_sym_tf_ts", "symbol", "timeframe", "bar_timestamp"),
    )

    id            = Column(Integer, primary_key=True, autoincrement=True)
    symbol        = Column(String(10), nullable=False)
    timeframe     = Column(String(5), nullable=False)
    bar_timestamp = Column(String, nullable=False)       # ISO 8601
    features      = Column(JSONB, nullable=False)         # {feature_name: value, ...}
    created_at    = Column(String, default=lambda: datetime.utcnow().isoformat())


# feature_store partition window (Phase 1D). Concrete partition DDL lives
# in db_migrations._create_feature_store_partitions; constants live here
# so persist callers can filter out-of-range rows without circular imports.
FEATURE_STORE_PARTITION_START = (2000, 1)   # inclusive — earliest partition
FEATURE_STORE_PARTITION_END   = (2036, 1)   # exclusive — first uncovered month


class FeatureStoreRecord(Base):
    """
    Raw-source feature cache (Phase 1D).

    Holds untransformed external-source data (FRED macro, COT, ECB yield
    curve, Stooq sovereign yields, yfinance cross-asset) keyed by the
    source's own timestamp (release date / bar close). Distinct from
    ``feature_vectors`` — that stores the final 170-dim model input vector
    assembled per OHLCV bar; this stores the raw upstream values that feed
    into it. Kept wide (``values`` JSONB) so new source columns land
    without a migration.

    PK ``(symbol, timestamp, feature_group)`` matches
    docs/forex_expansion_plan.md §1D. ``symbol`` is the instrument the row
    is attached to (e.g. ``GBPUSD``, or ``_GLOBAL`` for source-wide data
    like the full ECB curve that isn't symbol-specific). ``feature_group``
    is the source label (``fred_macro`` / ``cot_tff`` / ``ecb_yield_curve``
    / ``stooq_yields`` / ``yfinance_cross_asset``).

    Partitioned BY RANGE (timestamp) monthly — the partition DDL lives in
    ``db_migrations._create_feature_store_partitions`` since SQLAlchemy's
    ``create_all`` only emits the parent. Partition children are created
    eagerly (2000-01 → 2035-12) to avoid runtime auto-partition triggers.

    Cache semantics (immutable-by-timestamp):
    - Closed-bar rows never change after write. ``(symbol, timestamp,
      feature_group)`` conflict → UPDATE (re-fetch wins, for upstream
      revisions).
    - ``schema_version`` bumps when a source's ``values`` layout changes
      so consumers can detect and ignore incompatible old rows.
    - ``written_at`` is ingest wall-clock; used by the weekly TTL job.
    """
    __tablename__ = "feature_store"
    __table_args__ = (
        PrimaryKeyConstraint(
            "symbol", "timestamp", "feature_group",
            name="pk_feature_store",
        ),
        # Note: no separate (symbol, timestamp) index — the PK's leftmost
        # prefix already serves any (symbol, timestamp) query. The
        # (feature_group, timestamp) index serves _GLOBAL-scope reads
        # (e.g. ECB curve) where feature_group is the leading filter.
        Index("ix_fs_group_ts", "feature_group", "timestamp"),
        {"postgresql_partition_by": "RANGE (timestamp)"},
    )

    symbol         = Column(String(16), nullable=False)
    timestamp      = Column(PG_TIMESTAMP(timezone=False), nullable=False)
    feature_group  = Column(String(40), nullable=False)
    values         = Column(JSONB, nullable=False)
    schema_version = Column(
        Integer,
        nullable=False,
        default=1,
        server_default=text("1"),  # also at DB layer so raw INSERTs succeed
    )
    written_at     = Column(
        PG_TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ModelVersion(Base):
    """
    Model training audit trail.
    One row per retrain event. version increments per model_name.
    hyperparameters stored as JSONB for flexible querying.
    """
    __tablename__ = "model_versions"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    model_name           = Column(String(50), nullable=False)   # lstm_XAUUSD, hmm_BTCUSD
    version              = Column(Integer, nullable=False)
    trained_at           = Column(String, default=lambda: datetime.utcnow().isoformat())
    trained_data_start   = Column(String)
    trained_data_end     = Column(String)
    val_loss             = Column(Float)
    directional_accuracy = Column(Float)
    hyperparameters      = Column(JSONB)    # full config snapshot


class ModelPrediction(Base):
    """
    Every prediction logged — both HMM regime and LSTM price return.
    prediction_type: "regime" | "price_return"
    Indexed on (symbol, bar_timestamp, model_name) for feedback loop joins.
    """
    __tablename__ = "model_predictions"
    __table_args__ = (
        Index("ix_pred_symbol_ts_model", "symbol", "bar_timestamp", "model_name"),
    )

    id              = Column(Integer, primary_key=True, autoincrement=True)
    symbol          = Column(String(10), nullable=False)
    bar_timestamp   = Column(String, nullable=False)
    model_name      = Column(String(50), nullable=False)
    model_version   = Column(Integer, nullable=False)
    prediction_type = Column(String(20))         # regime | price_return
    predicted_value = Column(Float)
    confidence      = Column(Float)
    created_at      = Column(String, default=lambda: datetime.utcnow().isoformat())


class ActualOutcome(Base):
    """
    Ground-truth labels computed from next-bar prices.
    actual_next_return = log(close[t+1] / close[t])
    actual_next_regime = HMM state index at bar t+1 (0–4)
    Indexed on (symbol, bar_timestamp) for fast feedback loop lookup.
    """
    __tablename__ = "actual_outcomes"
    __table_args__ = (
        UniqueConstraint("symbol", "bar_timestamp", name="uq_outcome_symbol_ts"),
        Index("ix_outcome_symbol_ts", "symbol", "bar_timestamp"),
    )

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    symbol             = Column(String(10), nullable=False)
    bar_timestamp      = Column(String, nullable=False)
    actual_next_return = Column(Float)
    actual_next_regime = Column(Integer)         # 0–4 (HMM state)
    # Triple-Barrier label computed by walking forward N H4 bars from this
    # bar's close — {-1.0, 0.0, +1.0}. Populated only once enough future
    # bars exist (20 H4 bars by default). Used by FeedbackLoop's
    # direction_correct metric so it compares like-for-like with TB-trained
    # LSTM predictions instead of raw log-return signs.
    actual_tb_label    = Column(Float)
    computed_at        = Column(String, default=lambda: datetime.utcnow().isoformat())


class PredictionError(Base):
    """
    Prediction vs actual outcome — computed by FeedbackLoop.
    direction_correct = sign(predicted_value) == sign(actual_next_return)
    Used for rolling accuracy / MSE metrics and LSTM sample weighting.
    """
    __tablename__ = "prediction_errors"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id    = Column(Integer, ForeignKey("model_predictions.id"), nullable=False)
    outcome_id       = Column(Integer, ForeignKey("actual_outcomes.id"), nullable=False)
    error_magnitude  = Column(Float)             # |predicted - actual|
    direction_correct = Column(Boolean)           # True if direction was right
    computed_at      = Column(String, default=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# ORM Models — Backtest Tables (Phase 10.2)
# ---------------------------------------------------------------------------

class BacktestRun(Base):
    """
    One walk-forward backtest execution. PK is a UUID string generated
    by the API layer so the client can poll for status immediately.
    """
    __tablename__ = "backtest_runs"

    id         = Column(String(36), primary_key=True)   # UUID
    status     = Column(String(20), nullable=False, default="pending")  # pending|running|done|failed
    symbol     = Column(String(10), nullable=False)
    timeframe  = Column(String(5), nullable=False)
    start_date = Column(String)     # ISO 8601
    end_date   = Column(String)
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())
    finished_at = Column(String, nullable=True)
    # Summary metrics (populated on completion)
    total_trades = Column(Integer, default=0)
    win_rate     = Column(Float, default=0.0)
    net_pnl      = Column(Float, default=0.0)
    max_drawdown_pct = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    error_message = Column(String, nullable=True)
    # Renamed from `mode` to avoid collision with PostgreSQL's `mode()`
    # ordered-set aggregate function — the old name triggered an asyncpg
    # WrongObjectTypeError "WITHIN GROUP is required" on every SELECT.
    run_mode = Column(String(10), default="simple")
    # Model architecture snapshot at run-creation time (so old backtests
    # remain attributable even after retrains). Nullable for legacy rows.
    model_name = Column(String(50), nullable=True)
    model_version = Column(String(50), nullable=True)
    model_trained_at = Column(String, nullable=True)


class BacktestEquity(Base):
    """Equity curve point for a backtest run."""
    __tablename__ = "backtest_equity"
    __table_args__ = (
        Index("ix_bt_eq_run", "run_id", "bar_timestamp"),
    )

    id            = Column(Integer, primary_key=True, autoincrement=True)
    run_id        = Column(String(36), ForeignKey("backtest_runs.id"), nullable=False)
    bar_timestamp = Column(String, nullable=False)
    equity        = Column(Float, nullable=False)
    drawdown_pct  = Column(Float, default=0.0)


class BacktestTrade(Base):
    """Individual trade in a backtest run."""
    __tablename__ = "backtest_trades"
    __table_args__ = (
        Index("ix_bt_tr_run", "run_id"),
    )

    id           = Column(Integer, primary_key=True, autoincrement=True)
    run_id       = Column(String(36), ForeignKey("backtest_runs.id"), nullable=False)
    symbol       = Column(String(10), nullable=False)
    direction    = Column(String(4))         # buy / sell
    entry_time   = Column(String)
    exit_time    = Column(String)
    entry_price  = Column(Float)
    exit_price   = Column(Float)
    pnl          = Column(Float, default=0.0)
    r_multiple   = Column(Float, default=0.0)
    exit_reason  = Column(String(30))        # sl / tier1 / tier2 / reversal / eod
    strategy_name = Column(String(40))       # LowVolAggressive / MidVolCautious / HighVolDefensive
    regime_label = Column(String(20))        # Crash / Bear / Neutral / Bull / Euphoria
    combined_score = Column(Float)           # Brain combined score at entry
    # the model bake-off: which primary's pipeline produced this trade.
    # Default 'lstm' covers all legacy rows (no primary tagging existed
    # pre-Sprint 4); GBM-primary backtest runs (Sprint 6) write 'gbm'.
    # Spec §1 anchor 4 — meta-labeler training filters by primary_kind.
    primary_kind = Column(String(10), nullable=False, default="lstm")


# ---------------------------------------------------------------------------
# ORM Models — Execution Quality (E-3 Phase 1)
# ---------------------------------------------------------------------------

class ExecutionEvent(Base):
    """
    Per-order execution snapshot (E-3 Phase 1).

    One row per mt5.order_send attempt. Captures the gap between what the
    strategy requested and what the broker filled, so R-1b can replace
    assumed friction with empirical slippage/spread distributions.

    Includes failed sends (retcode != DONE) — the retcode itself is
    diagnostic data for reject-cause analysis.
    """
    __tablename__ = "execution_events"
    __table_args__ = (
        Index("ix_exec_account_ts", "mt5_account", "timestamp"),
        Index("ix_exec_symbol_ts", "symbol", "timestamp"),
    )

    id                = Column(Integer, primary_key=True, autoincrement=True)
    timestamp         = Column(String, nullable=False)      # ISO 8601 UTC, post-send
    symbol            = Column(String(10), nullable=False)
    direction         = Column(String(4))                   # "buy" | "sell"
    ticket            = Column(Integer, nullable=True)      # NULL on reject
    requested_price   = Column(Float, nullable=True)        # price in our order request
    fill_price        = Column(Float, nullable=True)        # send_result.price (NULL on reject)
    slippage          = Column(Float, nullable=True)        # fill - requested (signed; NULL on reject)
    spread_at_send    = Column(Float, nullable=True)        # tick.ask - tick.bid at build time
    volume_requested  = Column(Float, nullable=True)
    volume_filled     = Column(Float, nullable=True)        # send_result.volume (NULL on reject)
    retcode           = Column(Integer, nullable=True)
    mt5_account       = Column(Integer, nullable=True)


# ---------------------------------------------------------------------------
# DataStore — Async PostgreSQL Interface
# ---------------------------------------------------------------------------

class DataStore:
    """
    Async PostgreSQL data layer for the entire trading system.

    Usage:
        store = DataStore()
        await store.connect()
        await store.bulk_insert_ohlcv(bars_list)
        df = await store.get_ohlcv_range("XAUUSD", "D1", start, end)
        await store.close()

    Or as async context manager:
        async with DataStore() as store:
            await store.bulk_insert_ohlcv(bars)
    """

    def __init__(self, dsn: Optional[str] = None):
        self._engine = build_engine(dsn)
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def connect(self) -> None:
        """Verify connection and create tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        # Hand-rolled idempotent ALTER TABLE for columns added after initial
        # table creation. PG ≥9.6 supports IF NOT EXISTS on ADD COLUMN.
        await self._apply_alter_migrations()
        logger.info("DataStore connected to PostgreSQL")

    async def _apply_alter_migrations(self) -> None:
        """Apply hand-rolled schema migrations (idempotent)."""
        from sqlalchemy import text as _text
        statements = [
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS model_name VARCHAR(50)",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS model_version VARCHAR(50)",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS model_trained_at VARCHAR",
        ]
        async with self._engine.begin() as conn:
            for sql in statements:
                try:
                    await conn.execute(_text(sql))
                except Exception as exc:
                    logger.warning("ALTER skipped (%s): %s", sql, exc)

    async def close(self) -> None:
        """Dispose connection pool."""
        await self._engine.dispose()
        logger.info("DataStore connection closed")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    # --- OHLCV ---

    async def bulk_insert_ohlcv(self, bars: list[dict]) -> int:
        """
        Bulk upsert OHLCV bars — ``ON CONFLICT DO UPDATE`` on
        ``(symbol, timeframe, bar_timestamp)``. MT5 is authoritative: a
        later fetch for the same timestamp (e.g. after the bar closes
        and a final tick lands) overwrites the stored OHLCV.

        Previously used DO NOTHING, which froze whatever provisional
        values the first tick captured — caused visible candle drift
        vs the the broker terminal on recently-closed bars.
        """
        if not bars:
            return 0
        stmt = pg_insert(OHLCVBar).values(bars)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_ohlcv_symbol_tf_ts",
            set_={
                "open":   stmt.excluded.open,
                "high":   stmt.excluded.high,
                "low":    stmt.excluded.low,
                "close":  stmt.excluded.close,
                "volume": stmt.excluded.volume,
            },
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            await session.commit()
        return result.rowcount

    async def get_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a symbol/timeframe within an optional date range.

        When ``limit`` is supplied the query returns the MOST RECENT
        ``limit`` bars via ``ORDER BY bar_timestamp DESC LIMIT N`` in
        SQL, then reverses to ascending order client-side. Pushing the
        LIMIT into the query (instead of the old pattern of fetching
        every row and ``.tail(N)`` in Python) collapses the candles
        endpoint from ~7-9s p95 to ~50-200ms on symbols with years of
        backfilled history — the P-1 candles bottleneck.

        Returns:
            DataFrame indexed by bar_timestamp (datetime) with columns:
            [open, high, low, close, volume]
            Empty DataFrame if no data found.
        """
        stmt = select(OHLCVBar).where(
            OHLCVBar.symbol == symbol,
            OHLCVBar.timeframe == timeframe,
        )
        if start is not None:
            stmt = stmt.where(OHLCVBar.bar_timestamp >= _dt_to_iso(start))
        if end is not None:
            stmt = stmt.where(OHLCVBar.bar_timestamp <= _dt_to_iso(end))

        if limit is not None and limit > 0:
            # Index on (symbol, timeframe, bar_timestamp) is scanned in
            # reverse. Then we reverse client-side to preserve the
            # documented ascending-order contract.
            stmt = stmt.order_by(OHLCVBar.bar_timestamp.desc()).limit(int(limit))
        else:
            stmt = stmt.order_by(OHLCVBar.bar_timestamp)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if limit is not None and limit > 0:
            rows = list(reversed(rows))

        df = pd.DataFrame([{
            "bar_timestamp": r.bar_timestamp,
            "open":   r.open,
            "high":   r.high,
            "low":    r.low,
            "close":  r.close,
            "volume": r.volume,
        } for r in rows])
        df["bar_timestamp"] = pd.to_datetime(df["bar_timestamp"])
        return df.set_index("bar_timestamp")

    async def get_latest_bar_timestamp(self, symbol: str, timeframe: str) -> Optional[str]:
        """Return the most recent bar_timestamp stored for a symbol/timeframe."""
        stmt = (
            select(OHLCVBar.bar_timestamp)
            .where(
                OHLCVBar.symbol == symbol,
                OHLCVBar.timeframe == timeframe,
            )
            .order_by(OHLCVBar.bar_timestamp.desc())
            .limit(1)
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    # --- Engineered Features ---

    async def save_engineered_features(
        self,
        symbol: str,
        timeframe: str,
        bar_timestamp: str,
        feature_dict: dict[str, float],
    ) -> None:
        """
        Persist computed features for one bar (idempotent upsert).

        Args:
            symbol, timeframe, bar_timestamp: Bar identifier
            feature_dict: {feature_name: value} e.g. {"rsi_14": 62.3, "macd": 0.05}
        """
        if not feature_dict:
            return

        rows = [
            {
                "symbol":        symbol,
                "timeframe":     timeframe,
                "bar_timestamp": bar_timestamp,
                "feature_name":  name,
                "feature_value": _safe_float(value),
            }
            for name, value in feature_dict.items()
        ]
        stmt = pg_insert(EngineeredFeature).values(rows)
        stmt = stmt.on_conflict_do_nothing(constraint="uq_feat_symbol_tf_ts_name")
        async with self._session_factory() as session:
            await session.execute(stmt)
            await session.commit()

    async def get_latest_feature_timestamp(
        self, symbol: str, timeframe: str
    ) -> Optional[str]:
        """Return the most recent bar_timestamp in engineered_features, or None."""
        stmt = (
            select(EngineeredFeature.bar_timestamp)
            .where(
                EngineeredFeature.symbol == symbol,
                EngineeredFeature.timeframe == timeframe,
            )
            .order_by(EngineeredFeature.bar_timestamp.desc())
            .limit(1)
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_features_range(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch features as a wide DataFrame: rows=bars, columns=features.

        Returns:
            DataFrame indexed by bar_timestamp with one column per feature.
            Empty DataFrame if no data found.
        """
        stmt = select(EngineeredFeature).where(
            EngineeredFeature.symbol == symbol,
            EngineeredFeature.timeframe == timeframe,
        )
        if start is not None:
            stmt = stmt.where(EngineeredFeature.bar_timestamp >= _dt_to_iso(start))
        if end is not None:
            stmt = stmt.where(EngineeredFeature.bar_timestamp <= _dt_to_iso(end))
        stmt = stmt.order_by(
            EngineeredFeature.bar_timestamp,
            EngineeredFeature.feature_name,
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()

        if not rows:
            return pd.DataFrame()

        long_df = pd.DataFrame([{
            "bar_timestamp": r.bar_timestamp,
            "feature_name":  r.feature_name,
            "feature_value": r.feature_value,
        } for r in rows])
        wide = long_df.pivot(
            index="bar_timestamp",
            columns="feature_name",
            values="feature_value",
        )
        wide.index = pd.to_datetime(wide.index)
        wide.columns.name = None
        return wide.sort_index()

    # --- Feature Vectors (JSONB) ---

    async def save_feature_vector(
        self,
        symbol: str,
        timeframe: str,
        bar_timestamp: str,
        feature_dict: dict[str, float],
    ) -> None:
        """
        Persist a full feature vector for one bar as a JSONB blob.

        Uses ON CONFLICT DO UPDATE so re-computing features for an
        existing bar overwrites the old blob (idempotent).

        Args:
            symbol, timeframe, bar_timestamp: Bar identifier
            feature_dict: {feature_name: value} — all features for this bar
        """
        if not feature_dict:
            return

        # Sanitize: convert NaN/inf to None for JSON compatibility
        clean = {
            k: _safe_float(v) for k, v in feature_dict.items()
        }

        stmt = pg_insert(FeatureVector).values(
            symbol=symbol,
            timeframe=timeframe,
            bar_timestamp=bar_timestamp,
            features=clean,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_fv_symbol_tf_ts",
            set_={"features": clean, "created_at": datetime.utcnow().isoformat()},
        )
        async with self._session_factory() as session:
            await session.execute(stmt)
            await session.commit()

    async def save_feature_vectors_bulk(
        self,
        rows: list[dict],
    ) -> int:
        """
        Bulk-insert feature vectors. Each dict must have:
        symbol, timeframe, bar_timestamp, features (dict).

        Uses ON CONFLICT DO NOTHING for speed during backfill.
        Returns number of newly inserted rows.
        """
        if not rows:
            return 0

        values = []
        for r in rows:
            values.append({
                "symbol":        r["symbol"],
                "timeframe":     r["timeframe"],
                "bar_timestamp": r["bar_timestamp"],
                "features":      {k: _safe_float(v) for k, v in r["features"].items()},
            })

        stmt = pg_insert(FeatureVector).values(values)
        stmt = stmt.on_conflict_do_nothing(constraint="uq_fv_symbol_tf_ts")
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            await session.commit()
        return result.rowcount

    async def get_feature_vectors_range(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch feature vectors as a wide DataFrame: rows=bars, columns=features.

        Reads from the JSONB ``feature_vectors`` table. Each row's
        ``features`` blob is unpacked into columns.

        Returns:
            DataFrame indexed by bar_timestamp with one column per feature.
            Empty DataFrame if no data found.
        """
        stmt = select(FeatureVector).where(
            FeatureVector.symbol == symbol,
            FeatureVector.timeframe == timeframe,
        )
        if start is not None:
            stmt = stmt.where(FeatureVector.bar_timestamp >= _dt_to_iso(start))
        if end is not None:
            stmt = stmt.where(FeatureVector.bar_timestamp <= _dt_to_iso(end))
        stmt = stmt.order_by(FeatureVector.bar_timestamp)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()

        if not rows:
            return pd.DataFrame()

        records = []
        for r in rows:
            row_dict = {"bar_timestamp": r.bar_timestamp}
            if r.features:
                row_dict.update(r.features)
            records.append(row_dict)

        df = pd.DataFrame(records)
        df["bar_timestamp"] = pd.to_datetime(df["bar_timestamp"])
        df = df.set_index("bar_timestamp").sort_index()
        # Convert all feature columns to float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    async def get_latest_feature_vector_timestamp(
        self, symbol: str, timeframe: str
    ) -> Optional[str]:
        """Return the most recent bar_timestamp in feature_vectors, or None."""
        stmt = (
            select(FeatureVector.bar_timestamp)
            .where(
                FeatureVector.symbol == symbol,
                FeatureVector.timeframe == timeframe,
            )
            .order_by(FeatureVector.bar_timestamp.desc())
            .limit(1)
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    # --- Feature Store (raw external-source cache, Phase 1D) ---

    async def upsert_feature_store(
        self,
        symbol: str,
        timestamp: datetime,
        feature_group: str,
        values: dict,
        schema_version: int = 1,
    ) -> None:
        """
        Write one feature_store row (idempotent upsert on PK).

        Conflict on ``(symbol, timestamp, feature_group)`` → UPDATE so a
        re-fetch of an existing row overwrites with fresh values. This is
        the immutable-by-timestamp + watermark contract: callers can re-run
        the same fetch safely; closed-bar timestamps are stable, but the
        TTL-checker job (Phase 1G) needs to be able to overwrite when an
        upstream source revises a value.

        Args:
            symbol: Instrument key. Use ``"_GLOBAL"`` for source-wide rows
                that aren't symbol-specific (e.g. the full ECB curve).
            timestamp: Naive UTC ``datetime`` — the source's timestamp,
                not ingest time. Routes to the monthly partition by year+month.
            feature_group: Source label, e.g. ``"fred_macro"``,
                ``"cot_tff"``, ``"ecb_yield_curve"``, ``"stooq_yields"``,
                ``"yfinance_cross_asset"``.
            values: Source columns as a dict. Will be JSON-serialized;
                NaN/inf are coerced to None.
            schema_version: Bump when ``values`` layout changes for this
                source so consumers can ignore incompatible old rows.
        """
        if not values:
            return

        clean = {k: _safe_float(v) if isinstance(v, (int, float)) else v
                 for k, v in values.items()}

        stmt = pg_insert(FeatureStoreRecord).values(
            symbol=symbol,
            timestamp=timestamp,
            feature_group=feature_group,
            values=clean,
            schema_version=schema_version,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="pk_feature_store",
            set_={
                "values": clean,
                "schema_version": schema_version,
                "written_at": func.now(),
            },
        )
        async with self._session_factory() as session:
            await session.execute(stmt)
            await session.commit()

    async def upsert_feature_store_bulk(
        self,
        rows: list[dict],
        *,
        mode: Literal["skip", "overwrite"] = "skip",
    ) -> int:
        """
        Bulk upsert feature_store rows. Each dict must contain
        ``symbol``, ``timestamp``, ``feature_group``, ``values``,
        and optionally ``schema_version`` (defaults to 1).

        Args:
            rows: List of row dicts to upsert.
            mode: Conflict policy:
                ``"skip"`` (default) — ``ON CONFLICT DO NOTHING``. Rows whose
                    PK already exists are silently kept. Right for backfill
                    loads where re-runs should be cheap no-ops.
                ``"overwrite"`` — ``ON CONFLICT DO UPDATE``. Existing rows
                    have ``values`` / ``schema_version`` / ``written_at``
                    overwritten with the new payload. Right for the TTL
                    safety-net job that picks up upstream revisions.

        Returns the number of rows touched (insert + update for ``"overwrite"``,
        insert-only for ``"skip"``).
        """
        if mode not in ("skip", "overwrite"):
            raise ValueError(f"mode must be 'skip' or 'overwrite', got {mode!r}")
        if not rows:
            return 0

        # Drop rows outside the partitioned timestamp window. Some sources
        # (e.g. Stooq US Treasury yields back to 1871) extend further than
        # the configured partition range; without filtering, asyncpg raises
        # CheckViolationError "no partition for given key" and the whole
        # bulk insert aborts. Better to silently drop the unreachable rows
        # and continue with the addressable history.
        lower = datetime(*FEATURE_STORE_PARTITION_START, 1)
        upper = datetime(*FEATURE_STORE_PARTITION_END, 1)
        in_range, dropped = [], 0
        for r in rows:
            ts = r["timestamp"]
            if ts < lower or ts >= upper:
                dropped += 1
                continue
            in_range.append(r)
        if dropped:
            # WARNING (not INFO) so operators tailing logs at the default
            # production level still see when source data extends beyond
            # the partition window — invisible drops are easy to miss.
            logger.warning(
                "feature_store: dropped %d row(s) outside partition window "
                "[%s, %s) — keeping %d",
                dropped, lower.date(), upper.date(), len(in_range),
            )
        if not in_range:
            return 0

        values = []
        for r in in_range:
            v = r["values"]
            clean = {k: _safe_float(x) if isinstance(x, (int, float)) else x
                     for k, x in v.items()}
            values.append({
                "symbol":         r["symbol"],
                "timestamp":      r["timestamp"],
                "feature_group":  r["feature_group"],
                "values":         clean,
                "schema_version": r.get("schema_version", 1),
            })

        # Each chunk commits in its own transaction so a transient failure on
        # chunk N (network blip, constraint violation on a single row) doesn't
        # silently roll back the prior N-1 successful chunks. Chunk size is
        # the module-level FEATURE_STORE_BULK_CHUNK constant.
        total = 0
        for i in range(0, len(values), FEATURE_STORE_BULK_CHUNK):
            chunk = values[i:i + FEATURE_STORE_BULK_CHUNK]
            stmt = pg_insert(FeatureStoreRecord).values(chunk)
            if mode == "overwrite":
                # `stmt.excluded.values` collides with the dict
                # .values() method — bracket access disambiguates.
                stmt = stmt.on_conflict_do_update(
                    constraint="pk_feature_store",
                    set_={
                        "values":         stmt.excluded["values"],
                        "schema_version": stmt.excluded.schema_version,
                        "written_at":     func.now(),
                    },
                )
            else:
                stmt = stmt.on_conflict_do_nothing(constraint="pk_feature_store")
            try:
                async with self._session_factory() as session:
                    result = await session.execute(stmt)
                    await session.commit()
                total += result.rowcount or 0
            except Exception:
                logger.exception(
                    "feature_store bulk chunk %d/%d failed (rows %d-%d) — "
                    "prior chunks committed; data may be partial",
                    (i // FEATURE_STORE_BULK_CHUNK) + 1,
                    (len(values) + FEATURE_STORE_BULK_CHUNK - 1) // FEATURE_STORE_BULK_CHUNK,
                    i, min(i + FEATURE_STORE_BULK_CHUNK, len(values)) - 1,
                )
                raise
        return total

    async def read_feature_store(
        self,
        symbol: str,
        feature_group: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch feature_store rows as a wide DataFrame indexed by timestamp.

        Each row's ``values`` JSONB blob is unpacked into columns. Suitable
        for joining onto OHLCV bars at feature-engineering time.

        Args:
            symbol: Instrument key (or ``"_GLOBAL"`` for source-wide rows).
            feature_group: Source label to filter to one source.
            start, end: Optional naive UTC bounds (inclusive). Pruned by the
                partition planner since `timestamp` is the partition key.

        Returns:
            DataFrame indexed by ``timestamp`` (datetime), one column per
            value key, sorted ascending. Empty DataFrame if no rows match.

        Performance contract — read this before adding new call sites:
            ``feature_store`` is partitioned monthly with 432 child partitions
            (2000-01 through 2035-12). The Postgres planner evaluates every
            partition's range bound on each query — bounded queries with both
            ``start`` and ``end`` prune to a handful of partitions
            (sub-millisecond planning). Queries with only ``end`` and no
            ``start`` cannot prune the early end of the timeline and incur
            ~400ms planning time per call. In a backtest loop calling this
            method per-bar, that's seconds of overhead per pass.

            Always pass both ``start`` and ``end`` for time-range reads. For
            point-in-time "as-of T" queries, pass ``start = T - lookback``
            where ``lookback`` covers the longest release-cadence in the
            requested feature_group (typically 365 days for FRED macro):

                lookback = timedelta(days=365)
                df = await store.read_feature_store(
                    symbol="GBPUSD", feature_group="fred_macro",
                    start=bar_ts - lookback, end=bar_ts,
                )
                latest = df.iloc[-1] if not df.empty else None

            For full-history bulk reads (e.g. building a backtest dataset
            once), it is fine to pass only ``start`` — the timeline upper
            end prunes naturally to ``now()`` partitions.

            See the database review report (memory/project_phase1f_feature_store.md)
            finding C-1 for the EXPLAIN ANALYZE numbers behind this.
        """
        if end is not None and start is None:
            logger.warning(
                "read_feature_store(%s, %s): point-in-time query with end "
                "but no start — partition planner cannot prune lower bound "
                "(~400ms planning overhead). Pass start=end-lookback for "
                "fast path. See data_store.py docstring for details.",
                symbol, feature_group,
            )

        stmt = select(FeatureStoreRecord).where(
            FeatureStoreRecord.symbol == symbol,
            FeatureStoreRecord.feature_group == feature_group,
        )
        if start is not None:
            stmt = stmt.where(FeatureStoreRecord.timestamp >= start)
        if end is not None:
            stmt = stmt.where(FeatureStoreRecord.timestamp <= end)
        stmt = stmt.order_by(FeatureStoreRecord.timestamp)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()

        if not rows:
            return pd.DataFrame()

        records = []
        for r in rows:
            row_dict = {"timestamp": r.timestamp}
            if r.values:
                row_dict.update(r.values)
            records.append(row_dict)

        df = pd.DataFrame(records)
        df = df.set_index("timestamp").sort_index()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    async def get_feature_store_watermark(
        self,
        symbol: str,
        feature_group: str,
    ) -> Optional[datetime]:
        """
        Return the latest ``timestamp`` written for this (symbol, feature_group).

        Used by collectors to skip already-cached history on incremental
        re-fetches. Returns ``None`` if no rows exist yet.
        """
        stmt = (
            select(FeatureStoreRecord.timestamp)
            .where(
                FeatureStoreRecord.symbol == symbol,
                FeatureStoreRecord.feature_group == feature_group,
            )
            .order_by(FeatureStoreRecord.timestamp.desc())
            .limit(1)
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    # --- Model Versions ---

    async def save_model_version(self, metadata: dict) -> int:
        """
        Save a model training event and return the new version ID.

        Auto-increments ``version`` per ``model_name`` (no DB sequence).

        Args:
            metadata: {
                "model_name": str,
                "trained_data_start": datetime | str,
                "trained_data_end":   datetime | str,
                "val_loss": float,
                "directional_accuracy": float,
                "hyperparameters": dict,
            }

        Returns:
            New version integer (auto-incremented per model_name).
        """
        model_name = metadata["model_name"]
        async with self._session_factory() as session:
            max_stmt = select(func.max(ModelVersion.version)).where(
                ModelVersion.model_name == model_name
            )
            result = await session.execute(max_stmt)
            max_v = result.scalar_one_or_none() or 0
            new_version = max_v + 1

            record = ModelVersion(
                model_name=model_name,
                version=new_version,
                trained_data_start=_to_iso_str(metadata.get("trained_data_start")),
                trained_data_end=_to_iso_str(metadata.get("trained_data_end")),
                val_loss=metadata.get("val_loss"),
                directional_accuracy=metadata.get("directional_accuracy"),
                hyperparameters=metadata.get("hyperparameters", {}),
            )
            session.add(record)
            await session.commit()
        return new_version

    async def get_latest_model_version(self, model_name: str) -> Optional[int]:
        """Return the most recent version integer for a model_name."""
        stmt = select(func.max(ModelVersion.version)).where(
            ModelVersion.model_name == model_name
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    # --- Predictions ---

    async def save_prediction(
        self,
        symbol: str,
        bar_timestamp: str,
        model_name: str,
        model_version: int,
        prediction_type: str,
        predicted_value: float,
        confidence: float,
    ) -> int:
        """
        Log a single model prediction. Returns inserted row id.

        prediction_type: "regime" or "price_return"
        """
        async with self._session_factory() as session:
            record = ModelPrediction(
                symbol=symbol,
                bar_timestamp=bar_timestamp,
                model_name=model_name,
                model_version=int(model_version),
                prediction_type=prediction_type,
                predicted_value=float(predicted_value),
                confidence=float(confidence),
            )
            session.add(record)
            await session.flush()
            inserted_id = record.id
            await session.commit()
        return inserted_id

    async def get_predictions_with_outcomes(
        self, symbol: str, limit: int = 500
    ) -> pd.DataFrame:
        """
        LEFT JOIN model_predictions with actual_outcomes and prediction_errors.

        Includes rows where the outcome or error record does not yet exist —
        the FeedbackLoop uses those NULLs to find work that still needs doing.

        Returns:
            DataFrame with columns:
            [prediction_id, outcome_id, bar_timestamp, model_name,
             prediction_type, predicted_value, confidence,
             actual_next_return, error_id, error_magnitude, direction_correct]
            Ordered by bar_timestamp DESC, limited to ``limit`` rows.
        """
        sql = text("""
            SELECT
                mp.id             AS prediction_id,
                ao.id             AS outcome_id,
                mp.bar_timestamp  AS bar_timestamp,
                mp.model_name     AS model_name,
                mp.prediction_type AS prediction_type,
                mp.predicted_value AS predicted_value,
                mp.confidence     AS confidence,
                ao.actual_next_return AS actual_next_return,
                ao.actual_tb_label AS actual_tb_label,
                pe.id             AS error_id,
                pe.error_magnitude AS error_magnitude,
                pe.direction_correct AS direction_correct
            FROM model_predictions mp
            LEFT JOIN actual_outcomes ao
                ON mp.symbol = ao.symbol
               AND mp.bar_timestamp = ao.bar_timestamp
            LEFT JOIN prediction_errors pe
                ON pe.prediction_id = mp.id
               AND pe.outcome_id = ao.id
            WHERE mp.symbol = :symbol
            ORDER BY mp.bar_timestamp DESC
            LIMIT :limit
        """)
        cols = [
            "prediction_id", "outcome_id", "bar_timestamp", "model_name",
            "prediction_type", "predicted_value", "confidence",
            "actual_next_return", "actual_tb_label",
            "error_id", "error_magnitude", "direction_correct",
        ]
        async with self._session_factory() as session:
            result = await session.execute(
                sql, {"symbol": symbol, "limit": int(limit)}
            )
            rows = result.fetchall()

        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows, columns=cols)

    # --- Actual Outcomes ---

    async def save_actual_outcome(
        self,
        symbol: str,
        bar_timestamp: str,
        actual_next_return: float,
        actual_next_regime: Optional[int] = None,
        actual_tb_label: Optional[float] = None,
    ) -> int:
        """
        Save ground-truth label for a bar. Returns the outcome row id
        (whether newly inserted or already existing).

        Uses ON CONFLICT DO NOTHING on (symbol, bar_timestamp) for the
        first-insert path, then conditionally updates ``actual_tb_label``
        on existing rows where it is still NULL — the TB label can only be
        computed once enough future bars exist, which is typically several
        days after the bar's `actual_next_return` is known.
        """
        async with self._session_factory() as session:
            stmt = pg_insert(ActualOutcome).values(
                symbol=symbol,
                bar_timestamp=bar_timestamp,
                actual_next_return=float(actual_next_return),
                actual_next_regime=actual_next_regime,
                actual_tb_label=(
                    float(actual_tb_label) if actual_tb_label is not None else None
                ),
            ).on_conflict_do_nothing(constraint="uq_outcome_symbol_ts")
            await session.execute(stmt)

            # If the row already existed and we now have a TB label that
            # wasn't there before, fill it in.
            if actual_tb_label is not None:
                update_stmt = text(
                    "UPDATE actual_outcomes SET actual_tb_label = :v "
                    "WHERE symbol = :s AND bar_timestamp = :t "
                    "AND actual_tb_label IS NULL"
                )
                await session.execute(update_stmt, {
                    "v": float(actual_tb_label),
                    "s": symbol,
                    "t": bar_timestamp,
                })
            await session.commit()

            fetch = select(ActualOutcome.id).where(
                ActualOutcome.symbol == symbol,
                ActualOutcome.bar_timestamp == bar_timestamp,
            )
            fetch_result = await session.execute(fetch)
            return fetch_result.scalar_one()

    async def backfill_tb_label(
        self,
        outcome_id: int,
        actual_tb_label: float,
    ) -> None:
        """Set actual_tb_label on an existing actual_outcomes row."""
        async with self._session_factory() as session:
            await session.execute(
                text(
                    "UPDATE actual_outcomes SET actual_tb_label = :v "
                    "WHERE id = :id"
                ),
                {"v": float(actual_tb_label), "id": int(outcome_id)},
            )
            await session.commit()

    async def get_next_bar(
        self, symbol: str, timeframe: str, bar_timestamp: str
    ) -> Optional[dict]:
        """
        Return the OHLCV bar immediately after bar_timestamp.
        Returns None if it doesn't exist yet (bar hasn't closed).
        """
        stmt = (
            select(OHLCVBar)
            .where(
                OHLCVBar.symbol == symbol,
                OHLCVBar.timeframe == timeframe,
                OHLCVBar.bar_timestamp > bar_timestamp,
            )
            .order_by(OHLCVBar.bar_timestamp)
            .limit(1)
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
        if row is None:
            return None
        return {
            "bar_timestamp": row.bar_timestamp,
            "open":   row.open,
            "high":   row.high,
            "low":    row.low,
            "close":  row.close,
            "volume": row.volume,
        }

    # --- Prediction Errors ---

    async def save_prediction_error(
        self,
        prediction_id: int,
        outcome_id: int,
        error_magnitude: float,
        direction_correct: bool,
    ) -> None:
        """Save a prediction error record (FeedbackLoop calls this)."""
        async with self._session_factory() as session:
            record = PredictionError(
                prediction_id=int(prediction_id),
                outcome_id=int(outcome_id),
                error_magnitude=float(error_magnitude),
                direction_correct=bool(direction_correct),
            )
            session.add(record)
            await session.commit()

    async def get_rolling_metrics(
        self, symbol: str, window: int = 500
    ) -> dict:
        """
        Compute rolling performance metrics from prediction_errors.

        Only counts price_return predictions (not regime), joined to
        the parent model_predictions row for the symbol filter.

        Args:
            symbol: Trading symbol
            window: Number of most recent predictions to include

        Returns:
            {
                "directional_accuracy": float,   # fraction correct
                "mse": float,                    # mean squared error
                "mae": float,                    # mean absolute error
                "n_predictions": int,            # actual count used
            }
        """
        sql = text("""
            SELECT pe.error_magnitude, pe.direction_correct
            FROM prediction_errors pe
            JOIN model_predictions mp ON pe.prediction_id = mp.id
            WHERE mp.symbol = :symbol
              AND mp.prediction_type = 'price_return'
            ORDER BY pe.computed_at DESC
            LIMIT :window
        """)
        async with self._session_factory() as session:
            result = await session.execute(
                sql, {"symbol": symbol, "window": int(window)}
            )
            rows = result.fetchall()

        if not rows:
            return {
                "directional_accuracy": 0.0,
                "mse":  0.0,
                "mae":  0.0,
                "n_predictions": 0,
            }

        errors = np.array([r[0] for r in rows], dtype=float)
        corrects = np.array([bool(r[1]) for r in rows])

        return {
            "directional_accuracy": float(corrects.mean()),
            "mse":  float((errors ** 2).mean()),
            "mae":  float(np.abs(errors).mean()),
            "n_predictions": int(len(rows)),
        }

    async def get_latest_model_versions(self) -> list[dict]:
        """
        Return one row per distinct model_name — the most recent
        retrain. Used by the Models dashboard screen.
        """
        sql = text("""
            SELECT DISTINCT ON (model_name)
                   model_name, version, trained_at, val_loss,
                   directional_accuracy,
                   trained_data_start, trained_data_end
            FROM model_versions
            ORDER BY model_name, version DESC
        """)
        async with self._session_factory() as session:
            result = await session.execute(sql)
            rows = result.fetchall()
        return [
            {
                "model_name": r[0],
                "version": int(r[1]),
                "trained_at": r[2],
                "val_loss": float(r[3]) if r[3] is not None else None,
                "directional_accuracy": (
                    float(r[4]) if r[4] is not None else None
                ),
                "trained_data_start": r[5],
                "trained_data_end":   r[6],
            }
            for r in rows
        ]

    async def get_model_version_history(
        self, model_name: str, limit: int = 20,
    ) -> list[dict]:
        """All retrain events for one model, newest version first."""
        stmt = (
            select(ModelVersion)
            .where(ModelVersion.model_name == model_name)
            .order_by(ModelVersion.version.desc())
            .limit(int(limit))
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return [
            {
                "model_name": r.model_name,
                "version": int(r.version),
                "trained_at": r.trained_at,
                "val_loss": float(r.val_loss) if r.val_loss is not None else None,
                "directional_accuracy": (
                    float(r.directional_accuracy)
                    if r.directional_accuracy is not None else None
                ),
                "trained_data_start": r.trained_data_start,
                "trained_data_end":   r.trained_data_end,
            }
            for r in rows
        ]

    async def get_accuracy_timeseries(
        self, symbol: str, window_days: int = 30,
    ) -> list[dict]:
        """
        Daily accuracy / MAE buckets for the last ``window_days`` days.
        Buckets by date portion of `bar_timestamp`.
        """
        sql = text("""
            SELECT
                substr(mp.bar_timestamp, 1, 10)  AS d,
                avg(CASE WHEN pe.direction_correct THEN 1.0 ELSE 0.0 END)
                                                  AS dir_acc,
                avg(pe.error_magnitude)           AS mae,
                count(*)                          AS n
            FROM prediction_errors pe
            JOIN model_predictions mp ON pe.prediction_id = mp.id
            WHERE mp.symbol = :symbol
              AND mp.prediction_type = 'price_return'
              AND mp.bar_timestamp >= :cutoff
            GROUP BY substr(mp.bar_timestamp, 1, 10)
            ORDER BY d ASC
        """)
        from datetime import datetime as _dt, timedelta as _td
        cutoff = (_dt.utcnow() - _td(days=int(window_days))).isoformat()
        async with self._session_factory() as session:
            result = await session.execute(
                sql, {"symbol": symbol, "cutoff": cutoff},
            )
            rows = result.fetchall()
        return [
            {
                "date": r[0],
                "directional_accuracy": float(r[1]) if r[1] is not None else 0.0,
                "mae": float(r[2]) if r[2] is not None else 0.0,
                "n": int(r[3]),
            }
            for r in rows
        ]

    # --- Original Methods (trades, signals, equity) ---

    async def save_trade(self, trade: TradeRecord) -> None:
        """Persist a completed trade."""
        async with self._session_factory() as session:
            session.add(trade)
            await session.commit()

    async def close_trade_record(
        self,
        ticket: int,
        exit_price: float,
        exit_time_iso: str,
        pnl_usd: float,
        exit_reason: Optional[str] = None,
        allow_overwrite: bool = False,
        commission_usd: Optional[float] = None,
        swap_usd: Optional[float] = None,
        # Trade-journal exit fields (Plan 1). All optional so legacy callers
        # keep working; when provided, they land on the matching row.
        close_reason_code: Optional[str] = None,
        r_multiple_at_exit: Optional[float] = None,
        bars_held: Optional[int] = None,
        exit_score: Optional[float] = None,
        regime_at_exit: Optional[str] = None,
        be_locked_at_close: Optional[bool] = None,
    ) -> bool:
        """
        Update the ``trades`` row (identified by broker ticket) with exit
        fields. Returns True on update, False if no matching row.

        When ``allow_overwrite=False`` (default), only OPEN rows are
        updated — used by full_close so we can't accidentally clobber a
        row that broker-truth has already reconciled.

        When ``allow_overwrite=True``, any matching row is overwritten —
        used by reconcile_closed_trades so authoritative broker-side PnL
        (profit + commission + swap from history_deals_get) replaces the
        gross estimate the bot wrote at close-time.
        """
        async with self._session_factory() as session:
            stmt = select(TradeRecord).where(TradeRecord.ticket == int(ticket))
            if not allow_overwrite:
                stmt = stmt.where(TradeRecord.timestamp_close.is_(None))
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return False
            row.exit_price = float(exit_price)
            row.timestamp_close = exit_time_iso
            row.pnl_usd = float(pnl_usd)
            if commission_usd is not None:
                row.commission_usd = float(commission_usd)
            if swap_usd is not None:
                row.swap_usd = float(swap_usd)
            # Trade journal — always set when caller provides values. The
            # string-ish fields are length-capped to match schema widths so
            # oversize inputs (noisy broker messages) can't fail the commit.
            if exit_reason is not None:
                row.close_reason = exit_reason[:200]
            if close_reason_code is not None:
                row.close_reason_code = close_reason_code[:30]
            if r_multiple_at_exit is not None:
                row.r_multiple_at_exit = float(r_multiple_at_exit)
            if bars_held is not None:
                row.bars_held = int(bars_held)
            if exit_score is not None:
                row.exit_score = float(exit_score)
            if regime_at_exit is not None:
                row.regime_at_exit = regime_at_exit[:10]
            if be_locked_at_close is not None:
                row.be_locked_at_close = bool(be_locked_at_close)
            await session.commit()
            return True

    async def save_signal(self, signal: SignalRecord) -> None:
        """Persist a signal event."""
        async with self._session_factory() as session:
            session.add(signal)
            await session.commit()

    async def save_equity_snapshot(self, record: EquityRecord) -> None:
        """Persist an account equity snapshot."""
        async with self._session_factory() as session:
            session.add(record)
            await session.commit()

    async def save_execution_event(self, record: ExecutionEvent) -> None:
        """Persist an order-send execution snapshot (E-3 Phase 1)."""
        async with self._session_factory() as session:
            session.add(record)
            await session.commit()

    async def save_drift_score(self, row: dict) -> None:
        """Persist a daily feature-drift score (A-8).

        Accepts a dict with the DriftScoreRecord columns and inserts it.
        """
        async with self._session_factory() as session:
            session.add(DriftScoreRecord(**row))
            await session.commit()

    async def get_trades(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Query trade history, optionally filtered by symbol and date."""
        stmt = select(TradeRecord)
        if symbol is not None:
            stmt = stmt.where(TradeRecord.symbol == symbol)
        if since is not None:
            stmt = stmt.where(TradeRecord.timestamp_open >= _dt_to_iso(since))
        stmt = stmt.order_by(TradeRecord.timestamp_open.desc())

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "id":                 r.id,
            "timestamp_open":     r.timestamp_open,
            "timestamp_close":    r.timestamp_close,
            "symbol":             r.symbol,
            "direction":          r.direction,
            "lot_size":           r.lot_size,
            "entry_price":        r.entry_price,
            "exit_price":         r.exit_price,
            "pnl_usd":            r.pnl_usd,
            "commission_usd":     getattr(r, "commission_usd", None),
            "swap_usd":           getattr(r, "swap_usd", None),
            "regime_at_entry":    r.regime_at_entry,
            "combined_score":     r.combined_score,
            "ticket":             r.ticket,
            # Trade journal
            "close_reason":       getattr(r, "close_reason", None),
            "close_reason_code":  getattr(r, "close_reason_code", None),
            "r_multiple_at_exit": getattr(r, "r_multiple_at_exit", None),
            "bars_held":          getattr(r, "bars_held", None),
            "entry_score":        getattr(r, "entry_score", None),
            "exit_score":         getattr(r, "exit_score", None),
            "regime_at_exit":     getattr(r, "regime_at_exit", None),
            "initial_stop":       getattr(r, "initial_stop", None),
            "tp_price":           getattr(r, "tp_price", None),
            "be_locked_at_close": getattr(r, "be_locked_at_close", None),
            "mt5_account":        getattr(r, "mt5_account", None),
        } for r in rows])

    async def get_daily_pnl(self, date: Optional[datetime] = None) -> float:
        """Return total realized P&L for a given date (default: today UTC)."""
        target = date or datetime.utcnow()
        date_prefix = target.strftime("%Y-%m-%d")
        stmt = select(func.coalesce(func.sum(TradeRecord.pnl_usd), 0.0)).where(
            TradeRecord.timestamp_close.like(f"{date_prefix}%")
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return float(result.scalar_one())

    async def get_equity_history(
        self, limit: int = 1000, mt5_account: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return equity history for drawdown calculation (most recent first)."""
        stmt = select(EquityRecord)
        if mt5_account is not None:
            stmt = stmt.where(EquityRecord.mt5_account == mt5_account)
        stmt = stmt.order_by(EquityRecord.timestamp.desc()).limit(int(limit))
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()

        if not rows:
            return pd.DataFrame(columns=["balance", "equity", "floating_pnl"])

        df = pd.DataFrame([{
            "timestamp":    r.timestamp,
            "balance":      r.balance,
            "equity":       r.equity,
            "floating_pnl": r.floating_pnl,
        } for r in rows])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.set_index("timestamp").sort_index()

    # --- Paginated Queries (Phase 10.2) ---

    async def get_trades_paginated(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 50,
        mt5_account: Optional[int] = None,
    ) -> tuple[list[dict], int]:
        """
        Return paginated trades and total count.

        Returns:
            (list_of_trade_dicts, total_count)
        """
        base = select(TradeRecord)
        # History tab should show CLOSED trades only — open positions
        # belong on the Positions screen with live ticker + current_stop.
        # Without this filter the Trades table shows rows with null
        # exit_price / null pnl / blank close-time.
        base = base.where(TradeRecord.timestamp_close.is_not(None))
        if mt5_account is not None:
            base = base.where(TradeRecord.mt5_account == mt5_account)
        if symbol is not None:
            base = base.where(TradeRecord.symbol == symbol)
        if since is not None:
            base = base.where(TradeRecord.timestamp_open >= _dt_to_iso(since))
        if until is not None:
            base = base.where(TradeRecord.timestamp_close <= _dt_to_iso(until))

        count_stmt = select(func.count()).select_from(base.subquery())
        data_stmt = (
            base.order_by(TradeRecord.timestamp_close.desc())
            .offset(offset)
            .limit(limit)
        )

        async with self._session_factory() as session:
            total = (await session.execute(count_stmt)).scalar_one()
            result = await session.execute(data_stmt)
            rows = result.scalars().all()

        items = [{
            "id":                 r.id,
            "timestamp_open":     r.timestamp_open,
            "timestamp_close":    r.timestamp_close,
            "symbol":             r.symbol,
            "direction":          r.direction,
            "lot_size":           r.lot_size,
            "entry_price":        r.entry_price,
            "exit_price":         r.exit_price,
            "pnl_usd":            r.pnl_usd,
            "commission_usd":     getattr(r, "commission_usd", None),
            "swap_usd":           getattr(r, "swap_usd", None),
            "regime_at_entry":    r.regime_at_entry,
            "combined_score":     r.combined_score,
            "ticket":             r.ticket,
            # Trade journal
            "close_reason":       getattr(r, "close_reason", None),
            "close_reason_code":  getattr(r, "close_reason_code", None),
            "r_multiple_at_exit": getattr(r, "r_multiple_at_exit", None),
            "bars_held":          getattr(r, "bars_held", None),
            "entry_score":        getattr(r, "entry_score", None),
            "exit_score":         getattr(r, "exit_score", None),
            "regime_at_exit":     getattr(r, "regime_at_exit", None),
            "initial_stop":       getattr(r, "initial_stop", None),
            "tp_price":           getattr(r, "tp_price", None),
            "be_locked_at_close": getattr(r, "be_locked_at_close", None),
        } for r in rows]
        return items, total

    async def get_signals_paginated(
        self,
        symbol: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
        mt5_account: Optional[int] = None,
    ) -> tuple[list[dict], int]:
        """Return paginated signal records and total count."""
        base = select(SignalRecord)
        if mt5_account is not None:
            base = base.where(SignalRecord.mt5_account == mt5_account)
        if symbol is not None:
            base = base.where(SignalRecord.symbol == symbol)

        count_stmt = select(func.count()).select_from(base.subquery())
        data_stmt = (
            base.order_by(SignalRecord.timestamp.desc())
            .offset(offset)
            .limit(limit)
        )

        async with self._session_factory() as session:
            total = (await session.execute(count_stmt)).scalar_one()
            result = await session.execute(data_stmt)
            rows = result.scalars().all()

        items = [{
            "id":                 r.id,
            "timestamp":          r.timestamp,
            "symbol":             r.symbol,
            "regime":             r.regime,
            "regime_probability": r.regime_probability,
            "lstm_prediction":    r.lstm_prediction,
            "combined_score":     r.combined_score,
            "should_trade":       r.should_trade,
            "direction":          r.direction,
        } for r in rows]
        return items, total

    # --- Backtest Persistence (Phase 10.2) ---

    async def create_backtest_run(self, run: dict) -> None:
        """Insert a new backtest_runs row (status='pending')."""
        async with self._session_factory() as session:
            session.add(BacktestRun(**run))
            await session.commit()

    async def update_backtest_run(self, run_id: str, updates: dict) -> None:
        """Update fields on an existing backtest run."""
        async with self._session_factory() as session:
            stmt = (
                select(BacktestRun)
                .where(BacktestRun.id == run_id)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return
            for k, v in updates.items():
                setattr(row, k, v)
            await session.commit()

    async def get_backtest_run(self, run_id: str) -> Optional[dict]:
        """Return a single backtest run as a dict, or None."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(BacktestRun).where(BacktestRun.id == run_id)
            )
            row = result.scalar_one_or_none()
        if row is None:
            return None
        return {
            "id": row.id, "status": row.status, "symbol": row.symbol,
            "timeframe": row.timeframe, "start_date": row.start_date,
            "end_date": row.end_date, "created_at": row.created_at,
            "finished_at": row.finished_at, "total_trades": row.total_trades,
            "win_rate": row.win_rate, "net_pnl": row.net_pnl,
            "max_drawdown_pct": row.max_drawdown_pct,
            "sharpe_ratio": row.sharpe_ratio,
            "profit_factor": row.profit_factor,
            "error_message": row.error_message,
            # DB column is `run_mode` (renamed to dodge PG mode() aggregate
            # collision); API contract still uses "mode". Without this key
            # BacktestRunSummary(**run) falls back to schema default "simple"
            # and the detail drawer shows incorrect mode.
            "mode": row.run_mode,
            "model_name": row.model_name,
            "model_version": row.model_version,
            "model_trained_at": row.model_trained_at,
        }

    async def list_backtest_runs(self, limit: int = 20) -> list[dict]:
        """Return recent backtest runs, newest first."""
        stmt = (
            select(BacktestRun)
            .order_by(BacktestRun.created_at.desc())
            .limit(limit)
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return [{
            "id": r.id, "status": r.status, "symbol": r.symbol,
            "timeframe": r.timeframe, "start_date": r.start_date,
            "end_date": r.end_date, "created_at": r.created_at,
            "finished_at": r.finished_at, "total_trades": r.total_trades,
            "win_rate": r.win_rate, "net_pnl": r.net_pnl,
            "max_drawdown_pct": r.max_drawdown_pct,
            "sharpe_ratio": r.sharpe_ratio,
            "profit_factor": r.profit_factor,
            # Python attr renamed to avoid PG mode() aggregate collision;
            # API contract still uses "mode".
            "mode": r.run_mode,
            "model_name": r.model_name,
            "model_version": r.model_version,
            "model_trained_at": r.model_trained_at,
        } for r in rows]

    async def bulk_insert_backtest_equity(self, rows: list[dict]) -> int:
        """Bulk-insert equity curve points for a backtest. Returns row count."""
        if not rows:
            return 0
        async with self._session_factory() as session:
            session.add_all([BacktestEquity(**r) for r in rows])
            await session.commit()
        return len(rows)

    async def bulk_insert_backtest_trades(self, rows: list[dict]) -> int:
        """Bulk-insert trades for a backtest. Returns row count."""
        if not rows:
            return 0
        async with self._session_factory() as session:
            session.add_all([BacktestTrade(**r) for r in rows])
            await session.commit()
        return len(rows)

    async def get_backtest_equity(self, run_id: str) -> list[dict]:
        """Return equity curve for a backtest run, ordered by timestamp."""
        stmt = (
            select(BacktestEquity)
            .where(BacktestEquity.run_id == run_id)
            .order_by(BacktestEquity.bar_timestamp)
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return [{
            "bar_timestamp": r.bar_timestamp,
            "equity": r.equity,
            "drawdown_pct": r.drawdown_pct,
        } for r in rows]

    async def get_backtest_trades(self, run_id: str) -> list[dict]:
        """Return all trades for a backtest run."""
        stmt = (
            select(BacktestTrade)
            .where(BacktestTrade.run_id == run_id)
            .order_by(BacktestTrade.entry_time)
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return [{
            "symbol": r.symbol, "direction": r.direction,
            "entry_time": r.entry_time, "exit_time": r.exit_time,
            "entry_price": r.entry_price, "exit_price": r.exit_price,
            "pnl": r.pnl, "r_multiple": r.r_multiple,
            "exit_reason": r.exit_reason,
        } for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt_to_iso(dt: datetime) -> str:
    """
    Convert a datetime to an ISO 8601 string for DB comparison.

    Uses second precision, timezone-naive format. bar_timestamps in the DB
    are stored as strings and compared lexicographically — ISO 8601 strings
    at this precision sort correctly.
    """
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _to_iso_str(v) -> Optional[str]:
    """Coerce a datetime, ISO string, or None into an ISO string or None."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


def _safe_float(v) -> Optional[float]:
    """Coerce a value to a finite float, returning None for NaN/inf/None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    return f
