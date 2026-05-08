"""
db_migrations.py — Database Schema Management

Creates and validates the PostgreSQL schema for the trading system.
Uses SQLAlchemy's ``Base.metadata.create_all`` for idempotent creation
and ``inspect()`` for verification.

Usage as a library:
    from src.data_pipeline.db_migrations import create_all_tables, verify_schema
    from src.data_pipeline.data_store import build_engine

    engine = build_engine()
    await create_all_tables(engine)
    report = await verify_schema(engine)
    assert report["ok"], report

Usage as a CLI tool:
    python -m src.data_pipeline.db_migrations              # create + verify
    python -m src.data_pipeline.db_migrations --verify-only
    python -m src.data_pipeline.db_migrations --stats      # + row counts
    python -m src.data_pipeline.db_migrations --drop-all   # DANGER
"""

import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine

# Ensure project root on sys.path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.data_store import Base, build_engine  # noqa: E402

logger = logging.getLogger(__name__)

# Expected schema — used by verify_schema() as source of truth.
# If you add/rename columns in data_store.py, update this dict too.
EXPECTED_SCHEMA: dict[str, dict] = {
    "trades": {
        "columns": [
            "id", "timestamp_open", "timestamp_close", "symbol", "direction",
            "lot_size", "entry_price", "exit_price", "pnl_usd",
            "commission_usd", "swap_usd",
            "regime_at_entry", "combined_score", "ticket",
            # Trade journal (Plan 1)
            "close_reason", "close_reason_code", "r_multiple_at_exit",
            "bars_held", "entry_score", "exit_score", "regime_at_exit",
            "initial_stop", "tp_price", "be_locked_at_close",
        ],
        "indexes": [],
    },
    "signals": {
        "columns": [
            "id", "timestamp", "symbol", "regime", "regime_probability",
            "lstm_prediction", "combined_score", "should_trade", "direction",
        ],
        "indexes": [],
    },
    "equity_history": {
        "columns": ["id", "timestamp", "balance", "equity", "floating_pnl"],
        "indexes": [],
    },
    "ohlcv_bars": {
        "columns": [
            "id", "symbol", "timeframe", "bar_timestamp",
            "open", "high", "low", "close", "volume", "created_at",
        ],
        "indexes": ["uq_ohlcv_symbol_tf_ts", "ix_ohlcv_symbol_tf_ts"],
    },
    "engineered_features": {
        "columns": [
            "id", "symbol", "timeframe", "bar_timestamp",
            "feature_name", "feature_value", "created_at",
        ],
        "indexes": ["uq_feat_symbol_tf_ts_name", "ix_feat_symbol_tf_ts"],
    },
    "model_versions": {
        "columns": [
            "id", "model_name", "version", "trained_at",
            "trained_data_start", "trained_data_end",
            "val_loss", "directional_accuracy", "hyperparameters",
        ],
        "indexes": [],
    },
    "model_predictions": {
        "columns": [
            "id", "symbol", "bar_timestamp", "model_name", "model_version",
            "prediction_type", "predicted_value", "confidence", "created_at",
        ],
        "indexes": ["ix_pred_symbol_ts_model"],
    },
    "actual_outcomes": {
        "columns": [
            "id", "symbol", "bar_timestamp", "actual_next_return",
            "actual_next_regime", "computed_at",
        ],
        "indexes": ["uq_outcome_symbol_ts", "ix_outcome_symbol_ts"],
    },
    "prediction_errors": {
        "columns": [
            "id", "prediction_id", "outcome_id", "error_magnitude",
            "direction_correct", "computed_at",
        ],
        "indexes": [],
    },
    "feature_store": {
        "columns": [
            "symbol", "timestamp", "feature_group", "values",
            "schema_version", "written_at",
        ],
        # ix_fs_symbol_ts dropped — redundant with PK leftmost prefix.
        "indexes": ["pk_feature_store", "ix_fs_group_ts"],
    },
}


# ---------------------------------------------------------------------------
# feature_store partitioning (Phase 1D)
#
# Postgres declarative range partitioning by month on the `timestamp` column.
# SQLAlchemy's create_all emits the parent table (with the
# `postgresql_partition_by` hint on FeatureStoreRecord) but doesn't create
# child partitions — those need raw DDL. We pre-create one child per month
# from FEATURE_STORE_PARTITION_START to FEATURE_STORE_PARTITION_END so writes
# never hit "no partition for given key" at runtime.
#
# Why pre-create:
# - Empty partitions are ~16 KB of metadata each, negligible cost
# - Avoids BEFORE INSERT trigger machinery (failure-prone, hides errors)
# - Promotion-friendly: same DDL re-runs idempotently on prod, finishes in
#   <1 second on an empty table, takes no locks against existing data
#
# Promotion to prod (`trading_bot`):
#   1. Take a backup snapshot:  scripts\db_backup.ps1
#   2. Merge `dev-env-setup` -> master from inside the prod workspace
#         (see project_phase1a_ohlcv_redo.md for the full promotion path).
#   3. Run the migration EXPLICITLY before restarting the bot:
#         python -m src.data_pipeline.db_migrations
#      This is REQUIRED. `DataStore.connect()` (called by main.py) only
#      runs `Base.metadata.create_all` + a 3-statement legacy
#      _apply_alter_migrations — it does NOT call _create_feature_store_partitions
#      or the comprehensive _apply_alter_migrations in this module. Skipping
#      step 3 would create the parent table without any of the 432 child
#      partitions, leading to "no partition for given key" the first time
#      anything writes to feature_store.
#   4. Restart the bot.
#   5. Verify post-restart:
#         python -m src.data_pipeline.db_migrations --verify-only
#      Expected output:
#         "Schema OK — 10/10 tables verified."
#         "feature_store partitions: 432/432 [OK]"
# ---------------------------------------------------------------------------

# Partition window constants — single source of truth in data_store.py
# (where FeatureStoreRecord lives). Re-exported here for the migration
# helpers and CLI.
from src.data_pipeline.data_store import (   # noqa: E402
    FEATURE_STORE_PARTITION_END,
    FEATURE_STORE_PARTITION_START,
)


def _iter_partition_months(start: tuple[int, int], end: tuple[int, int]):
    """Yield (year, month) for every month in [start, end). Inclusive of start, exclusive of end."""
    y, m = start
    end_y, end_m = end
    while (y, m) < (end_y, end_m):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def _partition_ddl(year: int, month: int) -> str:
    """Generate idempotent CREATE TABLE for one monthly partition.

    Embeds the H-3 autovacuum tuning directly so new partitions inherit
    the right scale factors. Postgres rejects ``ALTER TABLE`` storage
    parameters on the partitioned parent, so each leaf carries them itself.
    """
    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1
    name = f"feature_store_{year:04d}_{month:02d}"
    lo = f"{year:04d}-{month:02d}-01"
    hi = f"{next_year:04d}-{next_month:02d}-01"
    return (
        f"CREATE TABLE IF NOT EXISTS {name} "
        f"PARTITION OF feature_store "
        f"FOR VALUES FROM ('{lo}') TO ('{hi}') "
        f"WITH (autovacuum_vacuum_scale_factor = 0.01, "
        f"autovacuum_analyze_scale_factor = 0.05)"
    )


async def _apply_partition_storage_params(engine: AsyncEngine) -> int:
    """
    Retroactively apply autovacuum tuning to feature_store children that
    were created before the ``WITH (...)`` clause was added to the
    partition DDL.

    Idempotent — ``ALTER TABLE ... SET (...)`` overwrites existing values
    with the same ones. Returns the number of partitions touched.
    """
    sql = text("""
        SELECT child.relname
        FROM   pg_inherits inh
        JOIN   pg_class child ON child.oid = inh.inhrelid
        JOIN   pg_class parent ON parent.oid = inh.inhparent
        WHERE  parent.relname = 'feature_store'
        ORDER BY child.relname
    """)
    async with engine.connect() as conn:
        result = await conn.execute(sql)
        partitions = [row[0] for row in result.fetchall()]

    # Validate partition names against the expected pattern before
    # interpolating into raw SQL — defense in depth even though the
    # source is a Postgres system catalog.
    _NAME_RE = re.compile(r"^feature_store_\d{4}_\d{2}$")
    touched = 0
    for name in partitions:
        if not _NAME_RE.match(name):
            logger.warning("skipping partition with unexpected name: %r", name)
            continue
        try:
            async with engine.begin() as conn:
                await conn.execute(text(
                    f"ALTER TABLE {name} SET ("
                    f"autovacuum_vacuum_scale_factor = 0.01, "
                    f"autovacuum_analyze_scale_factor = 0.05)"
                ))
            touched += 1
        except Exception as exc:
            logger.warning("autovacuum SET skipped for %s: %s", name, exc)
    return touched


async def _create_feature_store_partitions(engine: AsyncEngine) -> dict:
    """
    Create monthly RANGE partitions on `feature_store` for the configured window.

    Idempotent — `CREATE TABLE IF NOT EXISTS PARTITION OF` is a no-op on
    existing partitions. Returns a dict with counts the caller can log.

    Skips entirely if the parent table doesn't exist yet (the caller should
    have run create_all_tables first).
    """
    async with engine.connect() as conn:
        def _check_parent(sync_conn):
            return "feature_store" in set(inspect(sync_conn).get_table_names())
        parent_exists = await conn.run_sync(_check_parent)

    if not parent_exists:
        logger.warning("feature_store parent table missing — skipping partition creation")
        return {"created_or_existing": 0, "skipped": True}

    months = list(_iter_partition_months(
        FEATURE_STORE_PARTITION_START, FEATURE_STORE_PARTITION_END,
    ))

    # Each partition DDL gets its own transaction so a failure on partition N
    # doesn't abort the transaction and silently undo the prior N-1 successes.
    # Same bug pattern fixed in _apply_alter_migrations earlier.
    for y, m in months:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(_partition_ddl(y, m)))
        except Exception as exc:
            logger.warning("partition skipped feature_store_%04d_%02d: %s", y, m, exc)

    return {"created_or_existing": len(months), "skipped": False}


async def count_feature_store_partitions(engine: AsyncEngine) -> int:
    """
    Return the number of child partitions currently attached to feature_store.

    Used by --verify-only and tests to confirm the partition wall is intact.
    """
    sql = text("""
        SELECT count(*)
        FROM   pg_inherits inh
        JOIN   pg_class child ON child.oid = inh.inhrelid
        JOIN   pg_class parent ON parent.oid = inh.inhparent
        WHERE  parent.relname = 'feature_store'
    """)
    async with engine.connect() as conn:
        result = await conn.execute(sql)
        return int(result.scalar() or 0)


# ---------------------------------------------------------------------------
# Core schema operations
# ---------------------------------------------------------------------------

async def create_all_tables(engine: AsyncEngine) -> None:
    """
    Create all ORM-defined tables if they don't already exist.

    Idempotent — safe to run on an existing database. Uses SQLAlchemy's
    metadata.create_all which issues ``CREATE TABLE IF NOT EXISTS`` semantics.

    Args:
        engine: Async SQLAlchemy engine pointing to the target PostgreSQL DB.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _apply_alter_migrations(engine)
    fs_result = await _create_feature_store_partitions(engine)
    if not fs_result["skipped"]:
        logger.info(
            "feature_store partitions: %d months ensured (%d-%02d → %d-%02d)",
            fs_result["created_or_existing"],
            *FEATURE_STORE_PARTITION_START,
            *FEATURE_STORE_PARTITION_END,
        )
        # Apply autovacuum tuning to existing children. New children inherit
        # via the WITH clause in _partition_ddl; this catches partitions
        # that were created before the WITH clause was added.
        touched = await _apply_partition_storage_params(engine)
        logger.info("feature_store autovacuum tuning applied to %d partitions", touched)
    logger.info("All tables created (or already existed).")


async def _apply_alter_migrations(engine: AsyncEngine) -> None:
    """
    Hand-rolled idempotent ALTER TABLE migrations for columns added after
    initial table creation. SQLAlchemy's create_all only creates new tables;
    column additions need explicit ALTER. PG ≥ 9.6 supports IF NOT EXISTS
    on ADD COLUMN so this is safe to run on every startup.
    """
    statements = [
        "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS model_name VARCHAR(50)",
        "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS model_version VARCHAR(50)",
        "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS model_trained_at VARCHAR",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS commission_usd DOUBLE PRECISION",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS swap_usd DOUBLE PRECISION",
        # Trade journal (Plan 1)
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS close_reason VARCHAR(200)",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS close_reason_code VARCHAR(30)",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS r_multiple_at_exit DOUBLE PRECISION",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS bars_held INTEGER",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS entry_score DOUBLE PRECISION",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS exit_score DOUBLE PRECISION",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS regime_at_exit VARCHAR(10)",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS initial_stop DOUBLE PRECISION",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS tp_price DOUBLE PRECISION",
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS be_locked_at_close BOOLEAN",
        # Predictor traceability (2026-05-02): every live trade now stamped
        # with `lstm_<SYMBOL>@<YYYY-MM-DD>` from loaded LSTM file mtime so
        # operators can trace any trade to a retrain. Backfilled by
        # scripts/backfill_trade_model_versions.py from MLflow run history.
        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS model_version VARCHAR(64)",
        "CREATE INDEX IF NOT EXISTS ix_trades_model_version ON trades (model_version)",
        # Phase A Sprint 4: tag backtest_trades with the primary that
        # produced them so the meta-labeler can train per-primary
        # (spec §1 anchor 4). Default 'lstm' covers legacy rows.
        "ALTER TABLE backtest_trades ADD COLUMN IF NOT EXISTS "
        "primary_kind VARCHAR(10) NOT NULL DEFAULT 'lstm'",
        # feature_store: ensure schema_version has a DB-level default so raw
        # INSERTs (psql, pg_restore, non-ORM scripts) don't fail NOT NULL.
        "ALTER TABLE feature_store ALTER COLUMN schema_version SET DEFAULT 1",
        # feature_store H-1 (review 2026-04-25): drop ix_fs_symbol_ts as
        # redundant with the PK's (symbol, timestamp, feature_group)
        # leftmost prefix. Cascades to all 432 child partition indexes.
        "DROP INDEX IF EXISTS ix_fs_symbol_ts",
        # feature_store N-2: revoke unneeded TRIGGER privilege from the
        # application role. Application code never creates triggers.
        # Idempotent — REVOKE is a no-op if the grant doesn't exist.
        "REVOKE TRIGGER ON TABLE feature_store FROM cortex",
        # H-3 autovacuum tuning is NOT here — Postgres rejects storage
        # parameters on partitioned parent tables. Applied per-leaf in
        # _create_feature_store_partitions instead.
    ]
    # Each statement gets its own transaction so a failure on one doesn't
    # roll back the rest. Using ``engine.begin()`` once for the whole loop
    # would put every statement in a single tx — first failure aborts the
    # tx and the try/except merely logs while subsequent inserts no-op.
    for sql in statements:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(sql))
        except Exception as exc:
            logger.warning("ALTER skipped (%s): %s", sql, exc)


async def drop_all_tables(engine: AsyncEngine) -> None:
    """
    Drop ALL tables defined on Base.metadata.

    DESTRUCTIVE — used only in development or testing.
    Callers should prompt for confirmation before invoking this.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("All trading system tables DROPPED.")


async def verify_schema(engine: AsyncEngine) -> dict:
    """
    Verify that all expected tables and their columns exist in the database.

    Returns:
        {
            "ok":              bool,
            "tables_found":    list[str],
            "tables_missing":  list[str],
            "column_errors":   list[str],   # "table.column" for each missing column
            "index_warnings":  list[str],   # "table: missing_index" for missing indexes
        }
    """
    async with engine.connect() as conn:
        def _inspect(sync_conn):
            inspector = inspect(sync_conn)
            existing_tables = set(inspector.get_table_names())
            col_errors: list[str] = []
            idx_warnings: list[str] = []

            for table_name, schema in EXPECTED_SCHEMA.items():
                if table_name not in existing_tables:
                    continue

                existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
                for col in schema["columns"]:
                    if col not in existing_cols:
                        col_errors.append(f"{table_name}.{col}")

                existing_idx = {i["name"] for i in inspector.get_indexes(table_name)}
                existing_uq = {c["name"] for c in inspector.get_unique_constraints(table_name)}
                pk_info = inspector.get_pk_constraint(table_name) or {}
                pk_name = {pk_info["name"]} if pk_info.get("name") else set()
                have = existing_idx | existing_uq | pk_name
                for idx_name in schema.get("indexes", []):
                    if idx_name not in have:
                        idx_warnings.append(f"{table_name}: {idx_name}")

            return existing_tables, col_errors, idx_warnings

        existing_tables, column_errors, index_warnings = await conn.run_sync(_inspect)

    expected_tables = set(EXPECTED_SCHEMA.keys())
    tables_missing = sorted(expected_tables - existing_tables)
    tables_found = sorted(expected_tables & existing_tables)

    ok = len(tables_missing) == 0 and len(column_errors) == 0

    return {
        "ok": ok,
        "tables_found": tables_found,
        "tables_missing": tables_missing,
        "column_errors": column_errors,
        "index_warnings": index_warnings,
    }


async def print_table_stats(engine: AsyncEngine) -> None:
    """Print row counts for all trading system tables.

    Security Audit H2: validate table_name against allowlist before
    interpolating into raw SQL (defense-in-depth, even though callers
    use a static dict today).
    """
    ALLOWED_TABLES = frozenset(EXPECTED_SCHEMA.keys())
    async with engine.connect() as conn:
        for table_name in EXPECTED_SCHEMA:
            # Defense: reject anything not in the allowlist
            if table_name not in ALLOWED_TABLES:
                print(f"  {table_name:<25} SKIPPED (not in allowlist)")
                continue
            try:
                result = await conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = result.scalar() or 0
                print(f"  {table_name:<25} {count:>12,} rows")
            except Exception as e:
                print(f"  {table_name:<25} ERROR: {e}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

async def _main_async(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv
    load_dotenv()

    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        print("ERROR: POSTGRES_DSN environment variable not set.", file=sys.stderr)
        print("  Set it in .env or export it:", file=sys.stderr)
        print("  POSTGRES_DSN=postgresql+asyncpg://user:pass@host:port/db", file=sys.stderr)
        return 1

    engine = build_engine(dsn)

    try:
        if args.drop_all:
            confirm = input(
                "WARNING: This will DROP all trading tables. "
                "Type 'yes' to confirm: "
            )
            if confirm.strip().lower() != "yes":
                print("Aborted.")
                return 0
            await drop_all_tables(engine)
            print("All tables dropped.")
            return 0

        if not args.verify_only:
            print("Creating tables...")
            await create_all_tables(engine)
            print("Tables created.")

        print("\nVerifying schema...")
        report = await verify_schema(engine)

        if report["ok"]:
            print(f"  Schema OK — {len(report['tables_found'])}/{len(EXPECTED_SCHEMA)} tables verified.")
            if report["index_warnings"]:
                print(f"  (non-fatal) missing indexes: {report['index_warnings']}")
            if "feature_store" in report["tables_found"]:
                fs_count = await count_feature_store_partitions(engine)
                expected_partitions = len(list(_iter_partition_months(
                    FEATURE_STORE_PARTITION_START, FEATURE_STORE_PARTITION_END,
                )))
                marker = "OK" if fs_count == expected_partitions else "WARN"
                print(f"  feature_store partitions: {fs_count}/{expected_partitions} [{marker}]")
        else:
            print("  Schema INVALID:")
            if report["tables_missing"]:
                print(f"    MISSING TABLES:  {report['tables_missing']}")
            if report["column_errors"]:
                print(f"    MISSING COLUMNS: {report['column_errors']}")
            if report["index_warnings"]:
                print(f"    MISSING INDEXES: {report['index_warnings']}")
            return 1

        if args.stats:
            print("\nRow counts:")
            await print_table_stats(engine)

    finally:
        await engine.dispose()

    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,  # suppress SQLAlchemy noise on the CLI
        format="%(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Create and verify the trading system PostgreSQL schema"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify existing schema — do not create tables",
    )
    parser.add_argument(
        "--drop-all", action="store_true",
        help="DROP all trading tables (DANGER — development only)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print row counts for each table after verification",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
