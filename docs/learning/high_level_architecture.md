# High-Level Architecture

Every component of the Cortex system in one view. This is the C4-style
container diagram — shows what runs where, what talks to what, and which
piece stores which data.

For **what flows through** these components, see
[data_flow.md](data_flow.md). For **internals** of any single component,
see the specialized diagram in its subgraph label.

## Diagram

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 50, 'rankSpacing': 65, 'padding': 18}, 'themeVariables': {'fontSize': '12px'}}}%%
flowchart TB
    subgraph external["External systems"]
        direction TB
        Broker[("MT5 broker<br/>the broker")]
        FRED[("FRED API<br/>economic data")]
        News[("NewsAPI<br/>headlines")]
        TG[("Telegram<br/>Bot API")]
        Email[("SMTP<br/>email relay")]
    end

    subgraph bot["Bot process · main.py"]
        direction TB
        subgraph brainModule["Brain<br/>(see brain_pipeline.md)"]
            direction TB
            Features["Feature engineering"]
            HMMModel["HMM"]
            LSTMModel["LSTM"]
            Combiner["Signal combiner"]
        end

        subgraph decisionModule["Decision"]
            direction TB
            Strategy["Strategy orchestrator"]
            Portfolio["PortfolioManager"]
            Sizer["PositionSizer"]
        end

        subgraph safetyModule["Safety thread · independent<br/>(see safety_architecture.md)"]
            direction TB
            Risk["RiskMonitor"]
            Breaker["CircuitBreaker"]
            Emerg["EmergencyClose"]
            Invar["Invariants log"]
        end

        subgraph execModule["Execution<br/>(see order_lifecycle.md)"]
            direction TB
            Order["OrderManager"]
        end

        Alerts["Alert manager"]
        Heartbeat["Heartbeat writer"]
    end

    subgraph persist["Persistence"]
        direction TB
        PG[("Postgres<br/>asyncpg<br/>ohlcv · trades · signals<br/>equity · drift · backtests")]
        CSV[("CSV logs<br/>signal_audit<br/>trade_events<br/>tick_summary · invariants")]
        Models[("Model artifacts<br/>data/models/<br/>*.pkl · *.pt<br/>training distributions")]
        MLflow[("MLflow registry<br/>data/mlflow/<br/>runs · params · metrics")]
        Backups[("DB backups<br/>GFS rotation<br/>7 daily · 4 weekly · 3 monthly")]
    end

    subgraph api["API + UI"]
        direction TB
        FastAPI["FastAPI · :8787<br/>routes: live · history<br/>config · backtest · models"]
        SSE["SSE stream<br/>2s interval"]
        React["React SPA<br/>static dist<br/>Overview · Signals · History<br/>Models · Backtest · System"]
    end

    subgraph offline["Scheduled offline jobs"]
        direction TB
        TrainLSTM["Monthly LSTM retrain<br/>1st @ 03:00 UTC"]
        TrainHMM["Monthly HMM retrain"]
        Drift["Daily drift monitor<br/>01:00 UTC"]
        Weekly["Weekly summary email<br/>Sun 23:55 UTC"]
        Daily["Daily summary alert"]
        DBBackup["Nightly DB backup<br/>22:00 UTC"]
        Backtest["Backtest + CPCV<br/>on-demand"]
    end

    subgraph runtime["Runtime infra"]
        direction TB
        Task["Windows Scheduled Task<br/>CortexTradingBot"]
        NSSM["NSSM service<br/>(optional)"]
        PID["PID lock<br/>bot_heartbeat.json"]
    end

    %% External → Bot
    Broker -->|ticks · orders| Order
    Broker -->|OHLCV| Features
    FRED -.->|macro features| Features
    News -.->|sentiment| Features
    Alerts -->|events| TG
    Alerts -->|digest/weekly| Email

    %% Brain internal flow (abbreviated — see brain_pipeline.md)
    Features --> HMMModel --> Combiner
    Features --> LSTMModel --> Combiner

    %% Brain → Decision → Execution
    Combiner --> Strategy
    Strategy --> Portfolio
    Portfolio --> Sizer
    Sizer --> Order

    %% Safety veto
    Risk -.->|monitors| Order
    Breaker -.->|size multiplier| Sizer
    Breaker -.->|halt| Portfolio
    Emerg -.->|force close| Order

    %% Bot → Persistence
    Order -->|trades| PG
    Combiner -->|signals| PG
    Combiner -->|audit| CSV
    Order -->|events| CSV
    Invar --> CSV
    LSTMModel --- Models
    HMMModel --- Models
    Heartbeat --> PID

    %% Offline jobs → Persistence
    TrainLSTM --> Models
    TrainLSTM --> MLflow
    TrainHMM --> Models
    TrainHMM --> MLflow
    DBBackup --> Backups
    Backtest --> PG
    Backtest --> MLflow
    Drift --> PG

    %% Runtime boots bot
    Task -.->|starts| bot
    NSSM -.->|alt start| bot

    %% Persistence → API → UI
    PG --> FastAPI
    CSV --> FastAPI
    Models --> FastAPI
    MLflow --> FastAPI
    FastAPI --> SSE
    SSE --> React
    FastAPI --> React

    %% Alerts sources
    Order --> Alerts
    Breaker --> Alerts
    Invar --> Alerts
    Drift --> Alerts
    Weekly --> Alerts
    Daily --> Alerts

    classDef extCls fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e
    classDef brainCls fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef decCls fill:#f0fdf4,stroke:#15803d,color:#14532d
    classDef safeCls fill:#fee2e2,stroke:#b91c1c,color:#7f1d1d
    classDef execCls fill:#faf5ff,stroke:#7c3aed,color:#4c1d95
    classDef persCls fill:#f1f5f9,stroke:#475569,color:#0f172a
    classDef apiCls fill:#fffbeb,stroke:#b45309,color:#78350f
    classDef offCls fill:#fdf4ff,stroke:#a21caf,color:#581c87
    classDef runCls fill:#ecfccb,stroke:#4d7c0f,color:#1a2e05
    classDef helperCls fill:#fef2f2,stroke:#b91c1c,color:#7f1d1d

    class Broker,FRED,News,TG,Email extCls
    class Features,HMMModel,LSTMModel,Combiner brainCls
    class Strategy,Portfolio,Sizer decCls
    class Risk,Breaker,Emerg,Invar safeCls
    class Order execCls
    class Alerts,Heartbeat helperCls
    class PG,CSV,Models,MLflow,Backups persCls
    class FastAPI,SSE,React apiCls
    class TrainLSTM,TrainHMM,Drift,Weekly,Daily,DBBackup,Backtest offCls
    class Task,NSSM,PID runCls
```

## Containers, grouped

### External systems (5)

| Container        | Role                                  | Criticality    |
|------------------|---------------------------------------|----------------|
| **MT5 broker**   | Market data + order routing           | Hard dependency |
| **FRED API**     | Economic/macro features               | Soft (cached, bot runs without it) |
| **NewsAPI**      | News sentiment per-symbol             | Soft (empty → sentiment=0.0) |
| **Telegram**     | Real-time operator alerts             | Soft (logs stay) |
| **Email (SMTP)** | Daily/weekly digests                  | Soft            |

### Bot process · `main.py` (4 logical modules + helpers)

Runs as a **single Python process** but with two threads: main (brain work)
and safety (watchdog). The PID is captured to
`data/logs/bot_heartbeat.json` so the launcher can detect duplicates.

| Module            | Key files                                  | Drill-down |
|-------------------|--------------------------------------------|------------|
| **Brain**         | `src/brain/`, `src/data_pipeline/`         | [brain_pipeline.md](brain_pipeline.md) |
| **Decision**      | `src/strategy/`, `src/allocation/`         | [order_lifecycle.md](order_lifecycle.md) |
| **Safety**        | `src/safety/`                              | [safety_architecture.md](safety_architecture.md) |
| **Execution**     | `src/broker/`                              | [gating_sequence.md](gating_sequence.md) |
| Alert manager     | `src/alerts/manager.py`                    | Telegram + email routing |
| Heartbeat writer  | in-process                                 | PID lock + equity snapshot |

### Persistence (5 stores, different jobs)

| Store                 | What lives there                                              | Why separate |
|-----------------------|---------------------------------------------------------------|--------------|
| **Postgres**          | OHLCV · trades · signals · equity · drift · backtests         | Structured, queryable, account-segmented |
| **CSV logs**          | signal_audit · trade_events · tick_summary · invariants       | Forensic detail Postgres doesn't carry; survives DB wipe |
| **Model artifacts**   | `data/models/{lstm,hmm}_SYMBOL.{pt,pkl}` + training distributions | Binary, regenerated monthly |
| **MLflow registry**   | Runs, params, metrics, dataset fingerprints                   | Training provenance + `model_bench.py` comparisons |
| **DB backups**        | Nightly `pg_dump -Fc` with GFS rotation                       | Disaster recovery (see `docs/operations/postgres_dr.md`) |

### API + UI

- **FastAPI** on port 8787. Single source of truth for the dashboard.
  Route modules: `live`, `history`, `config`, `backtest`, `models`,
  `accounts`, `system`, `auth`, `invariants`, `news`.
- **SSE stream** broadcasts live state (regime, positions, equity) every
  2 seconds.
- **React SPA** is statically built to `src/api/static/dist/` and served by
  FastAPI. Screens: Overview, Signals, History, Models, Backtest, System.
  Bot runs even if the frontend is broken — they're fully decoupled.

### Scheduled offline jobs (7)

Independent processes kicked off by Windows Task Scheduler or cron
(depending on deployment). Never run in-process — the bot can't accidentally
block training, and training can't accidentally touch the running bot.

| Job                   | Schedule                      | Writes to           |
|-----------------------|-------------------------------|---------------------|
| LSTM retrain          | 1st of month · 03:00 UTC      | Models + MLflow     |
| HMM retrain           | 1st of month                  | Models + MLflow     |
| Drift monitor         | Daily · 01:00 UTC             | `drift_scores` (PG) |
| DB backup             | Daily · 22:00 UTC             | Backups directory   |
| Daily summary alert   | End-of-day                    | Telegram + email    |
| Weekly summary email  | Sunday · 23:55 UTC            | Email only          |
| Backtest + CPCV       | On-demand                     | PG + MLflow         |

### Runtime infrastructure

- **Windows Scheduled Task** `CortexTradingBot` is the primary launcher.
- **NSSM service** is the alternate — wraps the bot as a Windows service
  (optional, not always enabled).
- **PID lock** via `bot_heartbeat.json` prevents double-start (learned from
  the April-13 "duplicate bot process" incident).

## Key invariants the architecture enforces

1. **Brain never calls MT5 directly** — only OrderManager talks to the
   broker. Every other module has to go through it.
2. **Safety thread is a separate thread, not a separate process** — but it
   doesn't share state with brain beyond the atomic `positions_lock`.
3. **Training is offline** — the running bot never retrains a model.
   Monthly subprocess is the only path.
4. **Dashboard is read-only from the bot's perspective** — operator actions
   (account switch, pause) go through FastAPI endpoints that write to
   `LiveState`, which the bot polls.
5. **Account segmentation** everywhere — trades, signals, equity all tagged
   with `mt5_account`. Switching accounts gives you a clean view.

## Trust boundaries

- **Outside the bot process** (external systems, operator UI) → never
  trusted. All input validated at the boundary.
- **Inside the bot process** → trusted (dataclasses over Pydantic for
  internal types).
- **Offline jobs** are separate trust zones — they write to persistent
  stores the bot will consume, so their output is effectively an external
  input after the fact. Hence MLflow fingerprinting, dataset hashes, and
  head-shape checks on model load.

## What this diagram intentionally hides

- Request/response details inside FastAPI routes (see `src/api/routes/`)
- Exact retry semantics inside `OrderManager` (3× 20s on transient only)
- The five specific strategies inside the orchestrator (high_vol,
  mid_vol, low_vol) — strategy selection is a runtime policy, not a
  container.
- Individual dashboard screens — the React app is one "container" here
  but is actually 6+ lazy-loaded screens.

Drill into the per-subsystem diagrams for any of these.
