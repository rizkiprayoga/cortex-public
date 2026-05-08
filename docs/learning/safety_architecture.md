# Safety Architecture — the independent veto

The load-bearing rule from CLAUDE.md:

> **Safety thread is independent of Brain** — absolute veto.

The brain can be confidently bullish and the risk monitor will still slam the
emergency brake if equity crashes. This file shows how that independence is
wired up and what each layer actually watches.

## Diagram

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 55, 'rankSpacing': 70, 'padding': 20}, 'themeVariables': {'fontSize': '13px'}}}%%
flowchart LR
    subgraph brainThread["Brain thread · main event loop"]
        direction TB
        Tick["M15 tick"] --> BrainWork["features → HMM/LSTM<br/>→ combiner → strategy<br/>→ portfolio → sizer"]
        BrainWork --> Order["OrderManager<br/>send/modify/close"]
    end

    subgraph safetyThread["Safety thread · independent"]
        direction TB
        RiskLoop["RiskMonitor loop<br/>src/safety/risk_monitor.py"]
        Breaker["CircuitBreaker<br/>src/safety/circuit_breaker.py"]
        Emerg["EmergencyClose<br/>src/safety/emergency_close.py"]
        Invar["Invariants log<br/>src/safety/invariants.py"]

        RiskLoop -->|every N seconds| Watch["watch: equity · positions<br/>exposure · drift"]
        Watch --> Breaker
        Breaker -->|soft trip| SoftAct["reduce size multiplier<br/>0.5× · 0.25×"]
        Breaker -->|hard trip| HardAct["halt new trades<br/>pause N hours"]
        Breaker -->|hard + threshold| Emerg
        Emerg -->|"force-close all"| Order
    end

    subgraph triggers["Breaker triggers"]
        direction TB
        T1["Daily loss<br/>soft 3% · hard 5%"]
        T2["Weekly loss<br/>soft 5% · hard 7%"]
        T3["Peak drawdown<br/>hard 10%"]
        T4["Consec SL<br/>4 losses → 4h pause"]
    end

    T1 --> Breaker
    T2 --> Breaker
    T3 --> Breaker
    T4 --> Breaker

    SoftAct -.->|consulted by| BrainWork
    HardAct -.->|veto| Order
    Breaker --> Invar
    Emerg --> Invar

    classDef brainCls fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef safetyCls fill:#fef2f2,stroke:#b91c1c,color:#7f1d1d
    classDef trigCls fill:#fff7ed,stroke:#c2410c,color:#7c2d12
    classDef actCls fill:#fee2e2,stroke:#991b1b,color:#7f1d1d

    class Tick,BrainWork,Order brainCls
    class RiskLoop,Watch,Breaker,Emerg,Invar safetyCls
    class T1,T2,T3,T4 trigCls
    class SoftAct,HardAct actCls
```

## Three layers, three jobs

### 1. RiskMonitor · continuous watchdog

- Lives in its own thread — started by `main.py` at boot, runs until shutdown.
- Polls equity + open positions + exposure on a short interval (seconds, not
  bars). Never blocked by brain work.
- Holds a reference to `positions_lock` so it can atomically read broker state
  while the brain is mid-evaluation.
- Doesn't make trading decisions — it **feeds** the CircuitBreaker with live
  measurements.

### 2. CircuitBreaker · rule engine

- Stateless *function*: given current equity + recent trade outcomes, it
  returns which breakers are tripped and what size-multiplier should apply.
- **Soft tier** (e.g., 3% daily loss) → position size × 0.5. Brain keeps
  trading but smaller.
- **Hard tier** (e.g., 5% daily loss) → no new trades. Existing positions
  keep running with their own stops.
- **Critical tier** (e.g., 10% peak drawdown) → triggers EmergencyClose.

### 3. EmergencyClose · last resort

- Only fires on catastrophic conditions (peak drawdown, consecutive critical
  breakers).
- Sends market-close orders for every open position.
- Writes a breaker event + invariant violation, alerts operator via Telegram.

## Trigger table

| Trigger              | Soft    | Hard    | Effect                                  |
|----------------------|--------:|--------:|-----------------------------------------|
| Daily loss           | 3%      | 5%      | ×0.5 sizing → no new trades             |
| Weekly loss          | 5%      | 7%      | ×0.5 sizing → no new trades             |
| Peak drawdown        | —       | 10%     | EmergencyClose                          |
| Consecutive stops    | —       | 4 in a row | 4h pause                             |

All numbers sourced from `config/settings.yaml` and validated by the doc-drift
linter.

## Why this separation matters

Three failure modes it prevents:

1. **Brain stuck in a bad loop** — if the combiner starts producing bad
   signals (bug, data corruption, bad retrain), RiskMonitor still sees equity
   bleeding and halts trading.
2. **Single-thread deadlock** — brain work can block (feature engineering on a
   slow day can take seconds). Running safety in the same thread would delay
   breaker checks by that long.
3. **Silent regression** — if a change to brain code accidentally bypasses a
   gate, safety is still watching the *outcome* (equity) and will catch it.

## Related concepts

- **Invariants system** — orthogonal layer that guards runtime truths
  (e.g., "every open position has a stop-loss"). Violations write to
  `data/logs/invariants.jsonl` and surface on the dashboard Health card.
  See `memory/project_invariants_system.md`.
- **Direction-conflict guard** — lives in the brain layer, but enforces a
  similar "don't fire if we're not sure" principle. Lives in
  `src/strategy/orchestrator.py`.

## Read next

- [order_lifecycle.md](order_lifecycle.md) — how the breaker's size
  multiplier actually lands on an order.
- [gating_sequence.md](gating_sequence.md) — all gates from bar close to
  broker fill, including where safety checks slot in.
