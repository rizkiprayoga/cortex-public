# Learning Diagrams

Visual cues for studying the Cortex system. Every diagram is a Mermaid code
block rendered by VSCode's Markdown preview (install the **Markdown Preview
Mermaid Support** extension if blocks show as raw text).

**How to view**: open any `.md` file → `Ctrl-Shift-V` → diagram renders inline.

## Reading order

Recommended order for a first pass. Each builds on the previous one.

1. **[high_level_architecture.md](high_level_architecture.md)** — every
   container in one view: external systems, bot process, persistence, API,
   scheduled jobs, runtime infra. Start here for the full system map.

2. **[data_flow.md](data_flow.md)** — end-to-end pipeline view. What moves
   through the components and in which direction. Zoomed in one level from
   the architecture map.

3. **[brain_pipeline.md](brain_pipeline.md)** — how a single H4 bar becomes a
   signal. Covers HMM + LSTM + combiner + combiner-internal gates.

4. **[safety_architecture.md](safety_architecture.md)** — the independent
   safety thread. Why Brain proposes but Safety disposes, and how the
   circuit breaker and emergency close interact.

5. **[order_lifecycle.md](order_lifecycle.md)** — position state machine
   from signal to close. Pyramiding, exit conditions, time budgets per
   symbol.

6. **[gating_sequence.md](gating_sequence.md)** — the complete gate chain
   (~17 gates from bar close to broker fill). This is essentially
   `brain_pipeline` + `order_lifecycle` unrolled into every check that
   actually runs.

## What each diagram answers

| Diagram                  | Core question                                               |
|--------------------------|-------------------------------------------------------------|
| high_level_architecture  | *What are all the moving parts and where do they live?*     |
| data_flow                | *What data moves between components?*                       |
| brain_pipeline           | *How does the bot decide there's a signal?*                 |
| safety_architecture      | *What stops the bot from blowing up?*                       |
| order_lifecycle          | *What states can a position be in, and how does it exit?*   |
| gating_sequence          | *Why does a signal get rejected at each point?*             |

## For newcomers

If you only have 10 minutes:

1. Skim `high_level_architecture.md` (2 min) — map all the pieces.
2. Read `brain_pipeline.md` diagram + legend (5 min) — get the decision logic.
3. Read `safety_architecture.md` "Three layers" section (3 min) — get the
   guardrails.

That's enough to reason about any feature request or bug report at a
high level. Come back for the other three when you need them.

## Keeping these current

These diagrams describe **design intent**, not configuration. Config values
(risk %, thresholds, retry counts) are enforced by the
`doc-check` linter — see CLAUDE.md for details. Run
`python scripts/check_docs_consistency.py -v` to verify numeric claims.

Design changes (new gates, new states, new subsystems) should update the
relevant diagram as part of the same PR.
