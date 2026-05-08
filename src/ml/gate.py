"""
Model-promotion decision gate for T-8 piece 2 (``scripts/model_bench.py``).

The gate encodes the rule from BACKLOG T-8: a candidate replaces the current
model only if it matches or beats the current on at least **2 of 3**:

- Portfolio profit factor (higher is better)
- Portfolio max drawdown % (lower is better)
- Trade-count stability (within a configurable tolerance, default ±20%)

Ties on any criterion count as "not worse" and therefore satisfy that
criterion — the gate's purpose is to block regressions, not force strict
improvements that a refactor producing identical outputs couldn't pass.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    passed: bool
    criteria_met: int
    pf_improved: bool
    dd_improved: bool
    trades_stable: bool
    rationale: list[str]


def _fmt_improvement(name: str, current: float, candidate: float,
                     higher_is_better: bool, fmt: str = ".2f") -> str:
    direction = "↑" if (candidate > current) == higher_is_better else (
        "=" if current == candidate else "↓"
    )
    better = "✓" if direction in ("↑", "=") else "✗"
    return (
        f"{name:<12} {better}  current={current:{fmt}} "
        f"candidate={candidate:{fmt}}  {direction}"
    )


def compute_gate_verdict(
    *,
    current_portfolio_pf: float,
    current_portfolio_dd: float,
    current_total_trades: int,
    candidate_portfolio_pf: float,
    candidate_portfolio_dd: float,
    candidate_total_trades: int,
    trade_stability_tolerance: float = 0.20,
) -> GateResult:
    """Apply the 2-of-3 decision gate.

    Args:
        current_portfolio_pf: Current (baseline) portfolio profit factor.
        current_portfolio_dd: Current max drawdown as a positive percent.
        current_total_trades: Current total trade count.
        candidate_portfolio_pf: Candidate portfolio profit factor.
        candidate_portfolio_dd: Candidate max drawdown as a positive percent.
        candidate_total_trades: Candidate total trade count.
        trade_stability_tolerance: Fractional tolerance on trade-count drift.
            Default 0.20 = ±20%. Value of 0.0 means strict equality required.

    Returns:
        A ``GateResult`` with per-criterion booleans, a pass/fail verdict,
        the count of criteria met, and a human-readable rationale.
    """
    # Criterion 1: PF
    pf_improved = candidate_portfolio_pf >= current_portfolio_pf

    # Criterion 2: DD (lower is better)
    dd_improved = candidate_portfolio_dd <= current_portfolio_dd

    # Criterion 3: trade-count stability
    if current_total_trades == 0:
        # Division is undefined; treat as stable only if both are zero.
        trades_stable = candidate_total_trades == 0
    else:
        drift = abs(candidate_total_trades - current_total_trades) / current_total_trades
        trades_stable = drift <= trade_stability_tolerance

    criteria_met = int(pf_improved) + int(dd_improved) + int(trades_stable)
    passed = criteria_met >= 2

    rationale = [
        _fmt_improvement(
            "PF", current_portfolio_pf, candidate_portfolio_pf,
            higher_is_better=True,
        ),
        _fmt_improvement(
            "DD %", current_portfolio_dd, candidate_portfolio_dd,
            higher_is_better=False,
        ),
        (
            f"trades       {'✓' if trades_stable else '✗'}  "
            f"current={current_total_trades} candidate={candidate_total_trades}  "
            f"(tolerance ±{int(trade_stability_tolerance * 100)}%)"
        ),
    ]

    return GateResult(
        passed=passed,
        criteria_met=criteria_met,
        pf_improved=pf_improved,
        dd_improved=dd_improved,
        trades_stable=trades_stable,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# the model bake-off decision gate (spec anchor 5)
# ---------------------------------------------------------------------------

# Minimum DSR for the bake-off result to be considered actionable.
# Below this, the observed Sharpe is statistically consistent with the
# null hypothesis "best of N strategies tried by chance" — no
# architectural conclusion is supported.
PHASE_A_DSR_FLOOR: float = 0.5


def decide_winner_per_symbol(cells: dict) -> dict:
    """Apply the the model bake-off decision gate per spec anchor 5.

    Two-phase decision:

    1. **DSR floor.** If neither tuned variant has DSR >= PHASE_A_DSR_FLOOR
       (0.5), the observed best-of-N Sharpe is consistent with search
       bias and we cannot conclude either architecture is better. Return
       inconclusive + keep LSTM (status quo / production default).

    2. **2-of-3 gate on tuned variants.** Compare lstm_tuned vs
       gbm_tuned across (PF higher, DD lower, stability higher).
       Whichever wins 2 or more takes the symbol. Strict comparisons —
       a tie on any single metric awards 0 to both sides on that metric.

    3. **Tiebreaker.** If neither side wins 2+ (e.g. both win 1 + 1 tie,
       or all 3 are tied), the simplicity tiebreaker selects GBM —
       Occam grounds: faster retrain, SHAP-interpretable, no GPU
       requirement, smaller artifact, lower live-inference latency.

    Args:
        cells: dict with the 4 keys lstm_default / lstm_tuned /
            gbm_default / gbm_tuned, each mapping to a sub-dict with
            keys 'pf' (higher better), 'dd' (lower better), 'stability'
            (higher better), 'dsr' (higher better).

    Returns:
        dict with 'winner' ('lstm' | 'gbm'), 'inconclusive' (bool),
        'reasoning' (str). The orchestrator prints the reasoning into
        the verdict report.
    """
    lstm = cells["lstm_tuned"]
    gbm = cells["gbm_tuned"]

    # Phase 1: DSR floor on tuned variants.
    lstm_clears = lstm["dsr"] >= PHASE_A_DSR_FLOOR
    gbm_clears = gbm["dsr"] >= PHASE_A_DSR_FLOOR

    if not lstm_clears and not gbm_clears:
        return {
            "winner": "lstm",
            "inconclusive": True,
            "reasoning": (
                f"Neither tuned variant clears DSR floor ({PHASE_A_DSR_FLOOR}). "
                f"LSTM={lstm['dsr']:.2f}, GBM={gbm['dsr']:.2f}. "
                f"Result is consistent with best-of-N search bias — no "
                f"architectural conclusion. Keeping LSTM (status quo)."
            ),
        }
    # Asymmetric clear — the side that DOESN'T clear is unfit; the
    # other wins by default regardless of 2-of-3 outcome.
    if lstm_clears and not gbm_clears:
        return {
            "winner": "lstm",
            "inconclusive": False,
            "reasoning": (
                f"Only LSTM clears DSR floor "
                f"(LSTM={lstm['dsr']:.2f} >= {PHASE_A_DSR_FLOOR}, "
                f"GBM={gbm['dsr']:.2f} below). LSTM wins by trustworthiness."
            ),
        }
    if gbm_clears and not lstm_clears:
        return {
            "winner": "gbm",
            "inconclusive": False,
            "reasoning": (
                f"Only GBM clears DSR floor "
                f"(GBM={gbm['dsr']:.2f} >= {PHASE_A_DSR_FLOOR}, "
                f"LSTM={lstm['dsr']:.2f} below). GBM wins by trustworthiness."
            ),
        }

    # Phase 2: 2-of-3 gate (PF higher, DD lower, stability higher).
    gbm_wins_pf = gbm["pf"] > lstm["pf"]
    gbm_wins_dd = gbm["dd"] < lstm["dd"]
    gbm_wins_stab = gbm["stability"] > lstm["stability"]
    gbm_strict = int(gbm_wins_pf) + int(gbm_wins_dd) + int(gbm_wins_stab)

    lstm_wins_pf = lstm["pf"] > gbm["pf"]
    lstm_wins_dd = lstm["dd"] < gbm["dd"]
    lstm_wins_stab = lstm["stability"] > gbm["stability"]
    lstm_strict = int(lstm_wins_pf) + int(lstm_wins_dd) + int(lstm_wins_stab)

    if gbm_strict >= 2:
        return {
            "winner": "gbm",
            "inconclusive": False,
            "reasoning": (
                f"GBM wins {gbm_strict} of 3 metrics (PF: "
                f"{'GBM' if gbm_wins_pf else 'LSTM' if lstm_wins_pf else 'tie'}, "
                f"DD: {'GBM' if gbm_wins_dd else 'LSTM' if lstm_wins_dd else 'tie'}, "
                f"stability: {'GBM' if gbm_wins_stab else 'LSTM' if lstm_wins_stab else 'tie'})."
            ),
        }
    if lstm_strict >= 2:
        return {
            "winner": "lstm",
            "inconclusive": False,
            "reasoning": (
                f"LSTM wins {lstm_strict} of 3 metrics (PF: "
                f"{'GBM' if gbm_wins_pf else 'LSTM' if lstm_wins_pf else 'tie'}, "
                f"DD: {'GBM' if gbm_wins_dd else 'LSTM' if lstm_wins_dd else 'tie'}, "
                f"stability: {'GBM' if gbm_wins_stab else 'LSTM' if lstm_wins_stab else 'tie'})."
            ),
        }

    # Phase 3: tied (1-1, 0-0, etc.) → simplicity tiebreaker → GBM.
    return {
        "winner": "gbm",
        "inconclusive": False,
        "reasoning": (
            f"Tied {lstm_strict}-{gbm_strict} on the 2-of-3 metrics; "
            f"simplicity tiebreaker → GBM (anchor 5: faster retrain, "
            f"SHAP-interpretable, no GPU)."
        ),
    }
