"""
Overfitting diagnostics for backtest runs (A-7).

Two metrics:

- **Deflated Sharpe Ratio (DSR)** — Bailey & López de Prado 2014.
  Given an observed Sharpe, the number of trials that produced it, and the
  shape of the return distribution (skewness + kurtosis), returns the
  probability that the observed Sharpe exceeds what would be expected by
  chance under the null hypothesis of zero skill. A DSR close to 1.0 means
  the Sharpe is very likely real; close to 0 means it is likely an artifact
  of multiple-hypothesis search.

- **Sharpe stability** — proprietary pragmatic metric. Splits the return
  series into ``n_windows`` chronological sub-windows and reports the
  fraction with a positive Sharpe. A stability of 1.0 means every sub-
  period was profitable on its own; 0.5 means half lose money. This is
  *not* canonical PBO (which requires multiple hyperparameter
  configurations) — see CLAUDE.md / SYSTEM_AUDIT.md for when to upgrade.
"""
from __future__ import annotations

import math

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Sharpe helper
# ──────────────────────────────────────────────────────────────────────────

def sharpe_from_returns(returns: np.ndarray) -> float:
    """Annualized-agnostic Sharpe on a 1-D return series.

    Sharpe = mean / std * sqrt(N). Returns 0.0 when std is 0 (flat series)
    or when N < 2.
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    std = float(np.std(r, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(r) / std * math.sqrt(r.size))


# ──────────────────────────────────────────────────────────────────────────
# Deflated Sharpe Ratio
# ──────────────────────────────────────────────────────────────────────────

_EULER_MASCHERONI = 0.5772156649015329


def _expected_max_sharpe_under_null(n_trials: int) -> float:
    """Expected maximum Sharpe under the null hypothesis of zero skill.

    Bailey & López de Prado 2014 approximation:
        E[max_SR] ≈ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))

    where γ is Euler-Mascheroni, Φ⁻¹ is the inverse standard-normal CDF,
    and N is the number of trials. Assumes variance(SR) = 1 which is the
    standard simplification when trial variances aren't tracked.
    """
    if n_trials <= 1:
        return 0.0

    from scipy.stats import norm
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float((1.0 - _EULER_MASCHERONI) * a + _EULER_MASCHERONI * b)


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skewness: float,
    kurtosis: float,
    n_trials: int = 1,
) -> float:
    """Probabilistic Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Args:
        sharpe: Observed (annualized-agnostic) Sharpe ratio.
        n_obs: Number of return observations (e.g. closed trades).
        skewness: Skewness γ₃ of the return distribution.
        kurtosis: Kurtosis γ₄ (not excess kurtosis; N(0,1) has γ₄=3).
        n_trials: Number of strategy variations tried. Higher = more bias
            correction. Default 1 = plain significance test.

    Returns:
        A scalar in [0.0, 1.0] representing P(SR > E[max_SR | null]).
        Above ~0.95 is strongly suggestive of real edge; below ~0.5 is
        consistent with search bias.

    Degenerate cases (too few observations) return 0.0.
    """
    if n_obs < 2:
        return 0.0

    from scipy.stats import norm

    sr_null = _expected_max_sharpe_under_null(n_trials)

    # Standard deviation of the observed Sharpe under the assumed moments:
    #   Var(SR) ≈ (1 - γ₃·SR + (γ₄-1)/4·SR²) / (n - 1)
    var_sr = (
        1.0
        - skewness * sharpe
        + (kurtosis - 1.0) / 4.0 * sharpe * sharpe
    ) / (n_obs - 1)
    # Numerical guard
    if var_sr <= 0.0 or not math.isfinite(var_sr):
        return 0.0

    z = (sharpe - sr_null) / math.sqrt(var_sr)
    return float(norm.cdf(z))


# ──────────────────────────────────────────────────────────────────────────
# Sharpe stability (pragmatic PBO proxy)
# ──────────────────────────────────────────────────────────────────────────

def sharpe_stability(returns: np.ndarray, n_windows: int = 4) -> float:
    """Fraction of non-overlapping chronological sub-windows with Sharpe > 0.

    Returns a value in [0.0, 1.0]:
      - 1.0 → every sub-period made money on its own
      - 0.5 → half the sub-periods lose money
      - 0.0 → every sub-period loses money

    Requires at least 2 × n_windows observations. With fewer, returns 0.0.

    Not canonical PBO (which requires multiple hparam configs + train/test
    splits across them). Intended as a stability check on a single-run
    trade series: does the strategy's edge show up throughout the backtest
    window or is it concentrated in one lucky period?
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 2 * n_windows:
        return 0.0
    chunks = np.array_split(r, n_windows)
    positive = sum(1 for chunk in chunks if sharpe_from_returns(chunk) > 0)
    return float(positive / n_windows)
