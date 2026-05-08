"""
model_head.py — Per-symbol model architecture resolution + guards.

Two orthogonal axes describe a primary model:

  * ``model_kind``: ``"lstm"`` or ``"gbm"`` (the model bake-off, spec §4.1).
    Defaults to ``"lstm"`` (production default).
  * ``model_head``: output-head shape.
    - LSTM: ``"softmax"`` (3-class) or ``"regression"`` (1-output).
    - GBM:  ``"classifier"`` (3-class) — uniform across symbols per
      spec anchor 7.

Only 3 (kind, head) combinations are valid: see ``VALID_KIND_HEAD_COMBINATIONS``.

Both fields live in ``config/settings.yaml`` under
``strategy.per_symbol_params.<SYMBOL>.{model_kind,model_head}`` and are
the version-controlled source of truth. This module exposes the
resolvers, the (kind, head) compatibility check, and a shape-preservation
guard that refuses to silently overwrite an existing model with a
mismatched architecture.

Why a guard: on 2026-04-18 the T-3 dry-run caught that the scheduled
monthly retrain omitted ``--softmax``, so the next May-1 run would have
silently swapped EUR/JPY/ETH from softmax(3) → regression(1). The guard
trips on any such mismatch so the operator decides explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


def resolve_softmax_for_symbol(
    symbol: str,
    cli_override: Optional[str] = None,
    settings_path: Optional[Path] = None,
) -> bool:
    """
    Decide whether ``symbol`` trains with softmax (3-class) or regression
    (1-output). Sources, in priority order:

    1. ``cli_override`` — ``"softmax"`` or ``"regression"`` forces that
       head for the whole run (operator opt-out from config).
    2. ``strategy.per_symbol_params.<symbol>.model_head`` in
       ``settings.yaml``.

    Raises ``ValueError`` if neither source yields a valid decision.
    Never silently defaults.
    """
    if cli_override == "softmax":
        return True
    if cli_override == "regression":
        return False
    if cli_override not in (None, "softmax", "regression"):
        raise ValueError(
            f"cli_override must be 'softmax', 'regression', or None "
            f"(got {cli_override!r})"
        )
    path = settings_path or _SETTINGS_PATH
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = (
        (cfg or {}).get("strategy", {})
        .get("per_symbol_params", {})
        .get(symbol, {})
    )
    head = params.get("model_head")
    if head not in ("softmax", "regression"):
        raise ValueError(
            f"[{symbol}] missing or invalid strategy.per_symbol_params."
            f"{symbol}.model_head in {path} (got {head!r}, expected "
            f"'softmax' or 'regression'). Pass cli_override or add the "
            f"per-symbol config."
        )
    return head == "softmax"


class HeadMismatchError(RuntimeError):
    """Raised when an existing on-disk model's output head disagrees
    with the resolved configuration."""


def assert_head_matches_existing(
    symbol: str,
    want_softmax: bool,
    *,
    allow_change: bool = False,
    models_dir: Optional[Path] = None,
    suffix: str = "",
) -> None:
    """
    Shape-preservation guard. If ``lstm_<symbol>{suffix}.pt`` already
    exists, compare its fc2 output dim to the resolved head. Mismatch +
    no ``allow_change`` → raise ``HeadMismatchError``. First-ever
    training (no existing file) is a no-op.

    ``suffix`` (Task 2.2b-2b) makes the guard path-aware so Phase A
    bake-off artifacts (``lstm_{symbol}_default.pt`` /
    ``lstm_{symbol}_tuned.pt`` / ``lstm_{symbol}_trial_N.pt``) are
    independent of the legacy unsuffixed production file. With the
    default empty suffix the behavior is unchanged.

    Corrupt/unreadable existing models log a warning and proceed — we
    don't want a file-system bit flip to block retraining indefinitely.
    """
    base = models_dir or (_PROJECT_ROOT / "data" / "models")
    model_path = base / f"lstm_{symbol}{suffix}.pt"
    if not model_path.exists():
        return
    try:
        import torch
        # weights_only=True is the safer default (rejects pickled
        # arbitrary objects). We only need fc2.weight.shape[0]; the
        # except below catches any deserialization failure so the guard
        # fails open per its existing contract.
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        sd = state.get("state_dict", state) if isinstance(state, dict) else state
        existing_out = int(sd["fc2.weight"].shape[0])
    except Exception:
        # Let caller log — we don't own logging here.
        return
    existing_softmax = (existing_out == 3)
    if existing_softmax == want_softmax:
        return
    existing_label = "softmax(3)" if existing_softmax else "regression(1)"
    want_label = "softmax(3)" if want_softmax else "regression(1)"
    msg = (
        f"[{symbol}] head mismatch: existing model is {existing_label} "
        f"but resolved config says {want_label}. "
    )
    if allow_change:
        return  # caller may log
    raise HeadMismatchError(
        msg
        + "Refusing to silently change architecture. To intentionally "
        + f"change the head, snapshot current models, delete "
        + f"{model_path.name}, and re-run with allow_change=True."
    )


VALID_KIND_HEAD_COMBINATIONS: frozenset[tuple[str, str]] = frozenset({
    ("lstm", "regression"),
    ("lstm", "softmax"),
    ("gbm", "classifier"),
})


def resolve_model_kind_for_symbol(
    symbol: str,
    settings_path: Optional[Path] = None,
) -> str:
    """
    Resolve ``model_kind`` for a symbol from settings.yaml. Defaults to
    ``"lstm"`` if unset (production default — spec §4.1).

    Returns:
        ``"lstm"`` or ``"gbm"``.

    Raises:
        ``ValueError`` if model_kind is set to an unsupported value.
    """
    path = settings_path or _SETTINGS_PATH
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    params = (
        cfg.get("strategy", {})
        .get("per_symbol_params", {})
        .get(symbol, {})
    )
    kind = params.get("model_kind", "lstm")
    if kind not in ("lstm", "gbm"):
        raise ValueError(
            f"[{symbol}] invalid model_kind {kind!r} in {path}. "
            f"Supported: 'lstm' or 'gbm'."
        )
    return kind


def validate_kind_head(symbol: str, kind: str, head: str) -> None:
    """
    Enforce the (model_kind, model_head) compatibility matrix per spec
    anchor 7. Only 3 combinations are valid:
      - (lstm, regression)
      - (lstm, softmax)
      - (gbm,  classifier)

    Raises ``ValueError`` on any other combination.
    """
    if (kind, head) not in VALID_KIND_HEAD_COMBINATIONS:
        valid = ", ".join(
            f"({k},{h})" for k, h in sorted(VALID_KIND_HEAD_COMBINATIONS)
        )
        raise ValueError(
            f"[{symbol}] invalid (model_kind, model_head) combination: "
            f"({kind}, {head}). Valid combinations: {valid}."
        )
