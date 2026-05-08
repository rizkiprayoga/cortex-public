"""
model_version_label.py — Build a human-readable version label for the
LSTM model currently loaded for a symbol, used to stamp `trades.model_version`
at trade-open so each live trade can be traced back to its predictor.

Format: ``lstm_<SYMBOL>@<YYYY-MM-DD>`` where the date is the file mtime of
``data/models/lstm_<SYMBOL>.pt`` (UTC). The ``model_versions`` audit table
is the long-term source of truth, but it has historically been empty —
this label is intentionally derivable from on-disk artifacts so backfill
of pre-existing trades works without any DB-side history.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_MODELS_DIR = Path("data/models")


def get_lstm_version_label(
    symbol: str,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Optional[str]:
    """Return ``lstm_<SYMBOL>@<YYYY-MM-DD>`` from the .pt file mtime, or None."""
    pt = models_dir / f"lstm_{symbol}.pt"
    try:
        if not pt.is_file():
            return None
        dt = datetime.fromtimestamp(pt.stat().st_mtime, tz=timezone.utc)
        return f"lstm_{symbol}@{dt.strftime('%Y-%m-%d')}"
    except OSError:
        return None
