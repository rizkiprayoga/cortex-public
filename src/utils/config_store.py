"""
config_store.py — Atomic YAML Config Read/Write

Provides thread-safe read and write access to the ``risk`` section
of ``config/settings.yaml``.  Writes are atomic: content goes to a
``.tmp`` file first, the current file is backed up to ``.bak``, and
``os.replace`` swaps ``.tmp`` → original in one OS call (atomic on
POSIX; best-effort on Windows).
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ConfigStore:
    """
    Thread-safe accessor for config/settings.yaml risk parameters.

    Usage:
        store = ConfigStore(Path("config/settings.yaml"))
        risk = store.read_risk_section()
        risk["max_daily_loss_soft_pct"] = 2.5
        store.write_risk_section(risk)
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def read_risk_section(self) -> dict[str, Any]:
        """Return a copy of the ``risk:`` section from settings.yaml."""
        with self._lock:
            data = self._load()
        return dict(data.get("risk", {}))

    def write_risk_section(self, risk: dict[str, Any]) -> None:
        """
        Merge *risk* into the full config and atomically rewrite the file.

        Steps:
            1. Load the full YAML (under lock).
            2. Replace ``risk:`` with the provided dict.
            3. Write to ``<path>.tmp``.
            4. Copy current file to ``<path>.bak``.
            5. ``os.replace("<path>.tmp", "<path>")`` — atomic swap.
        """
        with self._lock:
            data = self._load()
            data["risk"] = risk

            tmp = self._path.with_suffix(".tmp")
            bak = self._path.with_suffix(".bak")

            # Write to .tmp
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )

            # Backup current → .bak (best-effort)
            try:
                if self._path.exists():
                    import shutil
                    shutil.copy2(self._path, bak)
            except OSError as exc:
                logger.warning("ConfigStore: backup failed: %s", exc)

            # Atomic swap .tmp → original
            os.replace(str(tmp), str(self._path))
            logger.info("ConfigStore: risk section updated in %s", self._path)

    def _load(self) -> dict:
        """Load and return the full YAML content as a dict."""
        with open(self._path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict at top level in {self._path}")
        return data
