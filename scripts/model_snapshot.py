"""
model_snapshot.py — Backup, list, and restore trained model files.

Provides safety for retraining experiments — snapshot working models
before training new ones, restore if the new models perform worse.

Usage:
    # Snapshot current models with a label
    python scripts/model_snapshot.py save phase-a-baseline \\
        --note "PF avg 2.13 — Triple Barrier + risk 1.5% + Euphoria guards"

    # List all snapshots
    python scripts/model_snapshot.py list

    # Restore a snapshot (with safety prompt)
    python scripts/model_snapshot.py restore phase-a-baseline

    # Show what's in a snapshot
    python scripts/model_snapshot.py show phase-a-baseline

    # Delete an old snapshot
    python scripts/model_snapshot.py delete phase-a-baseline

Snapshot directory structure:
    data/model_snapshots/
        phase-a-baseline/
            metadata.json    — label, timestamp, note, optional metrics
            hmm_*.pkl        — all HMM model files
            lstm_*.pt        — all LSTM weight files
            lstm_scaler_*.pkl — all scaler files
            lstm_*.pca.pkl   — all PCA files (if any)

Snapshots are immutable — re-saving the same label requires explicit
--force or delete-then-save.
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")
SNAPSHOTS_DIR = Path("data/model_snapshots")

# Patterns of files to snapshot (relative to MODELS_DIR)
MODEL_FILE_PATTERNS = [
    "hmm_*.pkl",
    "lstm_*.pt",
    "lstm_scaler_*.pkl",
    "lstm_*.pca.pkl",
]


def _validate_label(label: str) -> None:
    """Allowlist label format — alphanumeric, dash, underscore only."""
    if not label or len(label) > 64:
        raise ValueError("Label must be 1-64 characters")
    for ch in label:
        if not (ch.isalnum() or ch in "-_"):
            raise ValueError(
                f"Label may contain only [a-zA-Z0-9_-]; got '{ch}'"
            )


def _list_model_files() -> list[Path]:
    """Return all model files matching MODEL_FILE_PATTERNS."""
    files: list[Path] = []
    if not MODELS_DIR.exists():
        return files
    for pattern in MODEL_FILE_PATTERNS:
        files.extend(MODELS_DIR.glob(pattern))
    return sorted(files)


def cmd_save(label: str, note: str = "", force: bool = False) -> int:
    _validate_label(label)
    snap_dir = SNAPSHOTS_DIR / label
    if snap_dir.exists() and not force:
        logger.error(
            "Snapshot '%s' already exists. Use --force to overwrite "
            "or delete it first.", label,
        )
        return 1

    files = _list_model_files()
    if not files:
        logger.error("No model files found in %s", MODELS_DIR)
        return 1

    snap_dir.mkdir(parents=True, exist_ok=True)
    # Clear if force-overwriting
    if force:
        for old_file in snap_dir.iterdir():
            if old_file.is_file():
                old_file.unlink()

    copied = 0
    for src in files:
        dst = snap_dir / src.name
        shutil.copy2(src, dst)
        copied += 1

    metadata = {
        "label": label,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "note": note,
        "file_count": copied,
        "files": [f.name for f in files],
    }
    (snap_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    logger.info(
        "Snapshot '%s' saved: %d files (%.1f MB) → %s",
        label, copied,
        sum(f.stat().st_size for f in snap_dir.iterdir() if f.is_file()) / 1e6,
        snap_dir,
    )
    if note:
        logger.info("  Note: %s", note)
    return 0


def cmd_list() -> int:
    if not SNAPSHOTS_DIR.exists():
        print("No snapshots found.")
        print(f"Snapshots dir: {SNAPSHOTS_DIR}")
        return 0

    snapshots = sorted([d for d in SNAPSHOTS_DIR.iterdir() if d.is_dir()])
    if not snapshots:
        print("No snapshots found.")
        return 0

    print(f"\n{'LABEL':<30} {'CREATED':<22} {'FILES':>6}  NOTE")
    print("-" * 100)
    for snap_dir in snapshots:
        metadata_path = snap_dir / "metadata.json"
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                created = meta.get("created_at", "?")[:19]
                count = meta.get("file_count", 0)
                note = (meta.get("note") or "")[:50]
                print(f"{meta.get('label', snap_dir.name):<30} "
                       f"{created:<22} {count:>6}  {note}")
            except Exception:
                print(f"{snap_dir.name:<30} (corrupt metadata)")
        else:
            files = list(snap_dir.glob("*"))
            print(f"{snap_dir.name:<30} (no metadata, {len(files)} files)")
    return 0


def cmd_show(label: str) -> int:
    _validate_label(label)
    snap_dir = SNAPSHOTS_DIR / label
    if not snap_dir.exists():
        logger.error("Snapshot '%s' not found", label)
        return 1
    metadata_path = snap_dir / "metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(json.dumps(meta, indent=2))
    else:
        print(f"No metadata for snapshot '{label}'")
    print("\nFiles:")
    for f in sorted(snap_dir.iterdir()):
        if f.is_file() and f.name != "metadata.json":
            print(f"  {f.name:<35} {f.stat().st_size / 1e3:>10.1f} KB")
    return 0


def cmd_restore(label: str, no_prompt: bool = False) -> int:
    _validate_label(label)
    snap_dir = SNAPSHOTS_DIR / label
    if not snap_dir.exists():
        logger.error("Snapshot '%s' not found", label)
        return 1

    metadata_path = snap_dir / "metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"\nRestoring snapshot: {label}")
        print(f"  Created: {meta.get('created_at')}")
        print(f"  Files:   {meta.get('file_count')}")
        if meta.get("note"):
            print(f"  Note:    {meta['note']}")

    if not no_prompt:
        print(f"\nThis will OVERWRITE current models in {MODELS_DIR}")
        # Auto-backup current state before restore
        auto_label = f"auto-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"Current models will be backed up to: {auto_label}")
        response = input("Proceed? [y/N]: ").strip().lower()
        if response != "y":
            print("Restore cancelled")
            return 0
        # Auto-backup before restore
        cmd_save(auto_label, note=f"Auto-backup before restoring '{label}'")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    restored = 0
    for src in snap_dir.iterdir():
        if src.is_file() and src.name != "metadata.json":
            dst = MODELS_DIR / src.name
            shutil.copy2(src, dst)
            restored += 1

    logger.info("Restored %d files from snapshot '%s'", restored, label)
    return 0


def cmd_delete(label: str, no_prompt: bool = False) -> int:
    _validate_label(label)
    snap_dir = SNAPSHOTS_DIR / label
    if not snap_dir.exists():
        logger.error("Snapshot '%s' not found", label)
        return 1

    if not no_prompt:
        response = input(f"Delete snapshot '{label}'? [y/N]: ").strip().lower()
        if response != "y":
            print("Delete cancelled")
            return 0

    shutil.rmtree(snap_dir)
    logger.info("Snapshot '%s' deleted", label)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Model snapshot tool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_save = sub.add_parser("save", help="Save current models as snapshot")
    p_save.add_argument("label", help="Snapshot label (alphanumeric, -, _)")
    p_save.add_argument("--note", default="", help="Optional note")
    p_save.add_argument("--force", action="store_true",
                        help="Overwrite existing snapshot with same label")

    sub.add_parser("list", help="List all snapshots")

    p_show = sub.add_parser("show", help="Show snapshot metadata")
    p_show.add_argument("label")

    p_restore = sub.add_parser("restore", help="Restore a snapshot")
    p_restore.add_argument("label")
    p_restore.add_argument("--yes", action="store_true",
                            help="Skip confirmation prompt")

    p_delete = sub.add_parser("delete", help="Delete a snapshot")
    p_delete.add_argument("label")
    p_delete.add_argument("--yes", action="store_true",
                           help="Skip confirmation prompt")

    args = parser.parse_args()

    try:
        if args.cmd == "save":
            return cmd_save(args.label, args.note, args.force)
        elif args.cmd == "list":
            return cmd_list()
        elif args.cmd == "show":
            return cmd_show(args.label)
        elif args.cmd == "restore":
            return cmd_restore(args.label, args.yes)
        elif args.cmd == "delete":
            return cmd_delete(args.label, args.yes)
    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
