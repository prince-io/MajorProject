"""Remove label files that have no matching image in the YOLO working tree.

Keeps ``data/yolo/<dataset>/labels/`` in exact correspondence with
``data/yolo/<dataset>/images/``. Only the working tree is touched; ``datasets/``
is never modified.

Usage::

    python src/data/prune_labels.py            # delete orphans
    python src/data/prune_labels.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA_YOLO  # noqa: E402

DATASETS: tuple[str, ...] = ("bdd_src", "acdc")
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png")


def _image_stems(dataset: Path) -> set[Path]:
    """Return image stems relative to the dataset's ``images/`` directory."""
    root = dataset / "images"
    return {
        p.relative_to(root).with_suffix("")
        for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES
    }


def prune(dry_run: bool) -> int:
    """Delete orphan labels across all datasets; return total removed."""
    removed_total = 0
    for name in DATASETS:
        dataset = DATA_YOLO / name
        if not dataset.exists():
            print(f"[prune] {name}: not found, skipping")
            continue
        image_stems = _image_stems(dataset)
        labels_root = dataset / "labels"
        orphans = [
            p
            for p in labels_root.rglob("*.txt")
            if p.relative_to(labels_root).with_suffix("") not in image_stems
        ]
        for path in orphans:
            if not dry_run:
                path.unlink()
        removed_total += len(orphans)
        action = "would remove" if dry_run else "removed"
        print(
            f"[prune] {name}: {action} {len(orphans)} orphan labels "
            f"({len(image_stems)} images kept)"
        )
    return removed_total


def main() -> None:
    """Parse args and run the prune."""
    parser = argparse.ArgumentParser(description="Prune orphan YOLO labels.")
    parser.add_argument("--dry-run", action="store_true", help="Report without deleting.")
    args = parser.parse_args()
    total = prune(args.dry_run)
    verb = "would be removed" if args.dry_run else "removed"
    print(f"[prune] total {verb}: {total}")


if __name__ == "__main__":
    main()
