"""Shared paths, constants, and class-map re-exports for the data pipeline.

Importing this module makes ``scripts/class_map.py`` importable so every converter
uses the single source of truth for the unified 6-class scheme.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
DATASETS: Path = PROJECT_ROOT / "datasets"

ACDC_ROOT: Path = DATASETS / "acdc"
BDD_ROOT: Path = DATASETS / "bdd100k"
BDD_IMAGES_ROOT: Path = BDD_ROOT / "bdd100k" / "bdd100k" / "images"
BDD_LABELS_ROOT: Path = BDD_ROOT / "bdd100k_labels_release" / "bdd100k" / "labels"

DATA_YOLO: Path = PROJECT_ROOT / "data" / "yolo"
SPLITS_DIR: Path = PROJECT_ROOT / "splits"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
RESULTS_SPLITS: Path = RESULTS_DIR / "splits"
EXPERIMENTS_DIR: Path = RESULTS_DIR / "experiments"
SUMMARY_DIR: Path = RESULTS_DIR / "summary"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

BDD_SIZE: tuple[int, int] = (1280, 720)
ACDC_WEATHERS: tuple[str, ...] = ("fog", "night", "rain", "snow")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from class_map import (  # noqa: E402
    UNIFIED_CLASSES,
    acdc_category_to_unified,
    bdd_category_to_unified,
)


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


_BDD_IMAGE_INDEX: dict[str, Path] | None = None


def _bdd_index() -> dict[str, Path]:
    """Build (once) a basename -> path index over the BDD image tree.

    BDD detection images are nested under ``100k/{train,val}/.../{trainA,trainB,
    testA,testB}``, so a flat candidate list misses most of them. ``100k`` is
    walked before ``10k`` so the full-resolution set wins on duplicates.
    """
    global _BDD_IMAGE_INDEX
    if _BDD_IMAGE_INDEX is None:
        index: dict[str, Path] = {}
        for split_root in (BDD_IMAGES_ROOT / "100k", BDD_IMAGES_ROOT / "10k"):
            for dirpath, _dirs, files in os.walk(split_root):
                for name in files:
                    if name.endswith(".jpg"):
                        index.setdefault(name, Path(dirpath) / name)
        _BDD_IMAGE_INDEX = index
    return _BDD_IMAGE_INDEX


def bdd_image_path(name: str) -> Path | None:
    """Resolve a BDD image basename to its on-disk path via a cached index."""
    return _bdd_index().get(name)


def yolo_line(class_id: int, x: float, y: float, w: float, h: float, width: int, height: int) -> str | None:
    """Convert an absolute pixel box to a normalized YOLO label line.

    Args:
        class_id: Unified class ID.
        x, y, w, h: Absolute top-left x/y and box width/height in pixels.
        width, height: Image dimensions in pixels.

    Returns:
        A ``"cls cx cy w h"`` string, or ``None`` if the box is degenerate.
    """
    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
        return None
    cx = min(max((x + w / 2.0) / width, 0.0), 1.0)
    cy = min(max((y + h / 2.0) / height, 0.0), 1.0)
    nw = min(max(w / width, 0.0), 1.0)
    nh = min(max(h / height, 0.0), 1.0)
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
