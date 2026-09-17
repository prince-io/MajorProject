"""Shared helpers for the offline synthesis stages (S2-S6).

The study's fixed comparison uses a deterministic split of the 10,000 BDD source
training images into two halves:

- **A (5,000 clear)** - the clear anchor, referenced from ``bdd_src`` (never copied).
- **B (5,000 clear)** - the synthesis source; a stage transforms each B image in place.

S1 trains on A + B (all clear), so it is the exact control for a stage that trains on
A (clear) + B (transformed). Per-image randomness is seeded from the filename so
generation is order-independent and reproducible (see ``PROJECT.md`` §5).
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA_YOLO, SPLITS_DIR  # noqa: E402  (re-exported for the package)

SEED = 42
CONDITIONS = ("fog", "night", "rain", "snow")
STAGES = ("s2", "s3", "s5", "s6a", "s6b")

A_CLEAR_MANIFEST = SPLITS_DIR / "bdd_src_A_clear.txt"
B_SOURCE_MANIFEST = SPLITS_DIR / "bdd_src_B_source.txt"


def stable_seed(*parts: object) -> int:
    """Return a deterministic 32-bit seed from the string form of ``parts``."""
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def load_manifest(path: Path) -> list[str]:
    """Read a newline-separated manifest of absolute image paths."""
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def source_halves(seed: int = SEED) -> tuple[list[str], list[str]]:
    """Split ``bdd_src_train.txt`` (10k) into sorted A (clear) and B (synth source)."""
    train = load_manifest(SPLITS_DIR / "bdd_src_train.txt")
    order = train[:]
    random.Random(seed).shuffle(order)
    half = len(order) // 2
    return sorted(order[:half]), sorted(order[half:])


def write_halves(a_clear: list[str], b_source: list[str]) -> None:
    """Persist the fixed A/B halves as manifests (idempotent)."""
    A_CLEAR_MANIFEST.write_text("\n".join(a_clear) + "\n", encoding="utf-8")
    B_SOURCE_MANIFEST.write_text("\n".join(b_source) + "\n", encoding="utf-8")


def label_path_for(image: str | Path) -> Path:
    """Map an image path to its YOLO label path (``.../images/x.jpg`` -> ``.../labels/x.txt``)."""
    path = Path(image)
    return path.parent.parent / "labels" / f"{path.stem}.txt"


def dataset_root(dataset_name: str) -> Path:
    """Return the generated-dataset root under ``data/yolo``."""
    return DATA_YOLO / dataset_name
