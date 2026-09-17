"""Copy sampled source images and all labeled ACDC images into the YOLO tree.

Images are copied once into ``data/yolo/<dataset>/images/``; split manifests then
reference these copies. Labels were already written by the converters. Never writes
inside ``datasets/``.

Usage::

    python src/data/materialize.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA_YOLO, SPLITS_DIR, ensure_dir  # noqa: E402


def _copy(image: Path, destination: Path) -> bool:
    """Copy an image if the destination does not already exist."""
    if destination.exists():
        return False
    ensure_dir(destination.parent)
    shutil.copy2(image, destination)
    return True


def materialize_source() -> None:
    """Copy the union of BDD source split manifests into the YOLO tree."""
    names: set[Path] = set()
    for manifest in ("bdd_src_train.txt", "bdd_src_val.txt"):
        for line in (SPLITS_DIR / manifest).read_text(encoding="utf-8").splitlines():
            if line.strip():
                names.add(Path(line.strip()))
    out_dir = DATA_YOLO / "bdd_src" / "images"
    copied = sum(_copy(p, out_dir / p.name) for p in sorted(names))
    print(f"[materialize] bdd_src: {copied} copied, {len(names)} total")


def materialize_target() -> None:
    """Copy all labeled ACDC images into the YOLO tree."""
    index = json.loads((DATA_YOLO / "acdc" / "index.json").read_text(encoding="utf-8"))
    out_dir = DATA_YOLO / "acdc" / "images"
    copied = 0
    for record in index:
        image = Path(record["image"])
        copied += _copy(image, out_dir / record["file_name"])
    print(f"[materialize] acdc: {copied} copied, {len(index)} total")


if __name__ == "__main__":
    materialize_source()
    materialize_target()
    print("[materialize] done")
