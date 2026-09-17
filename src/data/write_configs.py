"""Generate Ultralytics dataset YAML configs for every split.

Writes ``configs/<name>.yaml`` referencing absolute manifest paths under ``splits/``.

Usage::

    python src/data/write_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import CONFIGS_DIR, DATA_YOLO, SPLITS_DIR, UNIFIED_CLASSES, ensure_dir  # noqa: E402


def _yaml(name: str, data_root: Path, train: Path, val: Path) -> Path:
    """Write one dataset YAML and return its path."""
    names = "\n".join(f"  {i}: {n}" for i, n in UNIFIED_CLASSES.items())
    content = (
        f"path: {data_root}\n"
        f"train: {train}\n"
        f"val: {val}\n"
        f"nc: {len(UNIFIED_CLASSES)}\n"
        f"names:\n{names}\n"
    )
    path = CONFIGS_DIR / f"{name}.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    """Generate source, ACDC CV, and official configs."""
    ensure_dir(CONFIGS_DIR)
    bdd_root = DATA_YOLO / "bdd_src"
    acdc_root = DATA_YOLO / "acdc"

    written = [
        _yaml("bdd_src", bdd_root, SPLITS_DIR / "bdd_src_train.txt", SPLITS_DIR / "bdd_src_val.txt"),
        _yaml(
            "acdc_official",
            acdc_root,
            SPLITS_DIR / "acdc_official_train.txt",
            SPLITS_DIR / "acdc_official_val.txt",
        ),
    ]
    for k in range(5):
        written.append(
            _yaml(
                f"acdc_cv5_fold{k}",
                acdc_root,
                SPLITS_DIR / "acdc_cv5" / f"fold{k}_train.txt",
                SPLITS_DIR / "acdc_cv5" / f"fold{k}_test.txt",
            )
        )
    print(f"[configs] wrote {len(written)} configs -> {CONFIGS_DIR}")


if __name__ == "__main__":
    main()
