"""Convert ACDC COCO detection labels to YOLO format and build an index.

Writes one ``.txt`` per labeled image under ``data/yolo/acdc/labels/<file_name>.txt``
mirroring the original nested structure, and a machine-readable index at
``data/yolo/acdc/index.json`` for split building.

Usage::

    python src/data/convert_acdc.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    ACDC_ROOT,
    ACDC_WEATHERS,
    DATA_YOLO,
    UNIFIED_CLASSES,
    acdc_category_to_unified,
    ensure_dir,
    yolo_line,
)

ACDC_OUT = DATA_YOLO / "acdc"
SPLIT_FILES: tuple[tuple[str, str], ...] = (
    ("train", "train_gt_detection"),
    ("val", "val_gt_detection"),
)


def _label_lines(image: dict, annotations: list[dict]) -> list[str]:
    """Build YOLO label lines for one ACDC image, dropping unmapped classes."""
    width, height = image["width"], image["height"]
    lines: list[str] = []
    for ann in annotations:
        class_id = acdc_category_to_unified(ann["category_id"])
        if class_id is None:
            continue
        x, y, w, h = ann["bbox"]
        line = yolo_line(class_id, x, y, w, h, width, height)
        if line is not None:
            lines.append(line)
    return lines


def convert() -> list[dict]:
    """Convert all labeled ACDC images and return the index records."""
    ensure_dir(ACDC_OUT / "labels")
    index: list[dict] = []

    for weather in ACDC_WEATHERS:
        for split, kind in SPLIT_FILES:
            src = ACDC_ROOT / "labels" / weather / f"instancesonly_{weather}_{kind}.json"
            data = json.loads(src.read_text(encoding="utf-8"))
            prefix = f"{weather}/"
            images = [im for im in data["images"] if im["file_name"].startswith(prefix)]
            by_image: dict[int, list[dict]] = defaultdict(list)
            for ann in data["annotations"]:
                by_image[ann["image_id"]].append(ann)

            kept_instances = 0
            for image in images:
                lines = _label_lines(image, by_image.get(image["id"], []))
                file_name = image["file_name"]
                label_path = (ACDC_OUT / "labels" / file_name).with_suffix(".txt")
                ensure_dir(label_path.parent)
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                kept_instances += len(lines)
                class_counts = Counter(line.split()[0] for line in lines)
                index.append(
                    {
                        "file_name": file_name,
                        "weather": weather,
                        "split": split,
                        "image": str((ACDC_ROOT / "images" / file_name).resolve()),
                        "yolo_image": str(ACDC_OUT / "images" / file_name),
                        "label": str(label_path.relative_to(DATA_YOLO)),
                        "n_instances": len(lines),
                        "class_counts": {UNIFIED_CLASSES[int(k)]: v for k, v in class_counts.items()},
                    }
                )
            print(f"[acdc] {weather}/{split}: {len(images)} images, {kept_instances} kept instances")

    (ACDC_OUT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[acdc] wrote {len(index)} records -> {ACDC_OUT / 'index.json'}")
    return index


if __name__ == "__main__":
    convert()
