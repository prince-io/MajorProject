"""Convert BDD100K clear/daytime detection labels to YOLO format and build an index.

Writes one ``.txt`` per source image under ``data/yolo/bdd_src/labels/<name>.txt``
and a machine-readable index at ``data/yolo/bdd_src/index.json``.

Source definition (locked): ``weather == "clear"`` and ``timeofday == "daytime"``.

Usage::

    python src/data/convert_bdd.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    BDD_LABELS_ROOT,
    BDD_SIZE,
    DATA_YOLO,
    UNIFIED_CLASSES,
    bdd_category_to_unified,
    bdd_image_path,
    ensure_dir,
    yolo_line,
)

BDD_OUT = DATA_YOLO / "bdd_src"
TRAIN_JSON = BDD_LABELS_ROOT / "bdd100k_labels_images_train.json"
RARE_CLASSES: dict[int, str] = {4: "bus", 5: "bicycle", 1: "rider"}


def _label_lines(record: dict) -> list[str]:
    """Build YOLO label lines for one BDD record, dropping unmapped categories."""
    width, height = BDD_SIZE
    lines: list[str] = []
    for label in record.get("labels", []):
        box = label.get("box2d")
        if box is None:
            continue
        class_id = bdd_category_to_unified(label.get("category"))
        if class_id is None:
            continue
        w = box["x2"] - box["x1"]
        h = box["y2"] - box["y1"]
        line = yolo_line(class_id, box["x1"], box["y1"], w, h, width, height)
        if line is not None:
            lines.append(line)
    return lines


def convert() -> list[dict]:
    """Convert all clear/daytime BDD source images and return the index records."""
    ensure_dir(BDD_OUT / "labels")
    records = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    source = [
        r
        for r in records
        if r["attributes"].get("weather") == "clear"
        and r["attributes"].get("timeofday") == "daytime"
    ]
    print(f"[bdd] clear/daytime candidates: {len(source)}")

    index: list[dict] = []
    missing = 0
    for record in source:
        name = record["name"]
        image_path = bdd_image_path(name)
        if image_path is None:
            missing += 1
            print(f"[bdd][MISSING] {name}")
            continue
        lines = _label_lines(record)
        label_path = (BDD_OUT / "labels" / name).with_suffix(".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        class_counts = Counter(line.split()[0] for line in lines)
        present = {int(k) for k in class_counts}
        rare = sorted(RARE_CLASSES[c] for c in present if c in RARE_CLASSES)
        index.append(
            {
                "name": name,
                "image": str(image_path.resolve()),
                "yolo_image": str(BDD_OUT / "images" / name),
                "label": str(label_path.relative_to(DATA_YOLO)),
                "scene": record["attributes"].get("scene"),
                "n_instances": len(lines),
                "rare_present": rare,
                "class_counts": {UNIFIED_CLASSES[int(k)]: v for k, v in class_counts.items()},
            }
        )

    (BDD_OUT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    total_instances = sum(r["n_instances"] for r in index)
    print(
        f"[bdd] wrote {len(index)} records ({total_instances} instances), "
        f"{missing} missing images -> {BDD_OUT / 'index.json'}"
    )
    return index


if __name__ == "__main__":
    convert()
