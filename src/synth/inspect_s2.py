"""Validate and preview a generated S-stage dataset.

Checks (fail the run on violation):

- image count matches ``index.json``
- every degraded label is byte-identical to its source label
- no degraded image is all-black/white (mean in [5, 250], std >= 10)
- no degraded image equals its source

Also renders ``results/summary/figures/synth_preview_<dataset>.png`` (degraded images
with their boxes) and prints a short report.

Usage::

    python src/synth/inspect_s2.py --dataset-name bdd_s2          # full
    python src/synth/inspect_s2.py --dataset-name bdd_s2_smoke    # smoke
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import SUMMARY_DIR, ensure_dir  # noqa: E402
from synth import common as C  # noqa: E402


def _load_records(root: Path) -> list[dict]:
    return json.loads((root / "index.json").read_text(encoding="utf-8"))


def _boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse normalized YOLO boxes from a label file."""
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), *(float(v) for v in parts[1:])))
    return boxes


def _draw(image_bgr: np.ndarray, boxes: list[tuple[int, float, float, float, float]]) -> np.ndarray:
    """Draw normalized boxes on a copy of the image (returns RGB for matplotlib)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).copy()
    height, width = rgb.shape[:2]
    for _cls, cx, cy, bw, bh in boxes:
        x1, y1 = int((cx - bw / 2) * width), int((cy - bh / 2) * height)
        x2, y2 = int((cx + bw / 2) * width), int((cy + bh / 2) * height)
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return rgb


def _preview(records: list[dict], out_png: Path, n: int) -> None:
    """Render a grid of degraded images with boxes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = random.Random(C.SEED).sample(records, min(n, len(records)))
    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows))
    for ax, record in zip(np.ravel(axes), sample):
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        ax.imshow(_draw(image, _boxes(Path(record["label"]))))
        ax.set_title(";".join(record["ops"]), fontsize=7)
        ax.axis("off")
    for ax in np.ravel(axes)[len(sample):]:
        ax.axis("off")
    ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def inspect(dataset_name: str, n: int) -> int:
    """Run all checks; return 0 on success, 1 on failure."""
    root = C.dataset_root(dataset_name)
    if not (root / "index.json").exists():
        print(f"[inspect] no index.json under {root}; run build_dataset.py first")
        return 1
    records = _load_records(root)
    failures: list[str] = []

    images = sorted((root / "images").glob("*.*"))
    if len(images) != len(records):
        failures.append(f"image count {len(images)} != index entries {len(records)}")

    label_mismatch = identical = bad_stats = source_degenerate = 0
    means: list[float] = []
    for record in records:
        if Path(record["label"]).read_bytes() != C.label_path_for(record["source"]).read_bytes():
            label_mismatch += 1
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        if image is None:
            failures.append(f"unreadable image: {record['image']}")
            continue
        mean = float(image.mean())
        means.append(mean)
        source = cv2.imread(record["source"], cv2.IMREAD_COLOR)
        source_ok = (
            source is not None
            and 5.0 <= float(source.mean()) <= 250.0
            and float(source.std()) >= 10.0
        )
        if mean < 5 or mean > 250 or float(image.std()) < 10:
            # A usable source must not be degraded into a degenerate image. Sources that
            # are already near-black/near-constant (rare BDD tunnels/shadow scenes) are
            # exempt: we cannot restore content that is not there.
            if source_ok:
                bad_stats += 1
            else:
                source_degenerate += 1
        if source is not None and source.shape == image.shape and np.array_equal(source, image):
            identical += 1

    if label_mismatch:
        failures.append(f"{label_mismatch} labels differ from source")
    if bad_stats:
        failures.append(f"{bad_stats} images are all-black/white or near-constant")
    if identical:
        failures.append(f"{identical} synthetic images equal their source")

    preview = SUMMARY_DIR / "figures" / f"synth_preview_{dataset_name}.png"
    _preview(records, preview, n)

    print(f"[inspect] {dataset_name}: {len(records)} degraded images, {len(images)} on disk")
    print(f"[inspect] labels byte-identical: {len(records) - label_mismatch}/{len(records)}")
    print(f"[inspect] mean brightness: {np.mean(means):.1f} (min {np.min(means):.1f}, max {np.max(means):.1f})")
    print(f"[inspect] source-degenerate (exempt): {source_degenerate}")
    print(f"[inspect] preview -> {preview}")
    if failures:
        for failure in failures:
            print(f"[inspect] FAIL: {failure}")
        return 1
    print("[inspect] PASS")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inspect a generated S-stage dataset.")
    parser.add_argument("--dataset-name", default="bdd_s2")
    parser.add_argument("--n", type=int, default=20, help="Preview grid size.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(inspect(args.dataset_name, args.n))
