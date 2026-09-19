"""Validate and preview the S3 weather-synthesis dataset.

Hard checks (fail the run):

- image count matches ``index.json``
- every transformed label is byte-identical to its source label
- no transformed image is all-black/white/near-constant (unless the source was)
- no transformed image equals its source

Reported diagnostics (do not fail):

- per-condition counts (expect balanced 1,250 each on the full build)
- object visibility: distribution of GT-box local contrast after/before (weather
  legitimately occludes, so this is reported as a statistic, not a gate)
- per-condition preview grids for the paper

Usage::

    python src/synth/inspect_s3.py --dataset-name bdd_s3
    python src/synth/inspect_s3.py --dataset-name bdd_s3_smoke
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import SUMMARY_DIR, ensure_dir  # noqa: E402
from synth import common as C  # noqa: E402
from synth import stage_common  # noqa: E402

VISIBILITY_WARN_RATIO = 0.3
MIN_BOX_PIXELS = 4
MIN_SOURCE_STD = 1.0


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
    """Render a grid of transformed images with boxes."""
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
        ax.set_title(record["condition"], fontsize=9)
        ax.axis("off")
    for ax in np.ravel(axes)[len(sample):]:
        ax.axis("off")
    ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _visibility(records: list[dict]) -> dict[str, dict]:
    """Per-condition GT-box local-contrast retention (after / before)."""
    ratios: dict[str, list[float]] = defaultdict(list)
    for record in records:
        source = cv2.imread(record["source"], cv2.IMREAD_COLOR)
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        if source is None or image is None:
            continue
        gray_before = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray_before.shape[:2]
        for _cls, cx, cy, bw, bh in _boxes(Path(record["label"])):
            x1 = max(0, int((cx - bw / 2) * width))
            y1 = max(0, int((cy - bh / 2) * height))
            x2 = min(width, int((cx + bw / 2) * width))
            y2 = min(height, int((cy + bh / 2) * height))
            if (x2 - x1) * (y2 - y1) < MIN_BOX_PIXELS:
                continue
            before = float(gray_before[y1:y2, x1:x2].std())
            if before < MIN_SOURCE_STD:
                continue
            after = float(gray_after[y1:y2, x1:x2].std())
            ratios[record["condition"]].append(after / before)

    summary: dict[str, dict] = {}
    for condition, values in ratios.items():
        array = np.asarray(values, dtype=np.float32)
        summary[condition] = {
            "boxes": int(array.size),
            "frac_below_0.3": float(np.mean(array < VISIBILITY_WARN_RATIO)),
            "p05": float(np.percentile(array, 5)),
            "median": float(np.median(array)),
        }
    return summary


def inspect(dataset_name: str, n: int) -> int:
    """Run all checks; return 0 on success, 1 on failure."""
    root = C.dataset_root(dataset_name)
    if not (root / "index.json").exists():
        print(f"[inspect] no index.json under {root}; run build_s3.py first")
        return 1
    records = _load_records(root)
    failures: list[str] = []

    images = sorted((root / "images").glob("*.*"))
    if len(images) != len(records):
        failures.append(f"image count {len(images)} != index entries {len(records)}")

    counts = Counter(record["condition"] for record in records)
    label_mismatch = identical = bad_stats = source_degenerate = 0
    means: list[float] = []
    for record in records:
        if Path(record["label"]).read_bytes() != C.label_path_for(record["source"]).read_bytes():
            label_mismatch += 1
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        if image is None:
            failures.append(f"unreadable image: {record['image']}")
            continue
        means.append(float(image.mean()))
        source = cv2.imread(record["source"], cv2.IMREAD_COLOR)
        source_ok = source is not None and stage_common.source_usable(source)
        if not stage_common.stat_ok(image):
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

    figures = ensure_dir(SUMMARY_DIR / "figures")
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_condition[record["condition"]].append(record)
    previews: list[Path] = []
    for condition, members in sorted(by_condition.items()):
        out_png = figures / f"synth_preview_{dataset_name}_{condition}.png"
        _preview(members, out_png, n)
        previews.append(out_png)

    visibility = _visibility(records)

    report_lines = [
        f"dataset: {dataset_name}",
        f"images on disk: {len(images)}",
        f"index entries: {len(records)}",
        f"labels byte-identical: {len(records) - label_mismatch}/{len(records)}",
        f"mean brightness: {np.mean(means):.1f} (min {np.min(means):.1f}, max {np.max(means):.1f})",
        f"source-degenerate (exempt): {source_degenerate}",
        f"conditions: {dict(sorted(counts.items()))}",
        "object visibility (GT-box local contrast, after/before):",
    ]
    for condition in sorted(visibility):
        stats = visibility[condition]
        report_lines.append(
            f"  {condition:<6} boxes={stats['boxes']:>6} "
            f"frac<{VISIBILITY_WARN_RATIO:.1f}={stats['frac_below_0.3']:.3f} "
            f"p05={stats['p05']:.3f} median={stats['median']:.3f}"
        )
    report_lines.append("previews:")
    report_lines.extend(f"  {path}" for path in previews)
    report = "\n".join(report_lines) + "\n"
    report_path = SUMMARY_DIR / f"synth_report_{dataset_name}.txt"
    report_path.write_text(report, encoding="utf-8")

    print(f"[inspect] {dataset_name}: {len(records)} transformed images, {len(images)} on disk")
    print(f"[inspect] conditions: {dict(sorted(counts.items()))}")
    print(f"[inspect] labels byte-identical: {len(records) - label_mismatch}/{len(records)}")
    print(f"[inspect] mean brightness: {np.mean(means):.1f} (min {np.min(means):.1f}, max {np.max(means):.1f})")
    print(f"[inspect] source-degenerate (exempt): {source_degenerate}")
    for condition in sorted(visibility):
        stats = visibility[condition]
        print(
            f"[inspect]   {condition:<6} boxes={stats['boxes']} "
            f"frac<0.3={stats['frac_below_0.3']:.3f} p05={stats['p05']:.3f} median={stats['median']:.3f}"
        )
    print(f"[inspect] report -> {report_path}")
    if failures:
        for failure in failures:
            print(f"[inspect] FAIL: {failure}")
        return 1
    print("[inspect] PASS")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inspect the S3 weather-synthesis dataset.")
    parser.add_argument("--dataset-name", default="bdd_s3")
    parser.add_argument("--n", type=int, default=16, help="Preview grid size per condition.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(inspect(args.dataset_name, args.n))
