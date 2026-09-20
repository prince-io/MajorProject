"""Validate and preview the S5 appearance-calibrated dataset.

Hard checks (fail the run):

- image count matches ``index.json``
- every transformed label is byte-identical to its source
- no transformed image is all-black/white/near-constant (unless the source was)
- no transformed image equals its source
- condition assignment matches S3 (``splits/bdd_s3_conditions.csv``)

Reported diagnostics (do not fail):

- **closed-loop appearance**: per-channel mean/std and mean saturation of the generated
  half vs the calibrated target medians (the S5 claim)
- object visibility (GT-box local-contrast retention)
- per-condition counts and preview grids

Usage::

    python src/synth/inspect_s5.py --dataset-name bdd_s5
"""

from __future__ import annotations

import argparse
import csv
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
from synth import physics  # noqa: E402
from synth import stage_common  # noqa: E402

VISIBILITY_WARN_RATIO = 0.3
MIN_BOX_PIXELS = 4
MIN_SOURCE_STD = 1.0


def _load_records(root: Path) -> list[dict]:
    return json.loads((root / "index.json").read_text(encoding="utf-8"))


def _boxes(label_path: Path):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), *(float(v) for v in parts[1:])))
    return boxes


def _draw(image_bgr, boxes):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).copy()
    height, width = rgb.shape[:2]
    for _cls, cx, cy, bw, bh in boxes:
        x1, y1 = int((cx - bw / 2) * width), int((cy - bh / 2) * height)
        x2, y2 = int((cx + bw / 2) * width), int((cy + bh / 2) * height)
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return rgb


def _preview(records, out_png: Path, n: int) -> None:
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
        ax.set_title(record["condition"], fontsize=8)
        ax.axis("off")
    for ax in np.ravel(axes)[len(sample):]:
        ax.axis("off")
    ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _appearance_check(records) -> tuple[dict, dict]:
    """Generated per-condition appearance medians vs the calibrated targets."""
    gen: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        stats = physics_appearance(image)
        for key, value in stats.items():
            gen[record["condition"]][key].append(value)
    observed = {cond: {key: float(np.median(values)) for key, values in block.items()} for cond, block in gen.items()}
    target = physics.load_stats()["conditions"]
    return observed, target


def physics_appearance(img: np.ndarray) -> dict:
    means = [float(img[..., c].mean()) for c in range(3)]
    stds = [float(img[..., c].astype(np.float32).std()) for c in range(3)]
    return {
        "mean_b": means[0], "mean_g": means[1], "mean_r": means[2],
        "std_b": stds[0], "std_g": stds[1], "std_r": stds[2],
        "saturation": float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean()),
    }


def _visibility(records) -> dict[str, dict]:
    ratios: dict[str, list[float]] = defaultdict(list)
    for record in records:
        source = cv2.imread(record["source"], cv2.IMREAD_COLOR)
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        if source is None or image is None:
            continue
        before = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        after = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = before.shape[:2]
        for _cls, cx, cy, bw, bh in _boxes(Path(record["label"])):
            x1 = max(0, int((cx - bw / 2) * width)); y1 = max(0, int((cy - bh / 2) * height))
            x2 = min(width, int((cx + bw / 2) * width)); y2 = min(height, int((cy + bh / 2) * height))
            if (x2 - x1) * (y2 - y1) < MIN_BOX_PIXELS:
                continue
            base = float(before[y1:y2, x1:x2].std())
            if base < MIN_SOURCE_STD:
                continue
            ratios[record["condition"]].append(float(after[y1:y2, x1:x2].std()) / base)
    return {
        condition: {"boxes": len(values), "frac_below_0.3": float(np.mean(np.asarray(values) < VISIBILITY_WARN_RATIO)),
                    "p05": float(np.percentile(values, 5)), "median": float(np.median(values))}
        for condition, values in ratios.items()
    }


def inspect(dataset_name: str, n: int) -> int:
    root = C.dataset_root(dataset_name)
    if not (root / "index.json").exists():
        print(f"[inspect] no index.json under {root}; run build_s5.py first")
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

    # condition parity with S3 (full build only; smoke uses a different balanced subset)
    s3_path = C.SPLITS_DIR / "bdd_s3_conditions.csv"
    if dataset_name == "bdd_s5" and s3_path.exists():
        s3 = {row["source"]: row["condition"] for row in csv.DictReader(s3_path.open())}
        mismatch = sum(1 for record in records if s3.get(record["source"]) != record["condition"])
        if mismatch:
            failures.append(f"{mismatch} condition assignments differ from S3")

    observed, target = _appearance_check(records)
    figures = ensure_dir(SUMMARY_DIR / "figures")
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_condition[record["condition"]].append(record)
    previews = []
    for condition, members in sorted(by_condition.items()):
        out_png = figures / f"synth_preview_{dataset_name}_{condition}.png"
        _preview(members, out_png, n)
        previews.append(out_png)
    visibility = _visibility(records)

    lines = [
        f"dataset: {dataset_name}",
        f"images on disk: {len(images)}  index entries: {len(records)}",
        f"labels byte-identical: {len(records) - label_mismatch}/{len(records)}",
        f"mean brightness: {np.mean(means):.1f} (min {np.min(means):.1f}, max {np.max(means):.1f})",
        f"conditions: {dict(sorted(counts.items()))}",
        "closed-loop appearance (observed median vs target median):",
    ]
    for condition in sorted(observed):
        for field in physics_stats_fields():
            obs = observed[condition].get(field)
            tgt = target.get(condition, {}).get(field, {}).get("median")
            if obs is not None and tgt is not None:
                lines.append(f"  {condition:<6} {field:10} observed={obs:7.2f} target={tgt:7.2f} diff={obs - tgt:+.2f}")
    lines.append("object visibility (GT-box local contrast, after/before):")
    for condition in sorted(visibility):
        v = visibility[condition]
        lines.append(f"  {condition:<6} boxes={v['boxes']} frac<0.3={v['frac_below_0.3']:.3f} p05={v['p05']:.3f} median={v['median']:.3f}")
    lines.append("previews:")
    lines.extend(f"  {p}" for p in previews)
    report = "\n".join(lines) + "\n"
    report_path = SUMMARY_DIR / f"synth_report_{dataset_name}.txt"
    report_path.write_text(report, encoding="utf-8")

    print(f"[inspect] {dataset_name}: {len(records)} transformed, {len(images)} on disk")
    print(f"[inspect] labels byte-identical: {len(records) - label_mismatch}/{len(records)}")
    print(f"[inspect] conditions: {dict(sorted(counts.items()))}")
    print(f"[inspect] report -> {report_path}")
    if failures:
        for failure in failures:
            print(f"[inspect] FAIL: {failure}")
        return 1
    print("[inspect] PASS")
    return 0


def physics_stats_fields():
    return ("mean_b", "mean_g", "mean_r", "std_b", "std_g", "std_r", "saturation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the S5 appearance-calibrated dataset.")
    parser.add_argument("--dataset-name", default="bdd_s5")
    parser.add_argument("--n", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(inspect(args.dataset_name, args.n))
