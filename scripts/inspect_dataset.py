"""Inspect ACDC and BDD100K on disk and preview the unified class mapping.

This is a read-only, sample-based ground-truth check. It opens the datasets,
samples a few real records, runs them through :mod:`class_map`, and prints what
a conversion would produce. No images are loaded or drawn, and nothing is written
to disk except the text report under ``scripts/_reports/``.

Run with::

    python scripts/inspect_dataset.py
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from class_map import (
    UNIFIED_CLASSES,
    acdc_category_to_unified,
    bdd_category_to_unified,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACDC_ROOT = PROJECT_ROOT / "datasets" / "acdc"
BDD_ROOT = PROJECT_ROOT / "datasets" / "bdd100k"
BDD_IMAGES_ROOT = BDD_ROOT / "bdd100k" / "bdd100k" / "images"
BDD_LABELS_ROOT = BDD_ROOT / "bdd100k_labels_release" / "bdd100k" / "labels"
REPORT_PATH = Path(__file__).resolve().parent / "_reports" / "inspect_report.txt"

ACDC_WEATHERS: tuple[str, ...] = ("fog", "night", "rain", "snow")
SAMPLES: int = 3

_LINES: list[str] = []


def out(line: str = "") -> None:
    """Print a line and buffer it for the written report."""
    print(line)
    _LINES.append(line)


def header(title: str) -> None:
    """Print a prominent section header."""
    out()
    out("=" * 78)
    out(title)
    out("=" * 78)


def _load_json(path: Path) -> Any:
    """Load a JSON file, logging loudly and returning ``None`` on failure."""
    if not path.exists():
        out(f"[MISSING] {path}")
        return None
    out(f"[load] {path} ({path.stat().st_size / 1_048_576:.1f} MiB)")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _distribution(classes: Iterable[int | None]) -> Counter[str]:
    """Count unified class names, ignoring dropped (``None``) entries."""
    counter: Counter[str] = Counter()
    for class_id in classes:
        if class_id is not None:
            counter[UNIFIED_CLASSES[class_id]] += 1
    return counter


def _print_distribution(counter: Counter[str]) -> None:
    """Print a class distribution sorted by unified class ID."""
    if not counter:
        out("  (no mapped annotations)")
        return
    total = sum(counter.values())
    for class_id, name in UNIFIED_CLASSES.items():
        count = counter.get(name, 0)
        share = (count / total * 100.0) if total else 0.0
        out(f"  {class_id} {name:<8} {count:>8}  ({share:5.1f}%)")
    out(f"  {'TOTAL':<10} {total:>8}")


def inspect_acdc() -> None:
    """Inspect ACDC train COCO JSONs and preview the unified mapping."""
    header("ACDC -- per-weather train inspection")
    for weather in ACDC_WEATHERS:
        path = (
            ACDC_ROOT
            / "labels"
            / weather
            / f"instancesonly_{weather}_train_gt_detection.json"
        )
        out()
        out(f"--- ACDC weather: {weather} ---")
        data = _load_json(path)
        if data is None:
            continue

        raw_images = data.get("images", [])
        prefix = f"{weather}/"
        images = [im for im in raw_images if im["file_name"].startswith(prefix)]
        image_ids = {im["id"] for im in images}
        annotations = [a for a in data.get("annotations", []) if a["image_id"] in image_ids]
        categories = data.get("categories", [])
        id_to_name = {c["id"]: c["name"] for c in categories}

        if len(images) != len(raw_images):
            out(
                f"  [NOTE] file holds {len(raw_images)} images across multiple "
                f"weathers; filtered to {len(images)} '{weather}' images."
            )
        out(f"  images (weather subset): {len(images)}")
        out(f"  annotations (weather subset): {len(annotations)}")
        out(f"  categories: {len(categories)}")
        out(f"  raw categories: {[(c['id'], c['name']) for c in categories]}")

        out(f"  sample annotations (seed=42):")
        if annotations:
            for ann in random.sample(annotations, min(SAMPLES, len(annotations))):
                cat_id = ann["category_id"]
                mapped = acdc_category_to_unified(cat_id)
                rendered = (
                    "DROPPED" if mapped is None else f"{mapped} ({UNIFIED_CLASSES[mapped]})"
                )
                out(
                    f"    category_id={cat_id} ({id_to_name.get(cat_id, '?')}) "
                    f"-> {rendered}; bbox={ann.get('bbox')}"
                )
        else:
            out("    (no annotations)")

        kept = sum(1 for a in annotations if acdc_category_to_unified(a["category_id"]) is not None)
        dropped = len(annotations) - kept
        out(f"  kept: {kept}  dropped: {dropped}")
        out("  distribution after mapping:")
        _print_distribution(
            _distribution(acdc_category_to_unified(a["category_id"]) for a in annotations)
        )


def inspect_bdd() -> None:
    """Inspect BDD100K train labels and preview the unified mapping."""
    header("BDD100K -- train label inspection")
    path = BDD_LABELS_ROOT / "bdd100k_labels_images_train.json"
    data = _load_json(path)
    if data is None:
        return

    out()
    out(f"  total records: {len(data)}")
    attribute_keys = sorted(data[0].get("attributes", {}).keys()) if data else []
    out(f"  attributes keys: {attribute_keys}")
    out(f"  'weather' present: {'weather' in attribute_keys}")

    weather_counts = Counter(rec.get("attributes", {}).get("weather") for rec in data)
    out("  records per weather:")
    for weather, count in weather_counts.most_common():
        out(f"    {str(weather):<16} {count}")

    clear = [rec for rec in data if rec.get("attributes", {}).get("weather") == "clear"]
    out(f"  clear-weather records: {len(clear)}")

    out()
    out(f"  sample clear-weather records (seed=42):")
    for rec in random.sample(clear, min(SAMPLES, len(clear))):
        labels = rec.get("labels", [])
        out(f"    {rec['name']}: {len(labels)} labels")
        for label in labels:
            category = label.get("category")
            mapped = bdd_category_to_unified(category)
            rendered = (
                "DROPPED" if mapped is None else f"{mapped} ({UNIFIED_CLASSES[mapped]})"
            )
            geometry = "box2d" if "box2d" in label else ("poly2d" if "poly2d" in label else "none")
            out(f"      {category:<16} geometry={geometry:<7} -> {rendered}")

    out()
    out("  distribution after mapping (clear-weather records only):")
    _print_distribution(
        _distribution(
            bdd_category_to_unified(label.get("category"))
            for rec in clear
            for label in rec.get("labels", [])
        )
    )


_BDD_INDEX: dict[str, str] | None = None


def _bdd_index() -> dict[str, str]:
    """Build (once) a basename -> relative path index of BDD100K images."""
    global _BDD_INDEX
    if _BDD_INDEX is None:
        index: dict[str, str] = {}
        for directory, _subdirs, files in os.walk(BDD_IMAGES_ROOT):
            for name in files:
                if name.endswith(".jpg"):
                    rel = Path(directory, name).relative_to(BDD_IMAGES_ROOT)
                    index[name] = str(rel)
        _BDD_INDEX = index
    return _BDD_INDEX


def _resolve_bdd(name: str) -> tuple[Path | None, str]:
    """Resolve a BDD image name via ordered candidates, then a recursive index."""
    candidates: list[tuple[Path, str]] = [
        (BDD_IMAGES_ROOT / "100k" / "train" / name, "candidate 100k/train"),
        (BDD_IMAGES_ROOT / "10k" / "train" / name, "candidate 10k/train"),
        (BDD_IMAGES_ROOT / "100k" / "val" / name, "candidate 100k/val"),
        (BDD_IMAGES_ROOT / "10k" / "val" / name, "candidate 10k/val"),
    ]
    for candidate, label in candidates:
        if candidate.exists():
            return candidate, label
    rel = _bdd_index().get(name)
    if rel is not None:
        return BDD_IMAGES_ROOT / rel, f"recursive index ({Path(rel).parent})"
    return None, "NOT FOUND"


def inspect_paths() -> None:
    """Verify image files resolve on disk for both datasets."""
    header("Path resolution check")
    out()
    out("ACDC (candidate: datasets/acdc/images/<file_name>):")
    for weather in ACDC_WEATHERS:
        path = (
            ACDC_ROOT
            / "labels"
            / weather
            / f"instancesonly_{weather}_train_gt_detection.json"
        )
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        images = [im for im in data["images"] if im["file_name"].startswith(f"{weather}/")]
        if not images:
            continue
        for im in random.sample(images, min(SAMPLES, len(images))):
            candidate = ACDC_ROOT / "images" / im["file_name"]
            status = "HIT" if candidate.exists() else "MISS"
            out(f"  [{status}] {candidate.relative_to(PROJECT_ROOT)}")

    out()
    out("BDD100K (ordered candidates, then recursive index):")
    data = json.loads((BDD_LABELS_ROOT / "bdd100k_labels_images_train.json").read_text("utf-8"))
    clear = [rec for rec in data if rec.get("attributes", {}).get("weather") == "clear"]
    for rec in random.sample(clear, min(SAMPLES, len(clear))):
        resolved, how = _resolve_bdd(rec["name"])
        if resolved is None:
            out(f"  [MISS] {rec['name']} -- {how}")
        else:
            out(f"  [HIT] {rec['name']} -> {resolved.relative_to(PROJECT_ROOT)}  [{how}]")


def main() -> None:
    """Run the full inspection and write the text report."""
    random.seed(42)
    inspect_acdc()
    inspect_bdd()
    inspect_paths()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"\n[report written] {REPORT_PATH}")


if __name__ == "__main__":
    main()
