"""Build an offline S-stage training dataset from the fixed B half of the BDD source.

For a stage ``s`` the dataset is ``A_clear (referenced from bdd_src) + B_transformed``:
the A half is never copied, only the transformed B half is written under
``data/yolo/bdd_<stage>/``. Labels are copied byte-identically (transforms are
geometry-preserving). S1 (A + B, all clear) is therefore the exact control.

Usage::

    python src/synth/build_dataset.py --stage s2                 # full 5k
    python src/synth/build_dataset.py --stage s2 --limit 100     # smoke (separate dir)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import CONFIGS_DIR, SPLITS_DIR, UNIFIED_CLASSES, ensure_dir  # noqa: E402
from synth import common as C  # noqa: E402
from synth import photometric  # noqa: E402

# stage -> module exposing sample_ops / apply_ops (S3/S5/S6 added later)
STAGE_MODULES = {"s2": photometric}

LOG_COLUMNS = ("source", "image", "label", "stage", "ops", "seed", "attempt")

# Guard: if a source image is usable but the transform makes it near-black/white or
# near-constant, resample (deterministically) up to this many times.
MAX_STAT_ATTEMPTS = 6


def _stat_ok(image: np.ndarray) -> bool:
    """Usable image heuristic: mean in [5, 250] and std >= 10."""
    return 5.0 <= float(image.mean()) <= 250.0 and float(image.std()) >= 10.0


def _fmt_ops(names: list[str], params: dict[str, float]) -> str:
    """Render ops+params as a stable, loggable string."""
    ordered = [n for n in photometric.ORDER if n in set(names)]
    return ";".join(f"{n}={params[n]:.4f}" for n in ordered)


def _degrade_one(task: tuple[str, str, int, int, str]) -> tuple[dict, dict]:
    """Load one source image, degrade it, write the image + label; return records."""
    source, out_images, seed, jpeg_quality, stage = task
    out_images = Path(out_images)
    label_dir = out_images.parent / "labels"
    module = STAGE_MODULES[stage]

    img = cv2.imread(source, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {source}")

    source_ok = _stat_ok(img)
    attempt = 0
    while True:
        extra = () if attempt == 0 else (attempt,)
        rng = np.random.default_rng(C.stable_seed(Path(source).name, stage, seed, *extra))
        names, params = module.sample_ops(rng)
        degraded = module.apply_ops(img, names, params, rng)
        degraded_u8 = np.rint(degraded).astype(np.uint8)
        if source_ok and not _stat_ok(degraded_u8) and attempt < MAX_STAT_ATTEMPTS:
            attempt += 1
            continue
        break

    name = Path(source).name
    out_image = out_images / name
    cv2.imwrite(str(out_image), degraded_u8, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    out_label = label_dir / f"{Path(source).stem}.txt"
    shutil.copy2(C.label_path_for(source), out_label)

    image_seed = C.stable_seed(Path(source).name, stage, seed, *extra)
    record = {
        "source": source,
        "image": str(out_image),
        "label": str(out_label),
        "stage": stage,
        "ops": [n for n in photometric.ORDER if n in set(names)],
        "params": params,
        "seed": image_seed,
        "attempt": attempt,
    }
    log_row = {
        "source": source,
        "image": str(out_image),
        "label": str(out_label),
        "stage": stage,
        "ops": _fmt_ops(names, params),
        "seed": image_seed,
        "attempt": attempt,
    }
    return record, log_row


def _write_config(dataset_name: str, root: Path, train_manifest: Path, val_manifest: Path) -> Path:
    """Write the stage dataset YAML."""
    names = "\n".join(f"  {i}: {n}" for i, n in UNIFIED_CLASSES.items())
    content = (
        f"path: {root}\n"
        f"train: {train_manifest}\n"
        f"val: {val_manifest}\n"
        f"nc: {len(UNIFIED_CLASSES)}\n"
        f"names:\n{names}\n"
    )
    path = CONFIGS_DIR / f"{dataset_name}.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def build(stage: str, limit: int | None, seed: int, jobs: int, jpeg_quality: int) -> Path:
    """Build the stage dataset and return its root."""
    if stage not in STAGE_MODULES:
        raise NotImplementedError(f"stage '{stage}' not implemented yet (have: {sorted(STAGE_MODULES)})")

    dataset_name = f"bdd_{stage}" if not limit else f"bdd_{stage}_smoke"
    root = C.dataset_root(dataset_name)
    images_dir = ensure_dir(root / "images")
    ensure_dir(root / "labels")

    a_clear, b_source = C.source_halves(seed)
    if not limit:
        C.write_halves(a_clear, b_source)

    a_sel = a_clear if not limit else a_clear[:limit]
    b_sel = b_source if not limit else b_source[:limit]

    tasks = [(src, str(images_dir), seed, jpeg_quality, stage) for src in b_sel]
    print(f"[build] {dataset_name}: transforming {len(tasks)} B images, A referenced={len(a_sel)}")

    records: list[dict] = []
    log_rows: list[dict] = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, (record, log_row) in enumerate(pool.map(_degrade_one, tasks), 1):
                records.append(record)
                log_rows.append(log_row)
                if i % 500 == 0:
                    print(f"[build]   {i}/{len(tasks)}")
    else:
        for i, task in enumerate(tasks, 1):
            record, log_row = _degrade_one(task)
            records.append(record)
            log_rows.append(log_row)
            if i % 500 == 0:
                print(f"[build]   {i}/{len(tasks)}")

    records.sort(key=lambda r: r["source"])
    log_rows.sort(key=lambda r: r["source"])

    manifest = SPLITS_DIR / f"{dataset_name}_train.txt"
    manifest.write_text("\n".join(a_sel + [r["image"] for r in records]) + "\n", encoding="utf-8")

    (root / "index.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (root / "synthesis_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(log_rows)

    config = _write_config(dataset_name, root, manifest, SPLITS_DIR / "bdd_src_val.txt")
    print(f"[build] images -> {images_dir}")
    print(f"[build] manifest -> {manifest} ({len(a_sel) + len(records)} entries)")
    print(f"[build] config -> {config}")
    return root


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build an offline S-stage dataset.")
    parser.add_argument("--stage", default="s2", choices=C.STAGES)
    parser.add_argument("--limit", type=int, default=None, help="Smoke mode: first N of each half.")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.stage, args.limit, args.seed, args.jobs, args.jpeg_quality)
