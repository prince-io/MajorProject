"""Build all split manifests and distribution logs for the project.

Produces, under ``splits/``:

- ``bdd_src_train.txt`` / ``bdd_src_val.txt``
- ``acdc_cv5/fold{0..4}_{train,test}.txt``
- ``acdc_official_{train,val}.txt``
- ``acdc_design.txt`` (400 = 100/weather, the leak-free method-design set)
- ``acdc_pool_unlabeled.txt`` (official train minus design, 1,200)
- ``acdc_perweather/{fog,night,rain,snow}.txt``

and distribution logs under ``results/splits/``.

Sampling: source = proportional stratified over rare-class presence, ``seed=42``;
ACDC = weather-stratified 5-fold + design split, ``seed=42``.

Usage::

    python src/data/build_splits.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    ACDC_WEATHERS,
    DATA_YOLO,
    RESULTS_SPLITS,
    SPLITS_DIR,
    UNIFIED_CLASSES,
    ensure_dir,
)

SEED = 42
N_TRAIN = 10_000
N_VAL = 2_000
N_FOLDS = 5
N_DESIGN_PER_WEATHER = 100


def _allocate(total: int, sizes: list[int]) -> list[int]:
    """Allocate ``total`` proportionally across groups using largest remainder."""
    weight = sum(sizes)
    if weight == 0 or total == 0:
        return [0] * len(sizes)
    raw = [s * total / weight for s in sizes]
    alloc = [int(x) for x in raw]
    remainder = total - sum(alloc)
    order = sorted(range(len(sizes)), key=lambda i: raw[i] - alloc[i], reverse=True)
    for i in order[:remainder]:
        alloc[i] += 1
    return alloc


def _redistribute(alloc: list[int], deficit: int, sizes: list[int]) -> None:
    """Add a deficit to groups that still have capacity, largest slack first."""
    order = sorted(range(len(sizes)), key=lambda i: sizes[i] - alloc[i], reverse=True)
    for i in order:
        if deficit <= 0:
            break
        slack = sizes[i] - alloc[i]
        if slack <= 0:
            continue
        take = min(slack, deficit)
        alloc[i] += take
        deficit -= take


def stratified_train_val(
    records: list[dict],
    n_train: int,
    n_val: int,
    key_fn: Callable[[dict], tuple],
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split records into disjoint stratified train/val sets."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)
    keys = sorted(groups)
    sizes = [len(groups[k]) for k in keys]
    train_alloc = _allocate(n_train, sizes)
    val_alloc = _allocate(n_val, sizes)
    for i in range(len(keys)):
        while train_alloc[i] + val_alloc[i] > sizes[i]:
            if train_alloc[i] > 0:
                train_alloc[i] -= 1
            elif val_alloc[i] > 0:
                val_alloc[i] -= 1
            else:
                break
    _redistribute(train_alloc, n_train - sum(train_alloc), sizes)
    _redistribute(val_alloc, n_val - sum(val_alloc), sizes)

    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    for key, t_count, v_count in zip(keys, train_alloc, val_alloc):
        group = groups[key][:]
        rng.shuffle(group)
        val.extend(group[:v_count])
        train.extend(group[v_count : v_count + t_count])
    return train, val


def write_manifest(path: Path, records: Iterable[dict]) -> None:
    """Write a manifest of YOLO-tree image paths, one per line."""
    ensure_dir(path.parent)
    lines = [record["yolo_image"] for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _class_counts(records: Iterable[dict]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(record["class_counts"])
    return {name: counter.get(name, 0) for name in UNIFIED_CLASSES.values()}


def _strata_counts(records: Iterable[dict]) -> dict[str, int]:
    """Count records per rare-class-presence stratum with JSON-safe string keys."""
    counter: Counter[str] = Counter()
    for record in records:
        key = ",".join(record["rare_present"]) or "none"
        counter[key] += 1
    return dict(counter)


def build_source() -> dict:
    """Build BDD source train/val splits and their distribution log."""
    index = json.loads((DATA_YOLO / "bdd_src" / "index.json").read_text(encoding="utf-8"))
    key_fn = lambda r: tuple(r["rare_present"])  # noqa: E731
    train, val = stratified_train_val(index, N_TRAIN, N_VAL, key_fn, SEED)

    write_manifest(SPLITS_DIR / "bdd_src_train.txt", train)
    write_manifest(SPLITS_DIR / "bdd_src_val.txt", val)

    log = {
        "pool": len(index),
        "splits": {
            "train": {
                "images": len(train),
                "strata": _strata_counts(train),
                "class_instances": _class_counts(train),
            },
            "val": {
                "images": len(val),
                "strata": _strata_counts(val),
                "class_instances": _class_counts(val),
            },
        },
    }
    (RESULTS_SPLITS / "bdd_source_distribution.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    print(f"[splits] BDD source: train={len(train)} val={len(val)}")
    return log


def build_target() -> dict:
    """Build ACDC 5-fold CV, official, design/pool, and per-weather splits + log."""
    index = json.loads((DATA_YOLO / "acdc" / "index.json").read_text(encoding="utf-8"))

    by_weather: dict[str, list[dict]] = defaultdict(list)
    for record in index:
        by_weather[record["weather"]].append(record)

    rng = random.Random(SEED)
    folds: dict[int, list[dict]] = defaultdict(list)
    for weather in ACDC_WEATHERS:
        records = by_weather[weather][:]
        rng.shuffle(records)
        for i, record in enumerate(records):
            folds[i % N_FOLDS].append(record)

    log: dict = {
        "pool": len(index),
        "cv5": {},
        "official": {},
        "design": {},
        "pool_unlabeled": {},
        "perweather": {},
    }

    for k in range(N_FOLDS):
        test = folds[k]
        train = [r for f, recs in folds.items() if f != k for r in recs]
        write_manifest(SPLITS_DIR / "acdc_cv5" / f"fold{k}_train.txt", train)
        write_manifest(SPLITS_DIR / "acdc_cv5" / f"fold{k}_test.txt", test)
        log["cv5"][f"fold{k}"] = {
            "train_images": len(train),
            "test_images": len(test),
            "test_weather": dict(Counter(r["weather"] for r in test)),
            "test_class_instances": _class_counts(test),
        }

    official_train = [r for r in index if r["split"] == "train"]
    official_val = [r for r in index if r["split"] == "val"]
    write_manifest(SPLITS_DIR / "acdc_official_train.txt", official_train)
    write_manifest(SPLITS_DIR / "acdc_official_val.txt", official_val)
    log["official"] = {
        "train_images": len(official_train),
        "val_images": len(official_val),
        "val_weather": dict(Counter(r["weather"] for r in official_val)),
        "val_class_instances": _class_counts(official_val),
    }

    # Leak-free design split: N_DESIGN_PER_WEATHER per weather from the official train
    # pool (seed 42). Its complement is the unlabeled adaptation pool; the official val
    # is never touched. The official train manifest stays full (1,600) for the ceiling.
    train_by_weather: dict[str, list[dict]] = defaultdict(list)
    for record in official_train:
        train_by_weather[record["weather"]].append(record)
    design_rng = random.Random(SEED)
    design: list[dict] = []
    for weather in ACDC_WEATHERS:
        records = train_by_weather[weather][:]
        design_rng.shuffle(records)
        design.extend(records[:N_DESIGN_PER_WEATHER])
    design_ids = {record["yolo_image"] for record in design}
    pool_unlabeled = [r for r in official_train if r["yolo_image"] not in design_ids]

    write_manifest(SPLITS_DIR / "acdc_design.txt", sorted(design, key=lambda r: r["yolo_image"]))
    write_manifest(SPLITS_DIR / "acdc_pool_unlabeled.txt", pool_unlabeled)
    log["design"] = {
        "images": len(design),
        "per_weather": dict(Counter(r["weather"] for r in design)),
        "class_instances": _class_counts(design),
    }
    log["pool_unlabeled"] = {
        "images": len(pool_unlabeled),
        "per_weather": dict(Counter(r["weather"] for r in pool_unlabeled)),
    }

    for weather in ACDC_WEATHERS:
        write_manifest(SPLITS_DIR / "acdc_perweather" / f"{weather}.txt", by_weather[weather])
        log["perweather"][weather] = {
            "images": len(by_weather[weather]),
            "class_instances": _class_counts(by_weather[weather]),
        }

    (RESULTS_SPLITS / "acdc_distribution.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    print(
        f"[splits] ACDC: pool={len(index)} -> 5 folds, official, "
        f"design={len(design)}, pool_unlabeled={len(pool_unlabeled)}, per-weather"
    )
    return log


if __name__ == "__main__":
    ensure_dir(SPLITS_DIR)
    ensure_dir(RESULTS_SPLITS)
    build_source()
    build_target()
    print("[splits] done")
