"""S5 calibration — target appearance statistics for calibrated weather synthesis.

S5 keeps **S3's weather structure** (same operators and parameter ranges, rendered by
``weather.apply``) and calibrates **global appearance** to the **unlabeled** ACDC-train
pool (``splits/acdc_pool_unlabeled.txt``, 300/condition):

- per-channel mean and standard deviation (colour balance + contrast), and
- mean saturation.

At generation, ``physics.apply`` renders the S3 structure and then matches a target draw of
those statistics, so the synthetic half matches the measured target appearance
(closed-loop by construction). This is a **one-factor** change over S3 and a legitimate
unlabeled-domain-adaptation appearance calibration (cf. colour transfer
[Reinhard et al., 2001]).

Why not calibrate the operator parameters themselves: rendering from **BDD clear** scenes
means absolute image statistics do not track **ACDC** statistics (different cameras, cities,
tone), so parameter-level statistic inversion is confounded. See `PROJECT.md` §5.

Writes a versioned ``results/calibration/synth_stats.json`` (pool hash + commit); the
builder never recomputes it.

Usage::

    python src/synth/calibrate.py --pool-limit 20   # quick
    python src/synth/calibrate.py                   # full pool
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import RESULTS_DIR, ensure_dir  # noqa: E402
from synth import common as C  # noqa: E402
from synth import physics  # noqa: E402

SEED = C.SEED
POOL_MANIFEST = C.SPLITS_DIR / "acdc_pool_unlabeled.txt"
OUT_PATH = RESULTS_DIR / "calibration" / "synth_stats.json"

PROBS = np.linspace(0.01, 0.99, 33)
APPEARANCE_FIELDS = ("mean_b", "mean_g", "mean_r", "std_b", "std_g", "std_r", "saturation")


def _saturation_mean(img: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean())


def appearance_stats(img: np.ndarray) -> dict:
    """Per-channel mean/std and mean saturation of a BGR uint8 image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    means = [float(img[..., c].mean()) for c in range(3)]
    stds = [float(img[..., c].astype(np.float32).std()) for c in range(3)]
    return {
        "mean_b": means[0], "mean_g": means[1], "mean_r": means[2],
        "std_b": stds[0], "std_g": stds[1], "std_r": stds[2],
        "saturation": _saturation_mean(img),
        "_luma": float(gray.mean()),
    }


def _summarize(values, clip) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    lo, hi = clip
    if array.size == 0:
        return {"n": 0, "median": float("nan"), "mad": 0.0, "p10": float("nan"), "p90": float("nan"),
                "clip": [float(lo), float(hi)], "q": [], "rejected": 0, "clipped": 0}
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)) * 1.4826)
    kept = array[np.abs(array - median) <= 3.0 * mad] if mad > 0 else array
    if kept.size == 0:
        kept = array
    return {
        "n": int(kept.size), "median": median, "mad": mad,
        "p10": float(np.quantile(kept, 0.10)), "p90": float(np.quantile(kept, 0.90)),
        "clip": [float(lo), float(hi)], "q": [float(v) for v in np.quantile(kept, PROBS)],
        "rejected": int(array.size - kept.size), "clipped": int(np.sum((kept < lo) | (kept > hi))),
    }


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _code_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(Path(__file__).resolve().parents[2]), text=True).strip()
    except Exception:
        return "unknown"


def calibrate(pool_limit: int | None) -> Path:
    pool = C.load_manifest(POOL_MANIFEST)
    if pool_limit:
        grouped: dict[str, list[str]] = {cond: [] for cond in C.CONDITIONS}
        for path in pool:
            grouped[Path(path).parts[Path(path).parts.index("images") + 1]].append(path)
        pool = [p for cond in C.CONDITIONS for p in grouped[cond][:pool_limit]]

    collected: dict[str, dict[str, list[float]]] = {cond: {field: [] for field in APPEARANCE_FIELDS} for cond in C.CONDITIONS}
    loaded = 0
    for path in pool:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        condition = Path(path).parts[Path(path).parts.index("images") + 1]
        stats = appearance_stats(img)
        for field in APPEARANCE_FIELDS:
            collected[condition][field].append(stats[field])
        loaded += 1
        if loaded % 200 == 0:
            print(f"[calibrate]   {loaded}/{len(pool)}")

    conditions = {
        condition: {field: _summarize(collected[condition][field], (0.0, 255.0)) for field in APPEARANCE_FIELDS}
        for condition in C.CONDITIONS
    }

    stats_json = {
        "version": "3.0",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "code_commit": _code_commit(),
        "source_pool": str(POOL_MANIFEST),
        "source_pool_hash": _manifest_hash(POOL_MANIFEST),
        "n_pool": len(pool),
        "appearance_fields": list(APPEARANCE_FIELDS),
        "conditions": conditions,
        "self_test": _self_test(conditions),
    }
    ensure_dir(OUT_PATH.parent)
    OUT_PATH.write_text(json.dumps(stats_json, indent=2), encoding="utf-8")
    print(f"[calibrate] wrote {OUT_PATH}")
    for condition in C.CONDITIONS:
        block = conditions[condition]
        print(f"  {condition:6} mean_r={block['mean_r']['median']:.1f} std_r={block['std_r']['median']:.1f} sat={block['saturation']['median']:.1f}")
    return OUT_PATH


def _self_test(conditions: dict) -> dict:
    """Closed-loop check: the appearance match must reproduce the median target statistics."""
    from synth import weather

    reference = [img for img in (cv2.imread(p, cv2.IMREAD_COLOR) for p in C.load_manifest(C.A_CLEAR_MANIFEST)[:6]) if img is not None]
    report: dict[str, dict] = {}
    for condition in C.CONDITIONS:
        block = conditions[condition]
        target = {
            "b": (block["mean_b"]["median"], block["std_b"]["median"]),
            "g": (block["mean_g"]["median"], block["std_g"]["median"]),
            "r": (block["mean_r"]["median"], block["std_r"]["median"]),
            "saturation": block["saturation"]["median"],
        }
        errors = {field: [] for field in APPEARANCE_FIELDS}
        for img in reference:
            rng = np.random.default_rng(C.stable_seed("self_test", condition, "x"))
            names, params = weather.sample(rng, condition, img.shape)
            rendered = np.clip(weather.apply(img, names, params, rng), 0, 255).astype(np.uint8)
            matched = physics.appearance_match(rendered.astype(np.float32), target)
            measured = appearance_stats(np.clip(matched, 0, 255).astype(np.uint8))
            for field in APPEARANCE_FIELDS:
                target_value = {"mean_b": target["b"][0], "mean_g": target["g"][0], "mean_r": target["r"][0],
                                "std_b": target["b"][1], "std_g": target["g"][1], "std_r": target["r"][1],
                                "saturation": target["saturation"]}[field]
                errors[field].append(abs(measured[field] - target_value))
        report[condition] = {"mean_abs_err": {field: round(float(np.mean(values)), 3) for field, values in errors.items()}}
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate S5 target appearance statistics on the ACDC pool.")
    parser.add_argument("--pool-limit", type=int, default=None, help="Debug: N pool images per condition.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    calibrate(args.pool_limit)
