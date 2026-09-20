"""S5b calibration — ACDC-pool sharpness attenuation for condition-specific blur.

S5b keeps S5's rendering (S3 weather structure + target-appearance match) and adds a
condition-specific blur between the two (``PROJECT.md`` §5). This module fits the blur
strength to the **ACDC-train unlabeled pool** (``splits/acdc_pool_unlabeled.txt``,
300/condition; never official val, never the design split).

Method (pre-registered 2026-09-20; forward-curve base revised from pre-registered at Tier 0,
before any ACDC scoring):

1. Measure a sharpness metric on a common **normalized short side** (BDD 1280x720 and
   ACDC 1920x1080 are both resized) so blur strengths are scale-comparable.
2. Build an explicit **forward curve** strength -> metric on the **S5-rendered base**, i.e.
   the actual images S5b starts from. (The pre-registered plan used the clear BDD A-half; S5
   weather structure itself changes sharpness -- rain/snow raise the metric ~1.3x -- so
   calibrating against clear BDD and then blurring S5 double-counts it. Building the curve on
   the S5 base makes S5b's realized sharpness match the ACDC target.)
3. Invert the curve at the per-condition **absolute ACDC-pool target** (clear BDD is retained
   only to report the cross-domain attenuation ratio), subject to a **Tier 0 identifiability
   gate** (curve monotonic, target inside the curve range, clipping to a pre-registered bound
   recorded). This mirrors the S5 lesson: parameter-level matching across the BDD<->ACDC
   base-domain gap can saturate.
4. If the gate fails, fall back to a **preview-chosen strength inside the pre-registered
   range** (``FALLBACK_FRAC`` below); if ACDC is already sharper than the S5 base (no blur can
   help), no blur is applied.

The primary metric is the **variance-normalized Laplacian** (Laplacian variance divided by
image variance), which is monotonic under both defocus and motion blur and largely
contrast-robust; raw Laplacian variance and the high-band energy fraction are reported as
secondary diagnostics. Motion blur for rain follows [Garg & Nayar, TOG 2006]; defocus
follows the scattering view of [Koschmieder, 1924] / [Sakaridis et al., IJCV 2018].

Writes a versioned ``results/calibration/blur_stats.json`` (pool hash + commit); the S5b
builder reads it and never recomputes. S5's ``calibrate.py`` is untouched.

Usage::

    python src/synth/blur_calibrate.py --pool-limit 20 --ref-limit 40   # quick
    python src/synth/blur_calibrate.py                                 # full
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
from synth import blur  # noqa: E402
from synth import common as C  # noqa: E402

SEED = C.SEED
POOL_MANIFEST = C.SPLITS_DIR / "acdc_pool_unlabeled.txt"
REFERENCE_MANIFEST = C.A_CLEAR_MANIFEST
OUT_PATH = blur.DEFAULT_STATS_PATH

# Sharpness is measured after resizing each image so its short side is this many pixels.
NORMALIZED_SHORT_SIDE = 360

# Pre-registered sample sizes: the S5-base forward curve, the ACDC-pool target, and the
# clear-BDD reference used to report the cross-domain attenuation ratio.
REFERENCE_SAMPLE = 200
TARGET_SAMPLE = 300  # the full pool is 300/condition
S5_SAMPLE = 150

# High-band cutoff in normalized radial frequency (cycles/pixel) for the HF metric.
HIGH_CUTOFF = 0.25

# Forward-curve grids, as fractions of the (normalized) short side. Log-spaced (with a zero
# anchor) because both curves are steep near the origin; they cover beyond the pre-registered
# ranges so an out-of-range target is detectable rather than silently clipped.
def _log_grid(start: float, stop: float, count: int) -> list[float]:
    return [0.0] + [round(float(v), 6) for v in np.logspace(np.log10(start), np.log10(stop), count)]


GRID = {
    "gaussian": _log_grid(0.0004, 0.018, 15),
    "motion": _log_grid(0.0008, 0.045, 15),
}
MOTION_CURVE_ANGLE = 77.5  # mid of the S3 rain slant range (70-85 deg)

# Preview-chosen fallback strengths (fraction of short side) used only when the Tier 0 gate
# fails; inside the pre-registered ranges in blur.RANGES. Adjustable at Tier 0 by inspection.
FALLBACK_FRAC = {"fog": 0.004, "snow": 0.003, "rain": 0.012}

METRICS = ("normalized_laplacian", "laplacian_var", "hf_energy_ratio")
PRIMARY = "normalized_laplacian"


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _load_gray(path: str) -> np.ndarray | None:
    """Load an image as full-resolution float grayscale (blur is applied at this scale)."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None if image is None else image.astype(np.float32)


def _to_measure_scale(gray: np.ndarray) -> np.ndarray:
    """Resize a grayscale image so its short side is NORMALIZED_SHORT_SIDE (measurement only)."""
    height, width = gray.shape[:2]
    scale = NORMALIZED_SHORT_SIDE / float(min(height, width))
    if abs(scale - 1.0) <= 1e-3:
        return gray
    return cv2.resize(
        gray, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)


def _resize_gray(path: str) -> np.ndarray | None:
    """Load an image as grayscale at the measurement scale."""
    image = _load_gray(path)
    return None if image is None else _to_measure_scale(image)


def sharpness(gray: np.ndarray) -> dict:
    """Sharpness metrics of a grayscale float image.

    ``normalized_laplacian`` (primary) = Laplacian variance / image variance; it is monotonic
    under both defocus and motion blur and largely contrast-robust. ``laplacian_var`` and the
    high-band energy fraction ``hf_energy_ratio`` are secondary diagnostics. The input is
    resized to NORMALIZED_SHORT_SIDE first, so callers may pass full-resolution or measured-scale
    images interchangeably.
    """
    gray = _to_measure_scale(gray)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    window = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F)
    detrended = (gray - float(gray.mean())) * window
    power = np.abs(np.fft.fftshift(np.fft.fft2(detrended))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(gray.shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(gray.shape[1]))[None, :]
    radius = np.sqrt(fx**2 + fy**2)
    total = float(power.sum())
    energy_high = float(power[radius >= HIGH_CUTOFF].sum())
    return {
        "normalized_laplacian": laplacian_var / (float(gray.var()) + 1e-6),
        "laplacian_var": laplacian_var,
        "hf_energy_ratio": energy_high / max(total, 1e-9),
    }


def _median_metrics(images: list[np.ndarray]) -> dict:
    values = {name: [] for name in METRICS}
    for image in images:
        stats = sharpness(image)
        for name in METRICS:
            values[name].append(stats[name])
    return {name: float(np.median(values[name])) for name in METRICS}


def _blur_gray(gray: np.ndarray, kind: str, frac: float) -> np.ndarray:
    if frac <= 0.0:
        return gray
    pixels = frac * float(min(gray.shape[:2]))  # native-scale blur, matching blur.py generation
    if kind == "gaussian":
        return blur._defocus(gray, pixels)
    return blur._motion(gray, pixels, MOTION_CURVE_ANGLE)


def _forward_curve(reference: list[np.ndarray], kind: str) -> list[dict]:
    curve = []
    for frac in GRID[kind]:
        blurred = [_blur_gray(image, kind, frac) for image in reference]
        metrics = _median_metrics(blurred)
        metrics["strength"] = frac
        curve.append(metrics)
    return curve


def _is_monotonic(curve: list[dict], key: str) -> bool:
    values = [point[key] for point in curve]
    tolerance = 1e-6 * max(abs(values[0]), 1.0)
    return all(values[i + 1] <= values[i] + tolerance for i in range(len(values) - 1))


def _invert(curve: list[dict], target: float, key: str) -> float | None:
    """Interpolate the strength where the (descending) curve crosses ``target``.

    Interpolation is done on ``log`` of the metric (all positive), which tracks the steep
    convex falloff near the origin far better than linear interpolation in metric space.
    """
    strengths = [point["strength"] for point in curve]
    raw = [point[key] for point in curve]
    if target > raw[0] or target < raw[-1]:
        return None
    values = [np.log(max(value, 1e-9)) for value in raw]
    target_log = np.log(max(target, 1e-9))
    for index in range(len(values) - 1):
        high, low = values[index], values[index + 1]
        if high >= target_log >= low and high != low:
            weight = (high - target_log) / (high - low)
            return float(strengths[index] + weight * (strengths[index + 1] - strengths[index]))
    return float(strengths[-1])


def _group_pool(paths: list[str], limit: int | None) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {condition: [] for condition in C.CONDITIONS}
    for path in paths:
        parts = Path(path).parts
        grouped[parts[parts.index("images") + 1]].append(path)
    if limit:
        grouped = {condition: members[:limit] for condition, members in grouped.items()}
    return grouped


def _s5_base_images(condition: str, limit: int | None) -> list[np.ndarray]:
    """The S5-rendered B-half images for a condition (the actual S5b starting point)."""
    root = C.dataset_root("bdd_s5")
    index_path = root / "index.json"
    if not index_path.exists():
        return []
    records = json.loads(index_path.read_text(encoding="utf-8"))
    paths = sorted(record["image"] for record in records if record.get("condition") == condition)
    if limit:
        paths = paths[:limit]
    return [image for image in (_load_gray(p) for p in paths) if image is not None]


def calibrate(pool_limit: int | None, ref_limit: int | None, target_limit: int | None, out: Path) -> Path:
    load_limit = target_limit or pool_limit
    reference_paths = C.load_manifest(REFERENCE_MANIFEST)
    if ref_limit:
        reference_paths = reference_paths[:ref_limit]
    reference = [image for image in (_resize_gray(p) for p in reference_paths) if image is not None]
    if not reference:
        raise RuntimeError("no reference (BDD clear A-half) images loaded")
    reference_metrics = _median_metrics(reference)
    print(f"[blur] reference: {len(reference)} clear BDD images; primary={reference_metrics[PRIMARY]:.4f}")

    pool = _group_pool(C.load_manifest(POOL_MANIFEST), load_limit)
    fitted: dict[str, dict] = {}
    gate: dict[str, dict] = {}
    closed_loop: dict[str, dict] = {}
    targets: dict[str, dict] = {}
    bases: dict[str, dict] = {}
    curves: dict[str, list[dict]] = {}
    for condition in C.CONDITIONS:
        spec = blur.RANGES[condition]
        kind = spec["kind"]
        if kind == "none":
            fitted[condition] = {"kind": "none", "value": 0.0, "source": "none", "clipped": False, "bound": None}
            gate[condition] = {"passed": True, "reason": "condition has no blur"}
            targets[condition] = {}
            closed_loop[condition] = {}
            bases[condition] = {}
            curves[condition] = []
            continue

        # ACDC target (absolute) and the S5 base the blur will be applied to.
        images = [image for image in (_resize_gray(p) for p in pool[condition]) if image is not None]
        target = _median_metrics(images)
        targets[condition] = target
        base = _s5_base_images(condition, S5_SAMPLE)
        if not base:
            raise RuntimeError(f"no S5 base images for '{condition}'; build data/yolo/bdd_s5 first")
        bases[condition] = _median_metrics(base)
        curve = _forward_curve(base, kind)
        curves[condition] = curve

        monotonic = _is_monotonic(curve, PRIMARY)
        candidate = _invert(curve, target[PRIMARY], PRIMARY)
        in_range = candidate is not None
        low, high = spec["metric_frac"]
        passed = bool(monotonic and in_range)
        bound = None
        if passed:
            if candidate < low:
                bound, value = "lower", float(low)
            elif candidate > high:
                bound, value = "upper", float(high)
            else:
                value = float(candidate)
            source = "fit"
        elif monotonic and target[PRIMARY] > curve[0][PRIMARY]:
            value, source = 0.0, "no-blur"  # ACDC is sharper than the S5 base; blur would hurt
        else:
            value = float(np.clip(FALLBACK_FRAC[condition], low, high))
            source = "fallback"
        clipped = bound is not None
        if source == "no-blur":
            reason = "ACDC sharper than the S5 base; no blur applied"
        elif not monotonic:
            reason = "forward curve non-monotonic"
        elif not in_range:
            reason = "target outside forward-curve range (content/camera confound)"
        elif bound == "upper":
            reason = "fitted value hit the upper pre-registered bound (range may be too narrow)"
        elif bound == "lower":
            reason = "fitted value hit the lower pre-registered bound (little blur needed)"
        else:
            reason = "ok"
        fitted[condition] = {"kind": kind, "value": value, "source": source, "clipped": clipped, "bound": bound}
        gate[condition] = {
            "passed": passed, "reason": reason, "monotonic": monotonic,
            "target_in_range": in_range, "clipped": clipped, "bound": bound,
            "base_primary": bases[condition][PRIMARY], "target_primary": target[PRIMARY],
        }

        # closed-loop: apply the effective fraction to the S5 base and re-measure
        probe = base[: min(20, len(base))]
        measured = _median_metrics([_blur_gray(image, kind, value) for image in probe])
        closed_loop[condition] = {
            "applied_frac": value,
            "effective": {"kind": kind, "value": value, "source": source},
            "base_primary": bases[condition][PRIMARY],
            "target_primary": target[PRIMARY],
            "measured_primary": measured[PRIMARY],
            "reference_primary": reference_metrics[PRIMARY],
            "base_ratio": bases[condition][PRIMARY] / reference_metrics[PRIMARY],
            "target_ratio": target[PRIMARY] / reference_metrics[PRIMARY],
            "measured_ratio": measured[PRIMARY] / reference_metrics[PRIMARY],
            "abs_err": abs(measured[PRIMARY] - target[PRIMARY]),
        }
        print(
            f"[blur] {condition:<6} target={target[PRIMARY]:.4f} base={bases[condition][PRIMARY]:.4f} "
            f"fitted={value:.5f} source={source:<10} gate={'PASS' if passed else 'FAIL'} ({reason})"
        )

    stats = {
        "version": "1.0",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "code_commit": _code_commit(),
        "source_pool": str(POOL_MANIFEST),
        "source_pool_hash": _manifest_hash(POOL_MANIFEST),
        "reference_manifest": str(REFERENCE_MANIFEST),
        "n_pool": int(sum(len(members) for members in pool.values())),
        "n_reference": len(reference),
        "normalized_short_side": NORMALIZED_SHORT_SIDE,
        "high_cutoff": HIGH_CUTOFF,
        "primary_metric": PRIMARY,
        "metrics": list(METRICS),
        "ranges": {condition: blur.RANGES[condition] for condition in C.CONDITIONS},
        "reference_metrics": reference_metrics,
        "targets": targets,
        "bases": bases,
        "forward_curve": curves,
        "gate": gate,
        "fitted": fitted,
        "closed_loop": closed_loop,
    }
    ensure_dir(out.parent)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[blur] wrote {out}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate S5b blur strength on the ACDC-train pool.")
    parser.add_argument("--pool-limit", type=int, default=None, help="Debug: N pool images per condition.")
    parser.add_argument("--target-limit", type=int, default=TARGET_SAMPLE, help="N target images per condition.")
    parser.add_argument("--ref-limit", type=int, default=REFERENCE_SAMPLE, help="N clear BDD reference images.")
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="Output stats JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    calibrate(args.pool_limit, args.ref_limit, args.target_limit, args.out)
