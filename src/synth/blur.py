"""S5b stage module — calibrated condition-specific blur over appearance-calibrated S5.

S5b is an **ancillary one-factor ablation over S5** (``PROJECT.md`` §5). It keeps S5's
rendering (S3 weather structure rendered by ``weather.apply`` and global appearance matched
by ``physics.appearance_match``) and inserts a **condition-specific blur** between the two:

    weather.apply  ->  blur  ->  appearance_match

Blur is applied **before** the appearance match on purpose: the match sets per-channel
mean/std, and a normalized blur preserves the mean but lowers the std, so blurring *after*
the match would silently un-calibrate S5's appearance and make S5b differ from S5 by two
factors. Blurring first (capture-time blur, then global tone) keeps S5b a clean one-factor
change over S5.

Blur mapping (``PROJECT.md`` §2/§5):

- **rain**  -> directional motion blur aligned to the sampled streak slant;
- **fog**   -> isotropic defocus (Gaussian);
- **snow**  -> mild isotropic defocus (Gaussian);
- **night** -> none (illumination-limited; blur would only hurt the recall S5 recovers).

Strength is calibrated to the ACDC-train unlabeled pool by ``blur_calibrate.py`` (a
forward-curve sharpness estimator with a Tier 0 identifiability gate and a preview-chosen
fallback inside the pre-registered ranges below). Blur has **no S3 equivalent**, so the
ranges here are S5b's own pre-registered ranges, clipped and logged. This crosses the
"no frequency matching" line S5 respects, so it is an **S5b-only sensor/sharpness-calibration
exception**.

Physical models the blur draws on: motion blur for rain follows the rain-streak appearance
of [Garg & Nayar, TOG 2006]; defocus follows the atmospheric/optics scattering view of
[Koschmieder, 1924] and the synthetic fog protocol of [Sakaridis et al., IJCV 2018].

The calibrated strengths live in ``results/calibration/blur_stats.json`` (written by
``blur_calibrate.py``). For smoke tests the path can be overridden with the ``S5B_BLUR_STATS``
environment variable.

Harness interface: ``assign`` / ``sample`` / ``apply`` / ``describe``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import RESULTS_DIR  # noqa: E402
from synth import common as C  # noqa: E402
from synth import physics  # noqa: E402
from synth import weather  # noqa: E402

# Pre-registered S5b blur ranges (no S3 equivalent). Strength is expressed as a fraction of
# the image short side so BDD (1280x720) and ACDC (1920x1080) are comparable. Frozen (with
# the gate reporting any clip) before any ACDC scoring; see PROJECT.md §2/§5.
RANGES: dict[str, dict] = {
    "fog": {"kind": "gaussian", "metric_frac": (0.001, 0.012)},
    "snow": {"kind": "gaussian", "metric_frac": (0.0003, 0.008)},
    "rain": {"kind": "motion", "metric_frac": (0.0015, 0.030)},
    "night": {"kind": "none", "metric_frac": (0.0, 0.0)},
}

DEFAULT_STATS_PATH = RESULTS_DIR / "calibration" / "blur_stats.json"

_STATS_CACHE: dict | None = None


def stats_path() -> Path:
    """Return the blur-stats path (overridable for smoke tests via ``S5B_BLUR_STATS``)."""
    override = os.environ.get("S5B_BLUR_STATS")
    return Path(override) if override else DEFAULT_STATS_PATH


def load_blur_stats(path: Path | str | None = None) -> dict:
    """Load the calibrated blur stats (cached)."""
    global _STATS_CACHE
    if _STATS_CACHE is None:
        resolved = Path(path) if path is not None else stats_path()
        if not resolved.exists():
            raise FileNotFoundError(
                f"missing {resolved}; run `python src/synth/blur_calibrate.py` first "
                "(or set S5B_BLUR_STATS to a smoke file)"
            )
        _STATS_CACHE = json.loads(resolved.read_text(encoding="utf-8"))
    return _STATS_CACHE


def _fitted(condition: str) -> tuple[str, float, str]:
    """Return (kind, value, source) for a condition, clipped to the pre-registered range.

    Falls back to the range midpoint as a *provisional* value only if the calibrated file
    lacks the condition (recorded as ``source=provisional`` in the log).
    """
    spec = RANGES[condition]
    kind = spec["kind"]
    if kind == "none":
        return "none", 0.0, "none"
    low, high = spec["metric_frac"]
    block = load_blur_stats().get("fitted", {}).get(condition)
    if block and block.get("value") is not None:
        raw = float(block["value"])
        source = str(block.get("source", "fit"))
    else:
        raw = 0.5 * (low + high)
        source = "provisional"
    value = float(np.clip(raw, low, high))
    if value != raw:
        source = f"{source}+clipped"
    return kind, value, source


def effective_strength(condition: str) -> dict:
    """Return the effective blur config for a condition: kind, fraction, source."""
    kind, value, source = _fitted(condition)
    return {"kind": kind, "frac": value, "source": source}


def blur_params_for(condition: str, shape: tuple[int, ...]) -> dict:
    """Per-image blur parameters (JSON-loggable) for a condition of the given image shape."""
    short_side = float(min(shape[0], shape[1]))
    config = effective_strength(condition)
    kind = config["kind"]
    if kind == "none":
        return {"kind": "none"}
    frac = config["frac"]
    if kind == "gaussian":
        return {
            "kind": "gaussian",
            "sigma_frac": round(frac, 6),
            "sigma_px": round(frac * short_side, 4),
            "source": config["source"],
        }
    return {
        "kind": "motion",
        "length_frac": round(frac, 6),
        "length_px": round(frac * short_side, 4),
        "source": config["source"],
    }


def assign(sources: list[str], seed: int = C.SEED) -> dict[str, str]:
    """Reuse S3's balanced, seeded source->condition assignment (paired with S3/S5)."""
    return weather.assign(sources, seed)


def sample(rng: np.random.Generator, condition: str, shape: tuple[int, ...]) -> tuple[list[str], dict]:
    """S5 structure = S3 structure, plus the calibrated per-condition blur config."""
    names, params = weather.sample(rng, condition, shape)
    params["blur"] = blur_params_for(condition, shape)
    return names, params


def _defocus(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Isotropic Gaussian defocus."""
    if sigma_px <= 0.0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px, borderType=cv2.BORDER_REFLECT_101)


# Fixed number of shifted taps for the motion kernel so the blur scales smoothly with length
# (a length-dependent tap count introduces small non-monotonic steps in sharpness curves).
MOTION_SAMPLES = 17


def _motion(img: np.ndarray, length_px: float, angle_deg: float) -> np.ndarray:
    """Directional motion blur: average fractional-pixel shifted copies along the slant.

    ``angle_deg`` follows the S3 streak convention (from horizontal); the streak direction is
    ``(cos(angle), -sin(angle))`` in image coordinates (y down).
    """
    if length_px <= 0.0:
        return img
    offsets = np.linspace(-0.5, 0.5, MOTION_SAMPLES) * length_px
    theta = np.deg2rad(float(angle_deg))
    dx, dy = float(np.cos(theta)), -float(np.sin(theta))
    height, width = img.shape[:2]
    accumulator = np.zeros_like(img, dtype=np.float32)
    for offset in offsets:
        matrix = np.array([[1.0, 0.0, dx * offset], [0.0, 1.0, dy * offset]], dtype=np.float32)
        accumulator += cv2.warpAffine(
            img, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
    return accumulator / MOTION_SAMPLES


def apply_blur(img: np.ndarray, params: dict, blur: dict) -> np.ndarray:
    """Apply the condition-specific blur to a float32 BGR image."""
    kind = blur.get("kind", "none")
    if kind == "gaussian":
        return _defocus(img.astype(np.float32), float(blur["sigma_px"]))
    if kind == "motion":
        return _motion(img.astype(np.float32), float(blur["length_px"]), float(params["slant_deg"]))
    return img


def apply(img: np.ndarray, names: list[str], params: dict, rng: np.random.Generator) -> np.ndarray:
    """Render S3 structure -> blur -> calibrate global appearance to a target draw."""
    out = weather.apply(img, names, params, rng)
    out = apply_blur(out, params, params["blur"])
    target = physics.sample_target(names[0], rng)
    out = physics.appearance_match(out, target)
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def describe(names: list[str], params: dict) -> str:
    """Condition + S3 params + the calibrated blur config, as a stable loggable string."""
    blur = params.get("blur")
    clean = {key: value for key, value in params.items() if key != "blur"}
    base = weather.describe(names, clean)
    if not blur or blur.get("kind") in (None, "none"):
        return base
    if blur["kind"] == "gaussian":
        return f"{base};blur=gaussian(sigma_px={blur['sigma_px']:.3f},sigma_frac={blur['sigma_frac']:.6f})"
    return (
        f"{base};blur=motion(length_px={blur['length_px']:.3f},"
        f"length_frac={blur['length_frac']:.6f},angle_deg={float(params.get('slant_deg', 0.0)):.3f})"
    )
