"""S3 — hand-set, weather-structured synthesis (fog / rain / snow / night).

S3 is the first stage that models weather **explicitly** instead of by generic
photometric tone change (S2). Each source image in the fixed B half of the BDD
source is assigned exactly one condition (balanced 1,250 per condition) and
transformed by a geometry-preserving operator, so YOLO labels are copied
unchanged. No ACDC image, label, or statistic is used: S3 is zero-shot DG.

Design boundaries (``PROJECT.md`` §5, ``paper/literature/05_weather_synthesis.md``):

- **S3 vs S5.** S3 uses hand-set parameters; S5 fits the *same physical model
  families* to measured ACDC-train statistics. Fog is therefore a constant-
  transmission Koschmieder model, not a depth-modelled one.
- **No blur.** Blur is excluded from S3 (and from S2); it belongs to S5.
- **No mixed conditions, no local light sources, no snow accumulation, no depth.**
  Each is a single-condition, uniform-depth, hand-set operator.

Physical models the operators draw on: Koschmieder atmospheric scattering
[Koschmieder, 1924], rain rendering [Garg & Nayar, TOG 2006], and the synthetic
fog protocol of [Sakaridis et al., IJCV 2018]. Night follows the low-light
illumination view of [Guo et al., CVPR 2020].
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from synth import common as C

# Hand-set ranges (pre-registered in PROJECT.md §5 before any ACDC evaluation).
RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "fog": {
        "t": (0.35, 0.70),  # constant transmission t = exp(-beta*d)
        "airlight": (180.0, 235.0),  # atmospheric light level (0-255)
        "airlight_jitter": (0.0, 8.0),  # per-channel variation of the airlight
        "desat": (0.7, 1.0),
    },
    "rain": {
        "streaks_per_mpix": (366.0, 1221.0),  # 150-500 per 640x640, area-scaled
        "length_px": (10.0, 30.0),
        "width_px": (1.0, 2.0),
        "slant_deg": (70.0, 85.0),  # from horizontal
        "alpha": (0.3, 0.5),
        "contrast": (0.8, 0.95),
    },
    "snow": {
        "density": (0.02, 0.07),  # fraction of pixels covered by particles
        "radius_px": (2.0, 6.0),
        "alpha": (0.5, 0.7),
        "brightness": (1.0, 1.15),
        "contrast": (0.85, 1.0),
        "desat": (0.8, 1.0),
    },
    "night": {
        # gamma > 1 crushes midtones/shadows (low-light); gamma < 1 would lift
        # shadows and produce the known "dim daytime" artifact.
        "brightness": (0.35, 0.60),
        "gamma": (1.0, 1.4),
        "vignette": (0.1, 0.3),
    },
}

TINT_DELTA = 5.0  # fixed warm/cool channel shift (R+5,B-5) or (R-5,B+5)
_STREAK_GREY = 200.0
_PARTICLE_GREY = 245.0
_BGR_GRAY = np.asarray((0.114, 0.587, 0.299), dtype=np.float32)


def assign(sources: list[str], seed: int = C.SEED) -> dict[str, str]:
    """Map each source to a condition, balanced and deterministic (seed 42).

    The sorted source list is shuffled with a seeded RNG, then split into
    contiguous, near-equal blocks so the condition counts stay balanced even for
    smoke subsets (``limit``).
    """
    order = sorted(sources)
    random.Random(seed).shuffle(order)
    conditions = C.CONDITIONS
    total = len(order)
    result: dict[str, str] = {}
    for index, source in enumerate(order):
        slot = (index * len(conditions)) // max(total, 1)
        result[source] = conditions[min(slot, len(conditions) - 1)]
    return result


def sample(rng: np.random.Generator, condition: str, shape: tuple[int, ...]) -> tuple[list[str], dict]:
    """Sample the parameters of one condition for an image of ``shape``."""
    samplers = {
        "fog": _sample_fog,
        "rain": _sample_rain,
        "snow": _sample_snow,
        "night": _sample_night,
    }
    if condition not in samplers:
        raise ValueError(f"unknown condition: {condition}")
    return [condition], samplers[condition](rng, shape)


def apply(img: np.ndarray, names: list[str], params: dict, rng: np.random.Generator) -> np.ndarray:
    """Apply a sampled condition to a BGR uint8 image, returning float32."""
    appliers = {
        "fog": _apply_fog,
        "rain": _apply_rain,
        "snow": _apply_snow,
        "night": _apply_night,
    }
    condition = names[0]
    if condition not in appliers:
        raise ValueError(f"unknown condition: {condition}")
    out = appliers[condition](img.astype(np.float32), params, rng)
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def describe(names: list[str], params: dict) -> str:
    """Render condition + params as a stable, loggable string."""
    parts = [names[0]]
    for key, value in params.items():
        if isinstance(value, (bool, np.bool_)):
            rendered = str(bool(value))
        elif isinstance(value, (int, np.integer)):
            rendered = str(int(value))
        elif isinstance(value, (float, np.floating)):
            rendered = f"{float(value):.4f}"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return ";".join(parts)


def _contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Contrast around the image's own mean."""
    return (img - float(img.mean())) * float(factor) + float(img.mean())


def _desaturate(img: np.ndarray, factor: float) -> np.ndarray:
    """Blend toward the luma image (factor < 1 desaturates)."""
    gray = np.tensordot(img, _BGR_GRAY, axes=([2], [0]))[..., None]
    return gray + (img - gray) * float(factor)


def _sample_fog(rng: np.random.Generator, _shape: tuple[int, ...]) -> dict:
    low, high = RANGES["fog"]["t"]
    transmission = float(rng.uniform(low, high))
    base = float(rng.uniform(*RANGES["fog"]["airlight"]))
    jitter = float(rng.uniform(*RANGES["fog"]["airlight_jitter"]))
    air = np.clip(base + rng.uniform(-jitter, jitter, size=3), 0.0, 255.0)
    return {
        "t": transmission,
        "air_b": float(air[0]),
        "air_g": float(air[1]),
        "air_r": float(air[2]),
        "desat": float(rng.uniform(*RANGES["fog"]["desat"])),
    }


def _sample_rain(rng: np.random.Generator, shape: tuple[int, ...]) -> dict:
    height, width = shape[:2]
    area_mpix = (height * width) / 1.0e6
    rate = float(rng.uniform(*RANGES["rain"]["streaks_per_mpix"]))
    return {
        "n_streaks": int(max(1, round(rate * area_mpix))),
        "length_px": float(rng.uniform(*RANGES["rain"]["length_px"])),
        "width_px": float(rng.uniform(*RANGES["rain"]["width_px"])),
        "slant_deg": float(rng.uniform(*RANGES["rain"]["slant_deg"])),
        "alpha": float(rng.uniform(*RANGES["rain"]["alpha"])),
        "contrast": float(rng.uniform(*RANGES["rain"]["contrast"])),
    }


def _sample_snow(rng: np.random.Generator, shape: tuple[int, ...]) -> dict:
    height, width = shape[:2]
    density = float(rng.uniform(*RANGES["snow"]["density"]))
    radius_low, radius_high = RANGES["snow"]["radius_px"]
    mean_radius = 0.5 * (radius_low + radius_high)
    count = int(max(1, round(density * height * width / (np.pi * mean_radius**2))))
    return {
        "n_particles": count,
        "density": density,
        "radius_low": float(radius_low),
        "radius_high": float(radius_high),
        "alpha": float(rng.uniform(*RANGES["snow"]["alpha"])),
        "brightness": float(rng.uniform(*RANGES["snow"]["brightness"])),
        "contrast": float(rng.uniform(*RANGES["snow"]["contrast"])),
        "desat": float(rng.uniform(*RANGES["snow"]["desat"])),
    }


def _sample_night(rng: np.random.Generator, _shape: tuple[int, ...]) -> dict:
    return {
        "brightness": float(rng.uniform(*RANGES["night"]["brightness"])),
        "gamma": float(rng.uniform(*RANGES["night"]["gamma"])),
        "tint": str(rng.choice(("warm", "cool"))),
        "tint_delta": TINT_DELTA,
        "vignette": float(rng.uniform(*RANGES["night"]["vignette"])),
    }


def _apply_fog(img: np.ndarray, params: dict, _rng: np.random.Generator) -> np.ndarray:
    """Constant-transmission Koschmieder scattering: I = J*t + A*(1 - t)."""
    transmission = params["t"]
    air = np.asarray((params["air_b"], params["air_g"], params["air_r"]), dtype=np.float32)
    out = img * transmission + air * (1.0 - transmission)
    return _desaturate(out, params["desat"])


def _apply_rain(img: np.ndarray, params: dict, rng: np.random.Generator) -> np.ndarray:
    """Sparse directional streaks plus a mild contrast reduction."""
    height, width = img.shape[:2]
    overlay = np.zeros((height, width, 3), dtype=np.float32)
    slant = np.deg2rad(params["slant_deg"])
    length = params["length_px"]
    offset_x, offset_y = np.cos(slant) * length, np.sin(slant) * length
    thickness = max(1, int(round(params["width_px"])))

    for _ in range(params["n_streaks"]):
        start_x = float(rng.uniform(0.0, width))
        start_y = float(rng.uniform(0.0, height))
        direction = 1.0 if rng.random() < 0.5 else -1.0
        end = (int(round(start_x + direction * offset_x)), int(round(start_y - offset_y)))
        start = (int(round(start_x)), int(round(start_y)))
        cv2.line(overlay, start, end, (_STREAK_GREY,) * 3, thickness)

    overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
    out = img + params["alpha"] * overlay
    return _contrast(out, params["contrast"])


def _apply_snow(img: np.ndarray, params: dict, rng: np.random.Generator) -> np.ndarray:
    """Falling-particle overlay (no accumulation) plus brightness/contrast/desat."""
    height, width = img.shape[:2]
    overlay = np.zeros((height, width, 3), dtype=np.float32)
    radius_low, radius_high = params["radius_low"], params["radius_high"]
    for _ in range(params["n_particles"]):
        centre = (int(rng.integers(0, width + 1)), int(rng.integers(0, height + 1)))
        radius = int(round(rng.uniform(radius_low, radius_high)))
        cv2.circle(overlay, centre, radius, (_PARTICLE_GREY,) * 3, -1)

    overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
    coverage = np.clip(overlay / _PARTICLE_GREY, 0.0, 1.0)
    alpha = params["alpha"]
    out = img * (1.0 - alpha * coverage) + _PARTICLE_GREY * alpha * coverage
    out = out * params["brightness"]
    out = _contrast(out, params["contrast"])
    return _desaturate(out, params["desat"])


def _apply_night(img: np.ndarray, params: dict, _rng: np.random.Generator) -> np.ndarray:
    """Brightness scaling, gamma tone shift, warm/cool tint, and edge vignette."""
    out = np.clip(img * params["brightness"], 0.0, 255.0)
    out = 255.0 * np.power(np.clip(out, 0.0, 255.0) / 255.0, params["gamma"])

    delta = params["tint_delta"]
    if params["tint"] == "warm":  # R up, B down (BGR channels: 0=B, 2=R)
        out[:, :, 2] += delta
        out[:, :, 0] -= delta
    else:  # cool: R down, B up
        out[:, :, 2] -= delta
        out[:, :, 0] += delta

    out = np.clip(out, 0.0, 255.0)
    out = out * _vignette_mask(img.shape[:2], params["vignette"])[:, :, None]
    return out


def _vignette_mask(shape: tuple[int, int], strength: float) -> np.ndarray:
    """Smooth radial mask peaking at 1 in the centre and 1-strength at the edges."""
    height, width = shape
    kernel_x = cv2.getGaussianKernel(width, width * 0.5)
    kernel_y = cv2.getGaussianKernel(height, height * 0.5)
    mask = kernel_y * kernel_x.T
    mask = mask / float(mask.max())
    return 1.0 - strength * (1.0 - mask)
