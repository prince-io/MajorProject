"""Generic photometric degradations for S2.

Operators: brightness, contrast, gamma, saturation, additive Gaussian noise.
Blur and physical weather effects are intentionally **deferred to S3/S5** so that S2
isolates tone/exposure/sensor-noise change from weather structure (see ``PROJECT.md``
§5 and the ``PLAN.md`` review).

All operators are geometry-preserving, so YOLO labels are copied unchanged. The ranges
are deliberately narrow because ACDC contains many small, distant objects that
aggressive degradation would erase.

Prior work the operator family draws on: RandAugment [Cubuk et al., CVPRW 2020],
AugMix [Hendrycks et al., ICLR 2020], TrivialAugment [Mueller & Hutter, ICCV 2021].
"""

from __future__ import annotations

import numpy as np

# name -> (low, high); narrow ranges pre-registered in PROJECT.md §5.
RANGES: dict[str, tuple[float, float]] = {
    "brightness": (0.7, 1.3),
    "contrast": (0.7, 1.2),
    "gamma": (0.8, 1.4),
    "saturation": (0.6, 1.1),
    "noise": (0.0, 10.0),  # Gaussian sigma in 0-255 units
}

# Fixed application order; noise is applied last.
ORDER: tuple[str, ...] = ("brightness", "contrast", "gamma", "saturation", "noise")

# Parameters near the identity are pushed away so every degraded image differs from its
# source (a pre-registered S2 contract); "noise" has no effect at sigma=0.
_IDENTITY = {"brightness": 1.0, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0, "noise": 0.0}
_MIN_DEV = {"brightness": 0.03, "contrast": 0.03, "gamma": 0.03, "saturation": 0.03, "noise": 1.0}

_BGR_GRAY = (0.114, 0.587, 0.299)  # cv2 BGR channel order


def sample_param(rng: np.random.Generator, name: str) -> float:
    """Sample one parameter in range, avoiding the (near-)identity value."""
    low, high = RANGES[name]
    value = float(rng.uniform(low, high))
    identity = _IDENTITY[name]
    deviation = _MIN_DEV[name]
    if abs(value - identity) < deviation:
        value = identity + deviation if value >= identity else identity - deviation
        value = min(max(value, low), high)
    return value


def apply_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Multiplicative exposure change."""
    return img * float(factor)


def apply_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Contrast around the image's own mean, so brightness is not shifted."""
    mean = float(img.mean())
    return (img - mean) * float(factor) + mean


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Power-law tone curve (gamma < 1 brightens, > 1 darkens)."""
    return 255.0 * np.power(np.clip(img, 0.0, 255.0) / 255.0, float(gamma))


def apply_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    """Blend toward (factor<1) or away from (factor>1) the luma image."""
    gray = np.tensordot(img, np.asarray(_BGR_GRAY), axes=([2], [0]))[..., None]
    return gray + (img - gray) * float(factor)


def apply_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive zero-mean Gaussian noise."""
    if sigma <= 0.0:
        return img
    noise = rng.normal(0.0, float(sigma), size=img.shape).astype(np.float32)
    return img + noise


def sample_ops(rng: np.random.Generator) -> tuple[list[str], dict[str, float]]:
    """Sample 1-3 ops (without replacement) and one parameter each."""
    k = int(rng.integers(1, 4))
    names = [str(n) for n in rng.choice(list(ORDER), size=k, replace=False)]
    params = {name: sample_param(rng, name) for name in names}
    return names, params


def apply_ops(
    img: np.ndarray,
    names: list[str],
    params: dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the given ops to a uint8/float image in the fixed ``ORDER``, then clip."""
    out = img.astype(np.float32)
    selected = set(names)
    for name in ORDER:
        if name not in selected:
            continue
        if name == "noise":
            out = apply_noise(out, params[name], rng)
        else:
            out = {
                "brightness": apply_brightness,
                "contrast": apply_contrast,
                "gamma": apply_gamma,
                "saturation": apply_saturation,
            }[name](out, params[name])
    return np.clip(out, 0.0, 255.0).astype(np.float32)
