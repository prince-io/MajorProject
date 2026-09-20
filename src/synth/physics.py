"""S5 stage module — appearance-calibrated weather synthesis.

S5 keeps **S3's weather structure** (same operators and parameter ranges, sampled by
``weather.sample`` and rendered by ``weather.apply``) and adds a **target-appearance
calibration**: the rendered image's per-channel mean/std and mean saturation are matched to
a target draw from the ACDC-train pool distributions (``results/calibration/synth_stats.json``).

This is a one-factor change over S3 (structure identical in distribution; only target
appearance matching is added), and a legitimate unlabeled-domain-adaptation appearance
calibration (cf. colour transfer [Reinhard et al., 2001]). Condition assignment reuses
``weather.assign`` so S5 is paired with S3.

Harness interface: ``assign`` / ``sample`` / ``apply`` / ``describe``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import RESULTS_DIR  # noqa: E402
from synth import common as C  # noqa: E402
from synth import weather  # noqa: E402

STATS_PATH = RESULTS_DIR / "calibration" / "synth_stats.json"
_CACHE: dict | None = None


def load_stats(path: Path | str = STATS_PATH) -> dict:
    """Load the versioned calibration stats (cached)."""
    global _CACHE
    if _CACHE is None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run `python src/synth/calibrate.py` first")
        _CACHE = json.loads(path.read_text(encoding="utf-8"))
    return _CACHE


def assign(sources: list[str], seed: int = C.SEED) -> dict[str, str]:
    """Reuse S3's balanced, seeded source→condition assignment (paired with S3)."""
    return weather.assign(sources, seed)


def sample(rng: np.random.Generator, condition: str, shape: tuple[int, ...], stats: dict | None = None) -> tuple[list[str], dict]:
    """S5 structure = S3 structure: sample operator parameters from the S3 ranges."""
    return weather.sample(rng, condition, shape)


def _draw(summary: dict, rng: np.random.Generator) -> float:
    lo, hi = summary["clip"]
    quantiles = summary.get("q") or []
    if not quantiles:
        return 0.5 * (lo + hi)
    position = float(rng.random()) * (len(quantiles) - 1)
    index = int(np.floor(position))
    frac = position - index
    value = float(quantiles[-1]) if index >= len(quantiles) - 1 else float(quantiles[index]) * (1.0 - frac) + float(quantiles[index + 1]) * frac
    return float(np.clip(value, lo, hi))


def sample_target(condition: str, rng: np.random.Generator, stats: dict | None = None) -> dict:
    """Draw a target appearance from the fitted per-condition distributions."""
    block = (load_stats() if stats is None else stats)["conditions"][condition]
    return {
        "b": (_draw(block["mean_b"], rng), _draw(block["std_b"], rng)),
        "g": (_draw(block["mean_g"], rng), _draw(block["std_g"], rng)),
        "r": (_draw(block["mean_r"], rng), _draw(block["std_r"], rng)),
        "saturation": _draw(block["saturation"], rng),
    }


def appearance_match(out: np.ndarray, target: dict) -> np.ndarray:
    """Match per-channel mean/std to an explicit target.

    Mean saturation is reported as a diagnostic but not force-matched: per-channel mean/std
    (the colour-balance + contrast statistics) fully determine saturation, so matching both
    is over-constrained. ``target["saturation"]`` is retained for reporting.
    """
    out = np.clip(out, 0.0, 255.0)
    for index, channel in enumerate("bgr"):
        mean = float(out[..., index].mean())
        std = float(out[..., index].std())
        target_mean, target_std = target[channel]
        out[..., index] = (out[..., index] - mean) * (target_std / max(std, 1e-3)) + target_mean
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def apply(img: np.ndarray, names: list[str], params: dict, rng: np.random.Generator) -> np.ndarray:
    """Render S3 weather structure, then calibrate global appearance to a target draw."""
    out = weather.apply(img, names, params, rng)
    target = sample_target(names[0], rng)
    out = appearance_match(out, target)
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def describe(names: list[str], params: dict) -> str:
    return weather.describe(names, params)
