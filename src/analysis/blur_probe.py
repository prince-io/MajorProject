"""S5b Tier 1 — design-split blur sensitivity probe (screening only).

Evaluates the trained S5 model on the **400-image ACDC design split** with test-time blur at a
grid of strengths and reports a per-condition mAP@50 sensitivity curve. This is a cheap,
leak-free screen for whether a blur-augmented training arm (S5b Tier 2) is warranted.

Scope and limits (``PROJECT.md`` §5/§6, ``src/analysis/AGENTS.md``):

- uses the **design split only** (``splits/acdc_design.txt``, 100/condition), never the official
  val; it never writes a stage row and is not part of ``aggregate.py``/``visualize.py``;
- it is a **screening heuristic**: adding blur to already-degraded target images can only
  support, not prove, a training-time benefit.

Blur strengths are the calibrated S5b effective fractions (``results/calibration/blur_stats.json``)
scaled by the pre-registered multipliers below. Outputs go to tracked
``results/summary/blur_probe/``; scratch images/labels go to ``/tmp/opencode/blur_probe``.

Usage::

    python src/analysis/blur_probe.py --limit 5         # quick
    python src/analysis/blur_probe.py                   # full design split
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import SUMMARY_DIR, UNIFIED_CLASSES, ensure_dir  # noqa: E402
from synth import blur  # noqa: E402
from synth import blur_calibrate  # noqa: E402
from synth import common as C  # noqa: E402

DEFAULT_WEIGHTS = Path("results/experiments/S5/train/weights/best.pt")
DEFAULT_DESIGN = C.SPLITS_DIR / "acdc_design.txt"
DEFAULT_OUT = SUMMARY_DIR / "blur_probe"
DEFAULT_SCRATCH = Path("/tmp/opencode/blur_probe")
MULTIPLIERS = (0.0, 0.5, 1.0, 1.5, 2.0)
BLURRED_CONDITIONS = ("fog", "rain", "snow")


def _weather_of(image_path: str) -> str:
    parts = Path(image_path).parts
    return parts[parts.index("images") + 1]


def _acdc_label_for(image_path: str) -> Path:
    """Map an ACDC image to its label (``.../images/...`` -> ``.../labels/...``)."""
    parts = list(Path(image_path).parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def _blur_bgr(image: np.ndarray, kind: str, frac: float) -> np.ndarray:
    if frac <= 0.0 or kind == "none":
        return image
    pixels = frac * float(min(image.shape[0], image.shape[1]))
    if kind == "gaussian":
        return blur._defocus(image.astype(np.float32), pixels)
    return blur._motion(image.astype(np.float32), pixels, blur_calibrate.MOTION_CURVE_ANGLE)


def _write_dataset(images: list[str], scratch: Path, level: int, condition: str, kind: str, frac: float, limit: int | None) -> tuple[Path, int]:
    """Write blurred copies + copied labels for one (level, condition); return config path."""
    root = scratch / f"level_{level}"
    image_dir = root / "images" / condition
    label_dir = root / "labels" / condition
    ensure_dir(image_dir)
    ensure_dir(label_dir)
    manifest_paths: list[str] = []
    selected = images[:limit] if limit else images
    for source in selected:
        image = cv2.imread(source, cv2.IMREAD_COLOR)
        if image is None:
            continue
        blurred = np.clip(_blur_bgr(image, kind, frac), 0.0, 255.0).astype(np.uint8)
        out_image = image_dir / Path(source).name
        cv2.imwrite(str(out_image), blurred)
        shutil.copy2(_acdc_label_for(source), label_dir / f"{Path(source).stem}.txt")
        manifest_paths.append(str(out_image))

    manifests = ensure_dir(root / "manifests")
    manifest = manifests / f"{condition}.txt"
    manifest.write_text("\n".join(manifest_paths) + "\n", encoding="utf-8")

    config = ensure_dir(root / "configs") / f"{condition}.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": str(manifest),
                "val": str(manifest),
                "nc": len(UNIFIED_CLASSES),
                "names": {int(i): n for i, n in UNIFIED_CLASSES.items()},
            }
        ),
        encoding="utf-8",
    )
    return config, len(manifest_paths)


def _val_map50(model: YOLO, config: Path, scratch: Path, tag: str, args: argparse.Namespace) -> dict:
    metrics = model.val(
        data=str(config),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
        plots=False,
        project=str(scratch / "runs"),
        name=tag,
        exist_ok=True,
    )
    return {"mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map)}


def _figure(curves: dict, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    for condition, points in curves.items():
        axis.plot([p["frac"] for p in points], [p["mAP50"] for p in points], marker="o", label=condition)
    axis.set_xlabel("test-time blur (fraction of short side)")
    axis.set_ylabel("mAP@50 (design split)")
    axis.set_title("S5 model sensitivity to test-time blur (design split)")
    axis.grid(alpha=0.3)
    axis.legend()
    ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def probe(args: argparse.Namespace) -> Path:
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"weights not found: {args.weights}")
    design = C.load_manifest(Path(args.design))
    grouped: dict[str, list[str]] = {condition: [] for condition in C.CONDITIONS}
    for image in design:
        grouped[_weather_of(image)].append(image)

    effective = {condition: blur.effective_strength(condition) for condition in C.CONDITIONS}
    print(f"[probe] effective blur: { {c: effective[c] for c in BLURRED_CONDITIONS} }")
    scratch = Path(args.scratch)
    ensure_dir(scratch)
    model = YOLO(str(args.weights))

    curves: dict[str, list[dict]] = {}
    for condition in args.conditions:
        frac0 = effective[condition]["frac"]
        kind = effective[condition]["kind"]
        if kind == "none":
            continue
        points = []
        for level, multiplier in enumerate(MULTIPLIERS):
            frac = frac0 * multiplier
            config, count = _write_dataset(grouped[condition], scratch, level, condition, kind, frac, args.limit)
            if count == 0:
                continue
            scores = _val_map50(model, config, scratch, f"{condition}_m{multiplier}", args)
            points.append({"multiplier": multiplier, "frac": frac, "n": count, **scores})
            print(f"[probe] {condition:<6} x{multiplier:<3} frac={frac:.5f} mAP50={scores['mAP50']:.4f}")
        curves[condition] = points

    out_dir = ensure_dir(Path(args.out))
    payload = {
        "weights": str(Path(args.weights).resolve()),
        "design": str(Path(args.design).resolve()),
        "official_val_used": False,
        "screening_only": True,
        "multipliers": list(MULTIPLIERS),
        "motion_angle_deg": blur_calibrate.MOTION_CURVE_ANGLE,
        "effective": effective,
        "curves": curves,
    }
    json_path = out_dir / "blur_probe.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    figure = out_dir / "blur_probe_mAP50.png"
    _figure(curves, figure)

    lines = ["S5b Tier 1 design-split blur probe (screening only; never official val)", ""]
    for condition, points in curves.items():
        lines.append(f"{condition}:")
        for point in points:
            lines.append(f"  x{point['multiplier']:<3} frac={point['frac']:.5f} mAP50={point['mAP50']:.4f} n={point['n']}")
    (out_dir / "blur_probe.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[probe] wrote {json_path}")
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S5b Tier 1 design-split blur sensitivity probe.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--design", default=str(DEFAULT_DESIGN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--scratch", default=str(DEFAULT_SCRATCH))
    parser.add_argument("--conditions", nargs="+", default=list(BLURRED_CONDITIONS), choices=list(C.CONDITIONS))
    parser.add_argument("--limit", type=int, default=None, help="Debug: N design images per condition.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0)
    return parser.parse_args()


if __name__ == "__main__":
    probe(parse_args())
