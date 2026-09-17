"""Evaluate a YOLO checkpoint with overall, per-class, and per-weather metrics.

Runs Ultralytics validation on a dataset config's ``val`` manifest, then (optionally)
on per-weather subsets derived from the ACDC image layout. Writes a JSON report under
``results/experiments/<exp>/eval/<name>.json`` with ``--exp``, else ``results/metrics/``.

Usage::

    python src/eval.py --weights results/experiments/S1/train/weights/best.pt \
        --data configs/acdc_official.yaml --name S1_acdc_official --per-weather --exp S1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    ACDC_WEATHERS,
    CONFIGS_DIR,
    EXPERIMENTS_DIR,
    PROJECT_ROOT,
    ensure_dir,
)

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TMP_DIR = METRICS_DIR / "_tmp"
VAL_PROJECT = PROJECT_ROOT / "results" / "runs" / "_val"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a YOLO checkpoint.")
    parser.add_argument("--weights", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--data", required=True, help="Dataset YAML (name or path).")
    parser.add_argument("--name", required=True, help="Output report name.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0)
    parser.add_argument("--split", default="val")
    parser.add_argument("--per-weather", action="store_true", help="Break down ACDC by weather.")
    parser.add_argument(
        "--exp",
        default=None,
        help="Experiment ID; groups output under results/experiments/<exp>/eval/.",
    )
    return parser.parse_args()


def _resolve_config(data: str) -> Path:
    """Resolve a config name or path to an existing YAML."""
    path = Path(data)
    if not path.exists():
        path = CONFIGS_DIR / f"{data}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config not found: {data}")
    return path


def _weather_of(image_path: str) -> str:
    """Extract the ACDC weather segment from a YOLO image path."""
    parts = Path(image_path).parts
    index = parts.index("images")
    return parts[index + 1]


def _summarize(metrics) -> dict:
    """Convert an Ultralytics DetMetrics object into a JSON-safe dict."""
    box = metrics.box
    per_class = {}
    for i, class_index in enumerate(box.ap_class_index):
        name = metrics.names[int(class_index)]
        per_class[name] = {
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "mAP50": float(box.ap50[i]),
            "mAP50-95": float(box.ap[i]),
        }
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
        "per_class": per_class,
    }


def _val(model: YOLO, data, args: argparse.Namespace, tag: str, val_project: Path):
    """Run validation with shared settings, writing under a controlled directory."""
    return model.val(
        data=data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
        plots=False,
        project=str(val_project),
        name=tag,
        exist_ok=True,
    )


def main() -> None:
    """Evaluate overall and per-weather, then write the report."""
    args = parse_args()
    config_path = _resolve_config(args.data)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if args.exp:
        base = EXPERIMENTS_DIR / args.exp
        metrics_dir = base / "eval"
        tmp_dir = metrics_dir / "_tmp"
        val_project = metrics_dir / "_val"
    else:
        metrics_dir = METRICS_DIR
        tmp_dir = TMP_DIR
        val_project = VAL_PROJECT
    ensure_dir(metrics_dir)

    model = YOLO(args.weights)
    report: dict = {
        "run": args.name,
        "weights": str(Path(args.weights).resolve()),
        "data": str(config_path),
        "overall": _summarize(
            _val(model, str(config_path), args, f"{args.name}_overall", val_project)
        ),
        "per_weather": {},
    }

    if args.per_weather:
        images = [
            line.strip()
            for line in Path(config["val"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        groups: dict[str, list[str]] = {weather: [] for weather in ACDC_WEATHERS}
        for image in images:
            weather = _weather_of(image)
            groups.setdefault(weather, []).append(image)

        ensure_dir(tmp_dir)
        for weather, members in groups.items():
            if not members:
                continue
            manifest = tmp_dir / f"{args.name}_{weather}.txt"
            manifest.write_text("\n".join(members) + "\n", encoding="utf-8")
            temp_config = tmp_dir / f"{args.name}_{weather}.yaml"
            temp_config.write_text(
                yaml.safe_dump(
                    {
                        "path": config["path"],
                        "train": config["train"],
                        "val": str(manifest),
                        "nc": config["nc"],
                        "names": config["names"],
                    }
                ),
                encoding="utf-8",
            )
            report["per_weather"][weather] = {
                "images": len(members),
                **_summarize(
                    _val(model, str(temp_config), args, f"{args.name}_{weather}", val_project)
                ),
            }

    out_path = metrics_dir / f"{args.name}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    overall = report["overall"]
    print(
        f"[eval] {args.name}: mAP50={overall['mAP50']:.4f} "
        f"mAP50-95={overall['mAP50-95']:.4f} "
        f"P={overall['precision']:.4f} R={overall['recall']:.4f}"
    )
    for weather, values in report["per_weather"].items():
        print(
            f"  {weather:<6} mAP50={values['mAP50']:.4f} "
            f"mAP50-95={values['mAP50-95']:.4f} (n={values['images']})"
        )
    print(f"[eval] report -> {out_path}")


if __name__ == "__main__":
    main()
