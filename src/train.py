"""Thin Ultralytics YOLO training wrapper.

Usage::

    python src/train.py --data configs/bdd_src.yaml --model yolov8n.pt --epochs 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import EXPERIMENTS_DIR  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Project-wide training schedule. Kept identical across every config (B0, B1, B1aug,
# B2); only the augmentation preset varies, so comparisons stay controlled.
# optimizer=auto is left to Ultralytics (sets lr0=0.001 AdamW); do not hand-set lr0.
TRAIN_SCHEDULE: dict = {
    "optimizer": "auto",
    "amp": True,
    "pretrained": True,
}

# Augmentation presets. "none" disables all stochastic transforms (B0 baseline);
# "default" uses Ultralytics defaults (B2 standard-augmentation baseline).
AUG_NONE: dict = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "auto_augment": None,
    "close_mosaic": 0,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a YOLO detector on a project config.")
    parser.add_argument("--data", required=True, help="Dataset YAML (name or path).")
    parser.add_argument("--model", default="yolov8n.pt", help="Model weights or YAML.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32, help="-1 = auto.")
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument(
        "--aug",
        choices=("none", "default"),
        default="default",
        help="Augmentation preset: none (B0/B1), default (B1aug/B2).",
    )
    parser.add_argument("--project", default=str(PROJECT_ROOT / "results" / "runs"))
    parser.add_argument("--name", default="train")
    parser.add_argument(
        "--exp",
        default=None,
        help="Experiment ID; groups output under results/experiments/<exp>/train/.",
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--cos-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cosine LR schedule (uniform across configs).",
    )
    return parser.parse_args()


def main() -> None:
    """Resolve the data config and launch training."""
    args = parse_args()
    data = args.data
    if not Path(data).exists():
        data = str(PROJECT_ROOT / "configs" / f"{data}.yaml")

    # Resolve to an absolute path so Ultralytics never nests runs under its own
    # default runs_dir (e.g. runs/detect/results/runs/...). With --exp, output is
    # grouped under results/experiments/<exp>/train/.
    if args.exp:
        project = str((EXPERIMENTS_DIR / args.exp).resolve())
        name = "train"
    else:
        project = str(Path(args.project).resolve())
        name = args.name

    model = YOLO(args.model)
    overrides = AUG_NONE if args.aug == "none" else {}
    train_kwargs = dict(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        fraction=args.fraction,
        project=project,
        name=name,
        patience=args.patience,
        **TRAIN_SCHEDULE,
        cos_lr=args.cos_lr,
        **overrides,
    )
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
