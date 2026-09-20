"""Build the S5 dataset: A clear (referenced) + B calibrated-physics synthetic (5,000).

A/B come from the fixed seeded split of ``bdd_src_train.txt``; the A half is referenced
from ``bdd_src`` and only the transformed B half is written under ``data/yolo/bdd_s5/``.
Each B image gets exactly one condition with parameters sampled from the fitted
``results/calibration/synth_stats.json`` (``src/synth/physics.py``); condition assignment
matches S3 exactly, so S5↔S3 is a paired parameter-only comparison.

Usage::

    python src/synth/build_s5.py                 # full 5k (requires synth_stats.json)
    python src/synth/build_s5.py --limit 100     # smoke (separate dir)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth import common as C  # noqa: E402
from synth import stage_common  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build the S5 calibrated-physics dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Smoke mode: first N of each half.")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stage_common.run_build("s5", args.limit, args.seed, args.jobs, args.jpeg_quality)
