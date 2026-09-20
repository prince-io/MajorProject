"""Build the S5b dataset: A clear (referenced) + B calibrated-blur synthetic (5,000).

S5b is an ancillary one-factor ablation over S5: ``weather.apply`` (S3 structure) -> condition
blur -> ``appearance_match`` (global appearance calibrated to the ACDC-train pool). The blur
strength per condition comes from ``results/calibration/blur_stats.json`` (``blur_calibrate.py``);
see ``PROJECT.md`` §5 for the pre-registered design.

A/B come from the fixed seeded split of ``bdd_src_train.txt``; the A half is referenced from
``bdd_src`` and only the transformed B half is written under ``data/yolo/bdd_s5b/``. Condition
assignment matches S3/S5 exactly, so S5b<->S5 is a paired one-factor comparison.

Usage::

    python src/synth/build_s5b.py                 # full 5k (requires blur_stats.json)
    python src/synth/build_s5b.py --limit 100     # smoke (separate dir)
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
    parser = argparse.ArgumentParser(description="Build the S5b calibrated-blur dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Smoke mode: first N of each half.")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stage_common.run_build("s5b", args.limit, args.seed, args.jobs, args.jpeg_quality)
