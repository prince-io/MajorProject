"""Build the S3 dataset: A clear (referenced) + B weather-synthesized (5,000).

A/B come from the fixed seeded split of ``bdd_src_train.txt``; the A half is
referenced from ``bdd_src`` and only the transformed B half is written under
``data/yolo/bdd_s3/``. Each B image gets exactly one condition (balanced 1,250
each of fog/rain/snow/night) with hand-set, geometry-preserving parameters
(``src/synth/weather.py``). S1 (A + B, all clear) is the exact control.

Usage::

    python src/synth/build_s3.py                 # full 5k
    python src/synth/build_s3.py --limit 100     # smoke (separate dir)
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
    parser = argparse.ArgumentParser(description="Build the S3 weather-synthesis dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Smoke mode: first N of each half.")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stage_common.run_build("s3", args.limit, args.seed, args.jobs, args.jpeg_quality)
