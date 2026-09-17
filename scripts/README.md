# scripts

Utilities for inspecting the ACDC and BDD100K datasets and mapping their classes
into the project's unified 6-class scheme.

## `class_map.py`

Pure, side-effect-free module (stdlib only). Exports:

- `UNIFIED_CLASSES: dict[int, str]` - unified ID to name: `person, rider, car, truck, bus, bicycle`.
- `BDD_MAP: dict[str, int]` - BDD100K native category string to unified ID.
- `ACDC_MAP: dict[int, int | None]` - ACDC COCO category ID to unified ID (`None` = drop).
- `bdd_category_to_unified(cat: str) -> int | None`
- `acdc_category_to_unified(cat_id: int) -> int | None`

Motorcycle is folded into `bicycle`. Dropped: ACDC `train`; BDD `train`, `traffic light`, `traffic sign`, `lane`, `drivable area`.

Inspect the mappings directly:

```bash
python scripts/class_map.py
```

## `inspect_dataset.py`

Read-only, sample-based ground-truth check. Samples records from both datasets,
runs them through `class_map`, prints what a conversion would produce, and writes
the same output to `scripts/_reports/inspect_report.txt`. It does not load images
or write any dataset files.

```bash
python scripts/inspect_dataset.py
```

Uses `random.seed(42)`. Run from the project root; the script expects its own
directory on `sys.path` so it can `import class_map`.

## Dataset facts discovered (verified on this machine)

- ACDC COCO category IDs are exactly `24 person, 25 rider, 26 car, 27 truck, 28 bus, 31 train, 32 motorcycle, 33 bicycle`.
- ACDC train JSONs for `fog`/`night`/`snow` are per-weather (400 images each). The
  `rain` train JSON is the **combined all-weather** train file (1600 images = 400 per
  weather). It is the canonical ACDC train set; per-weather stats are obtained by
  filtering `file_name` on the weather prefix.
- ACDC image paths resolve as `datasets/acdc/images/<file_name>`.
- BDD100K detection images are nested under
  `datasets/bdd100k/bdd100k/bdd100k/images/100k/{train,val}/` (with
  `trainA/trainB/testA/testB` subfolders), not flat. Resolution needs an ordered
  candidate list plus a recursive index. `10k/` is a separate subset.
- BDD100K train labels: 69,863 records; `attributes` keys are `weather, scene, timeofday`.
  Weather values: `clear 37344, overcast 8770, undefined 8119, snowy 5549, rainy 5070,
  partly cloudy 4881, foggy 130`.
- BDD labels use both `box2d` (objects) and `poly2d` (`lane`, `drivable area`).

## Deviations from the original spec

- **ACDC rain:** treated as the combined all-weather file and filtered to `rain/` images
  for per-weather reporting (user-approved). The combined file will be the canonical
  ACDC train set in the conversion session.
- **BDD path resolution:** direct candidate paths alone do not hit; a recursive index is
  used as the final fallback, and the winning strategy is reported.
- **BDD record count:** the spec said ~70,000; the actual train label file has 69,863.

## Next session

Conversion pipeline: build YOLO-format detection datasets for both sources under a
separate top-level output directory (never inside `datasets/`), emitting image lists
and label `.txt` files with the unified class indices, plus a segmentation dataset
from `bdd100k_seg`.
