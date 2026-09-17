# acdc

## Purpose

- ACDC (Adverse Conditions Dataset with Correspondences): driving images captured under four adverse weather conditions, with COCO-format detection annotations.

## Ownership

- Owned by `datasets/AGENTS.md`.

## Local Contracts

- Layout:
  - `images/<weather>/<split>/<sequence>/<frame>.png` where weather is `fog|night|rain|snow` and split is `train|val|test`.
  - `labels/<weather>/instancesonly_<weather>_<split>_gt_detection.json` (train/val) and `instancesonly_<weather>_test_image_info.json` (test, images only).
- All images are PNG, 4006 total: fog 1000, night 1006, rain 1000, snow 1000.
- Detection label files are COCO-style dicts with `images`, `categories`, `annotations`.
- Classes (8): `person`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`; COCO IDs are exactly `24 person, 25 rider, 26 car, 27 truck, 28 bus, 31 train, 32 motorcycle, 33 bicycle`.
- `instancesonly_fog/night/snow_train_gt_detection.json` are per-weather (400 images each). `instancesonly_rain_train_gt_detection.json` is the **combined all-weather** train file (1600 images, 400 per weather) and is the canonical ACDC train set; filter `file_name` on the weather prefix for per-weather subsets.
- `val_gt` and `test_image_info` files are per-weather and correct.
- Test split has no ground-truth annotations.

## Work Guidance

- Use the per-weather `gt_detection.json` files for train/val; do not expect test labels.

## Verification

- `python src/data/convert_acdc.py` converts 2,006 labeled images (train 1,600 / val 406) with the combined-rain file filtered by prefix.

## Child DOX Index

- None.
