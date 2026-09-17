# datasets

## Purpose

- Read-only source datasets used to train YOLO **detection** models (the project is detection only).
- Two sources: ACDC (adverse-weather driving) and BDD100K (large-scale driving).

## Ownership

- Owned by the project root `AGENTS.md`.
- Dataset-specific layout, classes, and quirks are owned by each child `AGENTS.md`.

## Local Contracts

- Treat every file under `datasets/` as immutable source material. Never edit, rename, move, or delete raw dataset files.
- Do not write derived artifacts (converted YOLO datasets, splits, caches, training outputs) into `datasets/`; place them in a separate top-level directory.
- Preserve the original directory structure and file names so provenance stays traceable.
- Keep dataset roots self-contained: `datasets/acdc/` and `datasets/bdd100k/` do not share files.

## Work Guidance

- Confirm a dataset's class list and label format from its child `AGENTS.md` before writing conversion or training code.
- When converting to YOLO format, keep the raw source untouched and record the class-index mapping used.

## Verification

- `python src/data/convert_acdc.py` reports 2,006 index records; `python src/data/convert_bdd.py` reports 12,454 clear/daytime records with 0 missing images.
- `python src/data/build_splits.py` writes the `splits/` manifests reproducibly with `seed=42`: BDD source (10k/2k), ACDC 5-fold + official (1,600/406), design split (400 = 100/weather), unlabeled pool (1,200), and per-weather manifests.

## Child DOX Index

- `acdc/AGENTS.md` - ACDC adverse-weather images and COCO detection labels.
- `bdd100k/AGENTS.md` - BDD100K detection images/labels and segmentation masks.
