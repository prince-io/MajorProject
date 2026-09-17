# src

## Purpose

- Dataset conversion, split building, materialization, training, evaluation, metric aggregation, and figure generation for the BDD100K→ACDC adverse-weather detection project.

## Ownership

- Owned by the project root `AGENTS.md`.

## Local Contracts

- Class mappings come only from `scripts/class_map.py` (via `src/common.py`); never redefine class IDs here.
- `datasets/` is read-only. Converters may read it but must write only under `data/yolo/`.
- `splits/**` manifests are the source of truth for every split; trainers and analyses read manifests, never glob directories.
- All splits are generated with `seed=42` and must remain reproducible.
- `data/yolo/<dataset>/images/` and `labels/` are generated artifacts; do not hand-edit them.
- Labels must mirror images exactly (no orphans, no missing). Run `prune_labels.py` after `materialize.py`.
- `train.py --aug {none,default}` selects the augmentation preset: `none` = B0/B1 baseline, `default` = B1aug/B2 standard augmentation. All prior method/augmentation code was removed on 2026-09-17; only the baseline presets remain.
- `train.py` resolves `--project` to an absolute path so relative values cannot nest runs under Ultralytics' default `runs/` dir.
- `train.py` uses a uniform schedule for every config: `optimizer=auto`, `cos_lr=True`, `patience=30`, `batch=32`. Only augmentation/method varies across configs; never tune hyperparameters per config.
- `eval.py` writes overall, per-class, and per-weather metrics plus validator artifacts. Use it for every experiment so results are comparable.
- `train.py`/`eval.py` accept `--exp <ID>`; all artifacts for that experiment land under `results/experiments/<ID>/{train,eval,figures}`. Prefer `--exp` over raw `--project`/`--name`.
- Multi-run configs (e.g. B1, future F0/F1) get one experiment dir per run: `--exp <ID>_<run>` for training, `--name <ID>_acdc_<run>` for eval, so `aggregate.py` groups them under `<ID>_acdc`.
- `aggregate.py` scans `results/experiments/*/eval/*.json`, groups by experiment (stripping `_official`/`_foldN`), computes mean±std across folds, and writes `results/summary/summary.json`, `per_class.csv`, and `per_class.md`.
- `visualize.py` writes per-experiment figures (`training_curves.png`, `bars_*.png`), cross-experiment comparisons, and a `class_weather_<exp>.png` heatmap under `results/summary/figures/`.
- Pipeline order: `convert_acdc.py` → `convert_bdd.py` → `build_splits.py` → `write_configs.py` → `materialize.py` → `prune_labels.py` → `train.py --exp` → `eval.py --exp` → `aggregate.py` → `visualize.py`.

## Work Guidance

- Run modules from the project root, e.g. `python src/data/build_splits.py`.
- `src/common.py` is the shared path/constant layer; add new dataset facts there, not duplicated in each script.
- Keep the ACDC combined-rain quirk and the nested BDD image layout handled in the converters; see `PROJECT.md` §3.

## Verification

- `python src/train.py --help` lists only the baseline flags (`--aug {none,default}`), with no method arguments.
- `python src/aggregate.py` runs without error and writes `results/summary/summary.json`, `per_class.csv`, and `per_class.md`.
- `python src/visualize.py` writes per-experiment figures and `results/summary/figures/comparison_*.png` plus `class_weather_*.png`.
- `python src/data/prune_labels.py --dry-run` reports 0 orphan labels for both datasets.
- Current manifest sizes in `splits/` match the dataset facts: BDD train/val 10,000/2,000; ACDC official train/val 1,600/406; ACDC 5-fold (1,600 train / 406 test per fold).

## Child DOX Index

- None.
