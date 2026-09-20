# src

## Purpose

- Dataset conversion, split building, materialization, training, evaluation, metric aggregation, and figure generation for the BDD100K→ACDC adverse-weather detection project.

## Ownership

- Owned by the project root `AGENTS.md`.

## Local Contracts

- Class mappings come only from `scripts/class_map.py` (via `src/common.py`); never redefine class IDs here.
- `datasets/` is read-only. Converters may read it but must write only under `data/yolo/`.
- `splits/**` manifests are the source of truth for every split; trainers and analyses read manifests, never glob directories.
- All splits are generated with `seed=42` and must remain reproducible. `build_splits.py` emits only `bdd_src_{train,val}`, `acdc_cv5/fold{0..4}_{train,test}`, `acdc_official_{train,val}`, `acdc_design`, `acdc_pool_unlabeled`, and `acdc_perweather/*`.
- **Leakage control:** `acdc_design.txt` (400 = 100/weather, from official train) is the only ACDC set allowed for inspecting failure modes / choosing S6 policies. `acdc_pool_unlabeled.txt` (official train − design = 1,200) is the unlabeled adaptation pool (S4 style, S5 calibration). The official val (`acdc_official_val.txt`, 406) is the only scored set and is never trained on or sampled into a pool. `acdc_official_train.txt` stays full (1,600) for the T1aug ceiling.
- `data/yolo/<dataset>/images/` and `labels/` are generated artifacts; do not hand-edit them.
- Labels must mirror images exactly (no orphans, no missing). Run `prune_labels.py` after `materialize.py`.
- `train.py --aug {none,default}` selects the augmentation preset: `none` = S0/T1 baseline, `default` = T1aug/S1 standard augmentation. All prior method/augmentation code was removed on 2026-09-17; only the baseline presets remain.
- `train.py` resolves `--project` to an absolute path so relative values cannot nest runs under Ultralytics' default `runs/` dir.
- `train.py` uses a uniform schedule for every config: `optimizer=auto`, `cos_lr=True`, `patience=30`, `batch=32`. Only augmentation/method varies across configs; never tune hyperparameters per config.
- `eval.py` writes overall, per-class, and per-weather metrics plus validator artifacts. Use it for every experiment so results are comparable.
- **Per-experiment eval set (locked 2026-09-19).** Every stage reports the S2-matched set, single seed 42: official ACDC val (`--name <ID>_acdc_official --per-weather`; primary), ACDC 5-fold (`--name <ID>_acdc_fold{0..4} --per-weather`; supplementary), and in-domain BDD val (`--name <ID>_in_domain`, no `--per-weather`; forgetting check). Do not skip the fold or in-domain evals.
- `train.py`/`eval.py` accept `--exp <ID>`; all artifacts for that experiment land under `results/experiments/<ID>/{train,eval,figures}`. Prefer `--exp` over raw `--project`/`--name`.
- Multi-run configs (e.g. T1, future F0/F1) get one experiment dir per run: `--exp <ID>_<run>` for training, `--name <ID>_acdc_<run>` for eval, so `aggregate.py` groups them under `<ID>_acdc`.
- `aggregate.py` scans `results/experiments/*/eval/*.json`, groups by experiment (stripping `_official`/`_foldN`), computes mean±std across folds, and writes `results/summary/summary.json`, `per_class.csv`, and `per_class.md`.
- `visualize.py` writes per-experiment figures (`training_curves.png`, `bars_*.png`), cross-experiment comparisons, and a `class_weather_<exp>.png` heatmap under `results/summary/figures/`.
- Pipeline order: `convert_acdc.py` → `convert_bdd.py` → `build_splits.py` → `write_configs.py` → `materialize.py` → `prune_labels.py` → per-stage synthesis (`synth/build_dataset.py --stage s2`, `synth/build_s3.py`, ...) → `train.py --exp` → `eval.py --exp` → `aggregate.py` → `visualize.py`.
- `src/synth/` experiment code is **append-only**: a stage's generator is frozen once its result is locked (S2 = `photometric.py`/`build_dataset.py`/`inspect_s2.py`; S3 = `weather.py`/`build_s3.py`/`inspect_s3.py`). Later stages add their own modules on the shared `synth/stage_common.py` harness (S5 = `calibrate.py`/`physics.py`/`build_s5.py`/`inspect_s5.py`; S5b = `blur.py`/`build_s5b.py`). Never overwrite an earlier stage's code.

## Work Guidance

- Run modules from the project root, e.g. `python src/data/build_splits.py`.
- `src/common.py` is the shared path/constant layer; add new dataset facts there, not duplicated in each script.
- Keep the ACDC combined-rain quirk and the nested BDD image layout handled in the converters; see `PROJECT.md` §3.

## Verification

- `python src/train.py --help` lists only the baseline flags (`--aug {none,default}`), with no method arguments.
- `python src/aggregate.py` runs without error and writes `results/summary/summary.json`, `per_class.csv`, and `per_class.md`.
- `python src/visualize.py` writes per-experiment figures and `results/summary/figures/comparison_*.png` plus `class_weather_*.png`.
- `python src/data/prune_labels.py --dry-run` reports 0 orphan labels for both datasets.
- `splits/` sizes: BDD train/val 10,000/2,000; ACDC official train/val 1,600/406; ACDC 5-fold (1,600/406 per fold); design 400 (100/weather); pool 1,200 (300/weather). Design ∪ pool = official train, design ∩ val = ∅.
- No retired split/config tokens (`train_5k`, `smoke`, `acdc_loo`, `holdout_`) remain in `splits/`, `configs/`, or `src/`.
- `python src/synth/inspect_s2.py --dataset-name bdd_s2` passes (labels byte-identical, no synthetic image equals its source); `splits/bdd_src_A_clear.txt`/`bdd_src_B_source.txt` are disjoint 5,000 each and their union is `bdd_src_train.txt`.
- `python src/synth/inspect_s3.py --dataset-name bdd_s3` passes; `splits/bdd_s3_train.txt` is 10,000 (5,000 `bdd_src` + 5,000 `bdd_s3`); `splits/bdd_s3_conditions.csv` is balanced 1,250/condition; `configs/bdd_s3.yaml` has `nc: 6`.

## Child DOX Index

- `synth/AGENTS.md` - offline S2-S6 dataset synthesis (photometric/weather/physics transforms on the fixed B half).
