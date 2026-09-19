# src/synth

## Purpose

- Offline generation of the S2–S6 training datasets for the adverse-weather study (`PROJECT.md` §5): transform the fixed **B half (5,000 clear BDD)** of the 10k source into a stage-specific synthetic set, keeping the **A half (5,000 clear)** as the shared anchor.
- **S2** = generic photometric degradation (`photometric.py` + frozen `build_dataset.py`). **S3** = hand-set weather structure (`weather.py` + `build_s3.py`). S5/S6 (physics/combination) are added here later.

## Ownership

- Owned by `src/AGENTS.md`.

## Local Contracts

- **Append-only per experiment.** Once a stage has produced a locked result its generator is frozen: `photometric.py`, `build_dataset.py`, and `inspect_s2.py` are S2 and must not be edited. Later stages get their own modules and must never overwrite an earlier stage's code.
- Shared plumbing for S3+ lives in `stage_common.py` (worker loop, statistic guard, config/manifest/index/log writers). A new stage module implements `assign(sources, seed)`, `sample(rng, variant, shape)`, `apply(img, names, params, rng)`, and `describe(names, params)`; register it in `stage_common.MODULE_BY_STAGE`.
- **Fixed A/B split.** A/B halves come from `bdd_src_train.txt` via `common.source_halves(seed=42)` and are persisted to `splits/bdd_src_A_clear.txt` / `splits/bdd_src_B_source.txt`. Never change the split rule without a recorded reason — all S-stages share it, and **S1 (A+B clear) is the exact control** for every transformed stage.
- A-half images are **referenced from `bdd_src`, never copied**. Only the transformed B half is written under `data/yolo/bdd_<stage>/`.
- **Geometry preserving:** transforms must never move or change boxes. **Copy labels byte-identically** (`shutil.copy2`); never regenerate them.
- **Determinism:** per-image randomness is seeded by `common.stable_seed(filename, stage, seed, variant)` — order-independent and reproducible. Every op and parameter is logged.
- Machine-generated only: no ACDC images, no ACDC labels, and no ACDC-derived parameters unless a later stage explicitly declares a target-statistics calibration source. **S2 and S3 are zero-shot DG (no ACDC at all).**
- S2 is **purely photometric** (brightness, contrast, gamma, saturation, Gaussian noise). **S3 models weather structure** (Koschmieder fog, rain streaks, snow particles, night illumination). **Blur is excluded from both S2 and S3** and is deferred to S5, so S3-vs-S2 isolates weather structure and S3-vs-S5 isolates calibration, not blur.
- **S3 condition rules** (`PROJECT.md` §5): exactly one condition per image, balanced 1,250 each of fog/rain/snow/night, hand-set parameters only (never fit to ACDC), uniform depth, no mixed conditions, no local light sources, no snow accumulation. Assignment is `weather.assign` (seeded, balanced) and is recorded in `splits/bdd_s3_conditions.csv`.
- Ranges are deliberately narrow (ACDC has small distant objects): see `photometric.RANGES` and `weather.RANGES`. **S3 ranges are pre-registered before any ACDC evaluation** and must not be tuned from ACDC results.
- Cite prior work by full method/paper name and venue (e.g., "Koschmieder atmospheric scattering [Koschmieder, 1924]", "rain rendering [Garg & Nayar, TOG 2006]").

## Work Guidance

- Run from the project root.
- S2 (frozen): `python src/synth/build_dataset.py --stage s2 [--limit N]`, inspect with `python src/synth/inspect_s2.py --dataset-name bdd_s2 [--n 20]`.
- S3: `python src/synth/build_s3.py [--limit N] [--jobs 8]`, inspect with `python src/synth/inspect_s3.py --dataset-name bdd_s3 [--n 16]`.
- Generated artifacts live under `data/yolo/bdd_<stage>/` (gitignored); manifests and `bdd_s3_conditions.csv` under `splits/`; configs under `configs/`.
- S5/S6 (planned) will add their own modules and builders on top of `stage_common.py`, mirroring `build_s3.py`.

## Verification

- S2 (frozen): `python src/synth/build_dataset.py --stage s2 --limit 100` then `python src/synth/inspect_s2.py --dataset-name bdd_s2_smoke` pass (count matches, labels byte-identical, no all-black/white, each synthetic differs from its source).
- S3: `python src/synth/build_s3.py --limit 100 --jobs 4` then `python src/synth/inspect_s3.py --dataset-name bdd_s3_smoke` pass (balanced conditions, labels byte-identical, no degenerate image unless the source is exempt, each synthetic differs from its source).
- S3 full: `python src/synth/inspect_s3.py --dataset-name bdd_s3` passes; conditions are exactly 1,250 each; `splits/bdd_s3_train.txt` has 10,000 entries = 5,000 `bdd_src` + 5,000 `bdd_s3`; `configs/bdd_s3.yaml` has `nc: 6` and `val` = `splits/bdd_src_val.txt`; `data/yolo/bdd_s3` labels mirror images 1:1.
- Re-running a build yields identical image hashes (determinism).
- `splits/bdd_src_A_clear.txt` and `bdd_src_B_source.txt` are disjoint, each 5,000, and their union is exactly `bdd_src_train.txt`.

## Child DOX Index

- None.
