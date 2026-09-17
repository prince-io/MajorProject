# src/synth

## Purpose

- Offline generation of the S2–S6 training datasets for the adverse-weather study (`PROJECT.md` §5): transform the fixed **B half (5,000 clear BDD)** of the 10k source into a stage-specific synthetic set, keeping the **A half (5,000 clear)** as the shared anchor.
- **S2** = generic photometric degradation (`photometric.py`). S3/S5/S6 (weather/physics/combination) are added here later.

## Ownership

- Owned by `src/AGENTS.md`.

## Local Contracts

- **Fixed A/B split.** A/B halves come from `bdd_src_train.txt` via `common.source_halves(seed=42)` and are persisted to `splits/bdd_src_A_clear.txt` / `splits/bdd_src_B_source.txt`. Never change the split rule without a recorded reason — all S-stages share it, and **S1 (A+B clear) is the exact control** for every transformed stage.
- A-half images are **referenced from `bdd_src`, never copied**. Only the transformed B half is written under `data/yolo/bdd_<stage>/`.
- **Geometry preserving:** transforms must never move or change boxes. **Copy labels byte-identically** (`shutil.copy2`); never regenerate them.
- **Determinism:** per-image randomness is seeded by `common.stable_seed(filename, stage, SEED)` — order-independent and reproducible. Every op and parameter is logged.
- Machine-generated only: no ACDC images, no ACDC labels, and no ACDC-derived parameters unless a later stage explicitly declares a target-statistics calibration source.
- S2 is **purely photometric** (brightness, contrast, gamma, saturation, Gaussian noise). **Blur/physical effects are deferred to S3/S5** so S2/S3/S5 separate tone/exposure from weather structure.
- Ranges are deliberately narrow (ACDC has small distant objects): see `photometric.RANGES`.
- Cite prior work by full method/paper name and venue (e.g., "RandAugment [Cubuk et al., CVPRW 2020]").

## Work Guidance

- Run from the project root. `python src/synth/build_dataset.py --stage s2 [--limit N]`.
- Generated artifacts live under `data/yolo/bdd_<stage>/` (gitignored); manifests under `splits/`; configs under `configs/`.
- Inspect before training: `python src/synth/inspect_s2.py --dataset-name bdd_s2 [--n 20]`.
- STAGES beyond `s2` raise `NotImplementedError` until implemented; S3/S5/S6 will reuse `build_dataset.py`.

## Verification

- `python src/synth/build_dataset.py --stage s2 --limit 100` then `python src/synth/inspect_s2.py --dataset-name bdd_s2_smoke` pass (count matches, labels byte-identical, no all-black/white, each synthetic differs from its source).
- Re-running the same command yields identical image hashes (determinism).
- `splits/bdd_src_A_clear.txt` and `bdd_src_B_source.txt` are disjoint, each 5,000, and their union is exactly `bdd_src_train.txt`.

## Child DOX Index

- None.
