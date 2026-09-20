# src/synth

## Purpose

- Offline generation of the S2–S6 training datasets for the adverse-weather study (`PROJECT.md` §5): transform the fixed **B half (5,000 clear BDD)** of the 10k source into a stage-specific synthetic set, keeping the **A half (5,000 clear)** as the shared anchor.
- **S2** = generic photometric degradation (`photometric.py` + frozen `build_dataset.py`). **S3** = hand-set weather structure (`weather.py` + `build_s3.py`, frozen). **S5** = S3 structure + target-appearance calibration (`calibrate.py` + `physics.py` + `build_s5.py` + `inspect_s5.py`, implemented). **S5b** = calibrated-blur ablation over S5 (`blur.py` + `blur_calibrate.py` + `build_s5b.py` + `inspect_s5b.py`, implemented). S6 (combination) later. The S5b Tier 1 probe lives in `src/analysis/blur_probe.py` (not a stage row).

## Ownership

- Owned by `src/AGENTS.md`.

## Local Contracts

- **Append-only per experiment.** Once a stage has produced a locked result its generator is frozen: `photometric.py`, `build_dataset.py`, and `inspect_s2.py` are S2 and must not be edited. Later stages get their own modules and must never overwrite an earlier stage's code.
- Shared plumbing for S3+ lives in `stage_common.py` (worker loop, statistic guard, config/manifest/index/log writers). A new stage module implements `assign(sources, seed)`, `sample(rng, variant, shape)`, `apply(img, names, params, rng)`, and `describe(names, params)`; register it in `stage_common.MODULE_BY_STAGE`.
- **Fixed A/B split.** A/B halves come from `bdd_src_train.txt` via `common.source_halves(seed=42)` and are persisted to `splits/bdd_src_A_clear.txt` / `splits/bdd_src_B_source.txt`. Never change the split rule without a recorded reason — all S-stages share it, and **S1 (A+B clear) is the exact control** for every transformed stage.
- A-half images are **referenced from `bdd_src`, never copied**. Only the transformed B half is written under `data/yolo/bdd_<stage>/`.
- **Geometry preserving:** transforms must never move or change boxes. **Copy labels byte-identically** (`shutil.copy2`); never regenerate them.
- **Determinism:** per-image randomness is seeded by `common.stable_seed(filename, stage, seed, variant)` — order-independent and reproducible. Every op and parameter is logged.
- Machine-generated only: no ACDC images or labels and no ACDC-derived parameters, **except** where a stage's pre-registered contract declares a statistics-only calibration source. **S2 and S3 are zero-shot DG (no ACDC at all); S5 is unlabeled DA** (below).
- S2 is **purely photometric** (brightness, contrast, gamma, saturation, Gaussian noise). **S3 models weather structure** (Koschmieder fog, rain streaks, snow particles, night illumination). **Blur is excluded from S2, S3, and S5**, and is deferred beyond the S3/S5 comparison (to S6c), so S2/S3/S5 differences isolate tone, weather structure, and calibration — not blur.
- **S3 condition rules** (`PROJECT.md` §5): exactly one condition per image, balanced 1,250 each of fog/rain/snow/night, hand-set parameters only (never fit to ACDC), uniform depth, no mixed conditions, no local light sources, no snow accumulation. Assignment is `weather.assign` (seeded, balanced) and is recorded in `splits/bdd_s3_conditions.csv`.
- **S5 (appearance-calibrated synthesis; revised from pre-registered) rules** (`PROJECT.md` §5): keep S3's operators and parameter ranges **unchanged** (zero new structure). `calibrate.py` measures per-channel mean/std (+ saturation, diagnostic) over `splits/acdc_pool_unlabeled.txt` (unlabeled DA; never official val, never the design split) and writes the versioned `results/calibration/synth_stats.json`. `physics.py` samples S3 params (identical structure/distribution), renders via `weather.apply`, then matches a sampled target per-channel mean/std; **saturation is reported, not force-matched** (per-channel mean/std determine it). Reuse S3's source list and condition-per-source (`bdd_s3_conditions.csv`) so S5↔S3 is a one-factor paired comparison. The builder never recomputes calibration. **Parameter-level physics inversion (direct or forward-curve/severity) was tested and is not identifiable across the BDD↔ACDC base-domain gap — do not reintroduce it as if identifiable.** Structural changes (row-depth fog, local night, blur) are S6c.
- **S5b (calibrated blur ablation, over S5) rules** (`PROJECT.md` §5): ancillary arm, **not** folded into S5. Build on S5's pipeline with the pre-registered order **`weather.apply` (S3 structure) → blur → `appearance_match`** in `blur.py`/`build_s5b.py` (S5 files untouched). Blur is applied **before** the appearance match so S5b differs from S5 by blur alone (blurring after the match reduces per-channel std and makes it two factors). Condition-specific: **rain directional motion** (aligned to streak slant), **fog/snow isotropic defocus**, **night none**. Strength **calibrated to the ACDC-train pool** by sharpness attenuation via a **new estimator in `blur_calibrate.py`** (S5's `calibrate.py` untouched), writing `results/calibration/blur_stats.json`: a forward curve (strength → estimator) built on the **S5-rendered base** with **native-resolution blur**, measured at a **common normalized short side** (variance-normalized Laplacian primary + HF-energy ratio), and inverted to the **absolute ACDC-pool sharpness** under a **Tier 0 identifiability gate** (monotonic / target-in-range / bound clipping recorded); on gate failure fall back to preview-chosen values inside the ranges. (Revised from pre-registered at Tier 0: the clear-BDD base double-counted S5's own sharpness change.) Blur has **its own ranges** (no S3 equivalent), clipped + logged — frozen at Tier 0: fog σ [0.001,0.012], rain length [0.0015,0.030], snow σ [0.0003,0.008], night none. This crosses the "no frequency matching" line S5 respects, so it is an **S5b-only sensor/sharpness-calibration exception**. Evidence: Tier 0 preview → Tier 1 design-split probe (`splits/acdc_design.txt`, never val, never a stage row; `src/analysis/blur_probe.py` → tracked `results/summary/blur_probe/`) → Tier 2 train only if warranted. **S5b↔S5 is the only clean one-factor comparison.**
- Ranges are deliberately narrow (ACDC has small distant objects): see `photometric.RANGES` and `weather.RANGES`. **S3 ranges are pre-registered before any ACDC evaluation** and must not be tuned from ACDC results; S5 clips fitted values to them.
- Cite prior work by full method/paper name and venue (e.g., "Koschmieder atmospheric scattering [Koschmieder, 1924]", "rain rendering [Garg & Nayar, TOG 2006]").

## Work Guidance

- Run from the project root.
- S2 (frozen): `python src/synth/build_dataset.py --stage s2 [--limit N]`, inspect with `python src/synth/inspect_s2.py --dataset-name bdd_s2 [--n 20]`.
- S3: `python src/synth/build_s3.py [--limit N] [--jobs 8]`, inspect with `python src/synth/inspect_s3.py --dataset-name bdd_s3 [--n 16]`.
- Generated artifacts live under `data/yolo/bdd_<stage>/` (gitignored); manifests and `bdd_s3_conditions.csv` under `splits/`; configs under `configs/`.
- S5 (implemented): `python src/synth/calibrate.py` → `results/calibration/synth_stats.json`, then `python src/synth/build_s5.py --jobs 8` and `python src/synth/inspect_s5.py --dataset-name bdd_s5`. `synth.physics` is registered in `stage_common.MODULE_BY_STAGE`.
- S5b: `python src/synth/blur_calibrate.py` → `results/calibration/blur_stats.json` (Tier 0 gate), then `python src/analysis/blur_probe.py` (Tier 1 design-split probe → `results/summary/blur_probe/`), and only if warranted `python src/synth/build_s5b.py --jobs 8` + `python src/synth/inspect_s5b.py --dataset-name bdd_s5b` + train/eval. `synth.blur` is registered in `stage_common.MODULE_BY_STAGE`.

## Verification

- S2 (frozen): `python src/synth/build_dataset.py --stage s2 --limit 100` then `python src/synth/inspect_s2.py --dataset-name bdd_s2_smoke` pass (count matches, labels byte-identical, no all-black/white, each synthetic differs from its source).
- S3: `python src/synth/build_s3.py --limit 100 --jobs 4` then `python src/synth/inspect_s3.py --dataset-name bdd_s3_smoke` pass (balanced conditions, labels byte-identical, no degenerate image unless the source is exempt, each synthetic differs from its source).
- S3 full: `python src/synth/inspect_s3.py --dataset-name bdd_s3` passes; conditions are exactly 1,250 each; `splits/bdd_s3_train.txt` has 10,000 entries = 5,000 `bdd_src` + 5,000 `bdd_s3`; `configs/bdd_s3.yaml` has `nc: 6` and `val` = `splits/bdd_src_val.txt`; `data/yolo/bdd_s3` labels mirror images 1:1.
- Re-running a build yields identical image hashes (determinism).
- `splits/bdd_src_A_clear.txt` and `bdd_src_B_source.txt` are disjoint, each 5,000, and their union is exactly `bdd_src_train.txt`.
- S5b: `python src/synth/blur_calibrate.py` writes `results/calibration/blur_stats.json` with per-condition fitted strengths, the Tier 0 gate verdict, and a closed-loop sharpness ratio; `python src/synth/build_s5b.py --limit 100 --jobs 4` then `python src/synth/inspect_s5b.py --dataset-name bdd_s5b_smoke` pass. Full build: `inspect_s5b.py` PASS, labels byte-identical, condition parity with S3, sharpness below S5 (blur applied) and starved of frequency energy, `configs/bdd_s5b.yaml` `nc: 6`; `python src/analysis/blur_probe.py` writes only under `results/summary/blur_probe/`.

## Child DOX Index

- None.
