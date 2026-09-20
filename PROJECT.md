# PROJECT.md — Master Project Record

> Single source of truth for the final-year project. Update this file whenever a
> decision, fact, result, or plan changes. It is written to feed directly into the
> final paper and the DOX contracts (`AGENTS.md` files) remain the binding work rules.
> For a session-resumption summary, see `HANDOFF.md`.
>
> **Sync policy:** facts, results, and decisions are updated **immediately**;
> hypotheses, predictions, and method targeting are updated only at **defined evidence
> gates** and marked "revised from pre-registered". No stale number, path, or claim is
> left in place.

- **Status:** Baseline ladder **S0 / T1 / T1aug / S1 complete and locked** (official split, 1 seed; relabeled from B0/B1/B1aug/B2 on 2026-09-17). The project follows the agreed **S0–S6 comparative-study** design (§5): all stages train on clear BDD and are evaluated on real ACDC, with **S1 (Ultralytics defaults) as the anchor**. **S0/S1/T1/T1aug locked; S2 0.2833, S3 0.2741 (official mAP@50). S3 ties S2 on 5-fold (0.2947 vs 0.2933) and is the best BDD-trained stage on snow; night/fog expose the limits of hand-set parameters. S5 (calibrated physics) is next.**
- **Last updated:** 2026-09-20 (S5 design pre-registered + documented; S3 results recorded)
- **Owner:** student
- **Hardware:** RTX 3050 Laptop, 6 GB VRAM; Python 3.12 `.venv`; PyTorch 2.6.0+cu124; Ultralytics.
- **Raw data:** `datasets/` (read-only).
- **Reference material:** 3rd-year mini-project report `Mini_Project_G-1_Final.pdf` (root); pretrained weights `yolov8n.pt` (root, auto-downloaded by Ultralytics).

---

## 1. Overview and goal

Build and evaluate a **camera-only object detector** that stays robust under **fog, rain, night, and snow**, using **BDD100K** as the clear-weather source domain and **ACDC** as the real adverse-weather target. The end goal is a **publishable paper** with a clear, defensible contribution — not just "we trained YOLO."

The baseline ladder (§9) quantifies the domain gap and localizes where a method must gain. The current direction is a **systematic comparative study of augmentation and domain-adaptation strategies (S0–S6, §5)**: clear BDD is transformed into synthetic adverse-weather BDD, a single detector is trained per strategy, and all strategies are scored per weather on real ACDC. The scientific question is whether synthetic adverse weather generated from clear BDD improves detection on real adverse weather. Whether this becomes a novel paper or stands as a B.Tech comparative study is deferred until the per-condition results exist.

## 2. Locked decisions

| Decision | Choice | Date |
|---|---|---|
| Source domain | BDD100K **clear/daytime** | 2026-09-11 |
| Target domain | ACDC (fog, rain, night, snow) | 2026-09-11 |
| Adaptation setting | Supervised fine-tune **+** zero-shot DG (LOO and ablations are future work, not yet configured) | 2026-09-11 |
| Models | YOLOv8n for all experiments (locked) | 2026-09-11 |
| Class scheme | Unified 6-class (`scripts/class_map.py`) | 2026-09-11 |
| Task | Object **detection only** (ACDC has no segmentation labels) | 2026-09-11 |
| Source subsample | 10,000 train / 2,000 val | 2026-09-11 |
| Source definition | `weather == "clear"` AND `timeofday == "daytime"` (12,454 pool) | 2026-09-11 |
| Data storage | Copy tree under `data/yolo/` (each image copied once) | 2026-09-11 |
| Split policy | Stratified, `seed=42`; ACDC weather-stratified 5-fold | 2026-09-11 |
| Primary evaluation | Official ACDC split; 1 training per config; seed 42 | 2026-09-14 |
| Study framing | S0–S6 comparative study; **S1 = Ultralytics defaults = anchor** | 2026-09-17 |
| Experiment IDs | Relabeled `B0→S0`, `B2→S1`, `B1→T1`, `B1aug→T1aug` (weights preserved, eval re-run) | 2026-09-17 |
| Synthetic data budget | Fixed **10k = 5k clear + 5k synthetic**, seed 42 (compute-matched to S1) | 2026-09-17 |
| Synthesis implementation | **Offline** pre-generated datasets; S4 (FDA) remains online | 2026-09-17 |
| Leakage control | 400-image ACDC **design split**; official val scored once | 2026-09-17 |
| S5 definition | Physics-structured synthesis **calibrated to measured ACDC-train statistics** | 2026-09-17 |
| S3 condition allocation | Balanced **1,250 each** fog/rain/snow/night; exactly one condition per image | 2026-09-19 |
| S3 weather model | Hand-set, geometry-preserving: constant-transmission Koschmieder fog, directional rain streaks, falling snow particles (no accumulation), illumination night; uniform depth; **no blur** | 2026-09-19 |
| Per-experiment eval set | Every stage reports the **S2-matched set**: official ACDC val (primary) + ACDC 5-fold (supplementary) + in-domain BDD (forgetting), all single seed 42 | 2026-09-19 |
| S5 scope | **Parameter-only**: identical operators to S3, only parameter values change (hand-set → fitted). Structural changes (row-depth fog, local night, blur) are S6c | 2026-09-20 |
| S5 calibration source | **ACDC-train unlabeled pool statistics only** (`splits/acdc_pool_unlabeled.txt`, 1,200 = 300/condition); never official val, never the design split. Setting = **unlabeled domain adaptation** | 2026-09-20 |
| S5 rendering + sampling | Render via S3's `weather.apply` (parity by construction); **empirical/bounded sampling** from the fitted distributions; fitted values **clipped to S3's pre-registered ranges and every clip logged** | 2026-09-20 |
| S5 vignette | Fit to the radial luminance profile, **clip to S3's `[0.10, 0.30]`**, report the raw fitted distribution separately | 2026-09-20 |
| S5b blur ablation | Ancillary one-factor ablation over S5: **condition-specific blur** — rain directional motion (aligned to streak slant), fog/snow isotropic defocus, night none. Base = S5; S5b↔S5 is the only clean comparison | 2026-09-20 |
| S5b blur calibration | Strength **calibrated to the ACDC-train pool** by sharpness attenuation (new estimator); blur parameters have **their own pre-registered ranges** (no S3 equivalent) and are clipped+logged | 2026-09-20 |
| S5b boundary | S5b is an **S5b-only sensor/sharpness-calibration exception** (S5 excluded frequency/gradient matching); S5 stays parameter-only. Blur is not folded into S5 | 2026-09-20 |

**Model:** YOLOv8n only, locked for the whole matrix — best compute/coverage tradeoff on 6 GB (S0: ~1.9 min/epoch, 3.9 GB at batch 32). Revisit scale only if results are inconclusive.

## 3. Verified dataset facts

### ACDC
- 4,006 images, PNG; **train 1,600 (400/weather), val 406 (fog 100, night 106, rain 100, snow 100), test 2,000 (no GT)**.
- Image path: `datasets/acdc/images/<file_name>`, `file_name = <weather>/<split>/<seq>/<frame>.png`.
- Detection labels: COCO JSON in `datasets/acdc/labels/<weather>/`.
- Categories/IDs: `24 person, 25 rider, 26 car, 27 truck, 28 bus, 31 train, 32 motorcycle, 33 bicycle`.
- **Quirk:** `instancesonly_rain_train_gt_detection.json` is the **combined all-weather** train file (1,600 imgs); `fog`/`night`/`snow` train files are per-weather (400 each). `val_gt` and `test_image_info` are per-weather and correct.

### BDD100K
- Detection images: `datasets/bdd100k/bdd100k/bdd100k/images/100k/{train,val,test}/` (**nested** under `trainA/trainB/testA/testB`). `10k/` is a separate subset.
- Detection labels: `datasets/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_{train,val}.json`; train = **69,863 records**; keys `name, attributes, timestamp, labels`.
- `attributes` keys: `weather, scene, timeofday`.
- Weather counts (train): `clear 37,344`, `overcast 8,770`, `undefined 8,119`, `snowy 5,549`, `rainy 5,070`, `partly cloudy 4,881`, `foggy 130`.
- Labels carry `box2d` (objects) and `poly2d` (`lane`, `drivable area`).
- Native object categories: `person, rider, car, truck, bus, train, motorcycle/bike, motor, traffic light, traffic sign`.
- Segmentation: `bdd100k_seg/.../seg/{images,labels,color_labels}/{train,val,test}` (not used — detection only).

## 4. Unified class scheme (6 classes)

```
0 person   1 rider   2 car   3 truck   4 bus   5 bicycle (motorcycle folded in)
```
- ACDC: `24→0, 25→1, 26→2, 27→3, 28→4, 31→DROP, 32→5, 33→5`.
- BDD: `person→0, rider→1, car→2, truck→3, bus→4, bike→5, motor→5`; drop `train, traffic light, traffic sign, lane, drivable area`.
- Source of truth: `scripts/class_map.py`. IDs must stay stable once training begins.
- **Deviation from standard ACDC evaluation:** official ACDC keeps `motorcycle` and `bicycle` separate. Folding them is required for a unified BDD+ACDC scheme; report a separate-class ablation to answer reviewer questions.
- **Pretrained head:** COCO has no `rider` class, so Ultralytics remaps 5/6 head rows by name (`person, car, truck, bus, bicycle`) and `rider` is randomly initialized. It trains from scratch and lags (0.249 mAP50 on BDD vs 0.654 car). Document in methods; expect method gains to be largest there.

## 5. Experimental design — S0–S6 comparative study

**Framing.** A systematic comparison of augmentation and domain-adaptation strategies for YOLO detection under adverse weather. Every stage trains on **clear BDD100K** (source) and is evaluated on **real ACDC** (target) per weather. **S1 (Ultralytics defaults) is the anchor**; every other stage is S1 plus one change. The study is a comparative benchmark; the decision to pursue a paper novelty is deferred until the per-condition results exist.

| ID | Training data (10k unless noted) | Purpose | Status |
|---|---|---|---|
| S0 | clear BDD, no aug | floor | **done** |
| S1 | clear BDD + Ultralytics defaults | **anchor** | **done** |
| S2 | S1 + generic photometric degradation (offline dataset; blur deferred) | non-weather sensor degradation | **done** (mAP@50 0.2833) |
| S3 | S1 + simple weather-specific transforms (offline dataset) | fast weather simulation | **done** (mAP@50 0.2741) |
| S4 | S1 + Fourier Domain Adaptation (online), ACDC-train style (unlabeled) | appearance adaptation | planned (prior global-FDA run failed) |
| S5 | S1 + physics-structured, ACDC-calibrated synthesis (offline dataset) | principled weather simulation | planned |
| S6a | best fixed combination of S2–S5 | combination | planned |
| S6b | S6a + condition-aware selection | condition-aware policy | planned |
| S6c | per-condition policy from design-split inspection | custom policy | optional |
| T1 / T1aug | ACDC labels, no aug / + Ultralytics defaults | in-domain reference / **ceiling** | **done** |

**Naming (relabeled 2026-09-17).** The S-ladder is the study's ID scheme: `B0→S0`, `B2→S1`, `B1→T1`, `B1aug→T1aug`. Weights were preserved and `eval.py` re-run under new `--name` (because `aggregate.py` groups by the JSON `"run"` field). Physical dirs are now `S0`, `S1`, `T1_official`/`T1_foldN`, `T1aug_official`/`T1aug_foldN`.

**Dataset construction (S2 / S3 / S5 / S6).**
- **Offline** pre-generated datasets under `data/yolo/bdd_<stage>/`, so the training command is identical across stages — **only the training dataset differs**.
- Fixed **10,000-image budget** (compute-matched to S1): **5,000 clear + 5,000 synthetic**, seed 42. The synthetic half is generated one-per-source from the remaining 5,000 clear images; labels are byte-copied (geometry-preserving).
- In-training `val` = `bdd_src_val.txt` (clear) for every stage, so `best.pt` selection is consistent and never touches ACDC.
- The 1:1 clear:synthetic ratio is pre-registered; a ratio sweep is a later ablation.

**S2 (finalized 2026-09-17).** *A/B split of S1's 10k:* **A = 5k clear**, **B = 5k clear**. S1 = A + B (already trained) is therefore the **exact control — no S1 rerun**. S2 trains on **A (clear, referenced from `bdd_src`) + B (degraded)**, so both stages see the same scenes/objects/labels/count and only the appearance of the B half differs. Operators (offline, geometry-preserving): **brightness ×0.7–1.3, contrast ×0.7–1.2, gamma 0.8–1.4, saturation ×0.6–1.1, additive Gaussian noise σ 0–10**; **blur excluded** from S2, S3, and S5 (deferred beyond the S3/S5 comparison, to S6c; so S2 is purely *photometric*, not "generic image degradation"). Each B image gets a **random 1–3 ops (without replacement)** in fixed order brightness→contrast→gamma→saturation→noise (noise last), seeded by `hash(filename)+42` (order-independent). Every op/param is logged to `synthesis_log.csv`; labels are copied byte-identically (`shutil.copy2`). Outputs: `data/yolo/bdd_s2/`, `splits/bdd_s2_train.txt`, `configs/bdd_s2.yaml` (val = `bdd_src_val.txt`); the fixed halves are recorded as `splits/bdd_src_A_clear.txt` / `splits/bdd_src_B_source.txt` (seed 42). S2 is **zero-shot DG** (no ACDC at all).

**S2 status (2026-09-17).** Implemented in `src/synth/{common,photometric,build_dataset,inspect_s2}.py`; A/B halves at `splits/bdd_src_A_clear.txt` / `bdd_src_B_source.txt` (5,000 each, disjoint, union = 10k). Generated `data/yolo/bdd_s2/` = 5,000 degraded B images + 5,000 referenced clear (manifest `splits/bdd_s2_train.txt`, config `configs/bdd_s2.yaml`). Inspector PASS: labels 5000/5000 byte-identical, 0 synthetic equal to source, 1 already-degenerate source exempt (guard resampled 1 image). Full generation ~2–3 min (8 workers); determinism verified.

**S3 (finalized 2026-09-19).** Same A/B harness: **A (5k clear) + B weather-transformed**, one condition per image, **balanced 1,250 each of fog/rain/snow/night** (seeded assignment; mirrored to ACDC's balanced train/val), so **S1 is again the exact control**. Operators are hand-set and geometry-preserving — no ACDC data, no mixed conditions, no local light sources, no snow accumulation, uniform-depth, **no blur** (blur is **excluded from both S2 and S3**; deferred beyond the S3/S5 comparison, to S6c):
- **fog** — constant-transmission Koschmieder `I = J·t + A·(1−t)` [Koschmieder, 1924]; `t ∈ [0.35, 0.70]`, airlight `A ∈ [180, 235]` with per-channel jitter ≤ 8, desaturation `∈ [0.7, 1.0]`.
- **rain** — directional streak overlay [Garg & Nayar, TOG 2006]: 150–500 streaks per 640² scaled by pixel area, length 10–30 px, width 1–2 px, slant 70–85° from horizontal, alpha 0.3–0.5, contrast ×0.8–0.95.
- **snow** — falling particles only: density 0.02–0.07 of pixels, radius 2–6 px, alpha 0.5–0.7, brightness ×1.0–1.15, contrast ×0.85–1.0, desaturation ×0.8–1.0.
- **night** — illumination model: brightness ×0.35–0.60, gamma 1.0–1.4 (crushes shadows; γ<1 would lift them into a "dim daytime" artifact), warm/cool tint ±5 (BGR-correct), vignette 0.1–0.3.

Outputs: `data/yolo/bdd_s3/` (5,000 weather images; the 5,000 clear A images referenced from `bdd_src`), `splits/bdd_s3_train.txt` (10k), `splits/bdd_s3_conditions.csv`, `configs/bdd_s3.yaml` (val = `bdd_src_val.txt`). Logging: `index.json` + `synthesis_log.csv` record condition and every parameter. S3 is **zero-shot DG** (no ACDC).

**S3 status (2026-09-19).** Implemented per-experiment (S2 frozen) in `src/synth/{stage_common,weather,build_s3,inspect_s3}.py`. Full generation ~2.5 min (8 workers); inspector PASS: 5,000/5,000 labels byte-identical, conditions exactly 1,250 each, each synthetic differs from its source, 5 low-signal sources exempt (documented). Object-visibility (GT-box local-contrast retention, median): fog 0.52, rain 0.90, snow 1.04, night 0.40; night is the most destructive (11.7% of boxes below 0.3), as expected for the hardest condition. Determinism verified (identical image hashes on rebuild). Paper-facing preview grids at `results/summary/figures/synth_preview_bdd_s3_{fog,rain,snow,night}.png`. **Trained + evaluated 2026-09-20** (80 epochs, best @65; full S2-matched eval): official mAP@50 **0.2741**, 5-fold **0.2947 ± 0.0119**, in-domain 0.4799; see the S3 result in §9.

**S3 vs S5 (must not collapse into each other).** S3 uses hand-set parameters; S5 fits the same physical model families (Koschmieder fog / dark-channel transmission, rain streaks, snow particles, night illumination) to **measured ACDC-train statistics** — this **parameter-source** difference is the principled separation and the basis of S5's potential novelty. S3 parameters are **pre-registered before any ACDC evaluation** and must not be tuned from ACDC results.

**S5 (finalized 2026-09-20).** *Calibrated physics, parameter-only.* Same A/B harness, same source list and **same condition-per-source as S3** (`splits/bdd_s3_conditions.csv`, hash-guarded), so S5 vs S3 is a **paired** per-image comparison in which only parameter values differ. S5 is **unlabeled domain adaptation**: it reads ACDC-train pool images (no labels) only to measure statistics. `src/synth/calibrate.py` estimates physically interpretable parameters per image, rejects ±3 MAD outliers, pools them, and writes a versioned `results/analysis/synth_stats.json` (pool-manifest hash + code commit); the builder never recomputes. Estimators: **fog** airlight `A` (dark-channel-masked brightest 0.1%), transmission `t` (dark channel), desaturation; **rain** slant (gradient orientation near-vertical), length (run-length), density (adaptive gradient threshold), contrast; **snow** radius (top-hat blob components), density (small-blob count — ground accumulation cannot be represented and is a stated limitation), alpha/brightness/contrast/desaturation; **night** brightness, gamma (percentile tone fit), per-channel tint (BGR), vignette (radial fit, clipped to `[0.10, 0.30]`). Ratio statistics use the clear BDD A-half as reference (stated caveat: conflates weather with BDD↔ACDC sensor/domain tone). `src/synth/physics.py` samples parameters **empirically/bounded** (no Gaussian MAD) and **renders by delegating to S3's `weather.apply`** (structural parity by construction); fitted values are **clipped to S3's pre-registered ranges and every clip is logged**. Validation: a **calibration self-test** (render with known parameters → estimator recovery) plus a **closed-loop QA** comparing generated statistics against both the fitted targets and the real pool, plus label invariance, condition counts, determinism, object visibility, previews. Pre-registered expectation: gains in **rain/snow**; **fog/night limited by model structure** (uniform depth, no local illumination) — if unchanged, that is a finding motivating S6c, not a failed run. Explicitly excluded from S5 (structural items → S6c): row-depth fog, local night illumination, **blur (tested separately as S5b, below)**, per-image fitting, gradient/noise-energy matching, calibration sample-size ablation. Comparisons to report: S5↔S3 (calibration + target access), **S5↔S4 (controlled access — the headline)**, S5↔S2 (structure vs photometric).

**S5b (blur ablation, planned 2026-09-20).** Blur is deliberately **not** folded into S5 (that would make S5↔S3 change two things and muddy the calibration claim). Instead it is an **ancillary one-factor ablation over the calibrated base**: **S5b = S5 rendering + condition-specific blur**, everything else identical (same A/B, same source list and condition-per-source as S3/S5, labels byte-copied, seed 42, 80 epochs, S2-matched eval). Blur mapping: **rain → directional motion blur** aligned with the sampled streak slant; **fog → isotropic defocus**; **snow → mild isotropic defocus**; **night → none** (illumination-limited; blur would only hurt the recall S5 tries to recover). Strength is **calibrated to the ACDC-train pool**: a new estimator measures per-condition sharpness attenuation (Laplacian-variance / high-frequency energy ratio) against the clear BDD A-half reference, and the σ (defocus) / length (motion) that reproduces it is fitted in `calibrate.py`; a closed-loop check re-applies the fitted blur to clear images and verifies the ratio. Because blur has **no S3 equivalent**, it uses **its own pre-registered parameter ranges**, clipped and logged. This deliberately crosses the "no frequency matching" line that S5 respects, so S5b is recorded as an **S5b-only sensor/sharpness-calibration exception**. Evidence path: (Tier 0) calibrated-blur preview + object-visibility; (Tier 1) evaluate the trained S5 model on the **400-image design split** with test-time blur → per-condition sensitivity curve (`results/analysis/blur_probe/`; never official val, never a stage row); (Tier 2) build `data/yolo/s5b` and train only if warranted. Implemented append-only (`src/synth/blur.py` + `build_s5b.py`) composing S5's sampler → `weather.apply` → blur; S5's files untouched. **S5b↔S5 is the only clean one-factor comparison** (S5b↔S3/S1 differ by more than one factor). The outcome feeds **S6a (fixed combination)** and **S6b (condition-aware)**.

**S4 (FDA).** Faithful reference `low_freq_mutate` [Yang & Soatto, CVPR 2020], applied online; target pool = `splits/acdc_pool_unlabeled.txt` (ACDC official train minus the design split). β ∈ **{0.05, 0.10}** *(provisional — the `PLAN.md` review recommends restoring β=0.01; to be decided when S4 is built)*, headline = best β, reported honestly even if below S1 (the prior global-FDA run gave 0.273 at β=0.01 and 0.255 at β=0.05 vs S1 0.269).

**Implementation (append-only per experiment).** Each stage has its own frozen generator, so code is never overwritten once a result is locked: S2 = `photometric.py` + `build_dataset.py` + `inspect_s2.py` (frozen); S3 = `weather.py` + `build_s3.py` + `inspect_s3.py` on the shared `stage_common.py` harness (frozen). S5 adds `calibrate.py` + `physics.py` + `build_s5.py` + `inspect_s5.py` on the same harness; S6 later. `src/synth/AGENTS.md` is the binding contract.
- `src/aug/fda.py` + a minimal trainer hook in `src/train.py` for S4 only; fidelity check `max|ours-ref| = 0.0000`.
- ~~Clean `src/data/build_splits.py` / `write_configs.py`~~ — **done 2026-09-17**: they now emit `bdd_src`, `acdc_cv5`, `acdc_official`, `acdc_perweather`, `acdc_design`, `acdc_pool_unlabeled`; official val kept at 406 and all kept manifests hash-verified.

**Run order:** lock splits → confirm S0/S1 → S2 → S3 → S5 → S4 → S6a/S6b → optional S6c.

## 6. Evaluation protocol

**Terminology (use precisely in the paper):**
- **Zero-shot generalization** — no ACDC data at all (S0, S1).
- **Unlabeled domain adaptation** — ACDC images seen during training, labels never used (future methods).
- **Supervised fine-tuning** — ACDC labels used (future).
- **Leave-one-domain-out generalization** — train on 3 weather styles, test on the held-out 4th (future).

**Metrics and statistics (locked 2026-09-14):**
- Metrics: mAP@50, mAP@50-95, Precision, Recall, F1, **per-weather**, **per-class**, FPS.
- **Primary protocol = the official ACDC split, single training per config, single seed (42).** Source = BDD clear/daytime (10k/2k); target test = **ACDC official val (406)**. Applies to all baselines and to future methods.
- **Rationale (state in the paper):** the 406 official val is the published, citable benchmark; a single seed is used consistently across methods **and** baselines (no asymmetry).
- **Leakage control (locked 2026-09-17).** The 406-image official val is the **only final scorer** and is scored once per stage. A **design split** of 400 images (100/condition, stratified from `acdc_official_train.txt`, seed 42; `splits/acdc_design.txt`) is reserved for inspecting failure modes and choosing S6 policies; it is **never trained on, never used as an S4 style source, and never scored**. The S4 target pool is `splits/acdc_pool_unlabeled.txt` (official train minus design = 1,200).
- **Uncertainty:** none from seeds/folds under the primary protocol; variability is reported via the **per-weather and per-class** breakdown. The existing 5-fold results for S0/T1/T1aug/S1 are kept as supplementary only.
- **Per-experiment eval set (locked 2026-09-19).** Every stage S0–S6 reports the **S2-matched eval set** so aggregate/visualize tables stay comparable: (a) **official ACDC val** (406) with per-weather breakdown, `--name <ID>_acdc_official --per-weather` — **primary**; (b) **ACDC 5-fold** (`acdc_cv5_fold0..4`) with per-weather, `--name <ID>_acdc_foldN --per-weather` — **supplementary**; (c) **in-domain BDD val** (`bdd_src_val`, no `--per-weather`), `--name <ID>_in_domain` — forgetting check. Applies to S3, S4, S5, S6a/b/c.
- **Limitations to state:** single seed, single official split (no CV error bars). The 2,000 unlabeled ACDC test images are unused.
- Qualitative: before/after detection panels per condition.
- Comparison targets: S0 (floor), S1 (anchor to beat), T1/T1aug (in-domain reference and ceiling).

## 7. Compute and budget

- `imgsz=640`, AMP on; YOLOv8n for all runs (§2).
- ~1.3 h per 10k-image / 40-epoch run (S0); ~2.5–2.9 h at the 80-epoch aug budget (T1aug, S1). ACDC fine-tunes are minutes.
- **Observed (S0):** ~3.0 it/s train, ~1:46/epoch + ~7 s val ≈ **1.9 min/epoch**; 40 epochs in 1.26 h; ~3.9 GB VRAM at batch 32.
- **Observed (T1/T1aug, ACDC ~1.6k imgs):** ~13 min / 40 epochs, ~26 min / 80 epochs; 6 runs each (official + 5 folds).
- **Observed (S1, 10k imgs / 80 epochs):** 2.70 h.
- 6 GB constraint: keep batch sizes modest, avoid YOLOv8m unless needed.
- **Uniform training schedule (all configs):** `optimizer=auto` (AdamW, `lr0=0.001`), `cos_lr=True`, `patience=30`, `batch=32`, `imgsz=640`, AMP on, `seed=42`; **epochs = 40 for no-aug (S0, T1), 80 for augmented (S1, T1aug, S2–S6)**. Only the training data/augmentation varies across configs — no per-config hyperparameter tuning, to keep the comparison controlled. Batch 32 measured at 4.0 GB VRAM.
- **Projected (S0–S6 study, 1 seed, official split):** S2/S3/S5/S6a/S6b = 5 × ~2.7 h ≈ **13.5 GPU-h**; S4 FDA × 2 β ≈ **~7 GPU-h**; optional S6c ≈ 2.7 h. Plus CPU-only dataset generation (~10–30 min/stage) and per-weather evaluations. Fits comfortably on the single 6 GB GPU.

## 8. Repository structure and pipeline

```
PROJECT.md            # this file
AGENTS.md             # DOX rail
configs/              # generated Ultralytics dataset YAMLs
src/
  data/               # BDD/ACDC -> YOLO conversion, split builders
  synth/              # offline synthesis: S2 + S3 implemented (per-stage, frozen); S5 designed; S5b blur ablation planned; S6 planned
  aug/                # (planned) S4 Fourier Domain Adaptation online hook
  train.py  eval.py  aggregate.py  visualize.py  common.py
scripts/              # class_map.py, inspect_dataset.py
data/                 # generated YOLO datasets (never inside datasets/)
paper/                # manuscript sources + literature/ verified knowledge hub
datasets/             # raw source (read-only)
results/
  experiments/<EXP>/{train,eval,figures}
  summary/            # aggregate summary.json, per_class.csv/md, figures
  splits/             # split distribution logs
```

Generated outputs (`data/`, `results/`, `paper/`, `configs/`, `src/`) must never be written inside `datasets/`.

**Per-experiment grouping:** pass `--exp <ID>` to `train.py` and `eval.py`; all artifacts for that experiment land under `results/experiments/<ID>/`. `aggregate.py` and `visualize.py` scan the experiment tree.

**Pipeline run order:** `convert_acdc.py` → `convert_bdd.py` → `build_splits.py` → `write_configs.py` → `materialize.py` → `prune_labels.py` → `train.py --exp` → `eval.py --exp` → `aggregate.py` → `visualize.py`. Manifests under `splits/` are the source of truth for every split; `data/yolo/` holds the copied images and generated labels, kept in exact one-to-one correspondence.

## 9. Results tracker

Official experiment rows (1 seed 42, official ACDC val). **S0 = floor**, **S1 = anchor to beat**, **T1 = in-domain reference**, **T1aug = ceiling** (relabeled 2026-09-17, §5).

Study rows (official ACDC val; fill as runs complete):

| ID | Config | mAP@50 | mAP@50-95 | P | R | Seed | Status |
|---|---|---|---|---|---|---|---|
| S2 | BDD + photometric synthesis (offline) | **0.283** | **0.165** | 0.434 | 0.272 | 42 | **done** |
| S3 | BDD + simple weather synthesis (offline) | **0.274** | 0.157 | 0.532 | 0.244 | 42 | **done** |
| S4 | BDD + FDA (online, best β) → ACDC | | | | | 42 | planned |
| S5 | BDD + calibrated physics synthesis (offline) | | | | | 42 | **designed** (plan locked; awaiting build) |
| S5b | S5 + pool-calibrated condition-specific blur (ancillary ablation) | | | | | 42 | planned |
| S6a | best fixed combination | | | | | 42 | planned |
| S6b | S6a + condition-aware selection | | | | | 42 | planned |
| S6c | per-condition policy (optional) | | | | | 42 | optional |

**S2 result — photometric degradation (zero-shot), 2026-09-17.** S1 is the exact control (S2 = A clear + B degraded; S1 = A + B clear). Official val: mAP@50 0.2690 → **0.2833** (+0.0143), mAP@50-95 0.1559 → **0.1650** (+0.0091), P 0.461 → 0.434, R 0.266 → 0.272. 5-fold 0.2863 ± 0.0138 → 0.2933 ± 0.0161 (+0.0070, within spread). In-domain BDD 0.4911 → 0.4920 (no forgetting). S2 captures ~28% of the S1→T1aug headroom. **Per weather (official mAP@50):** fog 0.491→0.485, night 0.193→0.179, **rain 0.244→0.256**, snow 0.284→0.282. **Per class:** person 0.270→0.303, rider 0.071→0.109, bus 0.169→0.206, truck 0.289→0.301, car flat, bicycle 0.113→0.078. **Reading:** first stage above the anchor, driven by rain + rare classes; fog/night/snow flat → generic photometrics do not model structured weather (motivates S3/S5). **Caveat:** single seed; the 5-fold gain is within noise, so directionally consistent but not yet significant. Full paper-facing write-up in `paper/results_notes.md`.

**S3 result — simple weather synthesis (zero-shot), 2026-09-20.** S1 is the exact control (S3 = A clear + B weather; S1 = A + B clear); schedule identical to S1/S2 (80 epochs, best @65, no early stop). Official val: mAP@50 0.2690 → **0.2741** (+0.0051), mAP@50-95 0.1559 → 0.1573 (+0.0014), P 0.461 → **0.532**, R 0.266 → 0.244. 5-fold 0.2863 ± 0.0124 → **0.2947 ± 0.0119** (+0.0084; mAP@50-95 0.1609 → 0.1665 ± 0.0080). In-domain BDD 0.4911 → 0.4799 (−0.0112, mild forgetting). S3 captures ~10% of the S1→T1aug headroom, **below S2 (0.2833) on the official split but statistically tied with S2 on 5-fold (S2 0.2933 ± 0.0144)**. **Per weather (official mAP@50):** fog 0.491→0.489, night 0.193→0.180, **rain 0.244→0.264**, **snow 0.284→0.300**. **Per weather 5-fold mAP@50:** fog 0.476→0.466, night 0.196→0.182, **rain 0.299→0.306**, **snow 0.300→0.306**. **Per class (official):** rider 0.071→**0.117** (best of S1/S2), truck 0.289→0.304, car 0.701→0.693, person 0.270→0.273, bus 0.169→0.167, bicycle 0.113→0.091. **Reading:** hand-set weather structure clears the anchor and ties S2 on 5-fold, but does **not** beat generic photometrics on the official primary metric. The gains are **condition-specific and physical**: **snow is the headline — S3 gives the best BDD-trained snow result (0.300 official, 0.306 5-fold) and closes ~49% of the S1→ceiling snow gap (mAP@50-95 snow 0.159→0.164), with snow recall rising 0.232→0.319 at higher precision**; rain improves too (+0.019 official, +0.008 5-fold). Night (global dimming, no local sources) and fog (constant transmission) regress, and person/bus drop — the overall recall loss (0.266→0.244) lives there, not in rain/snow. **Caveat:** single seed; the S2↔S3 official gap (~0.010) is inside the fold spread (±0.012–0.014), so this is "no advantage over S2", not "harm". Full paper-facing write-up in `paper/results_notes.md`. **Motivates S5 (calibration to measured ACDC-train statistics).**

| ID | Config | mAP@50 | mAP@50-95 | P | R | Seed | Status |
|---|---|---|---|---|---|---|---|
| S0_in | BDD no-aug → BDD val (in-domain) | 0.376 | 0.207 | 0.511 | 0.374 | 42 | done |
| S0_zs | BDD no-aug → ACDC 5-fold (zero-shot) | 0.213 ± 0.011 | 0.113 ± 0.004 | 0.370 ± 0.053 | 0.230 ± 0.015 | 42 | done |
| S0_zs | BDD no-aug → ACDC official val (zero-shot) | 0.201 | 0.108 | 0.401 | 0.202 | 42 | done |
| T1 | ACDC no-aug → ACDC 5-fold | 0.254 ± 0.020 | 0.136 ± 0.010 | 0.419 ± 0.083 | 0.265 ± 0.016 | 42 | done |
| T1 | ACDC no-aug → ACDC official val | 0.216 | 0.116 | 0.339 | 0.229 | 42 | done |
| T1aug | ACDC + default aug → ACDC 5-fold (**ceiling**) | 0.383 ± 0.021 | 0.221 ± 0.012 | 0.580 ± 0.026 | 0.353 ± 0.024 | 42 | done |
| T1aug | ACDC + default aug → ACDC official val | 0.320 | 0.196 | 0.539 | 0.289 | 42 | done |
| S1_in | BDD + standard aug → BDD val (in-domain) | 0.491 | 0.287 | 0.678 | 0.445 | 42 | done |
| S1_zs | BDD + standard aug → ACDC 5-fold (zero-shot) | 0.286 ± 0.014 | 0.161 ± 0.007 | 0.517 ± 0.042 | 0.274 ± 0.019 | 42 | done |
| S1_zs | BDD + standard aug → ACDC official val (zero-shot) | 0.269 | 0.156 | 0.461 | 0.266 | 42 | done |

### Baseline ladder (mAP@50)

| Run | Recipe | Official (primary) | 5-fold (supplementary) | In-domain |
|---|---|---|---|---|
| **S0** | BDD no-aug — **floor** | **0.201** | 0.213 ± 0.011 | 0.376 |
| T1 | ACDC labels, no aug | 0.216 | 0.254 ± 0.020 | — |
| **T1aug** | ACDC + default aug — **ceiling** | **0.320** | 0.383 ± 0.021 | — |
| **S1** | BDD + standard aug | **0.269** | 0.286 ± 0.014 | 0.491 |

**Findings:**
- **Augmentation, not target labels, is the lever.** T1 (labels, no aug) 0.254 ≈ S0 0.213; T1aug (labels + aug) 0.383. The tiny target set needs regularization.
- **S1 (standard aug) closes ~43% of the domain gap** (5-fold 0.213 → 0.286 of the 0.170 to the ceiling) and **beats T1**. Headroom for a method (official, primary): **T1aug − S1 = 0.320 − 0.269 = +0.051** (5-fold supplementary: +0.097).
- **Remaining headroom is localized.** Closed by S1: fog 79%, rain 66%, night 47%, **snow 36%**.
- **Caveat:** the 5-fold ceiling (0.383) exceeds the official-split ceiling (0.320) — different test sets (official val is harder). The **official split is primary**; 5-fold numbers are supplementary. Never claim 5-fold "would be better".

**Per-class 5-fold mAP50 (S0 → S1 → ceiling):**

| class | S0 | S1 | T1aug |
|---|---|---|---|
| person | 0.221 | 0.295 | 0.384 |
| rider | 0.096 | 0.177 | 0.214 |
| car | 0.577 | 0.661 | 0.687 |
| truck | 0.194 | 0.287 | 0.495 |
| bus | 0.112 | 0.159 | 0.340 |
| bicycle | 0.077 | 0.139 | 0.177 |

Per class, **truck 0.287 vs ceiling 0.495** and **bus 0.159 vs 0.340** hold most of the remaining headroom, while `car` is nearly closed (0.661 → 0.687).

**Per-weather 5-fold mAP50 (S0 → S1 → ceiling):**

| | fog | rain | snow | night |
|---|---|---|---|---|
| S0 | 0.373 | 0.216 | 0.249 | 0.138 |
| S1 | 0.476 | 0.299 | 0.300 | 0.196 |
| T1aug | 0.504 | 0.341 | 0.391 | 0.262 |

Night remains the hardest condition; the method should target night/snow and truck/bus.

## 10. Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-09-11 | Unified 6-class scheme; fold motorcycle→bicycle; drop train/lights/signs | ACDC has no lights/signs; cross-dataset consistency |
| 2026-09-11 | ACDC rain train JSON treated as combined all-weather file; filter by prefix | Verified on disk; user-approved |
| 2026-09-11 | Detection only | ACDC has no segmentation labels |
| 2026-09-11 | BDD clear/daytime source; supervised fine-tune + zero-shot DG | Clean normal→adverse gap; matches plan |
| 2026-09-11 | B2 baseline = YOLOv8 defaults | Reviewers compare against defaults |
| 2026-09-11 | Copy-tree storage; manifests are the source of truth | Portable; CV folds avoid 5x image duplication |
| 2026-09-11 | Prune orphan labels after materialize (`prune_labels.py`) | `data/yolo` labels must mirror images exactly (454 orphans removed) |
| 2026-09-11 | Resolve `--project` absolute; eval grouped under `--exp` | Prevent `runs/detect/...` nesting and clutter from relative paths |
| 2026-09-11 | Uniform schedule: `cos_lr=True`, `patience=30`, `batch=32`, `optimizer=auto` | Consistent across all baselines; no per-config tuning keeps comparison controlled |
| 2026-09-11 | `rider` head randomly initialized (no COCO match) | Document in methods; expect gains largest there |
| 2026-09-12 | B0 promoted from `pilot_B0` to official baseline | `args.yaml` matched the locked schedule; re-eval was bit-identical |
| 2026-09-12 | Multi-run configs use per-run experiment IDs | train dirs `<ID>_<run>`, eval `--name <ID>_acdc_<run>` so `aggregate.py` groups them |
| 2026-09-12 | B1 split into `B1` (no-aug) and `B1aug` (default aug) | No-aug isolates domain shift and pairs with B0; aug gives the achievable target-domain ceiling and pairs with B2 |
| 2026-09-14 | **Primary protocol = official split**, 1 training/config, 1 seed (42); baselines reused | Standard benchmark; consistent across methods and baselines. 5-fold kept as supplementary |
| 2026-09-17 | **Baseline-only reset:** removed all method code (`src/aug`, `src/analysis`), method experiments (B3/B4/B5), the target-image cache, LOO/5k/smoke configs and manifests, the stray `weights/yolo26n.pt`, and the `paper/ideas_wsm.md`, `paper/methods_notes.md`, `paper/related_work.md` notes | Retain only the verified B0/B1/B1aug/B2 baselines and reset the method to be redefined; the removed directions (SM-WCFA, global Fourier Domain Adaptation, Direction A / WSM, spectral analysis) are superseded |
| 2026-09-17 | **S0–S6 comparative-study design agreed** (S1 = anchor; S2 photometric, S3 simple weather, S4 FDA, S5 calibrated physics, S6a/b/c) | Peer review (`PLAN.md`): make every method incremental over S1, keep the condition count controlled, and lead with the per-weather comparison |
| 2026-09-17 | **Leakage control:** 400-image ACDC design split (100/condition) from official train; official val scored once; S4 pool = official train − design | `PLAN.md`: never tune the custom policy on the scored benchmark. ACDC test has no GT, so official val is the only scorer |
| 2026-09-17 | **Synthesis implemented offline** into fixed `data/yolo/bdd_<stage>` datasets; S4 FDA stays online | Reproducible/inspectable/checksummable, and the training command is identical across stages so the dataset is the only variable. S4 is an adaptation method, kept faithful to reference FDA |
| 2026-09-17 | **Fixed 10k = 5k clear + 5k synthetic** per stage (seed 42) | Compute-matched to S1 so only data composition changes; 1:1 ratio pre-registered |
| 2026-09-17 | **S5 = calibrated physics** (parameters fitted to measured ACDC-train statistics) | Keeps S3 (hand-set) and S5 (measured) distinct; uses the unlabeled target legitimately |
| 2026-09-17 | Relabeled `B0→S0`, `B2→S1`, `B1→T1`, `B1aug→T1aug`; S4 β ∈ **{0.05, 0.10}** | Uniform S-ladder IDs; weights preserved and eval re-run because `aggregate.py` groups by the JSON `run` field. FDA β restricted to the two retained values (β=0.01 dropped) |
| 2026-09-17 | **Phase 0 executed:** baselines renamed, 26 evals re-run, aggregate/visualize regenerated | Re-eval is bit-identical to the pre-rename numbers (S0 0.2007, S1 0.2690, T1 0.2162, T1aug 0.3196), confirming reproducibility |
| 2026-09-17 | **Phase 1 executed:** design split `acdc_design.txt` (400 = 100/weather, seed 42) + unlabeled pool `acdc_pool_unlabeled.txt` (1,200 = 300/weather); `build_splits.py`/`write_configs.py`/`materialize.py` cleaned of LOO/5k/smoke | Leakage control: the 406 official val is the only scored set; design ∪ pool = official train (1,600, unchanged); all kept manifests byte-identical (hash guard passed) |
| 2026-09-17 | **S2 plan finalized:** split S1's 10k into A (5k clear) + B (5k); S2 = A clear + B degraded; ops brightness/contrast/gamma/saturation/Gaussian-noise (narrow ranges), 1–3 random per image, fixed order, filename-hash seed; **blur deferred**; labels copied | Resolves the S1-vs-S2 budget confound: **S1 (A+B clear) is the exact control**, no S1 rerun; only the appearance of the B half changes. S2 stays purely photometric so the S2-vs-S3/S5 comparison isolates weather structure |
| 2026-09-17 | **Literature knowledge hub created** at `paper/literature/` (`references.md` + `references.bib` + 9 topic notes), metadata verified via the arXiv API | Rebuilds the deleted prior-work notes to a citable standard for the manuscript; explicitly flags the old "MIC" and "ViSGA" tags as unverified |
| 2026-09-17 | **S2 implemented:** `src/synth/` (photometric ops, generic stage builder, inspector) + `data/yolo/bdd_s2/` (5,000 degraded B + 5,000 referenced clear); `configs/bdd_s2.yaml`; A/B halves recorded | S2 dataset ready; pure photometric (blur deferred); labels byte-identical; deterministic; a guard prevents degrading usable sources into near-black images (already-degenerate sources exempt) |
| 2026-09-17 | **S2 result:** official mAP@50 **0.2833 vs S1 0.2690** (+0.0143), mAP@50-95 0.1650 vs 0.1559 (+0.0091); 5-fold +0.0070 (within ±1.4–1.6 spread); in-domain unchanged | First stage above the anchor; gains concentrated in **rain + rare classes**; fog/night/snow flat → generic photometrics do not model structured weather (motivates S3/S5). Single-seed caveat: directional, not yet significant. Paper write-up in `paper/results_notes.md` |
| 2026-09-19 | **S3 parameter set pre-registered** (fog Koschmieder t∈[0.35,0.70]; rain 150–500/640² streaks; snow density 0.02–0.07; night brightness 0.35–0.60, gamma 1.0–1.4) | Locks S3 before any ACDC evaluation so S3 (hand-set) cannot collapse into S5 (ACDC-fitted). Corrected `PLAN.md`'s night γ<1 (which lifts shadows into a "dim daytime" artifact) to γ>1, which crushes shadows. Rain/snow density was raised from an initial 50–200/640² and 0.005–0.02 after a visual intensity sweep (before any ACDC eval), since the first pass read as drizzle/light flurries |
| 2026-09-19 | **S3 dataset implemented + built:** `stage_common.py`, `weather.py`, `build_s3.py`, `inspect_s3.py`; S2 code frozen | Append-only per-experiment code: each locked stage's generator is preserved and later stages add their own modules. Balanced 1,250/condition; inspector PASS; determinism verified |
| 2026-09-19 | **Blur boundary corrected:** blur is excluded from both S2 and S3 (was documented "deferred to S3/S5"); it is deferred beyond the S3/S5 comparison (later finalized to S6c on 2026-09-20) | `PLAN.md` wrongly assumed S2 included blur; S2 had deferred it. Keeping blur out of S3 preserves the S2-vs-S3 and S3-vs-S5 attributions |
| 2026-09-19 | **Per-experiment eval set locked:** every stage reports official (primary) + 5-fold (supplementary) + in-domain BDD, single seed 42, matching S2 | Keeps `aggregate.py`/`visualize.py` tables comparable across stages and gives each stage both the citable official number and the fold-spread/forgetting context |
| 2026-09-20 | **S3 result:** official mAP@50 **0.2741 vs S1 0.2690** (+0.0051) but **below S2 0.2833**; 5-fold **0.2947 ± 0.0119 ≈ S2 0.2933 ± 0.0144**; in-domain 0.4799 (−0.011). Per weather: **snow 0.284→0.300** (~49% of the S1→ceiling snow gap; snow recall 0.232→0.319) and **rain 0.244→0.264**; night 0.193→0.180 and fog flat | Hand-set weather **ties generic photometrics on 5-fold but not on the official split**; only the physically structured conditions (rain/snow) improve, while night (no local illumination) and fog (constant transmission) fail. Single seed: S2↔S3 gap inside fold spread. Directly motivates **S5 calibration** |
| 2026-09-20 | **S5 calibration rule pre-registered = ACDC-train unlabeled pool statistics only** (`splits/acdc_pool_unlabeled.txt`, 1,200; never val, never design). S5 is labeled **unlabeled DA**, not zero-shot | Resolves `PROJECT.md` §11 and `paper/literature/09`: hand-set priors (S3) and target access cannot be fully disentangled, so S5↔S3 is framed as calibration + access, and **S5↔S4 (equal pool access) is the controlled comparison** |
| 2026-09-20 | **S5 design locked (parameter-only):** render via S3's `weather.apply`; empirical/bounded sampling; clip fitted values to S3 ranges + log; vignette fit→clip `[0.10,0.30]`+report; add a calibration self-test; closed-loop is QA not a hard gate | Structural parity by construction (no operator duplication/drift), robustness against estimator blow-ups, and a defensible measurement-validation step (calibration is the paper's contribution) |
| 2026-09-20 | **Blur boundary finalized:** blur is excluded from S2, S3, **and S5**; deferred to S6c | Keeps the whole S2/S3/S5 comparison about weather structure vs parameter calibration, not blur |
| 2026-09-20 | **S5b blur ablation planned (ancillary, over S5):** condition-specific blur (rain directional motion, fog/snow defocus, night none), strength **calibrated to the ACDC-train pool** by sharpness attenuation; own pre-registered ranges; S5b↔S5 is the only clean comparison | Tests blur's effect cheaply before S6 design without contaminating S5; feeds S6a/S6b. Blur is not folded into S5 (would confound S5↔S3) |
| 2026-09-20 | **S5b is an S5b-only sensor/sharpness-calibration exception** (S5 excludes frequency/gradient matching) | Keeps S5's physics-side boundary clean while allowing a measured blur model in the ancillary arm; any S5b claim is scoped to S5b |
| 2026-09-20 | **S5b evidence path:** Tier 0 calibrated-blur preview + visibility; Tier 1 design-split (400) test-time sensitivity probe (never official val, never a stage row); Tier 2 training only if warranted | Cheap, leak-free screening before committing a ~3 h run; design split used for failure-mode inspection only |

## 11. Open questions

- Whether S6c (per-condition policy) earns its extra run.
- Whether to sweep the clear:synthetic ratio beyond the pre-registered 1:1.
- Whether the comparative study yields a publishable novelty or stands as a B.Tech study (decide from the per-condition results).
- Whether to include a YOLOv11 generalization check.
- Venue and submission deadline.
- **S5 open items:** (a) how often fitted values hit the S3-range clip — if frequent, it signals estimator bias or a mis-set hand range and must be reported, not hidden; (b) whether a calibration sample-size ablation (design split 100/condition vs pool 300/condition) earns an extra run.
- **S5b open items:** whether Tier 2 (training the blur arm) is warranted after the Tier 0/1 probe; how much design-split-probe tooling to build (minimal script vs config + script).
- Whether the "MIC" and "ViSGA" tags from earlier notes have a real source; otherwise drop them from the manuscript.
