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

- **Status:** Baseline ladder **B0 / B1 / B1aug / B2 complete and locked** (official split, 1 seed). The project now follows the agreed **S0–S6 comparative-study** design (§5): all stages train on clear BDD and are evaluated on real ACDC, with **S1 (Ultralytics defaults) as the anchor**. This is a documentation/cleanup checkpoint — **no method code or experiments have been run since the 2026-09-17 baseline-only reset**; implementation is the next phase. The in-domain references are to be relabeled `T1`/`T1aug` on execution.
- **Last updated:** 2026-09-17 (S0–S6 study documented; baseline-only reset)
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
| Experiment IDs | Relabel `B0→S0`, `B2→S1`, `B1→T1`, `B1aug→T1aug` on execution (weights preserved) | 2026-09-17 |
| Synthetic data budget | Fixed **10k = 5k clear + 5k synthetic**, seed 42 (compute-matched to S1) | 2026-09-17 |
| Synthesis implementation | **Offline** pre-generated datasets; S4 (FDA) remains online | 2026-09-17 |
| Leakage control | 400-image ACDC **design split**; official val scored once | 2026-09-17 |
| S5 definition | Physics-structured synthesis **calibrated to measured ACDC-train statistics** | 2026-09-17 |

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
| S0 | clear BDD, no aug | floor | **done** (physical dir `B0`) |
| S1 | clear BDD + Ultralytics defaults | **anchor** | **done** (physical dir `B2`) |
| S2 | S1 + generic photometric degradation (offline dataset) | non-weather sensor degradation | planned |
| S3 | S1 + simple weather-specific transforms (offline dataset) | fast weather simulation | planned |
| S4 | S1 + Fourier Domain Adaptation (online), ACDC-train style (unlabeled) | appearance adaptation | planned (prior global-FDA run failed) |
| S5 | S1 + physics-structured, ACDC-calibrated synthesis (offline dataset) | principled weather simulation | planned |
| S6a | best fixed combination of S2–S5 | combination | planned |
| S6b | S6a + condition-aware selection | condition-aware policy | planned |
| S6c | per-condition policy from design-split inspection | custom policy | optional |
| T1 / T1aug | ACDC labels, no aug / + Ultralytics defaults | in-domain reference / **ceiling** | **done** (physical dirs `B1` / `B1aug`) |

**Naming.** The S-ladder is the study's ID scheme. On execution the existing baseline artifacts are relabeled `B0→S0`, `B2→S1`, `B1→T1`, `B1aug→T1aug` (weights preserved; `eval.py` re-run under new `--name` because `aggregate.py` groups by the JSON `"run"` field). Until then the physical directories keep the B-names.

**Dataset construction (S2 / S3 / S5 / S6).**
- **Offline** pre-generated datasets under `data/yolo/bdd_<stage>/`, so the training command is identical across stages — **only the training dataset differs**.
- Fixed **10,000-image budget** (compute-matched to S1): **5,000 clear + 5,000 synthetic**, seed 42. The synthetic half is generated one-per-source from the remaining 5,000 clear images; labels are byte-copied (geometry-preserving).
- In-training `val` = `bdd_src_val.txt` (clear) for every stage, so `best.pt` selection is consistent and never touches ACDC.
- The 1:1 clear:synthetic ratio is pre-registered; a ratio sweep is a later ablation.

**S3 vs S5 (must not collapse into each other).** S3 uses hand-set parameters. S5 fits the same physical model families (Koschmieder fog / dark-channel transmission, rain streaks, snow particles, night illumination) to **measured ACDC-train statistics** (per-condition colour mean/std, RMS contrast, dark-channel haze, gradient/noise energy; unlabeled) — this calibration is the principled separation and the basis of S5's potential novelty.

**S4 (FDA).** Faithful reference `low_freq_mutate` [Yang & Soatto, CVPR 2020], applied online; target pool = `splits/acdc_pool_unlabeled.txt` (ACDC official train minus the design split). β ∈ **{0.05, 0.10}**, headline = best β, reported honestly even if below S1 (the prior global-FDA run gave 0.273 at β=0.01 and 0.255 at β=0.05 vs S1 0.269).

**Implementation plan (next phase; no code yet).**
- `src/synth/` (new package, needs its own `AGENTS.md`): `photometric.py` (S2), `weather.py` (S3), `physics.py` (S5), `calibrate.py` (measured stats → `results/analysis/synth_stats.json`), `build_dataset.py` (writes `data/yolo/bdd_<stage>`, manifests, configs), `inspect_synth.py`.
- `src/aug/fda.py` + a minimal trainer hook in `src/train.py` for S4 only; fidelity check `max|ours-ref| = 0.0000`.
- Clean `src/data/build_splits.py` / `write_configs.py` (they still reference the removed LOO/5k/smoke splits) to emit `bdd_src`, `acdc_cv5`, `acdc_official`, `acdc_perweather`, `acdc_design`, `acdc_pool_unlabeled`.

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
- **Uncertainty:** none from seeds/folds under the primary protocol; variability is reported via the **per-weather and per-class** breakdown. The existing 5-fold results for B0/B1/B1aug/B2 are kept as supplementary only.
- **Limitations to state:** single seed, single official split (no CV error bars). The 2,000 unlabeled ACDC test images are unused.
- Qualitative: before/after detection panels per condition.
- Comparison targets: S0 (floor), S1 (anchor to beat), T1/T1aug (in-domain reference and ceiling); physical dirs are still `B0`/`B2`/`B1`/`B1aug` until relabeling.

## 7. Compute and budget

- `imgsz=640`, AMP on; YOLOv8n for all runs (§2).
- ~1.3 h per 10k-image / 40-epoch run (B0); ~2.5–2.9 h at the 80-epoch aug budget (B1aug, B2). ACDC fine-tunes are minutes.
- **Observed (B0):** ~3.0 it/s train, ~1:46/epoch + ~7 s val ≈ **1.9 min/epoch**; 40 epochs in 1.26 h; ~3.9 GB VRAM at batch 32.
- **Observed (B1/B1aug, ACDC ~1.6k imgs):** ~13 min / 40 epochs, ~26 min / 80 epochs; 6 runs each (official + 5 folds).
- **Observed (B2, 10k imgs / 80 epochs):** 2.70 h.
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
  synth/              # (planned) offline S2/S3/S5/S6 photometric + weather synthesis
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

Official experiment rows (1 seed 42, official ACDC val). Study mapping: **S0 = B0 (floor)**, **S1 = B2 (anchor to beat)**, **T1 = B1 (in-domain reference)**, **T1aug = B1aug (ceiling)**. Physical directories keep the B-names until the relabel phase (§5).

Planned study rows (fill as runs complete):

| ID | Config | mAP@50 | mAP@50-95 | P | R | Seed | Status |
|---|---|---|---|---|---|---|---|
| S2 | BDD + photometric synthesis (offline) | | | | | 42 | planned |
| S3 | BDD + simple weather synthesis (offline) | | | | | 42 | planned |
| S4 | BDD + FDA (online, best β) → ACDC | | | | | 42 | planned |
| S5 | BDD + calibrated physics synthesis (offline) | | | | | 42 | planned |
| S6a | best fixed combination | | | | | 42 | planned |
| S6b | S6a + condition-aware selection | | | | | 42 | planned |
| S6c | per-condition policy (optional) | | | | | 42 | optional |

| ID | Config | mAP@50 | mAP@50-95 | P | R | Seed | Status |
|---|---|---|---|---|---|---|---|
| B0_in | BDD no-aug → BDD val (in-domain) | 0.376 | 0.207 | 0.511 | 0.374 | 42 | done |
| B0_zs | BDD no-aug → ACDC 5-fold (zero-shot) | 0.213 ± 0.011 | 0.113 ± 0.004 | 0.370 ± 0.053 | 0.230 ± 0.015 | 42 | done |
| B0_zs | BDD no-aug → ACDC official val (zero-shot) | 0.201 | 0.108 | 0.401 | 0.202 | 42 | done |
| B1 | ACDC no-aug → ACDC 5-fold | 0.254 ± 0.020 | 0.136 ± 0.010 | 0.419 ± 0.083 | 0.265 ± 0.016 | 42 | done |
| B1 | ACDC no-aug → ACDC official val | 0.216 | 0.116 | 0.339 | 0.229 | 42 | done |
| B1aug | ACDC + default aug → ACDC 5-fold (**ceiling**) | 0.383 ± 0.021 | 0.221 ± 0.012 | 0.580 ± 0.026 | 0.353 ± 0.024 | 42 | done |
| B1aug | ACDC + default aug → ACDC official val | 0.320 | 0.196 | 0.539 | 0.289 | 42 | done |
| B2_in | BDD + standard aug → BDD val (in-domain) | 0.491 | 0.287 | 0.678 | 0.445 | 42 | done |
| B2_zs | BDD + standard aug → ACDC 5-fold (zero-shot) | 0.286 ± 0.014 | 0.161 ± 0.007 | 0.517 ± 0.042 | 0.274 ± 0.019 | 42 | done |
| B2_zs | BDD + standard aug → ACDC official val (zero-shot) | 0.269 | 0.156 | 0.461 | 0.266 | 42 | done |

### Baseline ladder (mAP@50)

| Run | Recipe | Official (primary) | 5-fold (supplementary) | In-domain |
|---|---|---|---|---|
| **B0** | BDD no-aug — **floor** | **0.201** | 0.213 ± 0.011 | 0.376 |
| B1 | ACDC labels, no aug | 0.216 | 0.254 ± 0.020 | — |
| **B1aug** | ACDC + default aug — **ceiling** | **0.320** | 0.383 ± 0.021 | — |
| **B2** | BDD + standard aug | **0.269** | 0.286 ± 0.014 | 0.491 |

**Findings:**
- **Augmentation, not target labels, is the lever.** B1 (labels, no aug) 0.254 ≈ B0 0.213; B1aug (labels + aug) 0.383. The tiny target set needs regularization.
- **B2 (standard aug) closes ~43% of the domain gap** (5-fold 0.213 → 0.286 of the 0.170 to the ceiling) and **beats B1**. Headroom for a method (official, primary): **B1aug − B2 = 0.320 − 0.269 = +0.051** (5-fold supplementary: +0.097).
- **Remaining headroom is localized.** Closed by B2: fog 79%, rain 66%, night 47%, **snow 36%**.
- **Caveat:** the 5-fold ceiling (0.383) exceeds the official-split ceiling (0.320) — different test sets (official val is harder). The **official split is primary**; 5-fold numbers are supplementary. Never claim 5-fold "would be better".

**Per-class 5-fold mAP50 (B0 → B2 → ceiling):**

| class | B0 | B2 | B1aug |
|---|---|---|---|
| person | 0.221 | 0.295 | 0.384 |
| rider | 0.096 | 0.177 | 0.214 |
| car | 0.577 | 0.661 | 0.687 |
| truck | 0.194 | 0.287 | 0.495 |
| bus | 0.112 | 0.159 | 0.340 |
| bicycle | 0.077 | 0.139 | 0.177 |

Per class, **truck 0.287 vs ceiling 0.495** and **bus 0.159 vs 0.340** hold most of the remaining headroom, while `car` is nearly closed (0.661 → 0.687).

**Per-weather 5-fold mAP50 (B0 → B2 → ceiling):**

| | fog | rain | snow | night |
|---|---|---|---|---|
| B0 | 0.373 | 0.216 | 0.249 | 0.138 |
| B2 | 0.476 | 0.299 | 0.300 | 0.196 |
| B1aug | 0.504 | 0.341 | 0.391 | 0.262 |

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
| 2026-09-17 | Relabel `B0→S0`, `B2→S1`, `B1→T1`, `B1aug→T1aug` on execution; S4 β ∈ **{0.05, 0.10}** | Uniform S-ladder IDs; weights preserved and eval re-run because `aggregate.py` groups by the JSON `run` field. FDA β restricted to the two retained values (β=0.01 dropped) |
| 2026-09-17 | **Literature knowledge hub created** at `paper/literature/` (`references.md` + `references.bib` + 9 topic notes), metadata verified via the arXiv API | Rebuilds the deleted prior-work notes to a citable standard for the manuscript; explicitly flags the old "MIC" and "ViSGA" tags as unverified |

## 11. Open questions

- Whether S6c (per-condition policy) earns its extra run.
- Whether to sweep the clear:synthetic ratio beyond the pre-registered 1:1.
- Whether the comparative study yields a publishable novelty or stands as a B.Tech study (decide from the per-condition results).
- Whether to include a YOLOv11 generalization check.
- Venue and submission deadline.
- **S5 calibration protocol (decide before S5 runs):** whether the parameters are fit on ACDC-train unlabeled statistics (mild UDA), on the design split only, or on BDD-adverse statistics (fully zero-shot). See `paper/literature/09_gaps_and_positioning.md`.
- Whether the "MIC" and "ViSGA" tags from earlier notes have a real source; otherwise drop them from the manuscript.
