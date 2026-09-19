# HANDOFF — Session Resumption Point

> Purpose: a single, self-contained entry point for resuming work in a fresh session.
> Last updated: **2026-09-19**. Keep this current at each session close.
> `PROJECT.md` remains the single source of truth for facts, decisions, and results;
> this file is the operational summary and "where we are right now" pointer.
> DOX contracts (`AGENTS.md` files) remain the binding work rules — read the relevant
> chain before editing.

---

## 1. Project in one paragraph

Final-year thesis: train **YOLOv8n** (Ultralytics) for **object detection** on
adverse-weather driving data. Train on **clear/daytime BDD100K** (source), test on
**ACDC** fog/rain/night/snow (target). Current direction: an agreed **S0–S6 comparative
study** of augmentation/domain-adaptation strategies — transform clear BDD into synthetic
adverse-weather BDD, train one detector per strategy, and score all of them per weather on
real ACDC, with **S1 (Ultralytics defaults) as the anchor**. The previous method directions
(SM-WCFA, global Fourier Domain Adaptation [Yang & Soatto, CVPR 2020], Direction A / WSM,
spectral analysis) were removed in the 2026-09-17 baseline-only reset. **S0/S1/T1/T1aug are
locked; S2 is implemented, trained and evaluated (official mAP@50 0.2833 vs S1 0.2690);
S3's weather-synthesis dataset is built and validated (5k A clear + 5k B weather,
1,250/condition), with training/eval pending.**

---

## 2. Environment

| Item | Value |
|---|---|
| Working dir | `/home/ps/Desktop/major_project` (run all commands from here) |
| Python | 3.12.3 virtualenv at `.venv` |
| PyTorch | 2.6.0+cu124 (CUDA works) |
| Ultralytics | 8.4.147 |
| matplotlib | 3.11.1 (note: **no pandas**) |
| OpenCV | 5.0.0 (no scipy/skimage) |
| GPU | RTX 3050 Laptop, 6 GB |

---

## 3. Locked decisions (do not change without a recorded reason)

- **Model:** YOLOv8n only, locked for the whole matrix.
- **Uniform schedule (all configs):** `optimizer=auto` (AdamW, `lr0=0.001`),
  `cos_lr=True`, `patience=30`, `batch=32`, `imgsz=640`, AMP on, `seed=42`.
  **Only the training data/augmentation varies** — no per-config tuning.
- **Epochs:** 40 for no-aug (S0, T1); 80 for augmented (S1, T1aug, S2–S6).
- **Seeds:** **1 seed (42) everywhere.**
- **Source:** BDD `weather == "clear"` AND `timeofday == "daytime"` (12,454 pool →
  10k train / 2k val, seed 42, stratified by rare-class presence).
- **Target:** ACDC. **Detection only.**
- **Classes (unified, 6):** `0 person, 1 rider, 2 car, 3 truck, 4 bus, 5 bicycle`
  (motorcycle→bicycle). Source of truth: `scripts/class_map.py`.
- **`rider` head is randomly initialized** (no COCO match) — document in methods.
- **Primary eval:** **official split, 1 training/config, 1 seed** —
  test = ACDC official val (`splits/acdc_official_val.txt`, 406), per-weather/per-class.
  Baselines (relabeled 2026-09-17): **S0 0.201, S1 0.269, T1 0.216, T1aug 0.320**.
  S0/S1/T1 5-fold kept as supplementary only.
- **Study framing:** S0–S6 comparative study; **S1 = anchor**.
- **Experiment IDs:** relabeled `B0→S0`, `B2→S1`, `B1→T1`, `B1aug→T1aug` (2026-09-17; dirs are now S0/S1/T1_*/T1aug_*).
- **Synthetic data budget:** fixed **10k = 5k clear + 5k synthetic**, seed 42.
- **Synthesis implementation:** **offline** pre-generated datasets; S4 FDA stays online.
- **Leakage control:** 400-image ACDC **design split**; official val scored once.
- **S5:** physics-structured synthesis **calibrated to measured ACDC-train statistics**.
- **S4 β:** **{0.05, 0.10}** (provisional — decide when S4 is built; review suggests restoring 0.01).
- **Experiment grouping:** `--exp <ID>` → `results/experiments/<ID>/{train,eval,figures}`;
  `aggregate.py`/`visualize.py` scan the experiment tree.

---

## 4. Data and pipeline

### Verified dataset facts
- **ACDC:** 4,006 PNG. Train 1,600 (400/weather); val 406 (fog 100, night 106, rain 100,
  snow 100); test 2,000 (no GT). Images `datasets/acdc/images/<weather>/<split>/...`.
- **BDD100K:** detection images nested under `100k/{train,val}/.../{trainA,trainB,testA,testB}`;
  labels `bdd100k_labels_images_{train,val}.json` (train = 69,863 records).
- **Quirk:** `instancesonly_rain_train_gt_detection.json` is the **combined all-weather**
  train file (1,600); filter by weather prefix.

### Generated artifacts (never write inside `datasets/`)
- `data/yolo/bdd_src` — 12,000 images / 12,000 labels.
- `data/yolo/acdc` — 2,006 images / 2,006 labels (0 orphans).
- `data/yolo/bdd_s2` — 5,000 degraded B images / 5,000 labels (S2); the 5,000 clear A images are referenced from `bdd_src` (not copied). Manifest `splits/bdd_s2_train.txt`, config `configs/bdd_s2.yaml`.
- `data/yolo/bdd_s3` — 5,000 weather images / 5,000 labels (S3); the 5,000 clear A images are referenced from `bdd_src`. Manifest `splits/bdd_s3_train.txt` (10k), conditions `splits/bdd_s3_conditions.csv` (1,250/condition), config `configs/bdd_s3.yaml`.
- `splits/` — manifests are the source of truth for every split: `bdd_src_{train,val}`, `acdc_cv5/fold0..4`, `acdc_official_{train,val}` (1,600/406), `acdc_design` (400 = 100/weather), `acdc_pool_unlabeled` (1,200 = 300/weather), `acdc_perweather/*`, `bdd_s2_train`, `bdd_s3_train`.
- `configs/` — Ultralytics YAMLs (`bdd_src`, `acdc_official`, `acdc_cv5_fold0..4`, `bdd_s2`, `bdd_s3`).

### Pipeline order
```
convert_acdc.py → convert_bdd.py → build_splits.py → write_configs.py →
materialize.py → prune_labels.py → synth/build_dataset.py --stage s2 (or synth/build_s3.py) →
train.py --exp → eval.py --exp → aggregate.py → visualize.py
```
`src/synth/build_dataset.py` generates the S2 dataset and `src/synth/build_s3.py` the S3
dataset (both via `src/synth/stage_common.py`); S5/S6 add their own builders. S4 is online
and skips this step.

---

## 5. Baseline ladder (mAP@50)

| Study ID | Recipe | Official (primary) | 5-fold | In-domain |
|---|---|---|---|---|
| **S0** | BDD no-aug — **floor** | **0.201** | 0.213 ± 0.011 | 0.376 |
| T1 | ACDC labels, no aug | 0.216 | 0.254 ± 0.020 | — |
| **T1aug** | ACDC + default aug — **ceiling** | **0.320** | 0.383 ± 0.021 | — |
| **S1** | BDD + standard aug — **anchor** | **0.269** | 0.286 ± 0.014 | 0.491 |

### Key findings (all in `PROJECT.md` §9)
- **Augmentation, not target labels, is the lever.** T1 (labels, no aug) 0.254 ≈ S0 0.213;
  T1aug (labels + aug) 0.383. The tiny target set needs regularization.
- **S1 (standard aug) closes ~43% of the domain gap** (5-fold 0.213 → 0.286 of the 0.170
  to the ceiling) and **beats T1**. Headroom for a method (official, primary):
  **T1aug − S1 = 0.320 − 0.269 = +0.051** (5-fold supplementary: +0.097).
- **Remaining headroom is localized.** Closed by S1: fog 79%, rain 66%, night 47%,
  **snow 36%**. Per class: **truck 0.287 vs ceiling 0.495**, **bus 0.159 vs 0.340** hold
  most of the gap; `car` is nearly closed (0.661 → 0.687).
- **Caveat:** the 5-fold ceiling (0.383) exceeds the official-split ceiling (0.320) —
  different test sets (official val is harder). The **official split is primary**; 5-fold
  numbers are supplementary. Never claim 5-fold "would be better".

---

## 6. The agreed S0–S6 study

Every stage trains on **clear BDD** and is evaluated on **real ACDC** per weather; only the
training data/augmentation changes.

| ID | Training data (10k) | Purpose | Status |
|---|---|---|---|
| S0 | clear BDD, no aug | floor | done |
| S1 | clear BDD + Ultralytics defaults | **anchor** | done |
| S2 | S1 + generic photometric degradation (offline; blur deferred) | sensor degradation | **done** (0.2833) |
| S3 | S1 + simple weather transforms (offline) | fast weather sim | **built** (awaiting train) |
| S4 | S1 + FDA online, ACDC-train style (unlabeled) | appearance adaptation | planned |
| S5 | S1 + calibrated physics synthesis (offline) | principled weather sim | planned |
| S6a | best fixed combination | combination | planned |
| S6b | S6a + condition-aware selection | condition-aware policy | planned |
| S6c | per-condition policy (optional) | custom policy | optional |
| T1/T1aug | ACDC labels, no aug / + defaults | reference / ceiling | done |

**Dataset construction (S2/S3/S5/S6).** Offline datasets under `data/yolo/bdd_<stage>/`;
fixed **10k = 5k clear + 5k synthetic**, seed 42 (synthetic half generated one-per-source
from the remaining 5k clear images; labels byte-copied). In-training `val` =
`bdd_src_val.txt` for all stages (consistent `best.pt` selection, no ACDC leakage).
1:1 ratio pre-registered.

**S2 (finalized 2026-09-17).** A/B split of S1's 10k: **A = 5k clear**, **B = 5k clear**;
**S1 = A+B is the exact control (no S1 rerun)**. S2 = A clear (referenced from `bdd_src`) +
B degraded. Ops (offline, geometry-preserving): brightness ×0.7–1.3, contrast ×0.7–1.2,
gamma 0.8–1.4, saturation ×0.6–1.1, additive Gaussian noise σ 0–10; **blur deferred** to
S3/S5. Per image: random 1–3 ops (no replacement), fixed order brightness→contrast→gamma→
saturation→noise, seed = `hash(filename)+42`; ops logged to `synthesis_log.csv`; labels
copied byte-identically. Halves recorded as `splits/bdd_src_A_clear.txt` /
`splits/bdd_src_B_source.txt`. Outputs: `data/yolo/bdd_s2/`, `splits/bdd_s2_train.txt`,
`configs/bdd_s2.yaml` (val = `bdd_src_val.txt`). S2 is **zero-shot DG**.

**S3 (built 2026-09-19).** Same A/B harness: **A (5k clear) + B weather-transformed**, one
condition per image, **balanced 1,250 each** of fog/rain/snow/night, so **S1 is again the
exact control**. Hand-set, geometry-preserving operators (no ACDC, no blur, uniform depth,
no mixed conditions, no local lights, no snow accumulation): **fog** = constant-transmission
Koschmieder `I=J·t+A·(1−t)`, `t∈[0.35,0.70]`; **rain** = directional streak overlay
(150–500/640² area-scaled, length 10–30 px, slant 70–85°, alpha 0.3–0.5); **snow** = falling
particles (density 0.02–0.07, radius 2–6 px); **night** = brightness ×0.35–0.60, gamma
1.0–1.4, warm/cool tint ±5, vignette (γ>1 crushes shadows — `PLAN.md`'s γ<1 lifted them
into a "dim daytime" artifact). Params pre-registered in `PROJECT.md` §5. Built via
`src/synth/{stage_common,weather,build_s3,inspect_s3}.py`; outputs `data/yolo/bdd_s3/`,
`splits/bdd_s3_{train.txt,conditions.csv}`, `configs/bdd_s3.yaml`. Inspector PASS,
deterministic; per-condition previews `results/summary/figures/synth_preview_bdd_s3_*.png`.

**S3 vs S5.** S3 = hand-set parameters; S5 = same physical models with parameters fitted to
**measured ACDC-train statistics** (colour mean/std, RMS contrast, dark-channel haze,
gradient/noise energy; unlabeled).

**S4 (FDA).** Reference `low_freq_mutate` [Yang & Soatto, CVPR 2020], online; target pool =
`splits/acdc_pool_unlabeled.txt`; β ∈ **{0.05, 0.10}**; headline = best β.

**Leakage control.** 400-image design split (`splits/acdc_design.txt`, 100/condition from
official train) is used **only** to inspect failure modes and choose S6 policies — never
trained on, never an FDA style source, never scored. Official val (406) is the only final
scorer and is scored once per stage.

**Run order:** lock splits → confirm S0/S1 → S2 → S3 → S5 → S4 → S6a/S6b → optional S6c.

---

## 7. Where we are now

- **Baselines locked and relabeled.** `S0` (floor), `S1` (anchor), `T1`/`T1aug` (in-domain
  reference/ceiling). Phase 0 done: dirs renamed, 26 evals re-run (bit-identical numbers),
  `aggregate.py`/`visualize.py` regenerated with S/T names.
- **Phase 1 done:** `build_splits.py`/`write_configs.py`/`materialize.py` cleaned (retired
  LOO/5k/smoke no longer emitted); added `splits/acdc_design.txt` (400 = 100/weather) and
  `splits/acdc_pool_unlabeled.txt` (1,200 = 300/weather); official val untouched (406) and
  all kept manifests byte-identical (hash guard passed).
- **S0–S6 design documented** in `PROJECT.md` §5/§6/§9/§10 and this file. Decisions locked:
  offline synthesis, fixed 10k = 5k clear + 5k synthetic, S5 calibrated physics, design
  split, relabel to S/T IDs.
- **S2 implemented + trained (2026-09-17):** `src/synth/` + `data/yolo/bdd_s2/`; inspector
  PASS; deterministic. **Result (official mAP@50): S2 0.2833 vs S1 0.2690 (+0.0143)**;
  mAP@50-95 0.1650 vs 0.1559 (+0.0091); 5-fold +0.0070 (within ±1.4–1.6 spread); in-domain
  unchanged (0.491→0.492). Gains in **rain + rare classes**; fog/night/snow flat. Paper
  write-up: `paper/results_notes.md`.
- **S3 implemented + built (2026-09-19):** `src/synth/stage_common.py` (shared harness for
  S3+), `weather.py` (fog/rain/snow/night), `build_s3.py`, `inspect_s3.py`; all hand-set,
  geometry-preserving, no ACDC, no blur. Dataset `data/yolo/bdd_s3/` (5k weather + 5k clear
  A referenced), balanced 1,250/condition; inspector PASS; determinism verified. **S2 code is
  frozen (append-only per experiment).** **Next: train + eval S3.** The S4 FDA hook
  (`src/aug/fda.py`) is still planned.
- **`.gitignore` corrected (2026-09-17):** the earlier `data/` and `datasets/` patterns had
  hidden `src/data/` (all pipeline code) and the dataset `AGENTS.md` files from git. They are
  now tracked; only `/data/` and the heavy dataset subtrees are ignored.
- **Literature knowledge hub created** at `paper/literature/` — `references.md` + `references.bib`
  (metadata verified via the arXiv API) plus topic notes `01`–`08` and a novelty map `09`.
  It rebuilds the deleted prior-work notes and flags the old "MIC"/"ViSGA" tags as unverified.
- **`PLAN.md` (root)** is the raw peer-review conversation; `PROJECT.md` is authoritative
  where they differ (notably: ACDC test has **no GT**, so official val is the scorer).
- **`T1`/`T1aug` naming is locked**; **S4 β is provisional** (see §3 and §12).
- **Deferred cleanup (intentionally left as-is):** `PLAN.md`; `splits/acdc_perweather/*`
  (currently unused); `data/yolo/**/*.cache` (Ultralytics speed caches).
  `Mini_Project_G-1_Final.pdf` and `yolov8n.pt` are kept reference material.

---

## 8. Next steps (in order)

1. ~~**Relabel baselines (Phase 0)**~~ — **done 2026-09-17** (dirs renamed, 26 evals re-run,
   aggregate/visualize regenerated; numbers bit-identical).
2. ~~**Lock splits (Phase 1)**~~ — **done 2026-09-17**: `acdc_design.txt` (400) and
   `acdc_pool_unlabeled.txt` (1,200) added; builders cleaned; kept manifests hash-verified.
   - **Decide the S5 calibration rule** before S5 runs (ACDC-train stats = mild UDA vs design
     split only vs BDD-only = zero-shot); see `paper/literature/09_gaps_and_positioning.md`.
   - **Keep `paper/literature/` current**: verify venues marked `confirm`, resolve or drop the
     "MIC"/"ViSGA" tags, and confirm `shapiro2025bridging`/PAGen do not already cover our angle.
3. ~~**Build S2**~~ — **done 2026-09-17** (`src/synth/` + `data/yolo/bdd_s2/`; inspector PASS,
   determinism verified). Re-run with `python src/synth/build_dataset.py --stage s2 --jobs 8`.
4. ~~**Train + eval S2**~~ — **done 2026-09-17** (official mAP@50 0.2833 vs S1 0.2690;
   findings in `PROJECT.md` §9 and `paper/results_notes.md`).
5. ~~**Build S3**~~ — **done 2026-09-19** (`src/synth/{stage_common,weather,build_s3,inspect_s3}.py`
   + `data/yolo/bdd_s3/`; inspector PASS, determinism verified; params pre-registered in
   `PROJECT.md` §5). Re-run: `python src/synth/build_s3.py --jobs 8`.
6. **S3 train + eval (next):** train 80 epochs on `configs/bdd_s3.yaml`, eval on ACDC
   official per-weather, record vs S1/S2 (commands in §9). Then S5 → S6a/S6b; **S4 FDA**
   online (`src/aug/fda.py` + hook) — β chosen when we reach S4.
7. Later: S6c, ratio ablations, supervised fine-tune, LOO, paper.

---

## 9. Command recipes

```bash
# --- Phase 0: relabel baselines (DONE 2026-09-17; kept for reference) ---
# aggregate groups by the JSON "run" field, so re-eval with the new --name was required
python src/eval.py --weights results/experiments/S1/train/weights/best.pt \
  --data configs/acdc_official.yaml --name S1_acdc_official --per-weather --exp S1

# --- Baselines: train (already done; recipe for reference) ---
python src/train.py --data configs/bdd_src.yaml --model yolov8n.pt \
  --epochs 40 --batch 32 --seed 42 --aug none --exp S0
python src/train.py --data configs/bdd_src.yaml --model yolov8n.pt \
  --epochs 80 --batch 32 --seed 42 --aug default --exp S1

# --- S2 (DONE 2026-09-17: official mAP@50 0.2833 vs S1 0.2690) ---
python src/synth/build_dataset.py --stage s2 --jobs 8      # regenerate dataset
python src/synth/inspect_s2.py --dataset-name bdd_s2       # validate + preview
python src/train.py --data configs/bdd_s2.yaml --model yolov8n.pt \
  --epochs 80 --batch 32 --seed 42 --aug default --exp S2
python src/eval.py --weights results/experiments/S2/train/weights/best.pt \
  --data configs/acdc_official.yaml --name S2_acdc_official --per-weather --exp S2

# --- S3 (dataset built 2026-09-19; training/eval pending) ---
python src/synth/build_s3.py --jobs 8                     # regenerate dataset (~2.5 min)
python src/synth/inspect_s3.py --dataset-name bdd_s3      # validate + per-condition previews
python src/train.py --data configs/bdd_s3.yaml --model yolov8n.pt \
  --epochs 80 --batch 32 --seed 42 --aug default --exp S3
python src/eval.py --weights results/experiments/S3/train/weights/best.pt \
  --data configs/acdc_official.yaml --name S3_acdc_official --per-weather --exp S3
python src/eval.py --weights results/experiments/S3/train/weights/best.pt \
  --data configs/bdd_src.yaml --name S3_bdd_val --exp S3   # in-domain (no --per-weather)

# --- S4: online FDA (trainer hook not written yet) ---
# python src/train.py --data configs/bdd_src.yaml --model yolov8n.pt \
#   --epochs 80 --batch 32 --seed 42 --aug fda --beta 0.05 \
#   --target-pool splits/acdc_pool_unlabeled.txt --exp S4

# Aggregate + figures
python src/aggregate.py && python src/visualize.py
```

---

## 10. Key files

| Path | Role |
|---|---|
| `PROJECT.md` | Single source of truth: plans, facts, decisions, results (§5 study, §9 tracker, §10 log). |
| `paper/literature/` | Verified literature knowledge hub: `references.md`, `references.bib`, topic notes `01`–`09`. |
| `paper/results_notes.md` | Paper-facing running results narrative (baselines + S2, tables + interpretation). |
| `PLAN.md` | Raw peer-review conversation that motivated S0–S6 (authoritative source is `PROJECT.md`). |
| `AGENTS.md` (root) | DOX rail; user preferences; Child DOX Index. |
| `src/common.py` | Shared paths/constants. |
| `src/data/*.py` | Converters, split builder, config writer, materializer, label pruner. |
| `src/synth/` | Offline S2–S6 synthesis, **append-only per experiment**. S2 (frozen): `common.py`, `photometric.py`, `build_dataset.py`, `inspect_s2.py`. S3: `stage_common.py`, `weather.py`, `build_s3.py`, `inspect_s3.py`. |
| `src/aug/` | (planned) S4 FDA online hook. |
| `src/train.py` | Training wrapper; `--aug {none,default}`, `--exp`. |
| `src/eval.py` | Overall + per-class + per-weather metrics; `--exp`. |
| `src/aggregate.py` | Writes `results/summary/{summary.json,per_class.csv,per_class.md}`. |
| `src/visualize.py` | Per-exp + cross-exp figures under `results/summary/figures/`. |
| `datasets/` | Raw source, **read-only**. |
| `Mini_Project_G-1_Final.pdf` | 3rd-year report (bibliography unverified — rebuild). |

---

## 11. Conventions and gotchas

- **DOX:** read the applicable `AGENTS.md` chain before editing; run a DOX pass after.
  Keep `PROJECT.md` current — facts/results/decisions immediately; hypotheses/predictions
  at evidence gates, marked "revised from pre-registered".
- **`aggregate.py` groups by the JSON `"run"` field**, not the directory name. To relabel an
  experiment, re-run `eval.py` with the new `--name` (don't just rename files).
- **Multi-run configs** (e.g. T1): train dirs `<ID>_<run>`, eval `--name <ID>_acdc_<run>`
  so they group under `<ID>_acdc`.
- **Do NOT pass `--per-weather` for BDD in-domain evals** — `eval.py` extracts weather from
  the ACDC path layout and will fail on BDD paths.
- **`args.yaml` `project` fields reflect the original pre-rename run paths** (e.g. `pilot_B0`,
  `B1_official`) — that is provenance; the directory names (S0/S1/T1*/T1aug*) are authoritative.
- **`best.pt` for T1/T1aug is selected on the eval split** (in-training `val` = test) —
  mild optimism, acceptable for a bound; state it in the paper.
- **Fold-spread semantics differ:** S0/S1 fold spread = test-subset variation (one model
  tested 5 ways); T1/T1aug = training variance (5 models). Interpret separately.
- **Never write inside `datasets/`.** Generated outputs go to `data/`, `splits/`, `results/`,
  `configs/`, `paper/`.
- **No pandas** in the environment; use stdlib `csv`/`json`.
- **Ultralytics writes `.cache` files into label dirs** during validation. They are not
  labels — count only `*.txt` (2,006 ACDC labels, 12,000 BDD labels; `prune_labels.py
  --dry-run` reports 0 orphans).
- **Synthetic datasets are geometry-preserving** — transforms must never move boxes; inspect
  with `inspect_s2.py` / `inspect_s3.py` before training.
- **`src/synth/` code is append-only.** A stage's generator is frozen once its result is
  locked (S2 = `photometric.py`/`build_dataset.py`/`inspect_s2.py`); later stages add their
  own modules on `stage_common.py` and must never overwrite an earlier stage's code.
- **S3 low-signal-source exemption:** a handful of BDD "clear/daytime" sources are very dark
  (tunnels; mean < 20). Night darkens them to near-black, so they are exempt from the
  degeneracy guard (`stage_common.SOURCE_MIN_MEAN`) and reported as `source-degenerate`. The
  builder and inspector share the same rule.
- **S3 params are pre-registered** in `PROJECT.md` §5; never tune them from ACDC results (that
  would collapse S3 into S5).

---

## 12. Open questions

- **S4 FDA β set** — whether to restore β=0.01; decide when S4 is built.
- **S5 calibration protocol** (ACDC-train stats vs design split vs BDD-only) — decide before S5 runs.
- Resolve or drop the "MIC"/"ViSGA" tags in `paper/literature/09`.
- Whether S6c earns its extra run.
- Whether to sweep the clear:synthetic ratio beyond 1:1.
- Whether the study yields a publishable novelty or stands as a B.Tech study.
- Whether to include a YOLOv11 generalization check.
- Venue and submission deadline.
