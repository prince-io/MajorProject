# HANDOFF — Session Resumption Point

> Purpose: a single, self-contained entry point for resuming work in a fresh session.
> Last updated: **2026-09-17**. Keep this current at each session close.
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
spectral analysis) were removed in the 2026-09-17 baseline-only reset. **No code or
experiments have been run since; implementation is the next phase.**

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
  Study mapping: **S0 = B0 0.201, S1 = B2 0.269, T1 = B1 0.216, T1aug = B1aug 0.320**.
  B0/B1/B2 5-fold kept as supplementary only.
- **Study framing:** S0–S6 comparative study; **S1 = anchor**.
- **Experiment IDs:** relabel `B0→S0`, `B2→S1`, `B1→T1`, `B1aug→T1aug` on execution.
- **Synthetic data budget:** fixed **10k = 5k clear + 5k synthetic**, seed 42.
- **Synthesis implementation:** **offline** pre-generated datasets; S4 FDA stays online.
- **Leakage control:** 400-image ACDC **design split**; official val scored once.
- **S5:** physics-structured synthesis **calibrated to measured ACDC-train statistics**.
- **S4 β:** **{0.05, 0.10}** only.
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
- `splits/` — manifests are the source of truth for every split.
- `configs/` — baseline Ultralytics YAMLs (`bdd_src`, `acdc_official`, `acdc_cv5_fold0..4`).

### Pipeline order
```
convert_acdc.py → convert_bdd.py → build_splits.py → write_configs.py →
materialize.py → prune_labels.py → train.py --exp → eval.py --exp →
aggregate.py → visualize.py
```
The study adds an offline synthesis step (`src/synth/build_dataset.py`, planned) between
split building and training.

---

## 5. Baseline ladder (mAP@50) — becomes S0/S1 and T1/T1aug

| Study ID | Physical dir | Recipe | Official (primary) | 5-fold | In-domain |
|---|---|---|---|---|---|
| **S0** | `B0` | BDD no-aug — **floor** | **0.201** | 0.213 ± 0.011 | 0.376 |
| T1 | `B1` | ACDC labels, no aug | 0.216 | 0.254 ± 0.020 | — |
| **T1aug** | `B1aug` | ACDC + default aug — **ceiling** | **0.320** | 0.383 ± 0.021 | — |
| **S1** | `B2` | BDD + standard aug — **anchor** | **0.269** | 0.286 ± 0.014 | 0.491 |

### Key findings (all in `PROJECT.md` §9)
- **Augmentation, not target labels, is the lever.** B1 (labels, no aug) 0.254 ≈ B0 0.213;
  B1aug (labels + aug) 0.383. The tiny target set needs regularization.
- **S1/S2 (standard aug) closes ~43% of the domain gap** (5-fold 0.213 → 0.286 of the 0.170
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
| S0 | clear BDD, no aug | floor | done (`B0`) |
| S1 | clear BDD + Ultralytics defaults | **anchor** | done (`B2`) |
| S2 | S1 + generic photometric degradation (offline) | sensor degradation | planned |
| S3 | S1 + simple weather transforms (offline) | fast weather sim | planned |
| S4 | S1 + FDA online, ACDC-train style (unlabeled) | appearance adaptation | planned |
| S5 | S1 + calibrated physics synthesis (offline) | principled weather sim | planned |
| S6a | best fixed combination | combination | planned |
| S6b | S6a + condition-aware selection | condition-aware policy | planned |
| S6c | per-condition policy (optional) | custom policy | optional |
| T1/T1aug | ACDC labels, no aug / + defaults | reference / ceiling | done (`B1`/`B1aug`) |

**Dataset construction (S2/S3/S5/S6).** Offline datasets under `data/yolo/bdd_<stage>/`;
fixed **10k = 5k clear + 5k synthetic**, seed 42 (synthetic half generated one-per-source
from the remaining 5k clear images; labels byte-copied). In-training `val` =
`bdd_src_val.txt` for all stages (consistent `best.pt` selection, no ACDC leakage).
1:1 ratio pre-registered.

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

- **Baselines locked.** S0 (`B0`) and S1 (`B2`) exist plus the in-domain references
  (`B1`/`B1aug`). The full data pipeline is reproducible and the tracker is current.
- **S0–S6 design documented** in `PROJECT.md` §5/§6/§9/§10 and this file. Decisions locked:
  offline synthesis, fixed 10k = 5k clear + 5k synthetic, S5 calibrated physics, design
  split, relabel to S/T IDs.
- **Nothing implemented yet.** `src/synth/` and `src/aug/fda.py` are planned, not written.
  `build_splits.py`/`write_configs.py` still reference the removed LOO/5k/smoke splits and
  need cleaning in the implementation phase.
- **Literature knowledge hub created** at `paper/literature/` — `references.md` + `references.bib`
  (metadata verified via the arXiv API) plus topic notes `01`–`08` and a novelty map `09`.
  It rebuilds the deleted prior-work notes and flags the old "MIC"/"ViSGA" tags as unverified.
- **`PLAN.md` (root)** is the raw peer-review conversation; `PROJECT.md` is authoritative
  where they differ (notably: ACDC test has **no GT**, so official val is the scorer).
- **`T1`/`T1aug` naming is decided** (S4 β locked to {0.05, 0.10}); the physical dirs are
  renamed during Phase 0.
- **Deferred cleanup (intentionally left as-is):** `PLAN.md`; `splits/acdc_perweather/*`
  (currently unused); the retired LOO/5k/smoke emissions in `build_splits.py` /
  `write_configs.py` (Phase 1 code cleanup); `data/yolo/**/*.cache` (Ultralytics speed
  caches). `Mini_Project_G-1_Final.pdf` and `yolov8n.pt` are kept reference material.

---

## 8. Next steps (in order)

1. **Relabel baselines (Phase 0):** rename dirs `B0→S0`, `B2→S1`, `B1*→T1*`, `B1aug*→T1aug*`;
   re-run `eval.py` with new `--name`s; regenerate `results/summary/`. No retraining.
2. **Lock splits (Phase 1):** add `acdc_design.txt` and `acdc_pool_unlabeled.txt`; clean
   `src/data/build_splits.py` / `write_configs.py` to the current split set.
   - **Decide the S5 calibration rule** before S5 runs (ACDC-train stats = mild UDA vs design
     split only vs BDD-only = zero-shot); see `paper/literature/09_gaps_and_positioning.md`.
   - **Keep `paper/literature/` current**: verify venues marked `confirm`, resolve or drop the
     "MIC"/"ViSGA" tags, and confirm `shapiro2025bridging`/PAGen do not already cover our angle.
3. **Build `src/synth/`** (with its own `AGENTS.md`): photometric, weather, physics,
   calibration, dataset builder, inspector.
4. **Generate offline datasets** for S2/S3/S5/S6a/S6b; verify label invariance + inspect.
5. **S4 FDA online** (`src/aug/fda.py` + minimal trainer hook) with a fidelity check.
6. **Train S2–S6** (1 seed, official val), then `aggregate.py` + `visualize.py`.
7. Later: hybrid S6c, ratio ablations, supervised fine-tune, leave-one-domain-out, paper.

---

## 9. Command recipes

```bash
# --- Phase 0: relabel baselines (no retraining) ---
# (rename dirs, then re-eval; aggregate groups by the JSON "run" field)
python src/eval.py --weights results/experiments/S1/train/weights/best.pt \
  --data configs/acdc_official.yaml --name S1_acdc_official --per-weather --exp S1

# --- Baselines: train (already done; recipe for reference) ---
python src/train.py --data configs/bdd_src.yaml --model yolov8n.pt \
  --epochs 40 --batch 32 --seed 42 --aug none --exp S0
python src/train.py --data configs/bdd_src.yaml --model yolov8n.pt \
  --epochs 80 --batch 32 --seed 42 --aug default --exp S1

# --- Study stages (planned) ---
# S2/S3/S5/S6: train on pre-generated offline datasets
python src/train.py --data configs/bdd_s2.yaml --model yolov8n.pt \
  --epochs 80 --batch 32 --seed 42 --aug default --exp S2
# S4: online FDA (trainer hook)
# python src/train.py --data configs/bdd_src.yaml --model yolov8n.pt \
#   --epochs 80 --batch 32 --seed 42 --aug fda --beta 0.05 \
#   --target-pool splits/acdc_pool_unlabeled.txt --exp S4_fda001

# Evaluate every stage on the official split
python src/eval.py --weights results/experiments/S2/train/weights/best.pt \
  --data configs/acdc_official.yaml --name S2_acdc_official --per-weather --exp S2

# Aggregate + figures
python src/aggregate.py && python src/visualize.py
```

---

## 10. Key files

| Path | Role |
|---|---|
| `PROJECT.md` | Single source of truth: plans, facts, decisions, results (§5 study, §9 tracker, §10 log). |
| `paper/literature/` | Verified literature knowledge hub: `references.md`, `references.bib`, topic notes `01`–`09`. |
| `PLAN.md` | Raw peer-review conversation that motivated S0–S6 (authoritative source is `PROJECT.md`). |
| `AGENTS.md` (root) | DOX rail; user preferences; Child DOX Index. |
| `src/common.py` | Shared paths/constants. |
| `src/data/*.py` | Converters, split builder, config writer, materializer, label pruner. |
| `src/synth/` | (planned) offline S2/S3/S5/S6 synthesis package. |
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
- **Synthetic datasets are geometry-preserving** — weather/photometric transforms must never
  move boxes; verify with `inspect_synth.py` before training.

---

## 12. Open questions

- **S5 calibration protocol** (ACDC-train stats vs design split vs BDD-only) — decide before S5 runs.
- Resolve or drop the "MIC"/"ViSGA" tags in `paper/literature/09`.
- Whether S6c earns its extra run.
- Whether to sweep the clear:synthetic ratio beyond 1:1.
- Whether the study yields a publishable novelty or stands as a B.Tech study.
- Whether to include a YOLOv11 generalization check.
- Venue and submission deadline.
