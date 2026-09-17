# 08 — Benchmarks, Metrics, and Comparative Studies

## Why this matters
The study is a **comparative benchmark**. Its credibility depends on a locked protocol, one
detector, one seed, identical training budgets, and a clearly defined scored set. This note
collects the metric definitions and the closest comparative/benchmark papers.

## Metrics (as reported)
- **mAP@50** and **mAP@50-95** — from the COCO protocol [Lin et al., ECCV 2014]
  — `lin2014coco`. Ultralytics implements the same IoU-sweep AP.
- **Precision / Recall / F1** — Ultralytics aggregate values.
- **Per-weather** (fog/rain/night/snow) and **per-class** (6 unified classes).
- **FPS** on the target hardware.
- **Relative improvement over S1:** `(Method − S1) / S1 × 100`.

### Protocol facts (locked)
- Official ACDC val (406) = the only scored set, scored once per stage.
- 1 training per config, seed 42, batch 32, imgsz 640, AMP, 40/80 epochs.
- Source manifests fixed; ACDC official train = unlabeled pool only (S4) / calibration (S5).
- **No per-config hyperparameter tuning.**

## Closest existing benchmarks / comparative studies
### DA-RAW [Jeon et al., ICRA 2024] — `jeon2024daraw`
- Real adverse-weather DAOD; separates weather vs style gap. Closest protocol; use as the
  external reference (note detector/backbone differences).

### Robustness of detection in adverse weather [Pettersen & Zhu, 2026] — `pettersen2026robustness`
- A recent robustness/benchmark study; direct comparison point for our per-weather table.

### From Filters to VLMs [Aryashad et al., 2025] — `aryashad2025filters`
- Scores **defogging methods by downstream detection/segmentation**. Methodological sibling:
  "does the preprocessing/synthesis actually help the downstream task?" Cite as evidence this
  question is being asked, and position our study as the **augmentation/adaptation** analogue.

### Bridging Clear and Adverse Driving Conditions [Shapiro et al., 2025] — `shapiro2025bridging`
- Recent clear↔adverse bridging; check overlap carefully before claiming novelty.

### Object detection under rain (review) [Hnewa & Radha, T-ITS 2021] — `hnewa2021rainreview`
- Survey of rain-specific detection; good for citing per-condition difficulty.

### Rain 3D detection [Piroli et al., IV 2023] — `piroli2023rain3d`
- Rain robustness (3D); adjacent evidence.

## Comparative-study design lessons to state
- **Controls:** same detector/version, schedule, split, seed, eval script, thresholds; only
  the augmentation/adaptation differs. (Mirrors the peer-review advice in `PLAN.md`.)
- **Confound to acknowledge:** S2–S6 train on a 10k mix (5k clear + 5k synthetic) vs S0/S1 on
  10k clear. This holds compute constant (10k images, 80 epochs) but changes data composition.
  State it and (optionally) ablate the ratio.
- **Sim-to-real gap:** measure and report per-condition failures rather than only aggregate
  mAP. This is a contribution, not a weakness.

## What to cite where
- Metrics: `lin2014coco`.
- External comparison / protocol: `jeon2024daraw`, `pettersen2026robustness`,
  `aryashad2025filters`, `shapiro2025bridging`.
- Per-condition difficulty: `hnewa2021rainreview`, `piroli2023rain3d`.
