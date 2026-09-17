This is an unusually well-run project for a B.Tech thesis — arguably better documented than many M.Tech/Ph.D. repos. The baseline ladder is locked, leakage control is real, and the S0–S6 plan is coherent. Below is what I'd flag before you write a single line of `src/synth/`.

## What is genuinely strong

- **Leakage control is correct.** Design split (400) vs official val (406) vs FDA pool (1,200) is the right shape, and scoring official val once is exactly what a reviewer wants to see.
- **Baseline ladder is interpretable.** S0 floor, S1 anchor, T1/T1aug in-domain reference/ceiling, with the key finding (augmentation, not labels) already localized.
- **Reproducibility discipline.** Hash guards on kept manifests, re-eval bit-identical after rename, versioned configs, seed 42 everywhere. This is the part most students fail.
- **Compute budget is realistic.** 13.5 GPU-h for S2–S6 on a 3050 is believable and the 80-epoch/40-epoch split is defensible.

## Issues to resolve before S2–S6 runs

### 1. The S1 vs S2–S6 data-budget confound (biggest issue)

S1 trains on **10k clear BDD**. S2–S6 train on **5k clear + 5k synthetic**. That is not compute-matched — it is *data-composition-changed* and *clear-diversity-halved* at the same time. Any S2–S6 gain over S1 is now confounded with "we removed 5k unique clear scenes."

Two clean fixes, pick one:

- **Option A (recommended):** S1 also trains on 5k clear + 5k clear-duplicates (or 5k clear + 5k clear from a different slice). Then S1, S2–S6 all see 10k images with 5k unique clear scenes; only the second half differs. The comparison becomes "what you fill the second 5k with."
- **Option B:** S2–S6 use 10k = 5k clear + 5k synthetic, and you additionally run a **control** S1-half = 5k clear only. Report S1 (10k) and S1-half (5k) so the reader can see how much of the S2–S6 effect is just "5k clear is enough."

Without one of these, a reviewer will ask "did S3 beat S1 because of weather synthesis, or because you dropped half the clear images?" You have no answer.

### 2. S4 FDA β choice looks self-defeating

You dropped β=0.01 — the only β that beat S1 in the prior run (0.273 vs 0.269). You kept {0.05, 0.10}, both of which previously underperformed (0.255 at 0.05; 0.10 untested but higher β = more distortion = typically worse). As locked, S4 is likely to lose to S1 and contribute nothing but "FDA doesn't help here."

That is a legitimate finding, but if you want S4 to have a fair shot:

- Keep **{0.01, 0.05, 0.10}** or at minimum **{0.01, 0.05}**. Reference FDA (Yang & Soatto) uses low β; 0.01 is well within range. Dropping it looks like pre-selecting a failure.
- If the reason was "0.01 gave 0.273 which is within noise of 0.269," say so in the decisions log and keep it anyway as a reported data point.

### 3. Epoch mismatch S0 vs S1

S0 = 40 epochs, S1 = 80. That confounds "no aug" with "half the training." Fine for S0-as-floor, but do not let anyone read "S0 → S1 = +0.068" as pure augmentation effect. Either run S0 at 80 epochs as a supplementary row, or state explicitly in the paper that S0 is a floor, not an augmentation ablation.

### 4. Make mAP@50-95 the primary number, not mAP@50

The handoff already says mAP@50-95 + Recall matter. But the baseline ladder table and every headline in PROJECT.md is mAP@50. Switch the primary table to **mAP@50-95** (with mAP@50 as secondary) or you will be criticized for reporting the most saturated metric. Adverse weather hurts localization, not just classification — mAP@50-95 will separate methods better.

### 5. Categorize methods by adaptation setting in the paper

S0–S3 and S5-as-calibrated-to-ACDC-train are **not the same setting**:

- S0, S1, S2, S3 (hand-set): zero-shot DG — no ACDC at all.
- S4 FDA: unlabeled domain adaptation — ACDC images seen, no labels.
- S5 (if calibrated to ACDC-train stats): **also unlabeled DA** — you are using target statistics.
- T1/T1aug: supervised fine-tuning — ACDC labels used.

Your §6 terminology already distinguishes these, but the S0–S6 table in §5 flattens them. Make the setting column explicit in the paper's main table. Reviewers will jump on "S5 looks like zero-shot but used target stats."

The open question in §11 ("S5 calibration protocol: ACDC-train vs design split vs BDD-only") is the right question. My recommendation: **calibrate on ACDC-train (excluding design split) and label S5 as unlabeled DA**, because BDD-only calibration makes S5 and S3 nearly the same thing and destroys the "principled vs hand-set" distinction.

### 6. Minor but worth fixing

- **Per-class on the official split.** PROJECT.md shows per-class 5-fold only. Add per-class official-split numbers to §9 once S2–S6 run.
- **3 seeds for S1 and S6.** You have 1 seed everywhere. For the *anchor* and the *proposed method*, run 3 seeds and report mean ± std. One seed for S2–S5 is defensible; one seed for your headline method is not. ~8 extra GPU-h.
- **`rider` random init is consistent** across S0–S6, so comparisons are fair. Just say so explicitly in methods.
- **T1/T1aug best.pt on eval split** inflates the ceiling slightly. Already noted — keep that sentence in the paper.
- **454 unused BDD pool images.** Trivial, but if a reviewer asks, have the answer ("kept out of both train and val to preserve the exact 10k/2k split").

## Bottom line

The project is in very good shape. The two things I would not proceed past without resolving:

1. **The 5k-clear + 5k-synthetic vs 10k-clear confound** — add an S1-half control or change the budget.
2. **S4 β = {0.05, 0.10}** — restore β=0.01 or explicitly justify dropping it, otherwise S4 is a pre-rigged null result.

Everything else is polish. Fix those two, then build `src/synth/`.


------------------------------------------------------------


Good — this is exactly the right stage to argue about the design. S2 is well-scoped, but there is one methodological point that needs to be nailed down before any code: the data-budget framing. Everything else is straightforward once that is settled.

## 1. The biggest issue: Q1 data-budget confound

Your current framing says S1 trains on 10k clear, S2 trains on 5k clear + 5k degraded, so S2 has less clean data. But if S2’s 5k degraded images are generated from the *other* 5k clear images that S1 also saw, then S1 is already the exact control.

Let me restate it cleanly:

- Let the 10k clear images used by S1 be split into **A (5k)** and **B (5k)**.
- S1 trains on **A_clear + B_clear**.
- S2 should train on **A_clear + B_degraded**, where each B_degraded is a photometric transform of the corresponding B_clear.

Then the only difference between S1 and S2 is the appearance of the second half. Same scenes, same objects, same labels, same count. That is the clean comparison.

So my recommendation is:

- **Do not change the S1 anchor.** Keep S1 as the 10k-clear Ultralytics-defaults run.
- **Define A and B from S1’s training set.** If S1 was trained on exactly those 10k clear images, you do not need to re-run S1. You just need to split its training list into A and B.
- **If S1’s 10k is not available or not the same pool**, then re-run S1_control on A_clear + B_clear. That becomes your matched anchor for S2. But if S1 already used those images, re-running is wasted compute.
- **Option (b) — 5k-clear-only control — is not necessary** for the main S2 vs S1 comparison. It answers a different question (“what does halving clean data cost?”). Keep it in your back pocket only if you have spare GPU time and want a diagnostic.

So the first implementation step is a data audit: locate S1’s training list, verify it has 10k clear BDD images, and split it deterministically into A and B. If that checks out, Q1 is resolved without an extra 80-epoch run.

## 2. Op list and ranges

The op list is reasonable, but two things need tightening:

- **Blur is not photometric.** If you keep blur in S2, rename S2 to “generic image degradation” instead of “generic photometric degradation.” Otherwise the terminology will bite you in the thesis.
- **Ranges are a bit wide for small objects.** ACDC has many small, distant objects. Aggressive blur or noise can erase them. I’d narrow the ranges:

| Op | Proposed range | Suggested range |
|---|---:|---:|
| Brightness | ×0.6 – ×1.4 | ×0.7 – ×1.3 |
| Contrast | ×0.6 – ×1.3 | ×0.7 – ×1.2 |
| Gamma | 0.7 – 1.5 | 0.8 – 1.4 |
| Saturation | ×0.5 – ×1.2 | ×0.6 – ×1.1 |
| Gaussian blur | σ 0 – 1.5 | σ 0 – 1.0 (if kept) |
| Defocus blur | radius 0 – 3 | radius 0 – 2 (if kept) |
| Additive Gaussian noise | σ 0 – 15 | σ 0 – 10 |

These are still meaningful degradations but less likely to destroy labels.

## 3. Blur: include or defer?

My strong preference: **defer blur to S3/S5.** Keep S2 purely photometric (brightness, contrast, gamma, saturation, noise). Reasons:

- S2 is the control for “generic degradation robustness.” If you include blur, you are already touching an optical effect that weather synthesis will also model (rain streaks, fog scattering, snow, motion blur).
- A purely photometric S2 makes the S2 vs S3/S5 comparison cleaner: S2 tests tone/exposure/sensor noise, S3/S5 test weather structure.
- If S2 with only photometric ops helps ACDC a lot, that is already a strong result. If it helps little and S3/S5 help a lot, the weather-specific modelling argument is stronger.

If you do include blur, run a small ablation later (S2 with vs without blur) rather than mixing it into the main S2.

## 4. Per-image random vs fixed pipeline

Since synthesis is offline, each image is generated once. I’d use:

- **Per-image random subset of ops**, sampled from 1–3 ops without replacement.
- **Seeded per image** using a stable hash of the filename plus global seed 42. This makes generation order-independent and reproducible.
- **Log every op and parameter** to `synthesis_log.csv`.
- **Fixed order of application**: brightness → contrast → gamma → saturation → (blur) → noise. Noise last.

This gives diversity across images while remaining inspectable. The Ultralytics online augmentations (mosaic, HSV, flips) still apply on top during training, so the model sees additional variation each epoch.

## 5. Implementation plan

Here is the concrete plan I’d follow.

### Step 0 — Data audit and split
- Load S1’s training list.
- Verify it contains 10k clear BDD images.
- Split deterministically with seed 42 into `A_clear.txt` (5k) and `B_source.txt` (5k).
- Save these lists under `data/yolo/bdd_s2/`.
- Confirm A and B are disjoint and that no ACDC image appears anywhere.

### Step 1 — `src/synth/photometric.py`
- Functions: `apply_brightness`, `apply_contrast`, `apply_gamma`, `apply_saturation`, `apply_gaussian_noise`. (Add blur functions only if you keep blur.)
- Each function takes a float32 image and parameters, returns float32.
- Use OpenCV/NumPy. Clip to [0,255] at the end.
- Add `sample_ops(rng)` that returns a list of 1–3 op names and their parameters.

### Step 2 — `src/synth/build_s2.py`
- Read `B_source.txt`.
- For each image:
  - Read image.
  - Derive per-image RNG from filename + seed 42.
  - Sample 1–3 ops and parameters.
  - Apply in fixed order.
  - Save to `data/yolo/bdd_s2/images/train/` with the same filename.
  - Copy the label file byte-identically to `data/yolo/bdd_s2/labels/train/`.
  - Append a row to `synthesis_log.csv`.
- Also copy the 5k clear images from A and their labels into the same train directories. This makes the dataset self-contained.
- Write `bdd_s2_train.txt` listing all 10k images.

### Step 3 — `src/synth/inspect_s2.py`
Check:
- Count is exactly 10k.
- Every synthetic image has a byte-identical label to its source.
- No image is all-black or all-white (mean not < 5 or > 250, std > 10).
- No synthetic image is identical to its source (hash check).
- Save a visual grid of 20 random synthetic images with boxes drawn.
- Output a short report.

### Step 4 — `bdd_s2.yaml`
- `train: data/yolo/bdd_s2/bdd_s2_train.txt`
- `val: data/yolo/bdd_src_val.txt` (same clear 2k val as S1)
- `names`: same as S1.
- `path`: absolute or relative to dataset root.

### Step 5 — Training
- Use the exact same command as S1, only changing `--data` to `bdd_s2.yaml`.
- `model=yolov8n.pt`, `epochs=80`, `imgsz=640`, `batch=32`, `seed=42`, `patience=30`, `optimizer=auto`, `cos_lr=True`, `amp=True`.

### Step 6 — Evaluation
- Use the same evaluation script as S1.
- Score on ACDC official val (406) per weather.
- Report mAP@50, mAP@50-95, P, R per weather and per class.
- Compare S2 vs S1 with deltas.
- Inspect failure modes on the 400-image ACDC design split, but never train or score on it.

## 6. Red flags to watch

- **ACDC leakage:** Ensure ACDC official val and the design split never appear in training or in-training validation.
- **Label drift:** Copy labels with `shutil.copy2`. Do not regenerate them.
- **RNG reproducibility:** Use per-image seeds derived from filenames, not a single global RNG that depends on processing order.
- **Best.pt selection:** S2 will select `best.pt` on clear BDD val, same as S1. That is consistent, but note it may not pick the best ACDC model. That is a known limitation of the whole matrix.
- **Compute matching:** Keep the 10k budget. Do not add extra images.
- **Terminology:** If you keep blur, call S2 “generic image degradation.” If you drop blur, “generic photometric degradation” is correct.

## Bottom line

S2 is a good control. The main fix is to make sure S1 and S2 share the same 10k clear image pool, with S2 replacing the second 5k with degraded versions. If S1 already used that pool, you do not need a new S1 run. Keep S2 photometric-only, use moderate ranges, random 1–3 ops per image, log everything, and inspect labels plus image statistics before training. Once that is done, the implementation is straightforward.