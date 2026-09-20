# S5 Implementation Plan — Calibrated Physics Synthesis

**Status:** draft for approval. Supersedes the S5 brief where they differ.
**Locked principle:** S5 v1 uses **exactly S3's model structure**; only parameter *values* change, from hand-set to ACDC-train-fitted. Any structural change (row-depth fog, local night illumination, blur) is S6c, not S5.

---

## 1. Goal and framing

Fit the parameters of S3's four weather operators to statistics measured on the **unlabeled ACDC-train pool** (`splits/acdc_pool_unlabeled.txt`, 1,200 = 300/condition), then generate a 10k dataset (`data/yolo/bdd_s5/`) with the same A/B harness, same source images, same condition allocation, and same labels as S3. Train YOLOv8n once (80 epochs), evaluate on the S2-matched set.

**What S5 measures (state this in the paper):**
- **S5 vs S3** = calibration + target-statistics access vs hand-set priors + zero-shot. Two variables change; not a pure calibration ablation.
- **S5 vs S4** = the controlled UDA comparison: physics-structured synthesis vs Fourier appearance adaptation, same target pool, same access setting. This is the paper's headline S5 comparison.
- **S5 vs S2** = physics structure vs generic photometric degradation, both offline.

**What S5 does not measure:** calibration alone. Say so explicitly.

---

## 2. Scope locks

| Item | S5 v1 |
|---|---|
| Model structure | **Identical to S3** (uniform-depth Koschmieder fog, directional rain streaks, falling snow particles, illumination night with vignette) |
| Fog depth | Uniform (no row-depth). Row-depth → S6c. |
| Night illumination | Global + vignette only (no headlights/streetlamps). Local sources → S6c. |
| Blur | **Not in S5.** Update `PROJECT.md` to say blur is deferred beyond the S3/S5 comparison. |
| Parameter values | Fitted from ACDC-train pool (unlabeled), distribution per condition |
| Fitting targets | Colour mean/std, RMS contrast, dark-channel haze, saturation — **not** gradient/noise energy |
| Calibration source | `splits/acdc_pool_unlabeled.txt` (1,200), never the design split, never official val |
| Dataset harness | Same A/B split as S2/S3: A = 5k clear referenced from `bdd_src`, B = 5k synthetic |
| Source image list | Identical to S3 (seed 42, same file list) |
| Condition allocation | Identical to S3 — reuse `splits/bdd_s3_conditions.csv` (hash-verified) |
| Labels | Byte-copied |
| Seed | 42 |
| Schedule | Identical to S1/S2/S3 (80 epochs, batch 32, `optimizer=auto`, `cos_lr=True`, `patience=30`, AMP, imgsz 640) |
| Eval | S2-matched set: official (primary), 5-fold (supplementary), in-domain BDD |

---

## 3. Calibration protocol

`src/synth/calibrate.py` reads the 1,200-image pool, computes per-condition statistics, writes `results/analysis/synth_stats.json` **once**. Versioned. Never recomputed by the builder.

**Per-condition pipeline:**
1. Load all 300 images for that condition (RGB, native resolution, no resize).
2. Compute per-image parameter estimates (below).
3. Reject outliers (per parameter: drop beyond ±3 MAD from median; report rejection counts).
4. Report the **distribution**: median, MAD, p10, p90, and a physical clip range.
5. Write to `synth_stats.json`.

**Reference for relative statistics:** the 5k clear A half (`splits/bdd_src_A_clear.txt`), used as the "clear BDD" baseline for contrast/brightness/saturation ratios. Same reference for all conditions.

### 3.1 Per-condition estimates

**Fog**
- `A` (atmospheric light): per-image, mean of brightest 0.1% pixels per channel, using dark-channel prior to mask out small bright blobs (headlights). Median across images → single `A` per channel, plus a small per-channel jitter (≤ 8, as S3).
- `t` (transmission): per-image, `t = 1 − median(D)/median(A)` where `D` is the dark channel (min over channels of a 15×15 min-filter). Distribution of `t` across images → mean/MAD/clip.
- `desaturation`: per-image saturation (HSV S) mean, divided by mean saturation of the clear reference, clipped to `[0.6, 1.0]`. Distribution across images.

**Rain**
- `slant`: high-pass each image (subtract 5×5 median blur), compute gradient orientation weighted by magnitude, take dominant mode in `[60°, 90°]` (pick the mode closer to vertical). Median across images.
- `length`: run-length of high-gradient pixels along the dominant orientation.
- `density`: fraction of pixels above an adaptive gradient threshold (e.g., mean + 2σ of the high-passed image).
- `contrast`: RMS contrast of rain images divided by clear reference, clipped to `[0.75, 1.0]`.
- `alpha`, `width`: derived from streak contrast vs local background; width fixed at `[1, 2]` px per S3.

**Snow**
- `radius`: white top-hat (structuring element ≈ 1.5× expected flake size) → connected components → `r = sqrt(area/π)`. Distribution across blobs.
- `density`: blob count per pixel area.
- `alpha`: blob peak intensity minus local background, normalized.
- `brightness`: mean luminance of snow images / clear reference.
- `contrast`: RMS contrast ratio, clipped.
- `desaturation`: as fog.

**Night**
- `brightness`: mean luminance of night images / clear reference.
- `gamma`: per-image, fit a single γ that maps the 25/50/75 percentiles of a γ-adjusted darkened clear reference to those of the night image. Median across images.
- `tint`: per-channel mean differences `(R−L, G−L, B−L)` across night images, in BGR-correct order.
- `vignette`: fit α in `L(r) = L₀ · (1 − α·r²)` to the radial luminance profile; report distribution of α.

All estimates are **per-image → pooled → distribution**. No per-image fitting at generation time.

---

## 4. `synth_stats.json` schema

```json
{
  "version": "1.0",
  "generated": "2026-09-XX",
  "source_pool": "splits/acdc_pool_unlabeled.txt",
  "reference_clear": "splits/bdd_src_A_clear.txt",
  "n_per_condition": 300,
  "conditions": {
    "fog": {
      "A": {"r": 205, "g": 208, "b": 215, "jitter": 6},
      "t": {"median": 0.52, "mad": 0.06, "p10": 0.42, "p90": 0.63, "clip": [0.30, 0.80]},
      "desaturation": {"median": 0.85, "mad": 0.05, "clip": [0.60, 1.00]}
    },
    "rain": { "slant": {...}, "length": {...}, "density": {...}, "contrast": {...} },
    "snow": { "radius": {...}, "density": {...}, "alpha": {...}, "brightness": {...}, "contrast": {...}, "desaturation": {...} },
    "night": { "brightness": {...}, "gamma": {...}, "tint": {"b": ..., "g": ..., "r": ...}, "vignette": {...} }
  },
  "rejected": {"fog": 3, "rain": 5, "snow": 2, "night": 4},
  "notes": "..."
}
```

---

## 5. Dataset generation

`src/synth/build_s5.py`:

1. Read `synth_stats.json` (fail if missing; never recompute).
2. Read `splits/bdd_s3_conditions.csv`; assert it matches a seed-42 re-derivation (hash guard). This guarantees condition allocation is identical to S3.
3. For each source image `f` in the B half:
   - Seed RNG with `hash(f) + 42`.
   - Draw each parameter from the fitted distribution: `sample = median + MAD · z`, `z ~ N(0,1)`, then clip to `clip`.
   - Apply the operator from `physics.py` (identical structure to S3's `weather.py`).
   - Copy labels byte-identically (`shutil.copy2`).
4. Write `data/yolo/bdd_s5/` (5k synthetic images), `splits/bdd_s5_train.txt` (10k, A referenced from `bdd_src`), `configs/bdd_s5.yaml` (val = `bdd_src_val.txt`).
5. Log `index.json` + `synthesis_log.csv`: condition, sampled parameters, source file.

**`src/synth/physics.py`:** four operator functions with signatures identical to `weather.py` but taking explicit params from the stats file. Structural parity with `weather.py` is enforced by the inspector (§6).

---

## 6. Inspection — `src/synth/inspect_s5.py`

All S3 checks, plus:

- **Label invariance:** 5,000/5,000 byte-identical.
- **Condition counts:** 1,250 each.
- **Source/seed parity with S3:** for each source, condition matches `bdd_s3_conditions.csv`.
- **Structural parity:** call `weather.py` with S3's hand-set ranges and `physics.py` with the same explicit params on a fixed source; assert identical output (tolerance ≤ 1 pixel value). This proves the two operators are the same model.
- **Closed-loop statistics check:** recompute the fitting statistics (colour mean/std, RMS contrast, dark-channel haze, saturation, brightness) on the generated S5 synthetic half; assert per-condition median is within tolerance of the target (`synth_stats.json` medians). Tolerance: ±10% relative, or the fitted MAD, whichever is larger. Report a table.
- **Object visibility:** local-contrast retention (same metric as S3). Report per condition; flag any condition where median drops below 0.35.
- **Determinism:** rebuild and hash-compare.
- **Preview grids:** `results/summary/figures/synth_preview_bdd_s5_{fog,rain,snow,night}.png`.

PASS criteria: all of the above. Failures are documented, not hand-patched.

---

## 7. Training + evaluation

```bash
python src/synth/calibrate.py                         # writes synth_stats.json
python src/synth/build_s5.py --jobs 8
python src/synth/inspect_s5.py --dataset-name bdd_s5
python src/train.py --data configs/bdd_s5.yaml --model yolov8n.pt \
  --epochs 80 --batch 32 --seed 42 --aug default --exp S5
W=results/experiments/S5/train/weights/best.pt
python src/eval.py --weights $W --data configs/acdc_official.yaml \
  --name S5_acdc_official --per-weather --exp S5
for k in 0 1 2 3 4; do
  python src/eval.py --weights $W --data configs/acdc_cv5_fold$k.yaml \
    --name S5_acdc_fold$k --per-weather --exp S5
done
python src/eval.py --weights $W --data configs/bdd_src.yaml \
  --name S5_in_domain --exp S5
python src/aggregate.py && python src/visualize.py
```

Report S5 against S1 (anchor), S2 (photometric), S3 (hand-set weather), S4 (FDA), and T1aug (ceiling). Add a per-condition comparison table S2/S3/S5 on the official split.

---

## 8. Deliverables

| Path | Role |
|---|---|
| `src/synth/calibrate.py` | Fits stats → `synth_stats.json` |
| `src/synth/physics.py` | Four operators, param-injected, structurally identical to `weather.py` |
| `src/synth/build_s5.py` | Dataset builder |
| `src/synth/inspect_s5.py` | Inspector with closed-loop check |
| `results/analysis/synth_stats.json` | Versioned calibration output |
| `data/yolo/bdd_s5/` | 5k synthetic images |
| `splits/bdd_s5_train.txt` | 10k manifest |
| `configs/bdd_s5.yaml` | Training config |
| `results/experiments/S5/` | Train + eval + figures |
| `paper/results_notes.md` | Updated with S5 result |

Code is append-only on `stage_common.py`; S2/S3 modules are untouched.

---

## 9. Deferred to S6c (explicitly not in S5)

- Row-depth fog (monotone in image-row, correctly oriented so far = up).
- Local night illumination (headlights, streetlamps).
- Blur.
- Per-image parameter fitting.
- Gradient/noise-energy matching (keeps S5 clearly on the physics side of the style-transfer line).

If S5's night/fog results are the same as S3's, that is a **finding**, not a failure: it confirms the bottleneck is model structure, not parameter values, and motivates S6c.

---

## 10. Task checklist

- [ ] Approve scope locks (§2).
- [ ] Update `PROJECT.md` §5 to say blur is deferred beyond S5, not to S5.
- [ ] Write `calibrate.py`; produce `synth_stats.json`; review distributions for plausibility.
- [ ] Write `physics.py`; verify structural parity with `weather.py` on a fixed source.
- [ ] Write `build_s5.py`; generate `data/yolo/bdd_s5/`.
- [ ] Write `inspect_s5.py`; run all checks; confirm closed-loop tolerance.
- [ ] Train S5 (80 epochs).
- [ ] Evaluate on official + 5-fold + in-domain.
- [ ] Update `PROJECT.md` §9/§10, `HANDOFF.md` §7, `paper/results_notes.md`.
- [ ] Re-run `aggregate.py` + `visualize.py`.

---

## 11. Risks

- **Fitted statistics may be dominated by scene content.** Dark-channel haze and gradient statistics are scene-dependent; per-image → pooled → median mitigates this but doesn't eliminate it. The closed-loop check verifies generation, not detection; the detection result is the real test.
- **Night/fog likely to remain limited** for structural reasons. Pre-register the expectation that S5's gains, if any, are in rain and snow.
- **Overfitting to the pool.** The pool is 1,200 images; a single fitted distribution per condition has low capacity, so overfitting risk is low — but the design split must stay untouched to preserve the S6 policy story.
- **Structural drift between `weather.py` and `physics.py`.** The parity check in the inspector is the guard. If it can't be made to pass, that's a blocker, not a footnote.