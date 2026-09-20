# Results Notes (running)

> Paper-facing narrative of the S0–S6 comparative study. `PROJECT.md` §9 is the source of
> truth for numbers; this file turns them into draft-ready tables and sentences. Update it
> when each stage completes. Protocol/terminology per `PROJECT.md` §6.

## Protocol recap

- Model: YOLOv8n (locked), `seed=42`, `imgsz=640`, batch 32, AMP, `optimizer=auto`, `cos_lr=True`, `patience=30`, 80 epochs for augmented runs (40 for no-aug).
- Source: BDD100K clear/daytime, fixed 10k (5k A clear anchor + 5k B).
- **Every stage = S1's Ultralytics-default augmentation (online) + the stage's own change**; S0 is the only no-aug run.
- Target: ACDC official val (406), **the only scored set**; per-weather and per-class.
- Leakage control: a 400-image design split and the 1,200-image unlabeled pool are never scored; official val is scored once per stage.
- Setting labels: S0–S3 (and S5 if calibrated on BDD only) = **zero-shot DG**; S4 = unlabeled DA; S5 calibrated on target stats = unlabeled DA; T1/T1aug = supervised fine-tuning.

## Baseline ladder (official ACDC val / 5-fold / in-domain BDD val, mAP@50)

| Run | Recipe | Official | 5-fold | In-domain |
|---|---|---|---|---|
| **S0** | BDD no-aug — **floor** | 0.201 | 0.213 ± 0.011 | 0.376 |
| **S1** | BDD + Ultralytics defaults — **anchor** | **0.269** | 0.286 ± 0.014 | 0.491 |
| T1 | ACDC labels, no aug | 0.216 | 0.254 ± 0.020 | — |
| **T1aug** | ACDC + defaults — **ceiling** | **0.320** | 0.383 ± 0.021 | — |

Key baseline facts already established: augmentation (not target labels) is the dominant lever; S1 closes ~43% of the S0→ceiling gap; the remaining headroom is localized to **night/snow** and **truck/bus**. Note the epoch caveat: S0 is 40 epochs (floor), not an augmentation ablation; S1 is 80.

## S2 — generic photometric degradation (zero-shot)

**Design.** Split S1's 10k into **A (5,000 clear)** and **B (5,000 clear)**. S2 trains on **A clear + B degraded**; S1 trains on A + B clear, so **S1 is the exact control** (same scenes, objects, labels, count — only the appearance of the B half changes). Degradation = 1–3 random ops per image from {brightness, contrast, gamma, saturation, additive Gaussian noise} (narrow ranges; no blur/geometry), labels unchanged. In-training val = clear BDD.

### Main comparison (official ACDC val, 406)

| Metric | S1 (anchor) | S2 | Δ |
|---|---|---|---|
| mAP@50 | 0.2690 | **0.2833** | **+0.0143** |
| mAP@50-95 | 0.1559 | **0.1650** | **+0.0091** |
| Precision | 0.4613 | 0.4343 | −0.0270 |
| Recall | 0.2655 | 0.2715 | +0.0060 |
| In-domain BDD (mAP@50) | 0.4911 | 0.4920 | +0.0009 |
| 5-fold ACDC (mAP@50) | 0.2863 ± 0.0138 | 0.2933 ± 0.0161 | +0.0070 |

S2 captures ~28% of the remaining S1→ceiling headroom (official: 0.014 of 0.051).

### Per weather (official, 100–106 images each)

| Weather | S1 mAP@50 | S2 mAP@50 | Δ | S1 mAP@50-95 | S2 mAP@50-95 |
|---|---|---|---|---|---|
| fog | 0.4913 | 0.4849 | −0.0064 | 0.3268 | 0.3167 |
| night | 0.1929 | 0.1791 | −0.0138 | 0.0993 | 0.0941 |
| **rain** | 0.2442 | **0.2558** | **+0.0116** | 0.1314 | 0.1393 |
| snow | 0.2837 | 0.2819 | −0.0018 | 0.1593 | 0.1627 |

### Per class (official mAP@50 / mAP@50-95)

| Class | S1 | S2 | Δ (mAP@50) |
|---|---|---|---|
| person | 0.270 / 0.106 | 0.303 / 0.129 | +0.033 |
| rider | 0.071 / 0.038 | 0.109 / 0.054 | +0.038 |
| car | 0.701 / 0.443 | 0.703 / 0.451 | +0.002 |
| truck | 0.289 / 0.205 | 0.301 / 0.224 | +0.012 |
| bus | 0.169 / 0.105 | 0.206 / 0.105 | +0.037 |
| bicycle | 0.113 / 0.038 | 0.078 / 0.028 | −0.035 |

### Interpretation

- **First stage above the anchor**: +1.4 mAP@50 and +0.9 mAP@50-95 on the official split, with **no clear-weather forgetting**.
- **Where the gain lives:** essentially **rain**, plus the **rare classes** (rider/bus/person/truck). Fog, night and snow are flat-to-slightly-negative.
- **Precision falls, recall rises** — degrading input makes the detector slightly more willing to fire (fewer misses, more false positives), which nets a small mAP gain.
- **Takeaway for the study:** generic tone/exposure/noise changes help only partially; **structured weather (fog/snow/night) is not addressed by photometrics**, which is precisely the motivation for the weather-specific stages (S3/S5).

### Caveats to state

- **Single seed.** The 5-fold S2 gain (+0.7) sits inside the fold spread (±1.4–1.6); the effect is directionally consistent but not yet statistically strong. Do not over-claim S2 alone.
- **Bicycle drop (−3.5)** is on a small, noisy per-class cell — do not over-read.
- Comparisons must be read against the **matched S1 control** (A+B clear) per the pre-registered design, not stage-by-stage in isolation.

### Paper-ready sentences

> We introduce a photometric-degradation control (S2) that trains on the same 10,000 scenes as the anchor (S1), replacing half with mild photometric transforms (random subsets of brightness, contrast, gamma, saturation and Gaussian noise). On the official ACDC validation split, S2 improves mAP@50 from 0.269 to 0.283 (+1.4) and mAP@50-95 from 0.156 to 0.165 (+0.9), with no loss of in-domain (clear BDD) accuracy. The improvement is concentrated in rain (+1.2 mAP@50) and rare classes (rider, bus, person), while fog, night and snow remain essentially unchanged — indicating that generic photometric degradation does not model the structured effects that dominate those conditions.

## S3 — simple weather synthesis (zero-shot)

**Design.** Same A/B harness as S2: **A (5,000 clear) + B (5,000 weather)**, one condition per
image, **balanced 1,250 each** of fog/rain/snow/night, so **S1 (A + B clear) is the exact
control** and **S2 is the matched generic-photometric comparator**. Transforms are hand-set,
geometry-preserving, and use **no ACDC data**: constant-transmission Koschmieder fog
[Koschmieder, 1924]; directional rain streaks [Garg & Nayar, TOG 2006]; falling snow particles
(no accumulation); illumination night (brightness ×0.35–0.60, γ 1.0–1.4, tint, vignette). No
blur, no depth, no mixed conditions, no local light sources. Parameters are pre-registered in
`PROJECT.md` §5; training is the identical 80-epoch schedule (best epoch 65, no early stop).

**Dataset QC (2026-09-19).** 5,000/5,000 labels byte-identical; conditions exactly 1,250 each;
each synthetic differs from its source; deterministic. GT-box local-contrast retention (median):
fog 0.52, rain 0.90, snow 1.04, night 0.40. Previews:
`results/summary/figures/synth_preview_bdd_s3_{fog,rain,snow,night}.png`.

### Main comparison (official ACDC val, 406 / 5-fold / in-domain)

| Metric | S1 (anchor) | S2 (photometric) | **S3 (weather)** | T1aug (ceiling) |
|---|---|---|---|---|
| mAP@50 (official) | 0.2690 | **0.2833** | 0.2741 | 0.3196 |
| mAP@50-95 (official) | 0.1559 | **0.1650** | 0.1573 | 0.1957 |
| Precision (official) | 0.4613 | 0.4343 | **0.5315** | 0.5391 |
| Recall (official) | 0.2655 | **0.2715** | 0.2444 | 0.2889 |
| 5-fold mAP@50 | 0.2863 ± 0.0124 | 0.2933 ± 0.0144 | **0.2947 ± 0.0119** | 0.3827 ± 0.0187 |
| 5-fold mAP@50-95 | 0.1609 ± 0.0060 | 0.1643 ± 0.0087 | **0.1665 ± 0.0080** | 0.2214 ± 0.0106 |
| In-domain BDD mAP@50 | 0.4911 | 0.4920 | 0.4799 | — |
| Headroom captured (S1→ceiling) | — | 28% | 10% | 100% |

S3 clears the anchor but is **below S2 on the official primary metric** and **statistically tied
with S2 on 5-fold**. The two are the same 10k scenes; only the B-half transform differs.

### Per weather (official, mAP@50 / mAP@50-95; 100–106 images each)

| Weather | S1 | S2 | **S3** | T1aug (ceiling) |
|---|---|---|---|---|
| fog | 0.4913 / 0.3268 | 0.4849 / 0.3167 | 0.4893 / 0.2965 | 0.5233 / 0.3628 |
| night | 0.1929 / 0.0993 | 0.1791 / 0.0941 | 0.1799 / 0.0963 | 0.2247 / 0.1185 |
| rain | 0.2442 / 0.1314 | 0.2558 / 0.1393 | **0.2636 / 0.1546** | 0.3022 / 0.1801 |
| snow | 0.2837 / 0.1593 | 0.2819 / 0.1627 | **0.3002 / 0.1639** | 0.3176 / 0.1842 |

**Snow is the headline:** S3 is the best BDD-trained stage on snow (0.300 vs S1 0.284, S2 0.282)
and closes **~49%** of the S1→ceiling snow gap; snow **recall rises 0.232 → 0.319** at equal-or-
better precision. Rain also improves (0.244 → 0.264, ~33% of the gap). Night regresses and fog is
flat.

### Per weather — 5-fold mAP@50 (mean ± std)

| Weather | S1 | S2 | **S3** |
|---|---|---|---|
| fog | 0.4757 ± 0.0949 | 0.4695 ± 0.1095 | 0.4656 ± 0.1089 |
| night | 0.1957 ± 0.0402 | 0.1898 ± 0.0375 | 0.1816 ± 0.0298 |
| rain | 0.2989 ± 0.0709 | 0.2976 ± 0.0728 | **0.3056 ± 0.0658** |
| snow | 0.2997 ± 0.0218 | 0.3098 ± 0.0208 | **0.3062 ± 0.0193** |

Across folds the snow/rain ordering is stable in sign (S3 ≥ S1) but the S2↔S3 snow gap flips
(S2 0.310 vs S3 0.306 on 5-fold) — i.e. **S3's snow advantage over S2 is visible on the official
split but within noise across folds**.

### Per class (official mAP@50 / mAP@50-95)

| Class | S1 | S2 | **S3** | T1aug |
|---|---|---|---|---|
| person | 0.270 / 0.106 | 0.303 / 0.129 | 0.273 / 0.115 | 0.340 / 0.159 |
| rider | 0.071 / 0.038 | 0.109 / 0.054 | **0.117 / 0.051** | 0.145 / 0.067 |
| car | 0.701 / 0.443 | 0.703 / 0.451 | 0.693 / 0.442 | 0.704 / 0.462 |
| truck | 0.289 / 0.205 | 0.301 / 0.224 | **0.304 / 0.213** | 0.313 / 0.225 |
| bus | 0.169 / 0.105 | **0.206 / 0.105** | 0.167 / 0.087 | 0.283 / 0.203 |
| bicycle | 0.113 / 0.038 | 0.078 / 0.028 | 0.091 / 0.036 | 0.132 / 0.059 |

### Precision / recall mechanism

Overall S3 is **precision-heavy and recall-light** (P 0.461 → 0.532, R 0.266 → 0.244). That net
shift hides the condition detail: on **snow** S3 raises *both* P (0.477 → 0.478) and R
(0.232 → 0.319); on **rain** it raises P strongly (0.237 → 0.342) with a small recall dip
(0.307 → 0.274); the recall drag comes from **night** (P 0.382 → 0.260, R 0.224 → 0.272) and the
`person`/`bus` cells. So the structured operators genuinely help the structured conditions, while
the coarse night model spends precision it cannot recover.

### Interpretation

- **Above the anchor, on par with S2 in 5-fold, below S2 on the official split.** Hand-set
  weather structure does **not** beat generic photometric degradation on the primary metric; the
  S2↔S3 gap (~0.010) is inside the fold spread, so this is "no advantage", not "harm".
- **The gain is exactly where the physics is modelled: snow (strong) and rain.** Snow — S1's
  weakest closure — is where S3 delivers its best result and its clearest recall recovery.
- **The failure is exactly where the operator is too crude:** night (global dimming, no local
  illumination) and fog (constant transmission, no depth). These regress rather than help.
- **Direct motivation for S5:** fit the *same model families* to measured ACDC-train statistics
  (dark-channel transmission, illumination/noise) instead of hand-set ranges.

### Caveats to state

- **Single seed.** The S2↔S3 official difference (~0.010) is within the 5-fold spread
  (±0.012–0.014); do not claim S3 is better or worse than S2 from one seed.
- **Official vs 5-fold ordering differs** (official val is harder); the snow-over-S2 result is
  official-split-specific and within noise across folds.
- **Small per-class cells** (bus, bicycle, rider) are noisy — do not over-read single-class moves.
- **Mild in-domain forgetting** (0.491 → 0.480): aggressive weather augmentation costs a little
  clear-weather accuracy.

### Paper-ready sentences

> We replace half of the clear BDD training scenes with hand-set weather synthetics (S3:
> constant-transmission fog, directional rain streaks, falling snow particles, illumination
> night). On the official ACDC validation split S3 reaches mAP@50 0.274 (mAP@50-95 0.157) —
> above the clear-weather anchor (+0.5 mAP@50) but below the photometric control S2 (0.283);
> across the 5-fold split the two are statistically indistinguishable (0.295 vs 0.293).
> The effect is strongly condition-specific: S3 is the strongest BDD-trained stage on snow
> (0.300 mAP@50, closing ~49% of the anchor-to-ceiling snow gap, with snow recall rising from
> 0.232 to 0.319 at equal precision) and improves rain (0.264), whereas night and fog do not
> improve — consistent with global dimming and constant-transmission fog being too crude and
> motivating calibration (S5).

## What's next

- **S5 — appearance-calibrated synthesis (built 2026-09-20, revised from pre-registered):** S3's
  weather structure kept unchanged; S5 adds **target-appearance calibration** — per-channel
  mean/std measured on the ACDC-train unlabeled pool and matched at generation (unlabeled DA;
  saturation reported as a diagnostic). Parameter-level physics inversion was tested and found
  non-identifiable across the BDD↔ACDC base-domain gap, hence the appearance pivot. One-factor
  over S3; see `PROJECT.md` §5. **Awaiting train/eval.**
- **S5b — calibrated blur ablation (planned, after S5):** ancillary one-factor over S5;
  condition-specific blur (rain directional motion, fog/snow defocus, night none) with strength
  calibrated to the ACDC-train pool; screened by a preview + a design-split sensitivity probe,
  trained only if warranted. **S5b↔S5 is the clean comparison** (it is not folded into S5).
- **S4 FDA** online (`src/aug/fda.py` + hook); β chosen when we reach S4. S4↔S5 (same pool) is the
  controlled equal-access comparison.
- Decide per the reviewer: S4 β set (restore 0.01?), seeds (3 for S1/S6).
- Add per-class **official** numbers for all stages and the zero-shot vs unlabeled-DA setting column to the main table.

## Figure / table inventory

- `results/summary/figures/comparison_overall.png`, `comparison_per_weather.png`, `class_weather_<exp>.png`.
- `results/summary/{summary.json,per_class.csv,per_class.md}`.
- S3 result figures: `results/summary/figures/comparison_overall.png` / `comparison_per_weather.png`
  (S3 included), `class_weather_S3_acdc.png`, `results/experiments/S3/figures/{bars_S3_acdc_official,training_curves}.png`.
- `results/summary/figures/synth_preview_bdd_s2.png` (qualitative S2 samples).
- `results/summary/figures/synth_preview_bdd_s3_{fog,rain,snow,night}.png` (qualitative S3 samples).
- `results/summary/synth_report_bdd_s3.txt` (S3 inspector report incl. object-visibility).
