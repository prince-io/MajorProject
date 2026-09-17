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

## What's next

- **S3 simple weather** (hand-set rainbow/fog/snow/night), then **S5 calibrated physics**, both offline using the same A/B harness; **S4 FDA** online.
- Decide per the reviewer: S4 β set (restore 0.01?), primary metric (mAP@50-95), seeds (3 for S1/S6), S5 calibration source.
- Add per-class **official** numbers for all stages and the zero-shot vs unlabeled-DA setting column to the main table.

## Figure / table inventory

- `results/summary/figures/comparison_overall.png`, `comparison_per_weather.png`, `class_weather_<exp>.png`.
- `results/summary/{summary.json,per_class.csv,per_class.md}`.
- `results/summary/figures/synth_preview_bdd_s2.png` (qualitative S2 samples).
