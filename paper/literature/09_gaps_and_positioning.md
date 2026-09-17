# 09 — Gaps, Saturation, and Positioning

> This is the novelty map for the **S0–S6 comparative study**. It is deliberately skeptical:
> most of this space is saturated. Use it to avoid over-claiming and to decide, after the
> per-condition results exist, whether the study supports a paper.

## What is already saturated (do **not** claim as novel)
- "Apply augmentation to adverse weather" — broad literature (S3/restoration/augmentation).
- "Use FDA for adverse-weather adaptation" — FDA is standard and already applied; our S4 only
  *measures* it in a controlled setting.
- "Synthetic fog on Cityscapes → real fog" — established by `sakaridis2018syntheticfog`.
- "Adversarial DAOD" — heavily worked (C1–C4, C7, C8, C13).
- "Phase-guided / Fourier amplitude generation for DAOD" — now claimed by
  `du2025pagen`; the retired SM-WCFA idea is **no longer novel**.
- "Feature-statistic mixing" — `zhou2021mixstyle`, `huang2017adain`.

## Gaps that plausibly remain
1. **Per-condition, controlled comparison on real ACDC with one unified YOLO detector.**
   Most work is single-condition, synthetic-target, or two-stage. A clean
   "which strategy helps fog vs rain vs snow vs night, at equal compute" table is useful and
   currently undersupplied.
2. **Calibrated (measurement-driven) physics synthesis.** Fitting the synthesis parameters
   to *measured target statistics* (colour, RMS contrast, dark-channel haze, noise/gradient
   energy) rather than hand-setting them. This is the S5 angle and the most defensible
   methodological contribution.
3. **Hard-condition targeting / null results.** Reporting that standard photometric (S2) and
   global Fourier (S4) do **not** move the needle, and localizing *why* per condition, is a
   legitimate contribution (the project already has one negative Fourier result).
4. **Zero-inference-cost condition-aware policy.** A training-only policy (S6b) with a clear
   accounting of gains per condition, no architecture change.

## Candidate paper stories (decide after results)
- **A. Comparative study:** "Which augmentation/adaptation strategy helps which adverse
  weather? A controlled BDD→ACDC study." Workshop/ITSC/IV level; robust and honest.
- **B. Calibration paper:** lead with S5 (measured-statistic-calibrated synthesis) and use
  S0–S4/S6 as controls. Stronger novelty; must beat the hand-set S3 to be convincing.
- **C. Negative-result/analysis paper:** "When does weather synthesis help detection?" if the
  gains are small but the per-condition explanation is crisp.
- **Fallback:** B.Tech thesis only, no paper, if gains are inconclusive.

## Critical protocol caveat to resolve
- **S5 uses ACDC-train images (unlabeled) to calibrate parameters.** That means S5 is **not
  strictly zero-shot** — it is closer to unlabeled DA for the calibration step. Decide and
  state the rule precisely:
  - calibration-only statistics (colour/contrast/noise) = arguably mild UDA; or
  - calibrate on ACDC **design split** only, keeping the scored val untouched; or
  - calibrate on BDD **adverse-looking** stats only (fully zero-shot).
  This choice must be pre-registered in `PROJECT.md` before S5 runs.

## Unverified tags carried from earlier notes
- **"MIC (CVPR 2023)"** — no matching title found via the arXiv API. Likely refers to the
  Hnewa & Radha line (`hnewa2021multiscale` / `hnewa2022integrated`). **Do not cite as MIC.**
- **"ViSGA (ICCV 2021)"** — no matching title found. The verified ICCV'21 similarity-alignment
  paper is `rezaeianaran2021seeking`. **Do not cite as ViSGA.**

## Positioning statements to draft
- vs **PAGen** (`du2025pagen`): we do not propose a new Fourier generator; we benchmark
  Fourier adaptation against simpler/physics alternatives under one detector.
- vs **Prior-based DAOD** (`sindagi2020prior`): they embed priors in an adversarial adaptation
  framework; we calibrate a training-data synthesiser and measure per-condition detection.
- vs **IA-YOLO** (`liu2022iayolo`): they add an inference-time restoration module; we keep
  inference unchanged (zero cost) and only change training data.
- vs **DA-RAW** (`jeon2024daraw`): same real-adverse target spirit, but they focus on DAOD
  method; we focus on synthesis/augmentation comparison with a locked protocol.

## Open reading tasks (to close the gap checks)
- [ ] Confirm whether "Bridging Clear and Adverse Driving Conditions" (`shapiro2025bridging`)
      already performs a per-condition augmentation comparison.
- [ ] Check PAGen's exact evaluation (conditions, detector) to position S4 precisely.
- [ ] Find the true "MIC" reference if it exists; otherwise drop from the manuscript.
- [ ] Verify venues for entries marked "confirm" in `references.bib`.
