# 05 — Weather Synthesis and Image Translation (S3 / S5 / S6)

## Why this matters
S3 (simple weather transforms), S5 (calibrated synthesis), and S6 (combinations) are the core
of the study. This note separates **hand-designed physics**, **learned translation**, and
**measurement-driven calibration** so S3 and S5 stay distinct (a requirement from the peer
review). Note: the implemented S5 calibrates **appearance** (see the revision note below).

## A. Physics-based rendering (S3, S5)

### Koschmieder visibility law — `koschmieder1924`
- `I = J·t + A·(1 − t)`, `t = exp(−β·d)`; the standard atmospheric-scattering fog model.
- **S3:** use it with a simple/constant transmission (fast).
- **S5:** estimate a **depth/transmission proxy** and fit `A`, `β` to measured target stats.

### Dark Channel Prior [He et al., CVPR 2009] — `he2009darkchannel`
- Gives a monocular haze/transmission estimate **without** a depth network — the practical
  tool for calibrating fog transmission on BDD images (S5).

### Rain rendering [Garg & Nayar, TOG 2006] — `garg2006rain`; **Vision and Rain** [IJCV 2007] — `garg2007visionrain`
- Rain streaks are **dynamic, depth-dependent, and illuminated**; naive 2-D streak overlays
  are a weak model. This is the argument for S3 (simple) vs S5 (sized/smeared by depth proxy,
  motion-blurred) being genuinely different.

### Synthetic fog protocol [Sakaridis et al., IJCV 2018] — `sakaridis2018syntheticfog`
- Establishes that physically-based fog synthesis on real driving scenes transfers to real
  fog. The template for S3/S5 and the strongest external validation of the approach.

## B. Learned image-to-image translation

### CycleGAN [Zhu et al., ICCV 2017] — `zhu2017cyclegan`
- Unpaired clear↔weather translation; the standard learned synthesis baseline.

### MUNIT [Huang et al., ECCV 2018] — `huang2018munit`
- Multimodal translation → diverse weather styles from one source image.

### CUT [Park et al., ECCV 2020] — `park2020cut`
- Contrastive unpaired translation; lighter alternative to CycleGAN.

**Positioning note:** learned translation can hallucinate content and needs adversarial
training on 6 GB; our study deliberately uses **non-learned, physics-structured** synthesis
for reproducibility and label safety. If the study needs a learned baseline, S5 vs CycleGAN
is the natural comparison.

## C. Detection-specific adverse-weather adaptation

### IA-YOLO [Liu et al., AAAI 2022] — `liu2022iayolo`
- Differentiable image-processing front-end tuned jointly with detection.
- **Contrast:** IA-YOLO changes the architecture (inference cost); our synthesis is
  **training-time only, zero inference cost**. Cite as the main alternative philosophy.

### Prior-based DAOD [Sindagi et al., ECCV 2020] — `sindagi2020prior`
- Uses fog/rain physical priors within an adaptation framework. The closest prior art to a
  physics-driven pipeline; differentiate by our per-condition calibration and comparative
  scope.

## D. Night and low-light

### Zero-DCE [Guo et al., CVPR 2020] — `guo2020zerodce`
### EnlightenGAN [Jiang et al., TPAMI 2021] — `jiang2021enlightengan`
- Reference-free/low-light enhancement. Evidence that **night is an illumination/energy
  problem**, so S2/S5 night handling is gamma/brightness/colour + noise, not a texture model.

## E. Sim-to-real framing

### Domain Randomization [Tobin et al., IROS 2017] — `tobin2017domainrandomization`
- The conceptual basis for S3/S5: randomize synthetic appearance so the model is robust to
  the real distribution. Cite to frame "synthetic weather → real ACDC".

## S3 vs S5 — the distinction to defend
> **Revised 2026-09-20:** per-parameter physics inversion was found non-identifiable across the
> BDD↔ACDC base-domain gap; the implemented S5 keeps S3's structure and calibrates **global
> appearance** (per-channel mean/std) to measured ACDC-train pool statistics. See `PROJECT.md` §5.
> The table below is the *aspirational* physics-calibration framing retained for prior-art
> positioning, not the implemented S5.

| | S3 (simple) | S5 (calibrated) |
|---|---|---|
| Parameters | hand-set, fixed | **measured from ACDC-train pool statistics** |
| Calibrated quantity (implemented) | — | **per-channel appearance (mean/std)**; physics structure unchanged |
| Fog | constant transmission | (aspirational) dark-channel transmission + fitted `A`, `β` |
| Rain | fixed streak kernel | (aspirational) depth-proxy streaks + fitted density |
| Snow | fixed particle sprite | (aspirational) fitted density/size + high-frequency energy |
| Night | fixed gamma/blue shift | (aspirational) fitted illumination + black-level + noise |
| Justification | fast baseline | measurement-driven calibration |

**The S5 contribution is the calibration to measured target statistics** (implemented as
appearance calibration), not the physics equations themselves. State this.

## What to cite where
- S3/S5 method: `koschmieder1924`, `he2009darkchannel`, `garg2006rain`, `garg2007visionrain`,
  `sakaridis2018syntheticfog`.
- Learned-synthesis related work: `zhu2017cyclegan`, `huang2018munit`, `park2020cut`.
- Detection-specific contrast: `liu2022iayolo`, `sindagi2020prior`.
- Night: `guo2020zerodce`, `jiang2021enlightengan`.
- Sim-to-real: `tobin2017domainrandomization`.
