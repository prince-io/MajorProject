# 07 — Domain Generalization, Robustness, and Test-Time Adaptation

## Why this matters
Our S0/S1/S2/S3/S5 stages are effectively **zero-shot domain generalization** (no ACDC data
at training time), while S4 is unlabeled adaptation. Keeping this taxonomy precise is
required by `PROJECT.md` §6 and prevents over-claiming.

## Definitions (use exactly)
- **Zero-shot DG:** no target data at all → S0, S1, S2, S3, S5.
- **Unlabeled domain adaptation:** target images seen, labels never used → S4 (FDA), and the
  design of S6 if it uses target statistics.
- **Supervised fine-tuning:** target labels used → T1/T1aug (in-domain references).
- **Test-time adaptation:** adapt at inference on unlabeled test data → not in S0–S6 (future).

## Papers
### Domain randomization — `tobin2017domainrandomization`
- Randomize synthetic appearance so the model generalizes to reality. The framing for S3/S5.

### DANN — `ganin2016dann`
- Gradient-reversal domain-invariance; the adversarial DG/DA foundation.

### MixStyle — `zhou2021mixstyle`
- Feature-statistic mixing for DG; conceptually the "style mixing" family.

### TENT — `wang2021tent`
- Entropy-minimization test-time adaptation. A cheap, strong future add-on: adapt the trained
  detector on unlabeled ACDC test images at inference. Note it **breaks the zero-inference-cost
  property**, so treat as a separate contribution/ablation.

### Mean Teacher — `tarvainen2017meanteacher`; UMT — `deng2021umt`
- Consistency/teacher-student foundations for any semi-supervised target stage.

### STAC — `sohn2020stac`
- Semi-supervised detection with pseudo-labels; the simplest way to use ACDC train labels
  *without* full supervision (future work).

## Robustness to corruption (augmentation-as-robustness)
### AugMix — `hendrycks2020augmix`
- Mixing augmentations improves corruption robustness. Conceptual support for "synthetic
  weather aug improves real-weather robustness".

## Takeaways
- Keep the four terms above explicit in every table caption.
- S5 is **zero-shot** even though it uses ACDC **train images unlabeled for calibration** —
  carefully word this: calibration uses unlabeled statistics, which is closer to UDA for the
  calibration step but the detector never sees target images during training. Decide and state
  the exact rule in `PROJECT.md`.
- TENT is the most promising low-cost future extension if S6 plateaus.

## What to cite where
- Taxonomy and framing: `tobin2017domainrandomization`, `ganin2016dann`.
- Future work: `wang2021tent`, `sohn2020stac`.
- Robustness motivation: `hendrycks2020augmix`.
