# 04 — Frequency-Domain and Style-Statistic Methods

## Why this matters
**S4 is Fourier Domain Adaptation (FDA)**. The project previously ran global FDA (as B3)
and it did **not** beat the S1 anchor. This note records the exact method, the known
limitations, and the closest competing method (PAGen) so the S4 result is positioned, not
hand-waved.

## The methods

### FDA [Yang & Soatto, CVPR 2020] — `yang2020fda` (S4)
- **Mechanism:** FFT source and target; replace the low-frequency **amplitude** in a
  centered square of half-width fraction β with the target's, keep the **source phase**;
  inverse FFT. No training, no labels.
- **Why phase is kept:** phase encodes structure/geometry, so boxes stay valid. This is the
  same property we rely on for any amplitude-mixing augmentation.
- **Known limitation (verified in our earlier work):** the low-frequency amplitude swap also
  imposes the **target's coarse spatial layout** (its DC + low-frequency structure), not just
  a global colour cast. Across different driving scenes this yields radial/diagonal
  colour banding rather than a clean tint. A clean global style change needs **global colour
  statistics** (DC-only or mean+std / AdaIN), not a square amplitude swap.
- **Empirical status:** prior B3 runs: β=0.01 → 0.273, β=0.05 → 0.255 (official mAP50) vs
  S1/B2 0.269. Global FDA did not help. S4 re-runs it under the S-numbering with
  **β ∈ {0.05, 0.10}** for a fair, documented baseline.

### FACT [Xu et al., CVPR 2021] — `xu2021fact`
- Amplifies amplitude perturbation and mixes amplitude/phase across domains for **domain
  generalization**. Stronger than FDA for DG; a natural "next step" baseline if S4 is weak.
- Relevant because it separates the roles of amplitude (style) vs phase (semantics).

### PAGen [Du et al., 2025] — `du2025pagen`
- **Phase-guided amplitude generation** for domain-adaptive object detection. The closest
  prior work to the *retired* SM-WCFA idea; must be cited and differentiated. Our current
  study is a **comparative benchmark**, so we do not claim a new Fourier method — but any
  future Fourier claim must beat/position against PAGen.

### The importance of phase [Oppenheim & Lim, Proc. IEEE 1981] — `oppenheim1981phase`
- Classic evidence that phase carries structure. Cite to justify geometry preservation.

### AdaIN [Huang & Belongie, ICCV 2017] — `huang2017adain`
- Global feature-statistic (mean/std) transfer — the principled "clean tint" alternative to
  FDA. Relevant to illumination calibration in S2/S5 and to explaining why FDA's spatial
  square is the wrong operator for a global cast.

### MixStyle [Zhou et al., ICLR 2021] — `zhou2021mixstyle`
- Feature-statistic mixing across domains; the basis for feature-level style ideas (and the
  retired WSM direction). Cite only if feature mixing returns.

## Takeaways for our design
- **FDA's failure is explainable, not mysterious:** amplitude-square mixing injects target
  layout under source phase. Record this in the discussion as a *finding*.
- The S4 sweep is small (β ∈ {0.05, 0.10}); headline = best β, reported even if ≤ S1.
- Do **not** claim a Fourier novelty in the current study; if the comparative results
  motivate one, PAGen/FACT are the bar.

## What to cite where
- S4 method: D1.
- Fourier context/limitations: D2, D3, D4.
- Global-colour alternative: G6 (AdaIN).
