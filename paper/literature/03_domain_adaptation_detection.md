# 03 — Domain Adaptation for Object Detection

## Why this matters
S4 (FDA) is an **unlabeled domain-adaptation** method, and the broader DAOD literature is
the main comparator for any "does adaptation help" claim. It is also the space where our
comparative study must carve a defensible niche.

## Taxonomy of prior work

### 1. Adversarial feature alignment
- **Domain Adaptive Faster R-CNN** [Chen et al., CVPR 2018] — `chen2018domainadaptivefasterrcnn`.
  Image-level + instance-level domain classifiers on a two-stage detector.
- **SW-DA** [Saito et al., CVPR 2019] — `saito2019swda`. Weak (global) + strong (local)
  alignment; widely used baseline.
- **DANN** [Ganin et al., JMLR 2016] — `ganin2016dann`. The gradient-reversal foundation.

### 2. Teacher–student / self-training
- **UMT** [Deng et al., CVPR 2021] — `deng2021umt`. Unbiased mean teacher for DAOD.
- **STAC** [Sohn et al., 2020] — `sohn2020stac`. Semi-supervised detection.
- **Mean Teacher** [Tarvainen & Valpola, NeurIPS 2017] — `tarvainen2017meanteacher`. Basis.

### 3. Prior/physics-informed adaptation (closest to us)
- **Prior-based DA for Hazy and Rainy Conditions** [Sindagi et al., ECCV 2020]
  — `sindagi2020prior`. **Key comparator:** uses atmospheric-scattering and rain-physics
  priors for DAOD. Our S3/S5 physics-based synthesis must be positioned relative to this.
- **DA-RAW** [Jeon et al., ICRA 2024] — `jeon2024daraw`. **Closest protocol:** disentangles
  weather vs style gap for real adverse weather. Use for protocol and comparison framing.
- **Adaptation under Foggy Weather** [Li et al., WACV 2023] — `li2023foggydaod`.
- **AWADA** [Menke et al., WACV 2023] — `menke2023awada`. Attention-weighted adversarial DAOD.
- **Multimodal adverse-weather UDA** [Eskandar et al., IV 2022] — `eskandar2022multimodal`.
- **Similarity-based alignment** [Rezaeianaran et al., ICCV 2021] — `rezaeianaran2021seeking`.
- **D-YOLO** [Chu, 2024] — `chu2024dyolo`. YOLO-based adverse-weather detector.

### 4. Strong graph/semantic matching
- **SIGMA** [Li et al., CVPR 2022] — `li2022sigma`. High-complexity DAOD; sets a performance
  bar but is orthogonal to training-time augmentation.

## Where our study sits
We compare **training-time data/adaptation strategies** (S2 photometric, S3 simple weather,
S4 FDA, S5 appearance calibration, S6 combined) on a **single unified YOLO detector**, scored per
weather. This is **not** a new adversarial DA method. The defensible contributions are:
1. a controlled per-condition comparison (which strategy helps which weather);
2. the **calibration** angle of S5 (physics parameters fit to measured target statistics);
3. the negative/limited result for FDA in this setting (honest reporting).

## Key caveats to state in the paper
- Most adversarial DAOD uses **two-stage** detectors and **synthetic** target domains
  (Cityscapes→Foggy); our setting is a small one-stage detector on **real** ACDC.
- DAOD methods assume a target **unlabeled pool at training time**; S0–S3/S5 are **zero-shot**
  (no target images), which changes the comparison axis. Be explicit about this.
- Labeled target of 1,600 is tiny; adaptation methods requiring stable target statistics may
  underperform for data reasons, not method reasons.

## What to cite where
- Related work (adaptation): C1–C13.
- S4 framing: D1 + C1/C2 (adversarial DA context).
- Positioning/novelty: C5, C6, D3.
