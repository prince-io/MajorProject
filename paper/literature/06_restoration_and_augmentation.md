# 06 — Restoration and Augmentation Families

## Why this matters
**S2 is "S1 + photometric"** and **S3 uses simple weather transforms**. Both must be defined
against what Ultralytics already does (HSV, RandAugment, Mosaic, erasing), otherwise S2 is
confounded with S1. This note fixes that boundary and records the restoration context.

## A. What S1 already contains (do not duplicate in S2)
Ultralytics defaults (from the S1 `args.yaml`, formerly B2):
- `hsv_h 0.015`, `hsv_s 0.7`, `hsv_v 0.4` → **photometric colour jitter**
- `auto_augment: randaugment` → a learned policy that **already includes brightness/contrast/
  sharpness ops** [Cubuk et al., CVPRW 2020] — `cubuk2020randaugment`
- `mosaic 1.0` (from YOLOv4) — `bochkovskiy2020yolov4`
- `fliplr 0.5`, `translate 0.1`, `scale 0.5`, `erasing 0.4`, `close_mosaic 10`

**Consequence:** S2 must add operators **outside** this set — e.g. gamma correction, linear
brightness/contrast beyond HSV, Gaussian blur, additive noise, JPEG compression, CLAHE. S2 =
"generic sensor/photometric degradation", explicitly not a weather model.

## B. Augmentation literature
- **AutoAugment** [Cubuk et al., CVPR 2019] — `cubuk2019autoaugment`. Learned policy search.
- **RandAugment** [Cubuk et al., CVPRW 2020] — `cubuk2020randaugment`. The default in S1.
- **AugMix** [Hendrycks et al., ICLR 2020] — `hendrycks2020augmix`. Robustness-focused mixing.
- **TrivialAugment** [Müller & Hutter, ICCV 2021] — `muller2021trivialaugment`. Search-free
  strong augmentation; a sanity comparison for S2.
- **YOLOv4 / Mosaic** [Bochkovskiy et al., 2020] — `bochkovskiy2020yolov4`.
- **MixStyle** [Zhou et al., ICLR 2021] — `zhou2021mixstyle`. Feature-statistic mixing
  (style-level); context for condition-aware feature ideas.

## C. Restoration / low-level (context and restored-then-detect alternative)
- **MPRNet** [Zamir et al., CVPR 2021] — `zamir2021mprnet` (derain/dehaze).
- **Restormer** [Zamir et al., CVPR 2022] — `zamir2022restormer`.
- **FFA-Net** [Qin et al., AAAI 2020] — `qin2020ffanet` (dehazing).
- **Zero-DCE** [Guo et al., CVPR 2020] — `guo2020zerodce` (low-light).
- **EnlightenGAN** [Jiang et al., TPAMI 2021] — `jiang2021enlightengan` (low-light).

These are **pipeline alternatives** (restore then detect). Our S-study keeps detection
training-time only and adds no inference cost; cite restoration as a contrasting family and
as the basis for choosing which transforms mimic which degradation.

## D. Photometric operators we can safely use (label-preserving)
| Operator | Weather it approximates | Stage |
|---|---|---|
| Gamma / brightness / contrast | global illumination change | S2, night in S3/S5 |
| Gaussian blur | scattering/defocus, rain on lens | S2 |
| Additive Gaussian/Poisson noise | low light / sensor noise | S2 |
| JPEG compression | sensor/pipeline artefacts | S2 |
| Colour-temperature shift | illumination colour | S2, night |
| CLAHE | local contrast restoration (inverse) | S2 |
| Haze blend (Koschmieder) | fog | S3/S5 |
| Streak kernel + motion blur | rain | S3/S5 |
| Particle sprites + high-freq texture | snow | S3/S5 |

## Takeaways
- **S2 definition (locked in docs):** S1 + the operators above minus HSV/RandAugment overlap.
- All operators must be **geometry-preserving**; verify with `inspect_synth.py` (planned).
- Report S2 honestly: if it does not help, that is a result (standard augmentation already
  covers much of the photometric space).

## What to cite where
- S2 definition and augmentation related work: `cubuk2019autoaugment`, `cubuk2020randaugment`,
  `hendrycks2020augmix`, `muller2021trivialaugment`, `bochkovskiy2020yolov4`.
- Restoration contrast: `zamir2021mprnet`, `zamir2022restormer`, `qin2020ffanet`,
  `guo2020zerodce`, `jiang2021enlightengan`.
