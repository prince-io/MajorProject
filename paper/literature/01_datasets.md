# 01 — Datasets and Benchmarks

## Why this matters
Our entire study is defined by a **source** (clear BDD100K) and a **target** (real ACDC).
Every claim about "generalization" or "sim-to-real" depends on these datasets' exact
properties, and reviewers will check that we did not leak target labels.

## Target domain

### ACDC [Sakaridis et al., ICCV 2021] — `sakaridis2021acdc`
- **What:** 4,006 images across **fog / rain / night / snow**, with pixel-level and
  detection annotations and *correspondences* (near-identical scenes across conditions).
- **Our use:** detection only (no segmentation labels needed). Train 1,600 (400/weather),
  val 406, test 2,000 (no GT). The **406 official val is our only scored benchmark**.
- **Key property:** real adverse weather, not synthetic — this is what makes our
  synthetic-BDD study a genuine **sim-to-real** test.
- **Caveats:** small (406 val) → report per-weather/per-class and avoid over-reading rare
  classes; a single seed; state the no-CV-error-bar limitation.
- **Citations:** dataset + protocol; ACDC official web page for the split.

### Foggy Cityscapes / synthetic fog [Sakaridis et al., IJCV 2018] — `sakaridis2018syntheticfog`
- **What:** physically-based fog synthesis on Cityscapes, used to train and test fog models.
- **Our use:** the archetype for S3/S5. It establishes the standard of using a **physical
  scattering model** (Koschmieder) rather than ad-hoc haze, and it validates synthetic→real
  transfer for fog. Cite when justifying S5's physics and as prior art for synthetic-weather
  training.

### CADC / CADC+ — `pitropov2021cadc`, `tang2025cadcplus`
- **What:** CADC = real Canadian adverse (snow) driving; CADC+ = paired clear/snow.
- **Our use:** evidence that real adverse data is scarce and that paired data (the ideal for
  adaptation) is only recently available; supports the motivation for BDD→ACDC.

### DAWN — `kenk2020dawn`
- Small real adverse-weather detection set (rain/fog/snow). Cite as further evidence that
  real adverse detection benchmarks are few, hence ACDC is the right target.

### NightOwls — `neumann2019nightowls`
- Night pedestrian detection. Supports treating **night as its own low-light problem**
  (consistent with our plan for S2/S5 night handling).

## Source domain

### BDD100K [Yu et al., CVPR 2020] — `yu2020bdd100k`
- **What:** 100k driving images with attributes (`weather`, `scene`, `timeofday`) and boxes.
- **Our use:** filter `weather == clear` AND `timeofday == daytime` → 12,454 pool → 10k
  train / 2k val (seed 42). We deliberately keep the source **single-condition** to define a
  clean domain gap.
- **Caveat:** `rider` has no COCO equivalent → randomly-initialized head (document).

### Cityscapes [Cordts et al., CVPR 2016] — `cordts2016cityscapes`
- Not our data, but the origin of most DA-detection protocols (Cityscapes→Foggy, etc.).
  Cite for related-work context and metric conventions.

## Multimodal adverse-weather data (context only)

### Seeing Through Fog [Bijelic et al., CVPR 2020] — `bijelic2020seeingthroughfog`
- Camera+LiDAR+gated-NIR fog benchmark. Cite to contrast **camera-only** and to show fog
  perception is an active area.

## Takeaways for our design
- The **single scored oracle is ACDC official val (406)** — hence the design split (§6) to
  keep method selection off it.
- Per-weather breakdowns are essential; per-class only where counts support it.
- Our source is intentionally narrow (clear/daytime); say so explicitly and do not claim
  all-weather source generalization.

## What to cite where
- Datasets subsection: A1, A2, A3, A4, A5, A6.
- "Real adverse data is scarce" motivation: A5, A6, A7, A8.
- Night as illumination: A9, I6.
