# S3 Implementation Recommendation Document

**Project:** Synthetic Adverse Weather Augmentation for YOLOv8n — Stage S3

**Version:** 1.0

**Target audience:** AI coding agent / implementation engineer

**Prerequisites:** Access to the S0–S6 comparative study framework, clear BDD100K subset (5k images), ACDC official validation set (406 images), and the S1 baseline configuration.

---

## 1. Executive Summary

S3 is the first stage in the S0–S6 ladder that models **weather explicitly** using hand-set, geometry-preserving photometric transforms. The goal is to determine whether weather-structured synthetic data improves YOLOv8n detection on real adverse weather (ACDC) more effectively than generic degradation (S2). S3 also produces per-condition synthetic data that later stages (S6a/S6b) will consume.

**Locked decisions:**
- **Training data composition:** 5k clear BDD + 5k weather-synthesized BDD (10k total).
- **Condition allocation:** Balanced — 1,250 images each for rain, fog, snow, night.
- **One condition per image:** No mixed conditions.
- **Labels:** Byte-copied (geometry-preserving transforms only).
- **Synthesis:** Offline, pre-generated under `data/yolo/bdd_s3/`.
- **Implementation:** Custom `weather.py` module using Albumentations and Automold primitives, plus custom night transform.
- **S3/S5 boundary:** S3 uses hand-set parameters; S5 fits parameters to ACDC-train statistics. S3 must not use any ACDC data.

**Expected outcome:** A single 80-epoch YOLOv8n training run with the same command as S1 except `--data configs/bdd_s3.yaml`.

---

## 2. Design Decisions and Rationale

### 2.1 Condition Allocation (Q5.1 — Resolved)

**Decision:** Balanced allocation — 1,250 images per weather condition.

**Rationale:**
- ACDC train is 400/400/400/400, so balanced maps naturally to the test distribution.
- Symmetric allocation avoids the appearance of tuning to the test set.
- Difficulty-weighted allocation (more snow/night) risks conflating S3 with S6 condition-aware policies.
- Continuous sampling (each image gets a random condition) is effectively balanced in expectation but less interpretable in the paper.

**Implementation:** Deterministic assignment using seed 42. Shuffle the 5k source image list, then assign conditions cyclically: indices 0–1249 → rain, 1250–2499 → fog, 2500–3749 → snow, 3750–4999 → night.

### 2.2 One Condition Per Image (Q6.2 — Resolved)

**Decision:** Each synthetic image receives exactly one weather condition. No rain+fog, snow+night, or other combinations.

**Rationale:**
- ACDC has clean per-weather labels; combined conditions cannot be scored separately.
- Mixed conditions blur attribution in the per-weather analysis.
- Combinations belong to S6 (condition-aware policy).

### 2.3 Night Simulation (Q6.3 — Resolved)

**Decision:** Simple brightness reduction + gamma shift + subtle color tint. No local illumination (headlights, streetlamps) in v1.

**Rationale:**
- Local illumination is illumination-source modelling and crosses into S5 territory.
- Naive darkening creates a known realism gap ("dim daytime" look), but this is an honest limitation of S3 and a finding for S5 to address.
- If night performance is weak, that is itself a useful result for the paper.

**Parameters:**
- Brightness multiplier: ×0.4–0.7 (uniform per image).
- Gamma shift: γ ∈ 0.7–0.95.
- Color tint: subtle warm (R+5, B−5) or cool (R−5, B+5) shift, randomly chosen.
- Optional: mild vignette (darkening at edges) to simulate reduced peripheral illumination — use Albumentations `RandomVignette`.

### 2.4 Snow Realism (Q6.4 — Resolved)

**Decision:** Falling snow particles only. No ground accumulation or surface whitening in v1.

**Rationale:**
- Ground whitening is spatial and risks obscuring objects.
- Accumulated snow is a depth-aware effect that belongs to S5.
- Falling particles alone provide the high-frequency occlusion challenge that tests detector robustness.

**Parameters (conservative starting ranges):**
- Snow particle radius: 2–6 px.
- Density: 0.005–0.02 (fraction of affected pixels).
- Alpha: 0.5–0.7.
- Brightness lift: ×1.0–1.15.
- Contrast reduction: ×0.85–1.0.
- Mild desaturation: ×0.8–1.0.

### 2.5 Depth-Dependence (Q6.5 — Resolved)

**Decision:** Accept uniform-depth approximation for S3. No depth maps or depth-aware effects.

**Rationale:**
- Depth-aware fog (denser at distance) and depth-aware rain (shorter streaks far away) require monocular depth estimation, which adds complexity and crosses into S5.
- Uniform-depth is the honest simple baseline for S3.
- Depth-aware effects belong to S5 if feasible.

### 2.6 Blur Handling (Q6.7 — Resolved)

**Decision:** Exclude Gaussian/defocus blur from S3. Leave blur entirely to S2.

**Rationale:**
- S2 uses generic photometric degradation including Gaussian/defocus blur.
- Including blur in S3 would confound the S2 vs S3 comparison.
- Rain streaks themselves add high-frequency structure; no additional blur needed.
- If rain-specific motion blur is desired later, it must be clearly directional and absent from S2.

### 2.7 Rain Streak Realism (Q6.6 — Resolved)

**Decision:** Sparse, directional, conservative streaks. Define density as number of streaks per image, not an ambiguous probability.

**Parameters:**
- Number of streaks: 50–200 per 640×640 image.
- Streak length: 10–30 px.
- Streak width: 1–2 px.
- Slant angle: 70–85° from horizontal (near-vertical, slight tilt).
- Alpha: 0.3–0.5.
- Contrast reduction: ×0.8–0.95.
- No wet-road darkening in v1.

**Rationale:** Upper ranges from the original brief (density 0.05, alpha 0.6) are too aggressive and risk destroying small objects (pedestrians, cyclists). Start conservative; the inspector will verify object visibility.

### 2.8 Source Image Pairing (Red Flag — Resolved)

**Decision:** Use the **same 5k source images** for both the clear half and the synthetic half.

**Rationale:**
- Each scene appears once clear and once under one weather condition.
- This makes S2 vs S3 cleaner (both use 5k clear + 5k transformed from the same source pool).
- Helps the model learn weather invariance more effectively.
- Document this choice explicitly in the paper.

**Implementation:** The 5k source images are split into two groups of 2,500. Group A (2,500 images) is used directly as clear training data. Group B (2,500 images) is used as the source for synthetic weather generation. Wait — this would mean only 2,500 unique scenes are augmented, producing 5k synthetic images. Alternative: use all 5k source images for both clear and synthetic, resulting in 5k clear + 5k synthetic = 10k, but then the clear and synthetic halves share the same 5k scenes.

**Correction:** The brief states "5k clear BDD + 5k weather-synthesized BDD." The cleanest interpretation is:
- 5k clear images (from the BDD source pool).
- 5k synthetic images generated from the **same 5k source images** (one synthetic per source image).
- This means the model sees each scene twice: once clear, once weather-transformed.
- Total dataset: 10k images (5k clear + 5k synthetic).

This is the recommended pairing. Document it clearly.

---

## 3. Implementation Plan

### 3.1 Directory Structure

```
project/
├── configs/
│   └── bdd_s3.yaml              # Dataset config for YOLOv8
├── data/
│   └── yolo/
│       └── bdd_s3/
│           ├── images/
│           │   ├── train/       # 10k images (5k clear + 5k synthetic)
│           │   └── val/         # bdd_src_val.txt images (2k clear)
│           ├── labels/
│           │   ├── train/       # Byte-copied from clear BDD
│           │   └── val/         # Byte-copied from clear BDD val
│           ├── metadata/
│           │   └── synth_metadata.csv   # Per-image condition, params, seed
│           └── synth_manifest.json      # Source image list, condition mapping
├── src/
│   └── synth/
│       ├── __init__.py
│       ├── weather.py           # Main transform module
│       ├── rain.py              # Rain transform
│       ├── fog.py               # Fog transform
│       ├── snow.py              # Snow transform
│       ├── night.py             # Night transform
│       ├── dataset_builder.py   # Builds bdd_s3 dataset
│       └── inspect_synth.py     # Dataset inspector
└── runs/
    └── detect/
        └── bdd_s3/              # Training outputs
```

### 3.2 `src/synth/weather.py` — Core Transform Interface

```python
"""
weather.py — Main interface for S3 weather transforms.

Each weather transform is a pure function with the signature:
    transform(image: np.ndarray, params: dict, rng: np.random.Generator)
        -> tuple[np.ndarray, dict]

No hidden state. No file I/O. No ACDC data.
"""

from typing import Callable
import numpy as np

# Registry of available weather transforms
WEATHER_TRANSFORMS: dict[str, Callable] = {}

def register(name: str):
    def decorator(fn):
        WEATHER_TRANSFORMS[name] = fn
        return fn
    return decorator

def apply_weather(
    image: np.ndarray,
    condition: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """
    Apply a single weather condition to an image.

    Args:
        image: RGB uint8 image (H, W, 3).
        condition: One of 'rain', 'fog', 'snow', 'night'.
        rng: NumPy random generator (seeded for reproducibility).

    Returns:
        (transformed_image, params): The transformed image and the
        sampled parameters (for metadata logging).
    """
    if condition not in WEATHER_TRANSFORMS:
        raise ValueError(f"Unknown condition: {condition}")

    transform_fn = WEATHER_TRANSFORMS[condition]
    return transform_fn(image, rng)
```

### 3.3 Rain Transform — `src/synth/rain.py`

**Libraries:** Albumentations `RandomRain` + custom contrast reduction.

**Parameters:**
- `slant_range`: (70, 85) degrees from horizontal.
- `drop_length`: (10, 30) px.
- `drop_width`: (1, 2) px.
- `drop_color`: (200, 200, 200) — light grey.
- `blur_value`: 0 (no blur in S3).
- `brightness_coefficient`: 0.8–0.95 (contrast reduction).
- `rain_type`: "default" (not drizzle/heavy/torrential presets).
- Number of streaks: 50–200 (controlled via custom streak generation or Albumentations' internal density).

**Implementation notes:**
- Albumentations `RandomRain` has a `rain_type` parameter with presets "drizzle", "heavy", "torrential". Use "default" and override with custom parameters.
- If Albumentations' internal density control is insufficient, generate streaks manually using OpenCV line drawing on a separate layer, then alpha-blend.
- Apply contrast reduction *after* rain streaks using `RandomBrightnessContrast` or a simple linear scaling.

**Pseudo-code:**
```python
@register("rain")
def apply_rain(image, rng):
    import albumentations as A

    # Sample parameters
    slant = rng.uniform(70, 85)
    drop_length = int(rng.uniform(10, 30))
    drop_width = int(rng.uniform(1, 2))
    alpha = rng.uniform(0.3, 0.5)
    contrast = rng.uniform(0.8, 0.95)

    transform = A.Compose([
        A.RandomRain(
            slant_range=(slant, slant),
            drop_length=drop_length,
            drop_width=drop_width,
            drop_color=(200, 200, 200),
            blur_value=0,
            brightness_coefficient=contrast,
            rain_type="default",
            p=1.0,
        ),
    ])

    result = transform(image=image)["image"]

    params = {
        "slant": slant,
        "drop_length": drop_length,
        "drop_width": drop_width,
        "alpha": alpha,
        "contrast": contrast,
        "num_streaks": "auto",
    }
    return result, params
```

**Note on alpha:** Albumentations `RandomRain` does not expose an alpha parameter directly. If fine-grained alpha control is needed, implement rain streaks manually with OpenCV:
1. Create a blank overlay (H, W, 3) of zeros.
2. For each streak: draw a line with random start point, length `drop_length`, angle `slant`, color `(200, 200, 200)`.
3. Apply Gaussian blur to the overlay with small kernel (1–2 px).
4. Alpha-blend: `result = image * (1 - alpha) + overlay * alpha`.

### 3.4 Fog Transform — `src/synth/fog.py`

**Libraries:** Albumentations `RandomFog` + custom contrast/brightness adjustment.

**Parameters:**
- `fog_coef`: 0.2–0.6 (alpha blend strength).
- `alpha_coef`: 0.08–0.2 (fog density variation).
- Contrast reduction: ×0.6–0.9.
- Brightness lift: ×1.0–1.2.
- Mild desaturation: ×0.7–1.0.

**Implementation notes:**
- Albumentations `RandomFog` uses patch-based fog simulation. The `fog_coef` parameter controls the intensity.
- Apply fog first, then contrast/brightness adjustments.
- Desaturation can be applied via `A.HueSaturationValue` with saturation_shift negative.

**Pseudo-code:**
```python
@register("fog")
def apply_fog(image, rng):
    import albumentations as A

    fog_coef = rng.uniform(0.2, 0.6)
    contrast = rng.uniform(0.6, 0.9)
    brightness = rng.uniform(1.0, 1.2)
    saturation = rng.uniform(0.7, 1.0)

    transform = A.Compose([
        A.RandomFog(
            fog_coef_range=(fog_coef, fog_coef),
            alpha_coef=rng.uniform(0.08, 0.2),
            p=1.0,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(brightness - 1, brightness - 1),
            contrast_limit=(contrast - 1, contrast - 1),
            p=1.0,
        ),
        A.HueSaturationValue(
            sat_shift_limit=(int((saturation - 1) * 100), int((saturation - 1) * 100)),
            p=1.0,
        ),
    ])

    result = transform(image=image)["image"]

    params = {
        "fog_coef": fog_coef,
        "contrast": contrast,
        "brightness": brightness,
        "saturation": saturation,
    }
    return result, params
```

### 3.5 Snow Transform — `src/synth/snow.py`

**Libraries:** Albumentations `RandomSnow` + custom brightness/contrast/desaturation.

**Parameters:**
- `snow_point_range`: (0.1, 0.3) — snow intensity threshold.
- `brightness_coeff`: 1.0–1.15.
- Contrast reduction: ×0.85–1.0.
- Desaturation: ×0.8–1.0.
- Particle density: 0.005–0.02.

**Implementation notes:**
- Albumentations `RandomSnow` uses a snow point threshold and brightness coefficient.
- The `snow_point_range` controls the intensity; lower values mean more snow.
- Falling particles are simulated by the transform's internal logic.
- No ground accumulation in S3.

**Pseudo-code:**
```python
@register("snow")
def apply_snow(image, rng):
    import albumentations as A

    snow_point = rng.uniform(0.1, 0.3)
    brightness = rng.uniform(1.0, 1.15)
    contrast = rng.uniform(0.85, 1.0)
    saturation = rng.uniform(0.8, 1.0)

    transform = A.Compose([
        A.RandomSnow(
            snow_point_range=(snow_point, snow_point),
            brightness_coeff=brightness,
            p=1.0,
        ),
        A.RandomBrightnessContrast(
            contrast_limit=(contrast - 1, contrast - 1),
            p=1.0,
        ),
        A.HueSaturationValue(
            sat_shift_limit=(int((saturation - 1) * 100), int((saturation - 1) * 100)),
            p=1.0,
        ),
    ])

    result = transform(image=image)["image"]

    params = {
        "snow_point": snow_point,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
    }
    return result, params
```

### 3.6 Night Transform — `src/synth/night.py`

**Libraries:** Custom implementation using OpenCV + NumPy. No suitable Albumentations transform exists for full night simulation.

**Parameters:**
- Brightness multiplier: ×0.4–0.7.
- Gamma shift: γ ∈ 0.7–0.95.
- Color tint: warm (R+5, B−5) or cool (R−5, B+5).
- Optional vignette: `A.RandomVignette` with strength 0.1–0.3.

**Implementation notes:**
- Convert to float, apply brightness scaling, then gamma correction, then color tint, then clip.
- Gamma correction: `output = (input / 255.0) ** gamma * 255.0`.
- Color tint: add/subtract small values to R and B channels.
- Vignette: use Albumentations `RandomVignette` or implement manually with a radial mask.

**Pseudo-code:**
```python
@register("night")
def apply_night(image, rng):
    import cv2
    import numpy as np

    brightness = rng.uniform(0.4, 0.7)
    gamma = rng.uniform(0.7, 0.95)
    tint = rng.choice(["warm", "cool"])

    # Brightness scaling
    img = image.astype(np.float32) * brightness

    # Gamma correction
    img = np.clip(img, 0, 255)
    img = (img / 255.0) ** gamma * 255.0

    # Color tint
    if tint == "warm":
        img[:, :, 0] += 5   # R
        img[:, :, 2] -= 5   # B
    else:
        img[:, :, 0] -= 5   # R
        img[:, :, 2] += 5   # B

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Optional vignette
    vignette_strength = rng.uniform(0.1, 0.3)
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols * 0.5)
    kernel_y = cv2.getGaussianKernel(rows, rows * 0.5)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    mask = 1 - vignette_strength * (1 - mask)
    mask = np.dstack([mask] * 3)
    img = (img * mask).astype(np.uint8)

    params = {
        "brightness": brightness,
        "gamma": gamma,
        "tint": tint,
        "vignette_strength": vignette_strength,
    }
    return img, params
```

### 3.7 Dataset Builder — `src/synth/dataset_builder.py`

**Responsibilities:**
1. Load the 5k source image list from BDD clear subset.
2. Shuffle with seed 42.
3. Assign conditions cyclically (rain, fog, snow, night) — 1,250 each.
4. For each source image:
   - Read image.
   - Sample parameters via `rng`.
   - Apply weather transform.
   - Write synthetic image to `data/yolo/bdd_s3/images/train/`.
   - **Byte-copy** the label file to `data/yolo/bdd_s3/labels/train/`.
   - Record metadata (source filename, condition, params, seed) to CSV.
5. Copy the 5k clear images and their labels to `data/yolo/bdd_s3/images/train/` and `labels/train/` (with a clear prefix or separate subdirectory to avoid name collisions).
6. Copy the 2k clear validation images to `data/yolo/bdd_s3/images/val/` and labels to `labels/val/`.
7. Write `synth_manifest.json` with the full mapping.

**Naming convention:**
- Clear images: `clear_<original_filename>.jpg`
- Synthetic images: `synth_<condition>_<original_filename>.jpg`
- Labels: match the image basename with `.txt` extension.

**Metadata CSV columns:**
`source_image, output_image, condition, param_1, param_2, ..., seed, code_commit`

**Reproducibility:** Save `code_commit` (git hash) and the RNG seed in the metadata. Use a single `np.random.default_rng(42)` for the entire build.

### 3.8 Dataset Config — `configs/bdd_s3.yaml`

```yaml
# BDD S3 dataset config for YOLOv8n training
path: data/yolo/bdd_s3
train: images/train
val: images/val

nc: 1  # Adjust to match BDD classes (BDD100K detection has 10 classes; use the same as S1)
names:
  0: pedestrian
  1: rider
  2: car
  3: bus
  4: truck
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign
```

**Note:** Use the same class mapping as S0/S1 to ensure comparability. If S1 used a subset of classes, replicate exactly.

### 3.9 Inspector — `src/synth/inspect_synth.py`

**Checks to perform:**

| Check | Method | Pass criterion |
|---|---|---|
| **Label invariance** | Compare label files byte-for-byte with source | 100% identical |
| **Object visibility** | For each bbox, compute local contrast (std of pixel values) before and after transform | No bbox drops below 30% of original local contrast |
| **Occlusion check** | Compute fraction of bbox pixels covered by rain/snow overlay (if manual overlay) | No bbox >70% occluded |
| **Value-range sanity** | Check min/max/mean pixel values | No image with mean <20 or >235 (near-black or near-white) |
| **Clipping check** | Count saturated pixels (>250) and dead pixels (<5) | <5% of pixels clipped |
| **Class distribution** | Count objects per class before and after | No class loses >10% of instances |
| **Per-condition samples** | Generate 4×4 grid of synthetic images per condition | Visual inspection for paper |
| **Metadata completeness** | Verify every synthetic image has a metadata row | 100% coverage |

**Output:** A report `inspect_report.txt` and visual grids `samples_rain.png`, `samples_fog.png`, `samples_snow.png`, `samples_night.png`.

**Object visibility implementation:**
```python
def check_visibility(image_before, image_after, bboxes):
    """
    For each bbox, compute local contrast (std of grayscale values)
    before and after. Flag if after < 0.3 * before.
    """
    gray_before = cv2.cvtColor(image_before, cv2.COLOR_RGB2GRAY)
    gray_after = cv2.cvtColor(image_after, cv2.COLOR_RGB2GRAY)

    for (x, y, w, h) in bboxes:
        patch_before = gray_before[y:y+h, x:x+w]
        patch_after = gray_after[y:y+h, x:x+w]

        contrast_before = patch_before.std()
        contrast_after = patch_after.std()

        if contrast_before > 0 and contrast_after < 0.3 * contrast_before:
            print(f"WARNING: Bbox ({x},{y},{w},{h}) visibility dropped "
                  f"from {contrast_before:.2f} to {contrast_after:.2f}")
```

---

## 4. Training Protocol

### 4.1 Training Command

Same as S1 except for the dataset config:

```bash
yolo detect train \
  model=yolov8n.pt \
  data=configs/bdd_s3.yaml \
  epochs=80 \
  imgsz=640 \
  batch=32 \
  optimizer=auto \
  cos_lr=True \
  patience=30 \
  seed=42 \
  amp=True \
  project=runs/detect \
  name=bdd_s3
```

**Note:** `optimizer=auto` resolves to AdamW with `lr0=0.001`. `cos_lr=True` enables cosine learning rate scheduling. `patience=30` enables early stopping. All other hyperparameters remain at Ultralytics defaults to maintain the "S1 plus exactly one change" principle.

### 4.2 Model Selection

- **Best checkpoint:** Selected on `bdd_src_val.txt` (clear BDD validation, 2k images).
- **Rationale:** Consistent with S1. Note in limitations that S3 may overfit to synthetic weather while `best.pt` is chosen on clear images.

### 4.3 Reproducibility

- Seed: 42 (same as S1).
- Save the full training command and environment info (CUDA version, PyTorch version, Ultralytics version) in the run directory.
- Save the `synth_manifest.json` and `synth_metadata.csv` alongside the training outputs for auditability.

---

## 5. Evaluation Protocol

### 5.1 Metrics

| Metric | Description |
|---|---|
| mAP@50 | Primary detection metric |
| mAP@50-95 | Secondary detection metric |
| Precision (P) | Per-weather and overall |
| Recall (R) | Per-weather and overall |
| Per-weather mAP@50 | Rain, fog, snow, night separately |
| Per-class mAP@50 | All 10 BDD classes |

### 5.2 Evaluation Command

```bash
yolo detect val \
  model=runs/detect/bdd_s3/weights/best.pt \
  data=configs/acdc_val.yaml \
  imgsz=640 \
  batch=32 \
  split=val
```

**Note:** ACDC official validation set (406 images) is scored **once** per stage. No ACDC image, label, or statistic is used during S3 dataset construction.

### 5.3 Per-Weather Scoring

ACDC provides weather labels for each image. Group the 406 images by weather condition (rain, fog, snow, night) and compute mAP@50 and mAP@50-95 per group. Use the ACDC official evaluation protocol or a custom script that respects the per-weather grouping.

### 5.4 Leakage Control

- ACDC val is scored **once** after training completes.
- No ACDC data is used to choose S3 parameters or inspect synthetic data.
- The ACDC design split (400 images) is inspected only **after** training, to analyse failure modes — not used to choose S3 parameters.

---

## 6. Open Questions — Resolved

| Question | Decision | Rationale |
|---|---|---|
| Q5.1: Condition allocation | Balanced, 1,250 per weather | Symmetric, no test-set tuning appearance |
| Q6.2: Mixed conditions | No | Blurs per-weather attribution; belongs to S6 |
| Q6.3: Night light sources | No for v1 | Illumination modelling belongs to S5 |
| Q6.4: Snow ground accumulation | No for v1 | Spatial, risks obscuring objects; belongs to S5 |
| Q6.5: Depth-dependence | Uniform-depth approximation | No depth maps; depth-aware belongs to S5 |
| Q6.6: Rain streak realism | Sparse, directional, conservative | Avoids destroying small objects |
| Q6.7: Blur in S3 vs S2 | Exclude from S3 | Avoids confounding S2 vs S3 comparison |
| Q6.8: Inspection criteria | See Section 3.9 | Label invariance, object visibility, value-range, class distribution |

---

## 7. Red Flags and Mitigations

| Red Flag | Mitigation |
|---|---|
| **S1 data composition confound:** S1 may have trained on 10k clear, while S3 uses 5k clear + 5k synthetic. | Frame S3 as "replace half the clear data with synthetic weather." The clean comparison for weather-specific vs generic is **S2 vs S3**, since both use 5k clear + 5k transformed. |
| **Label invariance ≠ object visibility:** Photometric transforms do not move boxes, but objects may become invisible. | Inspector checks object visibility (local contrast before/after). Flag if any bbox drops below 30% of original contrast. |
| **Offline synthesis reproducibility:** Without metadata, the dataset cannot be regenerated or audited. | Save per-image metadata (source, condition, params, seed, code commit) to CSV. |
| **S3 accidentally fitting to ACDC:** If S3 parameters are tuned by looking at ACDC results, S3 and S5 collapse. | S3 parameters are hand-set and locked **before** any ACDC evaluation. No ACDC data is used during dataset construction. |
| **Best.pt selection on clear val:** S3 may overfit to synthetic weather while best.pt is chosen on clear images. | Acceptable for the ladder; document in limitations. |

---

## 8. S3 vs S5 Boundary — Critical Design Rule

**S3 = hand-set parameters. S5 = same physical model families, but parameters fitted to measured ACDC-train statistics.**

| Aspect | S3 | S5 |
|---|---|---|
| Rain streak parameters | Chosen by eye and literature defaults | Fitted to ACDC-train rain image statistics (contrast, gradient energy, dark-channel haze) |
| Fog parameters | Hand-set range 0.2–0.6 | Fitted to ACDC-train fog statistics |
| Snow parameters | Hand-set range 0.1–0.3 | Fitted to ACDC-train snow statistics |
| Night parameters | Hand-set brightness/gamma | Fitted to ACDC-train night illumination statistics |
| Depth-awareness | None (uniform-depth approximation) | Depth-aware if feasible |
| ACDC data usage | None | ACDC-train only (not val) |

**If S3 accidentally fits to ACDC, S3 and S5 collapse and the comparison is meaningless.** Lock all S3 parameters before any ACDC evaluation.

---

## 9. Deliverables Checklist

- [ ] `src/synth/weather.py` — Main transform interface with registry
- [ ] `src/synth/rain.py` — Rain transform
- [ ] `src/synth/fog.py` — Fog transform
- [ ] `src/synth/snow.py` — Snow transform
- [ ] `src/synth/night.py` — Night transform
- [ ] `src/synth/dataset_builder.py` — Builds `data/yolo/bdd_s3/`
- [ ] `src/synth/inspect_synth.py` — Dataset inspector
- [ ] `configs/bdd_s3.yaml` — YOLOv8 dataset config
- [ ] `data/yolo/bdd_s3/` — Generated dataset (10k images + labels)
- [ ] `data/yolo/bdd_s3/metadata/synth_metadata.csv` — Per-image metadata
- [ ] `data/yolo/bdd_s3/synth_manifest.json` — Source-condition mapping
- [ ] `runs/detect/bdd_s3/weights/best.pt` — Trained model
- [ ] `runs/detect/bdd_s3/` — Training logs and results
- [ ] Per-weather evaluation results (rain, fog, snow, night)
- [ ] `inspect_report.txt` — Inspector output
- [ ] Visual sample grids for paper (`samples_*.png`)

---

## 10. Implementation Sequence (Recommended Order)

1. **Implement `weather.py` + individual transforms** (rain, fog, snow, night) as pure functions.
2. **Unit-test each transform** on a few sample images; verify no geometry changes and labels remain valid.
3. **Implement `dataset_builder.py`**; generate a small subset (e.g., 50 images) for testing.
4. **Implement `inspect_synth.py`**; run on the small subset; fix any issues.
5. **Generate full dataset** (10k images) using seed 42.
6. **Run full inspection**; verify all checks pass.
7. **Train YOLOv8n** with the S3 config (80 epochs).
8. **Evaluate on ACDC** per weather; record metrics.
9. **Generate paper artifacts** (sample grids, metadata summary).
10. **Lock S3 results**; proceed to S4/S5 planning.

---

## Appendix A: Parameter Summary Table

| Condition | Parameter | Range | Notes |
|---|---|---|---|
| **Rain** | Slant | 70–85° | From horizontal |
| | Streak length | 10–30 px | |
| | Streak width | 1–2 px | |
| | Alpha | 0.3–0.5 | |
| | Contrast reduction | ×0.8–0.95 | |
| | Streak count | 50–200 | Per 640×640 |
| **Fog** | Fog coefficient | 0.2–0.6 | Alpha blend |
| | Contrast reduction | ×0.6–0.9 | |
| | Brightness lift | ×1.0–1.2 | |
| | Desaturation | ×0.7–1.0 | |
| **Snow** | Snow point | 0.1–0.3 | Intensity threshold |
| | Brightness lift | ×1.0–1.15 | |
| | Contrast reduction | ×0.85–1.0 | |
| | Desaturation | ×0.8–1.0 | |
| | Particle density | 0.005–0.02 | Fraction of pixels |
| **Night** | Brightness | ×0.4–0.7 | |
| | Gamma | 0.7–0.95 | |
| | Tint | Warm or cool | Subtle ±5 |
| | Vignette | 0.1–0.3 | Optional |

---

## Appendix B: Example Training Output Log

```
Ultralytics YOLOv8.0.xxx 🚀 Python-3.10.x torch-2.x.x CUDA:0 (NVIDIA RTX 3050 Laptop, 6GB)
...

Overriding model.yaml nc=10 with nc=10

Transferred 319/355 items from pretrained weights
optimizer: AdamW(lr=0.001, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 64 bias(decay=0.0)
...

Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/80      3.2G      1.234      0.567      1.123        456        640
  ...
 80/80      3.4G      0.567      0.234      0.789        512        640

Results saved to runs/detect/bdd_s3
```

---

## Appendix C: Contacts and References

- **S1 baseline:** Ultralytics defaults, 0.269 mAP@50 on ACDC.
- **S0 floor:** Clear BDD no aug, 0.201 mAP@50.
- **T1 reference:** ACDC labels no aug, 0.216 mAP@50.
- **T1aug ceiling:** ACDC labels + defaults, 0.320 mAP@50.
- **Albumentations documentation:** https://albumentations.ai/docs/
- **Automold library:** https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
- **Ultralytics YOLOv8 docs:** https://docs.ultralytics.com/

---

**Document status:** Locked for implementation. No changes to parameter ranges or design decisions without explicit approval and documentation of rationale.