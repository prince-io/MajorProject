# 02 — Detection Architectures and Backbones

## Why this matters
The study is locked to **YOLOv8n** for compute reasons (6 GB VRAM). We must justify that
choice and be able to state what the results do and do not imply for other detectors.

## One-stage / YOLO family

### YOLOv1 [Redmon et al., CVPR 2016] — `redmon2016yolo`
- Origin of the single-shot detector: one network predicts boxes+classes in one pass.
- Cite for lineage only.

### YOLOv4 [Bochkovskiy et al., 2020] — `bochkovskiy2020yolov4`
- **Critical for us:** introduces **Mosaic** augmentation and the bag-of-freebies that
  became the Ultralytics defaults. Our **S1** is "Ultralytics defaults", so YOLOv4 is the
  origin of much of S1's recipe. It is also why S2 must be defined as *beyond* default HSV
  and RandAugment (already photometric).

### YOLOv7 [Wang et al., CVPR 2023] — `wang2023yolov7`
- Anchor-free, trainable bag-of-freebies; an optional robustness/generalization check.

### YOLOv9 [Wang et al., ICLR 2024] — `wang2024yolov9`
- Programmable gradient information; optional generalization check.

### Ultralytics YOLOv8/YOLO11 — `jocher2023ultralytics`
- The actual implementation we train; cite the software and pin the version (`8.4.147`).

## Two-stage and transformer detectors

### Faster R-CNN [Ren et al., NeurIPS 2015] — `ren2015fasterrcnn`
- The backbone of the **domain-adaptation detection** literature (C1–C4). Our study is
  YOLO-based, so we must note that most DAOD prior work uses two-stage detectors and that
  results may not transfer directly.

### FPN [Lin et al., CVPR 2017] — `lin2017fpn`
- Multi-scale features used inside modern detectors; context.

### RetinaNet / Focal Loss [Lin et al., ICCV 2017] — `lin2017focalloss`
- Dense one-stage with class imbalance handling; context.

### DETR [Carion et al., ECCV 2020] — `carion2020detr`
- Transformer detector; context, optional related-work breadth.

## Evaluation anchor

### COCO [Lin et al., ECCV 2014] — `lin2014coco`
- Defines the mAP protocol we report (mAP@50, mAP@50-95). Cite for metrics.

## Takeaways for our design
- **Justify YOLOv8n** by compute/coverage (6 GB), not by state-of-the-art claims.
- State the limitation explicitly: findings are for a **small one-stage detector**; two-stage
  DAOD results (C1–C4) may differ.
- When writing related work, separate (a) detectors, (b) DA methods, (c) weather synthesis —
  do not conflate them.

## What to cite where
- Method/experimental setup: B5, B10.
- Related work on detectors: B1, B3, B4, B6, B7, B8, B9.
- Augmentation origin (Mosaic): B2, G2.
