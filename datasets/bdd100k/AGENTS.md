# bdd100k

## Purpose

- BDD100K: large-scale driving dataset providing detection images/labels and semantic segmentation masks.

## Ownership

- Owned by `datasets/AGENTS.md`.

## Local Contracts

- Three components, each under its own subfolder:
  - `bdd100k/bdd100k/bdd100k/images/{100k,10k}/{train,val,test}/` - JPG detection images (100k set: 100000, 10k set: 10000).
  - `bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_{train,val}.json` - detection labels (train has 69863 entries).
  - `bdd100k_seg/bdd100k/seg/` - segmentation: `images/{train,val,test}` (JPG), `labels/{train,val,test}` (train-id PNG masks), `color_labels/{train,val,test}` (color PNG masks). Counts: train 7000, val 1000, test 2000.
- Detection labels are JSON entries with `name`, `attributes`, `timestamp`, `labels`; each label has `category`, `box2d` (x1,y1,x2,y2), and attributes.
- Object categories: `person`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`, `traffic light`, `traffic sign`. `drivable area` and `lane` also appear and are not objects.
- Some image split folders contain extra `trainA/trainB/testA/testB` subfolders from an image-translation split; detection label records resolve recursively under `100k/train/**` and `100k/val/**`, not to flat files.
- The `10k/` image set is a separate benchmark subset; the `100k/` set is the detection source.

## Work Guidance

- Use `bdd100k_labels_images_train.json` and `_val.json` for detection; the test split has no detection labels.
- Segmentation `labels/` are train-id masks; `color_labels/` are the human-readable color versions. Do not mix the two.

## Verification

- `python src/data/convert_bdd.py` converts 12,454 clear/daytime records with 0 missing images.

## Child DOX Index

- None.
