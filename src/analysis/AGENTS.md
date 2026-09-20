# src/analysis

## Purpose

- Post-hoc diagnostics that **inspect failure modes and screen a proposed stage without creating a stage row**. Current member: the S5b Tier 1 design-split blur probe (`blur_probe.py`).

## Ownership

- Owned by `src/AGENTS.md`.

## Local Contracts

- **No stage rows.** These scripts never write under `results/experiments/<ID>/` and never produce an experiment ID; they are not part of `aggregate.py`/`visualize.py`.
- **Leakage control.** Allowed ACDC inputs are the **design split only** (`splits/acdc_design.txt`, 400 = 100/condition), per `PROJECT.md` §6. Never read `splits/acdc_official_val.txt` and never score the official split here; the official val is the only scorer and is scored once per stage through `src/eval.py`.
- Probe outputs are tracked under `results/summary/blur_probe/`; transient blurred images/labels go to a scratch directory outside the repo (`/tmp/opencode/...`), never under `data/` or `datasets/`.
- `datasets/` stays read-only; read manifests from `splits/`, images from `data/yolo/`.
- Report a probe as a **screening heuristic**, not a result: test-time blur on already-degraded target images can only support, not prove, a training-time benefit.

## Work Guidance

- Run from the project root: `python src/analysis/blur_probe.py` (defaults: S5 `best.pt`, design split, blur strengths from `results/calibration/blur_stats.json`, output `results/summary/blur_probe/`).
- Use `--limit` for a quick probe and `--conditions` to restrict conditions.
- Keep probe parameters (multipliers, metrics) fixed before running and record them in the output JSON.

## Verification

- `python src/analysis/blur_probe.py --limit 5` writes `results/summary/blur_probe/blur_probe.json` and a figure, and creates no `results/experiments/*` directory.

## Child DOX Index

- None.
