# scripts

## Purpose

- Standalone utilities for inspecting the raw datasets and mapping native classes to the project's unified 6-class scheme.

## Ownership

- Owned by the project root `AGENTS.md`.

## Local Contracts

- `class_map.py` is the single source of truth for class mappings; conversion and training code must import it rather than redefine mappings.
- Scripts are stdlib-only and read-only against `datasets/`; they never write dataset files.
- The only permitted write is the generated report at `_reports/inspect_report.txt`.
- Run scripts from the project root so the script directory is importable.

## Work Guidance

- Keep `UNIFIED_CLASSES` IDs stable once training begins; changing them invalidates generated labels.
- See `README.md` for verified dataset facts and deviations before extending the pipeline.

## Verification

- `python scripts/class_map.py` prints both mappings without error.
- `python scripts/inspect_dataset.py` runs end to end and writes `_reports/inspect_report.txt`.

## Child DOX Index

- None.
