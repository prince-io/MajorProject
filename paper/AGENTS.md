# paper

## Purpose

- Manuscript drafts, related-work synthesis, methods notes, figures, and tables for the final paper.

## Ownership

- Owned by the project root `AGENTS.md`.

## Local Contracts

- `PROJECT.md` is the source of truth for facts, decisions, and results; paper text must stay consistent with it.
- Cite only verified sources; the canonical list lives in `literature/references.md` (+ `references.bib`). The 3rd-year mini-project bibliography is unverified and must be rebuilt before use.
- Report results only from the results tracker in `PROJECT.md`; never invent numbers.

## Work Guidance

- Maintain a research-grade standard: every claim needs evidence or a verified citation, novelty must be checked against prior art, and no naive or tutorial-level framing.
- Build related-work text from `literature/` (full method/paper names + venue); the previous `related_work.md`, `methods_notes.md`, and `ideas_wsm.md` were removed in the 2026-09-17 baseline-only reset.
- Use the terminology defined in `PROJECT.md` §6 (zero-shot vs unlabeled adaptation vs supervised fine-tuning vs leave-one-domain-out).
- Keep `results_notes.md` current as each stage completes: paper-ready tables, per-weather/per-class breakdowns, interpretation, and caveats. Numbers must match `PROJECT.md` §9.

## Verification

- Every citation in manuscript text resolves to a non-`unverified` entry in `literature/references.md` (+ `references.bib`); `references.bib` parses and its `eprint` IDs match `references.md`.
- Every number in `results_notes.md` matches the results tracker in `PROJECT.md` §9.

## Child DOX Index

- `literature/AGENTS.md` - verified literature knowledge hub (canonical references + per-topic synthesis) for the S0–S6 study.
