# paper/literature

## Purpose

- Verified literature knowledge base for the **S0–S6 adverse-weather detection study** (`PROJECT.md` §5): canonical references (`.md` + `.bib`) and per-topic synthesis that feed the Related Work and Method sections of the manuscript.

## Ownership

- Owned by `paper/AGENTS.md`.

## Local Contracts

- **Verified sources only.** Every entry carries a verification tag (`arXiv-ID`, `manual`, `unverified`). Never invent a citation, arXiv ID, author list, venue, or number.
- Always use the **full method/paper name with attribution** (e.g. "Fourier Domain Adaptation (FDA) [Yang & Soatto, CVPR 2020]"), never shorthand alone.
- `references.md` and `references.bib` are the canonical pair and must stay in sync; a new paper updates both.
- `PROJECT.md` remains the source of truth for project facts, results, and decisions; this hub holds prior work and analysis only — no project results or numbers.
- Venue fields marked `confirm` must be checked against the official proceedings before submission.
- Entries tagged `unverified` must not appear in the manuscript until confirmed.

## Work Guidance

- Verify metadata against the arXiv API (curl works; the Python `urllib` request is rejected with HTTP 406 — use `curl`/`subprocess`).
- Keep one topic note per theme (`01`–`08`) plus `09_gaps_and_positioning.md`; add a note only when a theme becomes a durable reading area.
- Record relevance to specific stages (S0–S6, T1/T1aug) so citations slot directly into the paper.
- Re-check saturation/novelty in `09` whenever a new related paper is found.

## Verification

- `references.bib` parses (balanced braces, one `\@` block per reference) and every `eprint` matches the `arXiv` link in `references.md`.
- Every citation used in paper text resolves to an entry here with a non-`unverified` tag.

## Child DOX Index

- None.
