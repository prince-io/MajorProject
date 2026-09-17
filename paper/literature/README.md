# Literature Knowledge Hub

> Purpose: a single, reusable knowledge base of the prior work relevant to the
> **S0–S6 adverse-weather detection study** (see `PROJECT.md` §5). It is written to feed
> directly into the manuscript — every entry uses the **full paper/method name with
> attribution** so it can be cited verbatim.
>
> This hub is **not** a manuscript draft. It is the verified reference layer plus
> per-topic synthesis that the Related Work and Method sections will be built from.

## How to use it

- **`references.md`** — the canonical list. Start here for the exact title/authors/venue
  and the one-line relevance of each work. Verification status is marked per entry.
- **`references.bib`** — BibTeX for paper drafting. Copy entries; do not re-key them.
- **`01`–`08` topic notes** — deeper, per-topic synthesis: what the method does, how it
  relates to our stages, its strengths/limitations, and how we should position against it.
- **`09_gaps_and_positioning.md`** — the novelty map: what is already saturated, what
  remains open, and where the study's contribution can live.

## Verification legend (read before citing)

- **`arXiv-ID`** — title/authors/date fetched from the arXiv API; the ID in this hub is
  the one returned. Venue (conference/journal) is stated from knowledge and **should be
  confirmed against the official proceedings before submission**.
- **`manual`** — canonical non-arXiv work (classic journal/conference or software); the
  bibliographic details are standard but confirm the page/DOI on first citation.
- **`unverified`** — referenced in earlier project notes but **not** confirmed here.
  Do not cite until checked. Currently: *MIC* and *ViSGA* (see `09`).

## Scope map (what belongs here)

| Topic file | Covers | Study stages served |
|---|---|---|
| `01_datasets.md` | ACDC, BDD100K, Cityscapes/Foggy, DAWN, CADC(+), NightOwls | all (source/target data) |
| `02_detection_backbones.md` | YOLO family, two-stage, DETR, FPN | model choice + related work |
| `03_domain_adaptation_detection.md` | adversarial DA, mean-teacher, prior-based, weather DAOD | S4, S6; positioning |
| `04_frequency_methods.md` | FDA, FACT, phase/amplitude, style statistics | **S4**, S6 |
| `05_weather_synthesis.md` | physics rendering, image-to-image translation, weather GANs | **S3, S5, S6** |
| `06_restoration_and_augmentation.md` | dehaze/derain/low-light, augmentation families | S2, S3, S5 |
| `07_dg_robustness_tta.md` | domain randomization, DG, test-time adaptation | S6, future work |
| `08_benchmarks_and_positioning.md` | adverse-weather detection/benchmark studies, metrics/protocol | all (evaluation) |
| `09_gaps_and_positioning.md` | novelty gaps, saturation, candidate contributions | paper framing |

## Rules

- Update this hub whenever a new paper is read/verified; keep `references.md` and
  `references.bib` in sync.
- Never invent a citation, arXiv ID, or number. Mark uncertain metadata explicitly.
- Prefer primary sources; state the venue and the exact method name.
