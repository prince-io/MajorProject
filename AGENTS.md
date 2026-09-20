# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Project

- Final-year project: train YOLO models (Ultralytics) for **object detection** on adverse-weather driving datasets (ACDC, BDD100K). Detection only — ACDC has no segmentation labels.
- Current direction: the agreed **S0–S6 comparative study** — transform clear BDD100K into synthetic adverse weather, train one detector per strategy, score per weather on real ACDC; **S1 (Ultralytics defaults) is the anchor**. See `PROJECT.md` §5.
- Python 3.12 virtualenv at `.venv`; CUDA-enabled PyTorch (`cu124`) on an RTX 3050 (6 GB).
- Raw datasets live under `datasets/` and are treated as read-only source material.
- Generated artifacts live under `data/` (YOLO datasets), `splits/` (manifests), and `results/` (experiments, summary, logs); nothing is ever written inside `datasets/`.
- `PROJECT.md` is the single source of truth for plans, verified facts, decisions, and results; update it on every meaningful change and use it when drafting the paper.
- `HANDOFF.md` is the session resumption point (operational summary, current state, next steps); keep it current at each session close.

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

- This is a serious thesis project targeting a **defensible, potentially publishable contribution** (current direction: the **S0–S6 comparative study** in `PROJECT.md` §5, with the novelty decision deferred until per-condition results exist). Every idea, recommendation, approach, solution, alternative, and default must be **sound and research-oriented** — not naive, not merely tutorial-level, and not the obvious off-the-shelf choice without justification.
- Prefer defensible, well-motivated decisions: state assumptions, tradeoffs, and failure modes; verify prior art before claiming novelty; distinguish what is standard from what is a genuine contribution.
- Keep `PROJECT.md` and every project-detail document current as the project advances: update facts, results, and decisions **immediately**; update hypotheses, predictions, and method targeting at **defined evidence gates** and mark changes as "revised from pre-registered". Never leave a stale number, path, or claim in the docs.
- **Cite research papers by name, not shorthand.** Whenever a doc, code comment, or note refers to prior work, write the full method/paper name and attribution (e.g., "Fourier Domain Adaptation (FDA) [Yang & Soatto, CVPR 2020]", not just "FDA"), so the reference is reusable directly in the manuscript. When new prior work is verified, keep canonical entries in the `paper/` reference notes.
- **Each experiment must push for genuine performance, not merely report a default.** Design every stage and, where a modeling/parameter choice is free, choose the option expected to raise the per-condition metric; do not settle for the naive default. Gains must stay **legitimate**: never tune on the official ACDC val or sample it into any pool; the 400-image design split is only for S6 policy inspection; the unlabeled pool may be used only as each stage's pre-registered contract allows. Pre-registered parameters are locked before scoring.
- When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Child DOX Index

- `datasets/AGENTS.md` - raw ACDC and BDD100K source datasets, layout, and read-only rules.
- `scripts/AGENTS.md` - dataset inspection tools and the unified class mapping module.
- `src/AGENTS.md` - data conversion, split building, materialization, training, evaluation, aggregation, and figures.
- `configs/AGENTS.md` - generated Ultralytics dataset YAMLs.
- `paper/AGENTS.md` - manuscript, related-work, and methods notes.