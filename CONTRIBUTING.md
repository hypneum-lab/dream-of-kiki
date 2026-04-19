# Contributing to dreamOfkiki

This repo maintains parallel English + French artifacts for the
research narrative (papers, specs). Code + tests + docs/ are
English-primary.

## Language policy

| Artifact | Primary | Secondary |
|----------|---------|-----------|
| Code (`kiki_oniric/`, `harness/`, `tests/`, `scripts/`) | English | — |
| Commits, issues, PRs | English | — |
| Specs (`docs/specs/`) | English | French (`docs/specs-fr/`) |
| Papers (`docs/papers/paper{1,2}/`) | English | French (`docs/papers/paper{1,2}-fr/`) |
| Plans (`docs/superpowers/plans/`) | French (historical) | — |
| Milestones (`docs/milestones/`) | English | — |
| Proofs (`docs/proofs/`) | English | — |
| Glossary (`docs/glossary.md`) | English | — |
| Invariants registry (`docs/invariants/`) | English | — |
| Changelog, README, STATUS | English | — |

## EN→FR propagation rule

Any modification to an English **paper** or **spec** must be
followed by a corresponding update to its French counterpart
**in the same PR** (or a dedicated FR-sync PR immediately
following). The FR version is not a freeze-at-v1.0 snapshot
(i.e. snapshot gelé à une version donnée — ex. v1.0) ; it
tracks EN changes.

Exception : typo-only fixes to EN papers/specs do not require
FR propagation unless the typo also exists in the FR version.

## Translation vocabulary

Canonical French technical terms established during Paper 1 FR
pilot (2026-04-18) :

| English | French |
|---------|--------|
| catastrophic forgetting | oubli catastrophique |
| consolidation | consolidation mnésique |
| synaptic homeostasis | homéostasie synaptique |
| free energy principle | principe d'énergie libre |
| predictive coding | codage prédictif |
| replay (pillar A) | réactivation |
| replay (technical) | replay (kept) |
| downscaling | régulation à la baisse |
| semigroup | semi-groupe |
| compositionality | compositionnalité |
| pre-registration | pré-enregistrement |
| benchmark | banc de test |
| accountability | redevabilité |
| substrate-agnosticism | indépendance du substrat |
| Conformance Criterion | Critère de Conformité |

Preserved unchanged in FR : axiom IDs (DR-0..DR-4), invariants
(I1/I2/I3/S1/S2/S3/S4/K1/K3/K4), profile names (P_min, P_equ,
P_max), substrate names (kiki-oniric, E-SNN), LaTeX math, code
blocks, file paths, BibTeX keys, URLs.

## Target publication venues

| Venue | Language | Scope |
|-------|----------|-------|
| arXiv | English | Paper 1 + Paper 2 preprints |
| Nature Human Behaviour | English | Paper 1 target |
| NeurIPS/ICML/TMLR | English | Paper 2 target |
| HAL (Hyper Articles en Ligne) | French | FR versions deposited for French academic archive |
| L'Electron Rare blog | French | Blog.md vulgarisation articles |

## Commit convention

All commits follow conventional-commits + validator-enforced
rules :
- Subject ≤50 chars
- Scope ≥3 chars (e.g., `paper1`, `fr`, `paper1-fr`, `dream`)
- Body lines ≤72 chars
- Body required (2-3 paragraphs for significant commits — e.g.
  commits that change functionality, add features, modify a public
  API, or touch multiple modules ; in French : commits qui changent
  la fonctionnalité, ajoutent des fonctionnalités, modifient une
  API publique, ou touchent plusieurs modules)
- NO AI attribution
- NO `--no-verify`

Scope conventions for FR artifacts :
- `fr` (generic FR update)
- `paper1-fr`, `paper2-fr` (specific FR paper)
- `specs-fr` (spec FR update)
