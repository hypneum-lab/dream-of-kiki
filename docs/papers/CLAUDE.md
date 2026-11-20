# docs/papers/ — Paper 1 / Paper 2 / Paper 3 source trees

**Distinct from `papers/` at repo root** : that dir holds the
submission-package CLAUDE.md (venue tracker, byline policy). This
dir holds the actual section sources (`abstract.md`,
`introduction.md`, `full-draft.md`, …) plus FR mirrors.

## Layout

| Subdir | Scope | Status |
|--------|-------|--------|
| `paper1/` | Paper 1 EN — Framework C (formal, Paper 1 cycle 1) | submission-ready (PLOS Comp Bio target since 2026-04-20; Nature HB retired) |
| `paper1-fr/` | Paper 1 FR mirror | parallel maintenance, HAL + L'Electron Rare blog |
| `paper2/` | Paper 2 EN — Engineering ablation (cycle 2) | NeurIPS / ICML / TMLR target |
| `paper2-fr/` | Paper 2 FR mirror | parallel maintenance |
| `paper3/` | Paper 3 — cycle-3 amorçage outline only | **PROVISIONAL** banner, do not flesh out before cycle-3 kickoff |

Each subdir carries its own `__init__.md` (file index +
cross-references + authorship byline). Read it before editing
section files.

## Conventions

- **Naming** : `paperN/` = canonical EN ; `paperN-fr/` = strict FR
  mirror. Section filenames match 1:1 (e.g.
  `paper1/methodology.md` ↔ `paper1-fr/methodology.md`).
- **EN leads** : every change starts in `paperN/`, FR mirror in the
  **same PR**. The `CONTRIBUTING.md` EN→FR propagation rule is
  reviewer-enforced and pre-commit-checked.
- **Citations are pinned, always** :
  - Cross-repo (e.g. paper1 §7.4 → nerve-wml v0.9 draft) :
    pin to a **tag**, never `HEAD`, never branch.
  - Internal proofs : pin to versioned filename
    (`docs/proofs/dr2-compositionality.md v0.2`).
  - Internal milestones : pin to dated filename
    (`docs/milestones/g2-pmin-report.md`).
  - Framework C : full DualVer tag (`C-v0.5.0+STABLE`), as shipped
    in `CHANGELOG.md`. No bare `C-v0.5`.
- **Numbers ↔ run_id** : every empirical number resolves to a
  registered `run_id` (`harness/storage/run_registry.py`) + a dump
  in `docs/milestones/`. Synthetic-pipeline numbers carry the
  `(synthetic placeholder, G2 pilot)` flag in caption.
- **Build artefacts** : `paper1/build/`, `paper2/build/` are
  pandoc outputs — gitignored or committed only as snapshots
  matching a tag.

## Coupling

- Paper 1 ↔ `docs/specs/` + `docs/proofs/` (formal substrate;
  axioms / invariants must use canonical IDs from `docs/glossary.md`).
- Paper 2 ↔ `kiki_oniric/` + `harness/` + `scripts/pilot_*.py`
  (engineering substrate; cite run_id and dumps).
- Submission gate : `papers/CLAUDE.md` (root) + `ops/CLAUDE.md`
  carry the venue tracker + reviewer recruitment ; this dir
  contains only sources.

## Anti-patterns

- Editing `paperN/` and `paperN-fr/` in separate PRs — drift is
  inevitable. Same PR or no PR.
- Citing a paper draft or proof by `HEAD` from a sibling paper or
  external doc — always a tag (`paper-1-v0.3`,
  `dr2-proof-v0.2`).
- Pasting numbers from a notebook into `results-section.md`
  without a registered `run_id` — the harness rule applies even
  inside the paper tree (parent `papers/CLAUDE.md` rule, restated
  here because section authors edit here, not in `harness/`).
- Removing or regenerating a figure without updating the
  milestone dump and committing a dated `run_id` reference —
  breaks R1 reproducibility on the figure. Note: the figure
  regeneration script (`scripts/render_figures.py`) is not yet
  in the repository (same note as `CLAUDE.md` l. 85–86). Once
  written, figures will live under
  `docs/papers/<paper>/figures/`.
- Fleshing out `paper3/outline.md` before cycle-3 kickoff — the
  PROVISIONAL banner is load-bearing : Phase 3+4 of cycle 2 must
  land first.
- Inlining a Paper 1 (formal, substrate-agnostic) section into a
  Paper 2 (engineering) draft (or vice versa) — scopes differ ;
  Paper 2 cites Paper 1 by version tag, never copies content.
