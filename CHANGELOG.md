# Changelog

All notable changes to dream-of-kiki are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
+ [Conventional Commits](https://www.conventionalcommits.org/).

Versioning scheme : **DualVer** (framework C formal+empirical axes,
see `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §12).

---

## [C-v0.5.0+STABLE] — 2026-04-17

End of setup phase (S1-S4). Program enters implementation phase (S5+).

### Added

- Formal framework C specs (master + framework C, 977 lines)
- Implementation plan phase-1 detailed (1416 lines, 80 checkboxes)
- Implementation plan S3-S4 atomic (1464 lines, 6 tasks)
- DR-3 Conformance Criterion strengthened (post-critic finding #3)
- Python project skeleton (uv, Python 3.12+, mlx, numpy, hypothesis,
  pytest, duckdb, pyarrow, plotly, yaml, click)
- RunRegistry with deterministic run_id (SHA-256-based, R1 contract)
- Invariants & Axioms registry (I/S/K families + DR-0..DR-4)
- Canonical glossary (primitives, profiles, DualVer, gates, metrics)
- T-Col outreach plan (3 fMRI labs + formal reviewer candidates)
- GitHub Actions CI workflow (lint + types + pytest + invariants)
- Fork decision document (kiki-oniric r3 jalonné S1/S8/S18)
- kiki_oniric skeleton
- Studyforrest RSA feasibility note (G1 Branch A locked)
- Story 0 expose typed Protocols (8 primitives, Conformance condition 1)
- Interface contracts : primitives.md + eval-matrix.yaml
- EvalMatrix loader with 6 contract tests
- OSF pre-registration draft (H1-H4 operationalized)
- Formal reviewer recruitment tracker + email template
- Retained benchmark (SHA-256 integrity, 50 synthetic items)
- fMRI schema lock (Studyforrest Branch A)
- Framework version bumped C-v0.3.1 → C-v0.5.0+STABLE

### Changed

- sqlite3 context manager fixed (leak + contextlib.closing)
- .gitignore now excludes .coverage artifacts

### Stats

- 21 commits across brainstorm → spec → plan → execution flow
- 16 tests passing, coverage 93.62% (gate ≥90%)
- 5 source files with C-v0.5.0+STABLE version consistency
- 0 BLOCKING invariant violations
- 0 AI attribution in any commit

### Milestones achieved

- **G1** — T-Col fallback locked Branch A (Studyforrest feasibility)
- **DR-3 Conformance Criterion condition (1)** — typed Protocols
  exposed, 3 tests passing

### Pending (S5+)

- DR-2 compositionality proof draft (G3-draft S6)
- P_min runtime functional (G2 S8)
- OSF pre-registration lock (upload via checklist)
- fMRI lab outreach replies (S3-S5)
- Formal reviewer recruitment (Q_CR.1 b, S3-S5)

---

## [C-v0.5.0+STABLE cycle-1 closeout] — 2026-04-18

End of cycle 1 (S5-S28). Programme essentially complete;
arXiv submission + Nature HB submission + DR-2 reviewer feedback
+ OSF lock = external user actions pending.

### Added — implementation phase (S5-S12)

- DreamEpisode 5-tuple dataclass + BudgetCap (S5.1)
- DreamRuntime scheduler with DR-0 log guarantee + try/finally
  exception handling (S5.2)
- DR-0 + DR-1 property tests via Hypothesis (S5.3)
- Replay operation skeleton (A-Walker source, S5.4)
- Downscale operation skeleton (B-Tononi SHY, S7.1)
- S2 finite guard (NaN/Inf check, S7.2)
- Swap protocol skeleton with S1+S2 guards (S7.3)
- P_min profile fully wired (S7.4 + S9.4 swap_now)
- P_equ profile skeleton then fully wired (S8.3 + S11.2)
- Restructure operation (D-Friston FEP, S10.1 skeleton +
  S13.1 MLX-native)
- S3 topology guard (validate_topology, S10.2)
- Recombine operation (C-Hobson, S11.1 skeleton + S13.2 MLX VAE)
- DR-4 profile inclusion proof + axiom test (S12.1)
- G2 P_min viability report (S8.1 + S9.5 pilot)
- G3 decision log skeleton (S8.2)
- G4 P_equ functional report (S12.2 + S15.3 ablation)

### Added — MLX integration phase (S9-S15)

- Replay handler MLX backend (MSE + SGD on real model, S9.1)
- Downscale handler MLX backend (real weight shrinkage, S9.2)
- Retained eval bridge (evaluate_retained → S1 guard, S9.3)
- mega-v2 dataset loader (real path + synthetic fallback, S13.3)
- Concurrent dream worker skeleton (Future API, S14.1)
- Statistical eval module H1-H4 (Welch / TOST / Jonckheere /
  one-sample t with Bonferroni, S15.1)
- Ablation runner harness (cartesian profile × seed grid, S15.2)
- G4 ablation real run on synthetic mega-v2 (3/4 hypotheses
  significant at α = 0.0125, S15.3)

### Added — closing phase (S16-S28)

- P_max profile skeleton with target_ops + target_channels_out
  metadata for DR-4 chain (S16.1)
- DR-4 P_max chain extension test (S16.2)
- Paper 1 outline + abstract + introduction (S17)
- Paper 1 results-section + discussion + future-work +
  references.bib (S18-S19)
- Paper 1 methodology + background + full-draft assembly (S20)
- arXiv preprint submission tracker (S21.1)
- G5 publication-ready gate report + Nature HB submission
  tracker (S22.1 + S18.1)
- Pivot B contingency decision tree (S18.3)
- Reviewer feedback collection skeleton (S25.1)
- Paper 2 cycle-2 outline (S27.1)
- G6 cycle-2 bootstrap decision report (S28.1)

### Changed — code-review iterations

- 5 CodeRabbit review cycles: 6 + 10 + 7 + 2 + 78 cumulative
  findings, ~95 fixes applied
- run_id width bumped 16 → 32 hex chars (collision safety)
- DR-3 Conformance Criterion strengthened with executable
  conditions
- Op-pair commutativity matrix corrected (downscale not
  idempotent)
- DR-2 free-semigroup softened to "semigroup generated by"
  pending universal property proof
- evaluate_retained accepts seed parameter (CRIT bug fix)
- AblationRunner registers run_id (R1 contract enforcement)

### Stats — cycle 1 close

- ~96 commits across brainstorm → spec → plan → 6 atomic plans →
  execution flow
- 116 tests passing, coverage 90.37% (gate ≥90%)
- DR-3 Conformance Criterion conditions (1) + (2) + (3) all
  green ; DR-2 reviewer pending (G3)
- 0 AI attribution in any commit
- 6 atomic plans covering S1-S28 calendar

### Milestones achieved

- **G1** Branch A Studyforrest LOCKED
- **G2** P_min GO-CONDITIONAL (synthetic pipeline PASS)
- **G4** P_equ GO-CONDITIONAL/PASS (3/4 hypotheses significant
  on synthetic ablation)

### Pending — external user actions

- **G3** DR-2 external reviewer feedback (T-Col Q_CR.1 b)
- **G5 → GO-FULL** Paper 1 arXiv preprint render + submission
- Nature HB submission (or Pivot B branch selection at S22)
- OSF DOI lock (`docs/osf-upload-checklist.md`)
- fMRI lab partnership formalization (T-Col extension)
- Real mega-v2 ablation run (closeout for GO-CONDITIONAL flip
  to GO-FULL)
- Cycle 2 bootstrap decision (G6, post-cycle-1 retrospective)
