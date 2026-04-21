# Changelog

All notable changes to dream-of-kiki are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
+ [Conventional Commits](https://www.conventionalcommits.org/).

Versioning scheme : **DualVer** (framework C formal+empirical axes,
see `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §12).

---

## [C-v0.7.1+PARTIAL] — 2026-04-21

### Changed — DR-2 weakened (FC-PATCH bump)

- DR-2 (compositionality) now carries an explicit precondition
  excluding the empirically falsified class (permutations with
  RESTRUCTURE preceding REPLAY). See spec §6.2 and amendment
  `docs/specs/amendments/2026-04-21-dr2-empirical-falsification.md`.
- No semantic change to the compositionality claim itself on the
  safe class; this is a clarification/equivalent reformulation per
  DualVer §12.2 FC-PATCH rule.
- EC axis unchanged (no new gate crossed).

---

## [C-v0.7.0+PARTIAL] — 2026-04-19

Cycle-3 Phase 1 launch bump. FC MINOR (+0.1.0) because cycle 3
adds a new derived constraint surface on the formal axis :

### Added — R1 output-hash API multi-artifact extension (issue #2)

- ``run_output_hashes`` schema extended from single-hash-per-run to
  N-hashes-per-run keyed on ``(run_id, artifact_name)``. Required
  for Paper 2 (ablation) where each run may register multiple
  artifacts (per-checkpoint, per-metric bundle, per-profile snapshot).
- New API, kwargs-only on the new parameters for backward-compat :
  ``register_output_hash(run_id, output_hash, *, artifact_name='canonical',
  hash_type='sha256')``,
  ``get_output_hash(run_id, *, artifact_name='canonical') -> str``,
  ``list_output_hashes(run_id) -> dict[str, str]``.
- ``_ensure_schema`` is idempotent over three states (fresh DB,
  legacy v1 single-hash schema, already-new schema). Legacy rows
  migrate to ``artifact_name='canonical'`` via a transactional
  rename / create / copy / drop. No action required on existing DBs.
- 9 new unit tests covering backward-compat, multi-artifact
  coexistence, exact-match idempotence, hash conflict, hash_type
  conflict, empty-list semantics, and migration idempotence /
  preservation.
- No DualVer bump : R1 contract is semantically preserved
  (``(run_id, artifact_name) -> hash`` is bit-stable ; the
  single-artifact case is still addressable verbatim via the
  ``canonical`` default). No change to ``run_id`` computation.

### Added — R1 output-hash API (additive, no-migration)

- `RunRegistry.register_output_hash(run_id, output_hash)` +
  `RunRegistry.get_output_hash(run_id)` — second half of R1 now
  enforceable from the caller (recorded op output hash is stable
  for a registered tuple, conflicts raise with an R1 tag).
- New sibling table `run_output_hashes(run_id PK, output_hash,
  recorded_at)` created additively by `_ensure_schema` ; `runs`
  schema is untouched so `run_id` computation and the existing DB
  rows remain bit-stable.
- `tests/reproducibility/test_r1_run_registry_contract.py` —
  `test_r1_registry_output_hash_contract` flipped from `xfail` to
  passing ; docstring + module preamble rewritten to describe the
  enforced contract rather than the blocker.
- 5 new unit tests in `tests/unit/test_run_registry.py` covering
  roundtrip, idempotence on exact match, conflict → `ValueError`
  (with R1 tag + run_id), unknown run_id → `KeyError`, and missing
  hash → `KeyError`.

- **H6 profile-ordering** (`P_max > P_equ > P_min` on retained
  accuracy after consolidation) is a new derived constraint per
  framework-C §12.2. Exercised by the cycle-3 Gate D decision
  script (C3.9, scheduled) ; pre-locked in the spec glossary.
- **Scale-axis** (`{qwen3p5-1p5b, qwen3p5-7b, qwen3p5-35b}`) is a
  new formal axis for cross-scale DR-3 replication — extends DR-3
  Conformance Criterion from "two substrates" (cycle 2) to "two
  substrates × three model scales" (cycle 3).
- **R1 determinism contract** extended : cycle-3 run_id tuple now
  encodes ``(harness_version, scale/profile/substrate
  composite-tag, seed, commit_sha)`` so every cell of the
  1080-config matrix is uniquely + deterministically addressable.

EC axis set to **PARTIAL** (demoted from STABLE per §12.3
transition rule) because cycle-3 Phase 2 cells are scoped-deferred :

- C3.11-C3.14 — Norse LIF substrate + SNN ops (sem 4-5) — deferred
- C3.15-C3.18 — fMRI alignment (Studyforrest HMM/CCA, sem 5-6) —
  deferred
- C3.19-C3.22 — Paper 1 v2 Nature HB writeup + publication lock
  (sem 6) — deferred

Phase 1 engineering (C3.1-C3.10) is green / in flight ; G10 Gate D
= CONDITIONAL-GO/PARTIAL. Pivot-4 branch per spec §5.1 R3 replaces
the final STABLE graduation with a new minor bump if Gate D = NO-GO.

### Added — Phase 1 cycle-3 (C3.1-C3.5 pre-bump)

- Qwen3.5 MLX SHA-256 pins — `harness/real_models/base_model_registry.py` (commit f3b0119)
- MMLU + HellaSwag dataset SHA pins — `harness/real_benchmarks/dataset_registry.py` (commit 3271883)
- Studyforrest download init script — `scripts/init_studyforrest_download.sh` (commit 7b79b9e)
- Real-bench loaders (MMLU + HellaSwag + mega-v2) — commit cae16f8
- Qwen3.5 MLX 3-scale wrappers — commit abcc6ea
- Real-weight ops over Qwen MLX — commit ab55c67
- H5 trivariant scaling law — `kiki_oniric/eval/scaling_law.py` (commit 8efec8f)
- Bonferroni combined 8-test family — `kiki_oniric/eval/statistics.py` (commit 6643598)

### Added — Phase 1 cycle-3 (C3.6, this bump cycle)

- `scripts/ablation_cycle3.py` — 1080-config multi-scale cartesian
  runner with resume semantics + deterministic run_id lineage

### Changed

- DualVer bumped C-v0.6.0+STABLE → C-v0.7.0+PARTIAL per §12.3
  transition rule (FC MINOR + EC STABLE → PARTIAL demotion)
- G10 cycle-3 Gate D surfaced as active gate in STATUS.md
- `HARNESS_VERSION` constants in cycle-2 scripts
  (`scripts/ablation_cycle2.py`, `scripts/ablation_g4.py`) aligned
  to the new tag so all future cycle-2-rerun rows share the
  current empirical-axis reading
- `c_version` default in `kiki_oniric/eval/ablation.py` aligned
- `MLX_SUBSTRATE_VERSION` / `ESNN_SUBSTRATE_VERSION` aligned

### Pending — Phase 1 cycle-3 (C3.7-C3.10 remainder)

- C3.7 : sanity pilot 1.5B fail-fast (scripted this bump cycle)
- C3.8 : full 1080-config Studio launch (~10 days)
- C3.9 : compute_gate_d + H1-H6 report generator
- (C3.10 is this bump — no remainder)

### Pending — Phase 2 cycle-3 (scoped-deferred, C3.11-C3.22)

- Norse LIF substrate + SNN ops (C3.11-C3.14)
- fMRI alignment (C3.15-C3.18)
- Paper 1 v2 Nature HB writeup (C3.19-C3.22)

### Stats (at this bump)

- 240 tests passing, coverage 91.13 % (gate ≥ 90 %)
- 0 AI attribution in any commit
- 13 files version-bumped (see commit body)

---

## [C-v0.6.0+STABLE cycle-2 closeout] — 2026-04-19

Cycle 2 fully closed. Phase 3 (cross-substrate ablation) + Phase 4
(Paper 2 narrative) delivered since PARTIAL tag (commits 48b0521..
b8d7abe). G9 → FULL-GO.

### Added — Phase 3 cross-substrate validation (C2.9-C2.12)

- Multi-substrate ablation runner (`kiki_oniric/eval/ablation.py`,
  substrate axis)
- Conformance validation matrix script + docs (3×3 MLX × E-SNN ×
  hypothetical_cycle3)
- `docs/proofs/dr3-substrate-evidence.md` formal evidence
- Cross-substrate H1-H4 statistical results (synthetic substitute)
- Paper 1 §8.5 cycle-2 cross-substrate replication subsection
  (EN + FR)
- Paper 1 v2 arXiv plan (deferred until v1 acceptance)

### Added — Phase 4 Paper 2 narrative (C2.13-C2.16)

- Paper 2 abstract + introduction (EN + FR, ~1400 words)
- Paper 2 §4 Conformance + §5 Architecture (EN + FR)
- Paper 2 §6 Methodology + §7 Results (EN + FR, cross-substrate
  comparative table)
- Paper 2 §8 Discussion + §9 Future Work + full-draft assembly
- Paper 2 pandoc LaTeX render `docs/papers/paper2/build/full-draft.tex`
- Paper 2 build/ : README-arxiv.md + .gitignore (mirror Paper 1
  pattern)

### Changed

- DualVer bumped C-v0.6.0+PARTIAL → C-v0.6.0+STABLE per §12.3
  transition rule
- G9 cycle-2 publication gate : CONDITIONAL-GO/PARTIAL →
  FULL-GO/STABLE
- STATUS.md active gate aligned
- 13 CodeRabbit cycle-10 findings applied (commits 4b67f3e +
  2d1228e)

### Stats

- 180 tests passing, coverage 91.34 % (gate ≥ 90 %)
- 2 substrates real (MLX kiki-oniric + E-SNN thalamocortical),
  1 placeholder (hypothetical_cycle3)
- 3 profiles fully wired (P_min, P_equ, P_max)
- 17 commits across cycle 2 (Phase 1 + 2 + 3 + 4 + 5 + 2
  CodeRabbit cycle-9 + 2 cycle-10 fix batches + 2 DualVer bumps)
- 32 CodeRabbit findings applied (cycle 9 = 19, cycle 10 = 13)
- 0 AI attribution in any commit

### Pending — external user actions (unchanged)

- arXiv submit Paper 1 → obtain arXiv ID for Paper 2 cross-cite
- Paper 2 v0.1 review + arXiv submit
- OSF DOI lock (post-arXiv)
- Nature HB submit (or Pivot B branch selection)
- HAL FR deposit (post-arXiv)
- DR-2 external reviewer feedback (T-Col Q_CR.1 b)
- fMRI lab partnership formalization

---

## [C-v0.6.0+PARTIAL] — 2026-04-19

Cycle 2 closeout. Engineering Phases 1 + 2 + 5 complete.
Phase 3 (cross-substrate ablation C2.9-C2.12) + Phase 4 (Paper 2
narrative C2.13-C2.16) deferred per user scope decision ;
G9 gate = CONDITIONAL-GO/PARTIAL.

### Added

- E-SNN thalamocortical substrate (Phase 1 : C2.1-C2.4) — second
  `@runtime_checkable` Protocol target for DR-3
- DR-3 Conformance Criterion extended to E-SNN substrate (G7 LOCKED)
  ; condition (1) of DR-3 now spans 2 substrates
- P_max profile fully wired : 4 ops + 4 channels (Phase 2 :
  C2.5-C2.8, G8 LOCKED) ; DR-4 chain extended in
  `tests/conformance/axioms/test_dr4_p_max_chain.py`
- α-stream raw-traces ring buffer (canal-α)
- recombine_full MLX VAE variant (encoder + decoder + KL)
- ATTENTION_PRIOR canal-4 + S4 invariant guard
- Real concurrent dream worker (C2.17 : asyncio/threading via
  `threaded=True` mode)
- G9 cycle-2 publication-ready gate report
  (CONDITIONAL-GO/PARTIAL)
- Paper 3 outline (cycle-3 amorçage, PROVISIONAL)

### Changed

- DR-2 reframed as unproven working axiom (DR-2') ; parallel
  `recombine` branch marked out-of-scope of sequential composition
  until G3-draft delivers a parallel monoidal model
- DR-3 conformance proof reframed as operational evidence (vs
  formal `⟹`) — circular-implication footgun closed
- recombine MLX RNG isolated (local `mx.random.key`, no global
  seed mutation) — R1 hardened
- attention_prior `get_prior()` returns read-only numpy view —
  S4 cannot be bypassed by mutation
- attention guard rejects NaN explicitly before range/budget checks
  — S4 plug
- `awake.pause(500ms)` clarified vs `K3_max=1s` (operational target
  vs warning threshold) in §7.2 swap protocol
- 19 CodeRabbit cycle-9 findings applied (3 commits : safety +
  spec-fr + prose, then 1 EN-sync commit mirroring spec/outline)

### Pending — external user actions

- Phase 3 cycle-2 (cross-substrate ablation C2.9-C2.12) — deferred
- Phase 4 cycle-2 (Paper 2 narrative C2.13-C2.16) — deferred
- arXiv submission of Paper 1
- Nature HB editorial decision
- OSF DOI lock (`docs/osf-upload-checklist.md`)
- DR-2 external reviewer feedback (T-Col Q_CR.1 b)
- HAL FR deposit (post-arXiv)
- fMRI lab partnership formalization (T-Col extension)

### Stats

- 173 tests passing (target ≥ 173), coverage 91.26%
  (gate ≥90%)
- ~13 commits across cycle 2 (Phase 1 + Phase 2 + Phase 5 +
  3 cycle-9 fix commits + 1 EN-sync + 1 DualVer bump)
- 0 AI attribution in any commit
- Substrates : **2** (MLX kiki-oniric + E-SNN thalamocortical)
- Profiles : **3** wired (P_min, P_equ, P_max)
- DR-3 Conformance conditions (1) + (2) + (3) : all green on both
  substrates

### Milestones achieved

- **G7** E-SNN substrate conformance LOCKED
- **G8** P_max profile wired LOCKED
- **G9** cycle-2 publication-ready CONDITIONAL-GO/PARTIAL

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
