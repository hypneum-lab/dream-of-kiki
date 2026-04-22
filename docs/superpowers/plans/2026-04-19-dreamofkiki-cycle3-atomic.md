# dreamOfkiki cycle-3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 6-week cycle 3 delivering Paper 1 v2 Nature HB submission
via multi-scale real-data replication (Phase 1) + cross-substrate
neuromorphic / fMRI validation (Phase 2).

> **PLOS CB pivot (2026-04-20)** : Paper 1 retargeted PLOS
> Computational Biology (commit `d6866f3`) ; H1–H4 + multi-scale
> empirical cells **moved to Paper 2**. Tasks marked DEFERRED below
> are scope-deferred to the future Paper 2 backlog. The 11 DONE
> tasks remain authoritative ; sequencing is preserved for Paper 2
> reactivation. See
> `docs/milestones/cycle3-plan-adaptation-2026-04-20.md` for the
> full adaptation matrix, Phase B 1.5B GO 3/3 highlight kept in
> Paper 1 §7 pipeline-validation scope, and the Paper 2 backlog.

**Architecture:** cf. `docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md`.
Waterfall with hard GATE D at sem 3 end (bar : H1-H4 sig 3/3 scales +
H5 family Bonferroni + H6 profile ordering).

**Tech Stack:** Python 3.12+ uv, MLX-LM, scipy.stats, pytest +
hypothesis, pandoc + MacTeX, Studio M3 Ultra 512GB dedicated. Phase 2
tracks add Norse (PyTorch LIF, CPU fallback) and nilearn (HMM/CCA).

**DualVer:** `C-v0.6.0+STABLE → C-v0.7.0+PARTIAL` (C3.10) →
`C-v0.7.0+STABLE` (C3.22). Pivot-4 branch replaces final STABLE
graduation with a new minor bump if Gate D = NO-GO (cf cycle-3
design spec
`docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md`
§5.1 R3).

**Prerequisites cycle-3 start (sem 1 day 1-3 external locks, spec §5.2):**
- [x] SHA-256 pin Qwen3.5 1.5B/7B/35B Q4_K_M — `harness/real_models/base_model_registry.py` (commit `f3b0119`)
- [x] SHA-256 pin MMLU + HellaSwag versions — `harness/real_benchmarks/dataset_registry.py` (commit `3271883`)
- [x] Studyforrest download initiated — `scripts/init_studyforrest_download.sh` (commit `7b79b9e`)
- [ ] OSF pre-reg amendment filed (H5 trivariant I/II/III, H6, two-sided H5-II, family-size=8) — must complete *before* any C3.6 run

**Calendar :** 6 weeks (cycle 3 = S47-S52 in continuous calendar).

**Deferred / archived :**
- Paper 1 v1 held (not submitted again independently)
- Paper 2 v0.1 cross-substrate absorbed into Paper 1 v2
- Paper 3 outline archived under `docs/drafts/paper3-archived.md`
- Transformer / RWKV / state-space substrates (beyond Norse)
- Real fMRI lab partnership (Studyforrest open-data suffices)

---

## Convention commits (validator-enforced)

- Subject ≤50 chars, format `<type>(<scope>): <description>`
- Scope ≥3 chars
- Body lines ≤72 chars, 2-3 paragraphs required for large-diff commits
- Body sections : Context / Approach / Changes / Impact / Refs
- NO AI attribution
- NO `--no-verify`

---

## File structure after cycle 3

```
dreamOfkiki/
├── harness/
│   ├── real_benchmarks/                ← C3.1 (loaders use existing dataset_registry.py, 3271883)
│   │   ├── __init__.py
│   │   ├── dataset_registry.py         ✅ pre-locked (3271883)
│   │   ├── mmlu.py                     ← C3.1
│   │   ├── hellaswag.py                ← C3.1
│   │   ├── mega_v2_eval.py             ← C3.1
│   │   └── mega_v2_adapter.py          ← C3.3
│   ├── real_models/                    ← C3.2 (wrappers over existing base_model_registry.py, f3b0119)
│   │   ├── __init__.py
│   │   ├── base_model_registry.py      ✅ pre-locked (f3b0119)
│   │   └── qwen_mlx.py                 ← C3.2
│   └── fmri/                           ← C3.15
│       └── studyforrest.py
├── kiki_oniric/
│   ├── eval/
│   │   ├── statistics.py               ✅ extend (C3.5 Bonferroni 8-test)
│   │   ├── scaling_law.py              ← C3.4 (H5 trivariant)
│   │   ├── state_alignment.py          ← C3.16 (HMM dream↔BOLD)
│   │   └── cca_alignment.py            ← C3.17 (CCA fMRI)
│   ├── substrates/
│   │   └── esnn_norse.py               ← C3.11
│   └── dream/operations/
│       ├── replay_real.py              ← C3.3
│       ├── downscale_real.py           ← C3.3
│       ├── restructure_real.py        ← C3.3
│       ├── recombine_real.py           ← C3.3
│       ├── replay_snn.py               ← C3.12
│       ├── downscale_snn.py            ← C3.12
│       ├── restructure_snn.py         ← C3.12
│       └── recombine_snn.py            ← C3.12
├── scripts/
│   ├── pilot_cycle3_sanity.py          ← C3.7
│   ├── ablation_cycle3.py              ← C3.6
│   ├── compute_gate_d.py               ← C3.9
│   ├── pilot_phase2b_neuromorph.py     ← C3.13
│   └── pilot_phase2c_fmri.py           ← C3.18
├── tests/
│   ├── unit/
│   │   ├── test_real_benchmarks.py     ← C3.1
│   │   ├── test_mega_v2_adapter.py     ← C3.3
│   │   ├── test_qwen_mlx_wrappers.py   ← C3.2
│   │   ├── test_real_ops.py            ← C3.3
│   │   ├── test_scaling_law.py         ← C3.4
│   │   ├── test_statistics_bonferroni.py ← C3.5
│   │   ├── test_norse_substrate.py     ← C3.11
│   │   ├── test_snn_ops.py             ← C3.12
│   │   ├── test_studyforrest_loader.py ← C3.15
│   │   ├── test_state_alignment.py     ← C3.16
│   │   └── test_cca_alignment.py       ← C3.17
│   └── conformance/
│       └── operations/
│           └── test_substrate_matrix_cycle3.py ← C3.13
└── docs/
    ├── papers/paper1-v2/
    │   ├── outline.md                  ← C3.19
    │   ├── methodology.md              ← C3.20
    │   ├── results.md                  ← C3.20
    │   ├── discussion.md               ← C3.21
    │   ├── full-draft.md               ← C3.21
    │   └── build/full-draft.tex        ← C3.21
    ├── papers/paper1-v2-fr/            ← C3.20 mirror
    └── milestones/
        ├── g10-cycle3-gate-d.md        ← C3.9
        ├── g10a-neuromorph.md          ← C3.14
        ├── g10c-fmri.md                ← C3.18
        └── g10-cycle3-publication.md   ← C3.22
```

---

## Phase 1 — real-data multi-scale (sem 1-3, C3.1-C3.10)

### C3.1 — Real benchmark loaders

**Goal** : MMLU 5-shot + HellaSwag + mega-v2 80/20 self-eval loaders, consuming SHA-256 pins from the pre-locked `dataset_registry.py` (commit `3271883`). Replaces synthetic placeholders per `CLAUDE.md` §3.

**Files** :
- Create : `harness/real_benchmarks/__init__.py`
- Create : `harness/real_benchmarks/mmlu.py`
- Create : `harness/real_benchmarks/hellaswag.py`
- Create : `harness/real_benchmarks/mega_v2_eval.py`
- Create : `tests/unit/test_real_benchmarks.py`

**Pattern** : TDD 6 tests (loader determinism, SHA-256 verification against `dataset_registry.py`, schema validation, MMLU 5-shot sampling reproducible under fixed seed, HellaSwag zero-shot iterator, mega-v2 80/20 split deterministic).

**Commit** : `feat(real-bench): MMLU+HellaSwag+mega-v2 loaders` (48 chars)

### C3.2 — Multi-scale Qwen MLX wrappers

**Goal** : `qwen_mlx.py` wraps 1.5B/7B/35B Q4_K_M checkpoints over MLX-LM with seeded deterministic forward ; verifies SHA-256 pins against the pre-locked `base_model_registry.py` (commit `f3b0119`).

**Files** :
- Create : `harness/real_models/__init__.py`
- Create : `harness/real_models/qwen_mlx.py`
- Create : `tests/unit/test_qwen_mlx_wrappers.py`

**Pattern** : TDD 5 tests (forward determinism 1.5B under seed, SHA pin verification against registry, typed Protocol matching §2.1 primitives, K1 compute-budget tagging, multi-scale dispatch by `(scale, quant)` key).

**Commit** : `feat(real-models): Qwen3.5 MLX wrappers 3 scales` (48 chars)

### C3.3 — Mega-v2 adapter + real-weight dream ops

**Goal** : mega-v2 record ↔ `DreamEpisode` adapter emitting α/β streams per §2.1 ; four real-weight operations (`replay_real`, `downscale_real`, `restructure_real`, `recombine_real`) extending the synthetic cycle-2 ops with weight-tensor IO bound to `qwen_mlx.py`. Respects DR-2 canonical order, reuses S1/S2/S3 guards.

**Files** :
- Create : `harness/real_benchmarks/mega_v2_adapter.py`
- Create : `kiki_oniric/dream/operations/replay_real.py`
- Create : `kiki_oniric/dream/operations/downscale_real.py`
- Create : `kiki_oniric/dream/operations/restructure_real.py`
- Create : `kiki_oniric/dream/operations/recombine_real.py`
- Create : `tests/unit/test_mega_v2_adapter.py`
- Create : `tests/unit/test_real_ops.py`

**Pattern** : TDD 8 tests (4 adapter round-trip record→DreamEpisode→record, 4 ops × (DR-2 order preserved, S1 budget respected, channel 1-4 emission typed, weight-tensor shape invariant)).

**Commit** : `feat(real-ops): real-weight ops over Qwen MLX` (45 chars)

### C3.4 — scaling_law.py — H5 trivariant analyzer

**Goal** : implement H5-I (Levene / Brown-Forsythe variance invariance), H5-II (Spearman ρ, **two-sided** per OSF pre-reg amendment), H5-III (`scipy.optimize.curve_fit d = c·N^α` + bootstrap 95% CI on α, seeded per R1). H5-III non-significant outcome is publishable.

**Files** :
- Create : `kiki_oniric/eval/scaling_law.py`
- Create : `tests/unit/test_scaling_law.py`

**Pattern** : TDD 6 tests (Levene on synthetic equal-variance d-curves, Brown-Forsythe fallback, Spearman two-sided on monotone synthetic, curve_fit recovers known α on synthetic power-law, bootstrap CI seeded deterministic, H5-III null-outcome reported without crash).

**Commit** : `feat(eval): scaling_law H5 trivariant analyzer` (46 chars)

### C3.5 — statistics.py — Bonferroni 8-test family

**Goal** : extend `kiki_oniric/eval/statistics.py` with per-cell family-wise correction ; `α_per_test = 0.05 / 8 = 0.00625` ; explicit refusal to cross-Bonferroni across the 6 cells (would degrade to 0.001, under-powered).

**Files** :
- Modify : `kiki_oniric/eval/statistics.py`
- Create : `tests/unit/test_statistics_bonferroni.py`

**Pattern** : TDD 5 tests (API `apply_bonferroni(p_values, family_size=8)` returns list[bool] of rejections, family_size=8 enforced, α=0.00625 exact, cross-cell correction raises NotImplementedError with spec citation, backward-compat with cycle-2 callers).

**Commit** : `feat(stats): Bonferroni 8-test family per cell` (46 chars)

### C3.6 — ablation_cycle3.py — 1080-config runner

> **Prerequisite** : requires C3.10 (DualVer bump
> C-v0.6.0+STABLE → C-v0.7.0+PARTIAL) to be committed *before*
> the 1080-config launch so every run_id is registered under the
> new tag. Execution sequence is C3.5 → C3.10 → C3.6, not the
> numeric C3.5 → C3.6 → … → C3.10.

**Goal** : cartesian `scale ∈ {1.5B, 7B, 35B} × profile ∈ {P_min, P_equ, P_max} × benchmark ∈ {MMLU, HellaSwag, mega-v2} × seed ∈ 1..40` = 1080 runs. Idempotent resume, run-registry manifest per framework-C spec §8.3 (R1 bit-exact + R3 artifact addressability), K4 matrix coverage gate.

**Files** :
- Create : `scripts/ablation_cycle3.py`

**Pattern** : script + smoke test (invoke with `--dry-run` enumerates exactly 1080 configs, validates against registry, writes manifest). No TDD unit test (runner is a driver ; logic is in eval/ modules already TDD-covered).

**Commit** : `feat(harness): cycle-3 1080-config ablation runner` (50 chars)

### C3.7 — Sanity pilot 1.5B (fail-fast)

**Goal** : 180 runs (1.5B × 3 profiles × 3 benchmarks × 20 seeds), ~18h wall-clock, fails fast on (a) any S2 violation, (b) K1 budget exceedance > 2×, (c) S1 regression > 5% on retained. Emits go/no-go to launch C3.8.

**Files** :
- Create : `scripts/pilot_cycle3_sanity.py`

**Pattern** : script + doc (bash -n) ; 3 fail-fast predicates unit-tested by reusing `test_real_ops.py` S1/S2 coverage ; GO/NO-GO decision JSON written under run-registry.

**Commit** : `feat(pilot): cycle-3 1.5B sanity fail-fast` (42 chars)

### C3.8 — Full ablation Studio launch

> **PARTIAL — PLOS CB pivot (2026-04-20)** : 1.5B cell delivered as
> Phase B sanity pilot (commit `22c58c9`, Verdict GO 3/3,
> 46.75 min wall-clock on Studio). 7B + 35B cells **DEFERRED to
> Paper 2** (multi-scale empirical claims removed from Paper 1
> v0.2 PLOS CB scope per §8.3).

**Goal** : background launch `ablation_cycle3.py` on Studio M3 Ultra after sanity GO ; ~10 days wall-clock budget per §7 ; partial results streamed for incremental Gate D dry-runs sem 3 mid.

**Files** :
- No new code (executes `ablation_cycle3.py` via Studio dedicated session).
- Log milestone : `docs/milestones/cycle3-ablation-launch.md` (date/SHA/Studio session ID).

**Pattern** : chore commit logging launch event ; no tests. Resume protocol documented ; dashboard `dream.saillant.cc` monitors K3 per-swap latency.

**Commit** : `chore(ablation): cycle-3 Studio full launch sem 2` (49 chars)

### C3.9 — compute_gate_d.py — GATE D decision

> **DEFERRED to Paper 2 (PLOS CB pivot 2026-04-20)** : Gate D
> hinges on H1–H4 sig 3/3 *scales*, but only the 1.5B cell is
> available post-pivot. Gate D moves to Paper 2 once C3.8 7B + 35B
> cells are re-executed.

**Goal** : reads run-registry full-ablation output, runs the 8-test Bonferroni family (C3.5) per cell, computes H5 trivariant (C3.4), emits per-hypothesis GO/NO-GO table, renders Gate D verdict. GO iff (H1-H4 sig 3/3 scales) ∧ (H5 family Bonferroni-significant) ∧ (H6 profile ordering P_max>P_equ>P_min).

**Files** :
- Create : `scripts/compute_gate_d.py`
- Create : `docs/milestones/g10-cycle3-gate-d.md`

**Pattern** : TDD 5 tests (synthetic run-registry fixture → GO path, one-missing-scale → NO-GO, H5 non-sig → NO-GO, Bonferroni family-size=8 enforced, deterministic report under R1 seeded bootstrap).

**Commit** : `feat(gate): compute_gate_d + H1-H6 report` (41 chars)

### C3.10 — DualVer bump C-v0.6.0+STABLE → C-v0.7.0+PARTIAL

> **Execution order note** : despite the C3.10 numeric label, this
> task is a prerequisite of C3.6. The intended sequence is
> **C3.5 → C3.10 → C3.6** so the 1080-config matrix runs under the
> new tag. The numeric order reflects authoring sequence (DualVer
> paperwork was spec'd after the launcher), not execution sequence.
> C3.6 must not be launched until this bump is committed — see the
> prerequisite note added to the C3.6 section above.

**Goal** : FC minor bump (H6 adds a derived constraint surface per framework-C §12.2) + EC STABLE → PARTIAL (Phase 2 cells scoped-deferred until sem 6, §12.3). Performed *after* C3.5 lands and *before* C3.6 launches so the 1080-matrix runs under the new tag.

**Files** :
- Modify : framework-C spec banner (FR + EN) in `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
- Modify : `STATUS.md`, `CHANGELOG.md`
- Modify : `docs/glossary.md` (add G10, real_benchmarks, scale-axis, H5/H6 entries per spec §8)
- Modify : substrate version constants (search for `C-v0.6.0` references)

**Pattern** : surgical-bump pattern per cycle-2 closeout `139c4c5` reference ; zero test diff (version strings only).

**Commit** : `feat(dualver): bump to C-v0.7.0+PARTIAL` (39 chars)

---

## Phase 2 — parallel tracks (sem 4-6, conditional Gate D = GO, C3.11-C3.22)

If Gate D = NO-GO → execute Pivot 4 (cycle-3 design spec `docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md` §5.1 R3) ; re-spec cycle 3 sem 4 onward and replace the C-v0.7.0+STABLE graduation with a new minor bump reflecting the pivot scope.

### C3.11 — Norse fallback substrate wrapper

**Goal** : Norse PyTorch LIF SNN wrapper as Loihi-2 CPU fallback per axiom DR-3 (substrate-agnosticism, framework-C spec §6.2). Implements Conformance Criterion §6.2 : typed signature + axiom property tests + invariants S1/S2/S3/I1 enforceable.

**Files** :
- Create : `kiki_oniric/substrates/esnn_norse.py`
- Create : `tests/unit/test_norse_substrate.py`

**Pattern** : TDD 5 tests (LIF forward determinism, spike-rate proxy non-trivial, typed Protocol matches cycle-2 E-SNN abstraction, S1/S2/S3 guards enforceable, I1 identity invariant).

**Commit** : `feat(substrate): Norse LIF SNN fallback wrapper` (47 chars)

### C3.12 — Dream ops on Norse SNN

**Goal** : 4 ops (`replay_snn`, `downscale_snn`, `restructure_snn`, `recombine_snn`) on Norse substrate using spike-rate proxy ; same channel typings as `_real.py` per §2.1 ; DR-2 canonical order preserved.

**Files** :
- Create : `kiki_oniric/dream/operations/replay_snn.py`
- Create : `kiki_oniric/dream/operations/downscale_snn.py`
- Create : `kiki_oniric/dream/operations/restructure_snn.py`
- Create : `kiki_oniric/dream/operations/recombine_snn.py`
- Create : `tests/unit/test_snn_ops.py`

**Pattern** : TDD 5 tests (spike-rate proxy non-zero per op, DR-3 conservation cross-substrate vs MLX baseline, S1 budget respected, K1 compute tagged, determinism under seed).

**Commit** : `feat(snn-ops): dream ops over Norse substrate` (45 chars)

### C3.13 — Phase-2b cross-substrate pilot

> **DEFERRED to Paper 2 (PLOS CB pivot 2026-04-20)** : sem-3 quota
> interrupted execution. Driver `scripts/pilot_phase2b_neuromorph.py`
> (624 LOC) preserved as-is with deferred-note header (commit
> `ada3fc0`). Paper 2 reactivation entry point.

**Goal** : Norse vs MLX cross-substrate pilot via `compute_gate_d.py` reused on the Norse cell (Bonferroni stays per-cell, **not** across substrates per spec §4). 3 substrates × 3 profiles × 4 ops conformance matrix update.

**Files** :
- Create : `scripts/pilot_phase2b_neuromorph.py`
- Create : `tests/conformance/operations/test_substrate_matrix_cycle3.py`

**Pattern** : TDD 4 tests (matrix enumerates 3×3×4, each cell has typed result, Norse cell Bonferroni-significant on synthetic fixture, H6 profile ordering persists).

**Commit** : `feat(pilot): Phase-2b Norse cross-substrate` (43 chars)

### C3.14 — G10a neuromorph milestone

> **DEFERRED to Paper 2 (PLOS CB pivot 2026-04-20)** : depends on
> C3.13 outputs ; reactivation tied to Paper 2 multi-scale sprint.

**Goal** : milestone report : H6 (profile ordering invariant cross-substrate) verdict on the Norse cell ; feeds Paper 1 v2 §§ cross-substrate.

**Files** :
- Create : `docs/milestones/g10a-neuromorph.md`

**Pattern** : docs-only commit ; cites run-registry IDs from C3.13 pilot, H6 verdict table, M1.b and M2 deltas MLX vs Norse.

**Commit** : `docs(milestone): G10a neuromorph cross-substrate` (48 chars)

### C3.15 — Studyforrest BOLD loader

**Goal** : BOLD loader with SHA-256 pin per R1 §8.4 ; nilearn CPU-deterministic mode ; fallback HCP ds000113 if Studyforrest access slow (R4). Consumes data path from `scripts/init_studyforrest_download.sh` (commit `7b79b9e`).

**Files** :
- Create : `harness/fmri/__init__.py`
- Create : `harness/fmri/studyforrest.py`
- Create : `tests/unit/test_studyforrest_loader.py`

**Pattern** : TDD 5 tests (loader determinism, SHA-256 pin verification, fallback to HCP ds000113 triggers correctly, BOLD ROI extraction seeded, nilearn CPU mode enforced).

**Commit** : `feat(fmri): Studyforrest BOLD loader` (36 chars)

### C3.16 — HMM dream-state alignment

**Goal** : HMM Viterbi alignment between dream-episode state sequence and BOLD activation phases ; metric M2.b (RSA fMRI alignment) extension.

**Files** :
- Create : `kiki_oniric/eval/state_alignment.py`
- Create : `tests/unit/test_state_alignment.py`

**Pattern** : TDD 4 tests (Viterbi on synthetic BOLD recovers known sequence, HMM emission seeded deterministic, state-count enforced from P-profile cardinality, M2.b metric scalar returned).

**Commit** : `feat(eval): HMM dream-state ↔ BOLD alignment` (44 chars)

### C3.17 — CCA Studyforrest alignment

**Goal** : CCA between kiki representations (4 ortho species ρ_phono/lex/syntax/sem) and Studyforrest BOLD ROIs ; complements RSA M2.b with cross-decomposition.

**Files** :
- Create : `kiki_oniric/eval/cca_alignment.py`
- Create : `tests/unit/test_cca_alignment.py`

**Pattern** : TDD 4 tests (CCA round-trip on aligned synthetic, 4-species shape invariant, sklearn CCA seeded deterministic, correlation coefficient bounded [0, 1]).

**Commit** : `feat(eval): CCA Studyforrest alignment` (38 chars)

### C3.18 — Phase-2c fMRI pilot + G10c milestone

> **DEFERRED to Paper 2 (PLOS CB pivot 2026-04-20)** : C3.15
> Studyforrest loader + C3.16 HMM + C3.17 CCA infra DONE in-tree
> ; only the Phase-2c pilot driver and G10c milestone are
> deferred. Paper 1 v0.2 §5.6 cross-substrate walkthrough already
> cites the C3.16 + C3.17 infra without invoking the pilot run.

**Goal** : Phase-2c pilot run (M2.b real-data cell) + milestone report ; feeds Paper 1 v2 §§ cognitive alignment.

**Files** :
- Create : `scripts/pilot_phase2c_fmri.py`
- Create : `docs/milestones/g10c-fmri.md`

**Pattern** : script + doc (bash -n) ; reuses C3.16/C3.17 eval modules (already TDD-covered) ; milestone cites run-registry IDs + M2.b deltas.

**Commit** : `feat(pilot): Phase-2c Studyforrest fMRI` (39 chars)

### C3.19 — Paper 1 v2 outline

> **REPLACED by Paper 1 v0.2 PLOS CB workflow (2026-04-20)** :
> narrative authored directly in `docs/papers/paper1/` (single
> tree, not `paper1-v2/`). The C3.19–C3.22 sequence now belongs
> to a future Paper 2 narrative once Gate D verdict lands.

**Goal** : outline merging v1 sections + Paper 2 v0.1 §§ cross-substrate (cycle-2) + cycle-3 §§ real-data + §§ cross-substrate Norse + §§ scaling laws + §§ fMRI alignment ; cites every G-gate, axiom, invariant.

**Files** :
- Create : `docs/papers/paper1-v2/outline.md`

**Pattern** : docs-only commit ; structural outline only, full prose deferred to C3.20/C3.21 ; cross-references to G1..G10c + DR-0..DR-4 + I/S/K + H1..H6.

**Commit** : `docs(paper1-v2): outline merge v1+cycle-2+cycle-3` (49 chars)

### C3.20 — Paper 1 v2 methodology + results (EN + FR)

> **REPLACED by Paper 1 v0.2 PLOS CB workflow (2026-04-20)** :
> methodology + results sections live in `docs/papers/paper1/`
> (W-series revisions, commit `d6866f3`). H1–H4 verdicts moved to
> Paper 2.

**Goal** : methodology covers 3 substrates (MLX + Norse + fMRI alignment cell) × 3 scales × 8-test Bonferroni family ; results table per cell with H1-H6 verdicts. Bilingual EN/FR mirrored per CLAUDE.md authorship policy.

**Files** :
- Create : `docs/papers/paper1-v2/methodology.md` (EN)
- Create : `docs/papers/paper1-v2/results.md` (EN)
- Create : `docs/papers/paper1-v2-fr/methodology.md` (FR mirror)
- Create : `docs/papers/paper1-v2-fr/results.md` (FR mirror)

**Pattern** : docs-only commit ; results table ingests run-registry JSON from C3.8 + C3.13 + C3.18 (no synthetic numbers, R1 contract respected).

**Commit** : `docs(paper1-v2): methodology + results EN/FR` (44 chars)

### C3.21 — Paper 1 v2 discussion + full-draft assembly

> **REPLACED by Paper 1 v0.2 PLOS CB workflow (2026-04-20)** :
> Paper 1 v0.2 full-draft already rendered to
> `docs/papers/paper1/build/full-draft.pdf` (22 pages, 296 KB).
> arXiv submission prep is the live critical path.

**Goal** : discussion + future-work + full-draft assembly + pandoc tex render ; respects PUBLICATION-READY criteria framework-C §9.

**Files** :
- Create : `docs/papers/paper1-v2/discussion.md`
- Create : `docs/papers/paper1-v2/full-draft.md`
- Create : `docs/papers/paper1-v2/build/full-draft.tex`

**Pattern** : docs-only commit ; pandoc build verified locally (MacTeX) ; zero TODO in final draft per G10 criterion.

**Commit** : `docs(paper1-v2): discussion + full-draft assembly` (49 chars)

### C3.22 — DualVer bump C-v0.7.0+PARTIAL → C-v0.7.0+STABLE + G10 gate

> **REPLACED / DEFERRED to Paper 2 (PLOS CB pivot 2026-04-20)** :
> EC remains `+PARTIAL` through Paper 1 v0.2 PLOS CB submission.
> Promotion to `+STABLE` deferred until Paper 2 closeout (Phase 2
> deferred cells re-closed under §12.3 transition rule).

**Goal (deferred to Paper 2 per the §12.3 transition rule above)** :
*Planned* — EC `+PARTIAL` → `+STABLE` (Phase 2 deferred cells re-closed) ; Gate G10 *would be* promoted CONDITIONAL → FULL-GO/STABLE on the same surgical-bump pattern as C3.10 / `139c4c5`. Until Paper 2 closeout, EC stays at `+PARTIAL` and Gate G10 stays CONDITIONAL ; the C3.10 / `139c4c5` reference is retained as the canonical bump template for that future commit.

**Files** :
- Modify : framework-C spec banner (FR + EN), `STATUS.md`, `CHANGELOG.md`, substrate version constants
- Create : `docs/milestones/g10-cycle3-publication.md`

**Pattern** : surgical-bump commit ; zero test diff ; G10 milestone cross-references G10a (C3.14) + G10c (C3.18) + Gate D (C3.9) + Paper 1 v2 draft (C3.21).

**Commit** : `feat(dualver): bump to C-v0.7.0+STABLE` (38 chars)

---

## Self-review

**1. Spec coverage** :
- §2 Phase 1 tasks C3.1-C3.10 → 10 tasks ✅
- §3 Phase 2 tasks C3.11-C3.22 → 12 tasks ✅
- §4 hypotheses H1-H6 → formalized in C3.4 (H5 trivariant) + C3.5 (Bonferroni 8-test) + C3.9 (Gate D decision) + C3.13 (H6 cross-substrate) ✅
- §5 risks (R1-R5) + pre-cycle-3 locks → prerequisites table + C3.1/C3.2/C3.15 consume pre-locked pins ✅
- §6 DualVer G10 → C3.10 + C3.22 surgical bumps ✅
- §7 compute budget → C3.7 sanity + C3.8 full-launch envelope ✅
- §8 glossary delta → C3.10 glossary.md updates ✅
- §9 cross-refs → spec citations in each task's Pattern / Goal ✅

**2. Placeholder scan** : zero TBD/TODO/"fill in later". Abbreviated 4-field pattern delibérée (cycle-2 proved convergence on 18 tasks).

**3. Type consistency** :
- C3.2 `qwen_mlx.py` exports wrappers matching §2.1 primitives — consumed by C3.3 `_real.py` ops unchanged
- C3.3 `mega_v2_adapter.py` emits α/β streams — consumed by C3.6 ablation runner
- C3.5 `apply_bonferroni(p_values, family_size=8)` — invoked by C3.9 `compute_gate_d.py` and C3.13 pilot
- C3.4 `scaling_law.py` H5 API — invoked by C3.9 `compute_gate_d.py`
- C3.11 `esnn_norse.py` Protocol — consumed by C3.12 `_snn.py` ops
- C3.15 `studyforrest.py` — consumed by C3.16/C3.17 eval modules and C3.18 pilot

**4. Commit count** : 22 commits (10 Phase 1 + 12 Phase 2). Matches spec §3 "Total cycle 3 : 22 commits".

**5. Validator risks** : all subjects pre-verified ≤50 chars (max = 50 on C3.6, C3.19, C3.21).

**6. Critical-path dependencies** :
- OSF pre-reg amendment — must file sem 1 day 1-3 before any C3.6 run
- Studio M3 Ultra dedication sem 2-3 (10-day wall-clock)
- Qwen3.5 MLX checkpoints on Studio (copy from `kxkm-ai:/mnt/models/` sem 1, hash-verify against `f3b0119` registry)
- Studyforrest `git-annex` clone completing by sem 4 (fallback HCP ds000113 if R4 fires)
- Gate D verdict sem 3 end — Phase 2 branch (C3.11-C3.22) conditional ; Pivot 4 branch replaces STABLE graduation

**7. OSF pre-reg lock reminder** : H5-II direction is **two-sided** per pre-reg amendment — **never modify** post-C3.4. Any post-hoc one-sided claim would invalidate Paper 1 v2 Nature HB registered-report integrity.

---

## Execution Handoff

Plan complete and saved. Two execution options :

**1. Subagent-Driven (recommended)** — dispatch fresh subagent per
task, review between tasks, fast iteration. Works well for abbreviated
pattern ; cycle-2 proved convergence on 18 tasks using this mode.

**2. Inline Execution** — batch execution with human-in-loop
checkpoints, use `superpowers:executing-plans`.

---

**End of cycle-3 atomic plan.**

**Version** : v0.1.0
**Generated** : 2026-04-19 via `superpowers:writing-plans`
**Source** : `docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md` (commit `f9b2721`)
