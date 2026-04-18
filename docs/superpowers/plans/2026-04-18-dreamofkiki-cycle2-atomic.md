# dreamOfkiki Cycle 2 Atomic Plan

> **Pour agents autonomes :** SKILL REQUIS — utiliser `superpowers:subagent-driven-development`. Les steps utilisent la syntaxe checkbox (`- [ ]`).

**Goal** : valider empiriquement la substrate-agnosticism du framework C via E-SNN thalamocortical, finaliser P_max profile, livrer Paper 2 (engineering ablation, NeurIPS/ICML/TMLR), produire la conformance validation matrix multi-substrat.

**Architecture** : 18 tasks atomiques organisées en 5 phases : C2.1-C2.4 (E-SNN substrate setup), C2.5-C2.8 (P_max real wiring), C2.9-C2.12 (cross-substrate validation), C2.13-C2.16 (Paper 2 narrative), C2.17-C2.18 (cycle 2 closeout).

**Tech Stack** : Python 3.12+ uv, **MLX** (existing), **NEST/Brian2/Norse** (E-SNN simulation if Loihi-2 unavailable), **Intel NxSDK / NxNet** (if Loihi-2 access granted), pytest + hypothesis, scipy.stats (existing).

**Préréquis cycle-1 closeout** :
- 99 commits dreamOfkiki, dernier `7760b47 fix: 7 CodeRabbit cycle-6 findings`
- 119 tests passing, coverage 91.86%
- Framework C-v0.5.0+STABLE (target bump → C-v0.7.0+STABLE post-G3)
- Paper 1 narrative complete + LaTeX render
- 6 atomic plans cycle-1 livrés
- Cycle 2 amorçage : Paper 2 outline (S27.1) + G6 decision report (S28.1)

**Préréquis externes (utilisateur)** :
- ✅ Paper 1 arXiv submitted (preprint URL available for Paper 2 citation)
- ✅ Paper 1 Nature HB submitted (or Pivot B branch active)
- ✅ Intel NRC Loihi-2 access granted OR fallback E-SNN simulation accepted
- ✅ Real mega-v2 dataset access OR synthetic-acceptance documented
- ✅ DR-2 reviewer feedback received (G3 closed)

**Calendar** : ~12-18 weeks (cycle 2 = S29-S46 in continuous calendar)

**Deferred to cycle 3** :
- Transformer / RWKV / state-space substrates (3+ substrate matrix)
- Production deployment patterns
- Real fMRI lab partnership extension

---

## Convention commits (validator-enforced)

- Subject ≤50 chars, format `<type>(<scope>): <description>`
- Scope ≥3 chars
- Body lines ≤72 chars, 2-3 paragraphs required
- NO AI attribution
- NO `--no-verify`

---

## File structure après cycle 2

```
dreamOfkiki/
├── kiki_oniric/
│   ├── dream/operations/
│   │   ├── replay.py                  ✅ existing
│   │   ├── downscale.py               ✅ existing
│   │   ├── restructure.py             ✅ existing
│   │   ├── recombine.py               ✅ existing → recombine_full added
│   │   └── concurrent.py              ✅ existing → real asyncio worker
│   ├── substrates/                    ← NEW C2.1-C2.4
│   │   ├── __init__.py
│   │   ├── mlx_kiki_oniric.py         (existing kiki-oniric reorg)
│   │   ├── esnn_thalamocortical.py    ← C2.2 (Norse or NxNet backend)
│   │   └── conformance.py             ← C2.4 (suite reusable)
│   └── profiles/
│       ├── p_min.py                   ✅ existing
│       ├── p_equ.py                   ✅ existing
│       └── p_max.py                   ✅ skeleton → C2.5-C2.8 fully wired
├── tests/
│   ├── unit/
│   │   ├── test_esnn_substrate.py     ← C2.2
│   │   ├── test_p_max_wiring.py       ← C2.6-C2.7
│   │   ├── test_alpha_stream.py       ← C2.5
│   │   └── test_attention_prior.py    ← C2.7
│   └── conformance/
│       ├── axioms/
│       │   └── test_dr3_esnn_substrate.py  ← C2.4 (DR-3 second instance)
│       └── operations/
│           └── test_substrate_matrix.py    ← C2.10 (cross-substrate)
├── scripts/
│   ├── ablation_g4.py                 ✅ existing
│   ├── ablation_cycle2.py             ← C2.9 (multi-substrate ablation)
│   └── conformance_matrix.py          ← C2.10 (validation matrix runner)
├── docs/
│   ├── papers/
│   │   ├── paper1/                    ✅ cycle-1 complete
│   │   └── paper2/
│   │       ├── outline.md             ✅ existing
│   │       ├── abstract.md            ← C2.13
│   │       ├── introduction.md        ← C2.13
│   │       ├── conformance-section.md ← C2.14
│   │       ├── architecture.md        ← C2.14
│   │       ├── methodology.md         ← C2.15
│   │       ├── results.md             ← C2.15
│   │       ├── discussion.md          ← C2.16
│   │       └── full-draft.md          ← C2.16
│   ├── milestones/
│   │   ├── g7-esnn-conformance.md     ← C2.4 (E-SNN passes Conformance)
│   │   ├── g8-p-max-functional.md     ← C2.8 (P_max wired + tested)
│   │   ├── g9-cycle2-publication.md   ← C2.18 (Paper 2 ready)
│   │   └── conformance-matrix.md      ← C2.10 (multi-substrate report)
│   └── proofs/
│       └── dr3-substrate-evidence.md  ← C2.10 (formal evidence multi-substrate)
└── papers/paper2/build/               ← C2.16 (LaTeX render)
```

---

# Phase 1 — E-SNN substrate setup (C2.1-C2.4)

## C2.1 — Substrate abstraction refactor

**Goal** : extract substrate-specific code from kiki_oniric into `substrates/` directory. Cycle-1 code becomes `substrates/mlx_kiki_oniric.py`. Foundation for cycle-2 second substrate.

**Files** :
- Create : `kiki_oniric/substrates/__init__.py`
- Create : `kiki_oniric/substrates/mlx_kiki_oniric.py` (move from existing)
- Update imports in `kiki_oniric/profiles/{p_min,p_equ,p_max}.py`

**Pattern** : pure refactor, no test changes. Verify all 119 tests still pass.

**Commit** : `refactor(substrate): extract MLX substrate module` (47 chars)

## C2.2 — E-SNN substrate skeleton

**Goal** : implement `kiki_oniric/substrates/esnn_thalamocortical.py` skeleton. Choose backend : **Norse** (PyTorch-based, install via `pip install norse`) for simulation OR **NxNet** if Loihi-2 access granted. Skeleton declares the substrate identity and provides stubs for the 8 typed Protocols.

**Files** :
- Create : `kiki_oniric/substrates/esnn_thalamocortical.py`
- Create : `tests/unit/test_esnn_substrate.py`

**Pattern** : TDD 4 tests (substrate instantiable, backend choice persisted, Protocol signatures present, skeleton ops raise NotImplementedError as expected for cycle-2 work-in-progress)

**Commit** : `feat(esnn): add thalamocortical substrate skel` (47 chars)

## C2.3 — E-SNN operations wiring

**Goal** : wire 4 operations (replay/downscale/restructure/recombine) on E-SNN substrate. Each operation maps to spike-rate dynamics rather than dense matrix ops.

**Files** :
- Modify : `kiki_oniric/substrates/esnn_thalamocortical.py`
- Create : `tests/unit/test_esnn_operations.py`

**Pattern** : TDD 4 tests (one per operation, verify spike-rate output is non-trivial). For Norse backend, use `LIFCell` (Leaky Integrate-and-Fire) as the basic unit.

**Commit** : `feat(esnn): wire 4 operations on substrate` (43 chars)

## C2.4 — DR-3 Conformance Criterion validation on E-SNN

**Goal** : run the existing conformance test suite (`tests/conformance/axioms/test_dr3_substrate.py`) with the E-SNN substrate plugged in. Verify all 3 conditions pass : signature typing + axiom property tests + BLOCKING invariants enforceable.

**Files** :
- Create : `tests/conformance/axioms/test_dr3_esnn_substrate.py`
- Create : `docs/milestones/g7-esnn-conformance.md` (G7 gate report)

**Pattern** : the conformance suite is generic ; instantiate it with E-SNN substrate as parameter. If all 3 conditions pass, G7 gate locked.

**Commit** : `test(dr3): E-SNN passes Conformance` (35 chars)

---

# Phase 2 — P_max real wiring (C2.5-C2.8)

## C2.5 — α-stream raw traces input channel

**Goal** : implement the α channel (raw forward-pass traces firehose) as a ring buffer with bounded retention. Currently P_max-only declared but not consumed.

**Files** :
- Create : `kiki_oniric/dream/channels/alpha_stream.py`
- Create : `tests/unit/test_alpha_stream.py`

**Pattern** : TDD 4 tests (ring buffer wraps at capacity, retention bounded, FIFO/LIFO selectable, integrity check on overflow)

**Commit** : `feat(alpha): add raw traces ring buffer` (39 chars)

## C2.6 — recombine_full operation variant (full VAE)

**Goal** : implement `recombine_full` operation beyond the cycle-1 light variant. Full VAE encoder/decoder with proper reparameterization + KL divergence loss.

**Files** :
- Modify : `kiki_oniric/dream/operations/recombine.py` (add `recombine_handler_full_mlx`)
- Create : `tests/unit/test_recombine_full.py`

**Pattern** : TDD 3 tests (full VAE round-trip dimension check, KL divergence non-negative, sample diversity over N runs).

**Commit** : `feat(recombine): add full VAE variant` (38 chars)

## C2.7 — ATTENTION_PRIOR canal-4 emission + S4 invariant

**Goal** : implement the ATTENTION_PRIOR output channel emission with the S4 invariant (each component in [0, 1], sum ≤ budget_attention). Currently declared in framework spec but unwired.

**Files** :
- Create : `kiki_oniric/dream/channels/attention_prior.py`
- Create : `kiki_oniric/dream/guards/attention.py` (S4 invariant enforcer)
- Create : `tests/unit/test_attention_prior.py`
- Create : `tests/conformance/invariants/test_s4_attention.py`

**Pattern** : TDD 6 tests (4 unit + 2 conformance) covering channel emission + S4 enforcement.

**Commit** : `feat(canal4): wire ATTENTION_PRIOR + S4 guard` (45 chars)

## C2.8 — P_max profile fully wired + G8 gate

**Goal** : wire P_max profile = 4 ops + 4 channels (α+β+δ → 1+2+3+4) + recombine_full + ATTENTION_PRIOR. Test end-to-end.

**Files** :
- Modify : `kiki_oniric/profiles/p_max.py` (replace skeleton with full wiring)
- Create : `tests/unit/test_p_max_wiring.py`
- Create : `docs/milestones/g8-p-max-functional.md` (G8 gate report)

**Pattern** : TDD 5 tests (handler registration, 4-op sequential execution, all 4 channels emitted, log integrity, status flipped to "wired").

**Commit** : `feat(profile): wire P_max (4 ops + 4 chans)` (45 chars)

---

# Phase 3 — Cross-substrate validation (C2.9-C2.12)

## C2.9 — Multi-substrate ablation runner

**Goal** : extend `AblationRunner` to support multi-substrate runs (MLX kiki-oniric × E-SNN). Cartesian product (substrate × profile × seed).

**Files** :
- Modify : `kiki_oniric/eval/ablation.py` (add `substrate` parameter)
- Create : `scripts/ablation_cycle2.py` (full multi-substrate run)
- Create : `tests/unit/test_ablation_multi_substrate.py`

**Commit** : `feat(eval): multi-substrate ablation runner` (43 chars)

## C2.10 — Conformance validation matrix

**Goal** : produce the matrix : for each substrate (MLX, E-SNN, hypothetical 3rd) × each conformance condition (signature typing, axiom tests, invariants enforceable), run the suite and report pass/fail.

**Files** :
- Create : `scripts/conformance_matrix.py`
- Create : `docs/milestones/conformance-matrix.md` (results dump)
- Create : `docs/proofs/dr3-substrate-evidence.md` (formal evidence)

**Commit** : `feat(eval): conformance matrix runner` (38 chars)

## C2.11 — Cross-substrate H1-H4 statistical validation

**Goal** : re-run the H1-H4 hypotheses on E-SNN substrate. Verify consistency of profile chain effects across MLX and E-SNN.

**Files** :
- Modify : `scripts/ablation_cycle2.py` (add cross-substrate stats)
- Create : `docs/milestones/cross-substrate-results.md`

**Commit** : `docs(milestone): cross-substrate H1-H4 results` (45 chars)

## C2.12 — Paper 1 update with cycle-2 substrate evidence

**Goal** : update Paper 1 §7.x or §9 with cycle-2 evidence (E-SNN replication strengthens DR-3 claim). May require Paper 1 v2 arXiv update.

**Files** :
- Modify : `docs/papers/paper1/results-section.md` OR `discussion.md` (add cycle-2 section)
- Update : `docs/milestones/arxiv-submit-log.md` with v2 plan

**Commit** : `docs(paper1): add cycle-2 substrate evidence` (44 chars)

---

# Phase 4 — Paper 2 narrative (C2.13-C2.16)

## C2.13 — Paper 2 abstract + introduction

**Goal** : draft Paper 2 abstract (200 words) + introduction (~1 page) per cycle-1 outline (S27.1).

**Files** :
- Create : `docs/papers/paper2/abstract.md`
- Create : `docs/papers/paper2/introduction.md`

**Commit** : `docs(paper2): abstract + intro drafts` (37 chars)

## C2.14 — Paper 2 architecture + Conformance sections

**Goal** : draft §4 Conformance Criterion in Practice + §5 Engineering Architecture.

**Files** :
- Create : `docs/papers/paper2/conformance-section.md`
- Create : `docs/papers/paper2/architecture.md`

**Commit** : `docs(paper2): conformance + architecture` (40 chars)

## C2.15 — Paper 2 methodology + results

**Goal** : draft §6 Methodology + §7 Results (cross-substrate ablation data from C2.11).

**Files** :
- Create : `docs/papers/paper2/methodology.md`
- Create : `docs/papers/paper2/results.md`

**Commit** : `docs(paper2): methodology + results drafts` (44 chars)

## C2.16 — Paper 2 discussion + full-draft assembly

**Goal** : draft §8 Discussion + §9 Future Work + assemble full draft + LaTeX render.

**Files** :
- Create : `docs/papers/paper2/discussion.md`
- Create : `docs/papers/paper2/future-work.md`
- Create : `docs/papers/paper2/full-draft.md`
- Create : `docs/papers/paper2/build/full-draft.tex` (pandoc render)

**Commit** : `docs(paper2): assemble full draft v0.1` (40 chars)

---

# Phase 5 — Cycle 2 closeout (C2.17-C2.18)

## C2.17 — Real concurrent dream worker

**Goal** : replace the cycle-1 sync skeleton in `concurrent.py` with real asyncio/threading dream worker. Foundation for production deployment.

**Files** :
- Modify : `kiki_oniric/dream/operations/concurrent.py`
- Create : `tests/unit/test_concurrent_async.py`

**Pattern** : TDD 4 tests (real concurrent execution, ordering preserved, exception aggregation, cancellation safe).

**Commit** : `feat(dream): real async dream worker` (37 chars)

## C2.18 — G9 cycle-2 publication-ready gate + cycle 3 amorçage

**Goal** : G9 gate report (Paper 2 ready for arXiv + NeurIPS submission). Optional Paper 3 outline if cycle-3 already planned.

**Files** :
- Create : `docs/milestones/g9-cycle2-publication.md`
- Optional : `docs/papers/paper3/outline.md`

**Commit** : `docs(milestone): G9 cycle-2 publication ready` (45 chars)

---

# Self-review

**1. Spec coverage** :
- E-SNN substrate (Phase 1) → C2.1-C2.4 ✅
- P_max real wiring (Phase 2) → C2.5-C2.8 ✅
- Cross-substrate validation (Phase 3) → C2.9-C2.12 ✅
- Paper 2 narrative (Phase 4) → C2.13-C2.16 ✅
- Cycle 2 closeout (Phase 5) → C2.17-C2.18 ✅

**2. Placeholder scan** : aucun TBD non-intentionnel. Pattern abrégé délibéré (cycle-1 a éprouvé le format sur 6 plans).

**3. Type consistency** :
- `substrates/__init__.py` exports `MlxKikiOnericSubstrate`, `EsnnThalamocorticalSubstrate`
- `Conformance` test suite generic, parametrized by substrate
- `AblationRunner` extended with `substrate` parameter (backward-compat: default = MLX)
- P_max metadata fields (target_ops, target_channels_out from cycle-1) used as wiring contract

**4. Commit count** : 18 commits.

**5. Validator risks** : tous subjects pré-vérifiés ≤50 chars.

**6. Critical-path dependencies** :
- Loihi-2 access (Intel NRC) for native E-SNN ; fallback Norse simulation
- Real mega-v2 for non-synthetic ablation ; cycle-1 closeout may already deliver
- Paper 1 acceptance for cross-citation in Paper 2

---

**End of cycle-2 atomic plan.**

**Version** : v0.1.0
**Generated** : 2026-04-18 via cycle-2 amorçage
**Source** : cycle-1 G6 decision report (`docs/milestones/g6-cycle2-decision.md`)
