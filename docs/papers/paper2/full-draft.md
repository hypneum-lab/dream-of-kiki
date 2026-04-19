---
title: "dreamOfkiki Paper 2: Cross-substrate conformance of a dream-based knowledge consolidation framework (synthetic-substitute replication)"
author: "dreamOfkiki project contributors"
contact: "Clement Saillant <clement@saillant.cc>"
affiliation: "L'Electron Rare, France"
date: "2026"
draft: "v0.1 (cycle-2, C2.16 assembly)"
---

# Paper 2 — Full Draft Assembly

⚠️ **Status** : draft assembly. Section `.md` files remain the
sources of truth ; this file inlines their content as the
assembled source for pandoc rendering of
`build/full-draft.tex`.

⚠️ **Synthetic substitute — methodology / replication paper.**
Every empirical table, figure, and verdict reported below is
produced by a shared Python mock predictor wired across two
substrate registrations (MLX + E-SNN numpy LIF skeleton). No
Loihi-2 hardware and no fMRI cohort participate in any result.
Paper 2 documents the cross-substrate replication pipeline, not
a fresh empirical claim.

---

## 1. Abstract

Paper 1 introduced Framework C-v0.6.0+PARTIAL (axioms DR-0..DR-4,
8 typed primitives, 4 canonical operations, 3 profiles) and its
executable Conformance Criterion (DR-3). Paper 2 asks the
engineering question Paper 1 could not : *does a single compliant
substrate constitute evidence for substrate-agnosticism ?*

We answer by replicating the cycle-1 pre-registered H1-H4
pipeline across two registered substrates — MLX kiki-oniric on
Apple Silicon and a numpy LIF spike-rate E-SNN thalamocortical
skeleton — and by exhibiting a two-substrate DR-3 Conformance
matrix (3 conditions × 2 substrates, all PASS ; a third
`hypothetical_cycle3` row stays N/A). Under Bonferroni α =
0.0125, both substrates agree on 4 / 4 hypotheses (H1 reject,
H2 reject, H3 fail to reject, H4 reject) — *synthetic
substitute ; the agreement is trivial by shared-predictor
construction and validates the pipeline, not the framework's
empirical efficacy on real data*. All code, pre-registration,
and run_ids are MIT / CC-BY-4.0.

---

## 2. Introduction

### 2.1 From a single substrate to a Conformance claim

Paper 1 [dreamOfkiki, cycle 1] introduced Framework C — an
executable formal framework for dream-based knowledge
consolidation, with axioms DR-0..DR-4, a free semigroup of 4
canonical dream operations (replay, downscale, restructure,
recombine), 8 typed primitives (α, β, γ, δ inputs + 4 output
channels), and three profiles (P_min, P_equ, P_max) chained by
DR-4 inclusion. The strongest theoretical move in Paper 1 was
**DR-3** : a Conformance Criterion that specifies the three
conditions (signature typing, axiom property tests, BLOCKING
invariants enforceable) any candidate substrate must satisfy to
inherit the framework's guarantees.

DR-3 is the load-bearing bridge between the theory and the
engineering. Paper 1 proved it exists as a formal device, and
exercised it on a single substrate : MLX `kiki_oniric` on Apple
Silicon. One substrate, however, is an existence proof — not
substrate-agnosticism. A *second* substrate is where the claim
transitions from theoretical to defensible.

### 2.2 The cycle-2 goal : cross-substrate replication

Cycle 2 pursues exactly this second substrate and the
replication infrastructure around it. The engineering
contribution of Paper 2 is fourfold :

1. **A second substrate registration.** We wire
   `esnn_thalamocortical`, a numpy LIF spike-rate skeleton of a
   thalamocortical E-SNN, exposing the same 4 op factories and
   consuming the same 8 primitives as MLX `kiki_oniric`. The
   skeleton is explicitly *not* Loihi-2 hardware ; it is a
   structural second implementation of the framework's
   Protocols, which is what DR-3 requires.
2. **A DR-3 Conformance matrix.** We exhibit the substrate ×
   condition matrix (3 conditions × 2 real substrates + 1
   placeholder row) and back every cell with a concrete test
   artifact (`tests/conformance/axioms/` +
   `tests/conformance/invariants/`). The matrix lives at
   `docs/milestones/conformance-matrix.md`.
3. **A cross-substrate H1-H4 replication.** The cycle-1 pre-
   registered statistical chain (Welch / TOST / Jonckheere /
   one-sample t under Bonferroni α = 0.0125) is re-run *per
   substrate* by `scripts/ablation_cycle2.py`. Results at
   `docs/milestones/cross-substrate-results.md`.
4. **An engineering architecture fit for replication.** Paper
   2 documents the async dream worker (C2.17), the 3-profile
   pipeline, the swap protocol with S1 / S2 / S3 / S4 guards,
   the run registry with 32-hex R1 determinism contract, and
   the DualVer tag `C-v0.6.0+PARTIAL`.

### 2.3 Scope honesty : synthetic substitute throughout

Paper 2 is a **methodology / replication paper**, not a fresh
empirical claim paper. The two substrate rows in the cross-
substrate matrix share the same Python mock predictor. A real
substrate-specific predictor is a cycle-3 deliverable.
Consequently : all H1-H4 p-values in §7 are flagged
*(synthetic substitute)* ; the cross-substrate agreement
verdict is trivially YES by construction ; no biological,
neuromorphic, or hardware-performance claim is made. The
architectural half of DR-3 Conformance is what Paper 2 earns.

### 2.4 Differentiation from Paper 1 and roadmap

Paper 1 was theoretical : axioms + proofs + formal definition
of Conformance. Paper 2 is engineering : operational substrates
+ reusable test suite + replication runner + synthetic-
substitute evidence. §3 background situates Paper 2 relative to
Paper 1. §4 reports the Conformance Criterion evidence. §5 lays
out the engineering architecture. §6 details the methodology.
§7 reports the cross-substrate results. §8 discusses
limitations. §9 outlines cycle-3.

---

## 3. Background

### 3.1 Paper 1 in one paragraph

Paper 1 introduced Framework C : 8 typed primitives (α, β, γ, δ
inputs + 4 output channels), 4 canonical dream operations
(replay, downscale, restructure, recombine) forming a free
semigroup under DR-2 compositionality, a 5-tuple Dream Episode
ontology, and axioms DR-0 (accountability), DR-1 (episodic
conservation), DR-2 (compositionality), DR-3 (substrate-
agnosticism via the Conformance Criterion), DR-4 (profile chain
inclusion).

### 3.2 Four pillars, briefly

Paper 2 inherits Paper 1 §3's four-pillar mapping :
Walker / Stickgold replay [@walker2004sleep; @stickgold2005sleep]
as `replay` ; Tononi SHY [@tononi2014sleep] as `downscale`
(commutative, non-idempotent) ; Hobson / Solms creative
dreaming [@hobson2009rem; @solms2021revising] as `recombine` ;
Friston FEP [@friston2010free] as `restructure` under the S3
topology guard. Paper 2 does not re-argue these mappings.

### 3.3 DR-3 Conformance Criterion, briefly

DR-3 specifies three conditions (C1 signature typing via typed
Protocols, C2 axiom property tests pass, C3 BLOCKING invariants
enforceable) that a candidate substrate must satisfy. The
criterion is executable : it ships as a reusable pytest battery
in `tests/conformance/` parameterized on the substrate's state
type.

### 3.4 Prior multi-substrate replication work in continual learning

Cross-substrate replication of continual-learning mechanisms is
under-represented in prior art. [@vandeven2020brain] demonstrate
brain-inspired replay but on a single architecture.
[@kirkpatrick2017overcoming] introduce EWC but do not pursue
substrate-agnosticism. [@rebuffi2017icarl] and
[@shin2017continual] are architecture-specific. No prior work,
to our knowledge, ships a reusable conformance test suite
binding a formal framework to operational substrates across
qualitatively distinct hardware models (dense-tensor MLX vs
spiking LIF). This is Paper 2's distinguishing engineering
contribution.

### 3.5 Synthetic-substitute framing, briefly

Paper 2 is a **methodology / replication paper**, not a fresh
empirical claim paper. Readers who elide the synthetic-
substitute tag misread the paper's scope.

---

## 4. Conformance Criterion in Practice

### 4.1 The three DR-3 conditions, recapped

Framework C §6.2 defines the **Conformance Criterion** as three
independently checkable conditions : **C1** signature typing
(typed Protocols), **C2** axiom property tests pass, **C3**
BLOCKING invariants enforceable (S1..S4 guards). DR-3 is
existential : it does not claim every substrate is conformant ;
it claims every *conformant* substrate inherits the framework's
guarantees. Paper 1 exhibited one ; Paper 2 exhibits two + one
placeholder.

### 4.2 Substrate 1 — `mlx_kiki_oniric` (cycle 1 reference)

| Condition | Verdict | Evidence |
|-----------|---------|----------|
| C1 | PASS | `tests/conformance/axioms/test_dr3_substrate.py` |
| C2 | PASS | `tests/conformance/axioms/` |
| C3 | PASS | `tests/conformance/invariants/` |

The MLX row carries no synthetic-substitute flag at the
*conformance* level : the three conditions concern the
framework surface, not the data fed through it.

### 4.3 Substrate 2 — `esnn_thalamocortical` (cycle 2, synthetic substitute)

**(synthetic substitute — no Loihi-2 HW.)**

| Condition | Verdict | Evidence |
|-----------|---------|----------|
| C1 | PASS | 4 op factories callable + core registry shared with MLX *(synthetic substitute)* |
| C2 | PASS *(synthetic substitute — no Loihi-2 HW)* | DR-3 E-SNN suite passes on numpy LIF skeleton |
| C3 | PASS *(synthetic substitute — spike-rate numpy LIF)* | S2 + S3 guards enforceable on LIFState |

Evidence file :
`tests/conformance/axioms/test_dr3_esnn_substrate.py`. The
E-SNN row is the key new artifact of cycle 2. Passing C1..C3
on a structurally independent implementation (spike-rate
dynamics rather than gradient updates) is the architectural
evidence that typing + axiom + guard surfaces do not secretly
depend on MLX internals.

### 4.4 A third row : `hypothetical_cycle3` (placeholder)

| Condition | Verdict | Evidence |
|-----------|---------|----------|
| C1 | N/A | not yet implemented |
| C2 | N/A | not yet implemented |
| C3 | N/A | not yet implemented |

The row is explicitly marked `N/A` and must not be read as
passing or failing. Candidate cycle-3 substrates :
transformer-based, SpiNNaker [@furber2014spinnaker] / Norse, or
real Loihi-2 [@davies2018loihi].

### 4.5 The matrix as a reusable artifact

Running `uv run python scripts/conformance_matrix.py` re-derives
the matrix from the test suite and writes Markdown + JSON
dumps. The test suite is substrate-parameterized — the
operational promise of DR-3.

### 4.6 What Conformance does and does not certify

**Certifies** : substrate's typed surface is framework-
compatible ; state satisfies axiom property tests ; state can
be refused by BLOCKING guards. **Does NOT certify** : empirical
performance on real data ; hardware energy or latency ;
biological fidelity of E-SNN LIF parameters.

---

## 5. Engineering Architecture

### 5.1 Two substrates, same Protocols

`mlx_kiki_oniric` (Apple Silicon, dense tensors,
`kiki_oniric/substrates/mlx_kiki_oniric.py` [@mlx2023]) and
`esnn_thalamocortical` (numpy LIF skeleton,
`kiki_oniric/substrates/esnn_thalamocortical.py`, *synthetic
substitute*) expose the same 4 op factories and consume the
same 8 primitives.

### 5.2 Three profiles : P_min, P_equ, P_max

- **P_min** — replay only, canal-1. `kiki_oniric/profiles/
  p_min.py`. Cycle 1 wired.
- **P_equ** — `{replay, downscale, restructure}` on canals
  1 + 2 + 3. Cycle 1 wired.
- **P_max** — 4 ops on 4 channels, fully wired cycle 2 (G8).

DR-4 profile chain inclusion is enforced structurally and
verified by `tests/conformance/axioms/test_dr4_*`.

### 5.3 Four canonical operations on a free semigroup

`replay` (Walker / Stickgold), `downscale` (Tononi SHY,
commutative non-idempotent), `restructure` (Friston FEP under
S3 topology guard), `recombine` (Hobson / Solms, VAE-light →
full VAE in cycle 2). Of the 16 op-pairs, 12 are non-
commutative — ordering is load-bearing.

### 5.4 Eight primitives : 4 inputs + 4 output channels

Inputs α (raw traces, P_max only), β (episodic buffer), γ
(semantic snapshot), δ (latent snapshot). Outputs canal-1
UPDATED_WEIGHTS, canal-2 RECOMBINED_LATENTS, canal-3
RESTRUCTURED_TOPOLOGY, canal-4 ATTENTION_PRIOR (P_max ; S4
guard ≤ 1.5). Each output gated by the swap protocol : S1
retained non-regression, S2 finite, S3 topology, S4
attention_budget.

### 5.5 The async dream worker (C2.17)

Cycle 2 C2.17 (`018fd05`) lands the real `asyncio`-based
worker, superseding the cycle-1 Future-API skeleton.
Concurrency enables parallel per-cell dispatch at the
orchestration layer and prepares for cycle-3 divergent-
predictor replication.

### 5.6 Run registry + R1 determinism

R1 : every run is keyed by a 32-hex SHA-256 prefix of
`(c_version, profile, seed, commit_sha)`. Cycle-2 run_ids :
`cycle2_batch_id` = `3a94254190224ca82c70586e1f00d845`,
`ablation_runner_run_id` =
`45eccc12953e758440fca182244ddba2`, `harness_version` =
`C-v0.6.0+PARTIAL`.

### 5.7 DualVer lineage : `C-v0.6.0+PARTIAL`

Formal MINOR bump (E-SNN Conformance extension). Empirical
axis `+PARTIAL` — the cross-substrate rows share a predictor,
`+STABLE` requires cycle-3 divergent-predictor evidence.

---

## 6. Methodology

### 6.1 Pre-registered hypotheses (OSF, inherited)

H1 forgetting (Welch one-sided), H2 equivalence (TOST ε=0.05),
H3 monotonicity (Jonckheere-Terpstra), H4 energy budget
(one-sample t vs 2.0). OSF-locked in cycle 1 (DOI pending
[@osf]).

### 6.2 Statistical pipeline + Bonferroni α

All tests at `α_per_hypothesis = 0.05 / 4 = 0.0125`.
Implementation in `kiki_oniric.eval.statistics`
[@virtanen2020scipy] (unchanged cycle 2). All return
`StatTestResult(test_name, p_value, reject_h0, statistic)`.

### 6.3 Ablation matrix : 2 × 3 × 3 = 18 cells

Generated by `scripts/ablation_cycle2.py`, dumped at
`docs/milestones/cross-substrate-results.{md,json}`. Seeds
`[42, 123, 7]`.

### 6.4 Synthetic-substitute predictor (the critical caveat)

**(synthetic substitute — not empirical claim.)**  The two
substrate rows share the same Python mock predictor.
Consequences : p-values trivially identical in the shared-
predictor limit ; agreement verdict trivially YES ; the
*architectural* claim (pipeline executes on two registrations
without duplication) is validated ; the *empirical* claim is
not made. Divergent-predictor replication is cycle-3.

### 6.5 Reproducibility : R1 contract + benchmark integrity

R1 run_ids listed in §5.6. R3 (benchmark integrity) via
SHA-256 reference file ; cycle-2 uses synthetic fallback
`synthetic:c8a0712000b641...` pending real mega-v2. DualVer tag
`C-v0.6.0+PARTIAL` attached to every artifact.

### 6.6 What changed vs Paper 1 methodology

Same hypotheses, same stats module, same seeds, same
Bonferroni, same predictor class. **New** : substrate dimension
(2 rows), cross-substrate agreement verdict, conformance matrix
artifact. One-dimension-at-a-time discipline.

---

## 7. Results

> ⚠️ **Caveat (synthetic substitute — not empirical claim).**
> Every quantitative claim is produced by a shared Python mock
> predictor across two substrate registrations. The agreement
> across substrates is trivially YES by construction ; the
> values validate the cross-substrate replication pipeline, not
> empirical efficacy on real data.

### 7.1 Provenance (synthetic substitute — not empirical claim)

| field | value |
|-------|-------|
| harness_version | `C-v0.6.0+PARTIAL` |
| cycle2_batch_id | `3a94254190224ca82c70586e1f00d845` |
| ablation_runner_run_id | `45eccc12953e758440fca182244ddba2` |
| benchmark | mega-v2 stratified, 500 synthetic items |
| benchmark_hash | `synthetic:c8a0712000b641...` |
| seeds | `[42, 123, 7]` |
| substrates | `mlx_kiki_oniric`, `esnn_thalamocortical` |
| data_provenance | synthetic — mock predictors shared across substrate labels |

### 7.2 Cross-substrate H1-H4 comparative table (synthetic substitute)

**Table 7.2 — MLX vs E-SNN hypotheses at Bonferroni α = 0.0125
(synthetic substitute — not empirical claim).**

| hypothesis | test | MLX p | MLX verdict | E-SNN p | E-SNN verdict | agree |
|------------|------|-------|-------------|---------|---------------|-------|
| H1 forgetting | Welch one-sided | 0.0000 | reject H0 | 0.0000 | reject H0 | YES |
| H2 self-equivalence | TOST (ε=0.05) | 0.0000 | reject H0 | 0.0000 | reject H0 | YES |
| H3 monotonicity | Jonckheere-Terpstra | 0.0248 | fail to reject | 0.0248 | fail to reject | YES |
| H4 energy budget | one-sample t (upper) | 0.0101 | reject H0 | 0.0101 | reject H0 | YES |

MLX 3 / 4 significant, E-SNN 3 / 4 significant, full
consistency *(synthetic substitute — trivially YES by shared-
predictor construction)*. H3 fails at α = 0.0125 on both
substrates due to the mock predictor's constant dispersion —
a property of the mock, not of the framework.

### 7.3 Agreement matrix (synthetic substitute)

**Table 7.3 — Verdict agreement across substrates (synthetic
substitute).**

| hypothesis | verdicts equal ? | MLX reject | E-SNN reject |
|------------|------------------|------------|--------------|
| H1 forgetting | YES | true | true |
| H2 self-equivalence | YES | true | true |
| H3 monotonicity | YES | false | false |
| H4 energy budget | YES | true | true |

4 / 4 agreement. Expected outcome when the predictor is shared.

### 7.4 DR-3 Conformance matrix (cross-ref §4)

**Table 7.4 — 3 conditions × 2 real substrates + 1 placeholder.**

| substrate | C1 | C2 | C3 |
|-----------|----|----|----|
| `mlx_kiki_oniric` | PASS | PASS | PASS |
| `esnn_thalamocortical` | PASS *(synthetic substitute)* | PASS *(synthetic substitute — no Loihi-2 HW)* | PASS *(synthetic substitute — spike-rate numpy LIF)* |
| `hypothetical_cycle3` | N/A | N/A | N/A |

Source : `docs/milestones/conformance-matrix.md` (commit
`fd54df7`).

### 7.5 What §7 establishes and does not

Establishes : cross-substrate pipeline executes end-to-end ;
per-substrate stats emit identical verdicts on identical
inputs ; DR-3 matrix PASSes on two rows ; R1 holds. Does NOT
establish : biological / neuromorphic performance ; empirical
consolidation efficacy ; divergent-predictor replication ;
hardware energy or latency.

### 7.6 Gate summary (synthetic substitute)

H1 PASS *(synthetic)*, H2 PASS *(synthetic, self-equivalence
smoke)*, H3 FAIL at 0.0125 / PASS at 0.05 *(synthetic, mock-
dispersion limited)*, H4 PASS *(synthetic)*. Aggregate cycle-2
G9 verdict : **CONDITIONAL-GO / PARTIAL** per
`docs/milestones/g9-cycle2-publication.md`.

---

## 8. Discussion

### 8.1 What cross-substrate convergence means in Paper 2

Both substrates pass DR-3 (C1 + C2 + C3) and emit the same
4 / 4 hypothesis verdicts. The right reading : the framework's
Conformance Criterion is operational on two structurally
independent implementations of its 8 primitives. The wrong
reading : the framework is empirically substrate-agnostic.
The synthetic-substitute predictor is the distinction.

### 8.2 What synthetic-substitute implies for claims

We **do not claim** equivalent behaviour on real data, nor
neuromorphic hardware advantage. We **do claim** the
architectural half of DR-3 : two independent substrate
registrations pass the same test battery without code
duplication.

### 8.3 Limitations (honest enumeration)

**(i)** Shared predictor across substrates — the dominant
limitation. **(ii)** Only two real substrates + one placeholder
— DR-3 evidence is inductive. **(iii)** Synthetic mega-v2
benchmark. **(iv)** Paper 1 publication sequencing risk — if
Paper 1 acceptance is delayed > 6 months, Pivot B applies (§9.5).

### 8.4 The T-Col pivot to real fMRI data

T-Col outreach targets Huth Lab / Norman Lab / Gallant Lab for
task-controlled fMRI data, strengthening H3. Cycle-3
deliverable.

### 8.5 Comparison with Paper 1 discussion §8.5

Paper 1 §8.5's retro-active cross-substrate paragraph is
Paper 2's full narrative. No contradictions under the
synthetic-substitute framing.

### 8.6 Engineering trade-offs (MLX vs E-SNN)

Shared-predictor setup does not admit a head-to-head energy /
latency benchmark. Under a divergent predictor (cycle 3), MLX
optimises for sequential dense-tensor updates, E-SNN on
Loihi-2 [@davies2018loihi] would optimise for event-driven
sparse computation. Paper 2 makes the comparison *possible* ;
Paper 3 (if it emerges) makes the comparison.

---

## 9. Future Work — Cycle 3

### 9.1 Finish Phase 3 — divergent-predictor replication

Replace shared mock with substrate-specific inference : MLX
forward pass + LIFState spike-rate read-out. Unlocks
informative cross-substrate verdicts. Enables
`C-v0.7.0+STABLE`.

### 9.2 Real Loihi-2 hardware mapping

If Intel NRC partnership materialises [@davies2018loihi], port
numpy LIF skeleton to Loihi-2. Fallbacks : SpiNNaker via Norse
[@furber2014spinnaker], Lava SDK simulation, or status-quo
numpy skeleton.

### 9.3 Real fMRI cohort (T-Col pivot)

Lab partnership producing task-controlled fMRI data. May slip
to cycle 4 if IRB / scanning timelines extend.

### 9.4 Paper 3 emergence

Paper 3 becomes plausible only if 9.1 + (9.2 or 9.3) land with
strong data. Until then, provisional placeholder at
`docs/papers/paper3/outline.md`.

### 9.5 Phase 4 sequencing with Paper 1 acceptance

**Preprint-first** : arXiv-submit Paper 2 concurrently with
Paper 1 preprint. **Pivot B** : if Paper 1 delayed > 6 months,
re-self-contain Paper 2 §3 + §4 and submit NeurIPS / TMLR
standalone.

---

## 10. References

See `references.bib`. Bibliography rendered via pandoc
`--citeproc` or `--biblatex` at render time.

---

## Notes on the assembly

- All section source files remain individually editable in
  `docs/papers/paper2/{abstract,introduction,background,
  conformance-section,architecture,methodology,results,
  discussion,future-work}.md`. This full-draft inlines their
  content at C2.16 assembly ; future revisions should update
  both the section file and this assembly.
- Every table caption in §7 carries a `(synthetic substitute)`
  flag. Do not strip on length passes.
- FR mirror : `docs/papers/paper2-fr/full-draft.md`.
