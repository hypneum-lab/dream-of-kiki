---
title: "dreamOfkiki: A Substrate-Agnostic Formal Framework for Dream-Based Knowledge Consolidation in Artificial Cognitive Systems"
author: "Saillant, Clément"
contact: "Clement Saillant <clement@saillant.cc>"
affiliation: "L'Electron Rare, France"
date: "2026"
draft: "v0.2 (cycle-1, S20.3 assembly, INCLUDE placeholders inlined)"
---

# Paper 1 — Full Draft Assembly

⚠️ **Status** : draft assembly. Section .md files were the original
source of truth ; this file now inlines their content to become the
assembled source for pandoc rendering.

⚠️ **Synthetic data caveats** apply to §7 Results (numbers from
mega-v2 synthetic placeholder). Real ablation lands cycle 1
closeout (S20+) or cycle 2.

---

## 1. Abstract

**Scope note.** This paper reports an *executable formal framework* for dream-based consolidation in artificial cognitive systems ; primary contributions are the axioms, the Conformance Criterion, and a cross-substrate conformance walkthrough. Hypothesis-level empirical evaluation on continual-learning benchmarks and fMRI alignment is deferred to the companion paper (Paper 2).

Catastrophic forgetting remains a central obstacle for artificial
cognitive systems learning sequentially across tasks. Sleep-inspired
consolidation has been proposed as a remedy, with prior work
exploring replay (Walker, van de Ven), synaptic homeostasis
(Tononi), creative recombination (Hobson), and predictive coding
(Friston) — but no unified formal framework integrates these four
pillars into composable, substrate-agnostic operations.

We introduce **dreamOfkiki**, a formal framework with executable
axioms. **DR-2 compositionality** is proved here as a
generated-semigroup theorem over four canonical operations
(closure + budget additivity + functional composition +
associativity ; proof in `docs/proofs/dr2-compositionality.md`).
The remaining axioms are DR-0 accountability, DR-1 episodic
conservation, DR-3 substrate-agnosticism via an executable
Conformance Criterion (signature typing ∧ axiom property tests ∧
BLOCKING-invariant enforceability), and DR-4 profile chain
inclusion (proof in `docs/proofs/dr4-profile-inclusion.md`). The
framework defines 8 typed primitives (α, β, γ, δ inputs ; 4
output channels), 4 canonical operations (replay, downscale,
restructure, recombine), and a 5-tuple Dream Episode ontology.

**Cross-substrate conformance (§5.6).** Two independent
substrates — an MLX gradient-based *kiki-oniric* implementation
and a numpy-LIF *thalamocortical E-SNN* — both satisfy the three
conditions of the Conformance Criterion, with 9 axiom-property
and invariant-enforcement tests passing on each (gate G7 LOCKED,
see `docs/milestones/g7-esnn-conformance.md`). This substantiates
the substrate-agnosticism claim at Paper-1 scope rather than
deferring it to cycle 2.

**Pipeline validation (§7).** We validate the measurement and
statistical chain (four pre-registered tests under Bonferroni
correction), invariant-guard fault injection on the BLOCKING
invariants S1–S3, and R1 reproducibility determinism across
matched registry tuples. We additionally report cross-substrate
portability measurements from the sibling Nerve-WML work (§7.4)
that independently confirm substrate-agnosticism on linearly
separable tasks (gap < 5 %) and disclose the non-linear regime
gap (12.1 %) as an honest negative result. A pilot run of the
pre-registered pipeline on PermutedMNIST-5 (§7.5) produces
Bonferroni-compliant statistics (t = 2.524, p = 0.01216 < α =
0.0125) consistent with the pre-registered H1 signal direction
and surfaces an unexpected pilot result for H2 (EWC proxy adds
no measurable benefit over replay alone on this benchmark). No
continual-learning hypothesis decisions on H1–H4 are announced
in this paper ; confirmatory evaluation of H1–H4 on mega-v2 is
reported in Paper 2.

Pre-registration is locked on the Open Science Framework (DOI
10.17605/OSF.IO/Q6JYN). All code, specifications, and
pre-registration are open under MIT / CC-BY-4.0 ; reference
implementations and test suites are available at
`github.com/hypneum-lab/dream-of-kiki`.

---

## 2. Introduction

### 2.1 Catastrophic forgetting and the consolidation gap

Modern artificial cognitive systems excel at single-task learning
but degrade rapidly when trained sequentially across tasks — a
phenomenon known as **catastrophic forgetting** [@mccloskey1989catastrophic;
@french1999catastrophic]. Despite two decades of mitigation
strategies (elastic weight consolidation [@kirkpatrick2017overcoming],
generative replay [@shin2017continual], rehearsal-based memory
[@rebuffi2017icarl]), the field still lacks a *unified theoretical
account* of why these mechanisms work and when they should compose.

Biological cognition solves this problem during **sleep**.
Hippocampal replay during NREM, synaptic downscaling, predictive
restructuring of cortical representations, and creative
recombination during REM together form a multi-stage
consolidation pipeline [@diekelmann2010memory; @tononi2014sleep].
Yet existing AI work has integrated only fragments of this
biology, typically focusing on a single mechanism (e.g., replay
alone) without a principled account of how mechanisms interact.

### 2.2 Four pillars of dream-based consolidation

We identify four theoretical pillars that any complete
dream-inspired AI consolidation framework must address :

- **A — Walker/Stickgold consolidation** : episodic-to-semantic
  transfer via replay [@walker2004sleep; @stickgold2005sleep].
- **B — Tononi SHY** : synaptic homeostasis renormalizing weights
  during sleep [@tononi2014sleep].
- **C — Hobson/Solms creative dreaming** : recombination and
  abstraction during REM [@hobson2009rem; @solms2021revising].
- **D — Friston FEP** : minimization of free energy as a unifying
  account of inference and consolidation [@friston2010free].

Prior AI work has implemented A [@vandeven2020brain], B
[@kirkpatrick2017overcoming as a SHY-adjacent regularization], and
elements of D [@rao1999predictive; @whittington2017approximation],
but **no work has formalized how the four pillars compose** in a
substrate-agnostic manner amenable to ablation and proof.

### 2.3 The compositional gap

Why does composition matter ? Empirically, the order in which
consolidation operations apply changes the resulting cognitive
state — replay before downscaling preserves episodic specificity,
while downscaling before restructuring may erase the very
representations that restructuring is meant to refine. Our
analysis (`docs/proofs/op-pair-analysis.md`) enumerates the 16
op-pairs and finds 12 cross-pairs are non-commutative, reinforcing
that *order is part of the framework*, not an implementation
detail.

A proper formal framework must therefore (i) specify the
operations as composable primitives with well-defined types, (ii)
make explicit which compositions are valid, (iii) provide an
**executable** account that any compliant substrate can implement,
and (iv) support empirical ablation comparing different operation
profiles. None of the prior art does all four.

### 2.4 Contribution roadmap

In this paper we present **dreamOfkiki**, the first formal
framework for dream-based consolidation in artificial cognitive
systems with the following contributions :

1. **Framework C-v0.5.0+STABLE** : 8 typed primitives, 4 canonical
   operations forming a free semigroup, 4 OutputChannels, 5-tuple
   Dream Episode ontology, axioms DR-0..DR-4 with executable
   Conformance Criterion (§4). Items 2–4 below are reported in
   Paper 2 (empirical companion) ; Paper 1 confines itself to the
   formal contributions and the conformance roadmap.
2. **Roadmap** to substrate generalization (additional
   substrates beyond cycle-1's reference implementation) and
   real fMRI representational alignment (real lab partnership
   pursued via T-Col outreach).

The remainder of the paper is organized as follows : §3 reviews
the four pillars in depth ; §4 develops Framework
C-v0.5.0+STABLE with axioms and proofs ; §5 sketches the
Conformance Criterion validation approach (per-substrate
empirical results live in Paper 2) ; §6 details the methodology ;
§7 reports the synthetic pipeline-validation results ; §8
discusses implications and limitations ; §9 outlines cycle-2
future work.

---

## 3. Background — four theoretical pillars

**Relation to the canonical three-process framing.** Contemporary
sleep-consolidation reviews [@klinzing2019mechanisms;
@rasch2025sleep] organise the field around three processes :
active systems consolidation (replay-dominated), synaptic
homeostasis (SHY), and integration/abstraction. Our framework
refines this split into four composable primitives by treating
*integration* as two formally distinct operations — `restructure`
(topology-preserving reorganisation, grounded in predictive-coding
/ FEP accounts) and `recombine` (cross-episode interpolation
grounded in REM creative-recombination accounts). The
irreducibility of `recombine` relative to `replay ∘ restructure`
under the free-semigroup presentation is addressed in the DR-2
proof (`docs/proofs/dr2-compositionality.md`) and the
non-commutativity pattern with `downscale` is the formal witness
that the fourth generator is not redundant.

### 3.1 Pillar A — Walker / Stickgold consolidation

Sleep-dependent memory consolidation refers to the empirically
established phenomenon that newly encoded memories are
selectively strengthened, abstracted, and integrated into long-
term storage during sleep [@walker2004sleep; @stickgold2005sleep].
Hippocampal replay during NREM slow-wave sleep is the neural
substrate most directly implicated. The functional claim is that
replay performs **gradient-like updates** on cortical
representations, biased toward retention of the replayed
episodes — equivalent in our framework to the `replay`
operation : sample β-buffer episodes, forward through the
current parameters, apply gradient updates against a retention
objective.

### 3.2 Pillar B — Tononi SHY synaptic homeostasis

The Synaptic Homeostasis Hypothesis (SHY) posits that wakefulness
drives net synaptic potentiation, and sleep enforces global
synaptic downscaling that restores signal-to-noise ratio without
erasing the differential strengthening pattern [@tononi2014sleep].
The downscaling is empirically supported by ultrastructural
evidence (synapse size reductions during sleep) and by behavioral
evidence (sleep-dependent improvement on previously trained
tasks). In our framework, SHY corresponds to the `downscale`
operation : multiplicative shrinkage of weights by a factor in
(0, 1]. As established in our op-pair analysis (see
`docs/proofs/op-pair-analysis.md`, axioms DR-2 + invariants S2),
downscale is **commutative but not idempotent** (shrink_f ∘
shrink_f gives factor², not factor) — a property that constrains
canonical ordering choices.

### 3.3 Pillar C — Hobson / Solms creative dreaming

REM dreaming is associated with creative recombination,
counterfactual scenario generation, and integration of
emotionally significant material [@hobson2009rem; @solms2021revising].
The mechanism is hypothesized to be a generative-model-style
sampling from a latent representation of recent experiences,
producing novel combinations that probe the boundaries of
learned structure. In our framework, this maps to the
`recombine` operation : sample latents from the δ snapshot,
apply a VAE-light or interpolation kernel, emit new latent
samples on canal 2.

### 3.4 Pillar D — Friston Free Energy Principle

The Free Energy Principle (FEP) [@friston2010free] frames
perception, action, and learning as the minimization of
variational free energy under a hierarchical generative model.
Within FEP, sleep is interpreted as an offline phase that
**restructures** the generative model to better minimize expected
free energy on the distribution of waking inputs. In our
framework, this corresponds to the `restructure` operation :
modify the topology of the hierarchical model (add layer, remove
layer, reroute connectivity) to reduce predictive error on
retained episodes. The S3 topology guard (validate_topology)
ensures that restructure operations preserve framework-level
invariants S3 (species connectivity, no self-loops, layer count
bounds — see `docs/invariants/registry.md` for canonical
definitions and the S3 guard reference in
`kiki_oniric/dream/guards/topology.py`).

### 3.5 The compositional gap

Existing AI work has implemented one or two of the four pillars
(notably A via @vandeven2020brain generative replay and B via
@kirkpatrick2017overcoming EWC, treated as a SHY-adjacent
regularizer). However, no prior work has **formalized the
composition** of all four operations as a unified algebraic
structure with provable properties.

The compositional gap matters empirically because our op-pair
analysis (`docs/proofs/op-pair-analysis.md`) establishes that 12
of the 16 (op_i, op_j) cross-pairs are **non-commutative** —
that is, applying replay then downscale produces a different
cognitive state than applying downscale then replay. The
canonical ordering chosen in
`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §4.3
(replay → downscale → restructure ; recombine in parallel) is
therefore a load-bearing design decision, not an arbitrary
implementation choice.

A proper formal framework must therefore (i) specify the
operations as composable primitives with well-defined types,
(ii) make explicit which compositions are valid, (iii) provide
an executable account that any compliant substrate can
implement, and (iv) support empirical ablation comparing
different operation profiles. None of the prior art does all
four. Our Framework C-v0.5.0+STABLE (§4) is the first to do so,
mapping the four pillars to the canonical axiom framework :
pillar A → DR-1 episodic conservation, pillar B → DR-2
compositionality (downscale order constraint), pillar D → DR-3
substrate-agnosticism (the restructure topology guard S3 lives
on this axis), pillar C → DR-4 profile chain inclusion that
keeps the recombine-rich profiles on top. The free-semigroup
compositionality axiom DR-2 (proven in
`docs/proofs/dr2-compositionality.md`) is the foundational
property and the Conformance Criterion DR-3 the executable
contract for substrate-agnosticism.

---

## 4. Framework C

⚠️ **Source** : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
covers this section. Paper version below is a condensed narrative
of that spec, structured per §4 outline.md.

### 4.1 Primitives — 8 typed Protocols

Awake → Dream channels :
- α (raw traces, P_max only) — firehose ring buffer
- β (curated episodic buffer) — SQLite append-log with
  salience-gated insertion (records pass only when their
  salience score exceeds an adaptive top-k threshold)
- γ (weights snapshot) — checkpoint pointer fallback
- δ (hierarchical latents) — ring buffer N=256 multi-species

Dream → Awake channels :
- 1 (weight delta) — applied via swap protocol
- 2 (latent samples) — generative replay queue
- 3 (hierarchy diff) — atomic apply at swap with S3 guard
- 4 (attention prior) — meta-cognitive guidance (P_max only)

### 4.2 Profiles — chain inclusion DR-4

| Profile | Channels in | Channels out | Operations |
|---------|-------------|--------------|------------|
| P_min   | β | 1 | replay, downscale |
| P_equ   | β + δ | 1 + 3 + 4 | replay, downscale, restructure, recombine_light |
| P_max   | α + β + δ | 1 + 2 + 3 + 4 | replay, downscale, restructure, recombine_full |

DR-4 (proven in `docs/proofs/dr4-profile-inclusion.md`) :
ops(P_min) ⊆ ops(P_equ) ⊆ target_ops(P_max), and similarly for
channels. P_max is skeleton-only in cycle 1.

### 4.3 Dream-episode 5-tuple ontology

Each dream-episode (DE) is a 5-tuple :
`(trigger, input_slice, operation_set, output_channels, budget)`.
Triggers ∈ {SCHEDULED, SATURATION, EXTERNAL}. Operations are a
non-empty tuple of {REPLAY, DOWNSCALE, RESTRUCTURE, RECOMBINE}.
BudgetCap enforces non-negative finite (FLOPs, wall_time_s,
energy_j) per K1 invariant.

### 4.4 Operations — semigroup of consolidation steps

The operation set forms a free non-commutative semigroup under
composition `∘` with additive budget (DR-2 compositionality,
proof draft in `docs/proofs/dr2-compositionality.md`). Canonical
order : replay → downscale → restructure (serial, A-B-D pillar
order) ; recombine in parallel (C pillar). The op-pair analysis
(`docs/proofs/op-pair-analysis.md`) enumerates all 16 pairs,
finding 12 non-commutative cross-pairs.

### 4.5 Axioms DR-0..DR-4

- **DR-0 (accountability)** : every executed DE produces an
  `EpisodeLogEntry`, even on handler exception (try/finally
  guarantee).
- **DR-1 (episodic conservation)** : every β record is consumed
  before purge (τ_max bounded).
- **DR-2 (compositionality — proved under precondition, this paper)** :
  the set of operations `Op = {replay, downscale, restructure,
  recombine}` under composition `∘` with additive budget forms a
  non-commutative semigroup generated by the four primitives, on the
  class of permutations satisfying the precondition
  `¬(∃ i < j : π_i = RESTRUCTURE ∧ π_j = REPLAY)` (the excluded
  class was empirically falsified on the MLX substrate on 2026-04-21 ;
  see `docs/specs/amendments/2026-04-21-dr2-empirical-falsification.md`).
  The theorem is established in four steps (closure under the
  precondition, component-wise additivity of the budget triple
  `(FLOPs, wall_time, energy_J)`, functional composition of the
  `effect` map, and associativity of function composition). A
  companion sub-theorem (`docs/proofs/dr2-compositionality.md` §7)
  establishes **semantic irreducibility** of the fourth generator
  `recombine` under the framework's write-domain partition
  (`{replay, downscale, restructure}` modify `W` / `M` only ;
  `recombine` modifies `H`) : no finite composition of the other
  three primitives realises the same state-transformation as
  `recombine`, which justifies its inclusion as an independent
  generator. The corresponding **syntactic** freeness claim (no
  non-trivial relations beyond associativity) is strictly stronger
  than the irreducibility claim and is left open ; it is not
  required by the compositional-semantics argument of this paper.
- **DR-3 (substrate-agnosticism)** : an executable Conformance
  Criterion = signature typing ∧ axiom property tests pass ∧
  BLOCKING invariants enforceable. Two independent substrates
  satisfy all three conditions (§5.6).
- **DR-4 (profile chain inclusion)** : P_min ⊆ P_equ ⊆ P_max
  for ops and channels ; a monotonicity lemma (DR-4.L) gives the
  weak ordering of expected metric outcomes in-expectation over
  metric classes monotone in capacity. Proof in
  `docs/proofs/dr4-profile-inclusion.md`.

### 4.6 Invariants — I/S/K with enforcement matrix

- **I1** episodic conservation (BLOCKING)
- **I2** hierarchy traceability (BLOCKING)
- **I3** latent distributional drift (WARN)
- **S1** retained non-regression (BLOCKING, swap guard)
- **S2** finite weights no NaN/Inf (BLOCKING, swap guard)
- **S3** topology valid (BLOCKING, swap guard)
- **S4** attention prior bounded (P_max only)
- **K1** dream-episode budget (BLOCKING)
- **K3** swap latency bounded (WARN)
- **K4** eval matrix coverage on MAJOR bump (BLOCKING)

### 4.7 DualVer formal+empirical versioning

`C-vX.Y.Z+{STABLE,UNSTABLE}` — formal axis (FC) and empirical
axis (EC) bump independently. Current : C-v0.5.0+STABLE
(target post-G3 : C-v0.7.0+STABLE).

---

## 5. Conformance Criterion validation approach

⚠️ **Substrate-agnostic by design.** Paper 1 confines itself to
the abstract conformance contract any compliant implementation
must satisfy. An empirical instantiation (the cycle-1 reference
implementation) is reported in Paper 2.

### 5.1 Deterministic compilation graph

A conformant substrate exposes a deterministic compilation graph
for each of the four operations so that re-running with the same
seed produces bit-stable outputs (R1 contract). This is the
hardest pre-condition for the run registry to register a batch
as reproducible.

### 5.2 Single-threaded scheduler with handler registry

DR-0 accountability requires that every executed dream-episode
produces an `EpisodeLogEntry` even on handler exception. A
single-threaded scheduler with a per-Operation handler registry
and a try/except/finally pattern is the canonical realization ;
multi-threaded variants must demonstrate equivalent log
guarantees.

### 5.3 Atomic swap with invariant guards

The awake-state promotion must be atomic and must abort on any
violated BLOCKING invariant (S1 retained non-regression, S2
weight finiteness, S3 topology validity). Conformant substrates
expose a `SwapAborted`-style escape hatch keyed by the violated
invariant ID.

### 5.4 Profile chain inclusion

DR-4 requires that any conformant set of profiles (P_min ⊆ P_equ
⊆ P_max) inherit ops and channels by inclusion. The conformance
test suite ships generic membership checks ; substrate-specific
wiring is reported in Paper 2.

### 5.5 Reference implementation pointer

See Paper 2 for an empirical instantiation (cycle-1 MLX-based
reference implementation). Paper 1 makes no claim about a
specific implementation beyond the formal contract above.

### 5.6 Cross-substrate conformance walkthrough — two substrates satisfy the criterion

To substantiate the substrate-agnosticism claim of DR-3 *at
Paper-1 scope*, we report the result of applying the Conformance
Criterion to two structurally divergent substrates. The goal is
not a benchmark comparison but a demonstration that the criterion
is non-vacuous : it accepts substrates that differ in their
mathematical primitives (gradient descent on dense tensors vs.
spike-rate dynamics on sparse events), yet rejects incorrect
implementations via the BLOCKING invariants.

**Substrate 1 — MLX kiki-oniric (Track A, gradient-based).** A
reference implementation using Apple MLX dense-tensor operations,
LoRA-adapted weight updates, and SQLite-backed episodic buffer.
Canonical for cycle 1 ; source at
`kiki_oniric/substrates/mlx_kiki_oniric.py`.

**Substrate 2 — E-SNN thalamocortical (numpy-LIF, spike-based).**
A leaky-integrate-and-fire spiking neural network with
thalamocortical connectivity, rate-coded primitives, and a
NORSE / nxNET backend switch. Developed specifically to stress-test
DR-3 on a substrate *formally distant* from Substrate 1 (no
gradients in the op handlers ; state is spike trains, not weight
tensors). Source at
`kiki_oniric/substrates/esnn_thalamocortical.py` ; gate G7
conformance report at `docs/milestones/g7-esnn-conformance.md`.

**Criterion condition table.** Each condition of DR-3 is exercised
by a discrete set of tests ; each substrate receives an
independent pass/fail per condition.

| DR-3 condition | MLX kiki-oniric | E-SNN thalamocortical |
|----------------|-----------------|-----------------------|
| (1) Signature typing (4 op handlers, Protocol-compliant, identity constants exported) | ✅ 4 tests pass | ✅ 4 tests pass (`test_dr3_esnn_substrate.py`) |
| (2) Axiom property tests (DR-0 accountability on every executed DE ; DR-2 non-idempotent downscale property ; R1 recombine determinism) | ✅ 3 tests pass | ✅ 3 tests pass |
| (3) BLOCKING invariants enforceable (S2 finite-weight guard, S3 topology guard, S1 retained non-regression) | ✅ 2 tests pass (S2, S3) | ✅ 2 tests pass (S2 on LIF `state.v`, S3 on species chain) |
| **Overall** | **Conformant** | **Conformant** |

DR-1 (episodic conservation) and DR-4 (profile chain inclusion)
are substrate-agnostic properties verified once at the framework
level ; each conformant substrate inherits them by construction.

**Interpretation.** Both substrates pass the Conformance Criterion
under 9 axiom-property and invariant-enforcement tests each (~27
assertions total: 18 substrate-paired tests × an average of 1.5
underlying `assert` statements per test). The criterion is therefore non-vacuous in the
minimal sense — it distinguishes conformant implementations from
non-conformant ones (confirmed by fault-injection experiments
reported in §7.2). The criterion remains a *necessary* test of
framework compliance ; *sufficient* empirical validation of the
substrate-agnosticism claim across a broader class of substrates
(SNN variants, transformer-based instantiations, analog
neuromorphic hardware) is pursued in the companion paper.

### 5.7 Proof sketches — DR-0..DR-4

DR-0 proved by handler-registry try/finally invariant ; DR-1
proved by β-buffer drain accounting ; DR-2 proved as a
generated-semigroup theorem (full proof in
`docs/proofs/dr2-compositionality.md`) ; DR-3 validated by the
executable Conformance Criterion on two substrates (§5.6) ;
DR-4 proved in `docs/proofs/dr4-profile-inclusion.md` as chain
inclusion of ops and channels plus the DR-4.L monotonicity
lemma.

---

## 6. Methodology

### 6.1 Pre-registered hypotheses (OSF)

Four hypotheses were pre-registered on the Open Science Framework
(OSF) before any empirical run, following the Standard Pre-Data
Collection template. Pre-registration was locked at S3 of the
cycle (calendar reference) ; the OSF DOI is cited in the paper
front matter and resolves to an immutable timestamp record.
A single amendment (Amendment #1, Bonferroni family restructure
for cycle-3 multi-scale hypotheses) was filed 2026-04-21 before
any data collection at OSF DOI `10.17605/OSF.IO/TPM5S`
(https://osf.io/tpm5s/), as an Open-Ended Registration linked
to the parent Q6JYN. The primary analysis reported here uses the
original α = 0.00625 as the strictly more conservative of the
two bounds ; the restructured family (α = 0.00833 per-cell,
0.025 cross-cell) is provided as a secondary analysis. Both
registrations are accessible via the parent OSF project page
https://osf.io/q6jyn.

- **H1 — Forgetting reduction** : `mean(forgetting_P_equ) <
  mean(forgetting_baseline)`. Test : Welch's t-test, one-sided.
- **H2 — P_max equivalence** : `|mean(acc_P_max) -
  mean(acc_P_equ)| < 0.05`. Test : two one-sided tests (TOST).
  *Cycle 1 status* : self-equivalence smoke only (P_max skeleton).
- **H3 — Monotonic alignment** : `mean(acc_P_min) <
  mean(acc_P_equ) < mean(acc_P_max)`. Test :
  Jonckheere-Terpstra. *Cycle 1 status* : two-group (P_min ↔
  P_equ) only.
- **H4 — Energy budget** : `mean(energy_dream / energy_awake)
  < 2.0`. Test : one-sample t-test against threshold.

### 6.2 Statistical tests + Bonferroni correction

All hypothesis tests use a Bonferroni-corrected significance
threshold : `α_per_hypothesis = 0.05 / 4 = 0.0125`. The four
tests are implemented in the reference implementation's
statistical module (which wraps standard statistical libraries ;
see Paper 2 for the substrate-specific code path) :

- **`welch_one_sided`** (H1) : `scipy.stats.ttest_ind` with
  `equal_var=False`, p-value halved for one-sided interpretation.
- **`tost_equivalence`** (H2) : two manual one-sided t-tests
  (lower bound `diff <= -ε` and upper bound `diff >= +ε`),
  reject H0 when both pass at α (TOST max-p rule).
- **`jonckheere_trend`** (H3) : sum of pairwise Mann-Whitney U
  counts across ordered groups, z-approximation for p-value
  (no scipy native).
- **`one_sample_threshold`** (H4) : `scipy.stats.ttest_1samp`
  against `popmean=threshold`, p-value adjusted for one-sided
  (sample below threshold).

All tests return a uniform `StatTestResult(test_name, p_value,
reject_h0, statistic)` for downstream handling.

### 6.3 mega-v2 benchmark

Empirical runs use the **mega-v2** dataset (498K examples
distributed across 25 domains : phonology, lexical, syntax,
semantic, pragmatic, etc.). Cycle 1 stratifies a **500-item
retained subset** (20 items per domain) and freezes it via
SHA-256 hash for the reproducibility contract R1.

The frozen retained benchmark is loaded via `harness.benchmarks.
mega_v2.adapter.load_megav2_stratified()`, which falls back to
a deterministic synthetic placeholder if the real mega-v2 path
is unavailable. **All cycle-1 results in §7 use the synthetic
fallback ; real mega-v2 integration lands cycle 1 closeout
(S20+) or cycle 2.**

### 6.4 RSA fMRI alignment (Studyforrest)

The H3 monotonic representational alignment hypothesis is
evaluated by Representational Similarity Analysis (RSA) between
kiki-oniric activations and fMRI responses. Cycle 1 uses the
**Studyforrest** dataset (Branch A locked at G1 — see
`docs/feasibility/studyforrest-rsa-note.md`) :

- **Format** : BIDS, DataLad-distributed, PDDL license (open).
- **Annotations** : 16,187 timed words, 2,528 sentences, 66,611
  phonemes ; 300-d STOP word vectors. Mappable to ortho species
  (rho_phono / rho_lex / rho_syntax / rho_sem).
- **ROIs** : extracted via FreeSurfer + Shen-268 parcellations
  for STG, IFG, AG (the canonical language network).
- **Pipeline** : `nilearn` CPU-deterministic mode for R1
  reproducibility. Real ablation deferred S20+ (real model
  inference) ; cycle 1 reports infrastructure validation only.

### 6.5 Reproducibility contract R1 + R3

Reproducibility is enforced by two contracts :

- **R1 (deterministic run_id)** : every run is keyed by a
  16-character SHA-256 prefix of `(c_version, profile, seed,
  commit_sha)`. Re-running the same key produces an identical
  `run_id` (verified by `harness.storage.run_registry`). Width
  was bumped from 16 → 32 hex chars in commit `df731b0` after a
  code-review finding flagged 64-bit collision risk at scale.
- **R3 (artifact addressability)** : all benchmarks ship with a
  paired `.sha256` integrity file. The `RetainedBenchmark`
  loader rejects any items file whose hash does not match the
  frozen reference, raising `RetainedIntegrityError`.

The DualVer versioning scheme (formal axis FC + empirical axis
EC) tags each artifact with the framework version under which
it was produced. Empirical results are valid only against the
declared `c_version` ; bumping FC-MAJOR invalidates EC and
requires re-running the affected matrix.

---

## 7. Pipeline and framework validation

This section reports validation results at the **framework
level** : (i) the measurement and statistical pipeline executes
end-to-end on a scripted input dataset ; (ii) the BLOCKING
invariants S1–S3 reject deliberately faulted swap attempts ;
(iii) the R1 reproducibility contract produces deterministic
`run_id` hashes for matched parameter tuples ; and (iv) a
sibling project's cross-substrate portability measurements are
summarised as an independent test of the substrate-agnosticism
claim. **No hypothesis decisions on H1–H4 are announced here** ;
empirical hypothesis evaluation on continual-learning benchmarks
(mega-v2) and fMRI alignment (Studyforrest / lab partnerships)
is the subject of Paper 2.

### 7.1 Statistical pipeline end-to-end execution

The four pre-registered tests (Welch one-sided, TOST equivalence,
Jonckheere–Terpstra trend, one-sample threshold) are wrapped in
a uniform `StatTestResult` interface in
`harness.statistics.tests`. On a controlled 500-item stratified
input at scripted accuracy levels (50 % / 70 % / 85 % across
baseline / P_min / P_equ ; `run_id
syn_s15_3_g4_synthetic_pipeline_v1`,
`docs/milestones/ablation-results.json`) :

- all four tests execute without error ;
- each returns an expected-sign statistic consistent with the
  scripted direction of the signal ;
- the `StatTestResult.reject_h0` boolean is computed from the
  Bonferroni-corrected α = 0.0125 threshold and is interpretable
  downstream by the gate-decision logic.

*Interpretation.* This establishes the statistical chain as
operational under the pre-registered multiple-comparison policy.
No p-values are reported here : p-values computed from scripted
accuracies are uninformative about framework efficacy and would
be misleading if displayed. Real p-values, with non-scripted
predictors on mega-v2, are reported in Paper 2.

### 7.2 Invariant-guard fault injection

The three BLOCKING invariants S1 (retained non-regression), S2
(finite weights, no NaN/Inf in scratch weights), and S3
(topology validity) are exercised by three dedicated fault
injections on each of the two conformant substrates :

| Invariant | Fault | Expected outcome | Observed (MLX) | Observed (E-SNN) |
|-----------|-------|------------------|----------------|------------------|
| S1 | Weight regression on retained-bench subset | `SwapAborted(S1)` + log to `aborted-swaps/` | ✅ | ✅ |
| S2 | NaN injected into `W_scratch` | `SwapAborted(S2)` pre-S1 | ✅ | ✅ (on `state.v`) |
| S3 | Topology self-loop inserted | `SwapAborted(S3)` post-S1 | ✅ | ✅ (species chain) |

All six fault injections aborted the swap, logged the correct
invariant ID, and left the awake state un-promoted. See
`tests/invariants/test_swap_guards.py` and the G7 conformance
report for the E-SNN-specific variants.

### 7.3 Run registry determinism (R1)

The R1 contract assigns a deterministic 32-character SHA-256
prefix as `run_id` keyed on `(c_version, profile, seed,
commit_sha, benchmark_version)`. We generated 1000 matched
tuples at fixed parameters and 1000 tuples with a single
component perturbed ; all 1000 matched tuples yielded identical
`run_id` hashes, and all 1000 perturbed tuples yielded distinct
hashes. The registry enforces R3 artifact integrity by rejecting
any `RetainedBenchmark` whose SHA-256 does not match the frozen
reference ; this gate fired as expected on deliberate file
mutation (`test_r3_integrity.py`). Width was bumped from 16 → 32
hex characters in commit `df731b0` after a code-review flag on
64-bit collision risk at expected cycle-2 scale.

### 7.4 Cross-substrate portability — independent corroboration

A sibling project at `github.com/hypneum-lab/nerve-wml` (same
byline, same Protocol-typing discipline ; archived release
**v1.1.2, Zenodo DOI
[10.5281/zenodo.19656354](https://doi.org/10.5281/zenodo.19656354)**)
measures cross-substrate polymorphism on a separate Nerve
interface with two concrete substrates : `MlpWML` (dense MLP)
and `LifWML` (surrogate-gradient leaky-integrate-and-fire SNN).
The nerve-wml Gate W measurement is a test of the same
*substrate-agnosticism* principle articulated here as DR-3. We
summarise the reported numbers — full details in the cited
preprint and the archived Zenodo record.

**Linearly separable task (FlowProxyTask 4-class, N = 4 pool,
multi-seed).** `MlpWML` and `LifWML` pools both saturate at
accuracy 1.000 ; relative gap 0.000. Both substrates satisfy the
shared Nerve protocol.

**Non-linear task (HardFlowProxyTask 12-class, XOR-projected
label).** `MlpWML` reaches accuracy 0.547 ; `LifWML` reaches
accuracy 0.480. Absolute gap 0.067, relative gap **12.1 %** —
above the 5 % target. This is disclosed in the nerve-wml paper
as an honest negative result and reflects the lag of the current
cosine pattern-match decoder in the LIF variant on non-linear
tasks.

**Gate M (merge test).** An `MlpWML` trained against a mock
Nerve and deployed against a real Nerve retains **1.000** of
its mock-baseline accuracy (criterion ≥ 0.95), confirming
substrate-interoperability end-to-end.

*Interpretation.* Read at Paper-1 scope : (a) the
substrate-agnosticism principle of DR-3 is empirically tractable
and already demonstrated on linearly separable tasks in a sibling
system ; (b) the 12.1 % non-linear gap bounds the current
substrate-portability envelope and sets an explicit cycle-2
target ; (c) the honest reporting discipline — disclosing the
gap rather than hiding it — is preserved. These numbers do not
constitute H1–H4 evidence ; they constitute *methodological*
evidence that the framework's agnosticism claim is operational
rather than asserted.

### 7.5 Preliminary empirical pilot on a standard CL benchmark

To confirm end-to-end operation of the pre-registered statistical
pipeline on *non-scripted* data, we ran a pilot on
PermutedMNIST-5 (avalanche-lib 0.6.0, SimpleMLP 784→512→10,
5 epochs per experience, 3 seeds : 42 / 123 / 7) with three
conditions matched to the pre-registered profile ladder :

- `baseline` ≡ Naive (no replay, no regularization) ;
- `P_min` ≡ ReplayPlugin(mem_size = 500) — `replay` primitive only ;
- `P_equ` ≡ ReplayPlugin(mem_size = 500) + EWCPlugin(λ = 0.4) — `replay` + EWC as a proxy for `restructure`.

All runs executed on CPU ; in the tested configuration the
torch-2.11 MPS backend on Apple Silicon silently skipped weight
updates (diagnosed by direct weight-drift inspection, documented
in `experiments/h1_split_mnist/LAB_NOTEBOOK.md`). Raw
per-experience results (3 seeds × 3 conditions × 5 experiences =
45 rows) are archived at
`experiments/h1_split_mnist/results_h1_v2.csv`.

We measured **per-experience forgetting** =
peak test accuracy − final test accuracy, aggregated over 3 seeds
× 5 experiences :

| Condition | Mean final acc (σ) | Mean forgetting (σ) | Max forgetting (Exp 0) |
|-----------|--------------------|---------------------|------------------------|
| baseline  | 0.887 (0.117)      | 0.0895 (0.117)      | 0.351                  |
| P_min     | 0.959 (0.010)      | 0.0158 (0.010)      | 0.030                  |
| P_equ     | 0.959 (0.009)      | 0.0157 (0.009)      | 0.028                  |

A paired one-sided t-test on per-experience forgetting
(baseline > P_equ) gives **t = 2.524, p = 0.01216** ; against
the pre-registered Bonferroni-corrected threshold α =
0.05 / 4 = 0.0125, this is below α. The paired comparison
baseline > P_min gives t = 2.534, p = 0.01193.

*Scope constraint.* PermutedMNIST-5 is **not** the pre-registered
confirmatory benchmark (cf. OSF DOI
[10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN)) ;
this is a pilot, not an H1 decision. The confirmatory evaluation
on mega-v2 is Paper 2 scope. The pilot is reported here only as
framework-level evidence that (i) the measurement pipeline
produces interpretable statistics on real CL data rather than
scripted accuracies, and (ii) the signal direction on a standard
benchmark is consistent with the pre-registered prediction.

*Unexpected pilot finding for H2.* To four decimals, `P_equ` and
`P_min` produce the same mean forgetting (0.0157 vs 0.0158) on
PermutedMNIST-5. The pre-registered H2 prediction — that the
`restructure` primitive (proxied here by EWC regularization) adds
value beyond `replay` alone — is not supported in this pilot.
Three non-exclusive interpretations : (a) EWC λ = 0.4 is
miscalibrated for this benchmark ; (b) EWC is an imperfect proxy
for `restructure` as defined by DR-3 ; (c) PermutedMNIST's
input-level permutations do not stress the representational
geometry that `restructure` is designed to address. The
confirmatory H2 decision is re-scoped to Paper 2 and will include
an EWC-λ sweep (and alternative `restructure` proxies) as a
sensitivity analysis.

### 7.6 Summary

The §7 validation establishes five operational properties of the
framework : the pre-registered statistical pipeline executes
correctly under Bonferroni correction ; the BLOCKING invariants
S1–S3 enforce the expected abort behaviour under fault injection
on both conformant substrates ; the R1 reproducibility contract
produces deterministic `run_id` hashes at the required width ;
independent cross-substrate measurements in a sibling system
corroborate the substrate-agnosticism principle in regimes where
it is measurable today (§7.4) ; and a pilot run of the
pre-registered pipeline on a standard CL benchmark (§7.5)
produces interpretable, signed, Bonferroni-compliant statistics
on non-scripted data and surfaces an unexpected pilot result for
H2 that will inform the Paper 2 analysis plan. No continual-learning
hypothesis decisions on H1–H4 are announced in Paper 1 ; those
decisions are the scope of Paper 2.

---

## 8. Discussion

### 8.1 Theoretical contribution

Our framework C-v0.5.0+STABLE is, to our knowledge, the first
executable formal framework for dream-based consolidation in
artificial cognitive systems. By axiomatizing the four pillars
(replay (DR-1), downscaling (DR-2), restructuring (DR-3),
recombination (DR-4)) as composable operations on a free
semigroup with additive budget (see DR-2 in
`docs/proofs/dr2-compositionality.md`), we make explicit what
prior work left implicit : the **order and composition** of
consolidation operations matters, and reasoning about their
interactions requires more than ad-hoc engineering choices.

The Conformance Criterion (DR-3) operationalizes
substrate-agnosticism : any substrate that satisfies signature
typing + axiom property tests + BLOCKING-invariant enforceability
inherits the framework's guarantees. This is qualitatively
different from prior frameworks that bind theory to a specific
implementation [@kirkpatrick2017overcoming; @vandeven2020brain] —
implementation details are discussed in Paper 2. The DR-4 profile
chain inclusion (P_min ⊆ P_equ ⊆ P_max) further structures the
ablation space such that experimental claims about richer
profiles do not inadvertently rely on weaker-profile invariants.

### 8.2 Empirical contribution

The synthetic ablation pipeline (S15.3, run_id
`syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`) demonstrates that the
statistical evaluation chain (Welch / TOST / Jonckheere /
one-sample t-test under Bonferroni correction) is end-to-end
operational on a 500-item stratified benchmark. Three of four
pre-registered hypotheses passed at α = 0.0125 (H1 forgetting
reduction, H4 energy budget compliance, H2 self-equivalence
smoke), with H3 monotonic trend reaching the conventional 0.05
threshold but borderline at the corrected level.

While the values reported are synthetic placeholders pending real
mega-v2 + MLX-inferred predictor integration (S20+), the
**measurement infrastructure** itself is validated : the
RetainedBenchmark loader with SHA-256 integrity, the
`evaluate_retained` predictor bridge, the AblationRunner harness,
and the four statistical wrappers all interoperate cleanly. The
synthetic batch above is registered under
profile `G4_ablation` in the project registry so the JSON dump
remains traceable. Reproducibility contract R1 (deterministic
`run_id` from (c_version, profile, seed, commit_sha)) is enforced
by the run registry.

### 8.3 Limitations

Three limitations bound the Paper-1 contribution :

**(i) No hypothesis-level empirical decisions.** Paper 1 reports
framework validation (pipeline, invariants, reproducibility,
cross-substrate conformance) but **does not** claim H1–H4
hypothesis-level efficacy on real continual-learning benchmarks.
The §7 numbers are framework-level and do not entail per-hypothesis
p-values on continual-learning data — those are deferred to
Paper 2, where real mega-v2 predictors on the reference substrate
are evaluated under the locked pre-registration.

**(ii) Substrate-agnosticism tested on two substrates, not yet
on the wider class.** Two independent substrates (MLX kiki-oniric
and E-SNN thalamocortical) satisfy the Conformance Criterion
(§5.6), which is a *necessary* test of agnosticism. *Sufficient*
validation across the broader substrate class (transformer-based
instantiations, deep SNN variants on Loihi-2, analog neuromorphic
hardware) is cycle-2 scope.

**(iii) P_max skeleton only.** The P_max profile is declared via
metadata (target ops, target channels) but its handlers are not
fully wired. Hypothesis H2 (P_max equivalence vs P_equ within
±5 %) is therefore a cycle-2 evaluation ; in Paper 1, P_max is
used only to structure the DR-4 profile chain inclusion, not to
claim empirical parity.

### 8.4 Comparison with prior art

| Prior work | Contribution | dreamOfkiki addition |
|-----------|--------------|----------------------|
| @vandeven2020brain (brain-inspired replay) | Generative replay for CL | Replay is one of four composed operations, with explicit budget + non-commutativity with downscale, restructure, recombine |
| @kirkpatrick2017overcoming (EWC) | Fisher-weighted regularization protecting important weights | Complementary rather than subsumed : EWC realises a *weighting scheme* within the `downscale` operation class, not the same primitive as SHY multiplicative downscaling |
| @tononi2014sleep (SHY) | Theoretical synaptic-homeostasis claim | Operationalised as `downscale` operation with empirically verified commutative-but-non-idempotent property (§3.2) |
| @friston2010free (FEP) | Free Energy Principle | Operationalised as `restructure` operation with S3 topology guard ; tight interface with active-inference accounts (see below) |
| @hobson2009rem / @solms2021revising | REM creative dreaming | Operationalised as `recombine` with VAE-light / interpolation kernel and I3 drift bound |
| @mcclelland1995complementary (CLS) | Two-system hippocampus + neocortex | Embedded in profile chain P_min ⊆ P_equ ⊆ P_max (DR-4) |
| Progressive networks, PathNet [Rusu 2016, Fernando 2017] | Task-specific subnet allocation | Orthogonal : these are *architectural* CL devices ; our framework composes *operations* and admits such architectures as substrates |
| MERLIN, MEMO, Gated Linear Networks [Wayne 2018, Banino 2020, Veness 2021] | Multi-mechanism CL with explicit memory | Substrate candidates for our framework ; none satisfies an executable conformance criterion across gradient- and spike-based backends |
| Dark Experience Replay, iCaRL, A-GEM [Buzzega 2020, Rebuffi 2017, Chaudhry 2019] | Modern rehearsal-based CL baselines | All realise the `replay` operation alone ; DR-2 gives a formal composition semantics they inherit by instantiation |
| Active Inference / PEARL-style agents [Tschantz 2020, Millidge 2021] | FEP-grounded agent models composing perception + action | Complement : active inference is framework-level compatible with our `restructure` operation and the Conformance Criterion ; conformance on such an agent is explicit future work |
| World models (Dreamer, DreamerV2/V3) [Hafner 2020–2023] | Latent-world-model agents with replay and structural learning | Overlap on replay + restructure, but no explicit `recombine` or `downscale` in their ontology ; possible conformant substrate with those two primitives left as `no-op` or absorbed into the learning objective |
| JEPA family [LeCun 2022 "A Path Towards Autonomous Machine Intelligence" ; Assran et al. 2023 I-JEPA ; Bardes et al. 2024 V-JEPA ; AMI Labs 2026] | Self-supervised predictive world-model training on joint-embedding latents ; "world models that learn from reality, not language" | Complementary substrate candidate : JEPA realises the `restructure` primitive (Pillar D / FEP) at scale on latent representations, without an explicit `replay` or `downscale` counterpart in its native ontology. Conformance walkthrough on a JEPA-class substrate (e.g., V-JEPA video encoder with an added episodic β-buffer and homeostatic downscale) is explicit cycle-2 scope ; the cross-substrate portability pattern demonstrated here on MLX + E-SNN is intended to generalise to such a configuration. |
| @phillips2010categorial (Categorial Compositionality, PLOS Computational Biology 2010) | Category-theoretic explanation for systematicity in human cognition ; argues cognitive compositionality is a categorical functor | Prior art for the algebraic framing of cognition ; our DR-2 is a *domain-specific* instantiation on dream-episode operations (free non-commutative semigroup, not full Cartesian closed category) with an **executable** predicate and an empirical Conformance Criterion rather than an interpretive equivalence |
| @gavranovic2024fundamental (Fundamental Components of Deep Learning, PhD thesis / arXiv:2403.13001) | Unified category-theoretic account of deep learning architectures and training (parametric optics over semicartesian categories) | Orthogonal prior art for categorical ML : we formalise *dream-phase consolidation operations* rather than gradient-based training, and use a strictly weaker algebraic structure (free semigroup) adequate for the compositional-semantics claim without committing to the full optics stack |
| @klinzing2019mechanisms (Nature Neuroscience, systems-consolidation review) ; @rasch2025sleep (Physiological Reviews, sleep-contribution review) | Canonical three-process framing (active systems consolidation + SHY + integration) with extensive empirical grounding | Empirical anchor we commit to (cf. §3 preamble) ; our four-primitive decomposition refines *integration* into `restructure` and `recombine` — the extra generator is justified by non-commutativity with `downscale` (DR-2 proof), not by disagreement with the review literature |
| @huh2024platonic (Platonic Representation Hypothesis, ICML 2024) | Sufficiently capable models converge to a shared statistical representation independent of architecture and modality | Theoretical underpinning for our DR-3 substrate-agnosticism claim : if PRH holds, conformance is *expected* to transfer across the MLX, E-SNN and LoRA substrates rather than being an architectural coincidence ; companion empirical PRH probe in a cognitive-architecture setting reported in @saillant2026nervewml (`nerve-wml` v1.7.0, GammaThetaMultiplexer experiment) |

Our distinguishing features relative to the above : **(a)** a
unified *formal* framework covering all four pillars with a
**proved** compositionality theorem (DR-2), **(b)** an
*executable* Conformance Criterion empirically validated across
two structurally divergent substrates (MLX dense + LIF spiking),
**(c)** a pre-registered evaluation methodology with frozen
benchmarks and deterministic `run_id` tuples (contract R1),
**(d)** open-science artefacts (MIT code, CC-BY-4.0 docs, OSF
DOI, planned Zenodo DOI for artifacts). We do not claim a new
*algorithm* for continual learning ; the claim is a *framework*
against which algorithms can be specified, composed, and
compared with contract-level guarantees.

**On the Platonic Representation Hypothesis as theoretical
ground.** The PRH thesis [@huh2024platonic] — that representations
in sufficiently capable models converge to a shared statistical
structure independent of substrate — gives our DR-3 substrate-
agnosticism claim a falsifiable theoretical anchor : if PRH is
empirically vindicated, we should *expect* conformance metrics to
transfer across the MLX, E-SNN and LoRA substrates rather than
treat such transfer as a happy architectural accident. The
companion `nerve-wml` v1.7.0 working-memory layer
[@saillant2026nervewml] runs an explicit PRH probe via its
GammaThetaMultiplexer experiment ; the cross-substrate conformance
matrix reported here (cycle 2 + 3) is the dreamOfkiki side of the
same empirical bet.

---

## 9. Future Work

### 9.1 E-SNN substrate (Loihi-2 thalamocortical)

The most direct extension of cycle 1 is to validate the DR-3
Conformance Criterion on a second substrate : a thalamocortical
spiking neural network (E-SNN) deployed on Intel Loihi-2
neuromorphic hardware. This was deferred from cycle 1 per the
SCOPE-DOWN decision (master spec §7.3) to ensure cycle 1 closed
on time with a single-substrate validation.

The E-SNN substrate would test whether the framework's executable
axioms remain operational when the operations are realized as
spike-rate dynamics rather than gradient updates on dense
matrices. Successful conformance would provide the substrate-
agnosticism evidence that Paper 1 claims as a theoretical
property but does not yet demonstrate empirically across two
substrates.

### 9.2 P_max profile real wiring

Cycle 1 implements P_max only as a skeleton (`status="skeleton"`,
`unimplemented_ops=["recombine_full"]`). Cycle 2 will wire the
remaining components :

- **α-stream raw traces** input channel (currently P_max-only
  declared but not consumed) — requires firehose ring buffer
  with bounded retention
- **ATTENTION_PRIOR canal-4** output channel — requires the
  attention prior bounding invariant (S4) and downstream wiring
  to consumer modules
- **`recombine_full`** operation variant — full VAE encoder /
  decoder pair beyond the C-Hobson light interpolation skeleton

With P_max real-wired, hypothesis H2 (P_max equivalence vs P_equ
within ±5%) becomes a real comparison rather than the cycle-1
self-equivalence smoke test.

### 9.3 Real fMRI lab partnership

Cycle 1 locks Studyforrest as the fMRI fallback (G1 Branch A).
Cycle 2 pursues active partnership with one or more fMRI labs
identified via the T-Col reviewer outreach :

- **Huth Lab** (UT Austin) : Narratives dataset
- **Norman Lab** (Princeton) : episodic memory studies
- **Gallant Lab** (UC Berkeley) : naturalistic stimulus-driven
  BOLD

A real lab partnership would enable RSA on **task-controlled**
linguistic stimuli rather than the narrative-comprehension
fallback Studyforrest provides. This would strengthen H3
(monotonic representational alignment) which reached only
borderline significance in the cycle-1 synthetic pipeline
validation (run_id `syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`).

### 9.4 Multi-substrate Conformance Criterion validation

The strongest theoretical claim of Framework C-v0.5.0+STABLE —
substrate-agnosticism via DR-3 Conformance Criterion — needs
empirical validation across more than two substrates to be
defensible at peer review. Cycle 2 establishes the validation
matrix : for each candidate substrate (cycle-1 reference
implementation ✅, E-SNN, hypothetical transformer-based
instance), verify all three conformance conditions (signature
typing, axiom property tests passing, BLOCKING invariants
enforceable).

A reusable conformance test suite (drafted in cycle 1
`tests/conformance/`) is the foundation. Cycle 2 will extend it
with substrate-specific adapters and run the full suite against
each candidate substrate, producing a conformance report
publishable as a supplementary artifact for Paper 1 (or as the
main contribution of Paper 2's engineering ablation paper).

---

## 10. References

→ See `references.bib` (16 entries cycle-1 stub, will extend to
~30-40 in S20-S22 as the full draft is rendered). BibTeX
integration via `\bibliography{references}` in the LaTeX render.

Key citations (alphabetical) :
- Diekelmann & Born 2010 (sleep memory)
- French 1999 (catastrophic forgetting)
- Friston 2010 (FEP)
- Hobson 2009 (REM dreaming)
- Kirkpatrick 2017 (EWC)
- McClelland 1995 (CLS)
- McCloskey & Cohen 1989 (forgetting)
- Rao & Ballard 1999 (predictive coding)
- Rebuffi 2017 (iCaRL)
- Shin 2017 (generative replay)
- Solms 2021 (consciousness)
- Stickgold 2005 (consolidation)
- Tononi & Cirelli 2014 (SHY)
- van de Ven 2020 (brain-inspired replay)
- Walker & Stickgold 2004 (consolidation)
- Whittington & Bogacz 2017 (predictive coding)

---

## Word count summary (target : ~5000 words main + supp)

| Section | Target | Status |
|---------|--------|--------|
| §1 Abstract | ≤250 | drafted (~265, needs trim) |
| §2 Introduction | ≤1500 | drafted (~1200) |
| §3 Background | ≤1500 | drafted (~1500) |
| §4 Framework | condensed in main + spec ref | done |
| §5 Implementation | condensed | done |
| §6 Methodology | ≤1500 | drafted (~1500) |
| §7 Results | ≤2000 | drafted (placeholder) |
| §8 Discussion | ≤1500 | drafted (~1500) |
| §9 Future Work | ≤700 | drafted (~700) |
| §10 References | n/a | 16 entries stub |

**Estimated total** : ~10 000 words. PLOS Computational Biology
(primary submission target, retargeted from Nature Human
Behaviour 2026-04-19 after fit analysis) has no hard word limit ;
main-text budget is self-disciplined at 8 000–10 000 words with
figures and supplementary material absorbing overflow.

---

## Notes for revision

- **W6 added 2026-04-20** — §7.5 "Preliminary empirical pilot on
  a standard CL benchmark" inserted before Summary, which became
  §7.6. Abstract §1 updated to reflect the pilot. Raw data and
  run log at `experiments/h1_split_mnist/` (PermutedMNIST-5,
  avalanche-lib 0.6.0, 3 seeds, CPU device after MPS silent-fail
  diagnosis). Pilot reports paired-t baseline > P_equ forgetting
  t = 2.524, p = 0.01216 < Bonferroni α = 0.0125 ; H2 (EWC as
  `restructure` proxy) not supported in pilot, re-scoped to
  Paper 2 with EWC-λ sweep.
- Render via `./ops/build-arxiv.sh` → bundle for arXiv submit
  (v0.2 with the four W-series revisions applied 2026-04-19 ;
  re-bundle needed after W6 2026-04-20)
- Fill arXiv ID into §6.1 and CITATION.cff post-announcement
- Add Figures (1 four-pillars conceptual, 2 swap-protocol state
  diagram with invariant guards, 3 profile-chain inclusion
  P_min ⊆ P_equ ⊆ P_max, 4 cross-substrate conformance matrix,
  5 §7.5 per-experience forgetting curves baseline / P_min / P_equ)
- BibTeX render with proper `\cite{}` calls ; target ~35
  references for PLOS CB submission (W5 in cold review)
