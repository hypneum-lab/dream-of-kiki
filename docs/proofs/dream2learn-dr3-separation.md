# Dream2Learn — DR-3 separation argument (v0.1-draft)

**Version** : v0.1-draft (2026-05-20)
**Supersedes** : — (first issue)
**Amendment pointer** : none ; this is a separation argument, not a
weakening / strengthening of DR-3. The DR-3 statement (axioms
spec §6.2) is unchanged.
**Target venue** : Paper 2 §7.8 (ablation related-work) — see
`docs/papers/paper2/results.md` §7.8.
**Executable counterpart** : none. This is a **conceptual proof**
(formal argument on architectural primitives), not an axiom
mechanisation. No `tests/conformance/axioms/` companion ; the DR-3
executable counterpart remains `test_dr3_substrate.py` +
`test_dr3_esnn_substrate.py`, unaltered.
**Classification trigger** : memory note
`~/.claude/projects/-Users-electron/memory/project_hypneum_deepresearch_2026_05_19_classification.md`
(2026-05-20 c-alert decision table), entry
`Dream2Learn-arXiv-2026 → RECLASS to (a)`.

---

## 0. Executive summary

Dream2Learn (D2L), Calcagno et al. 2026 ([arXiv:2603.01935][d2l]),
is a continual-learning architecture in which a classifier
**conditions a frozen latent diffusion model through soft-prompt
optimisation** to generate novel "dreamed" classes that expand
its representation space. The 2026-05-19 deep-research synthesis
flagged D2L as a candidate c-alert against DR-3 (substrate-
agnosticism) on the suspicion that it might constitute, or
require, a framework-C primitive.

This document shows that D2L is **not** a framework-C primitive
and **not** a candidate framework-C substrate either. The
argument proceeds in two disjoint angles :

1. **Structural** — D2L is an *architecture*, not a *substrate*.
   It instantiates a specific generative mechanism (latent
   diffusion + learned soft prompt) that is orthogonal to the
   DR-3 Conformance Criterion (signature typing, axiom property
   tests, BLOCKING invariants enforceable).
2. **Behavioural** — D2L's consolidation dynamic (continuous
   denoising in a fixed latent geometry, no role partition
   between awake and dream, no DE-bounded accountability
   trace) does not instantiate framework-C's four-primitive
   dream-phase contract (α, β, γ, δ) nor the four channels
   (1 weight\_delta, 2 latent\_samples, 3 hierarchy\_chg,
   4 attention\_prior).

The conclusion is therefore **reclassification from (c-alert) to
(a) — citation-only ablation baseline** in Paper 2 §7.8 (mirrors
the §7.7 Wake-Sleep CL baseline pattern). DR-3 is *not* at risk ;
the axiom statement, the Conformance Criterion, and the
two-substrate evidence in `dr3-substrate-evidence.md` are
unchanged.

---

## 1. DR-3 — formal restatement

DR-3 (axioms spec §6.2, verbatim, AXIOMS.md L165-L196) is an
**operational criterion**, not a formal implication into the
other axioms :

```
∀ substrate S, if S satisfies the Conformance Criterion below,
then DR-0, DR-1, DR-2 (or DR-2') are empirically validated on S
(operational sense, see §6.2 "Operational statement" ; not a
formal implication — DR-2 itself remains an unproven working
axiom, §6.2).
```

**Conformance Criterion** (three conditions, all required) :

1. **Signature typing** — S implements the typed Protocol
   signatures of the eight framework-C primitives (awake → dream
   α, β, γ, δ ; dream → awake channels 1, 2, 3, 4) as defined in
   spec §2.1.
2. **Axiom property tests** — the DR-0, DR-1, DR-2 (or DR-2')
   property suite passes on S with ≥100 % coverage on BLOCKING
   cases (`tests/conformance/axioms/`).
3. **BLOCKING invariants enforceable** — invariants S1, S2, S3,
   I1 are implemented as runtime checks on S with
   abort-on-violation logging.

`conforms(S) ≜ typed(S) ∧ axiom_tests_pass(S) ∧ invariants_enforced(S, {S1, S2, S3, I1})`.

A substrate that satisfies `conforms(S)` provides empirical
evidence (validation, not proof) for the framework-C axioms on
that substrate ; a violation is a direct counter-example.

DR-3 is therefore neutral over the *implementation choice*
within a substrate (MLX tensors, spiking neurons, etc.) as long
as the three conditions above hold. The axiom protects the
framework from over-fitting to one implementation ; it does
**not** absorb every continual-learning architecture into a
single substrate-agnostic claim. An architecture that does not
expose the eight typed primitives is simply outside the
framework's scope — it is neither a substrate nor a violation.

---

## 2. Dream2Learn — architecture summary

Source : Calcagno et al. 2026, *Dream2Learn: Structured
Generative Dreaming for Continual Learning*, arXiv:2603.01935v1
([d2l][d2l]).

**Setting.** Class-incremental image classification on
Mini-ImageNet, FG-ImageNet, ImageNet-R. A classifier `f_θ`
learns a sequence of disjoint class subsets ; the goal is to
mitigate catastrophic forgetting without storing past samples.

**Core mechanism** (D2L's contribution to prior generative
replay) :

1. A **frozen latent diffusion model** `D` (pretrained off-task)
   is used as a fixed generative substrate.
2. The classifier `f_θ` drives **soft-prompt optimisation** : a
   continuous prompt embedding `p` is updated so that the
   diffusion samples conditioned on `p` are classified, by
   `f_θ` itself, into a **novel dreamed class** — i.e. a
   semantically distinct yet structurally coherent region of
   `D`'s latent space.
3. Samples drawn from `D(· | p*)` (with optimised `p*`) are
   then used as auxiliary training data for `f_θ` to *expand*
   its representation space ahead of the next real-task
   exposure. The dreamed samples are **not** intended to
   reconstruct past observations (contra classical generative
   replay) ; they prime future learning dynamics.

**Training contract.** `D` is frozen ; only `p` and `f_θ` move.
There is no separate dream-phase update rule applied to the
*generative* substrate ; the generative substrate is a black-box
sampler conditioned by a learned prompt.

**Inference contract.** At test time, `f_θ` predicts on real
inputs ; the diffusion model is not used.

[d2l]: https://arxiv.org/abs/2603.01935

---

## 3. Separation argument

We show that D2L does not instantiate framework-C's structural
predicate (§3.1) and does not exhibit framework-C's behavioural
predicate (§3.2). The two angles are independent : either is
sufficient to place D2L outside the DR-3 scope.

### 3.1 Structural — Conformance Criterion not exposed

D2L's externally observable primitives are :

- a **classifier-update** map `f_θ ↦ f_θ'` (gradient descent on
  combined real + dreamed loss),
- a **soft-prompt** map `(f_θ, D) ↦ p*` (inner optimisation
  loop),
- a **sample-draw** map `(D, p*) ↦ X_dream` (frozen diffusion
  forward pass).

We map these against the framework-C primitive registry (spec
§2.1, glossary.md L21-L34) :

| framework-C primitive | type signature (spec §2.1) | D2L correspondent |
|-----------------------|----------------------------|-------------------|
| α (raw traces, P\_max only) | `Iterable[Trace] → AlphaBuffer` | absent — D2L has no trace firehose ; `f_θ` does not emit a `Trace` stream during awake |
| β (curated episodic buffer) | `AlphaBuffer → EpisodicSlice` (or direct from awake) | absent — D2L explicitly *avoids* an episodic buffer ; dreamed samples replace one |
| γ (weights-only snapshot) | `Weights → GammaSnapshot` | partially present (`f_θ` checkpoint) but never consumed by a dream-phase operator |
| δ (hierarchical latent snapshots) | `LatentHierarchy → DeltaSnapshot` | absent — `D`'s latent geometry is fixed ; no hierarchy snapshot is exported |
| Canal 1 (weight\_delta) | `DreamPhase → WeightDelta` | absent as a typed channel ; D2L updates `f_θ` in-line on a combined batch, no isolated dream-phase delta to merge under invariant S1 |
| Canal 2 (latent\_samples) | `DreamPhase → LatentSamples` | partially present (`X_dream`) but emitted by a frozen black-box, not by a typed `DreamPhase` operator |
| Canal 3 (hierarchy\_chg) | `DreamPhase → HierarchyChange` | absent — `D` and `f_θ` carry no exported hierarchy |
| Canal 4 (attention\_prior) | `DreamPhase → AttentionPrior` | absent |

Out of eight primitives, **zero are exposed with the typed
contract** (Conformance Criterion condition 1). The two partial
matches (γ checkpoint, Canal-2-like sample emission) do not
satisfy the Protocol typing — they are byproducts of the
generative pipeline, not declared handlers wired into a registry
(`kiki_oniric/runtime/registry.py`).

Conformance condition 1 therefore fails. Conditions 2 and 3 are
inapplicable : the DR-0 / DR-1 / DR-2 property suite cannot be
evaluated on D2L because there is no `DreamEpisode` 5-tuple
`(trigger, input_slice, operation_set, output_delta, budget)`
to instrument, and no BLOCKING invariant set (S1, S2, S3, I1)
to enforce on a non-existent dream phase.

**Conclusion of §3.1** : D2L does not satisfy
`typed(D2L) ∧ axiom_tests_pass(D2L) ∧ invariants_enforced(D2L, {S1, S2, S3, I1})`,
not by negative evidence on a substrate that tried to conform,
but because D2L does not declare the eight-primitive interface
to begin with. D2L is **outside DR-3's universe of discourse**.
The axiom is not violated ; it is simply not addressed.

### 3.2 Behavioural — consolidation dynamics differ

A second, independent angle. Even granting (counterfactually) a
plausible "wrapping" of D2L's pipeline into framework-C typed
primitives, the *consolidation dynamic* it implements does not
match framework-C's dream-phase contract :

1. **No role partition between awake and dream worktrees.**
   Framework-C requires three weight copies (W\_awake, W\_dream,
   W\_scratch ; glossary.md L36-L41) with a swap protocol at
   the boundary of a DreamEpisode. D2L updates `f_θ` in-line on
   a single combined batch ; there is no frozen `W_dream`
   snapshot consumed by an operator that emits a typed
   `WeightDelta` for later swap. The DR-2 compositionality
   precondition (no
   `RESTRUCTURE` followed by `REPLAY` ; `dr2-compositionality.md`
   v0.2 §1) is therefore *not even definable* : D2L has no
   ordered sequence of dream-phase operators.
2. **No DE-bounded accountability trace.** DR-0 requires every
   `output_delta` to trace back to a DreamEpisode with a
   *finite budget* (`dr-0.md` v0.1-draft). D2L's soft-prompt
   inner loop is an unbounded gradient-descent on `p` with
   stop conditions defined by the classifier loss, not by a
   DE budget. There is no `(trigger, input_slice,
   operation_set, output_delta, budget)` 5-tuple to log to
   `harness/storage/run_registry.py`.
3. **Fixed latent geometry, no hierarchy primitive.** D2L
   operates entirely inside the frozen latent space of a
   pre-trained diffusion model. The δ primitive
   (hierarchical latent snapshots) requires a *mutable*
   hierarchy whose snapshots are emitted at episode close ;
   D2L's geometry is by design immutable. Channel 3
   (hierarchy\_chg) is structurally precluded.
4. **Continuous denoising vs discrete DreamEpisode unit.**
   The DE is the atomic dream unit (glossary.md L23-L24) ;
   framework-C reasons about DE counts, DE budgets, and
   DE-bounded invariants (I1 episodic conservation). D2L's
   diffusion reverse process is a continuous trajectory in
   latent space ; partitioning it into DE-equivalent units is
   possible only by an arbitrary outer wrapper that does not
   reflect the underlying mechanism.

These four divergences are mutually reinforcing : together they
mean that, even if D2L's pipeline were re-cast as a
framework-C substrate by an aggressive adapter layer, the
substrate's *operational behaviour* (continuous, unbounded,
single-trajectory denoising on a frozen geometry) would not
exhibit the discrete-DE / role-partition / accountability-trace
behaviour the axioms presuppose. The Conformance Criterion
would degrade from "fails condition 1" (§3.1) to "fails
conditions 2 and 3" (§3.2). The conclusion is unchanged.

---

## 4. Reclassification

Per the (a)/(b)/(b′)/(c-alert) taxonomy in the memory note :

- **(a) Citation-only** — no design impact, goes in `.bib`.
- **(b)** — design-input encapsulable inside `BioFieldWML.step()`.
- **(b′)** — design-input for the biophysical sub-theory
  (`docs/specs/2026-05-20-biophysical-stratification.md`).
- **(c-alert)** — architecture-level conflict with an axiom,
  requiring reformulation or drop.

§3 establishes that D2L is neither a framework-C primitive
(structural angle) nor a framework-C substrate (behavioural
angle). It is, however, a published continual-learning baseline
that *names* dreaming and is therefore worth citing as
related-work prior art alongside Wake-Sleep CL
([@alfarano2024wakesleep]). It matches the **(a)
citation-only** profile exactly :

- no DR-3 risk (§3) ;
- no integration into `BioFieldWML` (D2L's generative pipeline
  has no analog to the WML Protocol N-1..N-5 / W-1..W-4
  interface) ;
- no contribution to the biophysical sub-theory (D2L is not
  biophysically motivated ; latent diffusion is a deep-learning
  artefact) ;
- positive value as an ablation comparator in Paper 2 §7.8,
  matching the §7.7 Wake-Sleep CL pattern.

The memory note's c-alert decision table (2026-05-20 row
`Dream2Learn-arXiv-2026 → RECLASS to (a) — Paper 2 ablation
benchmark only`) is therefore **formally supported** by §1–§3
of this document.

---

## 5. Limits of this argument

The argument is conceptual, not statistical. Three classes of
future evidence could reopen the analysis :

1. **Architectural variant.** A future D2L variant that
   declares the eight framework-C primitives with typed
   handlers (in particular : a discrete-DE wrapper around
   the soft-prompt loop, a typed δ snapshot of the diffusion
   model's latent layer activations, and an explicit
   `W_dream` snapshot of `f_θ` at DE start) would re-open
   condition 1 of the Conformance Criterion. The argument
   in §3.1 would have to be revisited against the new
   surface API.
2. **Behavioural retrofit.** A bounded variant of D2L's
   soft-prompt loop with explicit (trigger, input\_slice,
   operation\_set, output\_delta, budget) instrumentation and
   a role-partitioned wake/dream split would address §3.2.
   The result would still need to pass the DR-2 / DR-2'
   property tests under the additional precondition.
3. **Empirical benchmark surprise.** If a future Paper 2 cycle
   includes D2L as a §7.8 published-reference row and that
   row outperforms all framework-C profiles on M1.a /
   M1.b / M3.b by a margin exceeding the
   underperforming-baseline rule (Paper 2 §6 / §7.1.13
   threshold), the framework would be required to *cite* D2L
   as a stronger baseline but would still not have to
   *adopt* its architecture — the citation-only
   classification is robust to empirical outperformance ;
   only architectural-primitive adoption would require a
   spec change.

Until any of these three triggers fires, D2L remains
category-(a) for the dreamOfkiki / framework-C development
cycle.

---

## 6. References

- **Dream2Learn** : Calcagno S., Pennisi M., Proietto
  Salanitri F., Sorrenti A., Palazzo S., Spampinato C.,
  Bellitto G. *Dream2Learn: Structured Generative Dreaming
  for Continual Learning.* arXiv:2603.01935, 2026.
  PeRCeiVe Lab, University of Catania. URL :
  <https://arxiv.org/abs/2603.01935>. BibTeX key
  `calcagno2026dream2learn` (see
  `docs/papers/paper2/references.bib`).
- **Framework-C DR-3** : axioms spec §6.2 ; `docs/axioms/AXIOMS.md`
  L165-L208 ; `docs/proofs/dr3-substrate-evidence.md` (C2.10).
- **Framework-C primitives & channels** : glossary.md L21-L34 ;
  `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
  §2.1.
- **DR-2 compositionality** : `docs/proofs/dr2-compositionality.md`
  v0.2-draft (2026-04-21).
- **DR-0 accountability** : `docs/proofs/dr-0.md` v0.1-draft
  (2026-05-02).
- **Wake-Sleep CL baseline** (parallel category-(a) treatment) :
  [@alfarano2024wakesleep] ; Paper 2 §7.7 ; arXiv:2401.08623.
- **Classification source** : memory file
  `~/.claude/projects/-Users-electron/memory/project_hypneum_deepresearch_2026_05_19_classification.md`
  (c-alert decision table, 2026-05-20).
- **Biophysical sub-theory exclusion** :
  `docs/specs/2026-05-20-biophysical-stratification.md` §6
  L239-L242 (Dream2Learn excluded from (b′) routing,
  citation-only).
