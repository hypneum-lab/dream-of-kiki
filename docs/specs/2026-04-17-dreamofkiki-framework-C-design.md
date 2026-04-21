# dreamOfkiki — Framework C Formal Design Spec

**Version** : C-v0.7.0+PARTIAL
**Date** : 2026-04-17 (initial) ; 2026-04-19 (cycle-2 closeout bump ; PARTIAL → STABLE after Phase 3+4 merge) ; 2026-04-19 (cycle-3 Phase 1 bump ; C-v0.6.0+STABLE → C-v0.7.0+PARTIAL, pre-C3.6 matrix launch)

> **Bump rationale (2026-04-19, cycle-3 Phase 1)**
>
> FC minor (0.6.0 → 0.7.0) : H6 adds a derived-constraint surface
> across substrates per §12.2 — the profile-ordering invariant
> (P_max > P_equ > P_min) is now required to hold across MLX and
> E-SNN substrates, not per substrate in isolation. No primitive
> signature / axiom ID changed ; the new constraint is derived from
> DR-3 condition 2 and §8.2.H6.
>
> EC STABLE → PARTIAL : the §8.2 evaluation matrix has cells
> deferred pending cycle-3 Phase 2 closure. Deferred cells
> (row × column) :
>
> - **Norse substrate × {P_min, P_equ, P_max} × {1.5B, 7B, 35B}** —
>   depends on C3.11 Norse fallback wrapper, scheduled sem 4-5 (Phase 2b).
> - **fMRI alignment × {P_min, P_equ, P_max}** — depends on C3.15
>   Studyforrest BOLD loader + C3.16 state-alignment + C3.17 CCA,
>   scheduled sem 5-6 (Phase 2c).
> - **H6 cross-substrate meta-test** — aggregates the above, closes
>   once both Phase 2 tracks land.
>
> Re-closure target : C3.22 DualVer bump EC PARTIAL → STABLE at
> cycle-3 end (sem 6), gated by Gate G10 promotion
> CONDITIONAL → FULL-GO/STABLE per §9.
**Author** : Clement Saillant (L'Electron Rare)
**Status** : Draft for user review
**Companion** : `2026-04-17-dreamofkiki-master-design.md` (master vision)

This spec is the **formal core** of the dreamOfkiki research program. It defines primitives, profiles, invariants, axioms, evaluation protocol, testing discipline, and cross-track interfaces. Designed to be substrate-agnostic; current instantiation is kiki-oniric (Track A), future cycle 2 instantiation is E-SNN thalamocortical.

---

## 1. Scope of this spec

**In scope** :
- Formal definition of primitives (α, β, γ, δ, 1, 2, 3, 4)
- Formal definition of profiles (P_min, P_equ, P_max) with inclusion chain
- Dream-episode ontology (DE = 5-tuple)
- Invariants (I information, S safety, K compute) with enforcement
- Axioms (DR-0..DR-4) with strict mathematical formalization + proof targets
- Evaluation protocol (8 metrics, stratified matrix, bit-exact reproducibility)
- Publication-ready maturity mode criteria
- Testing pyramid + coverage targets
- Cross-track interface contracts

**Out of scope** :
- Implementation details (see Track A design in `kiki-oniric/`)
- Business/commercial positioning (see master spec)
- Cycle-2 SNN substrate E (see master spec Section 2.2)

---

## 2. Primitives

### 2.1 Signatures

Primitives are typed communications between the awake process and the dream process. Each primitive has a formal signature with precondition, postcondition, and complexity bound.

#### Awake → Dream

**α — Raw traces**
- Type : `Stream<ForwardPassTrace>` where `ForwardPassTrace = (tokens, activations, attention_patterns, prediction_errors)`
- Purpose : firehose of awake inference data
- Activation : P_max only
- Complexity : O(tokens × model_size) per forward pass
- Storage : ring buffer mmap, LIFO rotation
- Precondition : awake process active
- Postcondition : trace appended to ring within bounded latency

**β — Curated episodic buffer**
- Type : `AppendLog<EpisodicRecord>` where `EpisodicRecord = (context, outcome, saillance_score, timestamp, consumed_by)`
- Purpose : hippocampal-like selection of salient episodes
- Activation : all profiles
- Complexity : O(1) append when saillance > threshold
- Storage : SQLite with index on saillance + consumed_by
- Precondition : saillance emitter running in awake
- Postcondition : record persisted atomically

**γ — Weights-only snapshot**
- Type : `Pointer<CheckpointHandle>`
- Purpose : minimal state, for pure SHY downscaling
- Activation : fallback / diagnostic
- Complexity : O(1) pointer access
- Precondition : checkpoint exists
- Postcondition : read-only view valid until next checkpoint

**δ — Hierarchical latent snapshots**
- Type : `RingBuffer<HierarchicalSnapshot>` where `HierarchicalSnapshot = Dict[SpeciesName, LatentTensor]`
- Purpose : multi-scale representation for thalamocortical-like restructuring
- Activation : P_equ, P_max
- Complexity : O(Σ species_dim) per snapshot
- Storage : ring buffer N=256
- Precondition : species activation hooks installed
- Postcondition : snapshot valid for at least N×τ_snapshot wall-clock

#### Dream → Awake

**1 — Weight delta**
- Type : `WeightUpdate` where `WeightUpdate = (LoRAdelta, FisherBump)`
- Purpose : parametric consolidation output
- Applied via : worktree swap (Section 6)
- Constraint : must satisfy invariant S1 (retained non-regression) + S2 (finite values)

**2 — Latent samples**
- Type : `Queue<LatentSample>` where `LatentSample = (species, latent_vector, provenance)`
- Purpose : generative replay / data augmentation input for awake
- Applied via : awake data augmenter consumes queue
- Constraint : must satisfy invariant I3 (distributional drift bounded)

**3 — Hierarchy change**
- Type : `TopologyDiff` where `TopologyDiff = List[TopoOp]`, `TopoOp ∈ {Add(layer), Remove(layer), Reroute(src, dst)}`
- Purpose : structural learning output
- Applied via : atomic application at swap time
- Constraint : must satisfy invariant S3 (topology valid)

**4 — Attention prior**
- Type : `Tensor[species, attention_weight]`
- Purpose : meta-cognitive guidance output
- Applied via : copy at swap or live read-only
- Constraint : must satisfy invariant S4 (attention bounded)

### 2.2 Channel coupling matrix

| Channel pair | Coupling | Rationale |
|--------------|----------|-----------|
| β + δ | Synergic | Salience + structure = hippocampus + thalamocortex coherent |
| γ alone | Anti-coupling | Kills A (replay), C (recombinaison), D (restructure) — keep γ only as complement |
| α + all | Expensive | Firehose makes everything else more costly |
| 1 + 3 | Synergic but risky | Coherent change, requires S3 guard |
| 2 alone | Isolated | Awake must actively consume (no param update) |
| 4 + 1 | Synergic | Attention priors guide next weight updates |

---

## 3. Profiles

### 3.1 Formal definitions

```
P_min = { primitives_in: {β},     primitives_out: {1},     ops: {replay, downscale} }
P_equ = { primitives_in: {β, δ},  primitives_out: {1,3,4}, ops: {replay, downscale, restructure, recombine_light} }
P_max = { primitives_in: {α,β,δ}, primitives_out: {1,2,3,4}, ops: {replay, downscale, restructure, recombine_full} }
```

### 3.2 Inclusion chain (Axiom DR-4 setup)

```
ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)
channels_in(P_min) ⊆ channels_in(P_equ) ⊆ channels_in(P_max)
channels_out(P_min) ⊆ channels_out(P_equ) ⊆ channels_out(P_max)
```

This inclusion is not merely cosmetic: it guarantees that any run under P_min is a valid special case of P_equ (with unused channels ignored), enabling monotonicity-based reasoning in ablation.

---

## 4. Dream-episode ontology (DE)

### 4.1 Definition

A **Dream-episode** is a 5-tuple:

```
DE = (trigger, input_slice, operation_set, output_delta, budget)
```

where:
- `trigger ∈ { scheduled(Δt), saturation(signal_type, threshold), external(event) }`
- `input_slice = snapshot(β_t) ∪ snapshot(δ_t)` (optionally ∪ `snapshot(α_t)` if P_max)
- `operation_set ⊆ {replay, downscale, restructure, recombine}` (composed)
- `output_delta ∈ {WeightUpdate, LatentSample, TopologyDiff, AttentionPrior}` (channels 1-4)
- `budget = max(FLOPs, wall_time, energy_J)` (resource cap)

### 4.2 Operations

Four canonical operations, typed `Op : State × Budget → State × Output`:

**replay** — sample from β, forward through current W, update via gradient on retention objective
- Source : A-Walker/Stickgold consolidation
- Input : β slice
- Output : WeightUpdate (channel 1)

**downscale** — apply homeostatic regularizer (SHY-style), shrink weights toward prior, reduce noise
- Source : B-Tononi synaptic homeostasis
- Input : γ or W directly
- Output : WeightUpdate (channel 1)

**restructure** — apply predictive processing update, minimize free energy over hierarchy, add/remove/reroute layers if needed
- Source : D-Friston FEP
- Input : δ snapshot
- Output : WeightUpdate + TopologyDiff (channels 1+3) + optional AttentionPrior (channel 4)

**recombine** — sample latents, interpolate/mix across species, generate new candidates (VAE-like or diffusion)
- Source : C-Hobson/Solms creative generation
- Input : δ or combined latents
- Output : LatentSample (channel 2), optionally WeightUpdate

### 4.3 Canonical composition

A typical P_equ dream-episode has operation_set with ordered composition:

```
DE_canonical = replay → downscale → restructure (parallel: recombine)
```

Note: the canonical order A→B→D in series is thermodynamically motivated (first preserve, then regulate, then restructure). Recombine (C) runs in parallel as a separate branch.

**Note on the `recombine` ∥ `DE_canonical` interaction and DR-2** :
because the `recombine` branch executes in parallel with the
sequential chain `replay → downscale → restructure`, it is
explicitly **out-of-scope of DR-2 sequential composition** (which
models a non-commutative monoid of in-series operations). A proper
parallel monoidal model — expressing effect isolation between the
serial branch and the `recombine` branch — is deferred to
`g3-draft.md` (cycle 2). Until then, the operational invariant is :
outputs from the `recombine` branch (channel-2 LatentSample) are
journalled independently and do not participate in `effect(...)` of
the DR-2 series until the atomic swap §7.2.

---

## 5. Invariants

All invariants follow structure: `name, formal statement, motivation, kiki example, refutation test, enforcement, criticality`.

### 5.1 Family I (Information)

**I1 — Episodic conservation until consolidation**
- Formal : `∀ e ∈ β_t, ∃ t' ≤ t + τ_max, e ∈ inputs(DE_{t'})` before purge
- Motivation : nothing salient is forgotten without chance to consolidate (core A-Walker)
- Kiki : each β record has `consumed_by_DE_id` flag; purge iff flag set
- Test : `SELECT COUNT(*) FROM beta WHERE consumed_by IS NULL AND created_at < now() - τ_max` == 0
- Enforcement : T-Ops hourly cron
- Criticality : **BLOCKING**

**I2 — Hierarchy traceability**
- Formal : every `TopologyDiff δ` is recorded with `(DE_id, C_version, before_hash, after_hash, applied_at)` and the recorded diff equals the actual topology delta
- Motivation : reproducibility + forensic
- Kiki : `hierarchy_diffs` table with FK constraints
- Test : diff between pre/post swap topologies equals recorded diff
- Enforcement : T-Ops post-swap validator
- Criticality : **BLOCKING**

**I3 — Latent distributional drift bounded**
- Formal : `KL(p_recombined || p_awake) ≤ ε_drift` per C-version, measured over sliding 1000-sample window
- Motivation : prevent dream from poisoning awake distribution
- Kiki : paired KL estimation over matched samples
- Test : integrated into metric M4.a
- Enforcement : A.5 experiment runner check
- Criticality : **WARN** (detect, not block)

### 5.2 Family S (Safety)

**S1 — Retained task non-regression**
- Formal : `acc(W_post_swap, retained) ≥ acc(W_pre_swap, retained) − δ_regression` (default δ = 2%)
- Motivation : prevent catastrophic forgetting via consolidation
- Kiki : retained = frozen 500-item subset of mega-v2
- Test : integrated into swap protocol step 3
- Enforcement : swap protocol aborts if fail, logs to `aborted-swaps/`
- Criticality : **BLOCKING**

**S2 — No NaN/Inf in W_scratch**
- Formal : `∀ w ∈ W_scratch, isfinite(w) ∧ |w| ≤ w_max`
- Motivation : avoid gradient explosion via mis-calibrated recombine
- Kiki : numpy/MLX norm scan + NaN detection pre-swap
- Enforcement : swap protocol step 2, before S1 guard
- Criticality : **BLOCKING**

**S3 — Hierarchy guard**
- Formal : `validate_topology(G_post) == True` where validate checks (i) species connectivity, (ii) no unwanted cycles, (iii) layer counts within bounds
- Motivation : restructure can disconnect ρ_sem and mute kiki
- Kiki : `validate_topology(G)` function with explicit rule list
- Enforcement : swap S1-guard integration + pre-emission check in A.4
- Criticality : **BLOCKING**

**S4 — Attention prior bounded**
- Formal : `∀ i, prior[i] ∈ [0, 1] ∧ Σ prior ≤ budget_attention` (default 1.5 × uniform)
- Motivation : prevent obsessive dream (dreaming 90% on ρ_phono blocks syntax)
- Kiki : clamp + normalize before application
- Enforcement : A.4 dream runtime emission
- Criticality : **WARN** (auto-correct + log)

### 5.3 Family K (Compute)

**K1 — Dream-episode budget respected**
- Formal : `∀ DE, FLOPs_actual ≤ budget.FLOPs ∧ wall_time_actual ≤ budget.wall ∧ energy_actual ≤ budget.energy`
- Motivation : prevent runaway DEs
- Kiki : context manager wrapper with FLOPs counter (MLX profile) + energy proxy
- Enforcement : A.4 runtime wrapper; on exceed → abort DE, discard partial output
- Criticality : **BLOCKING** (for this DE, not pipeline)

**K3 — Swap latency bounded**
- Formal : `wall_clock(swap_atomic) ≤ K3_max` (default 1s, configurable by topology)
- Motivation : perceived glitch if too long
- Kiki : measurement before/after swap
- Enforcement : swap monitoring, alert if exceeded
- Criticality : **WARN**

**K4 — Eval matrix coverage**
- Formal : for any MAJOR bump of C-version, stratified matrix fully executed before tag
- Motivation : prevent publication on un-validated C-version
- Kiki : T-Ops CI hook refuses tag without all runs present
- Enforcement : T-Ops CI
- Criticality : **BLOCKING** (for tagging)

---

## 6. Axioms (DR — Dream-episode Rules, strictly formalized)

### 6.1 Formal framework

Let:
- **State** = `(W, H, M)` where W = weights, H = hierarchy topology, M = episodic memory (β buffer)
- **Op** = monoid of operations {replay, downscale, restructure, recombine} typed `Op : State × Budget → State × Output`
- **DE** = ordered composition of elements of Op with additive budget

### 6.2 Axioms

#### DR-0 (Accountability)

```
∀ δ ∈ dream_output_channels,
∃ DE ∈ History : budget(DE) < ∞ ∧ δ ∈ outputs(DE)
```

**Interpretation** : no "ambient dreaming" — every dream contribution is traceable to a bounded DE.

**Enforcement** : I1 + I2 + K1 combined.

#### DR-1 (Episodic conservation, formalizes I1)

```
∀ e ∈ β_t, ∃ t' ∈ [t, t + τ_max] : e ∈ inputs(DE_{t'})
```

**Interpretation** : every buffered episode gets a chance at consolidation.

**Enforcement** : I1 runtime check + β purge gate.

#### DR-2 (Compositionality — weakened with precondition, 2026-04-21)

```
∀ permutation π = (op_0, ..., op_{n-1}) over Op such that
  ¬∃ i < j : (π_i = RESTRUCTURE ∧ π_j = REPLAY),
  π is composable into Op
  ∧ budget(π) = Σ_k budget(π_k)
  ∧ effect(π, s) = effect(π_{n-1}, ..., effect(π_0, s))
```

**Precondition rationale**: the predicate
`∃ i < j : π_i = RESTRUCTURE ∧ π_j = REPLAY` captures the empirically
falsified class identified by Hypothesis property testing on the
real-weight substrate (see
`tests/conformance/axioms/test_dr2_compositionality_empirical.py`,
2026-04-21). The layer swap performed by RESTRUCTURE leaves the MLP
non-callable with the canonical (2, 4) input shape consumed by a
subsequent REPLAY. The precondition excludes exactly those 12 out of
24 permutations of the four canonical operations; the 12 remaining
permutations preserve closure, budget additivity, and effect
chaining.

**Cycle-1 status**: as of 2026-04-21 DR-2 is **weakened with a precondition** (Option B, see `docs/specs/amendments/2026-04-21-dr2-empirical-falsification.md`). The precondition-bounded form is no longer unproven — its closure, budget additivity, and effect chaining are validated empirically by Hypothesis property testing over the remaining 12 safe permutations. The operational version DR-2' (canonical order only) is retained as the stricter contract used by the G2/G4 pilots.

**Proof target** : by cases over the 4 operations + associativity lemma.

**Proof sketch** (to be completed in track C output `formal-proofs.md`) :

1. **Closure** : show that applying op_2 after op_1 preserves State type and produces valid Output. By typing rules.
2. **Budget additivity** : by definition of budget as resource counter (FLOPs, wall-time, energy), additive by construction.
3. **Functional composition** : by definition of `effect` as function on State.
4. **Associativity** : show `(op_3 ∘ op_2) ∘ op_1 = op_3 ∘ (op_2 ∘ op_1)` by case analysis.

**Commutativity is NOT claimed**. In particular `recombine ∘ downscale ≠ downscale ∘ recombine` in general. The monoid is non-commutative.

**Fallback DR-2'** : if strict compositionality proof fails, adopt:

```
DR-2' (Compositionality with canonical ordering)
∀ op_1, op_2 ∈ Op_canonical_order,
  op_2 ∘ op_1 ∈ Op
  (canonical order = replay < downscale < restructure < recombine)
```

This is weaker but sufficient for the canonical DE composition defined in 4.3.

#### DR-3 (Substrate-agnosticism — operational criterion)

```
∀ substrate S, if S satisfies the Conformance Criterion below,
then DR-0, DR-1, DR-2 (or DR-2') are **empirically validated** on S
(operational sense, see §6.2 "Operational statement" ; not a formal
implication — DR-2 itself remains an unproven working axiom, §6.2).
```

**Interpretation** : kiki-oniric and hypothetical E-SNN are both valid
instantiations **iff they pass the Conformance Criterion**. Pure signature
typing is **not** sufficient — behavioral conformance is required. DR-3 is
presented here as an **operational/empirical criterion** rather than a formal
implication ; the formal epistemic status of each axiom is unchanged by
`conforms(S)`.

**Conformance Criterion (executable)** : a substrate S instantiates the framework iff all three conditions hold:

1. **Signature typing** — S implements the typed Protocol signatures of primitives α, β, γ, δ, 1, 2, 3, 4 as defined in §2.1 (Python Protocol types, TypeScript interfaces, or equivalent in target language).

2. **Axiom property tests** — the property test suite for DR-0, DR-1, and DR-2 (or DR-2') passes on S with ≥100% coverage on BLOCKING cases. Test suite is versioned in T-Ops under `tests/conformance/axioms/` and executed via `dream-harness conformance --substrate <S>`.

3. **BLOCKING invariants enforceable** — invariants S1 (retained non-regression), S2 (no NaN/Inf), S3 (hierarchy valid), and I1 (episodic conservation) are implemented as runtime checks on S with automated enforcement (abort-on-violation + logging to `aborted-swaps/` or equivalent).

**Operational statement** :

```
conforms(S) ≜ typed(S) ∧ axiom_tests_pass(S) ∧ invariants_enforced(S, {S1,S2,S3,I1})
```

`conforms(S)` is **not** a formal implication into DR-0 ∧ DR-1 ∧
DR-2 — that would be circular (the property tests defining
`axiom_tests_pass(S)` are precisely the axiom checks themselves).
It is an **operational criterion** : a substrate that satisfies
`conforms(S)` provides **empirical evidence** (validation, not
proof) that DR-0/DR-1/DR-2 hold on that substrate. A violation of
`conforms(S)`, on the other hand, is a direct counter-example to
the corresponding axioms.

**Evidence decomposition** : given `conforms(S)`, each of the three conditions provides evidence for one part of the axioms:
- `typed(S)` → operations are compositional in type (evidence for the closure part of DR-2)
- `axiom_tests_pass(S)` → behavioral axioms are empirically validated (evidence for the functional composition part of DR-2, accountability DR-0, conservation DR-1)
- `invariants_enforced(S)` → runtime guarantees preserve axioms under concurrent execution (swap worktree, async dream process)

Together, conforming substrate is validated both **statically** (typing + property tests) and **dynamically** (invariant enforcement), not merely asserted by construction.

**Cycle-1 status** : kiki-oniric conformance is established incrementally — signature typing locked by S2 (Story 0 expose-primitives), axiom tests passing by S6 (aligned with G3-draft milestone), invariants enforceable by S8 (full Track A P_equ runtime). Cycle-2 substrate E-SNN must pass the same three conditions before being claimed as instantiation.

#### DR-4 (Profile chain inclusion)

```
ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)
∧ channels(P_min) ⊆ channels(P_equ) ⊆ channels(P_max)
```

**Lemma DR-4.L** : if P_min is valid (invariants satisfied) on substrate S, then P_equ does not strictly worsen metrics monotone in capacity.

**Proof sketch** : P_equ extends P_min with additional channels and operations; a P_equ run can simulate a P_min run by ignoring added channels; therefore best(P_equ) ≥ best(P_min) in expectation over appropriate metric classes.

### 6.3 Axiom verification in implementation

Each axiom must be **empirically testable** via mechanical test:

| Axiom | Test form | Track | Frequency |
|-------|-----------|-------|-----------|
| DR-0 | Query all output deltas → trace to DE with finite budget | T-Ops property test | CI nightly |
| DR-1 | Query β records → all consumed within τ_max | I1 hourly cron | Runtime |
| DR-2 | Property test : random op pairs, verify composition types + additivity | T-Ops CI | Each PR |
| DR-3 | For each substrate, run full invariant suite → all pass | T-Ops | Per substrate |
| DR-4 | Compare ops/channels sets → inclusion holds | T-Ops contract test | Each profile change |

---

## 7. Runtime protocol (swap worktree)

### 7.1 State copies

- `W_awake` : active weights used by awake process (read/write)
- `W_dream` : frozen snapshot read by dream process (read-only)
- `W_scratch` : working copy modified by dream process (read/write, dream-only)

### 7.2 Swap protocol (atomic)

When dream signals `ready_to_commit`:

```
1. awake.pause(max=500ms)                      # operational target ; K3_max=1s = warning threshold (§5.3)
2. validate_scratch_finite(W_scratch)          # S2 guard
3. acc_post = eval(W_scratch, retained_bench)  # S1 guard
4. if acc_post < acc_pre - δ_regression:
      abort_swap()
      log_to('aborted-swaps/')
      return
5. if topology_changed:
      validate_topology(H_scratch)             # S3 guard
      if fail: abort_swap() ; log ; return
6. atomic_swap:
      W_awake ← W_scratch
      W_dream ← W_awake  (fresh baseline)
      W_scratch ← W_awake  (new working copy)
7. apply_channel_4_if_present(attention_prior)
8. awake.resume()
```

### 7.3 Post-swap cleanup

- Update dashboard (T-Ops.3) with swap result
- Log swap to `runs/<C-ver>/<profile>/<seed>/<run-id>/swaps/`
- Advance dream_episode state machine
- Reset `W_scratch` ready for next DE cycle

---

## 8. Evaluation protocol

### 8.1 Metrics (8 total, E3 cognitive + E4 engineering)

| Code | Name | Formula/Procedure | Deterministic strategy |
|------|------|-------------------|------------------------|
| M1.a | Forgetting rate | `(acc_task_1_initial - acc_task_1_after_task_N) / acc_task_1_initial` | Numpy seeded RNG |
| M1.b | Average accuracy | `mean(acc_task_i for i in 1..N after full sequence)` | Numpy seeded RNG |
| M2.b | RSA fMRI alignment | Pearson correlation between RDM(kiki representations) and RDM(fMRI activity) on matched stimuli | Stimulus ordering seeded; nilearn CPU mode |
| M3.a | FLOPs ratio | `FLOPs(dream) / FLOPs(awake)` over comparable window | MLX static profile |
| M3.b | Offline gain | `Δ(M1.b, post-dream) - Δ(M1.b, no-dream)` normalized by FLOPs-equivalent wall-clock | Simulated wall-clock from FLOPs |
| M3.c | Energy per episode | `energy_proxy = f(FLOPs, model_size, precision)` — calibrated function | Deterministic function |
| M4.a | Recomb quality | Teacher scorer (Qwen3.5-9B Q4_K_M — *SHA to be pinned at the benchmark v1.0 freeze, tracking target §8.4 SHA pinning*) evaluates plausibility + diversity of latent samples | Scorer temp=0, seed=0 |
| M4.b | Structure discovery | Permutation test on learned hierarchy invariants vs baseline | Permutation seeded |

### 8.2 Stratified matrix

| Bump type | Obligatory cells |
|-----------|------------------|
| **PATCH** (C-vX.Y.z+1) | Affected axis metric × P_equ × 1 seed |
| **MINOR** (C-vX.y+1) | All 8 metrics × P_equ × 3 seeds |
| **MAJOR** (C-vx+1) | **Full grid** : 8 metrics × 3 profiles × 3 seeds = 72 runs |
| **EC change** (STABLE → DIRTY) | Re-run only published-result metrics |

### 8.3 Reproducibility contracts

**R1 (bit-exact)** : every `MetricResult` is bit-identical reproducible from
`(c_version, profile, seed, run_id, commit_sha, benchmark_version)` **for the
metrics whose external dependencies are SHA-pinned**. Scope currently covers
M1.a, M1.b, M2.b, M3.a, M3.b, M3.c, M4.b. M4.a is **not R1-compliant** until
the Teacher scorer SHA is pinned at the benchmark v1.0 freeze (see §8.4).

**R3 (artifact addressability)** : all artifacts addressable by SHA-256 checksum stored in `metadata.yaml`, storage schema per master spec §5.4.

(R2 suppressed — all metrics now deterministic.)

### 8.4 SHA pinning (R1 scope completion)

External dependencies that must be SHA-pinned before a metric enters the R1
scope :

| Metric | Dependency | Status |
|--------|-----------|--------|
| M4.a | Teacher scorer Qwen3.5-9B Q4_K_M GGUF | pinned at benchmark v1.0 freeze |

Until the row is pinned, the corresponding metric is excluded from R1 and
explicitly marked in its `MetricResult` metadata as `r1_scope=false`.

---

## 9. Publication-ready maturity mode

Three-level pipeline maturity, with explicit transitions:

| Mode | Criteria | Authorizes |
|------|----------|-----------|
| **RED** | ≥1 BLOCKING violated OR ≥3 WARN in 24h | No action, escalate Dream-sync |
| **GREEN** | All BLOCKING respected, WARN below threshold | Development, experiments |
| **PUBLICATION-READY** | GREEN + additional criteria below | Paper submission |

### Additional criteria for PUBLICATION-READY

| Criterion | Threshold |
|-----------|-----------|
| Eval matrix coverage | 100% stratified cells (MAJOR/MINOR/PATCH) |
| Reproducibility | ≥3 seeds per (profile, metric), variance documented |
| Retained benchmark | No regression δ > 1% on last 10 swaps |
| Zero BLOCKING | 7 consecutive days without violation |
| DualVer status | `+STABLE` (not DIRTY, not INVALIDATED) |
| Pre-submission review | ≥1 positive return from T-Col.4 network |
| Axioms DR | DR-0..DR-4 formalized, DR-2 proven (or DR-2' adopted) |
| Ablation complete | P_min, P_equ, P_max tested with significance |
| Paper draft | Complete, peer-reviewed internal ≥1×, no TODO |

**Enforcement** : T-Ops CI blocks commit to `papers/*/submitted/` folder unless mode == PUBLICATION-READY.

**Transitions** :
- RED → GREEN : resolve blocker + 24h monitoring
- GREEN → PUBLICATION-READY : T-Ops CI check triggered manually (Dream-sync decision)
- PUBLICATION-READY → GREEN : any C-version MAJOR bump erases status

---

## 10. Testing discipline

### 10.1 Pyramid

```
      E2E tests (few, slow, scenario-level)
     Integration tests (cross-track contracts)
    Unit tests (per module, fast)
   Property tests (invariants, Hypothesis)
```

### 10.2 Coverage targets (strict mode)

| Family | Coverage target | Enforcement |
|--------|----------------|-------------|
| Unit tests | ≥90% line per module | CI gate blocking |
| Property tests | Invariants BLOCKING + WARN 100% | CI gate blocking |
| Axiom tests | DR-0..DR-4 all mechanically tested | CI gate blocking |
| Integration tests | Each cross-track interface ≥1× | CI gate blocking |
| Benchmark tests | Retained benchmark re-run per merge | CI gate blocking |
| Repro tests | Golden runs re-tested weekly | Nightly alert |

### 10.3 TDD discipline

- **Invariants I/S/K and DR axioms** : strict TDD — test first, code until test passes.
- **Data pipelines + metrics** : strict TDD — must be tested before production.
- **Application code** : TDD recommended but not enforced — 90% coverage gate guarantees outcome.

---

## 11. Cross-track interfaces

Each interface is a **first-class versioned artifact** (semver-tagged with C-DualVer), tested by contract tests in T-Ops.

| Interface | Artifact | Owners | Lock date |
|-----------|----------|--------|-----------|
| C ↔ A | `interfaces/primitives.md` + Python Protocol types | Track C + Track A | S2 |
| C ↔ T-Ops | `interfaces/eval-matrix.yaml` | Track C + T-Ops | S5 |
| A ↔ T-Ops | `interfaces/experiment-contract.md` | Track A + T-Ops | S6 |
| C ↔ T-Col | `interfaces/proposal-template.md` | Track C + T-Col | S3 |
| A ↔ T-Col | `interfaces/fmri-schema.yaml` | Track A + T-Col | S4 |

Interface changes require bump of **both** tracks' DualVer minor.

---

## 12. DualVer versioning scheme

### 12.1 Format

`C-v<FC-MAJOR>.<FC-MINOR>.<FC-PATCH>+<EC-STATE>`

### 12.2 FC axis (formal consistency — SemVer-like)

- **FC-MAJOR** : semantic change of axiom / primitive / invariant signature
- **FC-MINOR** : addition of new axiom / new optional primitive / new derived constraint
- **FC-PATCH** : clarification / typo / equivalent reformulation / pseudo-code fix

### 12.3 EC axis (empirical consistency)

- **STABLE** : all empirical results measured under current FC, consistent with axioms
- **PARTIAL** : engineering deliverables green on current FC, but part of the
  evaluation matrix (e.g. a publication-track phase, a cross-substrate ablation)
  is scoped-deferred rather than failing — coverage of §8.2 stratified matrix is
  incomplete but no axiom is violated. Allowed transitions : PARTIAL → STABLE
  (on re-closure of deferred cells) or PARTIAL → DIRTY (if a deferred cell is
  later run and does not re-verify under current FC).
- **DIRTY** : at least one result is orphan (measured under older FC, not re-verified)
- **INVALIDATED** : a published/submitted result is invalidated by current FC-MAJOR

### 12.4 Bump workflow

1. Track C owner proposes bump via PR
2. T-Ops runs **stratified compat suite** (per §8.2) against new C
3. If contract tests pass → EC remains STABLE
4. If tests break → EC auto-set to DIRTY, dead tests tagged, issue opened
5. Merge PR → official bump, glossary updated, T-Col notified

---

## 13. Open formal questions (to resolve S1-S8)

1. **DR-2 proof** — is strict compositionality provable, or must we fall back to DR-2'? (G3 decision S8)
2. **Commutativity boundary** — for which op pairs does `op_2 ∘ op_1 = op_1 ∘ op_2`? (informs canonical order)
3. **Budget additivity under parallel C-branch** — how does additivity compose when recombine runs in parallel with serial A→B→D branch? (needs parallel monoid extension)
4. **Substrate conformance criterion** — RESOLVED (see §6.2 DR-3) : conformance requires signature typing ∧ axiom property tests ∧ BLOCKING invariants enforceable. Typing alone is insufficient.
5. **Monotonicity metric class for DR-4.L** — formally characterize which metrics are monotone in capacity for the lemma to hold strictly.

These questions feed into the proof work of section C.4 (formal-proofs appendix), weeks S5-S8.

---

## 14. Appendices

### A. Canonical glossary (excerpt)

| Term | Definition |
|------|-----------|
| `dreamOfkiki` | Program name, logical identifier (camelCase). Repo technical name: `dream-of-kiki` (kebab-case for filesystem compat) |
| `kiki-oniric` | Dedicated fork of `kiki-flow-core` for A-track experiments |
| Dream-episode (DE) | 5-tuple (trigger, input_slice, operation_set, output_delta, budget) — atomic unit of dreaming |
| Awake process | Kiki inference/training process running in real-time |
| Dream process | Asynchronous offline consolidation process running on separate compute (per profile topology) |
| ortho species | 4 linguistic levels : ρ_phono, ρ_lex, ρ_syntax, ρ_sem (Baddeley + Levelt) |
| Swap | Atomic promotion of W_scratch to W_awake (replaces merge in original design) |
| PUBLICATION-READY | Pipeline maturity mode authorizing paper submission |
| DualVer | `C-v<FC>+<EC>` versioning scheme with separate formal + empirical axes |

### B. Reference to master spec

See `2026-04-17-dreamofkiki-master-design.md` for:
- Business/calendar integration
- Publication strategy details
- Risk register (sections 7-8)
- Track A/T-Ops/T-Col operational details
- Compute topology per profile
- Calendar S1-S28

---

**End of framework C design spec.**

Next step after user review : invoke `writing-plans` skill to generate implementation plan per track.
