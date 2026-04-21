# GENIAL axioms (DR-0 to DR-4)

Canonical reference. Source of truth:
`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §5 (invariants)
and §6 (axioms).

| Axiom | Name | Status | Test |
|-------|------|--------|------|
| DR-0 | Accountability | Proven / Exec | `tests/conformance/axioms/test_dr0_accountability.py` |
| DR-1 | Episodic conservation | Proven / Exec | `tests/conformance/axioms/test_dr1_episodic_conservation.py` |
| DR-2 | Compositionality (weakened) | Weakened with precondition (2026-04-21) | test_dr2_compositionality_empirical.py |
| DR-2' | Canonical-order compositionality | Fallback, Exec | `tests/conformance/axioms/test_dr2_prime_canonical_order.py` |
| DR-3 | Substrate-agnosticism | Proven / Exec | `tests/conformance/axioms/test_dr3_substrate.py`, `test_dr3_esnn_substrate.py` |
| DR-4 | Profile chain inclusion | Proven / Exec | `tests/conformance/axioms/test_dr4_profile_inclusion.py` |

## Formal framework (spec §6.1)

- **State** = `(W, H, M)` where W = weights, H = hierarchy topology,
  M = episodic memory (β buffer).
- **Op** = monoid of operations `{replay, downscale, restructure,
  recombine}` typed `Op : State × Budget → State × Output`.
- **DE** = ordered composition of elements of `Op` with additive
  budget.

---

## DR-0 — Accountability

**Formal statement** (verbatim, spec §6.2 DR-0):

```
∀ δ ∈ dream_output_channels,
∃ DE ∈ History : budget(DE) < ∞ ∧ δ ∈ outputs(DE)
```

**Intuition**: no "ambient dreaming" — every dream contribution on
any output channel is traceable to a bounded DE in the history log.

**Executable form**: hypothesis property test over `DreamRuntime`
outputs — every executed DE appears in the runtime log with a
finite `BudgetCap`.

**Enforcement**: I1 + I2 + K1 combined (spec §6.2).

---

## DR-1 — Episodic conservation (formalizes I1)

**Formal statement** (verbatim, spec §6.2 DR-1):

```
∀ e ∈ β_t, ∃ t' ∈ [t, t + τ_max] : e ∈ inputs(DE_{t'})
```

**Intuition**: every buffered episode gets a chance at consolidation
within the window `τ_max` before purge.

**Executable form**: property test on a fake β buffer — every record
added at time `t` is consumed by some DE within `[t, t + τ_max]`.
Real β lands S7+ alongside the swap protocol (test docstring).

**Enforcement**: I1 runtime check (hourly cron) + β purge gate.

---

## DR-2 — Compositionality (weakened with precondition, 2026-04-21)

**Formal statement** (verbatim, spec §6.2 DR-2, updated 2026-04-21):

```
∀ permutation π = (op_0, ..., op_{n-1}) over Op such that
  ¬∃ i < j : (π_i = RESTRUCTURE ∧ π_j = REPLAY),
  π is composable into Op
  ∧ budget(π) = Σ_k budget(π_k)
  ∧ effect(π, s) = effect(π_{n-1}, ..., effect(π_0, s))
```

**Status**: as of 2026-04-21 DR-2 is **weakened with a precondition**
(Option B). The precondition-bounded form is empirically validated by
Hypothesis property testing over the 12 safe permutations. The
operational contract used by G2/G4 pilots is the **DR-2'** fallback
(canonical order only), which remains the stricter contract.

**2026-04-21 update**: DR-2 weakened with a precondition excluding the empirically falsified class (RESTRUCTURE preceding REPLAY). See amendment `docs/specs/amendments/2026-04-21-dr2-empirical-falsification.md` and test `tests/conformance/axioms/test_dr2_compositionality_empirical.py`.

**Commutativity is NOT claimed**. In particular
`recombine ∘ downscale ≠ downscale ∘ recombine` in general. The
monoid is non-commutative.

**Parallel branch note** (spec §4.3): `recombine` runs in parallel
with the serial chain `replay → downscale → restructure` and is
**out-of-scope of DR-2 sequential composition**. Channel-2
`LatentSample` outputs from `recombine` are journalled independently
and do not participate in `effect(...)` of the DR-2 series until the
atomic swap §7.2.

**Proof target** (sketch, spec §6.2): by cases over the 4
operations + associativity lemma. Steps: (1) closure by typing,
(2) budget additivity by resource-counter construction,
(3) functional composition by definition of `effect`,
(4) associativity by case analysis.

---

## DR-2' — Canonical-order compositionality (fallback)

**Formal statement** (verbatim, spec §6.2 DR-2 "Fallback DR-2'"):

```
DR-2' (Compositionality with canonical ordering)
∀ op_1, op_2 ∈ Op_canonical_order,
  op_2 ∘ op_1 ∈ Op
  (canonical order = replay < downscale < restructure < recombine)
```

**Relationship to DR-2**: strict weakening — DR-2' restricts
composition to pairs respecting the canonical order. Sufficient for
the canonical DE composition defined in spec §4.3
(`DE_canonical = replay → downscale → restructure (parallel:
recombine)`).

**Adoption condition**: DR-2' is adopted **in lieu of** DR-2 until a
strict DR-2 proof is written. It is the **empirical contract**
retained by the G2/G4 pilots.

**Executable form**: determinism conformance test — chaining the
four canonical operations through a single `DreamRuntime` under an
identical seed produces a **byte-identical** final state across two
independent runs.

**Provenance**: DR-2' executable axiom test added 2026-04-21 in
commit `b13ab95` (`test(ops): determinism tests + DR-2' axiom`).

---

## DR-3 — Substrate-agnosticism (operational criterion)

**Formal statement** (verbatim, spec §6.2 DR-3):

```
∀ substrate S, if S satisfies the Conformance Criterion below,
then DR-0, DR-1, DR-2 (or DR-2') are **empirically validated** on S
(operational sense, see §6.2 "Operational statement" ; not a formal
implication — DR-2 itself remains an unproven working axiom, §6.2).
```

**Conformance Criterion** (spec §6.2, executable) — three
conditions, all required:

1. **Signature typing** — S implements the typed Protocol signatures
   of primitives α, β, γ, δ, 1, 2, 3, 4 as defined in §2.1 (Python
   Protocol types, TypeScript interfaces, or equivalent).
2. **Axiom property tests** — the property test suite for DR-0,
   DR-1, and DR-2 (or DR-2') passes on S with ≥100% coverage on
   BLOCKING cases (`tests/conformance/axioms/`, executed via
   `dream-harness conformance --substrate <S>`).
3. **BLOCKING invariants enforceable** — invariants S1 (retained
   non-regression), S2 (no NaN/Inf), S3 (hierarchy valid), and I1
   (episodic conservation) are implemented as runtime checks on S
   with automated enforcement (abort-on-violation + logging to
   `aborted-swaps/` or equivalent).

**Operational statement** (verbatim, spec §6.2):

```
conforms(S) ≜ typed(S) ∧ axiom_tests_pass(S) ∧ invariants_enforced(S, {S1,S2,S3,I1})
```

`conforms(S)` is **not** a formal implication into DR-0 ∧ DR-1 ∧
DR-2 — that would be circular. It is an operational criterion: a
substrate that satisfies `conforms(S)` provides **empirical
evidence** (validation, not proof) for the axioms on that substrate.
A violation of `conforms(S)` is a direct counter-example.

**Executable form**: `test_dr3_substrate.py` covers condition (1)
for kiki-oniric MLX; `test_dr3_esnn_substrate.py` covers all 3
conditions for the E-SNN thalamocortical substrate (C2.4).

---

## DR-4 — Profile chain inclusion

**Formal statement** (verbatim, spec §6.2 DR-4):

```
ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)
∧ channels(P_min) ⊆ channels(P_equ) ⊆ channels(P_max)
```

**Intuition**: each richer profile strictly extends the previous
one in both operations and channels — a P_min run is a valid
special case of P_equ (unused channels ignored), enabling
monotonicity-based reasoning in ablation (spec §3.2).

**Lemma DR-4.L** (spec §6.2): if P_min is valid (invariants
satisfied) on substrate S, then P_equ does not strictly worsen
metrics monotone in capacity.

**Proof sketch** (spec §6.2): P_equ extends P_min with additional
channels and operations; a P_equ run can simulate a P_min run by
ignoring added channels; therefore `best(P_equ) ≥ best(P_min)` in
expectation over appropriate metric classes.

**Executable form**: contract test comparing registered ops and
emitted output channels of `PMinProfile`, `PEquProfile`,
`PMaxProfile` (derived from runtime handler registration, not
static metadata). See `docs/proofs/dr4-profile-inclusion.md`.

---

## Axiom verification (spec §6.3)

| Axiom | Test form | Track | Frequency |
|-------|-----------|-------|-----------|
| DR-0 | Query all output deltas → trace to DE with finite budget | T-Ops property test | CI nightly |
| DR-1 | Query β records → all consumed within τ_max | I1 hourly cron | Runtime |
| DR-2 | Property test: random op pairs, verify composition types + additivity | T-Ops CI | Each PR |
| DR-3 | For each substrate, run full invariant suite → all pass | T-Ops | Per substrate |
| DR-4 | Compare ops/channels sets → inclusion holds | T-Ops contract test | Each profile change |

---

## Invariants touched by each axiom

Invariant families defined in spec §5 and `docs/invariants/registry.md`.

| Axiom | Invariants | Enforcement surface |
|-------|-----------|--------------------|
| DR-0 | I1, I2, K1 | Runtime log + hourly cron + DE budget wrapper |
| DR-1 | I1 | Hourly cron, β purge gate |
| DR-2 / DR-2' | (typing closure) | T-Ops property/determinism tests |
| DR-3 | S1, S2, S3, I1 (BLOCKING set per Conformance Criterion) | Swap protocol guards + conformance suite |
| DR-4 | (none runtime) | Contract test on profile registration |

---

## Changelog / provenance

- Source spec: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
  (version `C-v0.7.0+PARTIAL`, 2026-04-17 initial, 2026-04-19
  cycle-3 Phase 1 bump).
- Registry cross-ref: `docs/invariants/registry.md`.
- Glossary cross-ref: `docs/glossary.md`.
- Compiled: 2026-04-21.
- DR-2' executable axiom test added 2026-04-21 via commit
  `b13ab95` (`test(ops): determinism tests + DR-2' axiom`).
