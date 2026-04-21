# Amendment: DR-2 empirical falsification

- **Date**: 2026-04-21
- **Status**: Adopted — Option B (weakened precondition), 2026-04-21
- **Target**: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
  §6.2 (DR-2 compositionality)
- **Evidence**: `tests/conformance/axioms/test_dr2_compositionality_empirical.py`

## Context

DR-2 as stated in the framework-C design spec §6.2 asserts
compositionality for any ordering of operations:

```
∀ op_1, op_2 ∈ Op,
  op_2 ∘ op_1 ∈ Op
  ∧ budget(op_2 ∘ op_1) = budget(op_1) + budget(op_2)
  ∧ effect(op_2 ∘ op_1, s) = effect(op_2, effect(op_1, s))
```

with the caveat *"Commutativity is NOT claimed"*. DR-2 was flagged
as an **unproven working axiom**, with DR-2' (canonical-order
compositionality) adopted as the operational fallback (see §6.2
addendum and `test_dr2_prime_canonical_order.py`).

## Empirical result

Property-based testing with Hypothesis over all 24 permutations of
the 4 canonical operations `{REPLAY, DOWNSCALE, RESTRUCTURE,
RECOMBINE}` identifies a falsifying class on the current real-weight
substrate:

> Any permutation where **RESTRUCTURE precedes REPLAY** (at any
> distance) raises `ValueError` from MLX `addmm` during the REPLAY
> step. The topology mutation performed by RESTRUCTURE (layer swap)
> leaves the MLP non-callable with the canonical `(2, 4)` input
> shape consumed by REPLAY.

12 of the 24 permutations are in this class. See
`tests/conformance/axioms/test_dr2_compositionality_empirical.py`
for the full enumeration (Hypothesis `@parametrize`-materialised
plus a generic probe).

## Interpretation

This does **not** contradict the spec — the spec explicitly flags
DR-2 as unproven. The empirical result upgrades the flag from
*"unproven working axiom"* to *"empirically falsified on the current
real-weight substrate under the canonical input schema"*.

Scope of the falsification:

1. **Substrate-dependent**. Only the real-weight MLX substrate was
   tested. A substrate where RESTRUCTURE preserves signature
   compatibility with REPLAY inputs may not exhibit this failure.
   DR-3 (substrate-agnosticism) therefore guards against
   over-generalising the result.
2. **Input-schema-dependent**. The canonical input shape `(2, 4)` is
   what triggers the shape mismatch. A different input schema could
   in principle avoid the issue for specific topology mutations.
3. **Closure is the failure mode**, not budget additivity or
   effect-chaining. Closure is the first conjunct of DR-2's
   conjunction; its failure refutes the whole claim as stated.

## Proposed amendment to §6.2

Three options, ordered by conservatism:

### Option A — Annotation only

Add a footnote to §6.2 citing the empirical finding:

> **Note** (2026-04-21). The closure clause of DR-2 has been shown
> empirically to fail for the real-weight substrate in the class of
> permutations where RESTRUCTURE precedes REPLAY (see
> `tests/conformance/axioms/test_dr2_compositionality_empirical.py`).
> DR-2' remains the operational contract.

### Option B — Weaken DR-2 to the observed safe class [ADOPTED 2026-04-21]

Redefine DR-2 as compositionality for the subset of `Op × Op` where
REPLAY does not follow RESTRUCTURE. This introduces a precondition
but preserves a non-trivial post-canonical claim.

### Option C — Demote DR-2, promote DR-2'

Remove DR-2 from the axiom list; promote DR-2' to DR-2
(canonical-order is *the* compositionality claim). Requires an
errata commit and a DualVer bump on the formal axis.

## Recommendation

**Option B adopted 2026-04-21** (FC-PATCH bump C-v0.7.0 → C-v0.7.1). The precondition-bounded DR-2 is empirically validated by the 11 passing permutations in the test suite. Option C (demote DR-2 entirely) remains available for Paper 2 if substrate-survey results warrant it.

## DualVer implication

- **Option A**: no formal-axis bump (annotation only). Empirical-axis
  unchanged (no gate crossed).
- **Option B**: formal-axis bump `+UNSTABLE` until the weakened
  form has been re-tested substrate-wide.
- **Option C**: formal-axis bump `+UNSTABLE` with an errata commit.

## References

- `tests/conformance/axioms/test_dr2_compositionality_empirical.py`
  — the 12 xfailing cases and the closure-safe passing cases.
- `tests/conformance/axioms/test_dr2_prime_canonical_order.py` —
  DR-2' (fallback, still enforceable).
- `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §6.2 —
  DR-2 canonical statement.
- `docs/axioms/AXIOMS.md` — compiled axiom reference (2026-04-21).
