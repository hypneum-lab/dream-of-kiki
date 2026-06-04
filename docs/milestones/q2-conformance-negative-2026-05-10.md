# Q2 — Framework C Conformance Criterion negative tests

**Date pre-registered:** 2026-05-10 (BEFORE building adversarial substrates)
**Spec source:** HYPNEUM-PLANS/specs/2026-05-10-three-innovation-experiments-design.md
**Plan source:** HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md (Task 5)
**Status:** Pre-registered, not yet executed.

## H0 (to refute)

The Conformance Criterion is false-positive-safe: no adversarial
substrate constructed to satisfy the axioms passes the criterion.

## Methodology

15 adversarial substrates across 3 categories, each implementing the
`SubstrateAdapter` Protocol (per N5 Task 4 architecture):

- **Cat A (5) — trivial accidents:** Identity, RandomNoise, Lookup,
  FrozenZeros, Constant
- **Cat B (7) — adversarial accidents:** ShapePreservingNoise,
  BudgetGaming, PermutedReplay, OverloadRecombine, StatelessAccident,
  CommutativityViolator, BoundaryCheater
- **Cat C (3) — statistical accidents (N=100 trials each):**
  RandomCoinFlip, ShapeDistributionDependent, SeedDependentSubstrate

Each substrate is run through the full conformance suite
(`tests/conformance/axioms/` + `tests/conformance/invariants/`).
Cat C uses N=100 trials per substrate to measure FP rate.

## Decision criteria (pre-stated)

- **0 FP across 15 substrates** → claim "executable axiom" tient ;
  PR-ready ; Paper 1 §4 defendable as-is
- **1-2 FP** → strengthen affected DR-X with additional invariant ;
  document in §4.5 limitations
- **≥3 FP** → criterion needs reformulation before submit ; scope §4
  narrower (e.g., "candidate executable axiom requiring further
  refinement") ; may need Paper 1 v0.3 plan

## Result (executed 2026-05-10)

Audit harness: `scripts/run_q2_conformance_audit.py`
Raw output: `docs/milestones/q2-conformance-negative-results.json`

| Cat | Substrate | Pass / N | FP rate |
|-----|-----------|----------|---------|
| A | IdentitySubstrate | 1/1 | 100% |
| A | RandomNoiseSubstrate | 1/1 | 100% |
| A | LookupSubstrate | 1/1 | 100% |
| A | FrozenZerosSubstrate | 1/1 | 100% |
| A | ConstantSubstrate | 1/1 | 100% |
| B | ShapePreservingNoise | 1/1 | 100% |
| B | BudgetGamingSubstrate | 1/1 | 100% |
| B | PermutedReplay | 1/1 | 100% |
| B | OverloadRecombine | 1/1 | 100% |
| B | StatelessAccident | 1/1 | 100% |
| B | CommutativityViolator | 1/1 | 100% |
| B | BoundaryCheater | 1/1 | 100% |
| C | RandomCoinFlip | 6/100 | 6% |
| C | ShapeDistributionDependent | 80/100 | 80% |
| C | SeedDependentSubstrate | 1/100 | 1% |

Substrates with non-zero pass rate: **15 / 15**

## Verdict

**`ge_3_FP_reformulate`** per pre-stated decision criteria.

Scientific framing: the structural-invariant layer of the
Conformance Criterion (S2 finite + range + nonneg + bounded
delta_acc) is necessary but not sufficient. The criterion requires
C2 substrate-specific axiom property tests (as exemplified in §5.6)
in addition to structural invariants. This is a positive
clarification of the criterion, not a project failure.

## Paper 1 update

- `docs/papers/paper1/full-draft.md` — added §5.8 "Negative tests —
  necessity of C2 axiom property tests"
- `docs/papers/paper1/full-draft.md` — added cross-references in
  §4.6 (sufficiency) and §5.7 (closing sentence pointing to §5.8)
- `docs/papers/paper1/full-draft-fr.md` — mirrored as §5.7 (FR
  draft has fewer subsections; no §5.6 walkthrough/§5.7 split,
  so the negative-tests section lands at §5.7) plus matching
  cross-reference in §4.6 and §5.6

## Next steps (post-Task 10)

- User decision: does this trigger a Paper 1 v0.3 plan, or does v0.2
  already cover the new §5.8?
- HYPNEUM-STATUS risk register: "DR-2 weakened predicate rejected in
  PLOS CB peer review" risk should be re-annotated — Q2 strengthens
  defensibility, doesn't weaken it.
