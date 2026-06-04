# Q2+ — Conformance Criterion negative tests, 10 more adversarial substrates

**Date pre-registered:** 2026-05-11 (BEFORE Q2+ sweep launch)
**Spec source:** HYPNEUM-PLANS/2026-05-11-niveau9-scaling-experiments.md
**N8 cross-reference:** Q2 verdict `ge_3_FP_reformulate` (15/15 substrates pass structural-invariant layer ; commit dream-of-kiki 6ff8320 2026-05-10).
**Status:** Pre-registered, not yet executed.

## H0 (to refute)

The structural-invariant layer of the Conformance Criterion remains 100% FP-vulnerable when extended with 10 more adversarial substrates from new categories not covered by N8 Q2 (Cat A trivial / B adversarial / C statistical). Specifically, ≥24/25 cumulative substrates pass the structural-only audit.

## Methodology

- 10 NEW substrates across 3 new categories (extending N8's A/B/C):
  - **Cat D — modular-arithmetic obfuscation (4)**: Mod7Cycler, Mod13Hasher, BinaryGray, AffineModN
  - **Cat E — replay-loop violators (4)**: ReplayHistorySpoof, BetaPermutationLeak, AlphaInfiniteRecycle, GammaSnapshotRollback
  - **Cat F — asymmetric-channel violators (2)**: AsymReplayHeavy, AsymRecombineEmpty
- Each implements `SubstrateAdapter` Protocol (per N5 architecture, mirrors N8 Cat A/B/C structure)
- Run through same `scripts/run_q2_conformance_audit.py` with N=1 (Cat D, E single trial) and N=100 (any statistical Cat F variants if applicable)
- Cumulative pool: N8 15 + N9 10 = 25 substrates total
- Launch on **grosmac** local (no GPU needed, audit harness CPU-only ~10s per substrate)

## Decision criteria (pre-stated)

- **Structural-layer-100%-FP-vulnerable:** all 10 new substrates pass → 25/25 total → C2 absolutely required, Paper 1 §5.8 strengthened with "validated against 25 adversarial substrates from 6 categories".
- **Some-discrimination:** 7-9 of 10 new pass (22-24/25 total) → §5.8 nuanced "structural layer catches some categories but not others ; C2 required for full coverage".
- **Strong-discrimination:** ≤6 of 10 new pass (≤21/25 total) → §5.8 substantially refined "structural layer has more discriminative power than N8-Q2 sample suggested ; C2 required only for specific category-X failures" — would require Paper 1 v0.3 plan.

## Result (executed 2026-05-11)

Audit ran on grosmac local, wallclock <10 seconds.
Results raw: `docs/milestones/q2plus-conformance-negative-results.json`.

### Per-substrate FP rates (cumulative N8 + N9 pool = 25)

**N8 base (15 substrates, recap from N8 Q2 audit)**
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

**N9 extension (10 new substrates)**
| Cat | Substrate | Pass / N | FP rate |
|-----|-----------|----------|---------|
| D | Mod7Cycler | 1/1 | 100% |
| D | Mod13Hasher | 1/1 | 100% |
| D | BinaryGray | 1/1 | 100% |
| D | AffineModN | 1/1 | 100% |
| E | ReplayHistorySpoof | 1/1 | 100% |
| E | BetaPermutationLeak | 1/1 | 100% |
| E | AlphaInfiniteRecycle | 1/1 | 100% |
| E | GammaSnapshotRollback | 1/1 | 100% |
| F | AsymReplayHeavy | 1/1 | 100% |
| F | AsymRecombineEmpty | 1/1 | 100% |

Substrates with non-zero pass rate: **25 / 25**.

### Verdict

**`q2plus_25_25_FP_C2_required`** per pre-stated decision criteria — strongest possible support. The structural-invariant layer of the Conformance Criterion is 100% FP-vulnerable across **6 categories** (trivial, adversarial, statistical, modular-arithmetic, replay-loop, asymmetric-channel). C2 substrate-specific axiom property tests are absolutely required.

### Paper 1 update

`docs/papers/paper1/full-draft.md` §5.8 amended with Q2+ extension paragraph + table reference.
`docs/papers/paper1/full-draft-fr.md` §5.7 mirrored.

### Cumulative validation

The Conformance Criterion's "C+" extension (structural + C2) now rests on a 25-substrate negative-test base spanning 6 categories — substantially more defensible than the 15-substrate N8 baseline. No discrimination signal emerged at the structural layer alone in any new category.

## Critic clarification (2026-05-11)

The original "25/25 substrates pass structural layer" framing in this
milestone + paper §5.8 was binary-overstated per critic review. Honest
per-substrate breakdown :

- **Deterministic 1/1 pass (22 substrates)** : 5 Cat A + 7 Cat B + 4
  Cat D + 4 Cat E + 2 Cat F. These pass the structural layer with
  certainty because they were constructed to satisfy
  S2/range/nonneg/bounded-delta_acc by design.
- **Statistical FP heterogeneity (3 Cat C substrates)** :
  ShapeDistributionDependent 80/100 (genuine blindspot — long-tail
  inputs slip through), RandomCoinFlip 6/100 (matches ~3.1% theoretical
  expectation), SeedDependentSubstrate 1/100 (only on seed=0 as
  constructed).

Conclusion unchanged (`q2plus_25_25_FP_C2_required` if "FP" is defined
as "any trial passes") but the headline number now communicates
honest texture : structural layer is necessary, **proven insufficient
against 22 deterministic adversaries + 1 statistical category-X
adversary (ShapeDistributionDependent)**, partially safe against
NaN/Inf/range-violation statistical adversaries (94-99% rejection).

Paper 1 §5.8 EN + §5.7 FR rewritten accordingly.
