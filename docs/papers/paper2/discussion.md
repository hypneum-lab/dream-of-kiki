<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : dreamOfkiki project contributors
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# §8 Discussion (Paper 2, draft C2.16)

**Authorship byline** : *dreamOfkiki project contributors*
**License** : CC-BY-4.0

**Target length** : ~1 page markdown (≈ 900 words)

---

## 8.1 What cross-substrate convergence means in Paper 2

The headline of Paper 2 is a **structural convergence**, not a
numerical surprise. Both substrates — MLX `kiki_oniric` and
E-SNN `thalamocortical` — pass the three DR-3 Conformance
conditions (C1 typed Protocols, C2 axiom property tests, C3
BLOCKING invariants enforceable, §4). Both substrates are
exercised through the same ablation runner, the same pre-
registered H1-H4 chain, and the same Bonferroni α = 0.0125
(§6). Both emit the same 4 / 4 hypothesis verdicts (§7.3).

The right reading is : *the framework's Conformance Criterion
is operational on two structurally independent implementations
of its 8 primitives.* The wrong reading is : *the framework is
empirically substrate-agnostic.* The two readings are distinct,
and the distinction rests on the synthetic-substitute predictor
documented in §6.4.

## 8.2 What synthetic-substitute implies for claims

**(synthetic substitute — not empirical claim.)**  The two
substrate rows share the same Python mock predictor. Concretely,
this means :

- The H1-H4 p-values (§7.2) are trivially identical across
  substrates in the limit of a perfect shared predictor.
- The agreement verdict (§7.3, 4 / 4 agree) is trivially YES
  by construction.
- The verdict *pattern* (3 reject, 1 fail-to-reject) mirrors
  Paper 1 §7.7 because the mock predictor is the same. There is
  no independent E-SNN signal in the numbers.

What this means for claims :

- We **do not claim** that MLX and E-SNN produce equivalent
  behaviour on real data. We claim that the pipeline emits
  equivalent verdicts when the predictor is shared, which is a
  necessary condition for DR-3 Conformance but not a sufficient
  condition for substrate-agnostic empirical efficacy.
- We **do not claim** a neuromorphic hardware advantage. The
  E-SNN substrate is a numpy LIF skeleton ; Loihi-2 hardware
  has not been exercised.
- We **do claim** the *architectural* half of DR-3 : the
  framework's typed surfaces and axiom / guard suites are
  substrate-generic, as evidenced by two structurally
  independent substrate registrations passing the same test
  battery without code duplication.

Reviewers assessing Paper 2 should read the cross-substrate
agreement as **pipeline evidence**, not **biological** or
**neuromorphic** evidence. Any reader who elides the
`(synthetic substitute)` tag is reading Paper 2 wrong.

## 8.3 Limitations (honest enumeration)

Four limitations bound the cycle-2 contribution :

**(i) Shared predictor across substrates.** This is the
dominant limitation. A divergent-predictor replication —
substrate-specific inference on each substrate's native state —
is the cycle-3 target. Until it lands, Paper 2 cannot carry
a substrate-agnostic *performance* claim.

**(ii) Only two real substrates + one placeholder.** DR-3
Conformance is an inductive claim ; two is a beginning, not a
ceiling. The `hypothetical_cycle3` row in the conformance
matrix is explicitly marked `N/A`. Candidate cycle-3 substrates
include transformer-based instances, SpiNNaker / Norse
mappings, and — pending Intel NRC partnership — a real Loihi-2
deployment.

**(iii) Synthetic mega-v2 benchmark.** The 500-item retained
benchmark (§6.5) is a synthetic stratified fallback inherited
from Paper 1. Real mega-v2 inference is deferred to a later
cycle. Consequently, §7's p-values carry no claim about real
linguistic consolidation.

**(iv) Paper 1 publication sequencing risk.** Paper 2's
narrative depends on Paper 1 (framework + axioms + DR-3). If
Paper 1 acceptance is delayed or a major revision is required,
Paper 2 should either cite the arXiv preprint or adopt a
Pivot B contingency (a self-contained recap of the framework,
re-compressed from Paper 1 §4 + §5). Neither the arXiv ID nor
the HAL FR deposit is currently locked
(`docs/milestones/g9-cycle2-publication.md` § external actions).

## 8.4 The T-Col pivot to real fMRI data

Paper 1 §6.4 locked Studyforrest as the cycle-1 fMRI fallback
(Branch A). The cycle-2 T-Col (Tower-Cologne) outreach pursued
an active lab partnership — Huth Lab (UT Austin), Norman Lab
(Princeton), Gallant Lab (UC Berkeley) — to unlock
**task-controlled** linguistic stimuli beyond the narrative-
comprehension Studyforrest provides. The partnership would
strengthen H3 (monotonic representational alignment), which
reached only borderline significance in cycle-1 and failed at
Bonferroni in cycle 2 (§7.2) due to the mock predictor's
constant dispersion.

The T-Col pivot is a **cycle-3 deliverable**. For Paper 2's
purposes, the pivot is the mechanism by which future cycles
replace synthetic-substitute data with real biological signal ;
this paper does not claim any such data.

## 8.5 Comparison with Paper 1 discussion §8.5

Paper 1 §8.5 added a retro-active cross-substrate preliminary
replication paragraph pointing at cycle-2 artifacts. Paper 2 is
the full narrative of that paragraph. Three things are
coherent across the two papers :

- Paper 1 §8.5 flags the cross-substrate verdict as *trivially
  agreeing by construction* ; Paper 2 §7.3 and §8.2 double
  down on the same framing.
- Paper 1 §8.5 notes that the synthetic-substitute evidence
  *strengthens but does not substitute for* cycle-1 H1-H4 ;
  Paper 2 §8.1 restates this as the architectural-vs-empirical
  distinction.
- Paper 1 §8.3 (iii) flags P_max skeleton ; Paper 2 §5.2
  records that P_max is now fully wired (G8) and §7.2 reports
  the first H2 and H3 runs under a real three-group structure
  — even though the numerical verdict is shared-predictor
  limited.

No contradiction exists between Paper 1 and Paper 2 claims ;
the two papers are consistent under the synthetic-substitute
framing.

## 8.6 Engineering trade-offs (MLX vs E-SNN)

Cycle 2 exercises MLX and E-SNN under the same pipeline but
does not benchmark them head-to-head on energy or latency.
That benchmark is not possible under the shared-predictor
setup : any wall-clock or energy-ratio number would be
dominated by the predictor, not the substrate. When a
divergent-predictor replication lands in cycle 3, the trade-
off space becomes measurable :

- **MLX** optimises for sequential gradient-style updates and
  is the natural target for Apple Silicon / CUDA deployments.
- **E-SNN** (on real Loihi-2 hardware) optimises for
  event-driven sparse computation and is the natural target
  for neuromorphic research deployments.

A substrate-agnostic framework accepts both without demanding
one dominate. Paper 2's value here is that it **makes the
comparison possible** — the registry, the ablation runner,
the conformance matrix, the async worker are all in place.
The comparison itself is a cycle-3 paper (Paper 3 emergence,
per `docs/milestones/g9-cycle2-publication.md` § cycle-3
amorçage).

---

## Notes for revision

- Tighten to ≤ 800 words at NeurIPS pre-submission pass.
- Cross-reference §9 future work once §9 lands.
- If Paper 1 arXiv ID lands before submission, replace
  `Paper 1 §X.Y` references with proper `\cite{}` calls.
