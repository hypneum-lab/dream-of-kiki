<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : dreamOfkiki project contributors
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# Introduction (Paper 2, draft C2.13)

**Authorship byline** : *dreamOfkiki project contributors*
**License** : CC-BY-4.0

**Target length** : ~1 page markdown (≈ 900 words)

---

## 1. From a single substrate to a Conformance claim

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

## 2. The cycle-2 goal : cross-substrate replication

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
   placeholder row for a future third substrate) and back every
   cell with a concrete test artifact
   (`tests/conformance/axioms/` and
   `tests/conformance/invariants/`). The matrix lives at
   `docs/milestones/conformance-matrix.md` and is regenerated
   deterministically by `scripts/conformance_matrix.py`.
3. **A cross-substrate H1-H4 replication.** The cycle-1 pre-
   registered statistical chain (Welch / TOST / Jonckheere /
   one-sample t under Bonferroni α = 0.0125) is re-run *per
   substrate* by `scripts/ablation_cycle2.py`, producing the
   comparative table in `docs/milestones/cross-substrate-
   results.md`. The runner, the predictor bridge, and the
   statistical module are shared across substrate labels —
   which is precisely how a substrate-agnostic framework is
   supposed to behave.
4. **An engineering architecture fit for replication.** Paper 2
   documents the async dream worker (C2.17), the 3-profile
   pipeline (P_min / P_equ / P_max), the swap protocol with
   S1 / S2 / S3 / S4 guards, the run registry with 32-hex
   R1 determinism contract, and the DualVer tag
   `C-v0.6.0+PARTIAL` recording the formal-axis bump for the
   E-SNN Conformance extension.

## 3. Scope honesty : synthetic substitute throughout

Paper 2 is a **methodology / replication paper**, not a fresh
empirical claim paper. This framing is non-negotiable : the two
substrate rows in the cross-substrate matrix share the same
Python mock predictor. A real substrate-specific predictor is a
cycle-3 deliverable (`docs/milestones/g9-cycle2-publication.md`
§ cycle-3 amorçage ; the user scoped Phase 3 + 4 as explicitly
synthetic-only for this cycle). Consequently :

- All H1-H4 p-values in §7 are flagged *(synthetic substitute)*.
- The cross-substrate agreement verdict is trivially YES by
  construction — identical predictor → identical sample → identical
  p-value. The **test value** is that the pipeline (runner →
  stats → dump → Markdown) executes end-to-end on two
  structurally distinct substrate registrations, which is the
  *architectural* half of DR-3 Conformance.
- No biological, neuromorphic, or hardware-performance claim
  is made in Paper 2. Those claims belong to cycle 3 (real
  Loihi-2 mapping, real fMRI cohort, divergent-predictor
  replication).

Research-discipline rule §3 from the repo `CLAUDE.md` — *never
report synthetic results as empirical claims* — is the guardrail.
Every table caption in §7, every p-value cell, every verdict
row is required to carry the `(synthetic substitute)` tag or an
equivalent inline flag. Readers who elide that tag are reading
the paper wrong.

## 4. Differentiation from Paper 1 and roadmap

Paper 1 was theoretical : axioms + proofs + formal definition of
Conformance. Paper 2 is engineering : operational substrates +
reusable test suite + replication runner + synthetic-substitute
evidence that the framework is *ready* to be replicated on real
substrates once real predictors are wired. The two papers share
authorship, licensing (MIT code, CC-BY-4.0 docs), pre-
registration (OSF, inherited from Paper 1), and the DualVer
lineage.

The remainder of the paper is organized as follows. §3
background situates Paper 2 relative to Paper 1 and the four
pillars (short, because Paper 1 already did the heavy lifting).
§4 reports the Conformance Criterion evidence on both
substrates with matrix citations. §5 lays out the engineering
architecture (2 substrates, 3 profiles, 4 ops, 8 primitives,
async worker). §6 details the methodology (pre-registered H1-H4
chain, Bonferroni, synthetic-substitute predictor details,
R1 contract). §7 reports the cross-substrate results with every
caption flagged. §8 discusses limitations honestly — what
synthetic-substitute data does and does not imply. §9 outlines
cycle-3 follow-up : real Loihi-2, fMRI cohort, Paper 3
emergence path.

---

## Notes for revision

- Insert proper bibtex citations once `references.bib` is
  extended from Paper 1's stub (S20-S22 cross-cycle).
- Cross-reference §3..§9 line numbers once full draft is laid
  out in NeurIPS style file.
- Tighten to ≤ 900 words before NeurIPS submission (current
  ~ 820).
