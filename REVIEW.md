# External reviewer critique — dream-of-kiki Paper 1

> Status: external review (deep-research harness + opus formal audit), 2026-06-03.
> Leads to confirm before amending the paper; some workflow verification was flaky.
> Target venue under review: PLOS Computational Biology.

## Verdict
Strong formal scaffolding and **exemplary honesty**, but **overstated novelty** and
**zero real empirical content**. Reframe as a methodology/formal paper; defer empirical
claims until real predictors replace the synthetic placeholders.

## ① Novelty (overstated)
- CLS wake/NREM/REM framing is **prior art**: WSCL (Alfarano 2024, arXiv 2401.08623)
  already does CLS-phased consolidation with a positive REM forward-transfer result.
- Four ops + NREM/REM split near-isomorphic to Robinson 2022 (arXiv 2209.05245),
  **not cited**. Defensible delta = substrate-agnostic axiomatization + Conformance
  Criterion + DR-3 only. §8.4 omits WSCL from the differentiation table.

## ② Methodological soundness
- **Headline NULL is unsupported**: NHST cannot support the null (Lakens 2017); the
  pre-specified TOST is unexecuted (smoke-only, results-section.md 7.6). Need a powered
  TOST on real predictors.
- **All §7 results are synthetic mock predictors** (accuracy 50/70/85). Cross-substrate
  replication shares one mock predictor → agreement is **true by construction**
  (discussion.md 8.5, results 7.7 CAVEAT). No positive empirical claim exists.

## ③ Formal audit — DR-2 proof (dr2-compositionality.md v0.2)
- Genuinely-proved content is **thin**: (B)/(F)/(A) are trivial (physical accounting /
  tautologies of function composition); the only non-trivial part, (C) closure, is
  **empirical** (24-perm MLX test), not a proof.
- 🔴 Precondition **[P] is substrate/schema-specific** (MLX `addmm`, shape (2,4)),
  "vacuous" on other substrates → the one non-trivial result is an implementation
  artifact, in tension with the substrate-agnosticism thesis. Is [P] structural or a bug?
- 🔴 **§7 proves semantic irreducibility of `recombine` only, NOT freeness.** Paper §4.4/4.5
  claims "free semigroup on four generators"; the proof establishes only `recombine ∉
  φ(L_B)`; the other generators' mutual irreducibility is "expected… not proved". The
  free-semigroup claim is unsupported (and §12 flags an internal "we do not claim
  freeness" inconsistency).
- `sha256(H)` invariant is contingent on the write-domain design choice (authors' own Q1);
  no mechanization (Lean/Coq); the DR-2' fallback (§9) signals low confidence in §7.

## ④ Threats to validity
- Biological grounding rests on a physiology→behaviour dissociation: CLAS enhances
  oscillations/spindles but repeatedly fails to improve memory (Henin 2019 eNeuro
  PMC6831893; Harrington 2021). Discussion 8.6 concedes it.
- Replay is sparse/selective: only 10–30% of SWRs carry replay; large SWRs causally drive
  reactivation (Fernández-Ruiz, Neuron 2025). Don't treat replay as a uniform channel.
- Continual-learning scenario unspecified (task/domain/class-IL differ — van de Ven 2022,
  Nat Mach Intell); mega-v2 may be domain-incremental and inflate gains.

## Open questions
1. Will real mega-v2 clear the pre-registered effect-size floors (g=0.29), or never fire
   given the dissociation literature?
2. Defend "first framework" vs WSCL beyond substrate-agnosticism.
3. Restate §4.4/4.5 to claim only what DR-2 v0.2 proves (semantic irreducibility, not
   freeness).
