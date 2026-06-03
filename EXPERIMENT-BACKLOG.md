# Experiment backlog — dream-of-kiki Paper 1

From the external review (REVIEW.md). Run in a fresh session with the repo env
(`uv sync --all-extras`). R1 `run_id` + DualVer EC bump on any claim change.
**Heavy items need the Mac Studio M3 Ultra + KIKI-Mac_tunner mega-v2 pipeline — not
runnable from an orchestration session.**

## E1 — Powered TOST on real predictors  [HEAVY: real mega-v2 35B]
**Why**: the headline coupling NULL is unsupported — NHST cannot support the null
(Lakens 2017); the pre-specified TOST is smoke-only (results-section.md 7.6); all §7 cells
are synthetic mocks (50/70/85).
**Protocol**:
1. Pre-specify equivalence bounds (use the pre-reg floor g=0.29 as the SESOI) + a power
   analysis: n per arm for 80% power at α=0.0167 (Bonferroni family).
2. Replace synthetic predictors with **real mega-v2** outputs (cortical-update accuracy).
3. Run the TOST; register run_id.
**Acceptance**: a powered, pre-registered TOST that can actually *support* (not just fail
to reject) the coupling NULL.

## E2 — Replace §7 synthetic cells with real mega-v2  [HEAVY]
Run the real-LLM cells, test against the g=0.29 pre-registered floors; the cross-substrate
replication must NOT share a single mock predictor (current agreement is true by
construction).

## E3 — Scaling-law third scale point + CIs  [HEAVY]
Add a 3rd scale point beyond {1.5B, 7B, 35B}; bootstrap CIs on per-seed p-values;
numerical-precision / log-prob underflow audit.

## E4 — Formal: freeness vs restatement  [analysis, opus/human]
DR-2 proof §7 establishes only **semantic irreducibility of `recombine`** (recombine ∉
φ(L_B)), NOT syntactic freeness of the 4-generator semigroup. Either: (a) complete the
syntactic freeness proof (no non-trivial relations among all four generators), or (b)
**restate paper §4.4/4.5** from "free semigroup on four generators" to "semantically
independent generators (recombine irreducible)". Resolve the §12 internal inconsistency.
