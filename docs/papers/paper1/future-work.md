# Future Work — Cycle 2 (Paper 1, draft S19.2)

**Target length** : ~0.5-1 page markdown (≈ 700 words)

---

## 9.1 E-SNN substrate (Loihi-2 thalamocortical)

The most direct extension of cycle 1 is to validate the DR-3
Conformance Criterion on a second substrate : a thalamocortical
spiking neural network (E-SNN) deployed on Intel Loihi-2
neuromorphic hardware. This was deferred from cycle 1 per the
SCOPE-DOWN decision (master spec §7.3) to ensure cycle 1 closed
on time with a single-substrate validation.

The E-SNN substrate would test whether the framework's executable
axioms remain operational when the operations are realized as
spike-rate dynamics rather than gradient updates on dense
matrices. Successful conformance would provide the substrate-
agnosticism evidence that Paper 1 claims as a theoretical
property but does not yet demonstrate empirically across two
substrates.

## 9.2 P_max profile real wiring

Cycle 1 implements P_max only as a skeleton (`status="skeleton"`,
`unimplemented_ops=["recombine_full"]`). Cycle 2 will wire the
remaining components :

- **α-stream raw traces** input channel (currently P_max-only
  declared but not consumed) — requires firehose ring buffer
  with bounded retention
- **ATTENTION_PRIOR canal-4** output channel — requires the
  attention prior bounding invariant (S4) and downstream wiring
  to consumer modules
- **`recombine_full`** operation variant — full VAE encoder /
  decoder pair beyond the C-Hobson light interpolation skeleton

With P_max real-wired, hypothesis H2 (P_max equivalence vs P_equ
within ±5%) becomes a real comparison rather than the cycle-1
self-equivalence smoke test.

## 9.3 Real fMRI lab partnership

Cycle 1 locks Studyforrest as the fMRI fallback (G1 Branch A).
Cycle 2 pursues active partnership with one or more fMRI labs
identified via the T-Col reviewer outreach :

- **Huth Lab** (UT Austin) : Narratives dataset
- **Norman Lab** (Princeton) : episodic memory studies
- **Gallant Lab** (UC Berkeley) : naturalistic stimulus-driven
  BOLD

A real lab partnership would enable RSA on **task-controlled**
linguistic stimuli rather than the narrative-comprehension
fallback Studyforrest provides. This would strengthen H3
(monotonic representational alignment) which reached only
borderline significance in cycle 1.

## 9.4 Multi-substrate Conformance Criterion validation

The strongest theoretical claim of Framework C — substrate-
agnosticism via DR-3 Conformance Criterion — needs empirical
validation across more than two substrates to be defensible at
peer review. Cycle 2 establishes the validation matrix : for
each candidate substrate (kiki-oniric ✅, E-SNN, hypothetical
transformer-based instance), verify all three conformance
conditions (signature typing, axiom property tests passing,
BLOCKING invariants enforceable).

A reusable conformance test suite (drafted in cycle 1
`tests/conformance/`) is the foundation. Cycle 2 will extend it
with substrate-specific adapters and run the full suite against
each candidate substrate, producing a conformance report
publishable as a supplementary artifact for Paper 1 (or as the
main contribution of Paper 2's engineering ablation paper).

---

## Notes for revision

- Tighten to ≤700 words for Nature HB discipline
- Cross-reference cycle-2 plan documents once those are drafted
  (post-G6 cycle-2 decision report S28.1)
- Reorder subsections by priority once cycle-2 scope is locked
