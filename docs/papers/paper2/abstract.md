<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : dreamOfkiki project contributors
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# Abstract (Paper 2, draft C2.13)

**Authorship byline** : *dreamOfkiki project contributors*
**License** : CC-BY-4.0

**Word count target** : ≤ 200 words

---

## Draft v0.1 (C2.13, 2026-04-19)

> **⚠️ Synthetic substitute — methodology / replication paper.**
> Every empirical table, figure, and verdict reported below is
> produced by a shared Python mock predictor wired across two
> substrate registrations (MLX + E-SNN numpy LIF skeleton). No
> Loihi-2 hardware and no fMRI cohort participate in any result.
> Paper 2 documents the **cross-substrate replication pipeline**,
> not a fresh empirical claim ; headline biological or
> neuromorphic claims are deferred to cycle 3.

Paper 1 introduced Framework C-v0.6.0+PARTIAL (axioms DR-0..DR-4,
8 typed primitives, 4 canonical operations, 3 profiles) and its
executable Conformance Criterion (DR-3). Paper 2 asks the
engineering question Paper 1 could not : *does a single compliant
substrate constitute evidence for substrate-agnosticism ?*

We answer by replicating the cycle-1 pre-registered H1-H4 pipeline
across two registered substrates — MLX kiki-oniric on Apple
Silicon and a numpy LIF spike-rate E-SNN thalamocortical skeleton
— and by exhibiting a two-substrate DR-3 Conformance matrix
(3 conditions × 2 substrates, all PASS ; a third `hypothetical_
cycle3` row stays N/A). Under Bonferroni α = 0.0125, both
substrates agree on 4 / 4 hypotheses (H1 reject, H2 reject, H3 fail
to reject, H4 reject) — *synthetic substitute ; the agreement is
trivial by shared-predictor construction and validates the
pipeline, not the framework's empirical efficacy on real data*.
All code, pre-registration, and run_ids are MIT / CC-BY-4.0.

---

## Notes for revision

- Word count currently ≈ 200 ; tighten if NeurIPS format
  constrains to 150.
- Insert OSF DOI (inherited from Paper 1) once locked.
- Insert Zenodo DOI for the cycle-2 artifact bundle at
  submission tag.
- Every pass through this abstract must preserve at least one
  `synthetic substitute` flag in the caveat block.
