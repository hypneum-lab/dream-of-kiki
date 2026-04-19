# Abstract (Paper 1, draft)

**Word count target** : 250 words

---

## Draft v0.1 (S17.2, 2026-04-18)

Catastrophic forgetting remains a central obstacle for artificial
cognitive systems learning sequentially across tasks. Sleep-inspired
consolidation has been proposed as a remedy, with prior work
exploring replay (Walker, van de Ven), synaptic homeostasis
(Tononi), creative recombination (Hobson), and predictive coding
(Friston) — but no unified formal framework integrates these four
pillars into composable, substrate-agnostic operations.

We introduce **dreamOfkiki**, a formal framework with executable
axioms (DR-0 accountability, DR-1 episodic conservation, DR-2
compositionality on a free semigroup of dream operations, DR-3
substrate-agnosticism via a Conformance Criterion, DR-4 profile
chain inclusion). The framework defines 8 typed primitives (α, β,
γ, δ inputs ; 4 output channels), 4 canonical operations (replay,
downscale, restructure, recombine), and a 5-tuple Dream Episode
ontology. The framework admits multiple conformant substrates ;
exemplar implementations validate the design and are reported
separately (see Paper 2).

Pre-registered hypotheses (OSF DOI : 10.17605/OSF.IO/Q6JYN) are evaluated via
Welch's t-test, TOST equivalence, Jonckheere-Terpstra trend, and
one-sample t-test against compute budget under Bonferroni
correction.

**Pipeline Validation (synthetic placeholder, G2 pilot).** The
end-to-end measurement and statistical pipeline is exercised with
mock predictors at scripted accuracy levels ; numbers are
reported in §7 alongside their registered run_id and JSON dump
under `docs/milestones/`. Real mega-v2 inference and any fMRI
representational similarity analysis follow in cycle 2 (Paper 2).
All code, specifications, and pre-registration are open under
MIT/CC-BY-4.0.

---

## Notes for revision

- Replace synthetic results with real ablation numbers post-S20+
- Insert OSF DOI once locked (currently pending action)
- Insert Zenodo DOI for code+model artifacts at submission tag
- Tighten to ≤250 words (current ~265)
