# Abstract (Paper 1, draft)

**Word count target** : ≤ 250 words

---

## Draft v0.2 (2026-04-21 — biology-first reframe per PLOS CB scope fit)

Sleep-dependent memory consolidation is an empirically well-characterised
biological process in which freshly encoded experiences are selectively
strengthened, integrated, and reorganised during offline periods
[@walker2004sleep; @stickgold2005sleep; @tononi2014sleep]. Contemporary
reviews organise the phenomenon around three canonical processes :
active systems consolidation (replay-dominated), synaptic homeostasis
(SHY), and integration / abstraction
[@klinzing2019mechanisms; @rasch2025sleep]. Artificial cognitive
systems learning sequentially suffer from catastrophic forgetting, and
partial mitigations have drawn on individual biological processes, but
no unified formal framework has integrated them into composable,
substrate-agnostic operations instantiable across neural architectures.

We introduce **dreamOfkiki**, a formal framework that axiomatises this
space. Four canonical operations (`replay`, `downscale`, `restructure`,
`recombine`) refine the biological three-process framing by splitting
"integration" into two semantically independent generators ; the
resulting free non-commutative semigroup is characterised by five
executable axioms — DR-0 accountability, DR-1 episodic conservation,
DR-2 compositionality under an empirically derived precondition, DR-3
substrate-agnosticism via an executable Conformance Criterion, and
DR-4 profile chain inclusion. Eight typed primitives and a 5-tuple
Dream Episode ontology complete the framework ; the fourth generator's
irreducibility is proved via a write-domain invariant
(`docs/proofs/dr2-compositionality.md` §7).

The framework admits multiple conformant substrates ; two exemplar
implementations (MLX gradient-based runtime and E-SNN thalamocortical
spiking network) are reported in Paper 2. Pre-registered hypotheses
(OSF DOI `10.17605/OSF.IO/Q6JYN`) are evaluated via Welch's t-test,
TOST equivalence, Jonckheere-Terpstra trend, and one-sample t-test
under Bonferroni correction. The end-to-end pipeline is here exercised
on synthetic predictors (G2 pilot) ; real ablation follows in Paper 2.
All code, specifications, and pre-registration are open under MIT /
CC-BY-4.0.

---

## Notes for revision

- Word count this draft : ~255 ; trim by 5 for final submit
  (target ≤ 250).
- OSF DOI `10.17605/OSF.IO/Q6JYN` is **live** (DataCite-minted
  2026-04-19T00:28:05Z).
- Zenodo DOI for code + model artifacts : mint at submission tag
  `arxiv-v0.2`.
- v0.1 archived in git history (pre-2026-04-21 version :
  "Catastrophic forgetting remains a central obstacle for artificial
  cognitive systems..." — AI-first opening, retired per PLOS CB scope
  fit review 2026-04-21).
