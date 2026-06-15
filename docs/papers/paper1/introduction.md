<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : Saillant, Clément
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# Introduction (Paper 1, draft)

**Authorship byline** : *Saillant, Clément*
**License** : CC-BY-4.0

**Target length** : ~1.5 pages markdown (≈ 1200 words)

---

## 1. Catastrophic forgetting and the consolidation gap

Modern artificial cognitive systems excel at single-task learning
but degrade rapidly when trained sequentially across tasks — a
phenomenon known as **catastrophic forgetting**
[@mccloskey1989catastrophic; @french1999catastrophic]. Despite two
decades of mitigation strategies (elastic weight consolidation
[@kirkpatrick2017overcoming], generative replay
[@shin2017continual], rehearsal-based memory [@rebuffi2017icarl]),
the field still lacks a *unified theoretical account* of why these
mechanisms work and when they should compose. Recent surveys of
the continual-learning literature identify latent replay as the
emerging consensus mechanism across five years of methods
[@vandeven2024clmonograph], yet do not derive that consensus from
first principles.

Biological cognition solves this problem during **sleep**.
Hippocampal replay during NREM, synaptic downscaling, predictive
restructuring of cortical representations, and creative
recombination during REM together form a multi-stage
consolidation pipeline [@diekelmann2010memory; @tononi2014sleep;
@robertson2025nrn]. Yet existing AI work has integrated only
fragments of this biology, typically focusing on a single
mechanism (e.g., replay alone) without a principled account of
how mechanisms interact.

## 2. Four pillars of dream-based consolidation

We identify four theoretical pillars that any complete
dream-inspired AI consolidation framework must address :

- **A — Walker/Stickgold consolidation** : episodic-to-semantic
  transfer via replay [@walker2004sleep; @stickgold2005sleep].
- **B — Tononi SHY** : synaptic homeostasis renormalizing weights
  during sleep [@tononi2014sleep].
- **C — Hobson/Solms creative dreaming** : recombination and
  abstraction during REM [@hobson2009rem; @solms2021revising].
- **D — Friston FEP** : minimization of free energy as a unifying
  account of inference and consolidation [@friston2010free].

Prior AI work has implemented A [@vandeven2020brain;
@spens2024generative], B [@kirkpatrick2017overcoming as a
SHY-adjacent regularization], and elements of D
[@rao1999predictive; @whittington2017approximation], but **no work
has formalized how the four pillars compose** in a
substrate-agnostic manner amenable to ablation and proof.

## 3. The compositional gap

Why does composition matter ? Empirically, the order in which
consolidation operations apply changes the resulting cognitive
state — replay before downscaling preserves episodic specificity,
while downscaling before restructuring may erase the very
representations that restructuring is meant to refine. Our
analysis (`docs/proofs/op-pair-analysis.md`) enumerates the 16
op-pairs and finds 12 cross-pairs are non-commutative, reinforcing
that *order is part of the framework*, not an implementation
detail.

A proper formal framework must therefore (i) specify the
operations as composable primitives with well-defined types, (ii)
make explicit which compositions are valid, (iii) provide an
**executable** account that any compliant substrate can implement,
and (iv) support empirical ablation comparing different operation
profiles. None of the prior art does all four.

Three research communities have converged on the same architectural
pattern from independent directions. **Continual learning** has
settled on latent replay as the dominant mitigation for catastrophic
forgetting [@vandeven2024clmonograph]. **Industry LLM research** is
reinventing the construct as a metaphor: Berkeley's *sleep-time
compute* reports +13–18 % accuracy and 5× compute amortisation by
pre-processing context offline, but makes no explicit reference to
biological sleep [@berkeley2025sleeptimecompute]; Google Research's
Titans introduces a test-time neural memory module learnt at
inference up to 2 M tokens, without principled grounding in
consolidation theory [@behrouz2024titans]. **Concurrent academic
proposals** are beginning to close that gap — *Wake-Sleep
Consolidated Learning* [@alfarano2024wakesleep] is, to our
knowledge, the closest published NREM/REM dual-phase analog, and
the contemporaneous *Language Models Need Sleep* proposal couples
RL-based upward distillation with intentional forgetting
[@iclr2026lmsleep]. Neuromorphic substrates show the same
convergence: CLP-SNN on Loihi 2 reports 70× speed and 5,600×
energy efficiency over GPU baselines for offline replay
[@hajizada2025clpsnn]. Three communities, one architectural
shape, and **no published 2024–2026 work formally maps the
SO–spindle–ripple triad onto a substrate-agnostic computational
invariant** : neuroscience has the mechanism, AI has the function
(replay → no-forgetting), neither yet has the axiomatic bridge.
Framework C is positioned at that bridge, with the four pillars
above as conceptual anchors, the closest empirical analog
[@alfarano2024wakesleep] retained as Paper 2's primary ablation
comparator, and the concurrent proposals
[@iclr2026lmsleep; @berkeley2025sleeptimecompute] read as
independent corroboration of the underlying need rather than as
prior art on the formal contribution.

## 4. Contribution roadmap

In this paper we present **dreamOfkiki**, the first formal
framework for dream-based consolidation in artificial cognitive
systems with the following contributions :

1. **Framework C-v0.5.0+STABLE** : 8 typed primitives, 4 canonical
   operations forming a non-commutative semigroup (semantically independent generators), 4 OutputChannels, 5-tuple
   Dream Episode ontology, axioms DR-0..DR-4 with executable
   Conformance Criterion (§4). Items 2–4 below are reported in
   Paper 2 (empirical companion) ; Paper 1 confines itself to the
   formal contributions and the conformance roadmap.
2. **Roadmap** to substrate generalization (additional
   substrates beyond cycle-1's reference implementation) and
   real fMRI representational alignment (real lab partnership
   pursued via T-Col outreach).

The remainder of the paper is organized as follows : §3 reviews
the four pillars in depth ; §4 develops Framework
C-v0.5.0+STABLE with axioms and proofs ; §5 sketches the
Conformance Criterion validation approach (per-substrate
empirical results live in Paper 2) ; §6 details the methodology ;
§7 reports the synthetic pipeline-validation results ; §8
discusses implications and limitations ; §9 outlines cycle-2
future work.

---

## Notes for revision

- Insert proper bibtex citations once reference manager is set up
- Cross-reference §3-§9 line numbers once full paper is laid out
  in target journal template
- Tighten to ≤1500 words for Nature HB main text discipline
