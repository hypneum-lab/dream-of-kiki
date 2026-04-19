<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : dreamOfkiki project contributors
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# §3 Background (Paper 2, draft C2.16)

**Authorship byline** : *dreamOfkiki project contributors*
**License** : CC-BY-4.0

**Target length** : ~0.5 page markdown (≈ 500 words)

---

## 3.1 Paper 1 in one paragraph

Paper 1 introduced Framework C : 8 typed primitives (α, β, γ,
δ inputs + 4 output channels), 4 canonical dream operations
(replay, downscale, restructure, recombine) forming a free
semigroup under DR-2 compositionality, a 5-tuple Dream
Episode ontology, and axioms DR-0 (accountability), DR-1
(episodic conservation), DR-2 (compositionality), DR-3
(substrate-agnosticism via the Conformance Criterion), DR-4
(profile chain inclusion). Readers unfamiliar with the
framework should read Paper 1 §4 (axioms + proofs) and §5
(reference implementation) before §4 of this paper.

## 3.2 Four pillars, briefly

Paper 2 inherits Paper 1 §3's four-pillar mapping :

- **Pillar A — Walker / Stickgold** : episodic-to-semantic
  transfer via replay [@walker2004sleep; @stickgold2005sleep ;
  operationalized as `replay` in our framework].
- **Pillar B — Tononi SHY** : synaptic homeostasis / weight
  downscaling [@tononi2014sleep ; operationalized as
  `downscale`, commutative but non-idempotent].
- **Pillar C — Hobson / Solms** : creative recombination in
  REM dreaming [@hobson2009rem; @solms2021revising ;
  operationalized as `recombine`, VAE-light in cycle 1,
  full VAE in cycle 2].
- **Pillar D — Friston FEP** : free-energy minimization
  [@friston2010free ; operationalized as `restructure` under
  the S3 topology guard].

Paper 2 does not re-argue these mappings. It consumes them.

## 3.3 DR-3 Conformance Criterion, briefly

DR-3 specifies the three conditions (C1 signature typing via
typed Protocols, C2 axiom property tests pass, C3 BLOCKING
invariants enforceable) that a candidate substrate must
satisfy to inherit the framework's guarantees. The criterion
is executable : it ships as a reusable pytest battery in
`tests/conformance/` parameterized on the substrate's state
type. §4 of this paper exercises DR-3 on two substrates +
one placeholder.

## 3.4 Prior multi-substrate replication work in continual learning

Cross-substrate replication of continual-learning mechanisms
is under-represented in the prior art. [@vandeven2020brain]
demonstrate brain-inspired replay on artificial neural
networks but on a single architecture. [@kirkpatrick2017overcoming]
introduce EWC as a SHY-adjacent regularizer but do not pursue
substrate-agnosticism. [@rebuffi2017icarl] and
[@shin2017continual] are architecture-specific. No prior work,
to our knowledge, ships a **reusable conformance test suite**
binding a formal framework to operational substrates across
qualitatively distinct hardware models (dense-tensor MLX vs
spiking LIF).

This is Paper 2's distinguishing engineering contribution.

## 3.5 Synthetic-substitute framing, briefly

Paper 2 is a **methodology / replication paper**, not a fresh
empirical claim paper. This framing is non-negotiable : the
two substrate rows share the same Python mock predictor
(§6.4). A divergent-predictor replication on real data is a
cycle-3 target. Readers who elide the synthetic-substitute
tag misread the paper's scope. The introduction §2 and the
methodology §6.4 re-state this framing at strength ; §3 is
the last place a reader can skip the caveat before the
results tables of §7.

---

## Notes for revision

- Insert proper bibtex citations once `references.bib` is
  extended from Paper 1's stub.
- Tighten to ≤ 400 words at NeurIPS pre-submission pass.
- Cross-reference §4 and §6 once the full draft is laid out
  in NeurIPS style file.
