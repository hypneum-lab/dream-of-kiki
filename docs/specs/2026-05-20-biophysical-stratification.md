# Biophysical Stratification Sub-Theory

**Version** : BS-v0.1.0  
**Date** : 2026-05-20  
**Author** : Clement Saillant (L'Electron Rare)  
**Status** : Draft — Paper 2 appendix scaffolding (OQ-3 default; see §0)  
**Companion specs** :
- `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` (framework C)
- `nerve-wml/docs/superpowers/plans/2026-05-19-bio-substrate-wml.md`
  (conformant-substrate side, phase B artifact 1)

This document is a **sub-theory**, not a revision of framework C.
It stratifies a family of biophysical references beside framework C,
providing interpretive and empirical scaffolding for Paper 2 and
for the `BioFieldWML` design plan. It does not introduce new
primitives, channels, or axioms.

---

## 0. Preamble

### 0.1 Framing decision — (i) + (ii)

The 2026-05-19 deep-research synthesis endorsed the **(i) + (ii)
dual decision** in full:

- **(i) DR-3 universal** — substrate-agnosticism is preserved as a
  universal axiom. The formal statement
  "∀ substrate S satisfying the Conformance Criterion, DR-0..DR-2
  are empirically validated on S" remains unchanged and is NOT
  weakened by anything in this document.
- **(ii) Biophysical sub-theory beside framework C** — a stratified
  family of biophysical references is collected here as interpretive
  scaffolding. This family represents ONE class of conformant
  instances; it is not mandated as the only class, and it does NOT
  feed any primitive signature or WML Protocol channel.

This document is the direct deliverable of decision (ii).

### 0.2 Scope (OQ-3 default)

**Paper 2 appendix scaffolding** (unpublished empirical context).
Formal rigour is lower than framework C; claims are interpretive,
not axiomatic. Lombardi 2020 (Stratum 5) is included under this
default. A reviewer can override at PR review by annotating
"OQ-3: standalone scope" — which requires strengthening Stratum 5
or dropping it.

### 0.3 Non-revision contract

This spec NEVER weakens DR-0..DR-4, N-1..N-5, W-1..W-4. If any
section appears to imply an axiom revision, treat it as a bug.
DR-0: biophysical processes are always DE-bounded; continuous
oscillatory dynamics are interpretive context only. DR-3: the
stratification below describes a biophysical *family*, not the
mandated substrate; conformance bridge is `BioFieldWML` in
nerve-wml. OQ-1/OQ-2 defaults applied (see §7).

---

## 1. Stratum 1 — Coupled-field substrate (Blum-Moyse line)

### 1.1 References

- **Blum-Moyse 2023** (thèse INSA): three-level plasticity taxonomy
  — (L1) cortico-thalamo-striatal ITDP, (L2) neuron-astrocyte
  Up-Down states and epileptiform transitions, (L3)
  hippocampo-neocortical coupled field supporting systems
  consolidation theory (SCT).
- **Blum-Moyse & Berry JTB 2024**: journal extension of the thesis;
  formalises the cortico-thalamo-striatal + theta field coupling
  and provides the central theoretical articulation of the
  three-level model.
- **bioRxiv 2025 "A unified model of cortico-hippocampal
  interactions through neural field theory"**: separate hippocampal
  (theta, 4–8 Hz) and cortical (alpha, 8–12 Hz) neural fields
  coupled via a quasi-conformal mapping geometry; provides a field-
  theoretic account of SCT at the population level.

### 1.2 Sub-theory contribution

The three-level taxonomy (L1–L3) maps onto the dreamOfkiki DE
sequence: L1 ITDP changes occur within a single step, L2 Up-Down
coupling occurs at intermediate DE timescales, and L3
hippocampo-neocortical transfer corresponds to the multi-DE
consolidation arc. The quasi-conformal-mapping coupling from
bioRxiv 2025 shows that a *geometry* on the coupling constrains
admissible cross-compartment interactions — an interpretive input
for `BioFieldWML` parameter design, without mandating that geometry
at the WML Protocol level.

### 1.3 Relationship to DR-3 and DR-0

DR-3 universality is preserved: this stratum is ONE conformant
instance; `BioFieldWML` (see §6 sister doc) is the conformance
bridge. DR-0 safety: L2 Up-Down oscillations are bounded within
`BioFieldWML.step()` (OQ-1 default) — never unbounded background
processes. The Blum-Moyse model motivates cycle length; it does
not override DE-budget accountability.

---

## 2. Stratum 2 — Theta-gamma sequence organisation

### 2.1 Reference

- **Ursino, Cesaretti & Acar 2024 (Frontiers in Neural Circuits)**
  (Bologna lab): neural-mass model simulating theta-gamma sequence
  encoding and retrieval. The model includes a regime where,
  when isolated from external sensory input, it generates
  spontaneous sequences interpretable as imagination or dreaming.

### 2.2 Sub-theory contribution

1. **Neurobiological correlate of N-3**: Ursino's theta-gamma
   coupling model provides a *biological correlate* of nerve-wml
   N-3 (γ/θ multiplex fidelity) — a reason to expect that
   multiplexing violations degrade sequence fidelity. This is
   interpretive support for N-3, not a redefinition.

2. **Dream Episode trigger analogy**: the "isolated from external
   input → dreaming phase" regime in Ursino's model maps onto the
   dreamOfkiki DE initiation mechanism (β buffer triggers
   input-isolated mode). Interpretive bridge only; no new primitive.

Stratum 2 provides biological motivation for N-3 and the DE trigger;
the neural-mass formalism stays in this sub-theory.

---

## 3. Stratum 3 — Multimodal efficient + predictive (kiki-flow tie-in)

### 3.1 Reference

- **Młynarski & Hermundstad 2025 (bioRxiv)**: "Convergence of
  efficient and predictive coding in multimodal sensory processing".
  Normative result showing that, under broad conditions, the
  objectives of efficient coding (compression) and predictive coding
  (error minimisation) converge to the same representation in
  multimodal sensory systems.

### 3.2 Sub-theory contribution

Młynarski-Hermundstad 2025 shows that efficient coding (compression)
and predictive coding (error minimisation) converge to the same
representation in multimodal systems. For kiki-flow-research this
means: a model trained to minimise prediction error simultaneously
minimises coding cost, unifying two otherwise separate evaluation
criteria in the Paper 2 ablation.

Optional bridge to bouba_sens B-3: the cross-modal consistency
result can be read as an empirical test of whether this convergence
holds for artificial systems. This connection is speculative and
must not be presented as validated without running the ablation.

---

## 4. Stratum 4 — Embodied sensorimotor grounding (B-3 anchored)

### 4.1 References

- **Heinrich et al. 2020 (Frontiers in Neurorobotics)**:
  "Crossmodal Language Grounding in an Embodied Neurocognitive
  Model" — MTRNN (Multi-Timescale Recurrent Neural Network)
  architecture with the NICO robot; models how linguistic and
  sensorimotor representations become grounded across modalities.
- **Hwang et al. 2018 (arXiv:1804.06774)**:
  AFA-PredNet — fast/slow timescale separation in an embodied
  predictive coding model with motor modulation. The "asymmetric
  fast-slow" architecture achieves robustness to motor perturbations
  via hierarchical timescale structure.
- **arXiv:2505.09760 — NASM 2025**:
  "Neural Associative Skill Memories" — sensorimotor grounding in
  temporal predictive coding with biologically-plausible local
  (Hebbian-style) learning rules. The model demonstrates that
  skill-level sensorimotor associations can be formed without
  backpropagation through time.
- **Côte d'Azur 2020 (2020COAZ4085, thèse)**:
  Distributed cellular neuromorphic architectures with structural
  and synaptic plasticity; cross-modal activation via temporal
  correlation. Provides a hardware-relevant perspective on
  embodied plasticity.

### 4.2 Sub-theory contribution and bouba_sens anchor

All four references bear on **B-3** (PASS: perceptive/proprioceptive
asymmetry), NOT on B-1 (Final Retract, N9 verdict 2026-04-23).

- **Multi-timescale structure** (Heinrich MTRNN, Hwang AFA-PredNet):
  fast-slow timescale separation provides a biological correlate
  for the profile stack P_min → P_equ → P_max. Slower timescales
  correspond to higher profiles and longer DE sequences. Does not
  mandate MTRNN or AFA-PredNet as implementation.
- **Local learning plausibility** (NASM 2025): strengthens
  justification for STDP-style updates inside `BioFieldWML` (OQ-2
  context). YAGNI bound from nerve-wml spec §570 preserved.
- **Structural plasticity** (Côte d'Azur 2020): long-term context
  for how `BioFieldWML` might extend to structural weight changes.
  Paper 2 appendix pointer, not a current design input.
- **Cross-modal cell-assembly bias** (Heinrich 2020): interpretive
  anchor for B-3's asymmetry result — the asymmetry reflects known
  cross-modal grounding structure, not a measurement artefact.

MTRNN and AFA-PredNet are design *inputs*, not mandated architectures
(subject to DR-3 if any implementation is attempted).

---

## 5. Stratum 5 — Critical dynamics (speculative, conditional)

**Included only under OQ-3 default (Paper 2 appendix scope).**
If scope is promoted to standalone preprint, Stratum 5 must be
strengthened with cross-validated criticality metrics or dropped.

- **Lombardi 2020**: neural systems operating near criticality
  during consolidation; proximity to phase transition is
  functionally necessary for memory compression and generalisation.
  Under current scope, this is a plausible mechanistic account for
  why DE budgets near a threshold outperform longer or shorter
  episodes — a hypothesis worth testing in Paper 2 ablations, not
  a design primitive. If formalised, it would enter as an invariant
  candidate inside `BioFieldWML`, not as an axiom of framework C.

---

## 6. Cross-references

- **Sister doc (phase B artifact 1)**:
  `nerve-wml/docs/superpowers/plans/2026-05-19-bio-substrate-wml.md`
  — the conformant-substrate side. That plan specifies `BioFieldWML`
  (the WML Protocol conformant class). Present doc = biological
  motivation; nerve-wml plan = protocol conformant implementation.
- **Classification gating doc**:
  memory file `project_hypneum_deepresearch_2026_05_19_classification.md`
  — all (b′) routing decisions for the 9-10 refs are traceable there.
- **Paper 1 HNN anchor** (PR #18): Liu 2024 HNN establishes the
  HNN-taxonomy anchor; this sub-theory is complementary (biophysical
  consolidation angle, no conflict).
- **c-alert refs excluded**: Singh-2024-CVPRW-WSEBM (DR-0 conflict)
  and Dream2Learn-arXiv-2026 (DR-3 conflict) are NOT included.
  Singh-2024 superseded by Bellitto-WSCL-2024; Dream2Learn
  reclassified to citation-only.

---

## 7. Open questions retained

### OQ-1 — BioFieldWML / DR-0 boundary (default applied)

Up-Down and ITDP dynamics are fully encapsulated inside
`BioFieldWML.step()`. Each call = one synchronous cycle; budget
finite per call; no oscillatory process spans calls requiring a new
WML Protocol channel. DR-0 preserved. Multi-call oscillatory state
requires a spec amendment before implementation.

### OQ-2 — Tomé STDP scope gate (default applied)

STDP triplet scoped to `BioFieldWML`, requires validation against
W-1 (weight-norm bound) and N-3 (population-code fidelity) under
the nerve-wml test suite. Surrogate-gradient YAGNI (spec §570)
not revoked. W-1 or N-3 violation triggers a spec amendment.

### OQ-3 — Doc scope (default applied)

Paper 2 appendix scaffolding, not a standalone preprint. Override
at PR review by annotating: "OQ-3 standalone — Stratum 5 must be
strengthened or removed; formal rigour additions TBD."

---

## 8. Non-goals

- Not a redefinition of DR-3 — substrate-agnosticism is universal.
- Not a mandate for a single biophysical substrate — strata define
  a *family*; conformant implementations may draw on any subset.
- Not a paper outline — use `docs/papers/paper2/` for that.
- Not a revision of bouba_sens B-1 — Final Retract (N9) is final;
  Stratum 4 anchors on B-3 only.
- Not an operational spec for `kiki_oniric/` — no code changes are
  implied without a PR explicitly citing the guarded invariant.
