# Background — four theoretical pillars (Paper 1, draft S20.2)

**Target length** : ~1.5 pages markdown (≈ 1500 words)

---

## 3.1 Pillar A — Walker / Stickgold consolidation

Sleep-dependent memory consolidation refers to the empirically
established phenomenon that newly encoded memories are
selectively strengthened, abstracted, and integrated into long-
term storage during sleep [Walker & Stickgold 2004, Stickgold
2005]. Hippocampal replay during NREM slow-wave sleep is the
neural substrate most directly implicated. The functional claim
is that replay performs **gradient-like updates** on cortical
representations, biased toward retention of the replayed
episodes — equivalent in our framework to the `replay`
operation : sample β-buffer episodes, forward through the
current parameters, apply gradient updates against a retention
objective.

## 3.2 Pillar B — Tononi SHY synaptic homeostasis

The Synaptic Homeostasis Hypothesis (SHY) posits that wakefulness
drives net synaptic potentiation, and sleep enforces global
synaptic downscaling that restores signal-to-noise ratio without
erasing the differential strengthening pattern [Tononi & Cirelli
2014]. The downscaling is empirically supported by ultrastructural
evidence (synapse size reductions during sleep) and by behavioral
evidence (sleep-dependent improvement on previously trained
tasks). In our framework, SHY corresponds to the `downscale`
operation : multiplicative shrinkage of weights by a factor in
(0, 1]. As established in our op-pair analysis, downscale is
**commutative but not idempotent** (shrink_f ∘ shrink_f gives
factor², not factor) — a property that constrains canonical
ordering choices.

## 3.3 Pillar C — Hobson / Solms creative dreaming

REM dreaming is associated with creative recombination,
counterfactual scenario generation, and integration of
emotionally significant material [Hobson 2009, Solms 2021]. The
mechanism is hypothesized to be a generative-model-style
sampling from a latent representation of recent experiences,
producing novel combinations that probe the boundaries of
learned structure. In our framework, this maps to the
`recombine` operation : sample latents from the δ snapshot,
apply a VAE-light or interpolation kernel, emit new latent
samples on canal 2.

## 3.4 Pillar D — Friston Free Energy Principle

The Free Energy Principle (FEP) [Friston 2010] frames perception,
action, and learning as the minimization of variational free
energy under a hierarchical generative model. Within FEP, sleep
is interpreted as an offline phase that **restructures** the
generative model to better minimize expected free energy on the
distribution of waking inputs. In our framework, this corresponds
to the `restructure` operation : modify the topology of the
hierarchical model (add layer, remove layer, reroute
connectivity) to reduce predictive error on retained episodes.
The S3 topology guard (validate_topology) ensures that
restructure operations preserve framework-level invariants
(species connectivity, no self-loops, layer count bounds).

## 3.5 The compositional gap

Existing AI work has implemented one or two of the four pillars
(notably A via van de Ven 2020 generative replay and B via
Kirkpatrick 2017 EWC, treated as a SHY-adjacent regularizer).
However, no prior work has **formalized the composition** of all
four operations as a unified algebraic structure with provable
properties.

The compositional gap matters empirically because our op-pair
analysis (`docs/proofs/op-pair-analysis.md`) establishes that 12
of the 16 (op_i, op_j) cross-pairs are **non-commutative** —
that is, applying replay then downscale produces a different
cognitive state than applying downscale then replay. The
canonical ordering chosen in framework §4.3 (replay → downscale
→ restructure ; recombine in parallel) is therefore a load-
bearing design decision, not an arbitrary implementation choice.

A proper formal framework must therefore (i) specify the
operations as composable primitives with well-defined types,
(ii) make explicit which compositions are valid, (iii) provide
an executable account that any compliant substrate can
implement, and (iv) support empirical ablation comparing
different operation profiles. None of the prior art does all
four. Our Framework C (§4) is the first to do so, with the
free-semigroup compositionality axiom DR-2 (proven in
`docs/proofs/dr2-compositionality.md`) as the foundational
property and the Conformance Criterion DR-3 as the executable
contract for substrate-agnosticism.

---

## Notes for revision

- Insert proper bibtex citations (S19.3 references.bib) using
  `\cite{walker2004sleep}` etc. once full draft is rendered
- Tighten §3.5 to ~300 words for Nature HB main-text discipline
- Add Background supplementary figure : conceptual diagram
  of the four pillars with their dreamOfkiki op mappings
