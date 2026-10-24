# Background — four theoretical pillars (Paper 1, draft S20.2)

**Target length** : ~1.5 pages markdown (≈ 1500 words)

---

## 3.1 Pillar A — Walker / Stickgold consolidation

Sleep-dependent memory consolidation refers to the empirically
established phenomenon that newly encoded memories are
selectively strengthened, abstracted, and integrated into long-
term storage during sleep [@walker2004sleep; @stickgold2005sleep].
The 2025 integrative literature now reframes this process as
"active systems consolidation embedded in global synaptic
downscaling," partially reconciling the Born/Diekelmann
[@klinzing2019mechanisms] and Tononi/Cirelli traditions
[@pmc2025systems]. Hippocampal replay during NREM slow-wave
sleep is the neural substrate most directly implicated, gated by
the temporally precise nesting of slow oscillations,
thalamocortical spindles, and sharp-wave ripples
[@trends2024coupled]. A Bayesian meta-analysis of 297 effect
sizes establishes SO-fast-spindle phase coupling as a
quantitatively grounded predictor of memory benefit
(coupling = 0.33 [0.27, 0.39], BF > 58 vs. null, no detected
publication bias on the phase branch) [@elife2025bayesian].
Closed-loop optogenetic boosting of large sharp-wave ripples
during sleep is *sufficient* to convert sub-threshold learning
into successful next-day recall, moving the field from
correlational to interventional grade evidence
[@neuron2025largeswr]. The functional claim is that replay
performs **gradient-like updates** on cortical representations,
biased toward retention of the replayed episodes — equivalent in
our framework to the `replay` operation : sample β-buffer
episodes, forward through the current parameters, apply gradient
updates against a retention objective. Only ~10–30 % of sleep
SWRs carry detectable replay content, so the framework should
not over-claim event density [@annurev2025replay].

## 3.2 Pillar B — Tononi SHY synaptic homeostasis

The Synaptic Homeostasis Hypothesis (SHY) posits that wakefulness
drives net synaptic potentiation, and sleep enforces global
synaptic downscaling that restores signal-to-noise ratio without
erasing the differential strengthening pattern
[@tononi2014sleep]. The downscaling is empirically supported by ultrastructural
evidence (synapse size reductions during sleep) and by behavioral
evidence (sleep-dependent improvement on previously trained
tasks). In our framework, SHY corresponds to the `downscale`
operation : multiplicative shrinkage of weights by a factor in
(0, 1]. As established in our op-pair analysis (see
`docs/proofs/op-pair-analysis.md`, axioms DR-2 + invariants S2),
downscale is **commutative but not idempotent** (shrink_f ∘
shrink_f gives factor², not factor) — a property that constrains
canonical ordering choices.
A 2026 sleep-inspired replay model shows that a single
unsupervised sleep phase scheduled after *several* sequential
tasks restores all of them, with most synapses weakened and a
minority strengthened [@bazhenov2026notjustafterone] — a
computational instance of the replay-plus-downscale coupling
that Pillars A and B jointly describe.

## 3.3 Pillar C — Hobson / Solms creative dreaming

REM dreaming is associated with creative recombination,
counterfactual scenario generation, and integration of
emotionally significant material [@hobson2009rem; @solms2021revising].
The mechanism is hypothesized to be a generative-model-style
sampling from a latent representation of recent experiences,
producing novel combinations that probe the boundaries of
learned structure — operationalized in the Spens-Burgess
generative-consolidation model as hippocampal autoassociative
replay training neocortical variational autoencoders, which
account for episodic recall, imagination, and schema-based
distortions in a single substrate-agnostic latent space
[@spens2024generative]. The clean "REM = emotional /
NREM = declarative" dichotomy is collapsing : NREM TMR cueing
already enhances emotional-item memory with REM contributing
complementarily [@commsbio2025emotional], and the 2025
*Nature Reviews Neuroscience* synthesis supports a **sequential**
SWS-then-REM model rather than a competitive split
[@robertson2025nrn]. Human high-density EEG adds stage-specific
evidence : reactivation time-locked to NREM slow-oscillation–
spindle complexes and reactivation during phasic REM are
dissociable, the former supporting memory consolidation and the
latter emotional dissipation [@zhang2026nremremreactivation] —
consistent with distinct operators per stage rather than a single
"integration" process. In our framework, this maps to the
`recombine` operation : sample latents from the δ snapshot,
apply a VAE-light or interpolation kernel, emit new latent
samples on canal 2.

## 3.4 Pillar D — Friston Free Energy Principle

The Free Energy Principle (FEP) [@friston2010free] frames
perception, action, and learning as the minimization of
variational free energy under a hierarchical generative model. Within FEP, sleep
is interpreted as an offline phase that **restructures** the
generative model to better minimize expected free energy on the
distribution of waking inputs. In our framework, this corresponds
to the `restructure` operation : modify the topology of the
hierarchical model (add layer, remove layer, reroute
connectivity) to reduce predictive error on retained episodes.
The S3 topology guard (validate_topology) ensures that
restructure operations preserve framework-level invariants
S3 (species connectivity, no self-loops, layer count bounds —
see `docs/invariants/registry.md` for canonical definitions and
the S3 guard reference in
`kiki_oniric/dream/guards/topology.py`).

## 3.5 The compositional gap

Existing AI work has implemented one or two of the four pillars
(notably A via van de Ven 2020 generative replay
[@vandeven2020brain] and B via Kirkpatrick 2017 EWC
[@kirkpatrick2017overcoming], treated as a SHY-adjacent
regularizer). A two-network computational substrate alternating
NREM and REM offline replay has been shown to produce autonomous
hippocampus → cortex transfer with retention, mechanistically
bridging active-systems and SHY accounts [@pnas2022dualstage].
However, no prior work has **formalized the composition** of all
four operations as a unified algebraic structure with provable
properties.

The compositional gap matters empirically because our op-pair
analysis (`docs/proofs/op-pair-analysis.md`) establishes that 12
of the 16 (op_i, op_j) cross-pairs are **non-commutative** —
that is, applying replay then downscale produces a different
cognitive state than applying downscale then replay. The
canonical ordering chosen in
`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §4.3
(replay → downscale → restructure ; recombine in parallel) is
therefore a load-bearing design decision, not an arbitrary
implementation choice.

A proper formal framework must therefore (i) specify the
operations as composable primitives with well-defined types,
(ii) make explicit which compositions are valid, (iii) provide
an executable account that any compliant substrate can
implement, and (iv) support empirical ablation comparing
different operation profiles. None of the prior art does all
four. Our Framework C-v0.5.0+STABLE (§4) is the first to do so,
mapping the four pillars to the canonical axiom framework :
pillar A → DR-1 episodic conservation, pillar B → DR-2
compositionality (downscale order constraint), pillar D → DR-3
substrate-agnosticism (the restructure topology guard S3 lives
on this axis), pillar C → DR-4 profile chain inclusion that
keeps the recombine-rich profiles on top. The compositionality axiom DR-2 (proven in
`docs/proofs/dr2-compositionality.md`) is the foundational
property and the Conformance Criterion DR-3 the executable
contract for substrate-agnosticism.

---

## Notes for revision

- Insert proper bibtex citations (S19.3 references.bib) using
  `[@walker2004sleep]` etc. once full draft is rendered
- Tighten §3.5 to ~300 words for Nature HB main-text discipline
- Add Background supplementary figure : conceptual diagram
  of the four pillars with their dreamOfkiki op mappings
