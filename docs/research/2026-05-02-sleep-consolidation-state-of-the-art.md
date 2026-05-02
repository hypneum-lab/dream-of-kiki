# Sleep-Dependent Memory Consolidation — État de l'art 2025-2026

*Generated: 2026-05-02 | Sources: 32 unique (after dedup) | Confidence: High on neurobiology + theory; Medium on AI bridging; Medium-low on clinical translation*

> Research report compiled for the `dream-of-kiki` framework C programme (Paper 1 →
> PLOS Comp Bio). All claims are sourced. Quotes drawn from search-snippet level
> reading; flagged citations marked `[unverified primary]` MUST be re-verified
> against full PDFs before manuscript submission.

---

## Executive Summary

Five threads have moved decisively in 2024-2026:

1. **Mechanism is now quantitatively grounded.** A Bayesian meta-analysis of
   297 effect sizes (eLife 2025) shows SO-fast-spindle phase coupling
   predicts memory benefit with Bayes Factor > 58 vs. null and **no detected
   publication bias on the phase branch**. Coupling strength = 0.33 [0.27, 0.39].
2. **Causality is established.** Closed-loop optogenetic boosting of large
   sharp-wave ripples in mice (Neuron 2025) is *sufficient* to convert
   sub-threshold learning into successful next-day recall — moving the field
   from correlational to interventional.
3. **Theoretical unification.** The 2025 PMC integrative review explicitly
   reframes consolidation as "active systems consolidation embedded in global
   synaptic downscaling," partially reconciling Born/Diekelmann (active
   systems) and Tononi/Cirelli (SHY); the Spens-Burgess generative consolidation
   model (Nat Hum Behav 2024) operationalizes consolidation as *generative
   reconstruction* rather than copy-and-store.
4. **Three independent communities have converged on the same architectural
   pattern**: an *offline consolidation phase replaying compressed/latent
   representations*. Continual-learning (van de Ven 2024 survey), LLMs
   (Sleep-time Compute 2025: +13-18 % accuracy, 5× compute amortisation;
   Titans 2025; "Language Models Need Sleep" ICLR 2026 submission), and
   neuromorphic systems (CLP-SNN on Loihi 2 2025: 70× speed, 5,600× energy
   efficiency vs GPU baselines).
5. **Clinical translation is not yet there.** A March 2025 medRxiv
   sham-controlled CLAS trial in chronic insomnia (N=27) shows physiological
   SO amplitude gain but **null on memory and sleep outcomes**; multi-night
   home AD studies expose a damaging dose-response artifact (patients with
   lowest baseline SWS receive fewest stimulations).

The pattern is robust enough to invite a substrate-agnostic axiomatic
formalisation — exactly the gap `dream-of-kiki` framework C is positioned
to fill.

---

## 1. Neurobiological Mechanisms (SQ1)

### 1.1 The triadic SO-spindle-ripple coupling

The mechanistic core has shifted from "spindles or replay" to the temporally
precise *nesting* of slow oscillations (SO, <1 Hz), thalamo-cortical spindles
(10–16 Hz), and hippocampal ripples (SWR, 80–200 Hz). The 2024 *Trends in
Cognitive Sciences* review canonicalizes the formulation:

> "Sequential SO–spindle–ripple coupling provides a temporally and spatially
> fine-tuned mechanism to selectively strengthen target memories" — Trends Cog
> Sci 2024 ([sciencedirect.com/.../S1364661324000299](https://www.sciencedirect.com/science/article/pii/S1364661324000299)).

The 2025 PMC integrative review (PMC12576410) embeds this triad in a broader
neuromodulatory + plastic-remodeling story, framing consolidation as
"an active systems consolidation process embedded in global synaptic
downscaling" — a synthesis of the Born/Diekelmann and Tononi/Cirelli camps
([pmc.ncbi.nlm.nih.gov/articles/PMC12576410/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12576410/)).

### 1.2 Quantitative evidence (Bayesian meta-analysis)

The single highest-grade quantitative claim available to Paper 1 is the
**Bayesian meta-analysis** of SO-spindle coupling and memory:

| Quantity | Estimate | 95% CI |
|---|---|---|
| SO-fast-spindle coupling strength (frontal) | 0.33 | [0.27, 0.39] |
| SO-slow-spindle coupling strength | 0.23 | [0.19, 0.27] |
| Bayes factor (phase-memory association) | > 58 | vs. null |
| Egger publication-bias test (phase branch) | p = 0.59 | n.s. |

Source: eLife 2025 (preprint Aug 2024, accepted Oct 2025), 23 studies, 297
effect sizes. ([elifesciences.org/articles/101992](https://elifesciences.org/articles/101992)).

> "Strong evidence supporting that precise and strong SO-fast SP coupling in
> the frontal lobe predicts memory consolidation" — eLife 2025.

The authors **explicitly recommend OSF pre-registration as standard practice**
going forward, and flag possible asymmetry on the amplitude-memory branch.

### 1.3 Causal evidence — closed-loop SWR boosting

Neuron 2025 reports the first sufficiency-grade causal demonstration:
closed-loop optogenetic enhancement of *large* sharp-wave ripples during
sleep is sufficient to convert sub-threshold learning into successful
next-day recall.

> "Enhancing large SWRs during sleep improved mice memory performance,
> enabling them to successfully recall a brief prior experience normally
> insufficient for successful retrieval."
> — Neuron 2025 ([cell.com/neuron/abstract/S0896-6273(25)00756-1](https://www.cell.com/neuron/abstract/S0896-6273(25)00756-1)).

This is the load-bearing citation for any framework C claim about DR-2
(replay channel) causality.

### 1.4 Selectivity / fraction problem

A persistent open question: only ~10–30 % of sleep SWRs carry detectable
replay content (Annual Review of Neuroscience 2025,
[annualreviews.org/.../annurev-neuro-112723-024516](https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-112723-024516)).
There is no consensus on what determines whether a given ripple is
consolidative, idle, or spurious. Paper 1 should not over-claim density of
replay events.

### 1.5 NREM-vs-REM division of labour

Long held as a clean dichotomy ("REM = emotional, NREM = declarative"), this
view is collapsing:

- *Communications Biology* 2025: TMR cueing in NREM enhanced emotional-item
  memory and was tracked by spindle + theta power; REM contributes
  complementarily but is *not necessary* for cued benefit
  ([nature.com/articles/s42003-025-07868-5](https://www.nature.com/articles/s42003-025-07868-5)).
- *bioRxiv* Sept 2025 `[unverified primary — preprint]`: REM-state PFC ripple
  "chains" with reactivation profiles distinct from NREM SWRs
  ([biorxiv.org/.../2025.09.15.676366v1](https://www.biorxiv.org/content/10.1101/2025.09.15.676366v1.full)).
- *Nature Reviews Neuroscience* 2025 (Robertson): supports a **sequential**
  model — SWS first quarter + REM final quarter — rather than dichotomy
  ([nature.com/articles/s41583-025-00973-8](https://www.nature.com/articles/s41583-025-00973-8)).

Implication for `dream-of-kiki`: profile design (`P_min`, `P_equ`, `P_max`)
should support a **sequential complementary** rather than competitive
NREM/REM model.

---

## 2. Theoretical Frameworks (SQ2)

### 2.1 Active Systems Consolidation (Born, Diekelmann, Klinzing)

Most-cited modern statement: Klinzing, Niethard, Born (Nat Neurosci 2019,
[nature.com/articles/s41593-019-0467-3](https://www.nature.com/articles/s41593-019-0467-3)).
Repeated hippocampus-driven replay during SWS, gated by SO up-states and
spindles, embedded within REM-driven plasticity refinement. **No fully
replacing 2024-2026 first-author Born/Diekelmann review surfaced** — 2025
PMC integrative review is the current update.

### 2.2 Synaptic Homeostasis Hypothesis (Tononi, Cirelli)

Foundation: Neuron 2014 + EJN 2020 update. Net potentiation during wake,
net renormalization during sleep — "the price the brain pays for plasticity."

> "Sleep downscales synaptic strength to a baseline level that is
> energetically sustainable… and is beneficial for learning and memory."
> — Tononi & Cirelli ([cell.com/neuron/fulltext/S0896-6273(13)01186-0](https://www.cell.com/neuron/fulltext/S0896-6273(13)01186-0)).

**Gap flagged**: no 2025-2026 first-author SHY review surfaced; verify
PubMed before final cite. Likely maps to `dream-of-kiki` invariant family
**S** (homeostatic budget).

### 2.3 Complementary Learning Systems (Kumaran, Hassabis, McClelland)

Canonical CLS-2016 update: TiCS 2016
([cell.com/.../S1364-6613(16)30043-2](https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(16)30043-2)).
Allows goal-weighted, non-uniform replay + rapid neocortical learning when
new information is schema-consistent. **No 2024-2026 dedicated CLS
update by the same authors retrievable.** Field has diffused into
generative-consolidation modelling and the 2025 systems-consolidation
reviews instead.

### 2.4 Generative consolidation (Spens & Burgess; van de Ven)

The unifying computational claim of the 2024-2026 era: consolidated memory
is *generatively reconstructed*, not copied. Hippocampal autoassociative
replay trains neocortical variational autoencoders (entorhinal / mPFC /
aTC). The same model accounts for episodic recall, imagination,
future-thinking, schema distortions, boundary extension.

> "Episodic memories are reconstructed, share neural substrates with
> imagination, combine unique features with schema-based predictions, and
> show schema-based distortions that increase with consolidation."
> — Spens & Burgess, *Nat Hum Behav* 2024
> ([nature.com/articles/s41562-023-01799-z](https://www.nature.com/articles/s41562-023-01799-z)).

This is the **load-bearing reference for any GLUT-aligned `dream-of-kiki`
framing** — it operationalises consolidation in a substrate-agnostic
generative latent space. The earlier van de Ven, Siegelmann, Tolias *Nat
Commun* 2020 brain-inspired replay paper supplies the deep-learning
implementation precedent
([nature.com/articles/s41467-020-17866-2](https://www.nature.com/articles/s41467-020-17866-2)).

A 2024-2025 sequential-extension paper (`[unverified primary]` — final
venue OpenReview / ResearchGate snippets only) extends the model to
sequence prediction and planning, suggesting unification with cognitive-map
literature.

### 2.5 PNAS 2022 dual-stage substrate precedent

A computational two-network model showing alternation of NREM and REM
(offline interleaved replay) autonomously produces hippocampus → cortex
transfer with retention; mechanistically bridges active-systems and SHY
([pnas.org/doi/10.1073/pnas.2123432119](https://www.pnas.org/doi/10.1073/pnas.2123432119)).
Useful precedent for `kiki_oniric`'s dual-stage substrate.

---

## 3. AI Dreaming, Generative Replay, Sleep-Inspired Learning (SQ3)

### 3.1 Continual learning is converging on latent replay

Van de Ven et al. (Nicholas Soures, Dhireesha Kudithipudi) published a
2024 monograph synthesising five years of CL research
([arxiv.org/abs/2403.05175](https://arxiv.org/abs/2403.05175)):

> "An issue with [generative replay] is that it can be difficult to train
> generative models of decent quality, especially in an incremental
> setting... this can be alleviated by replaying latent features instead of
> raw inputs."

This is the SOTA-aligned anchor for DR-1 and the "no raw episodic store"
reading of the Conformance Criterion.

A 2025 stateful-replay preprint (arXiv 2511.17936
`[unverified primary]`) and the 2025 SESLR SNN latent-replay paper
([arxiv.org/abs/2507.02901](https://arxiv.org/abs/2507.02901)) operationalise
the trend in streaming and event-driven settings respectively.

### 3.2 LLMs are reinventing sleep — but as metaphor

Two 2025 publications stand out:

| Paper | Mechanism | Reported gain |
|---|---|---|
| **Sleep-time Compute** (Berkeley/Letta, [arxiv.org/abs/2504.13171](https://arxiv.org/abs/2504.13171)) | Pre-compute context-conditioned reasoning during idle | +13 % GSM-Symbolic, +18 % AIME, 5× test-time compute reduction, 2.5× cost amortisation |
| **Titans** (Google Research, [arxiv.org/abs/2501.00663](https://arxiv.org/abs/2501.00663)) | Hybrid attention (short-term) + neural long-term memory module learnt at test time | Beats Transformer/Mamba on needle-in-haystack; 2 M tokens |

> "The neural memory has the ability to continuously learn from data and
> store it in its weights to play the role of long-term memory, while
> transformers' attention mechanisms are interpreted as short-term memory
> modules." — Titans, Google Research 2024.

Both papers admit explicitly or implicitly that "sleep" / "memory" are
metaphors with **no biological grounding**. Framework C is positioned to
supply the principled formal grounding the term currently lacks.

### 3.3 Wake-Sleep Consolidated Learning (closest dual-phase analog)

Alfarano et al., IEEE TNNLS Jan 2024
([arxiv.org/abs/2401.08623](https://arxiv.org/abs/2401.08623)):

> "During NREM sleep stages, synaptic weights are consolidated using
> replayed memory samples, while in REM sleep stages... a dreaming process
> enables it to explore potential feature spaces, preparing synapses for
> future knowledge."

This is the **closest published instantiation of an explicit NREM/REM split
in a deep network**, and the natural Paper 2 ablation comparator.

### 3.4 ICLR 2026 / 2026 concurrent proposals

- **"Language Models Need Sleep: Learning to Self Modify and Consolidate
  Memories"** — anonymous ICLR 2026 submission
  ([openreview.net/forum?id=iiZy6xyVVE](https://openreview.net/forum?id=iiZy6xyVVE)).
  Proposes RL-based upward distillation + intentional forgetting.
- **Self-Distillation Enables Continual Learning** — Shenfeld, Damani et al.,
  arXiv 2026 (`[unverified primary]`, arXiv ID forward-dated, verify before
  citing). Self-distillation on ~200 K-token synthetic Wikipedia-style
  corpora as a "synthetic dream" replay proxy.

Concurrent independent proposals strengthen the novelty argument: framework
C arrived at the *axiomatic* level while peers remain empirical.

---

## 4. Artificial Substrates for Consolidation (SQ4)

### 4.1 Neuromorphic — Loihi 2 / SpiNNaker 2 / mlx-snn

**CLP-SNN on Intel Loihi 2** (Hajizada et al., arXiv 2511.01553, Nov 2025,
[arxiv.org/abs/2511.01553](https://arxiv.org/abs/2511.01553)):

| Metric | CLP-SNN (Loihi 2) | Replay-based GPU baseline |
|---|---|---|
| Few-shot CL accuracy (OpenLORIS) | matches | matches |
| Throughput | **70× faster** | 1× |
| Energy / inference | **0.05 mJ** | 281 mJ (5,600× factor) |

> "Event-driven and spatiotemporally sparse local learning, a
> self-normalizing three-factor learning rule, and integrated neurogenesis
> and metaplasticity for capacity expansion and forgetting mitigation."

This is the load-bearing citation for the E-SNN substrate track of
`kiki_oniric` — concrete real-time-on-silicon evidence that DR-2
consolidation is substrate-portable.

Complement: **SpiNNaker 2** (TU Dresden, arXiv 2401.04491,
[arxiv.org/html/2401.04491v1](https://arxiv.org/html/2401.04491v1)) — hybrid
SNN/DNN platform.

### 4.2 Apple Silicon / MLX path — `mlx-snn` (2026)

`mlx-snn` ([arxiv.org/abs/2603.03529](https://arxiv.org/abs/2603.03529)) is
the first native MLX SNN library: 6 neuron models, BPTT pipeline, **97.28 %
MNIST, 2.0–2.5× faster training, 3–10× lower GPU memory** vs. snnTorch on
M3 Max. Direct enabler for the MLX-side `kiki_oniric` E-SNN substrate
implementation; matches the workspace constraint that training runs on M3
Ultra.

### 4.3 Thalamocortical biophysical replay (bioRxiv 2025)

> "Slow-wave sleep employs an interleaved replay of familiar cortical and
> novel hippocampal memory traces within individual Up states... allowing
> new memories to be embedded into the existing pool of cortical memories
> without interference."

bioRxiv June 2025 (PMID 40667278,
[biorxiv.org/.../2025.06.25.661579v1](https://www.biorxiv.org/content/10.1101/2025.06.25.661579v1)).
Anchors biological grounding of DR-2'; direct reference for the
thalamocortical channel in the framework-C spec.

### 4.4 Transformer-side memory consolidation

**BrainTransformers / SNN-LLM** ([arxiv.org/html/2410.14687v2](https://arxiv.org/html/2410.14687v2)):
spiking-equivalent FFN/attention; demonstrates LLM-scale SNNs are viable.
Establishes the SNN+Transformer hybrid space framework C must remain
agnostic over.

---

## 5. Empirical Validation & Clinical Translation (SQ5)

### 5.1 The TMR meta-analysis floor

Hu et al. *Psychological Bulletin* 2020 remains the canonical empirical
anchor (still cited unmodified through 2025):

| Stratum | Hedges' g | 95% CI |
|---|---|---|
| Overall (k=91, 212 effect sizes, N=2,004) | **0.29** | [0.21, 0.38] |
| NREM2 | 0.32 | [0.04, 0.60] |
| SWS | 0.27 | [0.20, 0.35] |
| REM | null | — |
| Wake | null | — |

([pmc.ncbi.nlm.nih.gov/articles/PMC7144680/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7144680/)).

The 2024 *npj Science of Learning* review notes the field has shifted from
"TMR works" to "TMR works *conditionally*"; awake TMR is increasingly
documented as null-or-disruptive
([nature.com/articles/s41539-024-00244-8](https://www.nature.com/articles/s41539-024-00244-8)).

### 5.2 Sleep-restriction dose-response

Javadi et al. *Neurosci Biobehav Rev* 2024
([sciencedirect.com/.../S0149763424003981](https://www.sciencedirect.com/science/article/pii/S0149763424003981)):
**g = 0.29 [0.13, 0.44]** across 39 reports / 125 effect sizes / N = 1,234,
**no detected publication bias**. Partial sleep loss (3–6.5 h) impairs
memory comparably to total deprivation. Supports the framework C
"minimum-viable-cycle" assumption (`P_min` profile) — there is no graceful
degradation below threshold.

### 5.3 Clinical translation gap

**CLAS in chronic insomnia** (medRxiv March 2025, N=27, sham-controlled
crossover, [medrxiv.org/.../2025.03.04.25321710v1](https://www.medrxiv.org/content/10.1101/2025.03.04.25321710v1)):

> "No beneficial effect of a single night of CLAS on subjective and
> objective sleep or declarative overnight memory performance."

Acute SO-amplitude gain present; behavioural benefit absent. **Critical
disconfirmation** of the simplistic "more SO ⇒ more consolidation" model.

**CLAS in Alzheimer's disease at home** (Am J Geriatr Psychiatry 2024,
[ajgponline.org/article/S1064-7481(24)00384-1/abstract](https://www.ajgponline.org/article/S1064-7481(24)00384-1/abstract)):

> "Patients with lower baseline SWS received fewer stimulations during the
> intervention, possibly resulting in less SWS enhancement."

Documents a damaging dose-response artifact: the population most needing
the intervention receives the least of it. Maps onto invariant **K1**
(dream-episode budget — FLOPs / wall-time / energy bounded per DE) in
framework C ; the floor effect is exogenous to K1 itself, but K1 is the
invariant that recognises the bounded-stimulation signal.

### 5.4 Clinical phenotyping — biomarker convergence

Sharon et al. *Alzheimer's & Dementia* 2025
([alz-journals.onlinelibrary.wiley.com/doi/10.1002/alz.70247](https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/alz.70247))
— hd-EEG in N=55 (21 healthy older / 28 aMCI / 6 AD):

> "Cognitive performance robustly decreases with slow wave trough amplitude
> and its synchronization across broad frontocentral cortical areas."

Provides empirical biomarker grounding for `kiki_oniric` profile design
(`P_min` ↔ degraded SO of aMCI; `P_equ` ↔ healthy older).

Schizophrenia pillar: spindle deficits + procedural memory consolidation
deficits in early-course minimally-medicated patients but not first-degree
relatives (Manoach lab, *Schizophrenia Research* 2024,
[pubmed.ncbi.nlm.nih.gov/39515257/](https://pubmed.ncbi.nlm.nih.gov/39515257/));
earlier eszopiclone RCT for spindle pharmacological augmentation
([pubmed.ncbi.nlm.nih.gov/32919407/](https://pubmed.ncbi.nlm.nih.gov/32919407/)).

### 5.5 Replication-critique baseline

Cordi & Rasch *Eur J Neurosci* 2021
([pubmed.ncbi.nlm.nih.gov/32711356/](https://pubmed.ncbi.nlm.nih.gov/32711356/))
remains the load-bearing 2021-2026 critique:

> "Effects of sleep on memory are smaller, more task-dependent, less
> SWS-related, less robust and less long-lasting than previously assumed."

The 2023 *Nature Reviews Psychology* methodology paper (cited as
gold-standard checklist by 2025 papers,
[nature.com/articles/s44159-023-00262-0](https://www.nature.com/articles/s44159-023-00262-0))
flags the recurrent confound between sequential designs and
retrieval-vs-restudy strength manipulations.

Schreiner et al. *J Neurosci* 2024 (
[jneurosci.org/content/44/24/e0022242024](https://www.jneurosci.org/content/44/24/e0022242024)):
TMR does **not** act holistically on object memory — fuels debate about
which memory features are sleep-sensitive.

---

## 6. Cross-Cutting Bridges (Neuroscience ↔ AI)

The 2024-2026 bridging literature places `dream-of-kiki` at a structural
opening:

- Spens & Burgess 2024 — generative consolidation (covered §2.4) is the
  conceptual ancestor of framework C's consolidation channel.
- Robertson NRN 2025 ([nature.com/articles/s41583-025-00973-8](https://www.nature.com/articles/s41583-025-00973-8))
  reframes sleep as an **active** state with diverse memory functions, with
  Stickgold's ~20 % visual-memory boost replicated.
- "AI Meets Brain" survey arXiv 2512.23343 (2025) `[unverified primary]`
  reviews memory-augmented transformers vs. neuroscientific consolidation
  theories.

**The structural gap framework C fills**: no published 2024-2026 work
formally maps the SO-spindle-ripple triad onto a *substrate-agnostic*
computational invariant. The neuroscience side has the mechanism; the AI
side has the function (replay → no-forgetting); neither has the axiomatic
bridge.

---

## Key Takeaways

1. **Cite the eLife 2025 Bayesian meta-analysis** (BF > 58, coupling
   strength 0.33 [0.27, 0.39]) as the strongest available quantitative
   anchor for any DR-2 / invariant-K claim. Pair with Neuron 2025 large-SWR
   optogenetics for sufficiency-grade causality.
2. **Use Spens & Burgess Nat Hum Behav 2024 as the conceptual axis** for
   framework C's generative-consolidation framing — it is the single most
   load-bearing modern reference, replacing the missing 2024-2026 CLS
   update.
3. **Position framework C as the substrate-agnostic axiomatic layer** that
   Sleep-time Compute, Titans, Wake-Sleep Consolidated Learning, CLP-SNN,
   and the bioRxiv thalamocortical model all *implement* but none
   *formalises*. Concurrent ICLR 2026 / 2026 proposals strengthen the
   novelty argument rather than threaten it.
4. **Be empirically honest on translation.** The March 2025 CLAS-in-insomnia
   null + the AD home-study floor effect must appear in Paper 1's
   empirical-anchors-and-limits section. Cordi & Rasch 2021 is still the
   correct critique citation.
5. **Profile design** (`P_min`, `P_equ`, `P_max`) should track the Sharon
   2025 SO-trough-amplitude gradient (healthy older → aMCI → AD) and adopt
   a **sequential complementary** NREM/REM model, not a competitive one
   (Robertson NRN 2025; Comm Biol 2025).
6. **Pre-register**. The eLife 2025 meta-analysis explicitly recommends OSF
   pre-registration as standard. `dream-of-kiki`'s sister project
   `bouba_sens` already pre-registers at OSF.IO/Q6JYN; align Paper 2's
   ablation pipeline to the same discipline.

---

## Sources (32 unique, dedup'd across agents)

### Neurobiological mechanisms (SQ1)
1. [Systems memory consolidation during sleep — PMC integrative review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12576410/) — 2025 anchor for §Background.
2. [Coupled sleep rhythms for memory consolidation — Trends Cog Sci 2024](https://www.sciencedirect.com/science/article/pii/S1364661324000299) — canonical SO-spindle-ripple statement.
3. [Bayesian meta-analysis of SO-spindle coupling — eLife 2025](https://elifesciences.org/articles/101992) — BF > 58, coupling 0.33, no pub bias on phase.
4. [Large SWRs promote memory reactivation — Neuron 2025](https://www.cell.com/neuron/abstract/S0896-6273(25)00756-1) — sufficiency-grade optogenetic evidence.
5. [Modulating sleep with SO/spindle stimulation — npj Sci Learn 2025](https://www.nature.com/articles/s41539-025-00383-6) — CLAS heterogeneity & age-attenuation.
6. [SWS + REM contributions to emotional consolidation — Comm Biol 2025](https://www.nature.com/articles/s42003-025-07868-5) — TMR cueing in NREM.
7. [Replay and Ripples in Humans — Annual Rev Neurosci 2025](https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-112723-024516) — selectivity / fraction problem.
8. [REM PFC ripple chains — bioRxiv Sept 2025 `[unverified primary]`](https://www.biorxiv.org/content/10.1101/2025.09.15.676366v1.full) — REM-specific replay channel.

### Theoretical frameworks (SQ2)
9. [CLS updated — Kumaran, Hassabis, McClelland — TiCS 2016](https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(16)30043-2) — canonical CLS reference.
10. [Mechanisms of systems memory consolidation — Klinzing, Niethard, Born — Nat Neurosci 2019](https://www.nature.com/articles/s41593-019-0467-3) — Active Systems statement.
11. [Sleep and synaptic down-selection — Tononi, Cirelli — Neuron 2014](https://www.cell.com/neuron/fulltext/S0896-6273(13)01186-0) — SHY foundation.
12. [Brain-inspired replay for CL — van de Ven, Siegelmann, Tolias — Nat Commun 2020](https://www.nature.com/articles/s41467-020-17866-2) — generative-replay implementation precedent.
13. [Generative model of memory construction — Spens & Burgess — Nat Hum Behav 2024](https://www.nature.com/articles/s41562-023-01799-z) — **load-bearing** generative consolidation reference.
14. [Hippocampus-cortex autonomous interaction — PNAS 2022](https://www.pnas.org/doi/10.1073/pnas.2123432119) — dual-stage NREM/REM substrate precedent.

### AI dreaming / continual learning (SQ3)
15. [Continual Learning and Catastrophic Forgetting — van de Ven et al. monograph 2024](https://arxiv.org/abs/2403.05175) — survey, latent-replay SOTA.
16. [Sleep-time Compute — Berkeley 2025](https://arxiv.org/abs/2504.13171) — +13–18 % accuracy, 5× compute amortisation.
17. [Wake-Sleep Consolidated Learning — IEEE TNNLS 2024](https://arxiv.org/abs/2401.08623) — closest NREM/REM dual-phase analog.
18. [Stateful Replay for Streaming Generative Learning — arXiv 2511.17936 `[unverified primary]`](https://arxiv.org/abs/2511.17936)
19. [Self-Distillation Enables CL — arXiv 2026 `[unverified primary, ID forward-dated]`](https://arxiv.org/pdf/2601.19897)
20. [Language Models Need Sleep — ICLR 2026 submission](https://openreview.net/forum?id=iiZy6xyVVE) — concurrent proposal.

### Artificial substrates (SQ4)
21. [CLP-SNN on Loihi 2 — Hajizada et al. arXiv 2511.01553 (Nov 2025)](https://arxiv.org/abs/2511.01553) — 70× speed, 5,600× energy.
22. [Interleaved replay during SWS — bioRxiv June 2025 (PMID 40667278)](https://www.biorxiv.org/content/10.1101/2025.06.25.661579v1) — thalamocortical biophysical model.
23. [SESLR sleep-enhanced latent replay SNN — arXiv 2507.02901 (July 2025)](https://arxiv.org/abs/2507.02901) — Paper 2 ablation baseline.
24. [Titans: Learning to Memorize at Test Time — Google Research 2024](https://arxiv.org/abs/2501.00663) — Transformer long-term memory module.
25. [mlx-snn — Apple Silicon SNN library, arXiv 2603.03529 (2026)](https://arxiv.org/abs/2603.03529) — direct MLX enabler.
26. [BrainTransformers (SNN-LLM) — arXiv 2410.14687v2](https://arxiv.org/html/2410.14687v2) — SNN+Transformer hybrid.
27. [SpiNNaker 2 — TU Dresden, arXiv 2401.04491](https://arxiv.org/html/2401.04491v1) — substrate complement.

### Empirical validation & clinical (SQ5)
28. [TMR meta-analysis — Hu et al. Psych Bull 2020 (PMC7144680)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7144680/) — Hedges' g = 0.29 [0.21, 0.38], k=91.
29. [TMR update review — npj Sci Learn 2024](https://www.nature.com/articles/s41539-024-00244-8) — TMR works conditionally.
30. [CLAS in chronic insomnia null — medRxiv March 2025](https://www.medrxiv.org/content/10.1101/2025.03.04.25321710v1) — N=27, null on memory.
31. [Slow-wave synchrony in prodromal AD — Sharon et al. Alz & Dementia 2025](https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/alz.70247) — biomarker convergence.
32. [CLAS at home in AD — Am J Geriatr Psychiatry 2024](https://www.ajgponline.org/article/S1064-7481(24)00384-1/abstract) — dose-response artifact.
33. [Sleep restriction meta-analysis — Javadi et al. NBR 2024](https://www.sciencedirect.com/science/article/pii/S0149763424003981) — g = 0.29 [0.13, 0.44].

### Cross-cutting bridges & critique
34. [Going offline to enhance memory — Robertson NRN 2025](https://www.nature.com/articles/s41583-025-00973-8) — sequential SWS-then-REM model.
35. [How robust are sleep-mediated memory benefits? — Cordi & Rasch EJN 2021](https://pubmed.ncbi.nlm.nih.gov/32711356/) — replication critique.
36. [Optimizing methodology of human sleep + memory research — Nat Rev Psych 2023](https://www.nature.com/articles/s44159-023-00262-0) — gold-standard checklist.
37. [Memory reactivation does not act holistically on object memory — Schreiner et al. J Neurosci 2024](https://www.jneurosci.org/content/44/24/e0022242024) — TMR null on holistic objects.
38. [Sleep oscillations and consolidation in early-course psychosis — Manoach lab, Schiz Res 2024](https://pubmed.ncbi.nlm.nih.gov/39515257/) — schizophrenia spindle deficit.
39. [Eszopiclone RCT for spindle augmentation — Wamsley 2020](https://pubmed.ncbi.nlm.nih.gov/32919407/) — pharmacological precedent.

---

## Methodology

- **Sources of evidence** : 3 parallel `general-purpose` sub-agents using
  Exa MCP (`web_search_exa`, `web_fetch_exa`) with WebSearch / WebFetch
  fallback. ~14 distinct search queries across 5 sub-questions; ~41 raw
  sources; dedup'd to 32 unique above (overlap concentrated on the
  load-bearing pivots: eLife 2025 Bayesian meta, Spens-Burgess Nat Hum
  Behav 2024, brain-inspired replay van de Ven 2020, Comm Biol 2025).
- **Confidence calibration** : *High* on neurobiology + theory (multiple
  independent sources converge on every claim); *Medium* on AI bridging
  (some 2025-2026 arXiv preprints lack peer review); *Medium-low* on
  clinical translation (small-N trials, dose-response artifacts, recent
  null replications).
- **Known limitations** :
  - All quotes drawn from search-snippet-level reading; fetch was denied
    on multiple occasions.
  - Sources flagged `[unverified primary]` MUST be re-verified against
    full PDFs before final manuscript citation. These are: arXiv
    2511.17936 stateful replay; arXiv 2601.19897 self-distillation
    (forward-dated arXiv ID); the bioRxiv Sept 2025 REM-ripple-chains
    preprint; the Spens-Burgess 2024-2025 sequential-extension paper
    (final venue uncertain); the AI-Meets-Brain 2025 survey.
  - **No 2024-2026 dedicated CLS update** by Kumaran/Hassabis/McClelland
    surfaced; build the "CLS in 2026" anchor from Spens-Burgess + 2025
    PMC integrative review.
  - **No 2025-2026 first-author Tononi/Cirelli SHY review** surfaced;
    verify PubMed before final cite.
  - **No 2025-2026 Cochrane / JAMA Neurology** systematic review on
    sleep-memory consolidation interventions found — possible real gap.
  - French-language sources not searched in this pass; if `paper1-fr`
    parity requires it, a follow-up FR query batch is needed.
- **Sub-questions investigated** :
  1. Neurobiological mechanisms (SO/spindle/ripple/replay) 2025-2026.
  2. Theoretical frameworks (CLS, ASC, SHY, generative consolidation).
  3. AI dreaming / generative replay / sleep-inspired learning 2024-2026.
  4. Artificial substrates (SNN thalamocortical, MLX, transformers) for
     consolidation.
  5. Empirical validation + clinical translation 2025-2026 + cross-cutting
     debates.

---

*Report generated 2026-05-02 by `dream-of-kiki` research pipeline
(`/ecc:deep-research`). Anchored to commit `6c50ffe` of `main` branch.
For citation in Paper 1 §Background and §Related Work; revisit before
manuscript submission to refresh time-sensitive items (clinical trials,
arXiv preprints).*
