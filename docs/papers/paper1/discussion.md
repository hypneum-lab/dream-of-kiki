# Discussion (Paper 1, draft S19.1)

**Target length** : ~1.5 pages markdown (≈ 1500 words)

---

## 8.1 Theoretical contribution

Our framework C-v0.5.0+STABLE is, to our knowledge, the first
executable formal framework for dream-based consolidation in
artificial cognitive systems. By axiomatizing the four pillars
(replay (DR-1), downscaling (DR-2), restructuring (DR-3),
recombination (DR-4)) as composable operations on a free
semigroup with additive budget (see DR-2 in
`docs/proofs/dr2-compositionality.md`), we make explicit what
prior work left implicit : the **order and composition** of
consolidation operations matters, and reasoning about their
interactions requires more than ad-hoc engineering choices.

The Conformance Criterion (DR-3) operationalizes
substrate-agnosticism : any substrate that satisfies signature
typing + axiom property tests + BLOCKING-invariant enforceability
inherits the framework's guarantees. This is qualitatively
different from prior frameworks that bind theory to a specific
implementation [Kirkpatrick 2017, van de Ven 2020] — implementation
details are discussed in Paper 2. The DR-4 profile
chain inclusion (P_min ⊆ P_equ ⊆ P_max) further structures the
ablation space such that experimental claims about richer
profiles do not inadvertently rely on weaker-profile invariants.

## 8.2 Empirical contribution

The synthetic ablation pipeline (S15.3, run_id
`syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`) demonstrates that the
statistical evaluation chain (Welch / TOST / Jonckheere /
one-sample t-test under Bonferroni correction) is end-to-end
operational on a 500-item stratified benchmark. Three of four
pre-registered hypotheses passed at α = 0.0125 (H1 forgetting
reduction, H4 energy budget compliance, H2 self-equivalence
smoke), with H3 monotonic trend reaching the conventional 0.05
threshold but borderline at the corrected level.

While the values reported are synthetic placeholders pending real
mega-v2 + MLX-inferred predictor integration (S20+), the
**measurement infrastructure** itself is validated : the
RetainedBenchmark loader with SHA-256 integrity, the
`evaluate_retained` predictor bridge, the AblationRunner harness,
and the four statistical wrappers all interoperate cleanly. The
synthetic batch above is registered under
profile `G4_ablation` in the project registry so the JSON dump
remains traceable. Reproducibility contract R1 (deterministic
`run_id` from (c_version, profile, seed, commit_sha)) is enforced
by the run registry.

## 8.3 Limitations

Three limitations bound the cycle-1 contribution :

**(i) Synthetic data caveats.** All quantitative results in §7
are produced by mock predictors at scripted accuracy levels
(50% baseline, 70% P_min, 85% P_equ; run_id
`syn_s15_3_g4_synthetic_pipeline_v1`). They validate the
*pipeline*, not the *consolidation efficacy*. Real
mega-v2 + MLX-inferred predictors land cycle 1 closeout (S20+)
or cycle 2 ; until then, all numbers should be read as
infrastructure-validation evidence only.

**(ii) Single-substrate validation.** A single substrate is
exercised in cycle 1. While DR-3 Conformance Criterion is
formulated to be substrate-agnostic, only one instance has
passed all three conformance conditions. Cycle 2 introduces an
additional substrate to test the substrate-agnosticism claim
empirically per the DR-3 Conformance Criterion.

**(iii) P_max skeleton only.** The P_max profile is declared via
metadata (target ops, target channels) but its handlers are
not wired. Hypothesis H2 (P_max equivalence vs P_equ within ±5%)
is therefore tested only as a self-equivalence smoke test in
cycle 1. Real H2 evaluation requires P_max real wiring (cycle 2).

## 8.4 Comparison with prior art

| Prior work | Contribution | dreamOfkiki addition |
|-----------|--------------|----------------------|
| van de Ven 2020 | Generative replay | Composability + DR-2 axiom + Conformance |
| Kirkpatrick 2017 (EWC) | Synaptic consolidation regularizer | EWC subsumed under B-Tononi SHY operation in framework |
| Tononi & Cirelli 2014 (SHY) | Theoretical claim of synaptic homeostasis | Operationalized as `downscale` operation with non-idempotent property |
| Friston 2010 (FEP) | Free energy principle | Operationalized as `restructure` operation with topology guard S3 |
| Hobson 2009 (REM) | Creative dreaming theory | Operationalized as `recombine` operation with VAE-light skeleton |
| McClelland 1995 (CLS) | Two-system hippocampus + neocortex | Embedded in profile inclusion DR-4 (P_min minimal vs P_equ richer) |
| Huh 2024 (PRH, ICML 2024) | Convergent representation across model scales / modalities | Theoretical anchor for DR-3 substrate-agnosticism ; companion empirical probe in nerve-wml v1.7.0 (Zenodo DOI 10.5281/zenodo.19656342) |
| @liu2021dvnc (Discrete-Valued Neural Communication, NeurIPS 2021) | VQ-VAE shared global codebook for discrete messages between neural modules | Closest prior art for cross-module discrete communication ; DVNC uses one shared codebook and homogeneous modules, whereas our substrate-portability claim (DR-3) and the companion nerve-wml protocol use per-substrate codebooks with learned transducers and typed prediction/error roles — DVNC is a special case (single substrate, single codebook) of the more general portability our Conformance Criterion targets |
| @pedersen2024nir (Neuromorphic Intermediate Representation, Nature Communications 2024) | Static computational-graph IR : a hardware-agnostic instruction set for compiling one spiking model onto many neuromorphic backends | Operates at the *model-graph* layer ; orthogonal to our framework, which composes *consolidation operations* over a substrate and is agnostic to the IR that lowers any one substrate to hardware — NIR could serve as the compilation target for our E-SNN substrate (§9.1) without altering the axioms |
| @aer2025biohybrid (Address-Event Representation over UDP/Ethernet, arXiv:2501.09128, 2025) | Transport / encoding layer specifying how spike events are timestamped, addressed and physically carried, including biohybrid silicon–wetware links | Operates at the *transport* layer ; orthogonal to our framework — the Conformance Criterion constrains the semantics of consolidation operations, not the wire protocol that moves spikes, and AER is one admissible carrier for a conformant E-SNN substrate |
| @liu2024hybrid (Advancing brain-inspired computing with hybrid neural networks, National Science Review 2024) | Survey arguing hybrid ANN/SNN systems are the path to brain-inspired computing | Canonical hybrid-neural-network anchor for our cross-substrate claim : our cycle-1 validation across an MLX dense substrate and a LIF spiking substrate is precisely a hybrid-neural-network configuration, and the executable Conformance Criterion supplies the contract such hybrids currently lack |

Our distinguishing features : **(a)** unified formal framework
covering all four pillars, **(b)** executable Conformance
Criterion enabling multi-substrate validation, **(c)**
pre-registered ablation methodology with frozen benchmarks +
deterministic run IDs, **(d)** open-science artifacts (MIT code,
OSF pre-reg, Zenodo DOI artifacts).

The Platonic Representation Hypothesis (PRH) provides a falsifiable
theoretical floor under DR-3 : if convergent representations across
substrates is a real phenomenon, conformance transfer across the
MLX, E-SNN and LoRA substrates is *expected* rather than a happy
accident. The cross-repo `nerve-wml` PRH probe (GammaThetaMultiplexer
experiment, v1.7.0) is the empirical complement to the conformance
matrix reported here.

## 8.5 Cross-substrate preliminary replication (cycle 2)

Cycle 2 operationalizes limitation (ii) above by wiring a second
substrate — `esnn_thalamocortical`, a numpy LIF spike-rate
skeleton — alongside the canonical `mlx_kiki_oniric` substrate.
The DR-3 Conformance Criterion is re-evaluated on both substrates
(see `docs/milestones/conformance-matrix.md` and
`docs/proofs/dr3-substrate-evidence.md`), and the cycle-1 H1-H4
statistical chain is re-run per substrate
(`docs/milestones/cross-substrate-results.md`, runner
`scripts/ablation_cycle2.py`).

**Synthetic substitute — not empirical claim.** The two substrate
rows share the same Python mock predictor in cycle 2 : substrate-
specific inference is deferred to cycle 3. Consequently the
cross-substrate verdict is trivially agreeing by construction, and
the pipeline emits identical H1-H4 p-values on both substrates
(3 / 4 significant at Bonferroni α = 0.0125, H3 monotonic failing
on both due to constant mock dispersion). This **strengthens but
does not substitute for** the cycle-1 H1-H4 results reported in
§7 and §8.2. What it *does* demonstrate is that the framework's
conformance artifacts (typed Protocols, axiom property tests, S2/S3
guards) and the statistical evaluation chain execute end-to-end on
a structurally independent second substrate registration, which is
the architectural claim of DR-3. A divergent-predictor replication
on real biological or neuromorphic data is the cycle-3 target.

## 8.6 Empirical anchors and limits

Framework C makes claims about substrate-agnostic consolidation
invariants, but its biological grounding rests on a literature
whose **clinical translation remains fragile**. The strongest
2025 sham-controlled trial of closed-loop auditory stimulation
(CLAS) in chronic insomnia (N=27, crossover) reported acute
slow-oscillation amplitude gain without behavioural memory or
sleep-outcome benefit [@medrxiv2025clasinsomnia], a critical
disconfirmation of the simplistic "more SO ⇒ more consolidation"
narrative. Multi-night home interventions in Alzheimer's disease
expose a damaging dose-response artifact : patients with the
lowest baseline slow-wave sleep — the population most needing the
intervention — receive the fewest stimulations
[@ajgp2024clashome], a floor effect we treat as exogenous to the
framework's finite-budget invariant K1.

At the mechanism level, only ~10–30 % of sleep sharp-wave ripples
carry detectable replay content [@annurev2025replay], a
selectivity that DR-2 (downscaling) and the wider replay-driven
narrative must not over-claim. Schreiner et al. [@schreiner2024jneurosci]
further show that targeted memory reactivation does **not** act
holistically on object memory during sleep, fuelling the still-open
debate about which memory features are sleep-sensitive. The CLAS
replication landscape itself is heterogeneous and age-attenuated,
with weaker effects on procedural tasks [@npjscilearn2025clas].

Cordi & Rasch's 2021 critique [@cordi2021robust] remains the
load-bearing counterweight to enthusiastic narratives — sleep
effects on memory are smaller, more task-dependent, less
SWS-related, less robust and less long-lasting than previously
assumed. Any future empirical work in this programme will adopt
the gold-standard methodology checklist of *Nature Reviews
Psychology* 2023 [@natrevpsych2023methodology] (sham-controlled
designs, dose-response reporting, awareness of the
sequential-design / retrieval-vs-restudy confound). These
boundary conditions tighten the scope of every empirical claim
the framework can support and the kind of cycle-3 substrate
validation that would count as confirmatory.

---

## Notes for revision

- Replace "synthetic" caveats with real-data results post S20+
- Tighten to ≤1500 words for Nature HB main-text discipline
- Insert proper bibtex citations once references.bib is set up
  (S19.3)
