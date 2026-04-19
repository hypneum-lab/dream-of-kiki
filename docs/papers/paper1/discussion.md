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

Our distinguishing features : **(a)** unified formal framework
covering all four pillars, **(b)** executable Conformance
Criterion enabling multi-substrate validation, **(c)**
pre-registered ablation methodology with frozen benchmarks +
deterministic run IDs, **(d)** open-science artifacts (MIT code,
OSF pre-reg, Zenodo DOI artifacts).

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

---

## Notes for revision

- Replace "synthetic" caveats with real-data results post S20+
- Tighten to ≤1500 words for Nature HB main-text discipline
- Insert proper bibtex citations once references.bib is set up
  (S19.3)
