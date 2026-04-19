# Cross-substrate H1-H4 results (C2.11, cycle 2)

> **(synthetic substitute — not empirical claim)**  All numbers
> below are produced by the mock predictors shared across substrate
> labels in `scripts/ablation_cycle2.py`. They validate the
> **cross-substrate replication pipeline** (runner wiring + H1-H4
> statistical chain applied per substrate) and do **not** carry a
> biological or hardware claim. Real wiring to substrate-specific
> inference is deferred post-cycle-2.

## Context

DR-3 (Conformance Criterion, spec §6.2) claims substrate-
agnosticism. C2.9 delivered a multi-substrate ablation runner ;
C2.10 delivered the static conformance matrix ; C2.11 closes the
loop by re-running the cycle-1 **H1-H4 statistical chain** on
both real substrates and comparing verdicts.

The statistical chain is identical to the cycle-1 G4 gate :

- **H1** forgetting — Welch's one-sided t-test (treatment = P_equ,
  control = baseline).
- **H2** self-equivalence — TOST with ε = 0.05.
- **H3** monotonicity — Jonckheere-Terpstra trend.
- **H4** energy budget — one-sample t-test (upper bound 2.0).

Bonferroni α = 0.05 / 4 = **0.0125** per the OSF pre-registration.
Stats implementation : `kiki_oniric/eval/statistics.py`.

## Provenance (synthetic substitute — not empirical claim)

| field | value |
|-------|-------|
| harness_version | `C-v0.6.0+PARTIAL` |
| cycle2_batch_id | `3a94254190224ca82c70586e1f00d845` |
| ablation_runner_run_id | `45eccc12953e758440fca182244ddba2` |
| benchmark | mega-v2 stratified, 500 synthetic items |
| benchmark_hash | `synthetic:c8a0712000b641...` |
| seeds | `[42, 123, 7]` |
| substrates | `mlx_kiki_oniric`, `esnn_thalamocortical` |
| data_provenance | synthetic — mock predictors shared across substrate labels |

## Comparative table MLX vs E-SNN (synthetic substitute — not empirical claim)

| hypothesis | test | MLX p-value | MLX verdict | E-SNN p-value | E-SNN verdict | agree |
|------------|------|-------------|-------------|---------------|---------------|-------|
| H1 forgetting | Welch one-sided | 0.0000 | reject H0 | 0.0000 | reject H0 | YES |
| H2 self-equivalence | TOST (ε=0.05) | 0.0000 | reject H0 | 0.0000 | reject H0 | YES |
| H3 monotonicity | Jonckheere-Terpstra | 0.0248 | fail to reject | 0.0248 | fail to reject | YES |
| H4 energy budget | one-sample t (upper) | 0.0101 | reject H0 | 0.0101 | reject H0 | YES |

- MLX significant count : **3 / 4**
- E-SNN significant count : **3 / 4**
- **Fully consistent across substrates** : **YES**

Caption : *synthetic substitute — not empirical claim*. H3 fails
at α = 0.0125 on both substrates because the mock predictors emit
a constant per-profile accuracy (no per-seed dispersion), so the
Jonckheere-Terpstra trend statistic cannot cross the Bonferroni
threshold. This is a property of the mock, not of the framework.

## Agreement matrix (synthetic substitute — not empirical claim)

| hypothesis | verdicts equal? | MLX reject | E-SNN reject |
|------------|-----------------|------------|--------------|
| H1 forgetting | YES | true | true |
| H2 self-equivalence | YES | true | true |
| H3 monotonicity | YES | false | false |
| H4 energy budget | YES | true | true |

All four hypotheses agree across substrates (4 / 4 agreement).
This is the expected outcome when the underlying predictor is
identical across substrate rows — it shows the per-substrate
stats path is wired correctly and emits identical verdicts on
identical inputs. A divergent-predictor replication is the
cycle-3 real-wiring target.

## Reproduction

```bash
uv run python scripts/ablation_cycle2.py
```

- Runs the substrate × profile × seed grid (2 × 3 × 3 = 18 cells).
- Runs H1-H4 per substrate via `kiki_oniric.eval.statistics`
  (Welch one-sided, TOST, Jonckheere-Terpstra, one-sample t).
- Writes JSON + Markdown to `docs/milestones/ablation-cycle2-
  results.{json,md}`.

Seed grid `[42, 123, 7]` matches cycle-1. Benchmark hash is
deterministic, so the dump is byte-stable as long as the predictors
and the seed list are frozen.

## Scope and caveats (synthetic substitute — not empirical claim)

1. **Predictor is shared.** Until a substrate-specific predictor
   is wired, the two substrate rows are an *identity label* only.
   The consistency verdict is trivially YES by construction ; the
   test value is that the **pipeline** (runner → stats → dump)
   executes end-to-end on two distinct substrate registrations.
2. **No real cohort, no real HW.** The E-SNN substrate is a numpy
   LIF spike-rate skeleton (see
   `kiki_oniric/substrates/esnn_thalamocortical.py` docstring).
   No Loihi-2, no fMRI.
3. **Does not substitute for cycle-1 H1-H4.** The cycle-1 results
   (G4 gate, `docs/milestones/ablation-results.md`) remain the
   primary evidence for the hypotheses themselves. The cycle-2
   dump strengthens the *substrate-agnostic architecture* claim.
4. **Bonferroni α = 0.0125** is enforced per hypothesis, matching
   the OSF pre-registration.

## Cross-references

- Spec : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §6.2
- Runner : `scripts/ablation_cycle2.py`
- Conformance matrix : `docs/milestones/conformance-matrix.md`
- DR-3 evidence : `docs/proofs/dr3-substrate-evidence.md`
- Stats : `kiki_oniric/eval/statistics.py`
- Cycle-1 H1-H4 baseline : `docs/milestones/ablation-results.md`
