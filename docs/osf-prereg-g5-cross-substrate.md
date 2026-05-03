# OSF pre-registration — G5 cross-substrate pilot

**Date** : 2026-05-03
**Pre-registration target DOI** : 10.17605/OSF.IO/Q6JYN (parent;
G5 amendment to be uploaded **before** the production run)
**c_version** : C-v0.12.0+PARTIAL
**Substrate** : `esnn_thalamocortical`
**Parent pilots** :
- G4-bis (MLX, binary head, replay-coupled spectator on
  RESTRUCTURE/RECOMBINE) : `docs/milestones/g4-pilot-2026-05-03-bis.md`
  — H1+H2+H_DR4 null finding (g_h1 = -2.31, p_h1 = 1.0,
  spectator pattern across 4 arms).
- G4-ter (MLX, richer hierarchical head, HP sweep) :
  `docs/milestones/g4-ter-pilot-2026-05-03.md` — H2 strong reject
  on richer substrate (g_h2 = +2.77, p_h2 = 4.9e-14, n=30).

## §0 Adaptation note (G4-ter context, 2026-05-03)

This pre-registration was authored after G4-bis (binary MLP, null
finding) and G4-ter (richer hierarchical head, positive H2
finding) had both completed. The G5 plan was originally drafted to
benchmark binary-head retention against G4-bis only. We retain
that primary scope for G5 (binary-head cross-substrate
replication) so the comparison is apples-to-apples : same
classifier architecture, same Split-FMNIST 5-task protocol, same
seed family, same 4-arm sweep. G4-ter's positive richer-head
result is **not** the G5 baseline — porting the hierarchical
head + HP grid to E-SNN is scope-bounded follow-up ("G5-bis"). G5
therefore tests cross-substrate consistency on the G4-bis null
finding ; if E-SNN replicates the G4-bis null, that confirms DR-3
substrate-agnosticism on the *binary-head dispatch path*. The
positive-finding cross-substrate replication (richer head) is
explicitly deferred.

## §1 Purpose

Empirically validate DR-3 substrate-agnosticism by replicating the
G4-bis Split-FMNIST 5-task sweep on the E-SNN thalamocortical
substrate and statistically testing whether per-arm retention
distributions are consistent across the two substrates.

## §2 Sweep design

- Arms : `["baseline", "P_min", "P_equ", "P_max"]` (mirrors G4-bis).
- Seeds : `[0, 1, 2, 3, 4]` (5 seeds per arm).
- Total cells : 20.
- Substrate : `kiki_oniric.substrates.esnn_thalamocortical`,
  numpy-LIF backend (no Loihi-2 hardware involved).
- Classifier :
  `experiments.g5_cross_substrate.esnn_classifier.EsnnG5Classifier`
  — rate-coded SNN, in_dim = 28*28 = 784, hidden_dim = 64,
  n_classes = 2, n_steps = 20, tau = 10.0, threshold = 1.0.
- Dream wrapper :
  `experiments.g5_cross_substrate.esnn_dream_wrap.dream_episode`
  — DR-0 logging only at the dream-episode boundary, no classifier
  weight mutation through the runtime path. (G4-bis additionally
  runs a separate replay-coupling step *outside* the runtime ; G5
  intentionally isolates the runtime dispatch path so the
  cross-substrate comparison targets the substrate, not the
  coupling.)
- Pre-registered hypotheses (own-substrate) : H1, H3, H_DR4 from
  G4-bis carried over verbatim.
- Pre-registered hypotheses (cross-substrate, unique to G5) :
  - **H_DR3-CONSIST** : for each arm a in {baseline, P_min,
    P_equ, P_max}, Welch two-sided p > Bonferroni α/4 = 0.0125
    on `(MLX retention[a], E-SNN retention[a])`. Verdict
    `dr3_cross_substrate_consistency_ok = True` ⇔ all 4 arms pass.

## §3 Effect-size anchors

Same Hu 2020 (g = 0.29, CI [0.21, 0.38]) and Javadi 2024
(g = 0.29, CI [0.13, 0.44]) anchors as G4-bis for own-substrate
H1 / H3. H_DR3-CONSIST has no external anchor — the consistency
null is the substrate-agnosticism claim itself.

## §4 Power analysis

N = 5 per arm. Minimum detectable effect at 80 % power, two-sided
α = 0.0125 ≈ g ≈ 1.7. The pilot is therefore **exploratory** for
H_DR3-CONSIST : a non-rejection at this scale is consistent with
DR-3, not a strict confirmation. A confirmatory N ≥ 30 follow-up
is scheduled per G4-bis (same compute budget plan).

## §5 Exclusion rule

Cells with `acc_task1_initial < 0.5` are flagged
`excluded_underperforming_baseline = true` and dropped from the
verdict aggregation (mirrors G4-bis).

## §6 Outputs

- `docs/milestones/g5-cross-substrate-2026-05-03.{json,md}` —
  per-arm retention + own-substrate H1 / H3 / H_DR4 verdicts.
- `docs/milestones/g5-cross-substrate-aggregate-2026-05-03.{json,md}`
  — H_DR3-CONSIST cross-substrate verdict, comparing G5 (E-SNN)
  to G4-bis (MLX, binary head). Aggregator input :
  `docs/milestones/g4-pilot-2026-05-03-bis.json`.
- DR-3 evidence record :
  `docs/proofs/dr3-substrate-evidence.md` upgraded
  **conditionally** on `dr3_cross_substrate_consistency_ok = True`.

## §7 Amendments

This pre-registration is **append-only**. Any post-hoc change to
the sweep, classifier, or verdict logic requires a dated
amendment line below this section.
