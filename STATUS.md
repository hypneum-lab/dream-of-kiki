# dream-of-kiki — Status

**As of** : 2026-05-04 G4-septimo Tiny-ImageNet H6-C UNIVERSALITY CONFIRMED (240 cells N=30 Studio M3 Ultra ~3h11 wall, 54-58 s/cell) — H6-B confirmed (Welch p=0.9247, g=-0.0246, mean_mog=0.3864 vs mean_none=0.3891 on Split-Tiny-ImageNet G4MediumCNN), conjunction `H6-A AND H6-B` resolves to **CONFIRMED**, RECOMBINE-empty universality flag now spans full pre-registered scope {Split-FMNIST, Split-CIFAR-10, Split-CIFAR-100, Split-Tiny-ImageNet} × {3-layer MLP, 5-layer MLP, small CNN, medium CNN} ; DR-4 evidence v0.6 amended : DR-4 prediction "richer ops yield richer consolidation" empirically refuted across entire escalation ladder ; STABLE promotion blocked at this scope ceiling, transformer/hierarchical-E-SNN/ImageNet-1k/real-LLM remain open ; EC stays PARTIAL, FC stays C-v0.12.0. Prior : G4-sexto Step 1 CIFAR-100 100-class pilot (N=30 Option B, 240 cells, 80 min M1 Max) — **H6-A confirmed** (Welch p=0.8450, g=0.057, mean_mog=0.3622 vs mean_none=0.3580 on Split-CIFAR-100 G4SmallCNN), H6-B deferred (Tiny-IN locked under Option B → G4-septimo), H6-C universality conjunction deferred ; DR-4 evidence v0.5 amended : empirical-emptiness scope extended from {FMNIST, CIFAR-10} to {FMNIST, CIFAR-10, CIFAR-100} × {3-layer MLP, 5-layer MLP, small CNN}, robust to 10× class-budget scaling ; EC stays PARTIAL, FC stays C-v0.12.0 ; confirmatory N=95 Studio re-confirmed H6-A (760 cells, 76 effective per arm, mog=0.3701 vs none=0.3592, g=0.153, Welch p=0.3457, fail-to-reject at α=0.0167, empirical-emptiness claim survives precision increase). Prior : Studio CPU exploratory pilots (3 small pilots in parallel with G4-sexto Studio N=95 GPU run, ~6 min total wall) — (1) **K2 real-substrate** validation on 24 modules sampled from SpikingKiki-V4 35B-A3B (31 070 modules, LIF metadata applied), 276 pairwise MVL : mean=0.3855 inside K2 invariant band [0.27, 0.39] (eLife 2025 BF=58 SO-spindle anchor), 45.7% pairs strictly within → *partial empirical support* on real substrate ; (2) **R1 cross-machine verification** : `tests/reproducibility/` 9/9 PASS on Studio Python 3.14.4/MLX 0.31.1 with hashes identical to M1 Max Python 3.12 baseline → `golden_hashes.json` status promoted from `pending_review` to `validated_cross_machine_2026-05-04` (caveat : both Apple Silicon, not Linux/CUDA) ; (3) **Robertson 2018 sequential-ordering test** : 6 perms × 5 seeds = 30 cells, max pairwise Hedges' g vs canonical = 0.079 → H_RO-A (permutation effect SMALL, |g| < 0.2), descriptive support for DR-3 intra-cycle ordering corollary. All exploratory, no FC/EC bump. Prior : G5-ter (H8-C partial, CNN closes ~2/3 cross-substrate gap), G5-bis (H7-B MLX-only artefact for LIF MLP), G4-quinto (H5-C CONFIRMED, RECOMBINE-empty universalises FMNIST+CIFAR-CNN), G4-quater (H4-C CONFIRMED on FMNIST 3-layer).
**Version** : C-v0.12.0+PARTIAL
**Phase** : Paper 1 v0.2 PLOS CB submission preparation. Cycle-3
Phase 1 1.5B sanity GO 3/3 (commit `22c58c9`, 46.75 min Studio) ;
Phase 2 multi-scale (7B + 35B) + Norse cross-substrate + fMRI
pilot **deferred to Paper 2** per
`docs/milestones/cycle3-plan-adaptation-2026-04-20.md` adaptation
matrix.

Active gate : **Paper 1 v0.2 PLOS CB submission preparation**
(arXiv prep + DR-2 external review + PLOS CB cover letter).

Public repo : https://github.com/electron-rare/dream-of-kiki
Public org repo : https://github.com/hypneum-lab/dream-of-kiki

---

## Program progress

Cycle 1 calendar : 28 weeks total (S1-S28) — closed at C-v0.5.0+STABLE
Cycle 2 calendar : all 5 phases delivered (Phase 1 E-SNN substrate,
Phase 2 P_max wiring, Phase 3 cross-substrate ablation, Phase 4
Paper 2 narrative, Phase 5 async worker + closeout) ;
G9 = FULL-GO/STABLE
Cycle 3 calendar : 6 weeks (S47-S52) — **adapted 2026-04-20 for
PLOS CB pivot**. Phase 1 sem 1-3 DONE (loaders + Qwen MLX wrappers
+ real ops + scaling law + Bonferroni + 1080-runner + 1.5B sanity
GO 3/3 + DualVer +PARTIAL). Phase 2 sem 4-5 infra DONE (Norse
wrapper + SNN ops + Studyforrest BOLD + HMM + CCA). Multi-scale
empirical (7B + 35B), Gate D, Norse cross-substrate pilot, fMRI
pilot, Paper 1 v2 narrative : **all deferred to Paper 2**. Paper 1
v0.2 retargeted PLOS Computational Biology (commit `d6866f3`),
draft rendered (`docs/papers/paper1/build/full-draft.pdf`,
22 pages). Active gate : **Paper 1 v0.2 PLOS CB submission
preparation** (G10 cycle-3 Gate D folded into Paper 2 backlog).

## Test suite

```
277 tests passing
coverage 91.17% (gate 90%)
```

R1 output-hash API landed (`register_output_hash` /
`get_output_hash` on `RunRegistry` ; sibling table
`run_output_hashes`). The `test_r1_registry_output_hash_contract`
contract test is no longer `xfail` — second half of R1 is now
caller-enforceable. Multi-artifact support landed (issue #2) :
schema keyed on `(run_id, artifact_name)`, idempotent migration
from the v1 single-hash layout, plus `list_output_hashes(run_id)`.

**2026-04-21 DualVer bump (FC-PATCH)**: DR-2 weakened with precondition excluding RESTRUCTURE-before-REPLAY permutations. See CHANGELOG `[C-v0.7.1+PARTIAL]` and amendment doc.

## DualVer status

| Axis | Value | Meaning |
|------|-------|---------|
| FC   | v0.12.0 | MINOR bump (Wake-Sleep CL ablation baseline added per Alfarano 2024 [IEEE TNNLS, arXiv 2401.08623], `eval-matrix.yaml` `baselines:` block additive, variant-c published-reference, 2026-05-03). Prior bumps: v0.11.0 (K2 phase-coupling invariant + opt-in PhaseCouplingObservable Protocol, eLife 2025 BF=58 anchor, 2026-05-02), v0.7.0 (cycle-3 launch, H6 profile-ordering + scale-axis), v0.7.1 (DR-2 FC-PATCH weakening, 2026-04-21), v0.8.0 (kiki_oniric.axioms public API, 2026-04-21), v0.9.0 (micro-kiki LoRA substrate, PR #11, 2026-04-22 ; substrate-internal patch C-v0.8.1 for SpikingKiki-V4 shim PR #12 same day), v0.10.0 (micro-kiki recombine TIES-merge handler, PR #13, 2026-04-22) |
| EC   | PARTIAL | Cycle-3 Phase 1 sem 1-3 DONE (loaders + wrappers + real ops + scaling law + Bonferroni + 1080-runner + 1.5B sanity GO 3/3 + DualVer +PARTIAL bump). Cycle-3 Phase 2 sem 4-5 infra DONE (Norse wrapper + SNN ops + Studyforrest BOLD + HMM + CCA). Multi-scale empirical 7B + 35B, Gate D verdict, Norse cross-substrate pilot, fMRI pilot : deferred to Paper 2 per PLOS CB pivot 2026-04-20 (`docs/milestones/cycle3-plan-adaptation-2026-04-20.md`). STABLE → PARTIAL per framework-C §12.3 transition rule |

Next target : C-v0.12.0+STABLE deferred to Paper 2 closeout (Phase 2 multi-scale cells re-closed). Paper 1 v0.2 PLOS CB submission does **not** trigger STABLE promotion (scope reframing only).

## Gates

| Gate | Target week | Status |
|------|-------------|--------|
| G1 — T-Col fallback lock | S2 | ✅ LOCKED Branch A Studyforrest |
| G2 — P_min viable | S8 | ⏳ Pending S5-S8 |
| G3 — DR-2 proof peer-reviewed | S8 | ⏳ Draft S6 + review S6-S8 |
| G3-draft — DR-2 proof circulated | S6 | ⏳ Pending |
| G4 — P_equ fonctionnel | S12 | 🔶 G4-quinto PARTIAL (2026-05-03 — H5-A False, H5-B False, H5-C **CONFIRMED** (Welch fail-to-reject mog vs none on CNN, p=0.9918 g=-0.0026 N=30) ; RECOMBINE-empty universalises across 2 benchmarks × 2 substrates ; DR-4 partial refutation **universalises** across FMNIST + CIFAR-CNN ; EC stays PARTIAL per pre-reg §6 ; FC stays C-v0.12.0 ; promotion to STABLE blocked pending ImageNet / transformer / hierarchical E-SNN follow-up. Prior : G4-quater 2026-05-03 PARTIAL — H4-A False, H4-B False, H4-C **CONFIRMED** (FMNIST 3-layer Welch p=0.989 g=0.002 N=95)). |
| G5 — PUBLICATION-READY | S18 | ⏳ Pending |
| G6 — Cycle 2 decision | S28 | ⏳ Pending |
| G7 — E-SNN substrate conformance | cycle-2 Phase 1 | ✅ LOCKED |
| G8 — P_max profile wired | cycle-2 Phase 2 | ✅ LOCKED |
| G9 — cycle-2 publication-ready | cycle-2 closeout | ✅ FULL-GO/STABLE |
| G10 — cycle-3 Gate D (H1-H6) | cycle-3 sem 3 | ⏸ DEFERRED to Paper 2 (PLOS CB pivot 2026-04-20 — only 1.5B cell available, multi-scale 7B+35B + Gate D moved to Paper 2 backlog) |
| Paper 1 v0.2 PLOS CB | 2026-04-20 → submission | ▶ ACTIVE (draft rendered `docs/papers/paper1/build/full-draft.pdf` 22 pages 296 KB ; arXiv prep + DR-2 external review + cover letter pending) |
| G6 — micro-kiki real LLM CL stream | 2026-05-03 → milestone | 🔶 PARTIAL (Path B inference-only, M1 Max ; pre-reg amended for G4-bis g_h1=-2.31 ; spectator pattern observed across 4 arms ; H_NEW reformulated as exploratory infrastructure validation per `docs/osf-prereg-g6-pilot.md` §0 + §2 ; Path A Studio = future work, blocking STABLE promotion per pre-reg §6) |
| G5-pilot — cross-substrate DR-3 empirical | 2026-05-03 → milestone | 🔶 PARTIAL (E-SNN thalamocortical, numpy LIF, 20 cells, 4 min M1 Max ; within-substrate spectator pattern reproduces G4-bis (P_min ≡ P_equ ≡ P_max retention 0.5119) ; cross-substrate Welch vs MLX rejects H0 at α/4=0.0125 (baseline g=9.98, dream arms g=3.49) ; absolute retention diverges, qualitative spectator pattern matches ; no DR-3 evidence upgrade at absolute level ; G5-bis with G4-ter richer head = future work) |
| G5-bis — richer head cross-substrate | 2026-05-03 → milestone | 🔶 PARTIAL (E-SNN richer LIF, 40 cells Option B, ~16 min M1 Max ; own-substrate g_h7a=0.1043 fail-to-reject Welch p=0.4052 below H7B threshold 0.5 ; cross-substrate aggregate vs G4-ter MLX richer head all 4 arms reject H0 at α/4=0.0125 (baseline g=3.23, dream arms g=4.02, P_min g=4.20) ; **classification H7-B = MLX-only artefact**, G4-ter +2.77 effect does NOT transfer to E-SNN ; DR-3 evidence revised : axiom-test-level guarantee preserved, empirical effect-size transferability refuted at this N ; confirmatory N=30 Option A + spiking-CNN G5-ter + ImageNet escalation = future work) |
| G5-ter — spiking-CNN cross-substrate H8 | 2026-05-03 → milestone | 🔶 PARTIAL (E-SNN spiking-CNN 4 layers Conv-LIF + Conv-LIF + avg-pool + FC-LIF + Linear, STE backward, pure-numpy Conv2d ; 40 cells Option B, 36 min M1 Max, train shard subsampled to 1500/task per documented amendment D-1 ; own-substrate g_h8=-0.1093 fail-to-reject Welch p=0.5992 below H7B threshold 0.5 ; cross-substrate aggregate vs G4-quinto Step 2 MLX small-CNN all 4 arms reject H0 at α/4=0.0125 (g_mlx_minus_esnn ∈ [+1.21, +1.32], g_p_equ_cross=+1.31 between H8-A floor 2.0 and H8-B ceiling 1.0) ; **classification H8-C = partial**, architecture closes ~2/3 of G5-bis cross-substrate level gap (+4.02 → +1.31 at P_equ) but cycle-3 positive effect remains absent on E-SNN regardless of architecture ; DR-3 evidence revised : empirical effect-size transferability now refuted at two architectural depths ; confirmatory N=30 Option A = future work — see `docs/milestones/g5-ter-aggregate-2026-05-03.md`) |

## Critical risks watched

| ID | Risk | Status |
|----|------|--------|
| R-EXT-01 | fMRI lab outreach fail | **MITIGATED** via Branch A Studyforrest |
| R-CHA-01 | Cognitive overload > 15h/sem | Monitoring Dream-sync Monday |
| R-FRM-01 | DR-2 proof fails | Fallback DR-2' canonical order ready |
| R-IMP-01 | Swap guards too strict | Configurable thresholds, permissive start |
| R-CAL-01 | Paper 1 reject | Fallback PLoS CB / Cognitive Science |

## Outstanding human actions

1. **OSF upload + Amendment #1** — DONE. Primary DataCite DOI
   `10.17605/OSF.IO/Q6JYN` auto-registered 2026-04-19T00:28:05Z.
   Amendment #1 (Bonferroni family restructure for cycle-3)
   filed 2026-04-21 as Open-Ended Registration at DOI
   `10.17605/OSF.IO/TPM5S` (https://osf.io/tpm5s/), linked to
   parent Q6JYN. Paper 2 H1-H4 confirmatory status unlocked for
   S5+ runs. Both registrations accessible via the project page
   https://osf.io/q6jyn.
2. **Emails T-Col fMRI labs** — Huth, Norman, Gallant outreach
   using `ops/formal-reviewer-email-template.md` adapted.
3. **Formal reviewer recruitment** — Q_CR.1 b, 3 candidates from
   academic network for DR-2 proof review (S3-S5 target).

## Open science artifacts (planned)

- [x] Repo public GitHub `electron-rare/dream-of-kiki`
- [x] OSF project + pre-registration DOI `10.17605/OSF.IO/Q6JYN` (DataCite-minted 2026-04-19T00:28:05Z on OSF publish)
- [ ] HuggingFace models `clemsail/kiki-oniric-{P_min,P_equ,P_max}` (S22)
- [ ] Zenodo DOI for harness (S22)
- [ ] Zenodo DOI for models (S22)
- [ ] Public dashboard `dream.saillant.cc` (S13+)

## License

- Code : MIT
- Docs : CC-BY-4.0
- Authorship byline (Paper 1) : dreamOfkiki project contributors
