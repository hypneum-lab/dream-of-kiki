# dream-of-kiki — Status

**As of** : 2026-04-20 PLOS CB pivot codified (cycle-3 plan adapted)
**Version** : C-v0.7.0+PARTIAL
**Phase** : Paper 1 v0.2 PLOS CB submission preparation. Cycle-3
Phase 1 1.5B sanity GO 3/3 (commit `22c58c9`, 46.75 min Studio) ;
Phase 2 multi-scale (7B + 35B) + Norse cross-substrate + fMRI
pilot **deferred to Paper 2** per
`docs/milestones/cycle3-plan-adaptation-2026-04-20.md` adaptation
matrix.

Active gate : **Paper 1 v0.2 PLOS CB submission preparation**
(arXiv prep + DR-2 external review + PLOS CB cover letter).

Public repo : https://github.com/electron-rare/dream-of-kiki
Public org repo : https://github.com/c-geni-al/dream-of-kiki

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

## DualVer status

| Axis | Value | Meaning |
|------|-------|---------|
| FC   | v0.7.0 | MINOR bump (H6 profile-ordering derived constraint surface added per framework-C §12.2 ; scale-axis glossary entry ; cycle-3 cross-scale DR-3 formal feature add) |
| EC   | PARTIAL | Cycle-3 Phase 1 sem 1-3 DONE (loaders + wrappers + real ops + scaling law + Bonferroni + 1080-runner + 1.5B sanity GO 3/3 + DualVer +PARTIAL bump). Cycle-3 Phase 2 sem 4-5 infra DONE (Norse wrapper + SNN ops + Studyforrest BOLD + HMM + CCA). Multi-scale empirical 7B + 35B, Gate D verdict, Norse cross-substrate pilot, fMRI pilot : deferred to Paper 2 per PLOS CB pivot 2026-04-20 (`docs/milestones/cycle3-plan-adaptation-2026-04-20.md`). STABLE → PARTIAL per framework-C §12.3 transition rule |

Next target : C-v0.7.0+STABLE deferred to Paper 2 closeout (Phase 2 multi-scale cells re-closed). Paper 1 v0.2 PLOS CB submission does **not** trigger STABLE promotion (scope reframing only).

## Gates

| Gate | Target week | Status |
|------|-------------|--------|
| G1 — T-Col fallback lock | S2 | ✅ LOCKED Branch A Studyforrest |
| G2 — P_min viable | S8 | ⏳ Pending S5-S8 |
| G3 — DR-2 proof peer-reviewed | S8 | ⏳ Draft S6 + review S6-S8 |
| G3-draft — DR-2 proof circulated | S6 | ⏳ Pending |
| G4 — P_equ fonctionnel | S12 | ⏳ Pending |
| G5 — PUBLICATION-READY | S18 | ⏳ Pending |
| G6 — Cycle 2 decision | S28 | ⏳ Pending |
| G7 — E-SNN substrate conformance | cycle-2 Phase 1 | ✅ LOCKED |
| G8 — P_max profile wired | cycle-2 Phase 2 | ✅ LOCKED |
| G9 — cycle-2 publication-ready | cycle-2 closeout | ✅ FULL-GO/STABLE |
| G10 — cycle-3 Gate D (H1-H6) | cycle-3 sem 3 | ⏸ DEFERRED to Paper 2 (PLOS CB pivot 2026-04-20 — only 1.5B cell available, multi-scale 7B+35B + Gate D moved to Paper 2 backlog) |
| Paper 1 v0.2 PLOS CB | 2026-04-20 → submission | ▶ ACTIVE (draft rendered `docs/papers/paper1/build/full-draft.pdf` 22 pages 296 KB ; arXiv prep + DR-2 external review + cover letter pending) |

## Critical risks watched

| ID | Risk | Status |
|----|------|--------|
| R-EXT-01 | fMRI lab outreach fail | **MITIGATED** via Branch A Studyforrest |
| R-CHA-01 | Cognitive overload > 15h/sem | Monitoring Dream-sync Monday |
| R-FRM-01 | DR-2 proof fails | Fallback DR-2' canonical order ready |
| R-IMP-01 | Swap guards too strict | Configurable thresholds, permissive start |
| R-CAL-01 | Paper 1 reject | Fallback PLoS CB / Cognitive Science |

## Outstanding human actions

1. **OSF upload** — follow `docs/osf-upload-checklist.md`,
   lock H1-H4 pre-registration before S5 experiments.
   Blocking : pre-reg confirmatory status for S5+ results.
2. **Emails T-Col fMRI labs** — Huth, Norman, Gallant outreach
   using `ops/formal-reviewer-email-template.md` adapted.
3. **Formal reviewer recruitment** — Q_CR.1 b, 3 candidates from
   academic network for DR-2 proof review (S3-S5 target).

## Open science artifacts (planned)

- [x] Repo public GitHub `electron-rare/dream-of-kiki`
- [ ] OSF project + pre-registration DOI (S3 human action)
- [ ] HuggingFace models `clemsail/kiki-oniric-{P_min,P_equ,P_max}` (S22)
- [ ] Zenodo DOI for harness (S22)
- [ ] Zenodo DOI for models (S22)
- [ ] Public dashboard `dream.saillant.cc` (S13+)

## License

- Code : MIT
- Docs : CC-BY-4.0
- Authorship byline (Paper 1) : dreamOfkiki project contributors
