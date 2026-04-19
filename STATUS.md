# dream-of-kiki — Status

**As of** : 2026-04-19 end of cycle-2 closeout
**Version** : C-v0.6.0+PARTIAL
**Phase** : cycle-2 engineering Phases 1+2+5 complete ; Phase 3
(cross-substrate ablation) + Phase 4 (Paper 2 narrative) deferred
per user scope decision

Public repo : https://github.com/electron-rare/dream-of-kiki

---

## Program progress

Cycle 1 calendar : 28 weeks total (S1-S28) — closed at C-v0.5.0+STABLE
Cycle 2 calendar : Phases 1, 2 and 5 complete ; Phases 3-4
(publication track) deferred ; G9 = CONDITIONAL-GO/PARTIAL
Active gate : **G9 cycle-2 publication-ready (CONDITIONAL-GO/PARTIAL)**

## Test suite

```
173 tests passing
coverage 91.26% (gate 90%)
```

## DualVer status

| Axis | Value | Meaning |
|------|-------|---------|
| FC   | v0.6.0 | MINOR bump (DR-3 extended to E-SNN, P_max wired, DR-2 reframed as DR-2', RNG isolated, S4 read-only) |
| EC   | PARTIAL | Engineering deliverables green, publication track (Phases 3+4) deferred |

Next target : C-v0.7.0+STABLE post-G3 (DR-2 external reviewer feedback returned, parallel monoidal model in g3-draft)

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
| G9 — cycle-2 publication-ready | cycle-2 closeout | ⚠️ CONDITIONAL-GO/PARTIAL |

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
