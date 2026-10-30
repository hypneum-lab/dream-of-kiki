# OSF filing note — G4-nono cross-domain RECOMBINE (H12)

**Status:** prepared 2026-10-28 for the PI, **not submitted**. One PI
decision remains before filing: the filing host (§10 item 3, below).
The PI files on OSF himself. No confirmatory cell runs before the PI confirms
the filing.

## What is filed

| item | value |
|---|---|
| Title | G4-nono cross-domain RECOMBINE pre-registration (H12-A..C) |
| Authors | Clément Saillant (L'Electron Rare, Grandris, France), as in `CITATION.cff` |
| Parent OSF project | 10.17605/OSF.IO/Q6JYN (`https://osf.io/q6jyn`) |
| Source draft | `docs/osf-prereg-g4-nono-cross-domain-2026-10-23.md` |
| Lock commit | `14c3d5f864430649ea860e3b49d10528065af798`, merge of PR #71 into `main` (CI green: push run 40733556724 on `main`, success) |
| Draft SHA-256 | `9c514909289a231f122e52dda1266f01b332f052c6f430392c1a443a57e9b28a` |
| Frozen copy | `docs/osf-filing/g4-nono/osf-prereg-g4-nono-cross-domain-2026-10-23.frozen.md` (byte-identical to the draft at the lock commit, same SHA-256) |
| PDF rendering | `docs/osf-filing/g4-nono/osf-prereg-g4-nono-cross-domain-2026-10-23.frozen.pdf` (SHA-256 in `SHA256SUMS`) |
| Companion erratum | `docs/osf-prereg-erratum-g4-power-2026-10-23.md`, SHA-256 `c15c202f8706dc2155e5b9c5629869c7275438f7a66270fc8a9286db1c1371ed`; frozen copy and PDF in this folder. It corrects the power sentences of the **locked** G4-quinto / sexto / septimo / octavo and G5-bis pre-registrations (which are not edited) and is filed next to them; the G4-nono draft cites it. |
| Rendering | `render.sh` + `render.lua` (verbatim copy of the E1-small filter) + `strikeout.lua` (struck-through PI decisions keep every character): pandoc 3.12.1, xelatex (TeX Live 2026), layout only. The PDF is not byte-reproducible (embedded dates); the Markdown and its SHA-256 are the reference. |
| Merge PR | `https://github.com/hypneum-lab/dream-of-kiki/pull/71` (split of G4-nono + DR-3 out of PR #41, with the smoke records) |
| Working PR | `https://github.com/hypneum-lab/dream-of-kiki/pull/41` (history; keeps the E1 35B draft) |
| Commit | `https://github.com/hypneum-lab/dream-of-kiki/commit/14c3d5f864430649ea860e3b49d10528065af798` |
| Power script | `scripts/prereg_power_2026_10.py` (§3) at the lock commit |
| Code pinned by the lock commit | driver `experiments/g4_nono_test/run_cross_domain.py`, `recombine_cross.py`, `metrics.py`, aggregator `aggregator.py`, corpus `qd_cifar10.py` + `qd_cifar10_indices.json` + builder `scripts/g4_build_qd_cifar10.py`; tests `tests/unit/test_g4_nono_*.py` |

At filing, record on OSF the lock commit above and the lock timestamp;
the draft's own "Lock commit / Lock timestamp" fields are left to be
filled at filing, as for every pre-registration in this repository.

## PI decisions (draft header and §10)

- **2026-10-23:** ε = 0.01 (set to 0.005, then reverted to 0.01 the same day); N by the §3 N-selection rule with a N = 108 tier.
- **2026-10-24:** the N = 30 line of the N-selection rule is removed, so N is 95 or 108.
- **2026-10-24:** domain corpus CIFAR-10 (grayscale) + Quick, Draw! (§5), replacing the CIFAR-100 split.
- **2026-10-24:** un-mixed control arm registered as an **exploratory** arm, reported only (§5).
- **2026-10-24 (amendment after the first smoke):** epochs 3 → 10; acc20 and DOA descriptive only; `P_max` runs `recombine_full` (§2, §5, §6).
- **2026-10-24 (hardware, partly decided):** smokes on macM1; the hardware of the confirmatory campaign is chosen after the N10-C campaign. The protocol does not change; R1 run_ids are per machine. **To resolve before filing (§10 item 3):** §10 asks that open items be resolved before filing, and §9 defines the pre-filing smoke as run "on the filing hardware". All smokes ran on macM1. Either the PI names macM1 the filing host (as for DR-3), or he names another host, on which the §9 smoke and checklist item 5 are rerun before filing. The frozen draft is not edited; the PI's choice goes in the OSF filing.
- **2026-10-28:** split out of PR #41 so it is filed without waiting for E1; the smoke records are archived in the repository (draft header "Split" line).

## Pilots / smokes (draft §9) and archived records

All on macM1 (M1, 32 GB), seeds 900-902, never pooled with
confirmatory seeds. Records copied verbatim from macM1 on 2026-10-28,
SHA-256 in `docs/milestones/g4-nono-smoke-records-2026-10/SHA256SUMS`
(index `README.md` there).

| smoke | code | outcome | records |
|---|---|---|---|
| 1 (CIFAR-100, 3 epochs) | `28e18c5` | 24/24 cells ok; 2 of 3 seeds fail the manipulation check = §4 abort condition, N selection refused (`n: null`); led to the 2026-10-24 amendment | `docs/milestones/g4-nono-smoke-records-2026-10/smoke1/` |
| 2 (CIFAR-100, 10 epochs) | `c64bcee` | 24/24 ok; SD̂_D = 0.0022, N = 95 | `…/smoke2/` |
| 2-unmixed (exploratory) | `0b2b48e` | 3 cells, `P_max/unmixed` | `…/smoke2-unmixed/` |
| **3 (CIFAR-10 + Quick, Draw!)** | `6f11823` | 27/27 ok; manipulation check passes; **SD̂_D = 0.0015 < 0.015, N = 95** (the N filed) | `…/smoke3/` (`n-selection.json`) |

Re-check 2026-10-25: `aggregator.py --select-n` at `d81652a` gives a
file byte-identical to `smoke3/n-selection.json`; repeated on
2026-10-28 on the archived cells at the PR #71 head (identical again).

## Filing-gate status (draft §9)

| condition | status |
|---|---|
| (a) driver, `recombine_cross.py` and aggregator at a pinned commit | **done** — pinned by the lock commit `14c3d5f` |
| (b) passing test for every §5 fidelity-checklist item | **done** — items 1-4 and 6-8 on synthetic data; item 5 (new corpus) passes on macM1. Per the draft, item 5 must be rebuilt and pass on the filing host: met if the PI names macM1, otherwise to rerun there before filing (see PI decisions). |
| (c) aggregator on synthetic paired data covering every §2 state and §7 row | **done** — `tests/unit/test_g4_nono_aggregator.py`, CI |
| (d) smoke results and the selected N written into the filed version | **done** — §9, N = 95 |

## Package contents (`docs/osf-filing/g4-nono/`)

- `FILING-NOTE.md` — this note
- `osf-prereg-g4-nono-cross-domain-2026-10-23.frozen.md` — frozen draft
- `osf-prereg-g4-nono-cross-domain-2026-10-23.frozen.pdf` — PDF rendering
- `osf-prereg-erratum-g4-power-2026-10-23.frozen.md` — frozen erratum
- `osf-prereg-erratum-g4-power-2026-10-23.frozen.pdf` — PDF rendering
- `render.sh`, `render.lua`, `strikeout.lua` — how the PDFs were made
- `SHA256SUMS` — hashes of the files above
