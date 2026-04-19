<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : dreamOfkiki project contributors
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# §9 Future Work — Cycle 3 (Paper 2, draft C2.16)

**Authorship byline** : *dreamOfkiki project contributors*
**License** : CC-BY-4.0

**Target length** : ~0.5 page markdown (≈ 500 words)

---

## 9.1 Finish Phase 3 — divergent-predictor replication

The single highest-priority cycle-3 deliverable is to replace
the shared mock predictor (§6.4) with **substrate-specific
inference** :

- MLX forward pass reading the MLX state of
  `mlx_kiki_oniric`, producing real per-item predictions.
- LIFState spike-rate read-out reading the E-SNN state of
  `esnn_thalamocortical`, producing real per-item
  predictions.

With a divergent predictor, the cross-substrate H1-H4 verdicts
(§7.2) become **informative** : agreement is no longer trivial
by construction ; disagreement would indicate a real
substrate-dependent signal. Either outcome is publishable —
agreement strengthens DR-3 empirically, disagreement constrains
the framework's substrate-agnostic claim.

Landing this item closes the `+PARTIAL` qualifier on the
DualVer tag and enables a `C-v0.7.0+STABLE` bump
(`docs/milestones/g9-cycle2-publication.md` § DualVer
proposals).

## 9.2 Real Loihi-2 hardware mapping

If the Intel NRC partnership materializes (external action,
tracked in `docs/milestones/g9-cycle2-publication.md` §
external user actions), the numpy LIF skeleton can be ported
to Loihi-2. The port should preserve the op factory
signatures (so C1 signature typing remains PASS without
code churn) while swapping the LIFState simulator for
neuromorphic hardware execution.

Fallback paths, ranked by likelihood :

- **SpiNNaker** via Norse / PyNN — software-compatible
  spiking substrate, deployable without Intel partnership.
- **Lava SDK simulation** of Loihi-2 — still synthetic but
  bit-compatible with the real chip's execution model.
- **Status quo** — keep the numpy LIF skeleton and defer
  neuromorphic claims until hardware access is confirmed.

## 9.3 Real fMRI cohort (T-Col pivot)

The T-Col outreach (§8.4) targets a lab partnership
producing task-controlled linguistic fMRI data. A cohort of
even 20 participants on a pre-registered paradigm would be
sufficient to run RSA-style alignment of the dream pipeline's
γ-semantic snapshot against BOLD responses, strengthening
H3 (monotonic representational alignment) which currently
fails at Bonferroni due to mock-predictor degeneracy (§7.2).

Cycle-3 timelines for fMRI data acquisition are long (IRB,
scanning, preprocessing, QC) — this item may slip to cycle 4
if partnership formalization is delayed beyond Q2 2026.

## 9.4 Paper 3 emergence

A Paper 3 — cross-substrate empirical performance comparison
on real data — becomes plausible **only if** 9.1
(divergent-predictor) and at least one of 9.2 (Loihi-2) or
9.3 (fMRI) land with strong data. Until then, Paper 3 is a
provisional placeholder (`docs/papers/paper3/outline.md`) and
not a commitment.

Possible Paper 3 venues, if and when it emerges :

- **Nature Communications** or **PNAS** (if real fMRI signal
  carries a consolidation-specific claim).
- **Neuromorphic Computing and Engineering** (IOP) or
  **Frontiers in Neuroscience** (if Loihi-2 execution carries
  a hardware-energy claim).
- **NeurIPS** follow-up (if the divergent-predictor
  replication shows substrate-specific performance deltas on
  ML benchmarks).

## 9.5 Phase 4 sequencing with Paper 1 acceptance

Paper 2's arXiv submission is **not blocked** on Paper 1
acceptance, but NeurIPS / TMLR sequencing is. Two viable
sequencing strategies :

- **Preprint-first** : arXiv-submit Paper 2 concurrently with
  Paper 1 preprint, cite the arXiv ID cross-document.
  NeurIPS / TMLR submission follows once Paper 1 is at least
  under review.
- **Pivot B** : if Paper 1 acceptance is delayed > 6 months,
  re-self-contain Paper 2's framework recap (§3 background
  + §4 conformance prelim), add an explicit "Paper 1 is
  under review" disclaimer, and submit NeurIPS / TMLR
  standalone.

The Pivot B contingency is load-bearing for the cycle-2
closeout claim that Paper 2 *is* submission-ready even if
Paper 1 is delayed. The full-draft assembly (this commit
C2.16) is the last engineering deliverable needed to
execute Pivot B.

---

## Notes for revision

- Re-order 9.1..9.5 by priority once Intel NRC partnership
  status is confirmed.
- Insert concrete cycle-3 Gantt references once
  `docs/superpowers/plans/2026-04-XX-dreamofkiki-cycle3.md`
  lands.
- Tighten to ≤ 400 words at NeurIPS pre-submission pass.
