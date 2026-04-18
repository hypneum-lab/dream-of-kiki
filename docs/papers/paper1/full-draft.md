---
title: "dreamOfkiki: A Substrate-Agnostic Formal Framework for Dream-Based Knowledge Consolidation in Artificial Cognitive Systems"
author: "dreamOfkiki project contributors"
contact: "Clement Saillant <clement@saillant.cc>"
affiliation: "L'Electron Rare, France"
date: "2026"
draft: "v0.1 (cycle-1, S20.3 assembly)"
---

# Paper 1 — Full Draft Assembly

⚠️ **Status** : draft assembly. Section files are the source of
truth ; this file is the rendering target. Edits go in section
files, then re-assemble.

⚠️ **Synthetic data caveats** apply to §7 Results (numbers from
mega-v2 synthetic placeholder). Real ablation lands cycle 1
closeout (S20+) or cycle 2.

---

## 1. Abstract

→ See `abstract.md` (250-word target).

[INCLUDE: abstract.md]

---

## 2. Introduction

→ See `introduction.md` (~1.5 pages, ~1200 words).

[INCLUDE: introduction.md]

---

## 3. Background — four theoretical pillars

→ See `background.md` (~1.5 pages, ~1500 words).

[INCLUDE: background.md]

---

## 4. Framework C

⚠️ **Source** : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
covers this section. Paper version below is a condensed narrative
of that spec, structured per §4 outline.md.

### 4.1 Primitives — 8 typed Protocols

Awake → Dream channels :
- α (raw traces, P_max only) — firehose ring buffer
- β (curated episodic) — SQLite append-log with saillance gating
- γ (weights snapshot) — checkpoint pointer fallback
- δ (hierarchical latents) — ring buffer N=256 multi-species

Dream → Awake channels :
- 1 (weight delta) — applied via swap protocol
- 2 (latent samples) — generative replay queue
- 3 (hierarchy diff) — atomic apply at swap with S3 guard
- 4 (attention prior) — meta-cognitive guidance (P_max only)

### 4.2 Profiles — chain inclusion DR-4

| Profile | Channels in | Channels out | Operations |
|---------|-------------|--------------|------------|
| P_min   | β | 1 | replay, downscale |
| P_equ   | β + δ | 1 + 3 + 4 | replay, downscale, restructure, recombine_light |
| P_max   | α + β + δ | 1 + 2 + 3 + 4 | replay, downscale, restructure, recombine_full |

DR-4 (proven in `docs/proofs/dr4-profile-inclusion.md`) :
ops(P_min) ⊆ ops(P_equ) ⊆ target_ops(P_max), and similarly for
channels. P_max is skeleton-only in cycle 1.

### 4.3 Dream-episode 5-tuple ontology

Each dream-episode (DE) is a 5-tuple :
`(trigger, input_slice, operation_set, output_channels, budget)`.
Triggers ∈ {SCHEDULED, SATURATION, EXTERNAL}. Operations are a
non-empty tuple of {REPLAY, DOWNSCALE, RESTRUCTURE, RECOMBINE}.
BudgetCap enforces non-negative finite (FLOPs, wall_time_s,
energy_j) per K1 invariant.

### 4.4 Operations — semigroup of consolidation steps

The operation set forms a free non-commutative semigroup under
composition `∘` with additive budget (DR-2 compositionality,
proof draft in `docs/proofs/dr2-compositionality.md`). Canonical
order : replay → downscale → restructure (serial, A-B-D pillar
order) ; recombine in parallel (C pillar). The op-pair analysis
(`docs/proofs/op-pair-analysis.md`) enumerates all 16 pairs,
finding 12 non-commutative cross-pairs.

### 4.5 Axioms DR-0..DR-4

- **DR-0 (accountability)** : every executed DE produces an
  EpisodeLogEntry, even on handler exception (try/finally guarantee).
- **DR-1 (episodic conservation)** : every β record is consumed
  before purge.
- **DR-2 (compositionality)** : op composition forms a semigroup
  with type closure + budget additivity + functional composition.
  Free-generator universal property is open (G3 reviewer
  pending).
- **DR-3 (substrate-agnosticism)** : Conformance Criterion =
  signature typing ∧ axiom property tests pass ∧ BLOCKING
  invariants enforceable. kiki-oniric satisfies all three.
- **DR-4 (profile chain inclusion)** : P_min ⊆ P_equ ⊆ P_max
  for ops and channels.

### 4.6 Invariants — I/S/K with enforcement matrix

- **I1** episodic conservation (BLOCKING)
- **I2** hierarchy traceability (BLOCKING)
- **I3** latent distributional drift (WARN)
- **S1** retained non-regression (BLOCKING, swap guard)
- **S2** finite weights no NaN/Inf (BLOCKING, swap guard)
- **S3** topology valid (BLOCKING, swap guard)
- **S4** attention prior bounded (P_max only)
- **K1** dream-episode budget (BLOCKING)
- **K3** swap latency bounded (WARN)
- **K4** eval matrix coverage on MAJOR bump (BLOCKING)

### 4.7 DualVer formal+empirical versioning

`C-vX.Y.Z+{STABLE,UNSTABLE}` — formal axis (FC) and empirical
axis (EC) bump independently. Current : C-v0.5.0+STABLE
(target post-G3 : C-v0.7.0+STABLE).

---

## 5. Implementation — kiki-oniric

⚠️ **Source** : `kiki_oniric/` Python package. Paper version is
narrative of architecture per §5 outline.md.

### 5.1 Substrate choice — MLX on Apple Silicon

We selected MLX (Apple) as the cycle-1 substrate for its
deterministic compilation graph, native Apple Silicon support,
and Python-first ergonomics. Substrate-specific code is
isolated in `kiki_oniric/dream/operations/*_mlx.py` variants
(replay, downscale, restructure, recombine), with
substrate-agnostic skeleton variants alongside for testing
without GPU.

### 5.2 Runtime — DreamRuntime + swap protocol

`DreamRuntime` is a single-threaded scheduler with handler
registry per Operation. DR-0 accountability is enforced by a
try/except/finally pattern : every `execute()` call produces an
`EpisodeLogEntry` even when a handler raises (logged with
`completed=False` + error string + executed_ops_so_far).

`swap_atomic` orchestrates the W_awake ← W_scratch promotion
with S2 (finite) + S1 (retained non-regression) guards. S3
(topology) guard is wired to the restructure operation output.
`SwapAborted` is raised with a violated-invariant code on guard
failure ; the swap is rolled back.

### 5.3 Profiles wired

P_min : replay + downscale handlers registered ; `swap_now()`
method exposes the S1 retained-eval gating closure. Pilot G2
results documented in `docs/milestones/g2-pmin-report.md`.

P_equ : replay + downscale + restructure + recombine_light
handlers registered ; channels β+δ → 1+3+4. Ablation G4 results
documented in `docs/milestones/g4-pequ-report.md`.

P_max : skeleton with target metadata (target_ops,
target_channels_out) for DR-4 chain test ; handlers deferred
cycle 2.

### 5.4 Concurrent worker — Future API skeleton

`ConcurrentDreamWorker` exposes a Future-based API (`submit()
-> Future`, `drain() -> list[EpisodeLogEntry]`) with sync
execution under the hood for cycle 1. Cycle-2 swap to real
asyncio/threading is forward-compatible by design.

### 5.5 Open-source artifacts

Code : `github.com/electron-rare/dream-of-kiki` (MIT, frozen at
arXiv submission tag). Models : HuggingFace
`clemsail/kiki-oniric-{P_min,P_equ}` (cycle 1) +
`kiki-oniric-P_max` (cycle 2). Data : Zenodo DOI minted
post-S20+. Pre-registration : OSF DOI (pending lock).
Dashboard : `dream.saillant.cc` public read-only.

---

## 6. Methodology

→ See `methodology.md`.

[INCLUDE: methodology.md]

---

## 7. Results

→ See `results-section.md`.

[INCLUDE: results-section.md]

---

## 8. Discussion

→ See `discussion.md`.

[INCLUDE: discussion.md]

---

## 9. Future Work

→ See `future-work.md`.

[INCLUDE: future-work.md]

---

## 10. References

→ See `references.bib` (16 entries cycle-1 stub, will extend to
~30-40 in S20-S22 as the full draft is rendered).

Key citations (alphabetical) :
- Diekelmann & Born 2010 (sleep memory)
- French 1999 (catastrophic forgetting)
- Friston 2010 (FEP)
- Hobson 2009 (REM dreaming)
- Kirkpatrick 2017 (EWC)
- McClelland 1995 (CLS)
- McCloskey & Cohen 1989 (forgetting)
- Rao & Ballard 1999 (predictive coding)
- Rebuffi 2017 (iCaRL)
- Shin 2017 (generative replay)
- Solms 2021 (consciousness)
- Stickgold 2005 (consolidation)
- Tononi & Cirelli 2014 (SHY)
- van de Ven 2020 (brain-inspired replay)
- Walker & Stickgold 2004 (consolidation)
- Whittington & Bogacz 2017 (predictive coding)

---

## Word count summary (target : ~5000 words main + supp)

| Section | Target | Status |
|---------|--------|--------|
| §1 Abstract | ≤250 | drafted (~265, needs trim) |
| §2 Introduction | ≤1500 | drafted (~1200) |
| §3 Background | ≤1500 | drafted (~1500) |
| §4 Framework | condensed in main + spec ref | done |
| §5 Implementation | condensed | done |
| §6 Methodology | ≤1500 | drafted (~1500) |
| §7 Results | ≤2000 | drafted (placeholder) |
| §8 Discussion | ≤1500 | drafted (~1500) |
| §9 Future Work | ≤700 | drafted (~700) |
| §10 References | n/a | 16 entries stub |

**Estimated total** : ~10000 words (needs aggressive trim for
Nature HB 5000-word main-text discipline ; supp can absorb
overflow).

---

## Notes for revision

- Render via Quarto / pandoc into PDF + LaTeX for arXiv submit
  (S21.1)
- Insert OSF DOI in §6.1 once OSF lock is completed
- Replace synthetic placeholders in §7 with real ablation values
  post S20+
- Trim §1 abstract to ≤250 words
- Trim §3 + §6 + §8 to fit overall main-text budget
- Add Figures (1 architecture diagram, 2 results boxplot, 3
  Jonckheere trend, 4 four-pillars conceptual)
- Bibtex render with proper `\cite{}` calls
