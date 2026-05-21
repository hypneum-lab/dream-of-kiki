# dreamOfkiki

**A substrate-agnostic formal framework for dream-based knowledge consolidation in artificial cognitive systems.**

Research program producing two complementary papers, both released as arXiv preprints + Zenodo DOIs:

- **Paper 1** — formal framework **C** with axioms **DR-0..DR-4** and invariants families **I / S / K**. Journal submission (PLOS Computational Biology candidate) deferred until preprint signal.
- **Paper 2** — empirical ablation on the `kiki_oniric` substrate across profiles `P_min`, `P_equ`, `P_max`. Peer-review venue deferred — arXiv preprint first, then decide.

**Status** — DualVer `C-v0.24.0+PARTIAL` (2026-05-21 ; SemVer alias `0.22.0` in `pyproject.toml`). 914 tests, coverage 89 % (Linux gate 30 %, macOS nightly gate 90 %). Last bump 2026-05-21 Wave 3b M3 (MLX latent-diffusion substrate trainer + sampler + R1 entries). Historic Paper 1 v0.2 status: Paper 1 **v0.2** frozen (22 p), release path : arXiv preprint + Zenodo DOI. Journal submission (PLOS Computational Biology candidate) deferred — we want a preprint in the world first and a signal from it before committing to a months-long peer review. Gates **G1, G7, G8, G9 LOCKED** (cycle 2 closed) ; **G10 deferred to Paper 2**. 277 tests / 91.17 % coverage. OSF pre-registration **live** at DOI `10.17605/OSF.IO/Q6JYN` (https://osf.io/q6jyn, DataCite-minted 2026-04-19T00:28:05Z). arXiv deposit ready, only web-UI walkthrough pending. Bonferroni amendment (2026-04-19) still to be filed as a linked OSF registration before submit — see `docs/osf-amendment-submission-package.md`.
**Author** — Clément Saillant (L'Electron Rare), *dreamOfkiki* program author. Hypneum Lab.
**License** — MIT (code) + CC-BY-4.0 (docs).

---

## Preliminary observation (Paper 2 backlog, 2026-04-20 ; framing reviewed 2026-04-21)

**A directional trend consistent with substrate-size scaling is observed on the `p_max` profile between 1.5B and 7B substrates, pending a 3rd scale point.** Phase-B pilot over three profiles (`p_min`, `p_equ`, `p_max`) × three benchmarks (MMLU, HellaSwag, mega_v2), 30 seeds × 3 profiles = 90 cells. 7B result : H1 p_min rejects H₀ at p = 1.4 × 10⁻²⁴, H1 p_equ at p = 6.2 × 10⁻²⁷ ; H1 p_max (**p = 0.055**) **did not reject** at any pre-registered α (0.05, 0.0125, 0.00833, 0.00625). We report the observed effect-size shift on `p_max` as a *descriptive* trend, not a confirmatory scaling-law claim.

**Why we are *not* claiming a "15-order-of-magnitude collapse".** (a) Two data points do not identify a power-law ; Clauset, Shalizi & Newman 2009 recommend ≥ 50 points with explicit goodness-of-fit ; neural-scaling-law baselines (Kaplan et al. 2020) typically span 7–9 scales. (b) p = 0.055 sits above every pre-registered α and does not support a confirmatory rejection ; describing the observation as a "collapse" would overstate what the evidence carries (cf. Stumpf & Porter 2012 on power-law overclaiming). (c) The reported magnitude may include floating-point underflow near machine precision ; a numerical-precision audit is required before any quantitative statement.

**Gating before any Paper 2 scaling-law claim.** (i) 3rd scale point (3B or 14B substrate) to rule out monotonicity-from-two-points artefact. (ii) Bootstrap CI on the per-seed p-value distribution. (iii) Numerical-precision audit of the p-value tail. (iv) If the audit reveals underflow, re-compute under log-probability arithmetic. The concrete Paper 2 hypothesis (H7) is therefore *conditional* on (i)–(iv) passing ; see `docs/milestones/scaling-law-analysis-2026-04-20.md`.

**Lesson carried over from sister project `bouba_sens` v0.5.0** (`github.com/hypneum-lab/bouba_sens`, 2026-04-21) — three pre-registered findings in that programme were all downgraded to null by critical validation (null-model partition control, bootstrap CIs, multi-estimator MI). This repo adopts the same discipline : no scaling-law claim is advanced in Paper 1 ; the preliminary observation reported here is framework-level (the pipeline produces a registerable trend) and is *not* used to argue any substrate-agnosticism claim in Paper 1's confirmatory analyses.

---

## 2026-05-11 milestones — Paper 1 §5.8 honest framing + Q2+ audit

Paper 1 advanced to **v0.2 with §5.8 honest per-substrate FP framing**
(Cat C heterogeneity 1 % / 6 % / 80 %, no overstatement). The
Conformance Criterion was strengthened to **C+** and now requires
**both** structural invariants and **C2 substrate-specific axiom
property tests** — validated against 25 substrates from 6 categories
through the sister-repo `nerve-wml` N8 Q2 + N9 Q2+ negative-tests
audit (cumulative 25 / 25 FP; structural layer alone is insufficient).

| Item | State |
|---|---|
| Paper 1 §5.8 reformulation | shipped (per-substrate FP, no Cat C overstatement) |
| Conformance Criterion | upgraded **C → C+** (C1 + C2 + C3) |
| C2 axiom property tests | mandated for any substrate compliance claim |
| Negative-tests audit | 25 / 25 substrates rejected (Cat A–F), upstream `nerve-wml` |
| Critic-driven fixes | 6 / 6 closed (CRITICAL #1 ablation void, CRITICAL #2 degenerate metrics, MAJOR #3 Cat C overstatement, MAJOR #4 Jonckheere wrong test, MAJOR #5 β-VAE confound, MAJOR #6 paper/JSON count mismatch) |

Cumulative on main HEAD `15efb95` : **479 commits**, internal
DualVer `C-v0.10.0`, 277+ tests / 91 % coverage. Paper 1 v0.2 still
points at PLOS Computational Biology as primary venue, arXiv deposit
ready.

---

## 2026-05-19 update — Paper 1 §8.4 prior-art extended + biophysical stratification (i)+(ii)

Paper 1 §8.4 ("Comparison with prior art") now lists **DVNC, NIR,
AER, and Liu 2024 HNN** as adjacent prior art and engages the
2026 PRH critiques (`platoscave2026` arXiv:2604.18572 ;
`aristotelianprh2026` arXiv:2602.14486). DR-3 substrate-agnosticism
is now explicitly anchored to the **local** form of PRH — mutual-kNN
against a capacity-matched random baseline, per nerve-wml empirical
evidence — rather than the unqualified global form. DVNC is
positioned as a special case of nerve-wml's general portability. NIR
and AER are classified as orthogonal layers that could compile to
an E-SNN substrate without altering DR-0..DR-4.

A new sub-theory spec `docs/specs/2026-05-20-biophysical-stratification.md`
(280 lines, **non-revision contract** — never weakens DR-0..DR-4 /
N-1..N-5 / W-1..W-4) routes nine deep-research references into five
biophysical strata (coupled-field substrate, theta-gamma sequencing,
multimodal efficient + predictive coding, embodied sensorimotor
grounding, critical dynamics). This spec operates under the
**(i)+(ii) co-existent framing** agreed after the session:

| Item | Location | What |
|---|---|---|
| Paper 1 §8.4 extended | `docs/papers/paper1/full-draft.md` + `discussion.md` | 4 new prior-art rows (DVNC, NIR, AER, Liu 2024 HNN); new "2026 PRH critiques + global-vs-local refinement" paragraph anchoring DR-3 to local-form PRH |
| NEW sub-theory spec | `docs/specs/2026-05-20-biophysical-stratification.md` | 5 strata routing 9-10 (b′)-classified deep-research refs; non-revision contract; (ii) side of (i)+(ii) framing |

**(i)+(ii) framing** — DR-3 is preserved as **universal**:
`BioFieldWML` (in `nerve-wml`) is one conformant substrate among N
alongside MLX, LIF, and Transformer; this is the (i) side.
The new biophysical stratification spec is the **(ii) side**: a
family of empirical scaffolding beside framework-C, stratifying
biological evidence without weakening any axiom. OQ defaults applied:
OQ-1 BioFieldWML.step() = one synchronous Up-Down cycle (DR-0
preserved); OQ-2 STDP scoped to BioFieldWML (YAGNI bounded, not
revoked); OQ-3 spec scope = Paper 2 appendix scaffolding.

Follow-up: `dream-of-kiki#20` for paper1-fr EN→FR sync of PR #18
§8.4 changes.

---

## 2026-05-20 → 2026-05-21 session — Wave 3a Dream2Learn resolved + Wave 3b M1-M4 shipped

Session shipped **6 PRs + 2 direct commits** (master green at
`5e5582e`). Two work-streams advanced :

**Wave 3a — Paper 2 c-alert resolution.** PR #27 (`a55f3bc`)
ships a formal DR-3 separation proof for **Dream2Learn**
(`docs/proofs/dream2learn-dr3-separation.md`) : Dream2Learn does
**not** satisfy the Conformance Criterion (CC condition 1 typed-
interface failure ; DE / role-partition mismatch). The 2026-05-19
c-alert is resolved and Dream2Learn is reclassified from c-alert
→ category (a) baseline. §7.8 EN + FR updated, bib +
`calcagno2026`, glossary updated.

**Wave 3b — MLX latent-diffusion substrate (Track S).** Four
milestones closed :

| M | PR / commit | What |
|---|---|---|
| M1 | PR #29 (`92afef9`) | Plan (525 L, 6 milestones M1-M6, D1-D5 decisions) + DR-3 evidence skeleton v0.1 + OSF amendment draft for Q6JYN |
| M2 | PR #30 (`05f737f`) | `kiki_oniric/substrates/mlx_latent_diffusion.py` + `_diffusion/` E/U/σ class signatures only ; 8 tests, 98 % coverage. Track S chosen at M2 review |
| M3 | PR #31 (`4cd259c`) | Sampler R1 key tree (`mx.random.split`) + trainer + 3 DR conformance families (DR-3, DR-0, DR-1) + 3 R1 entries `apple_m5` (`pending_review`). DualVer `C-v0.23 → C-v0.24` framework + `C-v0.13 → C-v0.14` substrate-internal. +33 tests → 898 |
| M4 | direct `e17c17e` | `scripts/ablation_cycle3_diffusion.py` + `harness/diffusion_eval/{cifar100_split_loader,diffusion_metrics}.py` + 15 integration tests ; +11 tests → 914, coverage 89 %, smoke 1.3 s, R1 deterministic. PR #32 closed as superseded |

Operational PRs : PR #26 (`38fcefa`) B5 bounds-check on
`hierarchy_change.add` + `weight_delta`, closes #24 ; PR #28
(`2013176`) skip MLX-only tests on Linux runners (root
`conftest.py` `collect_ignore_glob` + Linux coverage gate
30 % / macOS nightly 90 %) — first green master CI since
2026-05-03. Direct CI fix `5e5582e` follows M4.

**No empirical claim is advanced by Wave 3b at this stage** :
M3 ships R1 entries with `status: "pending_review"` (M5 GrosMac)
and `pending_remote_validation` (M1 Max). M5 (ablation pilot)
remains gated on 5 listed blockers ; M6 (paper integration)
follows M5.

---

## What this repo is

- **Research code**, not a product. Correctness > performance.
- Python 3.12+, `uv`-managed. MLX backend on Apple Silicon.
- Two artifacts in one tree:
  - The **formal framework C** — 8 primitives, 4 channels, DR-axioms, Conformance Criterion.
  - The **`kiki_oniric`** substrate, forked from `kiki-flow-core`, implementing Track A.
- Dual-axis versioning (**DualVer**): `C-vX.Y.Z+{STABLE,UNSTABLE}` — the formal axis (FC) and the empirical axis (EC) bump independently. See `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §12.

## Repo layout

| Directory | Content |
|---|---|
| `docs/specs/` | Master design + framework C spec (canonical) |
| `docs/invariants/` | I / S / K families — every runtime guard cites one |
| `docs/proofs/` | Formal proofs (DR-0..DR-4, conformance) |
| `docs/glossary.md` | Canonical terminology — don't invent synonyms |
| `kiki_oniric/` | Substrate — 8 primitives, 4 channels, 3 profiles |
| `harness/` | Shared eval harness, stratified matrix, bit-exact run registry |
| `papers/` | Paper 1 (formal) + Paper 2 (ablation) drafts |
| `tests/` | Unit + conformance (axioms, invariants) — coverage ≥ 90 % |
| `scripts/` | Milestone drivers, one per G-gate |
| `ops/` | Outreach, reviewer recruitment, mail drafts |

Nested `CLAUDE.md` files give agent-specific guidance per directory.

## Read-first context

Before touching code or claims:

1. `STATUS.md` + `CHANGELOG.md` — current sprint, gate, DualVer version, open actions.
2. `docs/specs/2026-04-17-dreamofkiki-master-design.md` — vision, 5 tracks, 28-week cycle, G1..G6 gates.
3. `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` — formal framework, axioms, conformance.
4. `docs/glossary.md` — canonical terms.

## Reproducibility contract (R1)

`harness/storage/run_registry.py` enforces bit-stability:

```
(c_version, profile, seed, commit_sha) → run_id   (SHA-256 slice, 16 hex)
```

Every experimental claim in either paper resolves to a registered `run_id` or a proof file. Benchmarks ship with `.sha256` digests. Seeds are never edited in place — add a new seed and register a new run.

## Install

```bash
git clone https://github.com/hypneum-lab/dream-of-kiki.git
cd dream-of-kiki
uv sync --all-extras
```

Python 3.12+, macOS arm64 (MLX) or Linux x86_64 (CPU fallback).

## Reproduce the paper v0.4

The draft release includes full experimental data. To re-run:

```bash
# Run the gate-specific pilot
uv run python scripts/pilot_g4.py --profile P_equ --seed 42

# Run the conformance test suite (axioms + invariants)
uv run pytest tests/conformance/ -v

# Regenerate paper figures (reads from run registry)
uv run python scripts/render_figures.py --gate G4
```

Outputs land in `harness/registry/runs/` (gitignored). Figures go to `papers/figures/`.

## Public resources (planned)

- Dashboard: [`dream.saillant.cc`](https://dream.saillant.cc) (public read-only)
- Models: `huggingface.co/clemsail/kiki-oniric-{P_min,P_equ,P_max}`
- OSF pre-registration: H1-H4 locked at S3
- Zenodo DOIs: harness + models + datasets

## Citation

Draft pre-print, tag [`paper-v0.4-draft`](https://github.com/hypneum-lab/dream-of-kiki/releases/tag/paper-v0.4-draft):

```bibtex
@unpublished{dreamofkiki-2026,
  author = {Saillant, Clément},
  title  = {Dream-Based Knowledge Consolidation in Artificial
            Cognitive Systems: A Formal Framework},
  year   = 2026,
  url    = {https://github.com/hypneum-lab/dream-of-kiki},
  note   = {Draft v0.4; MIT code + CC-BY-4.0 docs}
}
```

## Related repos

| Repo | Relation |
|---|---|
| [**kiki-flow-research**](https://github.com/hypneum-lab/kiki-flow-research) | Upstream — Wasserstein flow engine, `kiki_oniric` forked from `kiki_flow_core` |
| [**micro-kiki**](https://github.com/hypneum-lab/micro-kiki) | Sibling — MoE-LoRA routing system consuming consolidated profiles |

## Contributing

Research-first discipline:

- Axioms / invariants are load-bearing — cite the ID (`DR-1`, `S2`, `I4`) in every guard, test, and commit message that enforces one.
- DualVer bumps require either a proof (formal axis) or a gate result (empirical axis). Both axes are recorded in `CHANGELOG.md`.
- No AI co-authorship trailer. Authorship byline: *dreamOfkiki project contributors*.
