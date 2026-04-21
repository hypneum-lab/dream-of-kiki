# dreamOfkiki

**A substrate-agnostic formal framework for dream-based knowledge consolidation in artificial cognitive systems.**

Research program producing two complementary papers:

- **Paper 1** (PLOS Computational Biology target ; Nature Human Behaviour retired as primary target 2026-04-20): formal framework **C** with axioms **DR-0..DR-4** and invariants families **I / S / K**.
- **Paper 2** (NeurIPS / ICML / TMLR target): empirical ablation on the `kiki_oniric` substrate across profiles `P_min`, `P_equ`, `P_max`.

**Status** — DualVer `C-v0.8.0+PARTIAL` (2026-04-21 ; SemVer alias `0.8.0` in `pyproject.toml` / `CITATION.cff`). Paper 1 **v0.2** frozen (22 p), primary submission target **PLOS Computational Biology** (retargeted from Nature Human Behaviour on 2026-04-20 — acknowledged trade-off : IF 15.9 → 4.3, decision time 3–6 months → 44 days). Gates **G1, G7, G8, G9 LOCKED** (cycle 2 closed) ; **G10 deferred to Paper 2** per PLOS CB pivot. 277 tests / 91.17 % coverage. OSF pre-registration **live** at DOI `10.17605/OSF.IO/Q6JYN` (https://osf.io/q6jyn, DataCite-minted 2026-04-19T00:28:05Z). arXiv deposit ready, only web-UI walkthrough pending. Bonferroni amendment (2026-04-19) still to be filed as a linked OSF registration before submit — see `docs/osf-amendment-submission-package.md`.
**Author** — Clément Saillant (L'Electron Rare), *dreamOfkiki* program author. Hypneum Lab.
**License** — MIT (code) + CC-BY-4.0 (docs).

---

## Preliminary observation (Paper 2 backlog, 2026-04-20 ; framing reviewed 2026-04-21)

**A directional trend consistent with substrate-size scaling is observed on the `p_max` profile between 1.5B and 7B substrates, pending a 3rd scale point.** Phase-B pilot over three profiles (`p_min`, `p_equ`, `p_max`) × three benchmarks (MMLU, HellaSwag, mega_v2), 30 seeds × 3 profiles = 90 cells. 7B result : H1 p_min rejects H₀ at p = 1.4 × 10⁻²⁴, H1 p_equ at p = 6.2 × 10⁻²⁷ ; H1 p_max (**p = 0.055**) **did not reject** at any pre-registered α (0.05, 0.0125, 0.00833, 0.00625). We report the observed effect-size shift on `p_max` as a *descriptive* trend, not a confirmatory scaling-law claim.

**Why we are *not* claiming a "15-order-of-magnitude collapse".** (a) Two data points do not identify a power-law ; Clauset, Shalizi & Newman 2009 recommend ≥ 50 points with explicit goodness-of-fit ; neural-scaling-law baselines (Kaplan et al. 2020) typically span 7–9 scales. (b) p = 0.055 sits above every pre-registered α and does not support a confirmatory rejection ; describing the observation as a "collapse" would overstate what the evidence carries (cf. Stumpf & Porter 2012 on power-law overclaiming). (c) The reported magnitude may include floating-point underflow near machine precision ; a numerical-precision audit is required before any quantitative statement.

**Gating before any Paper 2 scaling-law claim.** (i) 3rd scale point (3B or 14B substrate) to rule out monotonicity-from-two-points artefact. (ii) Bootstrap CI on the per-seed p-value distribution. (iii) Numerical-precision audit of the p-value tail. (iv) If the audit reveals underflow, re-compute under log-probability arithmetic. The concrete Paper 2 hypothesis (H7) is therefore *conditional* on (i)–(iv) passing ; see `docs/milestones/scaling-law-analysis-2026-04-20.md`.

**Lesson carried over from sister project `bouba_sens` v0.5.0** (`github.com/hypneum-lab/bouba_sens`, 2026-04-21) — three pre-registered findings in that programme were all downgraded to null by critical validation (null-model partition control, bootstrap CIs, multi-estimator MI). This repo adopts the same discipline : no scaling-law claim is advanced in Paper 1 ; the preliminary observation reported here is framework-level (the pipeline produces a registerable trend) and is *not* used to argue any substrate-agnosticism claim in Paper 1's confirmatory analyses.

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
