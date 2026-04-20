# dreamOfkiki

**A substrate-agnostic formal framework for dream-based knowledge consolidation in artificial cognitive systems.**

Research program producing two complementary papers:

- **Paper 1** (Nature HB / PLoS Comp Bio target): formal framework **C** with axioms **DR-0..DR-4** and invariants families **I / S / K**.
- **Paper 2** (NeurIPS / ICML / TMLR target): empirical ablation on the `kiki_oniric` substrate across profiles `P_min`, `P_equ`, `P_max`.

**Status** — Cycle 1, released `paper-v0.4-draft` (2026-04-18). Next gate: **G5** (full ablation).
**Author** — Clément Saillant (L'Electron Rare), *dreamOfkiki* program author. Hypneum Lab.
**License** — MIT (code) + CC-BY-4.0 (docs).

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
