# AGENTS.md

Guidance for AI coding agents (Claude Code, Aider, Cursor, etc.) working in this repo.

## Project

`dreamofkiki` — substrate-agnostic formal framework for dream-based knowledge consolidation in artificial cognitive systems. Two-paper research output of **Hypneum Lab** (framework C + ablation), 28-week cycle. Current state lives in `STATUS.md` — read it first. Main branch `main`, recent HEAD `81524c0` (2026-05-21), DualVer `C-v0.24.0+PARTIAL`, SemVer alias `0.22.1`.

## Tech stack

- Language: Python 3.12+
- Runtime: `uv` (PEP 668)
- Test: `pytest` — 914 tests, coverage 89% (Linux gate 30%, macOS nightly gate 90% via `r1-nightly.yml`)
- Build: `hatchling`; packages `harness/`, `kiki_oniric/`
- Backend: **MLX on Apple Silicon** (`mlx>=0.31,<0.32`, pin paired with `mlx-lm`); CPU NumPy fallback
- CLI: `dream-harness` → `harness.cli.dream_harness:cli`
- Extras: `[fmri]` (nilearn/nibabel), `[teacher]` (llama-cpp-python)

## Commands

```bash
uv sync
uv run pytest                                 # full
uv run pytest tests/axioms/ -v                # axiom-conformance subset
uv run dream-harness --help
```

## Conventions

- Commits: subject ≤ 50 chars, body ≤ 72, no underscore in scope, no AI attribution, never `--no-verify`.
- Branches: `feat/<name>`, `fix/<name>`, `docs/<name>`, `wave3b/<name>`.
- DualVer: `C-vX.Y.Z+{STABLE,UNSTABLE,PARTIAL}` — formal axis (FC) and empirical axis (EC) bump independently (`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §12). `pyproject.toml` keeps a SemVer alias.
- Tests: macOS Apple Silicon CI is the science gate (90% cov); Linux is a sanity gate (MLX-only tests must be skipped via `conftest.py` `collect_ignore_glob`).
- Critic-review mandatory for ship-impacting commits (paper submission, gate close, version bump). See `~/.claude/projects/-Users-electron/memory/feedback_critic_before_ship.md`.

## File layout

- `kiki_oniric/` — substrate (dream runtime, profiles `P_min`/`P_equ`/`P_max`, ops, guards).
- `kiki_oniric/axioms/` — public axioms API (DR-0..DR-4 + DR-2'), **frozen `Axiom` dataclasses**.
- `kiki_oniric/substrates/` — registered substrates: `mlx_kiki_oniric`, `esnn_norse`, `esnn_thalamocortical`, `wake_sleep_cl_baseline`, `micro_kiki`, `_adversarial`, `mlx_latent_diffusion` (Wave 3b Track S).
- `harness/` — eval harness, benchmarks, run registry, matrix config.
- `docs/specs/`, `docs/invariants/`, `docs/glossary.md`, `docs/proofs/`, `docs/plans/`, `docs/papers/`.
- `papers/` — submission tracker, byline policy.
- `scripts/` — pilot scripts, G-gate milestone drivers.
- `tests/` — unit + axiom conformance.

## Domain-specific gotchas

- **`STATUS.md` is the source of truth** for current DualVer + active gate; read before changing anything claim-bearing.
- **Axioms are frozen dataclasses** in `kiki_oniric/axioms/`. Never mutate at runtime; never add unrecognised axiom IDs. New axioms require a spec amendment first.
- **MLX pin is intentional** (`mlx>=0.31,<0.32`, paired with `mlx-lm`) — tightened to avoid silent R1 hash drift (REBASELINE_NOTE 2026-05-10). Do not relax without re-baselining.
- **`numpy>=2.4,<3.0`** belt-and-suspenders against silent NumPy 3.x absorption.
- **Linux CI must skip MLX-only tests** via root `conftest.py` `collect_ignore_glob` + lazy import in `kiki_oniric/substrates/micro_kiki/__init__.py`. Any new MLX-only path must be added there.
- **Wave 3b plan** at `docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md`: M1-M4 closed, M5/M6 deferred per §6. OSF amendment for Wave 3b (`docs/osf-amendment-wave3b.md`) is **drafted but not yet filed** against Q6JYN.
- **Numbers are load-bearing**: per the project's spec-driven discipline, axiom IDs / invariants / numerical results must trace to docs. Don't hand-edit JSON outputs.
- **Several subdirectories ship their own `CLAUDE.md`** — closest wins. Read it when you enter that tree.

## When in doubt

- Read `STATUS.md` and `CLAUDE.md`.
- Recent commits: `git log --oneline -20`.
- Spec: `docs/specs/2026-04-17-dreamofkiki-master-design.md`, framework-C: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`.
- Cluster context: `~/CLAUDE.md`. Memory: `~/.claude/projects/-Users-electron/memory/project_hypneum_*`.
- Run `uv run pytest` before non-trivial commits.
