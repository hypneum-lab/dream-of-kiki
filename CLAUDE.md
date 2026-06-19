# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when
working with code in this repository.

## dreamOfkiki

Substrate-agnostic formal framework for dream-based knowledge
consolidation in artificial cognitive systems. Research program,
two-paper output (framework C + ablation), 28-week cycle. Current
version and active gate live in `STATUS.md` — read it first.

## What this repo is

- **Research code**, not a product. Correctness > performance.
- Python 3.12+, `uv`-managed, MLX backend on Apple Silicon.
- Two artifacts in one tree : the **formal framework C** (axioms
  DR-0..DR-4, invariants I/S/K) and the **kiki-oniric** substrate
  fork (profiles `P_min`, `P_equ`, `P_max`).
- DualVer versioning : `C-vX.Y.Z+{STABLE,UNSTABLE}` — formal axis
  (FC) and empirical axis (EC) bump independently. See
  `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §12.

## Where to look

| Task | Directory |
|------|-----------|
| Spec / axioms / invariants / glossary / proofs | `docs/` |
| Substrate implementation (dream runtime, profiles, ops, guards) | `kiki_oniric/` |
| Public axioms API (DR-0..DR-4, DR-2' ; frozen `Axiom` dataclasses) | `kiki_oniric/axioms/` |
| Evaluation harness, benchmarks, run registry, matrix config | `harness/` |
| Unit + conformance (axioms, invariants) tests | `tests/` |
| Pilot scripts, milestone drivers (G-gates) | `scripts/` |
| Paper section sources (drafts EN/FR for Paper 1/2/3) | `docs/papers/` |
| Submission tracker, byline policy, cross-paper writing discipline | `papers/` |
| Outreach, reviewer recruitment, mail drafts | `ops/` |

Several of these directories have their own `CLAUDE.md` with
domain-specific guidance — read those when you enter them.

## Read-first context

Before writing any code or doc claim, read the relevant spec — this
project is spec-driven and numbers / axiom IDs / invariants are
load-bearing :

- `docs/specs/2026-04-17-dreamofkiki-master-design.md` — vision, 5
  tracks, 28-week cycle, G1..G6 gates.
- `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` —
  formal framework, 8 primitives, 4 channels, axioms DR-0..DR-4,
  DualVer rules, Conformance Criterion.
- `docs/glossary.md` — canonical terminology. Do not invent synonyms.
- `docs/invariants/` — I/S/K families. Every runtime check cites one.
- `STATUS.md` + `CHANGELOG.md` — current gate / version / open actions.

## Common commands

All invocations go through `uv` (no Makefile / justfile). Python ≥3.12.

```bash
# Install — pulls dev, fmri, teacher extras (MLX, hypothesis, nilearn, llama-cpp-python)
uv sync --all-extras

# Full test suite with coverage gate (fails under 90 %)
uv run pytest

# Narrow scopes
uv run pytest tests/conformance/ -v                 # axioms + invariants
uv run pytest tests/reproducibility/ -v --no-cov    # R1 bit-exact (MLX/Metal on Apple Silicon)
uv run pytest tests/unit/test_foo.py::test_bar      # single test
uv run pytest -k "dr2 and not slow"                 # keyword selection

# Lint / type / format — must pass before commit (CI enforces)
uv run ruff check .
uv run mypy harness tests          # strict mode (pyproject.toml)

# Harness CLI (entry point defined in pyproject.toml)
uv run dream-harness --help

# Pilot / gate drivers — one per G-gate, always pass explicit seed
uv run python scripts/pilot_g2.py --profile P_equ --seed 42
uv run python scripts/pilot_cycle3_sanity.py --seed 42
uv run python scripts/conformance_matrix.py

# Render paper figures from registered runs
uv run python scripts/render_figures.py --gate G4
```

CI runs on push/PR : `ci.yml` (ubuntu-latest, ruff + mypy + pytest)
and `r1-nightly.yml` (macos-14 Apple Silicon, nightly `tests/reproducibility/`
with `golden_hashes.json` as failure artifact). A change that breaks
R1 bit-exactness will only surface on the nightly macOS runner —
run `tests/reproducibility/` locally on Apple Silicon before pushing
harness changes.

## Working rules (research discipline)

1. **Determinism is a contract.** `harness/storage/run_registry.py`
   enforces R1 : `(c_version, profile, seed, commit_sha)` →
   `run_id` is bit-stable. Never change seeds in-place ; add a new
   seed and register a new run.
2. **Cite the invariant / axiom ID** in every guard, test, and
   commit message that enforces one (e.g. `S2 finite`, `DR-1`).
3. **Synthetic vs real data.** Today's retained benchmark and
   pilots are **synthetic placeholders** (see `scripts/pilot_g2.py`
   docstring). Never report synthetic results as empirical claims.
4. **DualVer bumps** : formal axis bump requires proof or spec
   change ; empirical axis bump requires gate result. Both are
   recorded in `CHANGELOG.md` and `STATUS.md`.
5. **Reproducibility over speed.** Prefer deterministic ops, seeded
   RNGs, hashed input artifacts (benchmarks ship with `.sha256`).
6. **Fork hygiene for `kiki_oniric/`** : jalonné rebase policy
   (S1/S8/S18, see `docs/fork-decision.md`). Do not cherry-pick
   upstream outside those windows.

## Agent workflow

- Start by reading `STATUS.md` to learn current sprint (S-number)
  and active gate.
- For any change touching an axiom, invariant, or primitive
  signature : propose a DualVer bump in the commit and update the
  changelog + framework-C spec consistently.
- Before claiming a gate passed : run the full test suite, confirm
  coverage ≥ 90 % (pytest config already enforces it), and verify
  no `.coverage` / run-registry leakage.
- When unsure which axiom / invariant applies, search `docs/` first
  — the naming is standardized.
- Commit rules are validator-enforced (see `CONTRIBUTING.md`) :
  subject ≤50 chars, scope ≥3 chars (e.g. `paper1`, `dream`, `fr`),
  body lines ≤72 chars, body required for functional changes, English
  only, no AI attribution, no `--no-verify`. EN→FR propagation : any
  change to an English paper/spec must update its FR counterpart under
  `docs/specs-fr/` or `docs/papers/paper{1,2}-fr/` in the same PR.

## Paper-to-code mapping

- **Paper 1** (formal) ↔ `docs/specs/`, `docs/proofs/`,
  `docs/invariants/`, `tests/conformance/axioms/`.
- **Paper 2** (ablation) ↔ `kiki_oniric/profiles/`, `harness/`,
  `scripts/pilot_*.py`, run-registry artifacts.

Experimental claims in either paper must resolve to a registered
`run_id` or a proof file.

## License

Code MIT, docs CC-BY-4.0. Authorship byline : *dreamOfkiki project
contributors*. No AI attribution in commit trailers.
