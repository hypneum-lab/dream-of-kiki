# Effect-Size Benchmarks (Hu 2020 + Javadi 2024) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Encode the empirical effect-size targets of [@hu2020tmr] (TMR meta-analysis, k=91, N=2004; overall Hedges' g = 0.29 [0.21, 0.38]; NREM2 g = 0.32 [0.04, 0.60]; SWS g = 0.27 [0.20, 0.35]) and [@javadi2024sleeprestriction] (sleep-restriction meta, k=39, N=1234; g = 0.29 [0.13, 0.44]) as typed, frozen, immutable constants inside the harness, with full unit-test coverage and a typed verdict helper, so any future G4 pilot can compare its observed effect sizes against `P_equ` "gain floors" and `P_min` "decrement floors" deterministically — without rewriting numbers in code or paper prose.

**Architecture:** One new module `harness/benchmarks/effect_size_targets.py` exporting a frozen `@dataclass(frozen=True) EffectSizeTarget` and four module-level constants (`HU_2020_OVERALL`, `HU_2020_NREM2`, `HU_2020_SWS`, `JAVADI_2024_OVERALL`). One new test file `tests/unit/harness/test_effect_size_targets.py` covering construction, immutability, CI bounds methods, signed distance, and bibtex-key resolution against `docs/papers/paper1/references.bib`. One small documentation update in `docs/papers/paper1/methodology.md` §6.6 to point readers at the new harness module. No mutation of existing harness modules. No new pyproject dependencies (uses stdlib `dataclasses` + existing `pytest` + `hypothesis`).

**Tech Stack:** Python 3.12+, stdlib `dataclasses` (frozen), stdlib `pathlib`, `typing.Literal`, `pytest`, `hypothesis` (already a hard dep, see `pyproject.toml:14`), `uv run` invocations, conventional-commit messages with scope ≥3 chars. No pydantic / attrs (project convention is stdlib `@dataclass(frozen=True)` — confirmed by `harness/benchmarks/retained/retained.py:20-27`).

---

## Background

The Hu 2020 and Javadi 2024 meta-analyses are already cited in `docs/papers/paper1/methodology.md` §6.1 as "effect-size floors" for hypotheses H1 (P_equ consolidation gain) and H3 (P_min decrement). The numbers currently live only in BibTeX `note=` fields in `docs/papers/paper1/references.bib` (lines 598-609 for `hu2020tmr`, lines 655-664 for `javadi2024sleeprestriction`) and in paper prose. They are not machine-readable, not type-checked, and cannot be referenced by a future pilot script without copy-paste — which violates the R1 "numbers in READMEs are claims, not prose" rule (`/Users/electron/hypneum-lab/CLAUDE.md` §"Reproducibility contract R1").

This plan does **not** add a new benchmark, run a pilot, or claim an empirical result. It encodes published meta-analytic targets as typed constants so a future G4 pilot can call `target.is_within_ci(observed_g)` deterministically. No DualVer bump is required (no axiom / invariant / primitive signature change, no benchmark hash change). The empirical-axis EC remains unchanged because no new run is registered.

## Files touched

| File | Action | Lines (approx) |
|------|--------|---------------|
| `harness/benchmarks/effect_size_targets.py` | **create** | ~120 |
| `tests/unit/harness/__init__.py` | **create** | 0 (empty marker) |
| `tests/unit/harness/test_effect_size_targets.py` | **create** | ~180 |
| `docs/papers/paper1/methodology.md` | **edit** §6.6 only | +5 lines |

No edits to `harness/storage/run_registry.py`, `harness/benchmarks/__init__.py`, `pyproject.toml`, `STATUS.md`, `CHANGELOG.md`, `docs/specs/`, or any conformance test. Out of scope: any helper that *runs* a pilot, any `effect_size_validator.py` runtime helper (Task 7 originally proposed it; downgraded to a future-work note in §6.6 because no caller exists yet — adding a validator without a caller is YAGNI and violates the "research code, not a product" rule).

## Self-review compliance checklist (pre-execution)

- Header matches required format: yes (line 1 `# … Implementation Plan`, line 3 `> **For agentic workers:** REQUIRED SUB-SKILL: …`).
- Goal / Architecture / Tech Stack sections present: yes.
- All file paths are absolute or repo-root-relative: yes.
- All `pytest` / `git` invocations are exact and copy-pasteable: yes (verified against `pyproject.toml:65` pytest config).
- No placeholders (TBD / TODO / "later" / "to be defined"): verified by Self-Review task at the end.
- Type signatures: `Literal["P_min", "P_equ", "P_max"]` matches the three-profile vocabulary already used in `harness/storage/run_registry.py:113` and `tests/unit/test_run_registry.py:21`. `stratum: str | None` uses PEP 604 syntax (Python 3.12+ OK).
- Numbers are exactly transcribed from `docs/papers/paper1/references.bib` (verified Hu lines 598-609, Javadi lines 655-664).
- All commits use conventional-commit scope ≥3 chars (`harness`, `tests`, `paper1`); subject ≤50 chars (validated per task).

---

### Task 0: Investigate harness/benchmarks/ structure (read-only)

**Goal:** Confirm conventions before writing any code.

- [ ] **Step 1:** Verify the existing benchmarks layout matches plan assumptions.
  ```bash
  ls /Users/electron/hypneum-lab/dream-of-kiki/harness/benchmarks/
  # expect: __init__.py  mega_v2  retained
  ```
- [ ] **Step 2:** Re-read the dataclass-style template that `effect_size_targets.py` will mirror.
  - Read `/Users/electron/hypneum-lab/dream-of-kiki/harness/benchmarks/retained/retained.py` lines 20-27 (the `@dataclass(frozen=True) class RetainedBenchmark`). The new module must use the same pattern: stdlib `dataclasses.dataclass(frozen=True)`, no pydantic, docstring-first.
- [ ] **Step 3:** Re-read the test template.
  - Read `/Users/electron/hypneum-lab/dream-of-kiki/tests/unit/test_retained_benchmark.py` lines 1-40 to confirm import style, fixture style, `REPO_ROOT = Path(__file__).resolve().parents[2]` pattern.
- [ ] **Step 4:** Confirm `tests/unit/harness/` does **not** yet exist.
  ```bash
  ls /Users/electron/hypneum-lab/dream-of-kiki/tests/unit/harness/ 2>&1 || echo "absent — Task 1 will create it"
  ```
- [ ] **Step 5:** Confirm `hypothesis` is importable (already a hard dep, `pyproject.toml:14`).
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run python -c "import hypothesis; print(hypothesis.__version__)"
  ```

No commit. No file modification.

---

### Task 1: Add failing test for `EffectSizeTarget` dataclass construction + immutability

**Files:**
- Create: `/Users/electron/hypneum-lab/dream-of-kiki/tests/unit/harness/__init__.py` (empty file, package marker)
- Create: `/Users/electron/hypneum-lab/dream-of-kiki/tests/unit/harness/test_effect_size_targets.py` (initial 4 tests)

- [ ] **Step 1: Create the test package marker.**
  ```bash
  : > /Users/electron/hypneum-lab/dream-of-kiki/tests/unit/harness/__init__.py
  ```

- [ ] **Step 2: Write 4 failing tests** in `tests/unit/harness/test_effect_size_targets.py`:
  ```python
  """Unit tests for empirical effect-size targets (Hu 2020 + Javadi 2024).

  Targets are typed, frozen constants encoding published meta-analytic
  Hedges' g and 95% CIs. Every constant must be immutable and resolve
  to a real BibTeX key in docs/papers/paper1/references.bib.
  """
  from dataclasses import FrozenInstanceError

  import pytest

  from harness.benchmarks.effect_size_targets import EffectSizeTarget


  def test_target_constructs_with_all_fields() -> None:
      target = EffectSizeTarget(
          name="dummy_overall",
          hedges_g=0.29,
          ci_low=0.21,
          ci_high=0.38,
          sample_size_n=2004,
          k_studies=91,
          source_bibtex_key="hu2020tmr",
          profile_target="P_equ",
          stratum=None,
      )
      assert target.name == "dummy_overall"
      assert target.hedges_g == 0.29


  def test_target_is_frozen() -> None:
      target = EffectSizeTarget(
          name="dummy",
          hedges_g=0.29,
          ci_low=0.21,
          ci_high=0.38,
          sample_size_n=2004,
          k_studies=91,
          source_bibtex_key="hu2020tmr",
          profile_target="P_equ",
          stratum=None,
      )
      with pytest.raises(FrozenInstanceError):
          target.hedges_g = 0.99  # type: ignore[misc]


  def test_target_rejects_inverted_ci() -> None:
      """ci_low must be <= hedges_g <= ci_high (sanity, not stat rule)."""
      with pytest.raises(ValueError, match="ci_low.*ci_high"):
          EffectSizeTarget(
              name="bad",
              hedges_g=0.29,
              ci_low=0.50,   # > ci_high — invalid
              ci_high=0.10,
              sample_size_n=10,
              k_studies=1,
              source_bibtex_key="hu2020tmr",
              profile_target="P_equ",
              stratum=None,
          )


  def test_target_rejects_g_outside_ci() -> None:
      with pytest.raises(ValueError, match="hedges_g.*ci"):
          EffectSizeTarget(
              name="bad",
              hedges_g=0.99,    # outside [0.21, 0.38]
              ci_low=0.21,
              ci_high=0.38,
              sample_size_n=10,
              k_studies=1,
              source_bibtex_key="hu2020tmr",
              profile_target="P_equ",
              stratum=None,
          )
  ```

- [ ] **Step 3: Run — must fail with `ModuleNotFoundError`.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run pytest tests/unit/harness/test_effect_size_targets.py -v --no-cov 2>&1 | tail -20
  ```
  Expected: 4 collection errors / `ModuleNotFoundError: No module named 'harness.benchmarks.effect_size_targets'`.

- [ ] **Step 4: Implement** `/Users/electron/hypneum-lab/dream-of-kiki/harness/benchmarks/effect_size_targets.py` with the dataclass only (constants come in Tasks 2-3):
  ```python
  """Empirical effect-size targets for the C framework.

  Encodes published meta-analytic Hedges' g and 95 % CIs from
  [@hu2020tmr] (TMR, k=91, N=2004) and [@javadi2024sleeprestriction]
  (sleep-restriction, k=39, N=1234). Treated as immutable empirical
  anchors that future pilots compare observed effect sizes against.

  These targets do **not** themselves trigger an EC bump : they are
  external published numbers, not registered run outputs. They are
  the floor against which `P_equ` consolidation gains and `P_min`
  decrement magnitudes are evaluated for plausibility.
  """
  from __future__ import annotations

  from dataclasses import dataclass
  from typing import Literal

  ProfileLabel = Literal["P_min", "P_equ", "P_max"]


  @dataclass(frozen=True)
  class EffectSizeTarget:
      """A single published Hedges' g target with 95% CI.

      Attributes :
          name : human-readable label (e.g. ``"hu2020_nrem2"``).
          hedges_g : point estimate of standardized mean difference.
          ci_low, ci_high : 95% confidence interval bounds (CI_low <=
              hedges_g <= CI_high enforced at construction).
          sample_size_n : aggregated participant count across studies.
          k_studies : number of independent studies in the meta.
          source_bibtex_key : key in
              ``docs/papers/paper1/references.bib`` (validated by the
              test suite, not at runtime).
          profile_target : which dream-of-kiki profile this target
              floors — ``"P_equ"`` for consolidation gains,
              ``"P_min"`` for sleep-restriction-style decrements.
          stratum : optional sleep-stage / scope label
              (``"NREM2"``, ``"SWS"``, ``None`` for overall).
      """

      name: str
      hedges_g: float
      ci_low: float
      ci_high: float
      sample_size_n: int
      k_studies: int
      source_bibtex_key: str
      profile_target: ProfileLabel
      stratum: str | None

      def __post_init__(self) -> None:
          if self.ci_low > self.ci_high:
              raise ValueError(
                  f"ci_low ({self.ci_low}) must be <= ci_high "
                  f"({self.ci_high}) for target {self.name!r}"
              )
          if not (self.ci_low <= self.hedges_g <= self.ci_high):
              raise ValueError(
                  f"hedges_g ({self.hedges_g}) must lie within ci "
                  f"[{self.ci_low}, {self.ci_high}] for target "
                  f"{self.name!r}"
              )
  ```

- [ ] **Step 5: Run — 4 tests must pass.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run pytest tests/unit/harness/test_effect_size_targets.py -v --no-cov
  ```
  Expected: `4 passed`.

- [ ] **Step 6: Lint + type-check the new files.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run ruff check harness/benchmarks/effect_size_targets.py tests/unit/harness/ && \
    uv run mypy harness/benchmarks/effect_size_targets.py tests/unit/harness/
  ```
  Both must report no errors.

- [ ] **Step 7: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    git add harness/benchmarks/effect_size_targets.py \
            tests/unit/harness/__init__.py \
            tests/unit/harness/test_effect_size_targets.py && \
    git commit -m "feat(harness): add EffectSizeTarget dataclass" -m "Frozen typed container for published Hedges' g + 95% CI. Validates
  ci_low <= hedges_g <= ci_high at construction. Constants and CI
  helper methods land in follow-up commits."
  ```
  Subject: 38 chars ≤ 50; scope `harness` is 7 chars ≥ 3.

---

### Task 2: Add `HU_2020_OVERALL`, `HU_2020_NREM2`, `HU_2020_SWS` constants

**Files:**
- Modify: `/Users/electron/hypneum-lab/dream-of-kiki/harness/benchmarks/effect_size_targets.py` (append constants)
- Modify: `/Users/electron/hypneum-lab/dream-of-kiki/tests/unit/harness/test_effect_size_targets.py` (append 3 constant tests)

- [ ] **Step 1: Write failing tests** by appending to `test_effect_size_targets.py`:
  ```python
  # ----------------------------------------------------------------------
  # Hu 2020 TMR meta-analysis constants
  # ----------------------------------------------------------------------


  def test_hu2020_overall_matches_published() -> None:
      """[@hu2020tmr] overall Hedges' g = 0.29 [0.21, 0.38], k=91, N=2004."""
      from harness.benchmarks.effect_size_targets import HU_2020_OVERALL

      assert HU_2020_OVERALL.hedges_g == 0.29
      assert HU_2020_OVERALL.ci_low == 0.21
      assert HU_2020_OVERALL.ci_high == 0.38
      assert HU_2020_OVERALL.k_studies == 91
      assert HU_2020_OVERALL.sample_size_n == 2004
      assert HU_2020_OVERALL.source_bibtex_key == "hu2020tmr"
      assert HU_2020_OVERALL.profile_target == "P_equ"
      assert HU_2020_OVERALL.stratum is None


  def test_hu2020_nrem2_matches_published() -> None:
      """[@hu2020tmr] NREM2 stratum g = 0.32 [0.04, 0.60]."""
      from harness.benchmarks.effect_size_targets import HU_2020_NREM2

      assert HU_2020_NREM2.hedges_g == 0.32
      assert HU_2020_NREM2.ci_low == 0.04
      assert HU_2020_NREM2.ci_high == 0.60
      assert HU_2020_NREM2.stratum == "NREM2"
      assert HU_2020_NREM2.profile_target == "P_equ"
      assert HU_2020_NREM2.source_bibtex_key == "hu2020tmr"


  def test_hu2020_sws_matches_published() -> None:
      """[@hu2020tmr] SWS stratum g = 0.27 [0.20, 0.35]."""
      from harness.benchmarks.effect_size_targets import HU_2020_SWS

      assert HU_2020_SWS.hedges_g == 0.27
      assert HU_2020_SWS.ci_low == 0.20
      assert HU_2020_SWS.ci_high == 0.35
      assert HU_2020_SWS.stratum == "SWS"
      assert HU_2020_SWS.profile_target == "P_equ"
  ```

- [ ] **Step 2: Run — 3 new tests must fail with `ImportError`.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run pytest tests/unit/harness/test_effect_size_targets.py -v --no-cov 2>&1 | tail -20
  ```
  Expected: `4 passed, 3 errors` (ImportError on the three new symbols).

- [ ] **Step 3: Append constants** at the bottom of `harness/benchmarks/effect_size_targets.py`:
  ```python


  # ----------------------------------------------------------------------
  # Hu et al. 2020 TMR meta-analysis (Psychological Bulletin)
  # k = 91 reports, 212 effect sizes, N = 2004 participants
  # Source : docs/papers/paper1/references.bib :: hu2020tmr
  # ----------------------------------------------------------------------

  HU_2020_OVERALL: EffectSizeTarget = EffectSizeTarget(
      name="hu2020_overall",
      hedges_g=0.29,
      ci_low=0.21,
      ci_high=0.38,
      sample_size_n=2004,
      k_studies=91,
      source_bibtex_key="hu2020tmr",
      profile_target="P_equ",
      stratum=None,
  )

  HU_2020_NREM2: EffectSizeTarget = EffectSizeTarget(
      name="hu2020_nrem2",
      hedges_g=0.32,
      ci_low=0.04,
      ci_high=0.60,
      sample_size_n=2004,
      k_studies=91,
      source_bibtex_key="hu2020tmr",
      profile_target="P_equ",
      stratum="NREM2",
  )

  HU_2020_SWS: EffectSizeTarget = EffectSizeTarget(
      name="hu2020_sws",
      hedges_g=0.27,
      ci_low=0.20,
      ci_high=0.35,
      sample_size_n=2004,
      k_studies=91,
      source_bibtex_key="hu2020tmr",
      profile_target="P_equ",
      stratum="SWS",
  )
  ```

- [ ] **Step 4: Run — all 7 tests must pass.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run pytest tests/unit/harness/test_effect_size_targets.py -v --no-cov
  ```
  Expected: `7 passed`.

- [ ] **Step 5: Lint + type-check.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run ruff check harness/benchmarks/effect_size_targets.py tests/unit/harness/ && \
    uv run mypy harness/benchmarks/effect_size_targets.py tests/unit/harness/
  ```

- [ ] **Step 6: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    git add harness/benchmarks/effect_size_targets.py tests/unit/harness/test_effect_size_targets.py && \
    git commit -m "feat(harness): encode Hu 2020 TMR g targets" -m "Add HU_2020_OVERALL (g=0.29 [0.21, 0.38], k=91, N=2004),
  HU_2020_NREM2 (g=0.32 [0.04, 0.60]), HU_2020_SWS (g=0.27
  [0.20, 0.35]) as P_equ consolidation-gain floors. Source :
  docs/papers/paper1/references.bib :: hu2020tmr."
  ```

---

### Task 3: Add `JAVADI_2024_OVERALL` constant

**Files:**
- Modify: `/Users/electron/hypneum-lab/dream-of-kiki/harness/benchmarks/effect_size_targets.py` (append)
- Modify: `/Users/electron/hypneum-lab/dream-of-kiki/tests/unit/harness/test_effect_size_targets.py` (append 1 test)

- [ ] **Step 1: Write failing test** by appending:
  ```python
  # ----------------------------------------------------------------------
  # Javadi 2024 sleep-restriction meta-analysis constants
  # ----------------------------------------------------------------------


  def test_javadi2024_overall_matches_published() -> None:
      """[@javadi2024sleeprestriction] g = 0.29 [0.13, 0.44], k=39, N=1234."""
      from harness.benchmarks.effect_size_targets import JAVADI_2024_OVERALL

      assert JAVADI_2024_OVERALL.hedges_g == 0.29
      assert JAVADI_2024_OVERALL.ci_low == 0.13
      assert JAVADI_2024_OVERALL.ci_high == 0.44
      assert JAVADI_2024_OVERALL.k_studies == 39
      assert JAVADI_2024_OVERALL.sample_size_n == 1234
      assert JAVADI_2024_OVERALL.source_bibtex_key == "javadi2024sleeprestriction"
      # P_min : sleep restriction = degraded substrate decrement floor
      assert JAVADI_2024_OVERALL.profile_target == "P_min"
      assert JAVADI_2024_OVERALL.stratum is None
  ```

- [ ] **Step 2: Run — must fail with ImportError.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run pytest tests/unit/harness/test_effect_size_targets.py::test_javadi2024_overall_matches_published -v --no-cov
  ```

- [ ] **Step 3: Append constant** to `effect_size_targets.py`:
  ```python


  # ----------------------------------------------------------------------
  # Javadi et al. 2024 sleep-restriction meta (Neurosci Biobehav Rev)
  # 39 reports, 125 effect sizes, N = 1234, no detected pub bias
  # Source : docs/papers/paper1/references.bib :: javadi2024sleeprestriction
  # ----------------------------------------------------------------------

  JAVADI_2024_OVERALL: EffectSizeTarget = EffectSizeTarget(
      name="javadi2024_overall",
      hedges_g=0.29,
      ci_low=0.13,
      ci_high=0.44,
      sample_size_n=1234,
      k_studies=39,
      source_bibtex_key="javadi2024sleeprestriction",
      profile_target="P_min",
      stratum=None,
  )
  ```

- [ ] **Step 4: Run — all 8 tests must pass.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run pytest tests/unit/harness/test_effect_size_targets.py -v --no-cov
  ```

- [ ] **Step 5: Lint + type-check.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run ruff check harness/benchmarks/effect_size_targets.py && \
    uv run mypy harness/benchmarks/effect_size_targets.py
  ```

- [ ] **Step 6: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    git add harness/benchmarks/effect_size_targets.py tests/unit/harness/test_effect_size_targets.py && \
    git commit -m "feat(harness): encode Javadi 2024 SR g target" -m "Add JAVADI_2024_OVERALL (g=0.29 [0.13, 0.44], k=39, N=1234)
  as P_min sleep-restriction decrement floor. Source :
  docs/papers/paper1/references.bib :: javadi2024sleeprestriction."
  ```

---

### Task 4: Implement `is_within_ci` method (with Hypothesis property test)

**Files:**
- Modify: `/Users/electron/hypneum-lab/dream-of-kiki/harness/benchmarks/effect_size_targets.py` (add method)
- Modify: `/Users/electron/hypneum-lab/dream-of-kiki/tests/unit/harness/test_effect_size_targets.py` (add 3 tests : 2 example + 1 Hypothesis property)

- [ ] **Step 1: Write failing tests** by appending:
  ```python
  from hypothesis import given, settings
  from hypothesis import strategies as st

  from harness.benchmarks.effect_size_targets import HU_2020_OVERALL


  def test_is_within_ci_inclusive_at_bounds() -> None:
      assert HU_2020_OVERALL.is_within_ci(HU_2020_OVERALL.ci_low) is True
      assert HU_2020_OVERALL.is_within_ci(HU_2020_OVERALL.ci_high) is True
      assert HU_2020_OVERALL.is_within_ci(HU_2020_OVERALL.hedges_g) is True


  def test_is_within_ci_outside_returns_false() -> None:
      assert HU_2020_OVERALL.is_within_ci(0.0) is False
      assert HU_2020_OVERALL.is_within_ci(0.5) is False
      assert HU_2020_OVERALL.is_within_ci(-1.0) is False


  @given(observed=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False))
  @settings(max_examples=200, deterministic=True)
  def test_is_within_ci_property(observed: float) -> None:
      """For any observed in [-5, 5], is_within_ci agrees with bounds check."""
      result = HU_2020_OVERALL.is_within_ci(observed)
      expected = HU_2020_OVERALL.ci_low <= observed <= HU_2020_OVERALL.ci_high
      assert result is expected
  ```

- [ ] **Step 2: Run — must fail with `AttributeError: ... has no attribute 'is_within_ci'`.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run pytest tests/unit/harness/test_effect_size_targets.py -v --no-cov 2>&1 | tail -20
  ```

- [ ] **Step 3: Add method** inside `class EffectSizeTarget` (right after `__post_init__`):
  ```python
      def is_within_ci(self, observed: float) -> bool:
          """Return True iff ``observed`` lies within the 95 % CI (inclusive).

          Used by future G4 pilots to decide whether an observed
          empirical effect size from a registered run sits inside the
          published meta-analytic interval — i.e. is empirically
          consistent with the published anchor.
          """
          return self.ci_low <= observed <= self.ci_high
  ```

- [ ] **Step 4: Run — all 11 tests must pass.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run pytest tests/unit/harness/test_effect_size_targets.py -v --no-cov
  ```
  Expected: `11 passed`.

- [ ] **Step 5: Lint + type-check.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run ruff check harness/benchmarks/effect_size_targets.py tests/unit/harness/ && \
    uv run mypy harness/benchmarks/effect_size_targets.py tests/unit/harness/
  ```

- [ ] **Step 6: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    git add harness/benchmarks/effect_size_targets.py tests/unit/harness/test_effect_size_targets.py && \
    git commit -m "feat(harness): add is_within_ci on target" -m "Inclusive bounds check used by future G4 pilot to decide whether
  observed Hedges' g sits in the published 95% CI. Hypothesis
  property test (200 examples, deterministic) covers the full
  [-5, 5] range against the bounds."
  ```

---

### Task 5: Implement `distance_from_target` method

**Files:**
- Modify: `/Users/electron/hypneum-lab/dream-of-kiki/harness/benchmarks/effect_size_targets.py`
- Modify: `/Users/electron/hypneum-lab/dream-of-kiki/tests/unit/harness/test_effect_size_targets.py`

- [ ] **Step 1: Write failing tests** by appending:
  ```python
  def test_distance_from_target_zero_at_point_estimate() -> None:
      assert HU_2020_OVERALL.distance_from_target(HU_2020_OVERALL.hedges_g) == 0.0


  def test_distance_from_target_signed() -> None:
      """observed - hedges_g, signed (positive = above target)."""
      assert HU_2020_OVERALL.distance_from_target(0.40) == pytest.approx(
          0.40 - HU_2020_OVERALL.hedges_g
      )
      assert HU_2020_OVERALL.distance_from_target(0.10) == pytest.approx(
          0.10 - HU_2020_OVERALL.hedges_g
      )
      # negative when observed below target
      assert HU_2020_OVERALL.distance_from_target(0.0) < 0.0
  ```

- [ ] **Step 2: Run — must fail with `AttributeError`.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run pytest tests/unit/harness/test_effect_size_targets.py::test_distance_from_target_signed -v --no-cov
  ```

- [ ] **Step 3: Add method** right after `is_within_ci` :
  ```python
      def distance_from_target(self, observed: float) -> float:
          """Signed distance ``observed - hedges_g`` (positive = above).

          Use the sign to decide whether an observation overshoots
          (positive : larger consolidation gain than the meta-analytic
          point estimate) or undershoots (negative : smaller gain).
          Magnitude has no statistical interpretation by itself —
          combine with :meth:`is_within_ci` for a verdict.
          """
          return observed - self.hedges_g
  ```

- [ ] **Step 4: Run — all 13 tests must pass.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run pytest tests/unit/harness/test_effect_size_targets.py -v --no-cov
  ```

- [ ] **Step 5: Lint + type-check.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run ruff check harness/benchmarks/effect_size_targets.py tests/unit/harness/ && \
    uv run mypy harness/benchmarks/effect_size_targets.py tests/unit/harness/
  ```

- [ ] **Step 6: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    git add harness/benchmarks/effect_size_targets.py tests/unit/harness/test_effect_size_targets.py && \
    git commit -m "feat(harness): add distance_from_target method" -m "Signed observed - hedges_g distance for G4-style verdicts.
  Sign indicates over/undershoot vs published meta-analytic point
  estimate; magnitude is descriptive only — combine with
  is_within_ci for a binary pass/fail."
  ```

---

### Task 6: Validate `source_bibtex_key` resolves against `references.bib`

**Files:**
- Modify: `/Users/electron/hypneum-lab/dream-of-kiki/tests/unit/harness/test_effect_size_targets.py` (append integration test)

This test reads `docs/papers/paper1/references.bib` and asserts every constant's `source_bibtex_key` appears as a `@…{key,` entry. Prevents typos and orphan-key drift.

- [ ] **Step 1: Append test:**
  ```python
  from pathlib import Path

  REPO_ROOT = Path(__file__).resolve().parents[3]
  REFERENCES_BIB = REPO_ROOT / "docs" / "papers" / "paper1" / "references.bib"


  def _bibtex_keys(bib_path: Path) -> set[str]:
      """Extract @…{KEY, entries from a BibTeX file."""
      import re

      text = bib_path.read_text(encoding="utf-8")
      # Matches: @article{key, OR @misc{key, etc.
      pattern = re.compile(r"^@[A-Za-z]+\{([A-Za-z0-9_:-]+),", re.MULTILINE)
      return set(pattern.findall(text))


  def test_references_bib_exists() -> None:
      assert REFERENCES_BIB.is_file(), (
          f"references.bib not found at {REFERENCES_BIB} — "
          "either the file moved or REPO_ROOT parents level is wrong"
      )


  def test_all_target_keys_resolve_in_references_bib() -> None:
      """Every EffectSizeTarget.source_bibtex_key must appear in references.bib."""
      from harness.benchmarks.effect_size_targets import (
          HU_2020_NREM2,
          HU_2020_OVERALL,
          HU_2020_SWS,
          JAVADI_2024_OVERALL,
      )

      keys = _bibtex_keys(REFERENCES_BIB)
      for target in (HU_2020_OVERALL, HU_2020_NREM2, HU_2020_SWS, JAVADI_2024_OVERALL):
          assert target.source_bibtex_key in keys, (
              f"target {target.name!r} cites bibtex key "
              f"{target.source_bibtex_key!r} but it is not in "
              f"{REFERENCES_BIB}"
          )
  ```

- [ ] **Step 2: Run — both new tests must pass on first try** (Hu and Javadi keys are at `references.bib:598` and `references.bib:655`):
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run pytest tests/unit/harness/test_effect_size_targets.py -v --no-cov
  ```
  Expected: `15 passed`.

- [ ] **Step 3: Lint + type-check.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run ruff check tests/unit/harness/ && \
    uv run mypy tests/unit/harness/
  ```

- [ ] **Step 4: Sanity — confirm REPO_ROOT computation is correct.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run python -c "
  from pathlib import Path
  p = Path('tests/unit/harness/test_effect_size_targets.py').resolve().parents[3]
  print(p)
  assert (p / 'docs' / 'papers' / 'paper1' / 'references.bib').is_file()
  "
  ```
  Expected: prints `/Users/electron/hypneum-lab/dream-of-kiki`.

- [ ] **Step 5: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    git add tests/unit/harness/test_effect_size_targets.py && \
    git commit -m "test(harness): bibtex key resolution check" -m "Verifies every EffectSizeTarget.source_bibtex_key resolves
  against docs/papers/paper1/references.bib so a typo or removal
  is caught at test time, not paper-build time."
  ```

---

### Task 7: Update `docs/papers/paper1/methodology.md` §6.6 to point to harness module

**Files:**
- Modify: `/Users/electron/hypneum-lab/dream-of-kiki/docs/papers/paper1/methodology.md` (§6.6 only)

The methodology already cites Hu 2020 and Javadi 2024 in §6.1 (lines 22-24, 32-34) and Hu 2020 again in §6.6 (line 125). The edit appends a single forward-reference paragraph at the end of §6.6 pointing readers and future pilot authors at the typed harness module.

- [ ] **Step 1: Edit §6.6 of `methodology.md`.** Use the `Edit` tool with this exact `old_string` (currently the last line of §6.6, before the `---` separator at line 131):
  - `old_string`:
    ```
    inherits
    its biological grounding from the SWS up-state interleaving
    result of [@biorxiv2025thalamocortical].
    ```
  - `new_string`:
    ```
    inherits
    its biological grounding from the SWS up-state interleaving
    result of [@biorxiv2025thalamocortical].

    The numerical anchors of [@hu2020tmr] (overall, NREM2, SWS) and
    [@javadi2024sleeprestriction] are encoded as typed, frozen
    constants in `harness.benchmarks.effect_size_targets` (`HU_2020_OVERALL`,
    `HU_2020_NREM2`, `HU_2020_SWS`, `JAVADI_2024_OVERALL`) so a future
    G4 pilot compares observed effect sizes against the published 95 %
    CIs deterministically (`is_within_ci`, `distance_from_target`).
    No empirical (EC) bump is implied : these constants encode external
    published numbers, not registered run outputs.
    ```

- [ ] **Step 2: Mirror in FR counterpart** if it exists. Check first :
  ```bash
  test -f /Users/electron/hypneum-lab/dream-of-kiki/docs/papers/paper1-fr/methodology.md && \
    echo "FR mirror exists — must update in same PR per CONTRIBUTING.md EN→FR rule" || \
    echo "no FR mirror — skip"
  ```
  If it exists, add the equivalent French paragraph at the end of `paper1-fr/methodology.md` §6.6. Translation guidance :
  ```
  Les ancres numériques de [@hu2020tmr] (global, NREM2, SWS) et
  [@javadi2024sleeprestriction] sont encodées comme constantes
  typées et figées dans `harness.benchmarks.effect_size_targets`
  (`HU_2020_OVERALL`, `HU_2020_NREM2`, `HU_2020_SWS`,
  `JAVADI_2024_OVERALL`) afin qu'un futur pilote G4 compare les
  tailles d'effet observées aux IC 95 % publiés de façon
  déterministe (`is_within_ci`, `distance_from_target`). Aucun
  bump empirique (EC) n'est impliqué : ces constantes encodent
  des nombres externes publiés, pas des sorties de runs
  enregistrés.
  ```

- [ ] **Step 3: Verify markdown renders cleanly.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run ruff check docs/ 2>&1 | tail -5 || true   # ruff doesn't lint md, but confirm no incidental .py edits
  ```

- [ ] **Step 4: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    git add docs/papers/paper1/methodology.md
  # Conditionally add FR mirror if it was edited:
  test -f docs/papers/paper1-fr/methodology.md && git add docs/papers/paper1-fr/methodology.md
  git commit -m "docs(paper1): cite effect_size_targets module" -m "Forward-reference §6.6 to the new harness.benchmarks.effect_size_targets
  module so future G4 pilot authors find the typed Hu 2020 + Javadi
  2024 anchors instead of re-typing numbers from the bib. Mirror in
  paper1-fr/ per CONTRIBUTING.md EN->FR rule (if present)."
  ```

---

### Task 8: Final verification + Self-Review

- [ ] **Step 1: Full test suite + coverage gate.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest
  ```
  Must report all green and `coverage >= 90 %`. The new module is small enough that its own tests give it 100 % coverage; adding it cannot drop the project total below 90 %.

- [ ] **Step 2: Targeted scope re-run.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run pytest tests/unit/harness/ -v --no-cov
  ```
  Expected exact count: `15 passed` (4 dataclass + 3 Hu constants + 1 Javadi + 3 is_within_ci + 2 distance + 2 bibtex).

- [ ] **Step 3: Strict lint + type.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    uv run ruff check . && \
    uv run mypy harness tests
  ```
  Both must report no errors.

- [ ] **Step 4: Confirm no leakage** of `.coverage`, run-registry SQLite, or `__pycache__` into git.
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && git status
  ```
  Expected: clean working tree (no untracked artefacts).

- [ ] **Step 5: Self-Review checklist.**
  - [ ] **Spec coverage** :
    - [ ] All 4 published numbers (Hu overall, Hu NREM2, Hu SWS, Javadi overall) are encoded with the exact values from `references.bib:598-664`.
    - [ ] Each has a unit test pinning every field.
    - [ ] `is_within_ci` and `distance_from_target` are implemented and tested (example + Hypothesis property).
    - [ ] `source_bibtex_key` is validated against the live `.bib` file.
    - [ ] §6.6 of `methodology.md` (and FR mirror if present) points readers to the harness module.
  - [ ] **Placeholder scan** : `grep -rn "TODO\|TBD\|FIXME\|XXX" harness/benchmarks/effect_size_targets.py tests/unit/harness/` returns no hits introduced by this PR.
  - [ ] **Type consistency** :
    - [ ] `ProfileLabel = Literal["P_min", "P_equ", "P_max"]` matches the three-profile vocabulary used throughout the repo (`harness/storage/run_registry.py`, `kiki_oniric/profiles/`).
    - [ ] All constants are typed `: EffectSizeTarget` (not inferred).
    - [ ] `stratum: str | None` uses PEP 604.
    - [ ] mypy strict passes.
  - [ ] **Discipline** :
    - [ ] No DualVer bump made (no axiom / invariant / primitive change, no benchmark hash change). Verified by inspection : neither `STATUS.md`, `CHANGELOG.md`, nor `pyproject.toml` `version` was touched.
    - [ ] No conformance test added (these are external published anchors, not framework-C axioms — they belong in `tests/unit/`, not `tests/conformance/`, per `tests/CLAUDE.md`).
    - [ ] No new pyproject dependency.
    - [ ] No mutation to `harness/storage/run_registry.py` or `RunRegistry` schema.
    - [ ] All commit subjects ≤ 50 chars; all scopes ≥ 3 chars; no `Co-Authored-By` trailer.
    - [ ] EN→FR mirror handled in same commit if `paper1-fr/methodology.md` exists.

- [ ] **Step 6: If Self-Review fails any item**, fix the issue and add a follow-up commit. Do **not** amend any prior commit (per `/Users/electron/hypneum-lab/CLAUDE.md` "Always create NEW commits rather than amending").

- [ ] **Step 7: Report** total commits added (expected : 6 — Tasks 1, 2, 3, 4, 5, 6, 7 each commit; Task 0 and Task 8 do not), final test count (`15 passed` in `tests/unit/harness/`, suite green ≥ 90 % coverage), and whether the FR mirror was edited.

---

## Out of scope (deferred, not implemented here)

- `harness/benchmarks/effect_size_validator.py` runtime helper that takes `(run_id, target)` and emits `Pass / Fail / Within-CI` : no caller exists yet (no G4 pilot script). Adding it would be YAGNI per the "research code, not a product. Correctness > performance" rule (`dream-of-kiki/CLAUDE.md`). When a G4 pilot is drafted, that pilot's plan adds the validator with its first real consumer.
- DualVer bump : not triggered. These constants are external published anchors, not framework-C axioms or invariants.
- Adding entries to `docs/interfaces/eval-matrix.yaml` : these targets are not metrics emitted by a run, they are external comparison anchors. Adding them to `eval-matrix.yaml` would force a bump rule + gate threshold (per `harness/CLAUDE.md` anti-pattern "Don't add a metric to eval-matrix.yaml without also adding its bump rule and its publication-ready-gate threshold") — premature without a pilot consumer.
- Conformance test under `tests/conformance/` : disallowed by `tests/CLAUDE.md` ("Don't put conformance assertions inside tests/unit/" applies in reverse — these external numbers are not axioms / invariants and do not belong in `tests/conformance/`).

## Open questions (for the executor)

None blocking — all required values are sourced and verified against `references.bib:598-664`. If the executor discovers `paper1-fr/methodology.md` does not have a §6.6, the FR mirror commit (Task 7 Step 2) is conditionally skipped without blocking the PR (no §6.6 to mirror).
