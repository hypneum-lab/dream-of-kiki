# Coverage Targets — micro_kiki package

**Last measured:** 2026-05-10 (N6 Task 7, post-M2 split + N6.4 spike_loader extraction)
**N1 audit baseline:** 23% (monolithic `micro_kiki.py`, 1188 LOC, 23% line coverage)

## Current measurements

Measured via `coverage run --source=kiki_oniric/substrates/micro_kiki --branch -m pytest tests/unit/ -k "micro_kiki or substrate"` (65 tests; coverage 7.13.5 + pytest-cov 7.1.0 on Python 3.14 cannot run `--cov` flag with numpy 2.4 due to `del sys.modules[m]` triggering "cannot load module more than once" — workaround uses `coverage run` then `coverage report`).

| Module | Stmts | Coverage | Missing lines | N7 target | Rationale |
|--------|------:|---------:|---------------|----------:|-----------|
| `__init__.py`     |   8 | 100% | —                                              | 100% | Re-exports only, already saturated |
| `handlers.py`     |  70 |  98% | 164                                            |  99% | Op handlers = user-facing API; 1 branch left |
| `oplora.py`       |  26 | 100% | —                                              | 100% | Math is unit-testable; already saturated |
| `ties.py`         |  37 | 100% | —                                              | 100% | Pure numpy port (Yadav et al. 2306.01708); already saturated |
| `loaders.py`      |  32 |  39% | 89-111                                         |  60% | Env-gated (`DREAM_MICRO_KIKI_REAL=1`); hard to test without artefacts |
| `substrate.py`    | 109 |  84% | 261, 288-296, 317-322, 350-352, 432            |  92% | Class orchestration; snapshot/load round-trip + env-gating combos |
| `spike_loader.py` |  71 |  81% | 77, 102, 109-111, 160-166, 175->180, 184-185   |  90% | Backend-dependent; mockable via fixtures |
| **TOTAL**         | **353** | **86%** | — | **92%** | Up from N1 baseline 23% (+63 pts) |

## Notes

- **Mechanical win:** the M2 split (1 monolith → 6 focused modules, +1 from N6.4)
  yielded a **+63 pt** package-level coverage jump (23% → 86%) without writing any
  new test — purely from the test suite now reaching modules that existed but
  were previously bundled inside the unreachable monolith.
- The `DREAM_MICRO_KIKI_REAL=1` env-gated paths (`loaders.py` plus parts of
  `substrate.py` and `spike_loader.py`) require Studio access to MLX + real
  backend artefacts; coverage targets reflect dev-machine reachability, not
  full real-backend exercise.
- Re-measure after each N7 task to track progress against per-module targets.
- The package goal of **92%** assumes loaders.py stays under-tested without
  Studio fixtures; if Studio CI lands during N7, push the goal to 95%+.

## N7 candidates (priority order)

1. **substrate.py → 92%** (lowest hanging in highest-LOC module): add
   snapshot/`load_snapshot` round-trip tests, then `awake()` under each
   env-gating combination. ~12 missing lines.
2. **spike_loader.py → 90%**: backend-mock tests via fixture (avoid real
   spike-engine dependency). Lines 160-166 + 184-185 are error paths.
3. **handlers.py → 99%**: trivial — line 164 is a single conditional branch
   that needs a 1-line edge case test.
4. **loaders.py → 60%**: env-gated; needs either a Studio CI runner or
   fixture-based mocks of the real-backend artefact loader.
5. **oplora.py / ties.py / `__init__.py`**: already at 100%. Maintenance only.

## Note: loaders.py 39% is the legitimate env-gated floor

The `loaders.py` module's missing-coverage block (lines 89-111) is
exclusively the `DREAM_MICRO_KIKI_REAL=1` env-gated real-backend
loader. Exercising those lines requires Studio access (or
significant fixture mocking) — not reachable from CI / dev macOS.

The 60% target above was aspirational. Acknowledged: 39% is the
realistic ceiling without backend mock infrastructure. Raising
it requires either (a) a `[real-backend]` extras + mocks, or
(b) running tests against real Studio M3 Ultra (out of scope
for routine CI).

micro_kiki **package-level coverage 86%** remains the canonical
metric for downstream consumers; loaders.py contribution is
weighted accordingly.
