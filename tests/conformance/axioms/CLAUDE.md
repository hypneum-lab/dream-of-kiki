# tests/conformance/axioms/ — DR-0..DR-4 conformance suites

**Load-bearing.** These tests are the executable counterpart of the
Conformance Criterion (framework spec §6.2). A failing axiom test is
not a CI nuisance — it is an empirical-axis (EC) signal that may
block a paper submission and trigger an OSF amendment.

Scope of this dir is narrower than the parent `tests/CLAUDE.md`: it
covers **only** axiom property tests. Invariants live in
`../invariants/`; integration suites (HMM, SNN evidence) live one
level up under `../`.

## File-to-axiom map

| File | Axiom | Substrate |
|------|-------|-----------|
| `test_dr0_accountability.py` | DR-0 every DE logged + finite budget | hypothesis property |
| `test_dr0_diffusion_de_budget.py` | DR-0 diffusion budget accountability | MLX diffusion substrate |
| `test_dr1_episodic_conservation.py` | DR-1 β record eventually consumed | hypothesis property (S5.3 fake buffer; real β lands S7+) |
| `test_dr1_diffusion_finite.py` | DR-1 diffusion finite episodic conservation | MLX diffusion substrate |
| `test_dr2_compositionality_empirical.py` | DR-2 (weakened, 2026-04-21) closure / additivity / effect-chaining under precondition | MLX kiki-oniric, 24 permutations, 12 PASS / 12 xfail |
| `test_dr2_prime_canonical_order.py` | DR-2' fallback (strict canonical order) | MLX, byte-identical determinism |
| `test_dr3_substrate.py` | DR-3 condition (1) — 8 typed Protocols | structural |
| `test_dr3_diffusion_profile.py` | DR-3 diffusion profile conformance | MLX diffusion substrate |
| `test_dr3_diffusion_substrate.py` | DR-3 diffusion substrate conditions (1)-(3) | MLX diffusion substrate |
| `test_dr3_esnn_substrate.py` | DR-3 conditions (1)-(3) on E-SNN | numpy LIF skeleton (synthetic substitute) |
| `test_dr3_esnn_norse_substrate.py` | DR-3 conditions (1)-(3) on Norse SNN | numpy-fallback path (no torch/norse required) |
| `test_dr3_g4_deeper_substrate.py` | DR-3 G4 deeper substrate conditions (1)-(3) | MLX deeper G4 substrate |
| `test_dr3_g4_hierarchical_substrate.py` | DR-3 G4 hierarchical substrate conditions (1)-(3) | MLX hierarchical G4 substrate |
| `test_dr3_micro_kiki_substrate.py` | DR-3 conditions (1)-(3) on MLX | MLX kiki-oniric |
| `test_dr3_ttt_substrate.py` | DR-3 TTT substrate conditions (1)-(3) | MLX TTT substrate |
| `test_dr4_profile_inclusion.py` | DR-4 ops/channels chain inclusion | P_min ⊆ P_equ ⊆ P_max metadata |
| `_dsl.py` | shared DSL — `seeded_runtime`, `make_episode`, `assert_states_equal`, `registered_ops`, `profile_channels` | helpers (not a test) |

## Conventions

- **Real substrate**, not mocks. DR-3 substrate tests instantiate
  the actual MLX runtime / E-SNN backend. The S5.3 `FakeBetaBuffer`
  in `test_dr1_*.py` is the single tolerated exception, scoped to
  the skeleton phase and pinned to "real β implementation lands
  S7+" in the docstring.
- **Cite the axiom ID** in module docstring + `Reference:` line
  pointing at framework-C spec §6.2. Add `docs/proofs/<file>.md`
  pointer when a proof exists.
- **Hypothesis seeding** — `@settings(deadline=None)` for the
  property tests; explicit seeds (`seed=7`) for determinism tests.
- **Falsification stance**. `test_dr2_compositionality_empirical.py`
  encodes Popperian xfail witnesses (12 falsified permutations) —
  this is intentional: silencing them would erase the empirical
  basis for the DR-2 v0.2 weakening.

## Coupling

- DR-N test fails → STATUS.md gate bumps to `+UNSTABLE` on EC axis
  → may block Paper 1 submission and require an OSF amendment
  (cf. `osf-amendment-bonferroni-cycle3.md` template).
- New axiom test → must ship with `docs/proofs/<dr>-*.md` proof
  stub + `docs/invariants/registry.md` entry (parent rule).
- Axiom statement edit (in spec §6.2) → FC bump + update test
  docstring + amendment under `docs/specs/amendments/`.

## Anti-patterns

- Skipping (`pytest.mark.skip`) a DR-N test to make CI green.
  Use `xfail(strict=True)` with axiom ID + open a CHANGELOG entry
  + propose `+UNSTABLE` in STATUS.md. A skipped axiom invalidates
  the Conformance Criterion for the substrate.
- Mocking the substrate inside an axiom test (e.g. swapping
  `DreamRuntime` for a stub). DR-3 evidence collapses to vacuous.
  S5.3 `FakeBetaBuffer` is the only sanctioned skeleton fake.
- Editing the axiom statement (closure / additivity / inclusion
  formula) inside a test docstring without bumping FC and
  appending a `docs/specs/amendments/` entry.
- Adding a new DR-N test without (a) the proof stub under
  `docs/proofs/`, (b) the invariant declaration under
  `docs/invariants/registry.md`, (c) a parent-spec §6.2 statement.
- Coupling axiom tests to filesystem state across runs — use
  `tmp_path` fixtures so the run registry stays isolated (parent
  rule, repeated here because conformance failures here are far
  more expensive than in `tests/unit/`).
- Renaming a primitive method or `Operation` enum value while
  touching only the test — these names are part of the DR-3
  signature contract and require a coordinated FC bump in
  `kiki_oniric/core/primitives.py`.
