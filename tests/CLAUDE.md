# tests — unit + conformance suites

Two flavours, treated differently :

- `unit/` — ordinary pytest coverage of modules in `harness/` and
  `kiki_oniric/`. Fast, deterministic, runs on every commit.
- `conformance/` — **normative tests**. They enforce the framework
  C axioms and invariants and are cited in the paper.

## Conformance structure

- `conformance/axioms/test_dr0_accountability.py` — DR-0 every
  state mutation is accountable.
- `conformance/axioms/test_dr1_episodic_conservation.py` — DR-1
  episodic information conservation.
- `conformance/axioms/test_dr3_substrate.py` — DR-3 Conformance
  Criterion (typed Protocols, 3 conditions).
- `conformance/invariants/test_s2_finite.py` — S2 all activations
  finite.
- `conformance/invariants/test_s3_topology.py` — S3 topology valid
  after `HierarchyChangeChannel.apply_diff`.
- `conformance/invariants/test_s4_attention.py` — S4 attention
  budget does not exceed K1 compute cap.
- `conformance/invariants/test_k2_coupling.py` — K2 coupling
  coefficient in invariant band [0.27, 0.39].

Each conformance test's docstring must cite the axiom / invariant
ID and reference the relevant spec section.

## Anti-patterns specific to tests

- **Don't** weaken a conformance test to make CI green. A failing
  conformance test is an empirical-axis (EC) signal → propose
  `+UNSTABLE` in DualVer, don't silence the test.
- **Don't** put conformance assertions inside `tests/unit/` — they
  belong under `tests/conformance/` so they are surfaced in
  `STATUS.md` reports.
- **Don't** use nondeterministic `random` in tests ; seed MLX /
  numpy / hypothesis explicitly. CI coverage gate is ≥ 90 %.
- **Don't** add a new axiom test without adding the axiom proof
  stub under `docs/proofs/` and the invariant declaration under
  `docs/invariants/`.
- **Don't** skip (`pytest.mark.skip`) a conformance test — mark it
  `xfail` with the axiom ID and open a CHANGELOG entry.
- **Don't** rely on filesystem state between tests ; use `tmp_path`
  fixtures so the run registry stays isolated.
