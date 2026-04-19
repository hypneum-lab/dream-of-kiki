<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : dreamOfkiki project contributors
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# §4 Conformance Criterion in Practice (Paper 2, draft C2.14)

**Authorship byline** : *dreamOfkiki project contributors*
**License** : CC-BY-4.0

**Target length** : ~1.5 pages markdown (≈ 1300 words)

---

## 4.1 The three DR-3 conditions, recapped

Framework C §6.2 (`docs/specs/2026-04-17-dreamofkiki-framework-
C-design.md`) defines the **Conformance Criterion** as three
independently checkable conditions that a candidate substrate
must satisfy to inherit the framework's guarantees :

- **C1 — signature typing (typed Protocols).** The 8 primitives
  (α, β, γ, δ inputs + canal-1..4 outputs) are declared as
  typed Python `Protocol`s. A substrate conforms to C1 by
  exposing handlers / factories whose signatures are
  assignable to those Protocols. The registry of operations
  (`kiki_oniric.dream.registry`) must be complete for the
  profiles the substrate exercises.
- **C2 — axiom property tests pass.** DR-0..DR-4 have property-
  based test suites in `tests/conformance/axioms/`. Each suite
  is parameterized on the substrate's state type. A substrate
  conforms to C2 by having every applicable axiom suite green
  under its native state representation.
- **C3 — BLOCKING invariants enforceable.** The S-family guards
  (S1 retained non-regression, S2 finite, S3 topology, S4
  attention_budget) must be callable on the substrate's state
  and must *refuse* ill-formed values by raising the documented
  exception class. C3 is verified by negative tests in
  `tests/conformance/invariants/` that wire deliberately
  malformed states through the guards.

DR-3 is **existential, not universal** : it does not claim
every substrate is conformant ; it claims every *conformant*
substrate inherits the framework's guarantees. Paper 1 exhibited
one conformant substrate (MLX) ; Paper 2 exhibits a second
(E-SNN skeleton), and reserves a third row for cycle 3.

## 4.2 Substrate 1 — `mlx_kiki_oniric` (cycle 1 reference)

The MLX reference substrate is the cycle-1 canonical
implementation. Every cell below is backed by a concrete test
artifact in-tree :

| Condition | Verdict | Evidence |
|-----------|---------|----------|
| C1 signature typing | PASS | all 8 primitives declared as Protocols + registry complete (`tests/conformance/axioms/test_dr3_substrate.py`) |
| C2 axiom property tests | PASS | DR-0, DR-1, DR-3, DR-4 suites green on MLX (`tests/conformance/axioms/`) |
| C3 BLOCKING invariants | PASS | S2 finite + S3 topology guards enforceable on MLX arrays (`tests/conformance/invariants/`) |

The MLX row carries no synthetic-substitute flag at the
*conformance* level : the three conditions concern the
framework surface (typing, axioms, guards), not the data fed
through it. The synthetic-substitute caveat for MLX kicks in
only when MLX is used as the **predictor** in §7 (where the
predictor is shared across substrate labels).

## 4.3 Substrate 2 — `esnn_thalamocortical` (cycle 2 addition, synthetic substitute)

The E-SNN substrate is a **numpy LIF spike-rate skeleton**. It
exposes the same 4 operation factories (replay, downscale,
restructure, recombine) and consumes the same 8 primitives as
MLX, but its internal state is a leaky-integrate-and-fire
population representation (`LIFState`) whose evolution is
simulated numerically — not deployed on Loihi-2 hardware.

**(synthetic substitute — no Loihi-2 HW.)**  The C2 and C3
rows below carry the explicit synthetic-substitute flag
inherited from `docs/proofs/dr3-substrate-evidence.md` :

| Condition | Verdict | Evidence |
|-----------|---------|----------|
| C1 signature typing | PASS | 4 op factories callable + core registry shared with MLX (spike-rate numpy LIF skeleton, synthetic substitute) (`tests/conformance/axioms/test_dr3_esnn_substrate.py`) |
| C2 axiom property tests | PASS *(synthetic substitute — no Loihi-2 HW)* | DR-3 E-SNN conformance suite passes on numpy LIF skeleton (`tests/conformance/axioms/test_dr3_esnn_substrate.py`) |
| C3 BLOCKING invariants | PASS *(synthetic substitute — spike-rate numpy LIF)* | S2 finite + S3 topology guards enforceable on LIFState (`tests/conformance/axioms/test_dr3_esnn_substrate.py`) |

The E-SNN row is the **key new artifact of cycle 2** for DR-3.
Passing C1..C3 on a structurally independent implementation —
spike-rate dynamics rather than gradient updates on dense
matrices — is the architectural evidence the framework's
typing + axiom + guard surfaces do not secretly depend on MLX
internals. A real Loihi-2 mapping would strengthen this to a
neuromorphic hardware claim ; until then, the spike-rate
skeleton is a **structural second implementation** adequate for
validating the Conformance Criterion shape, not its hardware
consequences.

## 4.4 A third row : `hypothetical_cycle3` (placeholder, not evidence)

The conformance matrix reserves a third row for a future
(cycle-3) substrate :

| Condition | Verdict | Evidence |
|-----------|---------|----------|
| C1 signature typing | N/A | not yet implemented |
| C2 axiom property tests | N/A | not yet implemented |
| C3 BLOCKING invariants | N/A | not yet implemented |

The row is explicitly marked `N/A` and **must not be read as
passing, failing, or even testable**. Its purpose is twofold :
(i) keep the matrix shape stable for cycle 3 so diffs in the
cycle-3 closeout report are minimal, (ii) visually remind
readers that DR-3 evidence is a *collection*, not a binary ;
two substrates is a beginning, not a ceiling. Candidate
cycle-3 substrates under consideration include transformer-
based instances, SpiNNaker / Norse mappings, or a real
Loihi-2 deployment if the Intel NRC partnership materializes
(`docs/milestones/g9-cycle2-publication.md` § cycle-3 agenda).

## 4.5 The matrix as a reusable artifact

The conformance matrix is not a one-shot claim ; it is a
regenerable artifact. Running :

```bash
uv run python scripts/conformance_matrix.py
```

re-derives the matrix from the test suite and writes both the
Markdown (`docs/milestones/conformance-matrix.md`) and JSON
(`docs/milestones/conformance-matrix.json`) dumps. The JSON is
structured for downstream automation : a future
`test_conformance_matrix_regression` could diff the JSON against
a committed snapshot to catch silent regressions.

The supporting test suite (`tests/conformance/`) is
substrate-parameterized. A reviewer or a user with a new
candidate substrate adds one registration line
(`kiki_oniric/substrates/__init__.py`), writes the 4 op
factories + state type, and the entire C1..C3 battery runs
against it automatically. This is the operational promise of
DR-3 : **a substrate-agnostic framework must ship a
substrate-agnostic test suite**, not a monolithic per-substrate
battery rewritten each time.

## 4.6 What Conformance does and does not certify

Two readings must be distinguished :

**What Conformance does certify (C1..C3 all PASS) :**

- The substrate's typed surface is framework-compatible.
- The substrate's state representation satisfies the
  axiom property tests DR-0..DR-4 relevant to its profile.
- The substrate's state can be *refused* by BLOCKING guards
  when malformed.
- The framework's core promise — that two compliant
  substrates are interchangeable at the Protocol level — is
  held operationally by at least these two substrates.

**What Conformance does NOT certify :**

- Any **empirical performance claim** on real data. The cross-
  substrate results in §7 are synthetic substitute ; C1..C3
  is an architectural property, not a data-efficacy property.
- Any **hardware energy or latency claim**. E-SNN on Loihi-2
  would be a hardware claim ; E-SNN on numpy is an
  architectural claim. The two are distinct and Paper 2 makes
  only the second.
- **Biological fidelity** of the E-SNN LIF parameters. Pillar
  mappings (Paper 1 §3) are inherited from Paper 1 ; Paper 2
  does not re-argue them.

Cross-references : `docs/proofs/dr3-substrate-evidence.md` is
the per-substrate evidence record ;
`docs/milestones/conformance-matrix.md` is the regenerable
matrix ; `docs/milestones/g7-esnn-conformance.md` is the G7
gate report that locked the E-SNN Conformance verdict.

---

## Notes for revision

- Insert the matrix as a rendered table figure once NeurIPS
  style file is chosen.
- Cross-reference §5 architecture (which describes the ops
  behind the registry) once §5 lands.
- Tighten to ≤ 1200 words at NeurIPS pre-submission pass.
