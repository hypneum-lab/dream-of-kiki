<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : Saillant, Clément
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# §5 Engineering Architecture (Paper 2, draft C2.14)

**Authorship byline** : *Saillant, Clément*
**License** : CC-BY-4.0

**Target length** : ~1.5 pages markdown (≈ 1300 words)

---

## 5.1 Two substrates, same Protocols

Paper 1 defined the framework surface as a set of typed Python
`Protocol`s : 8 primitives (α, β, γ, δ inputs + canal-1..4
outputs) and 4 operations (replay, downscale, restructure,
recombine). Paper 2 exercises that surface on two substrates :

- **`mlx_kiki_oniric`** (cycle 1 reference) — MLX arrays on
  Apple Silicon, gradient-style updates through dense tensors,
  source-of-truth for the 4 op handlers in
  `kiki_oniric/substrates/mlx_kiki_oniric.py`.
- **`esnn_thalamocortical`** (cycle 2 addition, synthetic
  substitute) — numpy LIF spike-rate skeleton in
  `kiki_oniric/substrates/esnn_thalamocortical.py`. The 4 op
  factories emit spike-rate state transitions rather than
  gradient updates. No Loihi-2 hardware is involved ; the
  skeleton exists to exercise the Protocol shape, not to
  claim neuromorphic-hardware efficiency.

The registration API
(`kiki_oniric/substrates/__init__.py`) is a single entry point
returning a named-tuple with op factories + state type ; the
ablation runner + conformance matrix consume the registration
uniformly, which is how DR-3 manifests operationally in the
codebase.

## 5.2 Three profiles : P_min, P_equ, P_max

Paper 1 defined the profile chain (DR-4 inclusion) as
`P_min ⊆ P_equ ⊆ P_max`. Cycle 2 wires all three end-to-end :

- **P_min** — replay only, canal-1 only. Smallest functional
  profile that still satisfies the axiom suite (DR-0
  accountability, R1 determinism). Spec :
  `kiki_oniric/profiles/p_min.py`. Cycle 1 already wired.
- **P_equ** — `{replay, downscale, restructure}` on canals
  1 + 2 + 3. The *equi*-profile matching the four-pillar
  consolidation narrative (Walker / Tononi / Friston), minus
  the Hobson recombination arm. Spec :
  `kiki_oniric/profiles/p_equ.py`. Cycle 1 already wired.
- **P_max** — 4 ops (+ `recombine`) on 4 channels (+ α-stream
  input + canal-4 ATTENTION_PRIOR output). Previously
  skeleton-only at end of cycle 1 (§ Paper 1 §9.2) ; **fully
  wired in cycle 2** per G8 (`docs/milestones/g8-p-max-
  functional.md`). With P_max real-wired, hypothesis H2
  (P_max equivalence vs P_equ within ±5%) becomes a real
  comparison rather than a self-equivalence smoke test.

DR-4 profile chain inclusion is enforced structurally : P_equ's
op set is a superset of P_min's, P_max's is a superset of
P_equ's, and the same inclusion holds at the channel level.
Violations would be caught by the DR-4 axiom suite
(`tests/conformance/axioms/test_dr4_*`).

## 5.3 Four canonical operations on a free semigroup

The 4 operations are, per DR-2 (compositionality,
`docs/proofs/dr2-compositionality.md`), generators of a **free
semigroup** of dream operation sequences :

- **`replay`** — Walker / Stickgold consolidation pillar.
  Sample β-buffer episodes, forward through current
  parameters, apply gradient-style retention updates. On
  MLX : dense tensor ops. On E-SNN : spike-rate re-injection
  over the LIF population.
- **`downscale`** — Tononi SHY pillar. Multiplicative shrinkage
  of weights in (0, 1]. Commutative but **not** idempotent :
  `downscale_f ∘ downscale_f` gives factor², not factor.
- **`restructure`** — Friston FEP pillar. Topology modification
  (add / remove layer, reroute) under the S3 topology guard.
  Any emission must pass `validate_topology` before swap.
- **`recombine`** — Hobson / Solms pillar. Generative
  re-sampling from the δ snapshot. On MLX : VAE-light
  (cycle 1 skeleton) then full VAE with KL upgrade (cycle 2
  `9906520`). On E-SNN : interpolation over latent spike-rate
  codes. The recombine emission targets canal-2
  RECOMBINED_LATENTS (and canal-4 ATTENTION_PRIOR in P_max).

Of the 16 `(op_i, op_j)` cross-pairs, 12 are non-commutative
(`docs/proofs/op-pair-analysis.md`) — meaning order matters
and the canonical ordering (replay → downscale → restructure ;
recombine in parallel) is a load-bearing design choice, not
incidental.

## 5.4 Eight primitives : 4 inputs + 4 output channels

The framework boundary is typed at 8 points :

- **Inputs (awake → dream)** :
  - α — raw traces stream (P_max only in cycle 2 : ring
    buffer 1024, FIFO, wired `8ee452b`).
  - β — episodic buffer (all profiles).
  - γ — semantic snapshot (P_equ + P_max).
  - δ — latent snapshot (P_max for recombine).
- **Outputs (dream → awake)** :
  - canal-1 UPDATED_WEIGHTS (all profiles).
  - canal-2 RECOMBINED_LATENTS (P_equ + P_max).
  - canal-3 RESTRUCTURED_TOPOLOGY (P_equ + P_max).
  - canal-4 ATTENTION_PRIOR (P_max only ; S4 guard ≤ 1.5 via
    `63af87d`).

Each output channel is gated by the **swap protocol** :

1. Compute the candidate emission under the current dream
   state.
2. Apply S1 (retained non-regression — candidate must not
   regress on the retained benchmark under the predictor's
   evaluation).
3. Apply S2 (finite — no NaN / Inf on the candidate).
4. Apply S3 (topology — ortho species connectivity, no self-
   loops, layer count bounds).
5. Apply S4 (attention_budget ≤ 1.5) on canal-4.
6. If all guards PASS, commit the swap and register the
   artifact ; else, abort with the guard's refusal exception.

The swap protocol is substrate-agnostic : both MLX arrays and
LIFState pass through the same guard sequence.

## 5.5 The async dream worker (C2.17)

Cycle 1 shipped the dream worker as a **Future-API skeleton**
— the worker interface existed but the execution was
sequential. Cycle 2 C2.17 (`018fd05`) lands the **real async
worker** (`asyncio`-based, concurrent execution, no longer a
skeleton). The worker :

- Accepts `DreamEpisode` 5-tuples via a queue.
- Dispatches the episode's op sequence through the registered
  substrate's op factories.
- Awaits guard application + swap commit asynchronously.
- Registers the run in the run-registry under the R1 key.

Concurrency buys two engineering properties Paper 2 cares
about : (i) the cross-substrate runner can issue MLX + E-SNN
cells in parallel at the orchestration layer (even though the
per-cell inference is still shared-predictor synthetic), and
(ii) the architecture is ready for cycle-3 substrate-specific
predictors where per-cell inference cost diverges.

## 5.6 Run registry + R1 determinism

The run registry (`harness/storage/run_registry.py`) enforces
the **R1 reproducibility contract** : every run is keyed by a
**32-hex SHA-256 prefix** of `(c_version, profile, seed,
commit_sha)`. Re-running the same key must produce an identical
`run_id`. The width was bumped from 16 → 32 hex in cycle 1
commit `df731b0` after a code-review flag on 64-bit collision
risk at scale.

Every artifact reported in §7 resolves to a registered run_id :

- `ablation_runner_run_id` : `45eccc12953e758440fca182244ddba2`
  (multi-substrate runner entry).
- `cycle2_batch_id` : `3a94254190224ca82c70586e1f00d845`
  (the cycle-2 batch wrapper).
- `harness_version` : `C-v0.6.0+PARTIAL`.

Artifacts that do not resolve to a run_id or a proof file are
not reported in Paper 2. This is the strong form of the
reproducibility contract : the paper's tables and the repo's
artifacts must round-trip.

## 5.7 DualVer lineage : `C-v0.6.0+PARTIAL`

Paper 1 shipped under `C-v0.5.0+STABLE`. Cycle 2 bumps the
**formal axis** to `0.6` (E-SNN Conformance extension — a
formal extension of the framework surface, not a breaking
change, hence MINOR) and leaves the **empirical axis** at
`+PARTIAL` (the cross-substrate ablation is synthetic
substitute by scope, not stable empirical evidence — hence
the `PARTIAL` qualifier per §12 DualVer rules).

The tag `C-v0.6.0+PARTIAL` is the honest label for Paper 2 :
engineering-complete on 2 substrates + 3 profiles + 4 ops +
4 channels, empirical-partial because the cross-substrate
rows share a predictor. A `+STABLE` qualifier requires cycle 3
real-predictor evidence.

## 5.8 Wake-Sleep CL baseline row (Alfarano 2024)

Paper 2 §7 includes a fourth row in the ablation table : the
**Wake-Sleep Consolidated Learning** baseline of Alfarano et al.
2024 [@alfarano2024wakesleep, IEEE TNNLS, arXiv 2401.08623].
Paper 1 §3 already names this work as the "closest published
NREM/REM dual-phase analog" and the natural Paper 2 ablation
comparator. The baseline is registered in the substrates
namespace via
`kiki_oniric/substrates/wake_sleep_cl_baseline.py` for
discoverability, but it is **not DR-3 conformant** — it does not
implement the 4 dream operations. Its single comparator API,
`WakeSleepCLBaseline.evaluate_continual(seed, task_split)`,
returns the two M1.* metrics (`forgetting_rate`, `avg_accuracy`)
on Split-FMNIST 5-task class-incremental — the same dataset
shape already used in `experiments/h1_split_mnist/`. The
eval-matrix schema gains a top-level `baselines:` block
(FC-MINOR additive addition,
`C-v0.11.0+PARTIAL → C-v0.12.0+PARTIAL`) registering the
bibkey, arXiv id, variant, and the metric IDs scored. The
baseline row is dumped to
`docs/milestones/wake-sleep-baseline-2026-05-03.{md,json}` and
each cell carries an R1 `run_id`. **Variant : c** (published
reference values from Alfarano 2024 Tables 2-3, frozen ;
Paper 2 §6.4 caveat applies).

---

## Notes for revision

- Consider a Figure 1 : pipeline diagram (α/β/γ/δ inputs →
  4 ops under guards S1..S4 → canals 1-4).
- Cross-reference §4 Conformance once both sections are in
  the full draft.
- Tighten to ≤ 1200 words at NeurIPS pre-submission pass.
