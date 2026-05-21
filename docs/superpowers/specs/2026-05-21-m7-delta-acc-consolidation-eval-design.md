# M7 — `delta_acc` consolidation→eval wiring (design)

**Date:** 2026-05-21
**Substrate:** `mlx_latent_diffusion` (Wave 3b, Track S)
**DualVer:** folds into substrate-internal `C-v0.15.0+PARTIAL`
(unreleased — not a new bump). Project axis stays `C-v0.25.0+PARTIAL`.
**Branch:** `feat/m7-substrate-dr3`

## Problem

In the M7 substrate, `execute_profile` reports `delta_acc` as a
per-cell continual-learning metric. The intent (per
`_diffusion/cl_eval_head.py`) is
`delta_acc = accuracy AFTER the dream cycle − accuracy BEFORE`.

The current implementation makes `delta_acc ≡ 0` **by construction**:

- the `ClEvalHead` is trained once on `dataset[0]` (raw encoder
  latents) and then frozen;
- `baseline_acc = eval(head, dataset[0])`;
- the dream cycle runs — it mutates the denoiser, never the head and
  never the latents;
- `post_acc = eval(head, dataset[0])` — the *same* head on the *same*
  latents → `post_acc ≡ baseline_acc` → `delta_acc ≡ 0`.

A 450-cell bench run confirmed every fresh M7 cell reports
`delta_acc = 0.0`. The consolidation target (denoiser) and the eval
path (encoder latents → frozen head) are disconnected subsystems.

Consequences this design corrects:

- the CHANGELOG claim "`delta_acc` is now a real CL head measurement"
  is an overclaim — the head is real but the *delta* is vacuous;
- `tests/unit/test_mlx_latent_diffusion_adapter.py::test_execute_profile_delta_acc_is_order_independent`
  passes degenerately (`0 == 0`).

## Decision summary

Two semantic decisions were taken during brainstorming:

1. **B5 awake/dream loop.** The dream episodes emit channel outputs;
   `apply_channel_outputs` applies them; the head probes the model
   before vs after. This is the framework-C canonical mechanism
   (the B0–B5 channel machinery exists for exactly this).
2. **The denoiser is the continual learner.** `delta_acc` measures
   the effect of the dream cycle on the *denoiser's* representation.
   The CL head probes a denoiser-derived feature; channel outputs
   apply to the denoiser.

Implementation approach: **Approach A — diffusion-native weight
channel.** A new `DenoiserWeightDeltaChannel` consumes `WeightUpdate`
outputs and applies the per-layer dense delta to the plain
`MLPDenoiser`. This reuses `apply_channel_outputs` and the
`WeightUpdate` type unchanged, stays substrate-local, and does not
touch the LoRA channel machinery.

Scope: **WeightUpdate-only.** Only the parametric-consolidation
channel (`replay` + `downscale` → `WeightUpdate`) is routed through
`apply_channel_outputs` and drives `delta_acc`. `restructure`
(`TopologyDiff`) and `recombine` (`LatentSample`) keep their current
in-episode behaviour and are *not* emitted as channel outputs at M7.
Routing those through diffusion-native channels is a follow-up.

## Architecture & data flow

New `delta_acc` flow inside `execute_profile`:

1. Train the `ClEvalHead` once on a **denoiser-derived feature**
   `feat(z) = denoiser(z, t_fixed)` over `dataset[0]`, where
   `t_fixed = t_steps // 2` is a deterministic timestep (R1
   stability).
2. `baseline_acc = eval(head, feat(dataset[0]))` — denoiser in its
   pre-dream state.
3. The dream episodes run. The bound `_real.py` handlers for
   `replay` / `downscale` emit `WeightUpdate` outputs into
   `runtime.log`.
4. `apply_channel_outputs(runtime.log, weight_channel=DenoiserWeightDeltaChannel(denoiser))`
   applies the per-layer deltas to the denoiser's weights;
   `runtime.reset_log()` follows.
5. `post_acc = eval(head, feat(dataset[0]))` — the *same* frozen
   head, `feat` recomputed against the now-consolidated denoiser.
6. `delta_acc = post_acc − baseline_acc`.

Invariant: the head is frozen between steps 2 and 5, so
`baseline_acc, post_acc ∈ [0, 1]` and `delta_acc ∈ [−1, 1]` — the
out-of-range values seen in the pre-M7 bench cannot recur. `delta_acc`
is non-zero iff the dream cycle's `WeightUpdate`s changed the
denoiser's feature geometry.

## Components

### 1. `DenoiserWeightDeltaChannel`

New file `kiki_oniric/substrates/_diffusion/denoiser_weight_channel.py`.

- Implements the `WeightDeltaChannel` protocol: `.apply(lora_delta,
  fisher_bump)`.
- Holds a reference to the `MLPDenoiser`.
- `apply`: for each `(layer_name, delta)` in `lora_delta` (a
  `dict[str, NDArray[float32]]` — the `lora_` prefix is a legacy
  misnomer; the value is a dense per-layer delta), locate the
  matching denoiser parameter and add `delta` in place. S2
  finiteness on the input is already enforced by
  `WeightUpdate.__post_init__`; the channel re-checks the post-apply
  parameter is finite (S2 guard with the invariant ID in the
  message).
- `fisher_bump` is recorded for traceability but does not gate the
  apply at M7.
- The layer-key convention must match whatever the bound `_real.py`
  handlers emit (they introspect `model.layers` via the
  `_DenoiserSingleArgAdapter`); the implementation plan pins the
  exact key mapping.
- Re-exported from `kiki_oniric/substrates/_diffusion/__init__.py`.

### 2. CL head re-wire

`kiki_oniric/substrates/_diffusion/cl_eval_head.py` gains a
`denoiser_feature(denoiser, z, t_fixed)` helper returning
`denoiser(z, t_fixed)` (the predicted-noise output, `d_latent`-dim).
`execute_profile` trains and evaluates the head on
`denoiser_feature(dataset[0], ...)` instead of the raw latents, both
for `baseline_acc` and `post_acc`.

### 3. Episode channel emission + apply

In `execute_profile`:

- ensure `bind_real_handlers` (in `dream_ops_adapter.py`) binds the
  **emitting** `_real.py` handler variants (signature
  `-> WeightUpdate | None`) for `replay` and `downscale`;
- populate the `DreamEpisode.output_channels` field (currently `()`)
  to declare the `WeightUpdate` contract;
- after the episode loop, call `apply_channel_outputs` with the
  `DenoiserWeightDeltaChannel`, then `runtime.reset_log()`.

`restructure` and `recombine` handlers are bound as today and do not
emit channel outputs (WeightUpdate-only scope) — so
`apply_channel_outputs` is never handed a `TopologyDiff` /
`LatentSample` it has no channel for.

## DualVer & honesty

- The `delta_acc` work folds into the unreleased substrate-internal
  `C-v0.15.0+PARTIAL`; no new bump. `C-v0.15.0` now means "M7
  substrate: DR-3 conformance **+ real `delta_acc`**".
- The project axis stays `C-v0.25.0+PARTIAL`.
- The CHANGELOG `C-v0.25.0` entry is corrected: the `delta_acc`
  description now matches reality (denoiser-feature probe + B5
  `apply_channel_outputs` loop). No overclaim.
- `+PARTIAL` remains on the substrate for its other reasons
  (Canal 4 documented no-op, DR-3 evidence 7/8 typed) — `delta_acc`
  is no longer one of them.

## Testing

- `tests/unit/test_denoiser_weight_channel.py` — `apply` mutates the
  denoiser parameters; S2 finiteness holds on the result.
- `delta_acc ≠ 0` regression test — for a cell whose dream cycle
  emits non-trivial `WeightUpdate`s, `delta_acc` is non-zero and
  within `[−1, 1]`. Replaces the degenerate test.
- The order-independence test is kept and strengthened: it asserts
  `delta_acc` is both identical across call orders *and* non-zero,
  so a future regression back to the structural zero is caught.
- DR-3 conformance (`test_dr3_diffusion_profile.py`) is unaffected —
  the per-profile activation set does not change.
- R1: the 3 diffusion golden-hash entries change (the denoiser is
  now mutated by `apply_channel_outputs`, and the head probes the
  denoiser). Rebaseline the 3 entries on apple_m5 and append a
  `REBASELINE_NOTE.md` entry; m3_ultra / m1_max stay `pending`.

## Associated cleanup (bench-discovered defects)

Folded into the same work because the bench run surfaced them:

- `harness/diffusion_eval/milestone.py:40` hardcodes
  `"c_version": "C-v0.14.0+PARTIAL"` — change it to derive the
  `c_version` from the aggregated cells (or `HARNESS_VERSION`).
- `.run_registry.sqlite` carries 25 stale `C-v0.14.0` cells and 15
  duplicate `C-v0.15.0` cells from earlier partial runs; purge the
  stale entries before re-running.
- After the wiring lands, regenerate `docs/milestones/wave3b-bench-pending.*`
  cleanly (450 cells, real non-zero `delta_acc`, real `commit_sha`,
  `c_version C-v0.15.0`).

## Out of scope

- Routing `restructure` (`TopologyDiff`) and `recombine`
  (`LatentSample`) through diffusion-native channels — follow-up.
- LoRA-wrapping the denoiser (Approach B) — not pursued.
- Any change to the encoder, sampler, or trainer.
