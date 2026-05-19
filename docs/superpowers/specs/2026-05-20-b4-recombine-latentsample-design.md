# B4 — `recombine` emits a `LatentSample`

**Date** : 2026-05-20
**Status** : design approved, pending spec review
**Tracking issue** : #15 (`hypneum-lab/dream-of-kiki`)
**Scope** : sub-project B4 of the issue-#15 "approach B" decomposition

---

## Context

Issue #15 / approach B makes the four dream operations produce real
channel outputs. Prior on `main`:

- **B0** (`C-v0.14.0`) — channel-output contract, including
  `LatentSample(species: str, latent_vector: NDArray[np.float32],
  provenance: str)` with a `__post_init__` finite check on
  `latent_vector`.
- **B1a / B1b** (`C-v0.15.0` / `C-v0.16.0`) — LoRA model and the
  `replay` op emitting a real `WeightUpdate` (channel 1).
- **B2** (`C-v0.17.0`) — `downscale` emitting a `WeightUpdate`.
- **B3** (`C-v0.18.0`) — `restructure` emitting a `TopologyDiff`
  (channel 3).

**B4** makes `recombine` the fourth and final emitting op, this
time on **channel 2** (`LatentSample`). Framework-C §4.2:

> **recombine** — Hobson VAE light source (creative branch).
> Source: C-Hobson VAE.
> Input: latent buffer. **Output: LatentSample (channel 2).**

## Problem

`recombine_real_handler` already exists with a full VAE
reparameterization pipeline: it takes an `encoder` / `decoder`
pair, reads `delta_latents` from the episode, samples `z = mu +
sigma * epsilon` using an *isolated* `mx.random.key` (the
process-wide MLX RNG is never touched), decodes `z`, and stores
the decoded sample in `state.last_sample`. It returns `None`.

B4 makes the handler return a `LatentSample` whose
`latent_vector` is the sampled latent `z`. No new factory; the
machinery is already correct. The patch is minimal.

## Approaches considered

**Substrate target.** Three options were considered:

1. **Keep the existing `encoder` / `decoder` external pair.**
   Reuses the cycle-3 VAE machinery as-is. Semantically faithful
   to Hobson — generative diversity comes from a free-standing
   generator, not from a parametric consolidation. Diverges
   from B1b/B2/B3's `LoRAModel` substrate, but the Hobson VAE
   simply has no use for LoRA adapters. **Chosen.**
2. New `recombine_lora_handler(state, *, model: LoRAModel, seed)`.
   Symmetric with B1b/B2/B3 but semantically forced — LoRAModel
   is not a VAE. Rejected.
3. Both (patch the existing + add a LoRA variant). Double scope
   for no design payoff. Rejected.

**`latent_vector` content.** Three options were considered:

1. **`z` — the sampled latent itself.** Semantically aligned
   with the channel name `LATENT_SAMPLE` and with invariant I3
   ("latent distributional drift bounded"), which describes the
   latent's statistics. `state.last_sample` continues to hold
   the decoded output for backwards compatibility. **Chosen.**
2. The decoded output. Mismatches the channel name and I3.
   Rejected.
3. `z` with the decoded output encoded into `provenance`. Pushes
   binary data through a string field; complicates the contract.
   Rejected.

**`species` and `provenance`.** Three options were considered:

1. **`species` via `input_slice` (default `"default"`),
   `provenance` auto-derived from `(episode_id, episode_count,
   key_seed)`.** Caller controls categorical tagging; the trace
   string is fully reconstructible and R1-friendly. **Chosen.**
2. Both auto-derived. Loses categorical flexibility. Rejected.
3. Both required from `input_slice`. Adds API noise and breaks
   the auto-trace R1 chain. Rejected.

## Design

### Patch to `recombine_real_handler` in `kiki_oniric/dream/operations/recombine_real.py`

Signature change:

```python
def recombine_real_handler(
    state: RecombineRealState,
    *,
    encoder,
    decoder,
    seed: int,
) -> Callable[[DreamEpisode], "LatentSample | None"]:
```

The return type widens from `Callable[[DE], None]` to
`Callable[[DE], LatentSample | None]`. The handler body keeps
its VAE step intact and adds the `LatentSample` construction.

### Handler behaviour (only the new lines flagged)

For each `DreamEpisode`:

1. Read `latents = episode.input_slice.get("delta_latents", [])`.
   Empty → `ValueError("I3: ...")` (existing behaviour preserved
   — see §"Asymmetry note" below).
2. Compute `key_seed = seed + state._episode_count`, then `key`
   / `sample_key` via `mx.random.split` (existing).
3. `x = mx.array(latents[0])`; `mu, log_var = encoder(x)`;
   `sigma = mx.exp(0.5 * log_var)`;
   `epsilon = mx.random.normal(shape=mu.shape, key=sample_key)`;
   `z = mu + sigma * epsilon`; `sample_arr = decoder(z)`;
   `mx.eval(sample_arr)` (existing).
4. `state.last_sample = [...]` flattening as today;
   `state.last_compute_flops = max(2*(mu.size + sample_arr.size), 1)`;
   `state._episode_count += 1` (existing).
5. **New** — build the `LatentSample`:
   ```python
   latent_vector = (
       np.asarray(z, dtype=np.float32).ravel().copy()
   )
   species = episode.input_slice.get("species", "default")
   if not isinstance(species, str):
       raise ValueError(
           f"I3: species must be str, got {type(species).__name__}"
       )
   provenance = (
       f"recombine:de={episode.episode_id}:"
       f"ep={state._episode_count - 1}:seed={key_seed}"
   )
   return LatentSample(
       species=species,
       latent_vector=latent_vector,
       provenance=provenance,
   )
   ```

The `state._episode_count - 1` in `provenance` is intentional: by
the time we build the LatentSample the counter has already been
bumped (step 4), so `ep=<count-1>` is the index of *this*
episode (matches the `key_seed` that drove sampling).

### `latent_vector` shape and dtype

- Shape: 1-D, length `mu.size` (the latent dim).
- Dtype: `float32` (enforced by `np.asarray(..., dtype=np.float32)`
  + `B0`'s `LatentSample.__post_init__` finite check).
- Detached from MLX buffers via `.copy()` — the array is safe to
  cache in `EpisodeLogEntry` after `mx.eval` materialises `z`.

### `species` and `provenance`

- `species`: arbitrary string, defaults to `"default"`. Type
  check raises `ValueError("I3: ...")` on non-str. No vocabulary
  restriction beyond `str`.
- `provenance` format (fixed):
  `f"recombine:de={episode_id}:ep={count}:seed={key_seed}"`.
  Reconstructible bit-by-bit from
  `(episode_id, episode_count, key_seed)` — R1 traceability.

### Asymmetry note — empty `delta_latents`

B1b/B2/B3 all have an S1 no-op branch (return `None`). B4's
existing contract raises `ValueError("I3: ...")` on empty
`delta_latents`. **B4 preserves the raise.** Justification:
recombine consumes one latent per call; an empty buffer means
the caller has nothing to generate from, which is an upstream
scheduling error rather than a silent skip. Invariant I3
("latent-dim coherent") presumes a non-empty input. The
asymmetry is intentional and documented in the changelog.

### State (`RecombineRealState`)

**Unchanged.** Keeps `last_sample`, `last_compute_flops`,
`_episode_count`. `latent_vector` lives in the returned
`LatentSample` (captured by `DreamRuntime.execute()` into
`EpisodeLogEntry.channel_outputs`), not on the state.

### K1 FLOPs

**Unchanged.** Already tagged as `2 * (mu.size + sample_arr.size)`.
`LatentSample` construction is `O(z.size)` — folded into the
existing tag.

### Invariants

- **I3** (latent-dim coherent): preserved by the existing
  empty-raise; the `__post_init__` finite check on
  `latent_vector` covers the post-sample side.
- **S2** (finite): enforced at construction time by
  `LatentSample.__post_init__`. A pathological encoder producing
  `log_var = inf` would propagate through `sigma = exp(0.5*inf)`
  → `z = inf` → `LatentSample` raises before return.
- **K1**: unchanged.
- **R1**: deterministic — same `seed + episode_count +
  delta_latents[0]` reproduces the same `z` bit-by-bit; the
  `provenance` string encodes the reproduction recipe.

## DualVer

FC-**MINOR** (`C-v0.18.0 → C-v0.19.0`):

- Return type widened from `Callable[[DE], None]` to
  `Callable[[DE], LatentSample | None]` (additive — no callsite
  outside the handler consumes the return value beyond the
  runtime, which already accepts `None`).
- No axiom, no primitive signature, no invariant ID change.

EC unchanged `+PARTIAL`. `pyproject.toml` version `0.16.0 → 0.17.0`.

## Testing — `tests/unit/test_recombine_latent_sample.py`

Test file name avoids "lora" — B4 is the only B-task **not**
operating on a `LoRAModel`. End-to-end via
`DreamRuntime.execute()` with `_TinyEncoder` / `_TinyDecoder`
fixtures (the cycle-3 test set already defines small MLX modules
for this op — the new test file may re-define minimal fixtures
inline to stay self-contained).

1. **emit_latent_sample** — single call → `runtime.log[-1]
   .channel_outputs[0]` is `LatentSample`.
2. **latent_vector_dtype_shape** — `dtype == np.float32`, 1-D,
   length equals the encoder's latent dim.
3. **latent_vector_finite_propagation** — encoder returning
   `log_var = [inf]` → `LatentSample.__post_init__` raises
   `ValueError(r"^S2:")` before return.
4. **species_default** — no `species` key in `input_slice` →
   `sample.species == "default"`.
5. **species_from_input** — `input_slice["species"] =
   "replay-mix"` → `sample.species == "replay-mix"`.
6. **species_non_str_rejected** — `input_slice["species"] = 42`
   → `ValueError(r"^I3:")`.
7. **provenance_format** — regex
   `r"^recombine:de=.+:ep=\d+:seed=\d+$"` matches.
8. **provenance_count_increments** — two episodes with the
   same `episode_id` → `ep=0` then `ep=1`.
9. **determinism_same_seed** — two handlers built with the
   same `seed`, same `delta_latents`, same `episode_id` →
   bit-identical `latent_vector` AND identical `provenance`.
10. **empty_delta_latents_raises** —
    `ValueError(r"^I3:")` (existing contract preserved).
11. **state_last_sample_preserved** — after a successful
    call, `state.last_sample` is still populated with the
    decoded output (not the latent z).

## Scope boundary

**B4 does** : patch `recombine_real_handler` to return
`LatentSample`; the type check on `species`; the new test file;
the spec/CHANGELOG/version sync (FC-MINOR `C-v0.19.0`).

**B4 does not** : wire the handler into any profile (`p_min` /
`p_equ` keep their skeleton recombine handler); rewire
`consolidate()` (B5); add a `LoRAModel` variant (rejected); add
a `RecombineRealState` field for the latent_vector (lives in the
returned value); soften the empty-`delta_latents` raise to an
S1 no-op (asymmetry documented).

## Convention compliance

`recombine_real.py` is the **only** `_real.py` op that
genuinely respects the "MLX-only" rule from
`kiki_oniric/dream/operations/CLAUDE.md` (the other `_real.py`
ops use numpy via `np.asarray`). B4 adds a top-of-module
`import numpy as np` plus `from kiki_oniric.dream.channels import
LatentSample` (lazy at handler-build time, matching the B1b/B2/B3
precedent). Numpy is needed for the `np.asarray(..., dtype=
np.float32).ravel().copy()` conversion.

## Acceptance criteria

1. `recombine_real_handler` returns `LatentSample` on success
   (continues to raise `ValueError("I3: ...")` on empty
   `delta_latents`).
2. `latent_vector = np.asarray(z, dtype=np.float32).ravel()
   .copy()` — the sampled latent, detached.
3. `species` is read from `input_slice` (default `"default"`);
   non-str raises `ValueError(r"^I3:")`.
4. `provenance` matches
   `r"^recombine:de=.+:ep=\d+:seed=\d+$"`.
5. State (`last_sample`, `last_compute_flops`, `_episode_count`),
   K1 FLOPs, RNG isolation, and determinism are all preserved
   (no regressions on the existing recombine tests).
6. `tests/unit/test_recombine_latent_sample.py` (11 tests)
   passes; full pytest suite green; `uv run mypy harness tests`
   clean; `uv run ruff check .` clean.
7. FC-MINOR DualVer bump recorded in `CHANGELOG.md`
   (`[C-v0.19.0+PARTIAL]`); framework-C spec §4.2 (EN + FR)
   notes that `recombine` now emits a real `LatentSample`,
   completing the 4-of-4 emitting-ops milestone;
   `pyproject.toml` bumped `0.16.0 → 0.17.0`.
