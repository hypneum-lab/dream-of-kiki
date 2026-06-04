# B2 — `downscale` emits a `WeightUpdate`

**Date** : 2026-05-20
**Status** : design approved, pending spec review
**Tracking issue** : #15 (`hypneum-lab/dream-of-kiki`)
**Scope** : sub-project B2 of the issue-#15 "approach B" decomposition

---

## Context

Issue #15 / approach B makes the four dream operations produce real
channel outputs. Prior sub-projects on `main`:

- **B0** (`C-v0.14.0`) froze the channel-output contract.
- **B1a** (`C-v0.15.0`) built the `LoRAModel` abstraction
  (`kiki_oniric/substrates/micro_kiki/lora_model.py`) — frozen base,
  trainable named A/B adapters, `adapter_parameters()` returning
  the named A/B arrays, plus the helper `adapter_delta(before,
  after)` (B1b extension, MLX→numpy float32 delta with key + shape
  guards).
- **B1b** (`C-v0.16.0`) shipped `replay_lora_handler` in
  `replay_real.py` — the first dream operation to emit a real
  channel-1 `WeightUpdate`.

**B2** makes `downscale` the second emitting op. The framework-C
spec §4.2 says:

> **downscale** — apply homeostatic regularizer (SHY-style), shrink
> weights toward prior, reduce noise.
> Source: B-Tononi synaptic homeostasis.
> Input: γ or W directly. **Output: WeightUpdate (channel 1).**

## Problem

`downscale_real.py`'s `downscale_real_handler` multiplies a plain
MLX module's `layer.weight` and `layer.bias` by a `shrink_factor`
in place and returns `None`. No handler operates on a `LoRAModel`,
and none emits a `WeightUpdate`. B2 adds the LoRA-adapter
shrinkage variant.

## Approaches considered

The handler-emission mechanism is already settled (B0 widened the
return type; `DreamRuntime.execute()` collects it). The choice is
where the LoRA shrinkage logic lives:

1. **Extend `downscale_real.py` with a new factory.** Mirrors the
   B1b decision for `replay_real.py`. The repo's
   `operations/CLAUDE.md` mandates the three-variant layout
   (skeleton / `_real` / `_snn`) and forbids a 4th variant file;
   substrate-specific paths live behind the existing variant names
   as additional factories. **Chosen.**
2. A new `downscale_lora.py` file — rejected (violates the
   three-variant rule).
3. Mutate `downscale_real_handler` to branch on model type —
   rejected (one factory carrying two contracts is brittle).

## Design

### New factory in `kiki_oniric/dream/operations/downscale_real.py`

```python
def downscale_lora_handler(
    state: DownscaleRealState,
    *,
    model: "LoRAModel",
) -> Callable[[DreamEpisode], "WeightUpdate | None"]:
```

The factory reuses the existing `DownscaleRealState` (it already
carries `compound_factor`, `last_compute_flops`,
`total_compute_flops`). No new state class — `WeightUpdate` is the
handler's return value, not stored state.

### Handler behaviour

On each `DreamEpisode`:

1. Read `factor = episode.input_slice.get("shrink_factor", 1.0)`.
2. Validate `0.0 < factor <= 1.0`; otherwise raise `ValueError`
   citing `S2`. Validation runs **before** any model mutation.
3. **`factor == 1.0` → S1 no-op**: return `None`, leave the model
   untouched, set `last_compute_flops = 0`. Symmetric with B1b's
   empty-records branch.
4. Snapshot the adapters before the step: copy each
   `model.adapter_parameters()` value via `mx.array(v)` (defensive
   detach — MLX param assignment rebinds, so old refs are stable,
   but the explicit copy makes the snapshot self-evident).
5. Walk `model.layers` and shrink each `LoRALinear`'s adapters
   in place: `layer.lora_a = layer.lora_a * factor`,
   `layer.lora_b = layer.lora_b * factor`. The frozen base weight
   is **not touched** — B1a's `freeze` excludes it from the
   trainable parameter set but in this case we simply do not
   reference it; SHY semantics on the LoRA substrate target the
   adaptation only.
6. `mx.eval(model.parameters())` — materialise.
7. Snapshot after: `after = model.adapter_parameters()`.
8. Compute `lora_delta = adapter_delta(before, after)` (helper
   already in `lora_model.py`).
9. Update state: `compound_factor *= factor` (Tononi
   non-idempotence preserved); FLOPs (see below).
10. **Return** `WeightUpdate(lora_delta=<deltas>,
    fisher_bump=None)`. The runtime captures it into
    `EpisodeLogEntry.channel_outputs` (B0).

### `lora_delta` content

`lora_delta` carries the **A/B adapter deltas** keyed exactly as
`model.adapter_parameters()` keys them (`layer<i>.lora_a`,
`layer<i>.lora_b`). For shrinkage with factor `f`:

- `Δ_lora_a[layer<i>] = lora_a · (f - 1)` (non-positive when
  `f ≤ 1`)
- `Δ_lora_b[layer<i>] = lora_b · (f - 1)`

The dense effective change is `scale · ((B·f)·(A·f) - B·A) =
scale · (f² - 1) · B·A` — shrinking the adapters by `f` shrinks
the effective `ΔW` by `f²`. This non-linear amplification is
exactly the SHY signature: weak connections are pruned faster
than strong ones.

`fisher_bump` is `None` (the contract allows it).

### K1 (budget) compliance

The cost is dominated by the per-layer elementwise scale of the
adapters: `rank * (in_features + out_features)` ops per layer,
times two (forward + scratch). Concretely:

```python
def _flop_estimate_downscale_lora(model: "LoRAModel") -> int:
    per_layer = sum(
        2 * layer.rank * (layer.in_features + layer.out_features)
        for layer in model.layers
    )
    return max(per_layer, 1)
```

This is K1's order-of-magnitude tag; replays with batched gradient
steps remain the more expensive op.

### Invariants

- **S2** (finite): enforced by `WeightUpdate.__post_init__` —
  finiteness is verified at construction. Shrinking by `f ≤ 1`
  cannot introduce NaN/Inf from finite starting values.
- **I-Wmag**: cited in `operations/CLAUDE.md` for downscale but
  not formally defined in `docs/invariants/registry.md`. Any
  reasonable reading ("bounded magnitude of weights / of the
  delta") is **trivially satisfied** by multiplicative shrinkage
  with `f ≤ 1` — magnitudes only decrease. B2 cites S2 and flags
  I-Wmag as an out-of-scope invariant-registry cleanup item.
- **S1** (retained non-regression): enforced by the
  swap-protocol, not by the handler. Whether a given
  `shrink_factor` is safe for retained accuracy is the caller's
  scheduling decision (the K1-budgeted policy).

## Testing — emit + composed-ΔW property

New `tests/unit/test_downscale_lora.py`. Handler exercised
end-to-end through a directly-constructed `DreamRuntime` (no
profile).

- The handler returns a `WeightUpdate`; `runtime.log[-1]
  .channel_outputs[0]` is a `WeightUpdate`.
- `lora_delta` keys match `LoRAModel.adapter_parameters()` keys;
  every value is a finite `float32` numpy array.
- The deltas are **non-positive** for `factor < 1` (shrinkage):
  every element ≤ 0.
- Magnitudes scale: `factor == 0.5` produces deltas whose absolute
  values equal `0.5 · |adapter_before|`.
- **Composed-ΔW property test** — for a chosen layer the dense
  effective change `scale · (B_after @ A_after - B_before @
  A_before)` matches `scale · (f² - 1) · (B_before @ A_before)`
  within tolerance. This is the SHY non-linear amplification
  property and the analogue of B1b's composed-ΔW test.
- `factor == 1.0` → handler returns `None`,
  `last_compute_flops == 0`, model untouched.
- `factor` out of bounds (`0`, negative, > 1) raises `ValueError`;
  the model is untouched after the exception (validation-before-
  mutation discipline from `operations/CLAUDE.md`).
- `compound_factor` updates multiplicatively across two calls:
  `0.9` then `0.8` → `state.compound_factor ≈ 0.72`.
- K1: `last_compute_flops > 0`, `total_compute_flops` accumulates.
- Determinism (R1): same model + factor → bit-identical
  `lora_delta`.

## Scope boundary

**B2 does** : the `downscale_lora_handler` factory in
`downscale_real.py`, the per-adapter shrinkage + delta capture,
the `WeightUpdate` return, K1 FLOP tagging, unit tests, the
spec/`CHANGELOG`/version sync.

**B2 does not** : wire the handler into any profile (`p_min` /
`p_equ` keep their skeleton handlers); rewire `consolidate()`
(B5); define the I-Wmag invariant formally (separate cleanup);
add a Fisher bump.

## Convention compliance

`downscale_real.py` is MLX-only per `operations/CLAUDE.md`. The
new factory imports MLX (`mlx.core`) lazily and imports
`adapter_delta` from `lora_model.py` and `WeightUpdate` from
`channels.py` — no `numpy` import lands in `downscale_real.py`,
matching the B1b precedent. A `TYPE_CHECKING` guard hosts
`LoRAModel` and `WeightUpdate` for the string forward-refs.

## DualVer

FC-**MINOR** (`C-v0.16.0 → C-v0.17.0`): a new substrate handler;
no axiom, invariant, or primitive-signature change. EC unchanged.
`pyproject.toml` version `0.14.0 → 0.15.0`.

## Acceptance criteria

1. `downscale_lora_handler` exists in `downscale_real.py`, takes
   a `LoRAModel`, reuses `DownscaleRealState`.
2. The handler validates `0 < factor ≤ 1` before any mutation;
   `factor == 1.0` returns `None` with FLOPs 0.
3. The handler shrinks `lora_a` and `lora_b` of every
   `LoRALinear` by `factor` and returns a `WeightUpdate` whose
   `lora_delta` carries the per-adapter A/B deltas keyed as
   `adapter_parameters()` keys, with `fisher_bump=None`.
4. The composed-`ΔW` property `(f² - 1) · B·A` holds within
   tolerance for the emitted deltas.
5. `compound_factor` compounds multiplicatively across episodes;
   K1 FLOP fields are tagged.
6. `tests/unit/test_downscale_lora.py` covers the test plan; full
   suite green, `uv run mypy harness tests` clean, no `numpy`
   import in `downscale_real.py`.
7. FC-MINOR DualVer bump recorded in `CHANGELOG.md`; framework-C
   spec §4.2 (EN + FR) notes that `downscale` now emits a real
   `WeightUpdate`.
