# B1b — `replay` emits a real `WeightUpdate`

**Date** : 2026-05-19
**Status** : design approved, pending spec review
**Tracking issue** : #15 (`hypneum-lab/dream-of-kiki`)
**Scope** : sub-project B1b — second half of B1, itself sub-project B1 of the issue-#15 "approach B" decomposition

---

## Context

Issue #15 / approach B makes the four dream operations produce real
channel outputs. Prior sub-projects, all on `main`:

- **B0** (`C-v0.14.0`) froze the channel-output contract:
  `WeightUpdate` type, `EpisodeLogEntry.channel_outputs`,
  `OperationHandler` widened to `Callable[[DreamEpisode],
  ChannelOutput | None]`, and `DreamRuntime.execute()` collects
  each handler's return value into `channel_outputs`.
- **B1a** (`C-v0.15.0`) built the LoRA-adapter model abstraction:
  `LoRALinear` / `LoRAModel` in
  `kiki_oniric/substrates/micro_kiki/lora_model.py` — a frozen
  base weight plus trainable named A/B adapters, a differentiable
  forward pass, and `adapter_parameters()` returning the named
  A/B arrays.

**B1b** is the payoff: the `replay` operation does a LoRA-only
gradient step on a `LoRAModel` and **returns** a real channel-1
`WeightUpdate`. Because B0 already widened the handler contract
and `execute()` already captures the return value, the
handler-to-runtime plumbing exists — B1b only has to produce the
value.

## Problem

`replay` today has a skeleton handler (counters only) and an MLX
handler (`replay_handler_mlx` / `replay_real_handler`) that does
forward + MSE + SGD on *all* parameters of a plain model and
returns `None`. No replay handler produces a `WeightUpdate`, and
none operates on a LoRA-adapter model. B1b adds one.

## Approaches considered

The handler-emission mechanism was an open question in early
exploration, but B0 settled it: a handler simply **returns** a
`ChannelOutput`, and `DreamRuntime.execute()` records it. So the
only real choice is *where the code lives*:

1. **Extend `replay_real.py` with a new factory.** The repo's
   `operations/CLAUDE.md` mandates the three-variant layout
   (skeleton / `_real` / `_snn`) and forbids a 4th variant file;
   new substrate-specific paths live behind the existing variant
   names. The LoRA-replay handler is an MLX-backed real path, so
   it belongs in `replay_real.py` as an additional factory.
   **Chosen.**
2. A new `replay_lora.py` file. Rejected — violates the
   three-variant rule.
3. Mutate `replay_real_handler` to branch on model type. Rejected
   — overloads one factory with two contracts; a separate factory
   is clearer.

## Design

### New factory in `kiki_oniric/dream/operations/replay_real.py`

```python
def replay_lora_handler(
    state: ReplayRealState,
    *,
    model: "LoRAModel",
    lr: float = 0.01,
) -> Callable[[DreamEpisode], "WeightUpdate | None"]:
```

The factory reuses the existing `ReplayRealState` (it already
carries the K1 FLOP-tagging fields `last_compute_flops` /
`total_compute_flops`, plus `total_records_consumed` /
`last_loss`). No new state class — the `WeightUpdate` is the
handler's *return value*, not stored state.

### Handler behaviour

On each `DreamEpisode`:

1. Read `records = episode.input_slice.get("beta_records", [])`.
   Validate each record has `x` and `y` keys (cite the schema in
   the error, as the existing replay handlers do).
2. **Empty records → return `None`** (S1 no-op): set
   `state.last_loss = None`, `state.last_compute_flops = 0`, and
   return without emitting a `WeightUpdate`.
3. Snapshot the adapters *before* the step: copy
   `model.adapter_parameters()` into a `dict[str, mx.array]`
   (the named A/B arrays).
4. Build `xs` / `ys` from the records, run the gradient step:
   `loss, grads = nn.value_and_grad(model, mse_loss)(model, xs,
   ys)`; `optimizer.update(model, grads)`; `mx.eval(...)`. Because
   B1a freezes each `LoRALinear`'s base weight, MLX excludes the
   base from the gradient tree — the SGD step updates **only the
   A/B adapters**, with no explicit scoping.
5. Snapshot the adapters *after* the step. Compute per-adapter
   delta `after[k] - before[k]` for each key `k` (the
   `layer<i>.lora_a` / `layer<i>.lora_b` keys from
   `adapter_parameters()`), and convert each to a contiguous
   numpy `float32` array.
6. Update state: `total_records_consumed += len(records)`,
   `last_loss = float(loss)`, and tag K1 FLOPs (see below).
7. **Return** `WeightUpdate(lora_delta=<the per-adapter delta
   dict>, fisher_bump=None)`. `WeightUpdate.__post_init__`
   enforces S2 finiteness; `DreamRuntime.execute()` records the
   return value into `EpisodeLogEntry.channel_outputs`.

### `lora_delta` content

`lora_delta` carries the **A/B adapter deltas** — the raw change
to the low-rank matrices — keyed exactly as
`model.adapter_parameters()` keys them (`layer<i>.lora_a`,
`layer<i>.lora_b`). This is the faithful low-rank representation;
the composed dense `ΔW` is derivable from it but is not emitted.
`fisher_bump` is `None` (deferred — the contract allows it).

### K1 (budget) compliance

As a `_real` handler, the LoRA replay handler tags FLOPs on
`ReplayRealState`. The low-rank step's cost is dominated by the
`B @ A` product per layer: estimate
`last_compute_flops ≈ 2 * rank * in * out * n_records` summed
over layers (forward + backward), which is smaller than a
full-weight replay step. Empty records → `last_compute_flops = 0`.

## Testing — "test both representations"

New `tests/unit/test_replay_lora.py`. The handler is exercised
end-to-end through a directly-constructed `DreamRuntime` (the
pattern B0's runtime tests use) — no profile involved.

- The handler registered on a `DreamRuntime` and invoked via
  `runtime.execute(episode)`; assert the resulting
  `EpisodeLogEntry.channel_outputs[0]` is a `WeightUpdate`.
- **Representation A — emitted A/B deltas:** `lora_delta` keys
  equal the `LoRAModel.adapter_parameters()` keys; every value is
  a finite `float32` numpy array; after a real step on non-empty
  records the deltas are non-zero.
- **Representation B — composed `ΔW` (property test):** for each
  layer, the dense effective change
  `scale * (B_after @ A_after - B_before @ A_before)` is
  consistent with the emitted A/B deltas — i.e. recomposing the
  low-rank deltas reproduces the effective weight change. This
  verifies the low-rank form composes correctly without the
  contract carrying the dense form.
- Empty `beta_records` → handler returns `None`,
  `channel_outputs[0]` is `None`, `last_compute_flops == 0`.
- Determinism (R1): same model + records + seed → bit-identical
  `lora_delta`.
- A malformed record (missing `x`/`y`) raises `ValueError`
  before any mutation.

## Scope boundary

**B1b does** : the `replay_lora_handler` factory in
`replay_real.py`, the per-adapter delta capture, the `WeightUpdate`
return, K1 FLOP tagging, unit tests, and the
spec/`CHANGELOG`/version sync.

**B1b does not** : wire the handler into any profile (`p_min` /
`p_equ` keep their skeleton replay handler; deciding which
profile receives a `LoRAModel` and with what configuration is a
separate concern). It does not rewire `consolidate()` (B5), and
it does not compute a Fisher bump.

## DualVer

FC-**MINOR** (`C-v0.15.0 → C-v0.16.0`): a new substrate handler,
no axiom, invariant, or primitive-signature change (the
`OperationHandler` type already permits a `ChannelOutput` return
since B0). EC unchanged. `pyproject.toml` version
`0.13.0 → 0.14.0`.

## Acceptance criteria

1. `replay_lora_handler` exists in `replay_real.py`, takes a
   `LoRAModel`, reuses `ReplayRealState`.
2. The handler runs an adapter-only SGD step (base frozen by B1a)
   and returns a `WeightUpdate` whose `lora_delta` carries the
   per-adapter A/B deltas keyed as `adapter_parameters()` keys,
   `fisher_bump=None`.
3. Empty `beta_records` → returns `None`, FLOPs tagged 0.
4. K1 FLOP fields populated on `ReplayRealState`.
5. `tests/unit/test_replay_lora.py` covers both representations
   (emitted A/B deltas + composed-`ΔW` property), the empty case,
   determinism, and malformed-record rejection; the handler is
   tested end-to-end via `DreamRuntime.execute()`.
6. Full suite green, `uv run mypy harness tests` clean, FC-MINOR
   DualVer bump recorded in `CHANGELOG.md`; framework-C spec §4.2
   (EN + FR) notes that `replay` now emits a real `WeightUpdate`.
