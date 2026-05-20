# B5 — `apply_channel_outputs()` closes the awake↔dream loop

**Date** : 2026-05-20
**Status** : design approved, pending spec review
**Tracking issue** : #15 (`hypneum-lab/dream-of-kiki`)
**Scope** : sub-project B5 of the issue-#15 "approach B" decomposition

---

## Context

Issue #15 / approach B made the four dream operations emit real
channel outputs. Sub-projects shipped on `main`:

- **B0** (`C-v0.14.0`) — channel-output contract.
- **B1a / B1b** (`C-v0.16.0`) — `LoRAModel` + `replay`
  emitting `WeightUpdate`.
- **B2** (`C-v0.17.0`) — `downscale` emitting `WeightUpdate`.
- **B3** (`C-v0.18.0`) — `restructure` emitting `TopologyDiff`
  + `TopologyDiff.__post_init__` S3 guard.
- **B4** (`C-v0.19.0`) — `recombine` emitting `LatentSample`.

All four dream operations now emit; the channel Protocols in
`kiki_oniric/core/primitives.py` (`WeightDeltaChannel`,
`LatentSampleChannel`, `HierarchyChangeChannel`,
`AttentionPriorChannel`) define what consumers should look like
— but only `AttentionPriorChannel` has a concrete implementation
today (`kiki_oniric/dream/channels/attention_prior.py`, with
`.emit` instead of `.set_prior`).

**B5** closes the awake↔dream loop by:

1. Adding **concrete LoRA-target channel implementations** for the
   three missing types.
2. Adding `apply_channel_outputs(log, *, channels)` — a single
   dispatch function that walks `EpisodeLogEntry.channel_outputs`,
   types each `ChannelOutput`, and routes it to the matching
   channel.

## Problem

The handlers mutate the dream-model in place; their channel
outputs (delta dicts, topology diffs, latent samples) currently
go nowhere beyond the runtime log. There is no consumer that
applies them to an awake-model. Without this consumer, the
"approach B" loop is open at the awake end — the dream side has
data nobody reads.

## Approaches considered

**Awake/dream model split.** Three options were considered:

1. **Explicit split.** `consolidate()`-style code holds two
   `LoRAModel` instances — `dream_model` (scratch space the
   handlers mutate) and `awake_model` (target of `channel.apply`).
   The end-to-end test is: clone `awake_model = LoRAModel(...,
   seed=K)`; dream-runtime mutates a fresh `LoRAModel(..., seed=K)`
   identically initialised; the channel apply on `awake_model`
   yields a model bit-equal to the post-handler dream model.
   **Chosen.** Clean swap-time semantics; R1-friendly via seeded
   `LoRALinear` reconstruction.
2. Single model — `channel.apply` is informational only (validates
   S2 but doesn't mutate). Loses semantic value. Rejected.
3. Refactor handlers to NOT mutate, push mutation into channels.
   Cleaner long-term but breaks the contracts of B1b/B2/B3 already
   shipped. Rejected (out of scope for B5).

**Apply API.** Three options were considered:

1. **New free function** `apply_channel_outputs(log, *, channels)`
   in `kiki_oniric/consolidate.py`, independent of `consolidate()`.
   Tested in isolation; `consolidate()`'s nerve-wml-facing
   interface stays untouched. **Chosen.**
2. Extending `consolidate()` with an optional `target_model`
   parameter. Risks breaking the nerve-wml stable interface.
   Rejected.
3. Method on each `Profile` (`profile.consolidate(log,
   target_model)`). Triplicates the code across three profiles.
   Rejected (YAGNI).

**Concrete channel implementations.** Three options:

1. **Minimal LoRA-target impls** — three new classes
   (`LoRAWeightDeltaChannel`, `LoRAHierarchyChangeChannel`,
   `LatentSampleQueue`). The first two take a `target: LoRAModel`
   and mutate it; the third is a substrate-agnostic FIFO.
   `fisher_bump=None` is ignored (Fisher is future work). **Chosen.**
2. Add Fisher bump handling. YAGNI — B1b/B2 never emit a non-None
   `fisher_bump`.
3. Stub channels that only validate. Doesn't close the loop.
   Rejected.

## Design

### New file `kiki_oniric/dream/channels/weight_delta.py`

```python
"""LoRA-target concrete implementation of WeightDeltaChannel."""
from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


class LoRAWeightDeltaChannel:
    """Concrete ``WeightDeltaChannel`` — applies ``lora_delta`` to
    a ``LoRAModel`` additively. ``fisher_bump`` is accepted to match
    the Protocol but ignored (B5 scope — Fisher is future work)."""

    def __init__(self, target: "LoRAModel") -> None:
        self._target = target

    def apply(
        self,
        lora_delta: dict[str, NDArray[np.float32]],
        fisher_bump: dict[str, NDArray[np.float32]] | None = None,
    ) -> None:
        for key, delta_arr in lora_delta.items():
            layer_idx, attr = self._parse_key(key)
            if layer_idx >= len(self._target.layers):
                raise ValueError(
                    f"S1: weight_delta key {key!r} references "
                    f"layer {layer_idx} but target has "
                    f"{len(self._target.layers)} layers"
                )
            layer = self._target.layers[layer_idx]
            current = getattr(layer, attr)
            new = current + mx.array(delta_arr)
            if not bool(mx.all(mx.isfinite(new)).item()):
                raise ValueError(
                    f"S2: weight_delta apply non-finite on {key!r}"
                )
            setattr(layer, attr, new)
        mx.eval(self._target.parameters())

    @staticmethod
    def _parse_key(key: str) -> tuple[int, str]:
        """Parse 'layer<i>.lora_a' / 'layer<i>.lora_b' → (i, attr)."""
        if "." not in key:
            raise ValueError(f"S1: invalid lora_delta key {key!r}")
        prefix, attr = key.rsplit(".", 1)
        if not prefix.startswith("layer"):
            raise ValueError(f"S1: invalid lora_delta key {key!r}")
        try:
            idx = int(prefix[len("layer"):])
        except ValueError as exc:
            raise ValueError(
                f"S1: invalid lora_delta key {key!r}"
            ) from exc
        if attr not in {"lora_a", "lora_b"}:
            raise ValueError(
                f"S1: invalid lora_delta attr {attr!r}"
            )
        return idx, attr
```

### New file `kiki_oniric/dream/channels/hierarchy_change.py`

```python
"""LoRA-target concrete implementation of HierarchyChangeChannel."""
from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


def _apply_topology_op(
    model: "LoRAModel",
    op: str,
    payload: dict[str, object],
) -> None:
    """Replay a single (op, payload) on ``model``.

    Shared between ``LoRAHierarchyChangeChannel.apply_diff`` (B5)
    and a future refactor of ``restructure_lora_handler`` (B3) —
    today B3 inlines the same logic.
    """
    from kiki_oniric.substrates.micro_kiki.lora_model import (
        LoRALinear,
    )

    if op == "add":
        new_layer = LoRALinear(
            in_features=int(payload["in_features"]),
            out_features=int(payload["out_features"]),
            rank=int(payload["rank"]),
            alpha=float(payload["alpha"]),
            key=mx.random.key(int(payload["seed"])),
        )
        model.layers.insert(int(payload["index"]), new_layer)
    elif op == "remove":
        model.layers.pop(int(payload["index"]))
    elif op == "reroute":
        i, j = payload["swap_indices"]  # type: ignore[misc]
        model.layers[int(i)], model.layers[int(j)] = (
            model.layers[int(j)],
            model.layers[int(i)],
        )
    else:
        raise ValueError(f"S3: unknown topology op {op!r}")


class LoRAHierarchyChangeChannel:
    """Concrete ``HierarchyChangeChannel`` — replays a ``TopologyDiff``'s
    ``diff`` entries onto a ``LoRAModel`` via the shared
    ``_apply_topology_op`` helper. ``TopologyDiff.__post_init__``
    already validated each entry — no re-validation here."""

    def __init__(self, target: "LoRAModel") -> None:
        self._target = target

    def apply_diff(
        self, diff: list[tuple[str, dict[str, object]]],
    ) -> None:
        for op, payload in diff:
            _apply_topology_op(self._target, op, payload)
```

### New file `kiki_oniric/dream/channels/latent_sample.py`

```python
"""Substrate-agnostic FIFO queue implementing LatentSampleChannel."""
from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray


class LatentSampleQueue:
    """Concrete ``LatentSampleChannel`` — FIFO queue of latent samples."""

    def __init__(self, capacity: int | None = None) -> None:
        self._queue: deque = deque(maxlen=capacity)

    def enqueue(
        self,
        species: str,
        latent_vector: NDArray[np.float32],
        provenance: str,
    ) -> None:
        if not np.isfinite(latent_vector).all():
            raise ValueError(
                "S2: LatentSampleQueue.enqueue latent_vector non-finite"
            )
        self._queue.append(
            {
                "species": species,
                "latent_vector": np.asarray(
                    latent_vector, dtype=np.float32,
                ).copy(),
                "provenance": provenance,
            }
        )

    def dequeue(self) -> dict | None:
        try:
            return self._queue.popleft()
        except IndexError:
            return None

    def __len__(self) -> int:
        return len(self._queue)
```

### Patch `kiki_oniric/dream/channels/attention_prior.py`

Add `set_prior` as a thin alias for `emit` (the existing API
keyword); preserves backwards compatibility.

```python
# Append inside AttentionPriorChannel:
def set_prior(self, prior: NDArray) -> None:
    """Alias of :meth:`emit` — matches the WeightDeltaChannel
    Protocol vocabulary."""
    self.emit(prior)
```

### New function in `kiki_oniric/consolidate.py`

```python
def apply_channel_outputs(
    log: list["EpisodeLogEntry"],
    *,
    weight_channel: "WeightDeltaChannel",
    hierarchy_channel: "HierarchyChangeChannel",
    latent_channel: "LatentSampleChannel",
    attention_channel: "AttentionPriorChannel | None" = None,
) -> int:
    """Replay every non-None channel output in ``log`` onto the
    matching channel. Returns the number of outputs dispatched.

    Raises propagate from the underlying ``.apply()`` / ``.enqueue()``
    / ``.set_prior()`` calls (S1, S2, S3 enforced by the channel
    impls).
    """
    from kiki_oniric.dream.channels import (
        AttentionPrior,
        LatentSample,
        TopologyDiff,
        WeightUpdate,
    )

    count = 0
    for entry in log:
        for output in entry.channel_outputs:
            if output is None:
                continue
            if isinstance(output, WeightUpdate):
                weight_channel.apply(
                    output.lora_delta, output.fisher_bump,
                )
            elif isinstance(output, TopologyDiff):
                hierarchy_channel.apply_diff(list(output.diff))
            elif isinstance(output, LatentSample):
                latent_channel.enqueue(
                    output.species,
                    output.latent_vector,
                    output.provenance,
                )
            elif isinstance(output, AttentionPrior):
                if attention_channel is None:
                    raise ValueError(
                        "apply_channel_outputs: attention_channel "
                        "required for AttentionPrior outputs"
                    )
                attention_channel.set_prior(output.prior)
            else:
                raise TypeError(
                    f"apply_channel_outputs: unknown ChannelOutput "
                    f"type {type(output).__name__}"
                )
            count += 1
    return count
```

### Tests — `tests/unit/test_apply_channel_outputs.py` (~12 tests)

End-to-end via `DreamRuntime` + dream/awake model clones. Each
test creates two `LoRAModel(seed=K)` (bit-identical at t=0),
runs the corresponding B1b/B2/B3 handler on the dream-model,
then calls `apply_channel_outputs(log, ...)` with the awake-model
as channel target, and asserts bit-equality between the two
models (or the queue/channel state).

1. **dispatch_weight_update** — log with 1 `WeightUpdate` → after
   `apply`, `awake_model.layers[i].lora_a/b == dream_model
   .layers[i].lora_a/b` bit-exact.
2. **dispatch_topology_diff_reroute** — log with 1
   `TopologyDiff(reroute)` → awake_model.layers swapped.
3. **dispatch_topology_diff_add** — log with 1 `TopologyDiff(add)`
   → awake_model gains a `LoRALinear` whose `lora_a/b` match the
   dream-model's inserted layer bit-exact (seed reconstruction).
4. **dispatch_topology_diff_remove** — log with 1 `TopologyDiff
   (remove)` → awake_model loses the layer at the right index.
5. **dispatch_latent_sample** — log with 1 `LatentSample` →
   `len(queue) == 1`; `queue.dequeue()` returns the same fields.
6. **dispatch_attention_prior** — log with 1 `AttentionPrior` →
   `attention_channel.get_prior()` matches.
7. **dispatch_attention_prior_required** — `AttentionPrior` in log
   but `attention_channel=None` → `ValueError`.
8. **dispatch_skip_none** — log with `(None, WeightUpdate(...),
   None)` → returns `count == 1`; only one apply called.
9. **dispatch_empty_log** — empty list → `count == 0`, no apply.
10. **dispatch_unknown_type_rejected** — feed a fake `ChannelOutput`
    of an unsupported type → `TypeError`.
11. **end_to_end_replay_consolidation** — run `replay_lora_handler`
    on dream, then `apply_channel_outputs` on awake → bit-equal
    LoRAModels.
12. **end_to_end_multi_op** — run an episode triggering both a
    `WeightUpdate` and a `TopologyDiff` (e.g. via runtime with two
    handlers registered) → both deltas apply, awake-model matches
    dream-model.

### Invariants

- **S1** (retained non-regression): `LoRAWeightDeltaChannel.apply`
  is additive; with bit-identical initial models and the dream
  handler's emitted delta, the awake-model converges to the
  dream-model.
- **S2** (finite): checked at `apply` time on the new tensor;
  raised with `"S2:"` prefix.
- **S3** (topology): not re-validated by the channel —
  `TopologyDiff.__post_init__` (B3) already enforced this at
  the diff's construction site.
- **R1** (reproducibility): `add` payload carries the per-op
  `seed`; `mx.random.key(payload["seed"])` reconstructs the
  inserted layer bit-exactly.

### Convention compliance

`weight_delta.py` and `hierarchy_change.py` use MLX at module
import time (the channels operate on a `LoRAModel`). Both
sit in `kiki_oniric/dream/channels/` next to the existing
`attention_prior.py` and `alpha_stream.py`. `latent_sample.py`
is pure numpy + stdlib.

## DualVer

FC-**MINOR** (`C-v0.19.0 → C-v0.20.0`):

- Three new concrete channel implementations (additive).
- `apply_channel_outputs` is a new free function (additive).
- `AttentionPriorChannel.set_prior` is an alias for the existing
  `.emit` (no signature change).
- No axiom, no Protocol change.

EC unchanged `+PARTIAL`. `pyproject.toml` version `0.17.0 → 0.18.0`.

## Scope boundary

**B5 does** : the three new channel files; the `set_prior` alias
on `AttentionPriorChannel`; `apply_channel_outputs` in
`consolidate.py`; the shared `_apply_topology_op` helper (used by
the new `LoRAHierarchyChangeChannel`; future B3 refactor may
delegate to it but is not required by B5); ~12 end-to-end tests;
CHANGELOG + framework-C §4.1 EN+FR note ; version bump.

**B5 does not** : wire the new channels into profiles
(`p_min`/`p_equ`/`p_max` continue to use their cycle-3 channel
wiring — a future **B6** sub-project addresses profile-level
integration); modify `consolidate()`'s nerve-wml-facing
signature; implement Fisher bump handling; refactor
`restructure_lora_handler` to delegate to the shared
`_apply_topology_op` helper (cleanup deferred to a follow-up).

## Acceptance criteria

1. `kiki_oniric/dream/channels/weight_delta.py` ships
   `LoRAWeightDeltaChannel` with `apply(lora_delta, fisher_bump=
   None)`; `fisher_bump` accepted but ignored; per-key parsing
   validates `"layer<i>.lora_a"` / `.lora_b"` format; per-tensor
   S2 finite check raises `"S2:"` on NaN/Inf.
2. `kiki_oniric/dream/channels/hierarchy_change.py` ships
   `LoRAHierarchyChangeChannel` and the shared
   `_apply_topology_op(model, op, payload)` helper; `apply_diff`
   replays every entry; `add` reconstructs via
   `mx.random.key(payload["seed"])` bit-exactly.
3. `kiki_oniric/dream/channels/latent_sample.py` ships
   `LatentSampleQueue` with FIFO `enqueue` / `dequeue`, optional
   capacity, S2 finite check on enqueue.
4. `AttentionPriorChannel` gains a `set_prior(prior)` method as
   an alias of `.emit`.
5. `kiki_oniric/consolidate.py` exports
   `apply_channel_outputs(log, *, weight_channel,
   hierarchy_channel, latent_channel, attention_channel=None)
   -> int`; dispatches by `isinstance` on
   `WeightUpdate / TopologyDiff / LatentSample / AttentionPrior`;
   skips `None` entries; raises `TypeError` on unknown types;
   raises `ValueError` if an `AttentionPrior` is emitted but
   `attention_channel is None`.
6. `tests/unit/test_apply_channel_outputs.py` (12 tests) passes;
   full pytest suite green; `uv run mypy harness tests` clean;
   `uv run ruff check .` clean.
7. FC-MINOR DualVer bump recorded in `CHANGELOG.md`
   (`[C-v0.20.0+PARTIAL]`); framework-C spec §4.1 (channels) EN+FR
   notes that the four channel Protocols now have concrete
   LoRA-target implementations; `pyproject.toml` bumped `0.17.0 →
   0.18.0`.
