# B5 — `apply_channel_outputs` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the awake↔dream loop by shipping three concrete LoRA-target channels (`LoRAWeightDeltaChannel`, `LoRAHierarchyChangeChannel`, `LatentSampleQueue`), a `set_prior` alias on the existing `AttentionPriorChannel`, and a free function `apply_channel_outputs(log, …)` that dispatches `EpisodeLogEntry.channel_outputs` to the right channel.

**Architecture:** The dream-runtime handlers (B1b/B2/B3/B4) emit typed `ChannelOutput`s into `EpisodeLogEntry.channel_outputs`. `apply_channel_outputs` walks the log, dispatches by `isinstance`, and lets each channel mutate its **awake-side target** (typically a `LoRAModel` clone of the dream-side scratch model, initialised with the same seed so `add` payloads can be reconstructed bit-exactly via `mx.random.key(payload["seed"])`). `consolidate()`'s nerve-wml-facing signature stays untouched; profile wiring is deferred to B6.

**Tech Stack:** Python 3.12, `uv`, MLX (`mlx.core`), numpy, `collections.deque`, pytest, mypy.

**Spec:** `docs/superpowers/specs/2026-05-20-b5-consolidate-apply-design.md`

**Conventions to respect (`kiki_oniric/CLAUDE.md`, `dream/operations/CLAUDE.md`, root `CLAUDE.md`):**

- Validate before mutating. Cite invariant IDs in error messages (`S1:`, `S2:`, `S3:`).
- Determinism is a contract — seed-driven `mx.random.key` reconstruction is the *only* way `add` can stay R1.
- Subject ≤ 50 chars; body lines ≤ 72; English; no AI attribution; no `--no-verify`.
- `uv run` for every command.

**Key invariant — `add` reconstruction:** `_apply_topology_op` must call `LoRALinear(in_features, out_features, rank, alpha, key=mx.random.key(payload["seed"]))`. The seed lives in the `add` payload (B3 puts it there). Reconstructing without the seed breaks R1; reconstructing with the wrong key shape (e.g. `mx.random.key(42)` vs `mx.random.key(payload["seed"])`) silently produces a different layer.

---

## File Structure

- **Create** `kiki_oniric/dream/channels/weight_delta.py` — `LoRAWeightDeltaChannel`, additive lora_delta apply with S1 key-parsing + S2 finite guard.
- **Create** `kiki_oniric/dream/channels/hierarchy_change.py` — `_apply_topology_op` shared helper + `LoRAHierarchyChangeChannel`.
- **Create** `kiki_oniric/dream/channels/latent_sample.py` — `LatentSampleQueue` (FIFO `collections.deque`).
- **Modify** `kiki_oniric/dream/channels/attention_prior.py` — add `set_prior` alias of `emit`.
- **Modify** `kiki_oniric/dream/channels/__init__.py` — re-export the new concrete classes from the channels package.
- **Modify** `kiki_oniric/consolidate.py` — append `apply_channel_outputs(...)`.
- **Create** `tests/unit/test_apply_channel_outputs.py` — 12 tests, mostly end-to-end with dream/awake LoRAModel clones.
- **Modify** `CHANGELOG.md`, `pyproject.toml`, `uv.lock` — `[C-v0.20.0+PARTIAL]`, version `0.17.0 → 0.18.0`.
- **Modify** `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` — §4.1 note on the new concrete channels.

---

## Task 1: `LoRAWeightDeltaChannel`

**Files:**
- Create: `kiki_oniric/dream/channels/weight_delta.py`
- Test: `tests/unit/test_apply_channel_outputs.py` (start the file; later tasks append to it)

- [ ] **Step 1: Write the first failing test**

Create `tests/unit/test_apply_channel_outputs.py` with the file header + the first test. Append-friendly structure (later tasks add tests at the bottom):

```python
"""Unit tests for apply_channel_outputs() and concrete channels (B5)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import mlx.core as mx
else:
    mx = pytest.importorskip("mlx.core")

from kiki_oniric.substrates.micro_kiki.lora_model import LoRALinear, LoRAModel


def _clones(seed: int = 0) -> tuple[LoRAModel, LoRAModel]:
    """Return two bit-identical LoRAModels — dream-side and awake-side."""
    return (
        LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=seed),
        LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=seed),
    )


def _assert_lora_models_equal(a: LoRAModel, b: LoRAModel) -> None:
    """Assert bit-equality of every layer's base + adapters."""
    assert len(a.layers) == len(b.layers)
    for la, lb in zip(a.layers, b.layers):
        np.testing.assert_array_equal(
            np.asarray(la.base_weight), np.asarray(lb.base_weight),
        )
        np.testing.assert_array_equal(
            np.asarray(la.lora_a), np.asarray(lb.lora_a),
        )
        np.testing.assert_array_equal(
            np.asarray(la.lora_b), np.asarray(lb.lora_b),
        )
        if la.use_bias:
            np.testing.assert_array_equal(
                np.asarray(la.bias), np.asarray(lb.bias),
            )


def test_lora_weight_delta_channel_additive_apply() -> None:
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )

    _, target = _clones(seed=0)
    before_a = np.asarray(target.layers[0].lora_a, dtype=np.float32).copy()
    delta_a = np.ones_like(before_a) * 0.5
    channel = LoRAWeightDeltaChannel(target)
    channel.apply({"layer0.lora_a": delta_a})

    after_a = np.asarray(target.layers[0].lora_a, dtype=np.float32)
    np.testing.assert_allclose(after_a, before_a + delta_a, rtol=1e-6)


def test_lora_weight_delta_channel_rejects_non_finite() -> None:
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )

    _, target = _clones(seed=0)
    bad = np.full_like(
        np.asarray(target.layers[0].lora_a, dtype=np.float32),
        np.inf,
    )
    channel = LoRAWeightDeltaChannel(target)
    with pytest.raises(ValueError, match=r"^S2:"):
        channel.apply({"layer0.lora_a": bad})


def test_lora_weight_delta_channel_rejects_bad_key_format() -> None:
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )

    _, target = _clones(seed=0)
    channel = LoRAWeightDeltaChannel(target)
    with pytest.raises(ValueError, match=r"^S1:"):
        channel.apply({"oops": np.zeros((2, 4), dtype=np.float32)})


def test_lora_weight_delta_channel_rejects_out_of_range_layer() -> None:
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )

    _, target = _clones(seed=0)
    channel = LoRAWeightDeltaChannel(target)
    with pytest.raises(ValueError, match=r"^S1:"):
        channel.apply({"layer99.lora_a": np.zeros((2, 4), dtype=np.float32)})
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kiki_oniric.dream.channels.weight_delta'`.

- [ ] **Step 3: Implement `LoRAWeightDeltaChannel`**

Create `kiki_oniric/dream/channels/weight_delta.py`:

```python
"""LoRA-target concrete implementation of WeightDeltaChannel (B5).

Applies a ``lora_delta`` dict (B1b/B2 output format) additively onto a
``LoRAModel``'s adapter parameters. Layer keys are ``layer<i>.lora_a``
or ``layer<i>.lora_b`` — matching the format produced by
``LoRAModel.adapter_parameters()`` and emitted by the dream-side
handlers.

``fisher_bump`` is accepted to match the ``WeightDeltaChannel``
Protocol signature but is ignored in B5 (B1b/B2 always emit
``fisher_bump=None``; Fisher consolidation is future work).

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.1
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


_VALID_ATTRS: frozenset[str] = frozenset({"lora_a", "lora_b"})


class LoRAWeightDeltaChannel:
    """Concrete ``WeightDeltaChannel`` for a ``LoRAModel`` target.

    ``apply`` is additive: ``target.lora_a += delta``. Per-key S1
    parsing ensures the delta really refers to a layer that exists.
    S2 finite guard rejects NaN / Inf deltas before materialising.
    """

    def __init__(self, target: "LoRAModel") -> None:
        self._target = target

    def apply(
        self,
        lora_delta: dict[str, NDArray[np.float32]],
        fisher_bump: dict[str, NDArray[np.float32]] | None = None,
    ) -> None:
        del fisher_bump  # B5: accepted to match Protocol, ignored
        for key, delta_arr in lora_delta.items():
            layer_idx, attr = self._parse_key(key)
            if layer_idx >= len(self._target.layers):
                raise ValueError(
                    f"S1: weight_delta key {key!r} references layer "
                    f"{layer_idx} but target has "
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
        """Parse ``layer<i>.lora_a`` / ``layer<i>.lora_b`` → (i, attr)."""
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
        if attr not in _VALID_ATTRS:
            raise ValueError(
                f"S1: invalid lora_delta attr {attr!r} in {key!r}"
            )
        return idx, attr
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py -v`
Expected: PASS — 4 tests.

- [ ] **Step 5: Full sanity (suite + mypy + ruff)**

Run: `uv run pytest -q` — all pass.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/dream/channels/weight_delta.py tests/unit/test_apply_channel_outputs.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/dream/channels/weight_delta.py tests/unit/test_apply_channel_outputs.py
git commit -m "feat: LoRAWeightDeltaChannel additive apply"
```

Commit body:
```
feat: LoRAWeightDeltaChannel additive apply

B5 / issue #15. First concrete channel impl — additively applies
a lora_delta dict (B1b/B2 emit format) onto a LoRAModel target.
Per-key S1 parsing rejects unknown layer indices and bad attrs;
S2 finite guard rejects NaN/Inf before materialising. fisher_bump
is accepted to match the Protocol but ignored (Fisher is future
work).
```

---

## Task 2: `_apply_topology_op` helper + `LoRAHierarchyChangeChannel`

**Files:**
- Create: `kiki_oniric/dream/channels/hierarchy_change.py`
- Test: `tests/unit/test_apply_channel_outputs.py` (append)

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/test_apply_channel_outputs.py`:

```python
def test_hierarchy_channel_reroute_swaps_layers() -> None:
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )

    _, target = _clones(seed=0)
    id0_before = id(target.layers[0])
    id1_before = id(target.layers[1])
    channel = LoRAHierarchyChangeChannel(target)
    channel.apply_diff(
        [("reroute", {"swap_indices": (0, 1), "model_sha256_post": "0" * 64})],
    )
    assert id(target.layers[0]) == id1_before
    assert id(target.layers[1]) == id0_before


def test_hierarchy_channel_add_reconstructs_layer_bit_exact() -> None:
    """add via channel must match a fresh LoRALinear(seed) bit-exact."""
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )

    _, target = _clones(seed=0)
    payload = {
        "index": len(target.layers),
        "in_features": 3,
        "out_features": 5,
        "rank": 2,
        "alpha": 4.0,
        "seed": 12345,
        "model_sha256_post": "0" * 64,
    }
    channel = LoRAHierarchyChangeChannel(target)
    channel.apply_diff([("add", payload)])
    inserted = target.layers[-1]
    rebuilt = LoRALinear(
        in_features=3, out_features=5, rank=2, alpha=4.0,
        key=mx.random.key(12345),
    )
    np.testing.assert_array_equal(
        np.asarray(inserted.base_weight), np.asarray(rebuilt.base_weight),
    )
    np.testing.assert_array_equal(
        np.asarray(inserted.lora_a), np.asarray(rebuilt.lora_a),
    )
    np.testing.assert_array_equal(
        np.asarray(inserted.lora_b), np.asarray(rebuilt.lora_b),
    )


def test_hierarchy_channel_remove_pops_layer() -> None:
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )

    _, target = _clones(seed=0)
    pre_len = len(target.layers)
    payload = {
        "index": 0,
        "snapshot": {},  # B5 channel doesn't read the snapshot
        "model_sha256_post": "0" * 64,
    }
    channel = LoRAHierarchyChangeChannel(target)
    channel.apply_diff([("remove", payload)])
    assert len(target.layers) == pre_len - 1


def test_hierarchy_channel_rejects_unknown_op() -> None:
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )

    _, target = _clones(seed=0)
    channel = LoRAHierarchyChangeChannel(target)
    with pytest.raises(ValueError, match=r"^S3:"):
        channel.apply_diff([("bogus", {"model_sha256_post": "0" * 64})])
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py -k hierarchy -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kiki_oniric.dream.channels.hierarchy_change'`.

- [ ] **Step 3: Implement the helper + channel**

Create `kiki_oniric/dream/channels/hierarchy_change.py`:

```python
"""LoRA-target concrete implementation of HierarchyChangeChannel (B5).

Replays a ``TopologyDiff``'s ``diff`` entries onto a ``LoRAModel``
via a shared ``_apply_topology_op`` helper. ``TopologyDiff
.__post_init__`` (shipped in B3) already validated each entry —
this channel performs no re-validation, only mutation.

The shared helper is intentionally module-level so a future
refactor of ``restructure_lora_handler`` (B3) can delegate to it.
B5 itself does not refactor B3.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.1
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


def _apply_topology_op(
    model: "LoRAModel",
    op: str,
    payload: dict[str, object],
) -> None:
    """Replay one ``(op, payload)`` entry on ``model``.

    Unknown ops raise ``ValueError`` with the literal ``"S3:"`` tag
    (defence-in-depth; B3's ``TopologyDiff.__post_init__`` already
    rejects them at construction time).
    """
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRALinear

    if op == "add":
        new_layer = LoRALinear(
            in_features=int(payload["in_features"]),  # type: ignore[arg-type]
            out_features=int(payload["out_features"]),  # type: ignore[arg-type]
            rank=int(payload["rank"]),  # type: ignore[arg-type]
            alpha=float(payload["alpha"]),  # type: ignore[arg-type]
            key=mx.random.key(int(payload["seed"])),  # type: ignore[arg-type]
        )
        model.layers.insert(int(payload["index"]), new_layer)  # type: ignore[arg-type]
    elif op == "remove":
        model.layers.pop(int(payload["index"]))  # type: ignore[arg-type]
    elif op == "reroute":
        swap = payload["swap_indices"]
        i, j = int(swap[0]), int(swap[1])  # type: ignore[index]
        model.layers[i], model.layers[j] = model.layers[j], model.layers[i]
    else:
        raise ValueError(f"S3: unknown topology op {op!r}")


class LoRAHierarchyChangeChannel:
    """Concrete ``HierarchyChangeChannel`` for a ``LoRAModel`` target.

    ``apply_diff`` walks the diff in order and applies each entry via
    ``_apply_topology_op``. The dream-side handler's payload format
    is the contract — ``add`` must carry ``seed`` so the channel can
    reconstruct the layer bit-exactly via ``mx.random.key(seed)``
    (R1).
    """

    def __init__(self, target: "LoRAModel") -> None:
        self._target = target

    def apply_diff(
        self, diff: list[tuple[str, dict[str, object]]],
    ) -> None:
        for op, payload in diff:
            _apply_topology_op(self._target, op, payload)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py -k hierarchy -v`
Expected: PASS — 4 tests.

- [ ] **Step 5: Full sanity**

Run: `uv run pytest -q` — all pass.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/dream/channels/hierarchy_change.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/dream/channels/hierarchy_change.py tests/unit/test_apply_channel_outputs.py
git commit -m "feat: LoRAHierarchyChangeChannel + shared helper"
```

Commit body:
```
feat: LoRAHierarchyChangeChannel + shared helper

B5 / issue #15. Concrete HierarchyChangeChannel impl on a
LoRAModel target; replays each (op, payload) entry from a
TopologyDiff via the shared _apply_topology_op helper. add
reconstructs the inserted LoRALinear via mx.random.key
(payload["seed"]) so the awake-side model matches the dream-side
bit-exactly (R1). The helper is module-level so a future B3
refactor can delegate to it. B5 itself does not refactor B3.
```

---

## Task 3: `LatentSampleQueue`

**Files:**
- Create: `kiki_oniric/dream/channels/latent_sample.py`
- Test: `tests/unit/test_apply_channel_outputs.py` (append)

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/test_apply_channel_outputs.py`:

```python
def test_latent_queue_enqueue_dequeue_round_trip() -> None:
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    q = LatentSampleQueue()
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    q.enqueue("species-a", vec, "prov-1")
    assert len(q) == 1
    out = q.dequeue()
    assert out is not None
    assert out["species"] == "species-a"
    assert out["provenance"] == "prov-1"
    np.testing.assert_array_equal(out["latent_vector"], vec)


def test_latent_queue_dequeue_empty_returns_none() -> None:
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    q = LatentSampleQueue()
    assert q.dequeue() is None


def test_latent_queue_rejects_non_finite() -> None:
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    q = LatentSampleQueue()
    bad = np.array([np.inf], dtype=np.float32)
    with pytest.raises(ValueError, match=r"^S2:"):
        q.enqueue("x", bad, "p")


def test_latent_queue_capacity_drops_oldest() -> None:
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    q = LatentSampleQueue(capacity=2)
    for i in range(3):
        q.enqueue("s", np.array([float(i)], dtype=np.float32), f"p{i}")
    assert len(q) == 2
    out = q.dequeue()
    assert out is not None
    assert out["provenance"] == "p1"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py -k latent_queue -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kiki_oniric.dream.channels.latent_sample'`.

- [ ] **Step 3: Implement `LatentSampleQueue`**

Create `kiki_oniric/dream/channels/latent_sample.py`:

```python
"""Substrate-agnostic FIFO queue implementing LatentSampleChannel (B5).

A bounded or unbounded FIFO of latent samples emitted by the
recombine op (B4). The producer-side schema mirrors B0's
``LatentSample`` value type; the consumer dequeues dicts and can
shape them back into ``LatentSample`` instances if desired.

S2 finite check at enqueue protects downstream consumers from
NaN / Inf propagation.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.1
"""
from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray


class LatentSampleQueue:
    """Concrete ``LatentSampleChannel`` — FIFO queue, optional capacity.

    When ``capacity`` is set and the queue is full, the oldest item
    is evicted on enqueue (``collections.deque`` semantics).
    """

    def __init__(self, capacity: int | None = None) -> None:
        self._queue: deque[dict] = deque(maxlen=capacity)

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

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py -k latent_queue -v`
Expected: PASS — 4 tests.

- [ ] **Step 5: Full sanity**

Run: `uv run pytest -q` — all pass.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/dream/channels/latent_sample.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/dream/channels/latent_sample.py tests/unit/test_apply_channel_outputs.py
git commit -m "feat: LatentSampleQueue FIFO channel impl"
```

Commit body:
```
feat: LatentSampleQueue FIFO channel impl

B5 / issue #15. Substrate-agnostic concrete LatentSampleChannel
backed by collections.deque with optional capacity. S2 finite
check at enqueue. Consumes the B4 recombine_real_handler emit
format and exposes dequeue() returning dict | None.
```

---

## Task 4: `set_prior` alias on `AttentionPriorChannel`

**Files:**
- Modify: `kiki_oniric/dream/channels/attention_prior.py`
- Test: `tests/unit/test_apply_channel_outputs.py` (append)

- [ ] **Step 1: Append failing test**

Append to `tests/unit/test_apply_channel_outputs.py`:

```python
def test_attention_prior_set_prior_is_alias_of_emit() -> None:
    from kiki_oniric.dream.channels.attention_prior import (
        AttentionPriorChannel,
    )

    ch = AttentionPriorChannel()
    prior = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    ch.set_prior(prior)
    got = ch.get_prior()
    assert got is not None
    np.testing.assert_array_equal(got, prior)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py -k set_prior -v`
Expected: FAIL — `AttributeError: 'AttentionPriorChannel' object has no attribute 'set_prior'`.

- [ ] **Step 3: Add the alias**

Append to `kiki_oniric/dream/channels/attention_prior.py` inside the `AttentionPriorChannel` class (immediately after `emit`):

```python
    def set_prior(self, prior: NDArray) -> None:
        """Alias of :meth:`emit` — matches the ``AttentionPriorChannel``
        Protocol vocabulary (Protocol uses ``set_prior``; the cycle-2
        class shipped with ``emit``)."""
        self.emit(prior)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py -k set_prior -v`
Expected: PASS — 1 test.

- [ ] **Step 5: Full sanity**

Run: `uv run pytest -q` — all pass.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/dream/channels/attention_prior.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/dream/channels/attention_prior.py tests/unit/test_apply_channel_outputs.py
git commit -m "feat: set_prior alias on AttentionPriorChannel"
```

Commit body:
```
feat: set_prior alias on AttentionPriorChannel

B5 / issue #15. Adds set_prior(prior) as a thin alias of emit()
on the cycle-2 AttentionPriorChannel so the channel matches the
AttentionPriorChannel Protocol vocabulary. apply_channel_outputs
calls set_prior; legacy callers using emit are unaffected.
```

---

## Task 5: `apply_channel_outputs` dispatch function

**Files:**
- Modify: `kiki_oniric/consolidate.py`
- Test: `tests/unit/test_apply_channel_outputs.py` (append)

- [ ] **Step 1: Append the dispatch tests**

Append to `tests/unit/test_apply_channel_outputs.py`:

```python
def _make_log_with_one_output(output) -> list:
    """Build a 1-entry EpisodeLogEntry log carrying ``output`` only."""
    from kiki_oniric.dream.episode import Operation
    from kiki_oniric.dream.runtime import EpisodeLogEntry

    return [
        EpisodeLogEntry(
            episode_id="de-test",
            operations_executed=(Operation.REPLAY,),
            completed=True,
            error=None,
            channel_outputs=(output,),
        ),
    ]


def test_apply_dispatches_weight_update() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    _, target = _clones(seed=0)
    delta_a = np.ones(
        np.asarray(target.layers[0].lora_a).shape, dtype=np.float32,
    ) * 0.25
    log = _make_log_with_one_output(WeightUpdate(lora_delta={"layer0.lora_a": delta_a}))
    count = apply_channel_outputs(
        log,
        weight_channel=LoRAWeightDeltaChannel(target),
        hierarchy_channel=LoRAHierarchyChangeChannel(target),
        latent_channel=LatentSampleQueue(),
    )
    assert count == 1


def test_apply_dispatches_topology_diff_reroute() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import TopologyDiff
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    _, target = _clones(seed=0)
    entry = ("reroute", {"swap_indices": (0, 1), "model_sha256_post": "0" * 64})
    log = _make_log_with_one_output(TopologyDiff(diff=(entry,)))
    apply_channel_outputs(
        log,
        weight_channel=LoRAWeightDeltaChannel(target),
        hierarchy_channel=LoRAHierarchyChangeChannel(target),
        latent_channel=LatentSampleQueue(),
    )
    # After one reroute, the model is mutated; here we just smoke-check
    # via length. End-to-end bit-exact behaviour is covered in Task 6.
    assert len(target.layers) == 2


def test_apply_dispatches_latent_sample() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import LatentSample
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    _, target = _clones(seed=0)
    q = LatentSampleQueue()
    log = _make_log_with_one_output(
        LatentSample(
            species="default",
            latent_vector=np.array([0.1, 0.2], dtype=np.float32),
            provenance="recombine:de=test:ep=0:seed=0",
        ),
    )
    apply_channel_outputs(
        log,
        weight_channel=LoRAWeightDeltaChannel(target),
        hierarchy_channel=LoRAHierarchyChangeChannel(target),
        latent_channel=q,
    )
    assert len(q) == 1


def test_apply_dispatches_attention_prior() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import AttentionPrior
    from kiki_oniric.dream.channels.attention_prior import (
        AttentionPriorChannel,
    )
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    _, target = _clones(seed=0)
    att = AttentionPriorChannel()
    prior = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    log = _make_log_with_one_output(AttentionPrior(prior=prior))
    apply_channel_outputs(
        log,
        weight_channel=LoRAWeightDeltaChannel(target),
        hierarchy_channel=LoRAHierarchyChangeChannel(target),
        latent_channel=LatentSampleQueue(),
        attention_channel=att,
    )
    got = att.get_prior()
    assert got is not None
    np.testing.assert_array_equal(got, prior)


def test_apply_attention_required_when_emitted() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import AttentionPrior
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    _, target = _clones(seed=0)
    log = _make_log_with_one_output(
        AttentionPrior(prior=np.array([0.1], dtype=np.float32)),
    )
    with pytest.raises(ValueError, match="attention_channel"):
        apply_channel_outputs(
            log,
            weight_channel=LoRAWeightDeltaChannel(target),
            hierarchy_channel=LoRAHierarchyChangeChannel(target),
            latent_channel=LatentSampleQueue(),
        )


def test_apply_skips_none_entries() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue
    from kiki_oniric.dream.episode import Operation
    from kiki_oniric.dream.runtime import EpisodeLogEntry

    _, target = _clones(seed=0)
    delta_a = np.zeros(
        np.asarray(target.layers[0].lora_a).shape, dtype=np.float32,
    )
    log = [
        EpisodeLogEntry(
            episode_id="de-skip",
            operations_executed=(Operation.REPLAY, Operation.DOWNSCALE, Operation.RECOMBINE),
            completed=True,
            error=None,
            channel_outputs=(
                None,
                WeightUpdate(lora_delta={"layer0.lora_a": delta_a}),
                None,
            ),
        ),
    ]
    count = apply_channel_outputs(
        log,
        weight_channel=LoRAWeightDeltaChannel(target),
        hierarchy_channel=LoRAHierarchyChangeChannel(target),
        latent_channel=LatentSampleQueue(),
    )
    assert count == 1


def test_apply_empty_log_returns_zero() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    _, target = _clones(seed=0)
    count = apply_channel_outputs(
        [],
        weight_channel=LoRAWeightDeltaChannel(target),
        hierarchy_channel=LoRAHierarchyChangeChannel(target),
        latent_channel=LatentSampleQueue(),
    )
    assert count == 0


def test_apply_rejects_unknown_output_type() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    _, target = _clones(seed=0)
    # A bare object — not in the ChannelOutput union.
    bogus = object()
    log = _make_log_with_one_output(bogus)
    with pytest.raises(TypeError, match="unknown"):
        apply_channel_outputs(
            log,
            weight_channel=LoRAWeightDeltaChannel(target),
            hierarchy_channel=LoRAHierarchyChangeChannel(target),
            latent_channel=LatentSampleQueue(),
        )
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py -k apply -v`
Expected: FAIL — `ImportError: cannot import name 'apply_channel_outputs' from 'kiki_oniric.consolidate'`.

- [ ] **Step 3: Append `apply_channel_outputs` to `kiki_oniric/consolidate.py`**

Open `kiki_oniric/consolidate.py`. **Add** the imports below (alongside the existing imports) — keep the existing imports intact:

```python
from kiki_oniric.dream.channels import (
    AttentionPrior,
    LatentSample,
    TopologyDiff,
    WeightUpdate,
)
from kiki_oniric.dream.runtime import EpisodeLogEntry
```

Then **append** at the end of the file (after `consolidate`), before the existing `__all__` if it sits at the bottom — otherwise just at end-of-file:

```python
def apply_channel_outputs(
    log: list[EpisodeLogEntry],
    *,
    weight_channel,
    hierarchy_channel,
    latent_channel,
    attention_channel=None,
) -> int:
    """Dispatch every non-``None`` channel output in ``log`` to the
    matching concrete channel and return the count.

    Parameters
    ----------
    log
        The ``DreamRuntime.log`` (list of ``EpisodeLogEntry``) produced
        by running episodes through a runtime registered with the
        B1b/B2/B3/B4 emitting handlers.
    weight_channel
        A ``WeightDeltaChannel`` implementation (e.g.
        ``LoRAWeightDeltaChannel``) that consumes ``WeightUpdate``
        outputs.
    hierarchy_channel
        A ``HierarchyChangeChannel`` implementation that consumes
        ``TopologyDiff`` outputs.
    latent_channel
        A ``LatentSampleChannel`` implementation that consumes
        ``LatentSample`` outputs.
    attention_channel
        Optional ``AttentionPriorChannel`` — required only if the
        log carries an ``AttentionPrior``; otherwise pass ``None``.

    Returns
    -------
    int
        Number of channel outputs dispatched (``None`` entries are
        skipped).

    Raises
    ------
    TypeError
        On a non-``None`` log entry whose type isn't in the
        ``ChannelOutput`` union.
    ValueError
        If an ``AttentionPrior`` is encountered but
        ``attention_channel is None``.
    """
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

If `__all__` is present in `consolidate.py`, add `"apply_channel_outputs"` to it. Today `consolidate.py:58` says `__all__ = ["consolidate", "Profile"]` — change to `__all__ = ["consolidate", "Profile", "apply_channel_outputs"]`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py -k apply -v`
Expected: PASS — 8 dispatch tests.

- [ ] **Step 5: Full sanity**

Run: `uv run pytest -q` — all pass.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/consolidate.py tests/unit/test_apply_channel_outputs.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/consolidate.py tests/unit/test_apply_channel_outputs.py
git commit -m "feat: apply_channel_outputs dispatch function"
```

Commit body:
```
feat: apply_channel_outputs dispatch function

B5 / issue #15. Free function in consolidate.py that walks
EpisodeLogEntry.channel_outputs, skips None entries, and
dispatches by isinstance to the matching channel. WeightUpdate
to weight_channel.apply, TopologyDiff to apply_diff, LatentSample
to enqueue, AttentionPrior to set_prior. Unknown types raise
TypeError; missing attention_channel for an AttentionPrior raises
ValueError. consolidate()'s nerve-wml-facing signature stays
untouched.
```

---

## Task 6: End-to-end clone-bit-equality test

**Files:**
- Modify: `tests/unit/test_apply_channel_outputs.py` (append)

- [ ] **Step 1: Append the end-to-end test**

Append:

```python
def test_end_to_end_replay_dream_to_awake_bit_equal() -> None:
    """Run replay_lora_handler on dream-model, apply log to awake-model,
    assert the two models are bit-equal."""
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue
    from kiki_oniric.dream.episode import (
        BudgetCap, DreamEpisode, EpisodeTrigger, Operation, OutputChannel,
    )
    from kiki_oniric.dream.operations.replay_real import (
        ReplayRealState, replay_lora_handler,
    )
    from kiki_oniric.dream.runtime import DreamRuntime

    dream, awake = _clones(seed=0)
    state = ReplayRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.REPLAY,
        replay_lora_handler(state, model=dream, lr=0.05),
    )
    episode = DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={
            "beta_records": [
                {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
                {"x": [0.5, 0.6, 0.7, 0.8], "y": [0.0, 1.0]},
            ],
        },
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-e2e-replay",
    )
    runtime.execute(episode)

    # Sanity: handler did mutate the dream model — at least one adapter
    # array differs from awake (which has not been touched yet).
    diff_before_apply = not np.array_equal(
        np.asarray(dream.layers[0].lora_b), np.asarray(awake.layers[0].lora_b),
    )
    assert diff_before_apply

    apply_channel_outputs(
        runtime.log,
        weight_channel=LoRAWeightDeltaChannel(awake),
        hierarchy_channel=LoRAHierarchyChangeChannel(awake),
        latent_channel=LatentSampleQueue(),
    )

    # After apply, dream and awake match bit-for-bit.
    _assert_lora_models_equal(dream, awake)
```

- [ ] **Step 2: Run to verify it passes**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py::test_end_to_end_replay_dream_to_awake_bit_equal -v`
Expected: PASS.

- [ ] **Step 3: Full sanity (all B5 tests + suite)**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py -v`
Expected: PASS — all B5 tests (4 weight + 4 hierarchy + 4 latent + 1 set_prior + 8 dispatch + 1 e2e = 22 in total, ≥ 12 spec target).

Run: `uv run pytest -q` — full suite green.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check .` — clean.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_apply_channel_outputs.py
git commit -m "test: end-to-end dream-to-awake bit equality"
```

Commit body:
```
test: end-to-end dream-to-awake bit equality

B5 / issue #15. Final acceptance test for the awake to dream
loop closure: clone two LoRAModels at seed=K, run
replay_lora_handler on the dream side, call apply_channel_outputs
with the awake side as the weight_channel target, then assert
both models match bit-for-bit. Demonstrates that the dream-side
emitted lora_delta really does carry enough information to
reconstruct the dream-side mutation on a fresh awake-side clone.
```

---

## Task 7: Documentation and DualVer sync

**Files:**
- Modify: `CHANGELOG.md`, `pyproject.toml`, `uv.lock`
- Modify: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/...`

- [ ] **Step 1: Add the CHANGELOG entry**

Insert at the top of the `CHANGELOG.md` body, immediately above the existing `[C-v0.19.0+PARTIAL]` entry:

```markdown
## [C-v0.20.0+PARTIAL] — 2026-05-20 — channel apply closes the loop (B5)

### Formal axis (FC) — MINOR (v0.19.0 → v0.20.0)

- **Three new concrete channels** in `kiki_oniric/dream/channels/`:
  - `LoRAWeightDeltaChannel` (`weight_delta.py`) — additively
    applies a `lora_delta` dict onto a `LoRAModel` target. Per-key
    `"S1:"` parsing rejects unknown layer indices and bad attrs;
    `"S2:"` finite guard rejects NaN/Inf on the new tensor.
    `fisher_bump=None` accepted to match the `WeightDeltaChannel`
    Protocol; ignored in B5 (Fisher is future work).
  - `LoRAHierarchyChangeChannel` (`hierarchy_change.py`) — replays
    every `TopologyDiff` entry via the new module-level
    `_apply_topology_op(model, op, payload)` helper. `add` payloads
    carry `seed`, which the channel passes to `mx.random.key` so
    the reconstructed `LoRALinear` matches the dream-side bit-
    exactly (R1).
  - `LatentSampleQueue` (`latent_sample.py`) — substrate-agnostic
    FIFO `collections.deque` with optional capacity. `S2:` finite
    check on enqueue.
- **`AttentionPriorChannel.set_prior`** — thin alias of `emit` so
  the cycle-2 class matches the `AttentionPriorChannel` Protocol
  vocabulary used by `apply_channel_outputs`.
- **`apply_channel_outputs`** in `kiki_oniric/consolidate.py` — a
  free function that walks `EpisodeLogEntry.channel_outputs`,
  skips `None`, and dispatches by `isinstance` to the matching
  channel. Returns the number of outputs dispatched. Raises
  `TypeError` on an unknown `ChannelOutput` type; raises
  `ValueError` if an `AttentionPrior` is encountered but
  `attention_channel is None`. `consolidate()`'s nerve-wml-facing
  signature is untouched.
- Sub-project B5 of issue #15 — closes the awake↔dream loop. Tests
  use a dream/awake `LoRAModel` clone pair (same `seed`); after
  running an emitting handler on the dream side and
  `apply_channel_outputs` on the awake side, the two models match
  bit-for-bit (R1).

### Empirical axis (EC) — UNCHANGED (PARTIAL)

- No new substrate, axiom, or empirical claim. EC stays
  `+PARTIAL`.

### Packaging

- `pyproject.toml` version bumped `0.17.0 → 0.18.0`.

### Deferred

- Profile wiring (`p_min` / `p_equ` / `p_max` continue to use
  their cycle-3 channel wiring). A future **B6** sub-project
  integrates `apply_channel_outputs` into the profile-driven
  consolidation flow.
- Refactoring `restructure_lora_handler` to delegate to
  `_apply_topology_op`. The helper is intentionally module-level
  so B3 can delegate later, but B5 does not change B3.
- Fisher bump handling on `WeightDeltaChannel.apply`.
```

- [ ] **Step 2: Bump `pyproject.toml`**

Change `version = "0.17.0"` to `version = "0.18.0"`.

- [ ] **Step 3: Sync the lockfile**

Run: `uv lock`
Expected: `uv.lock` updates the `dreamofkiki` pin to `0.18.0`.

- [ ] **Step 4: Update the framework-C spec (EN)**

In `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §4.1 (channels), at the end of the channel-definitions block (after the existing four channel descriptions), append:

```markdown
As of B5 (issue #15), the four channel Protocols have concrete
LoRA-target implementations: `LoRAWeightDeltaChannel`,
`LoRAHierarchyChangeChannel`, `LatentSampleQueue` and the
existing `AttentionPriorChannel` (with a new `set_prior` alias
of `emit`). The free function
`kiki_oniric.consolidate.apply_channel_outputs(log, …)` walks
the runtime log and dispatches each `ChannelOutput` to the
matching channel — closing the awake↔dream loop. Profile-level
wiring is deferred to **B6**.
```

- [ ] **Step 5: Update the framework-C spec (FR mirror)**

In `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` §4.1, at the matching location, append:

```markdown
Depuis B5 (issue #15), les quatre Protocols de canaux ont des
implémentations concrètes ciblant le substrat LoRA :
`LoRAWeightDeltaChannel`, `LoRAHierarchyChangeChannel`,
`LatentSampleQueue` et l'`AttentionPriorChannel` existant (avec
un nouvel alias `set_prior` de `emit`). La fonction libre
`kiki_oniric.consolidate.apply_channel_outputs(log, …)` parcourt
le journal du runtime et répartit chaque `ChannelOutput` vers le
canal correspondant — fermant la boucle éveil↔rêve. Le câblage
au niveau des profils est différé à **B6**.
```

- [ ] **Step 6: Verify**

Run: `uv run pytest -q` — all pass (docs/version change, no code touched).
Run: `uv run mypy harness tests` — `Success`.

- [ ] **Step 7: Commit**

```bash
git add CHANGELOG.md pyproject.toml uv.lock docs/specs/2026-04-17-dreamofkiki-framework-C-design.md docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md
git commit -m "docs: sync spec for B5 channel apply"
```

Commit body:
```
docs: sync spec for B5 channel apply

B5 / issue #15. Add the C-v0.20.0+PARTIAL changelog entry, note
in framework-C spec section 4.1 (EN + FR) that the four channel
Protocols now have concrete LoRA-target implementations and
that apply_channel_outputs closes the awake to dream loop, and
bump the package version to 0.18.0.
```

If `tests/reproducibility/golden_hashes.json` shows modified (R1 metadata drift), do NOT stage it.

---

## Task 8: Final verification

**Files:** none — verification only.

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -q`
Expected: all pass, 0 failures, coverage gate met.

- [ ] **Step 2: Type check**

Run: `uv run mypy harness tests`
Expected: `Success: no issues found`.

- [ ] **Step 3: Lint**

Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 4: Confirm working tree is clean**

Run: `git status --short`
Expected: empty (if `tests/reproducibility/golden_hashes.json` shows modified, restore it with `git checkout -- tests/reproducibility/golden_hashes.json`).

- [ ] **Step 5: Update issue #15**

Comment on issue #15: B5 complete — `apply_channel_outputs` plus three concrete LoRA-target channels close the awake↔dream loop. The dream-side handler emits a delta/diff/sample; the channel side replays it onto an awake-model clone (same `seed`) to bit-exact equality. Profile wiring deferred to B6.

---

## Self-Review

- **Spec coverage:**
  - `LoRAWeightDeltaChannel` (Task 1) ✓ — additive apply, S1 key parsing, S2 finite guard, fisher_bump ignored.
  - `LoRAHierarchyChangeChannel` + `_apply_topology_op` (Task 2) ✓ — replay add/remove/reroute; `add` reconstructs via `mx.random.key(payload["seed"])`.
  - `LatentSampleQueue` (Task 3) ✓ — FIFO, optional capacity, S2 finite check.
  - `set_prior` alias on `AttentionPriorChannel` (Task 4) ✓.
  - `apply_channel_outputs` dispatch (Task 5) ✓ — `isinstance` dispatch, `None` skip, `TypeError` on unknown, `ValueError` when `attention_channel` required and missing.
  - End-to-end clone-bit-equality (Task 6) ✓ — replay handler + awake apply → bit-equal.
  - CHANGELOG `[C-v0.20.0+PARTIAL]` + spec §4.1 EN+FR + version `0.18.0` (Task 7) ✓.
  - Final verification (Task 8) ✓.
  - No profile wiring; no B3 refactor — deferred as documented.
- **Placeholder scan:** no TBD/TODO; every code step has complete code.
- **Type consistency:**
  - `LoRAWeightDeltaChannel(target: LoRAModel)` referenced by every dispatch test under the same name.
  - `LoRAHierarchyChangeChannel(target: LoRAModel)` same.
  - `LatentSampleQueue(capacity=None)` same — Task 3 capacity test uses the same kwarg name.
  - `apply_channel_outputs(log, *, weight_channel, hierarchy_channel, latent_channel, attention_channel=None) -> int` — exact same signature used by every test that calls it (Tasks 5, 6).
  - `_apply_topology_op(model, op, payload)` (module-level in Task 2) — not directly imported in any test but reachable as the implementation under the channel.
  - Channel-output value types (`WeightUpdate`, `TopologyDiff`, `LatentSample`, `AttentionPrior`) — accessed through `kiki_oniric.dream.channels` (B0 module), consistent with B0..B4 usage.
  - `EpisodeLogEntry(episode_id, operations_executed, completed, error, channel_outputs)` — Task 5 helper `_make_log_with_one_output` mirrors the dataclass fields exactly (B0 contract).
- **Test count:** 4 (Task 1) + 4 (Task 2) + 4 (Task 3) + 1 (Task 4) + 8 (Task 5) + 1 (Task 6) = **22 tests**, exceeding the ~12 spec target.
- **Inter-task ordering:** Tasks 1-4 are independent channel impls (any order works). Task 5 dispatches to all three of them, so it depends on 1-3 (and on 4 for the AttentionPrior case). Task 6 is the integration test; it depends on Tasks 1-5. Task 7 is documentation; depends on Tasks 1-6. The plan ships them in dependency order.
