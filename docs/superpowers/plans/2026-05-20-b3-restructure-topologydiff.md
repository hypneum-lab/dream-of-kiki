# B3 — `restructure` emits a `TopologyDiff` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `restructure` operation mutate a `LoRAModel`'s adapter stack (`{add, remove, reroute}`) and return a real channel-3 `TopologyDiff` carrying an executable per-op payload + SHA-256 model fingerprint; harden `TopologyDiff` with a `__post_init__` S3 guard.

**Architecture:** Two changes ship in sequence. **(1)** `TopologyDiff.__post_init__` in `kiki_oniric/dream/channels/__init__.py` enforces structural S3 validity at construction (vocab, per-op required keys, finite snapshot arrays, 64-hex `model_sha256_post`). **(2)** A new `restructure_lora_handler` factory in `kiki_oniric/dream/operations/restructure_real.py` validates every op in `input_slice["topo_ops"]` *before* any mutation, mutates `model.layers: list[LoRALinear]` in place (full `{add, remove, reroute}` vocab; INSS soft cap on `add`; `remove` snapshots all four arrays for undo), hashes the model after each applied op, and emits `TopologyDiff(diff=tuple(...))` — or `None` if zero ops are applied (S1 no-op).

**Tech Stack:** Python 3.12, `uv`, MLX (`mlx.core` / `mlx.nn`), numpy, `hashlib`, pytest, mypy.

**Spec:** `docs/superpowers/specs/2026-05-20-b3-restructure-topologydiff-design.md`

**LoRA fact (test design):** `LoRALinear` keeps a *frozen* `base_weight` (and optional `bias`) alongside *trainable* `lora_a` / `lora_b`. The `remove` snapshot includes the frozen `base_weight` (so B5 can rebuild the layer exactly) and the optional `bias` (only if `use_bias`). The `model_sha256` digest includes all of them.

**Migration note:** The existing test `tests/unit/test_channels.py::test_topology_diff_holds_tuple` constructs `TopologyDiff(diff=(("add_node", {"id": "n1"}),))` — that diff will be rejected by the new `__post_init__` (op `"add_node"` not in vocab, missing `model_sha256_post`). Task 1 updates this test to use a valid shape.

---

## File Structure

- **Modify** `kiki_oniric/dream/channels/__init__.py` — add `_VALID_TOPO_OPS`, `_SHA256_HEX_LEN`, and `TopologyDiff.__post_init__`. Replace the existing `# No __post_init__ ...` comment.
- **Modify** `tests/unit/test_channels.py` — fix the now-invalid `test_topology_diff_holds_tuple` constructor; add negative cases for the new `__post_init__`.
- **Modify** `kiki_oniric/dream/operations/restructure_real.py` — extend `RestructureRealState` with 4 backwards-compat fields; add `_model_sha256`, `_flop_estimate_restructure`, `restructure_lora_handler`.
- **Create** `tests/unit/test_restructure_lora.py` — 14 end-to-end handler tests via `DreamRuntime`.
- **Modify** `CHANGELOG.md`, `pyproject.toml`, `uv.lock` — `[C-v0.18.0+PARTIAL]`, version `0.15.0 → 0.16.0`.
- **Modify** `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` — §4.2 note for `restructure`.

---

## Task 1: `TopologyDiff.__post_init__` S3 guard

**Files:**
- Modify: `kiki_oniric/dream/channels/__init__.py`
- Modify: `tests/unit/test_channels.py`

- [ ] **Step 1: Write the failing tests**

Replace `test_topology_diff_holds_tuple` in `tests/unit/test_channels.py` with the valid-shape constructor and add the negative cases. The replaced test plus the new ones look like this — locate the existing `test_topology_diff_holds_tuple` and substitute the block below for it; append the rest at the end of the file (before any `ChannelOutput` membership test):

```python
_VALID_SHA = "0" * 64


def _reroute_entry() -> tuple[str, dict[str, object]]:
    return (
        "reroute",
        {"swap_indices": (0, 1), "model_sha256_post": _VALID_SHA},
    )


def test_topology_diff_accepts_valid_reroute() -> None:
    td = TopologyDiff(diff=(_reroute_entry(),))
    assert td.diff[0][0] == "reroute"


def test_topology_diff_rejects_unknown_op() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(("add_node", {"model_sha256_post": _VALID_SHA}),))


def test_topology_diff_rejects_non_tuple_entry() -> None:
    bad = (["reroute", {"swap_indices": (0, 1), "model_sha256_post": _VALID_SHA}],)
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=bad)  # type: ignore[arg-type]


def test_topology_diff_rejects_non_dict_payload() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(("reroute", "not-a-dict"),))  # type: ignore[arg-type]


def test_topology_diff_rejects_missing_sha() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(("reroute", {"swap_indices": (0, 1)}),))


def test_topology_diff_rejects_short_sha() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(
            ("reroute", {"swap_indices": (0, 1), "model_sha256_post": "abc"}),
        ))


def test_topology_diff_rejects_add_missing_keys() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(
            ("add", {"index": 0, "model_sha256_post": _VALID_SHA}),
        ))


def test_topology_diff_rejects_add_non_positive_rank() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(
            ("add", {
                "index": 0,
                "in_features": 4,
                "out_features": 8,
                "rank": 0,
                "alpha": 4.0,
                "seed": 0,
                "model_sha256_post": _VALID_SHA,
            }),
        ))


def test_topology_diff_rejects_remove_missing_snapshot() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(
            ("remove", {"index": 0, "model_sha256_post": _VALID_SHA}),
        ))


def test_topology_diff_rejects_remove_non_finite_snapshot() -> None:
    bad_snap: dict[str, object] = {
        "base_weight": np.array([np.inf], dtype=np.float32),
        "lora_a": np.zeros(1, dtype=np.float32),
        "lora_b": np.zeros(1, dtype=np.float32),
        "bias": None,
        "in_features": 4,
        "out_features": 8,
        "rank": 2,
        "alpha": 4.0,
    }
    with pytest.raises(ValueError, match="S2"):
        TopologyDiff(diff=(
            ("remove", {
                "index": 0,
                "snapshot": bad_snap,
                "model_sha256_post": _VALID_SHA,
            }),
        ))


def test_topology_diff_rejects_reroute_bad_swap_indices() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(
            ("reroute", {
                "swap_indices": (0, 1, 2),
                "model_sha256_post": _VALID_SHA,
            }),
        ))
```

- [ ] **Step 2: Run the new tests — expect failure**

Run: `uv run pytest tests/unit/test_channels.py -v`
Expected: the new tests **fail** (`TopologyDiff` still has no `__post_init__`); `test_topology_diff_accepts_valid_reroute` may pass trivially.

- [ ] **Step 3: Implement `__post_init__`**

In `kiki_oniric/dream/channels/__init__.py`, add the constants below the existing imports and replace the `TopologyDiff` block. Locate this section near the file's middle:

```python
@dataclass(frozen=True)
class TopologyDiff:
    """Channel 3 output — topology diff.

    Consumed by ``HierarchyChangeChannel.apply_diff`` (invariant S3).
    S3 validity is enforced by sub-project B3 when restructure
    produces a real diff.
    """

    diff: tuple[tuple[str, dict[str, object]], ...]
    # No __post_init__ — S3 structural validity deferred to B3.
```

Replace it with:

```python
_VALID_TOPO_OPS: frozenset[str] = frozenset({"add", "remove", "reroute"})
_SHA256_HEX_LEN: int = 64


@dataclass(frozen=True)
class TopologyDiff:
    """Channel 3 output — topology diff.

    Consumed by ``HierarchyChangeChannel.apply_diff`` (invariant S3).
    Structural S3 validity is enforced by ``__post_init__`` (sub-project
    B3): each entry must be a ``(op, payload)`` tuple where ``op`` is in
    ``{add, remove, reroute}`` and ``payload`` carries the executable
    fields required to reconstruct or undo the mutation, plus a
    ``model_sha256_post`` provenance fingerprint.
    """

    diff: tuple[tuple[str, dict[str, object]], ...]

    def __post_init__(self) -> None:
        for idx, entry in enumerate(self.diff):
            if not (isinstance(entry, tuple) and len(entry) == 2):
                raise ValueError(
                    f"S3: TopologyDiff.diff[{idx}] must be (op, payload)"
                )
            op, payload = entry
            if op not in _VALID_TOPO_OPS:
                raise ValueError(
                    f"S3: TopologyDiff.diff[{idx}].op {op!r} unknown"
                )
            if not isinstance(payload, dict):
                raise ValueError(
                    f"S3: TopologyDiff.diff[{idx}].payload not a dict"
                )
            sha = payload.get("model_sha256_post")
            if not (isinstance(sha, str) and len(sha) == _SHA256_HEX_LEN):
                raise ValueError(
                    f"S3: TopologyDiff.diff[{idx}].model_sha256_post invalid"
                )
            if op == "add":
                for key in (
                    "index",
                    "in_features",
                    "out_features",
                    "rank",
                    "alpha",
                    "seed",
                ):
                    if key not in payload:
                        raise ValueError(
                            f"S3: add entry missing key {key!r}"
                        )
                rank = payload["rank"]
                if not (isinstance(rank, int) and rank > 0):
                    raise ValueError(
                        "S3: add rank must be a positive int"
                    )
            elif op == "remove":
                if "index" not in payload or "snapshot" not in payload:
                    raise ValueError(
                        "S3: remove entry missing index/snapshot"
                    )
                snap = payload["snapshot"]
                if not isinstance(snap, dict):
                    raise ValueError(
                        "S3: remove snapshot must be a dict"
                    )
                for arr_key in ("base_weight", "lora_a", "lora_b"):
                    if arr_key not in snap:
                        raise ValueError(
                            f"S3: remove snapshot missing {arr_key!r}"
                        )
                    arr = snap[arr_key]
                    if not np.isfinite(arr).all():
                        raise ValueError(
                            f"S2: remove snapshot[{arr_key!r}] non-finite"
                        )
            else:  # reroute
                swap = payload.get("swap_indices")
                if not (
                    isinstance(swap, tuple)
                    and len(swap) == 2
                    and all(isinstance(v, int) for v in swap)
                ):
                    raise ValueError(
                        "S3: reroute swap_indices invalid"
                    )
```

- [ ] **Step 4: Run the channels tests — expect pass**

Run: `uv run pytest tests/unit/test_channels.py -v`
Expected: all tests pass (existing ones plus the 10 new negatives + the rewritten happy-path).

- [ ] **Step 5: Full suite + mypy + ruff**

Run: `uv run pytest -q` — all pass.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/dream/channels/__init__.py tests/unit/test_channels.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/dream/channels/__init__.py tests/unit/test_channels.py
git commit -m "feat: TopologyDiff post_init enforces S3 structure"
```

Commit body:
```
feat: TopologyDiff post_init enforces S3 structure

B3 / issue #15. Adds the constructor-side guard B0 had deferred:
op vocab {add, remove, reroute}, 64-hex model_sha256_post,
per-op required keys, finite snapshot arrays for remove. Updates
the existing channel test to a valid reroute entry and adds the
10 negative cases for the new validation rules.
```

---

## Task 2: `restructure_lora_handler` factory

**Files:**
- Modify: `kiki_oniric/dream/operations/restructure_real.py`
- Test: `tests/unit/test_restructure_lora.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/unit/test_restructure_lora.py`:

```python
"""Unit tests for the LoRA restructure handler (B3)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import mlx.core as mx
else:
    mx = pytest.importorskip("mlx.core")

from kiki_oniric.dream.channels import TopologyDiff
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.restructure_real import (
    RestructureRealState,
    restructure_lora_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime
from kiki_oniric.substrates.micro_kiki.lora_model import LoRALinear, LoRAModel


def _episode(
    topo_ops: list[dict[str, object]],
    episode_id: str = "de-restr",
) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"topo_ops": topo_ops},
        operation_set=(Operation.RESTRUCTURE,),
        output_channels=(OutputChannel.TOPOLOGY_DIFF,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id=episode_id,
    )


def _run(
    model: LoRAModel,
    topo_ops: list[dict[str, object]],
    *,
    max_adds: int = 1,
    seed: int = 0,
    episode_id: str = "de-restr",
) -> tuple[RestructureRealState, object]:
    state = RestructureRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.RESTRUCTURE,
        restructure_lora_handler(
            state,
            model=model,
            max_adds_per_episode=max_adds,
            seed=seed,
        ),
    )
    runtime.execute(_episode(topo_ops, episode_id=episode_id))
    return state, runtime.log[-1]


def _reroute(i: int, j: int) -> dict[str, object]:
    return {"op": "reroute", "swap_indices": [i, j]}


def _add(
    index: int,
    in_features: int = 4,
    out_features: int = 8,
    rank: int = 2,
    alpha: float = 4.0,
) -> dict[str, object]:
    return {
        "op": "add",
        "index": index,
        "in_features": in_features,
        "out_features": out_features,
        "rank": rank,
        "alpha": alpha,
    }


def _remove(index: int) -> dict[str, object]:
    return {"op": "remove", "index": index}


def test_restructure_emits_topology_diff() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _, entry = _run(model, [_reroute(0, 1)])
    assert isinstance(entry.channel_outputs[0], TopologyDiff)


def test_restructure_s3_vocab_rejects_unknown_op() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    state = RestructureRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.RESTRUCTURE,
        restructure_lora_handler(state, model=model),
    )
    with pytest.raises(ValueError, match=r"^S3"):
        runtime.execute(_episode([{"op": "bogus"}]))
    assert len(model.layers) == pre_len  # untouched


def test_restructure_s3_add_missing_dims_rejected() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    state = RestructureRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.RESTRUCTURE,
        restructure_lora_handler(state, model=model),
    )
    with pytest.raises(ValueError, match=r"^S3"):
        runtime.execute(_episode([{"op": "add", "index": 0}]))
    assert len(model.layers) == pre_len


def test_restructure_add_grows_model() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    state, entry = _run(
        model, [_add(index=pre_len, in_features=2, out_features=2, rank=2)],
        max_adds=1,
    )
    assert len(model.layers) == pre_len + 1
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    assert td.diff[0][0] == "add"
    payload = td.diff[0][1]
    assert payload["index"] == pre_len
    assert payload["in_features"] == 2
    assert payload["out_features"] == 2
    assert payload["rank"] == 2
    assert state.total_adds == 1


def test_restructure_add_reconstruction_round_trip() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    _, entry = _run(
        model,
        [_add(index=pre_len, in_features=3, out_features=5, rank=2, alpha=4.0)],
        seed=42,
        episode_id="de-rt",
    )
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    payload = td.diff[0][1]
    inserted = model.layers[pre_len]
    # Reconstruct from the payload seed: same key seed → same init.
    rebuilt = LoRALinear(
        in_features=int(payload["in_features"]),
        out_features=int(payload["out_features"]),
        rank=int(payload["rank"]),
        alpha=float(payload["alpha"]),
        key=mx.random.key(int(payload["seed"])),
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


def test_restructure_remove_shrinks_model() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    state, entry = _run(model, [_remove(0)])
    assert len(model.layers) == pre_len - 1
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    snap = td.diff[0][1]["snapshot"]
    for k in ("base_weight", "lora_a", "lora_b"):
        assert snap[k].dtype == np.float32
        assert bool(np.isfinite(snap[k]).all())
    assert state.total_removes == 1


def test_restructure_remove_undo_round_trip() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_base = np.asarray(model.layers[0].base_weight)
    pre_a = np.asarray(model.layers[0].lora_a)
    pre_b = np.asarray(model.layers[0].lora_b)
    _, entry = _run(model, [_remove(0)])
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    snap = td.diff[0][1]["snapshot"]
    rebuilt = LoRALinear(
        in_features=int(snap["in_features"]),
        out_features=int(snap["out_features"]),
        rank=int(snap["rank"]),
        alpha=float(snap["alpha"]),
    )
    rebuilt.base_weight = mx.array(snap["base_weight"])
    rebuilt.lora_a = mx.array(snap["lora_a"])
    rebuilt.lora_b = mx.array(snap["lora_b"])
    if snap["bias"] is not None:
        rebuilt.bias = mx.array(snap["bias"])
    np.testing.assert_array_equal(np.asarray(rebuilt.base_weight), pre_base)
    np.testing.assert_array_equal(np.asarray(rebuilt.lora_a), pre_a)
    np.testing.assert_array_equal(np.asarray(rebuilt.lora_b), pre_b)


def test_restructure_reroute_swaps_in_place() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    layer_a_id = id(model.layers[0])
    layer_b_id = id(model.layers[1])
    _, _ = _run(model, [_reroute(0, 1)])
    assert id(model.layers[0]) == layer_b_id
    assert id(model.layers[1]) == layer_a_id


def test_restructure_multi_op_diff_length() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    _, entry = _run(
        model,
        [
            _add(index=pre_len, in_features=2, out_features=2),
            _reroute(0, 1),
            _remove(0),
        ],
        max_adds=1,
    )
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    ops_in_order = tuple(e[0] for e in td.diff)
    assert ops_in_order == ("add", "reroute", "remove")


def test_restructure_inss_soft_cap_one_skip() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    state, entry = _run(
        model,
        [
            _add(index=pre_len, in_features=2, out_features=2),
            _add(index=pre_len + 1, in_features=2, out_features=2),
        ],
        max_adds=1,
    )
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    assert len(td.diff) == 1
    assert state.total_adds == 1
    # Only one add applied → model grew by exactly one.
    assert len(model.layers) == pre_len + 1


def test_restructure_inss_soft_cap_full_skip_returns_none() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    state, entry = _run(
        model,
        [_add(index=pre_len, in_features=2, out_features=2)],
        max_adds=0,
    )
    assert entry.channel_outputs[0] is None
    assert state.last_compute_flops == 0
    assert len(model.layers) == pre_len


def test_restructure_sha256_changes_under_mutation() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _, entry = _run(model, [_reroute(0, 1), _reroute(0, 1)])
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    sha_first = td.diff[0][1]["model_sha256_post"]
    sha_second = td.diff[1][1]["model_sha256_post"]
    # Two independent reroutes (0,1) → first swap differs from base,
    # second swap restores order, so the two shas must differ.
    assert sha_first != sha_second


def test_restructure_sha256_stable_under_identity_swap() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    # Capture the pre-handler hash via a dummy run that does nothing.
    from kiki_oniric.dream.operations.restructure_real import _model_sha256
    pre_sha = _model_sha256(model)
    _, entry = _run(model, [_reroute(0, 0)])
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    assert td.diff[0][1]["model_sha256_post"] == pre_sha


def test_restructure_is_deterministic() -> None:
    m1 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=3)
    m2 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=3)
    pre = len(m1.layers)
    _, e1 = _run(
        m1, [_add(index=pre, in_features=2, out_features=2)],
        max_adds=1, seed=7, episode_id="de-det",
    )
    _, e2 = _run(
        m2, [_add(index=pre, in_features=2, out_features=2)],
        max_adds=1, seed=7, episode_id="de-det",
    )
    td1 = e1.channel_outputs[0]
    td2 = e2.channel_outputs[0]
    assert isinstance(td1, TopologyDiff) and isinstance(td2, TopologyDiff)
    assert td1.diff[0][1]["model_sha256_post"] == td2.diff[0][1]["model_sha256_post"]
    assert td1.diff[0][1]["seed"] == td2.diff[0][1]["seed"]
```

- [ ] **Step 2: Run to verify the suite fails**

Run: `uv run pytest tests/unit/test_restructure_lora.py -v`
Expected: FAIL — `ImportError: cannot import name 'restructure_lora_handler'` (and `_model_sha256`).

- [ ] **Step 3: Extend state, add hashing + FLOP helpers, add factory**

In `kiki_oniric/dream/operations/restructure_real.py`, **replace the entire file body** (preserving the module docstring) with:

```python
"""Real-weight restructure op — topology mutation with S3 guard.

Cycle-3 C3.3 introduced ``restructure_real_handler`` for the
``reroute``-only swap on a list-of-layers model. Sub-project B3
(issue #15) adds ``restructure_lora_handler``: the full
``{add, remove, reroute}`` vocab on a ``LoRAModel`` adapter stack,
emitting a channel-3 ``TopologyDiff`` whose entries carry executable
payloads and a per-op SHA-256 model fingerprint.

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from kiki_oniric.dream.episode import DreamEpisode

if TYPE_CHECKING:
    from kiki_oniric.dream.channels import TopologyDiff
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


# Subset of topo_ops supported by the legacy real-weight reroute op.
_SUPPORTED_TOPO_OPS: frozenset[str] = frozenset({"reroute"})

# Full vocab supported by the LoRA variant (B3).
_LORA_TOPO_OPS: frozenset[str] = frozenset({"add", "remove", "reroute"})


@dataclass
class RestructureRealState:
    """K1-tagged restructure state across multiple episodes."""

    diff_history: list[str] = field(default_factory=list)
    last_compute_flops: int = 0
    # B3 additions — defaulted so existing call sites are unaffected.
    adds_this_episode: int = 0
    total_adds: int = 0
    total_removes: int = 0
    total_reroutes: int = 0


def restructure_real_handler(
    state: RestructureRealState,
    *,
    model,
) -> Callable[[DreamEpisode], None]:
    """Build a real-weight restructure handler bound to ``state``.

    Legacy cycle-3 handler — only ``"reroute"`` is supported; unknown
    ops raise a :class:`ValueError` whose message contains the literal
    ``"S3"`` tag (cycle-3 plan §C3.3 invariant 3 / test 7).
    """

    def handler(episode: DreamEpisode) -> None:
        topo_op = episode.input_slice.get("topo_op", "")
        if topo_op not in _SUPPORTED_TOPO_OPS:
            raise ValueError(
                f"S3: DE {episode.episode_id!r}: unknown topo_op "
                f"{topo_op!r} ; real-weight op supports "
                f"{sorted(_SUPPORTED_TOPO_OPS)}"
            )

        swap_indices = episode.input_slice.get("swap_indices", [0, 1])
        if len(swap_indices) != 2:
            raise ValueError(
                "S3: reroute requires swap_indices of length 2"
            )
        i, j = swap_indices
        if not (
            isinstance(i, int)
            and isinstance(j, int)
            and 0 <= i < len(model.layers)
            and 0 <= j < len(model.layers)
        ):
            raise ValueError(
                f"S3: reroute swap_indices {swap_indices!r} out of "
                f"bounds for layers of length {len(model.layers)}"
            )

        model.layers[i], model.layers[j] = (
            model.layers[j],
            model.layers[i],
        )

        state.diff_history.append(topo_op)
        state.last_compute_flops = max(len(model.layers), 1)

    return handler


def _model_sha256(model: "LoRAModel") -> str:
    """SHA-256 of the full LoRA parameter tree (R1 fingerprint)."""
    h = hashlib.sha256()
    for layer in model.layers:
        h.update(np.asarray(layer.base_weight, dtype=np.float32).tobytes())
        h.update(np.asarray(layer.lora_a, dtype=np.float32).tobytes())
        h.update(np.asarray(layer.lora_b, dtype=np.float32).tobytes())
        if layer.use_bias:
            h.update(np.asarray(layer.bias, dtype=np.float32).tobytes())
    return h.hexdigest()


def _flop_estimate_restructure(
    applied: list[tuple[str, dict[str, object]]],
    model: "LoRAModel",
) -> int:
    """Rough FLOP cost summed across applied restructure ops."""
    total = 0
    for op, payload in applied:
        if op == "add":
            r = int(payload["rank"])
            n = int(payload["in_features"])
            m = int(payload["out_features"])
            total += 2 * r * (n + m)
        elif op == "remove":
            snap = payload["snapshot"]  # type: ignore[index]
            total += (
                int(snap["base_weight"].size)
                + int(snap["lora_a"].size)
                + int(snap["lora_b"].size)
            )
        else:  # reroute
            total += max(len(model.layers), 1)
    return max(total, 1)


def _validate_topo_op(
    op_dict: dict[str, object],
    layers_len: int,
    idx: int,
) -> None:
    """Raise S3 ValueError on any structural defect; no mutation."""
    op = op_dict.get("op")
    if op not in _LORA_TOPO_OPS:
        raise ValueError(
            f"S3: topo_ops[{idx}].op {op!r} unknown; "
            f"must be one of {sorted(_LORA_TOPO_OPS)}"
        )
    if op == "add":
        for key in ("index", "in_features", "out_features", "rank", "alpha"):
            if key not in op_dict:
                raise ValueError(
                    f"S3: topo_ops[{idx}] add missing key {key!r}"
                )
        if not (
            isinstance(op_dict["rank"], int) and int(op_dict["rank"]) > 0
        ):
            raise ValueError(
                f"S3: topo_ops[{idx}] add rank must be positive int"
            )
        ins_at = op_dict["index"]
        if not (isinstance(ins_at, int) and 0 <= ins_at <= layers_len):
            raise ValueError(
                f"S3: topo_ops[{idx}] add index {ins_at!r} out of bounds"
            )
    elif op == "remove":
        rm_at = op_dict.get("index")
        if not (isinstance(rm_at, int) and 0 <= rm_at < layers_len):
            raise ValueError(
                f"S3: topo_ops[{idx}] remove index {rm_at!r} out of bounds"
            )
    else:  # reroute
        swap = op_dict.get("swap_indices")
        if not (hasattr(swap, "__len__") and len(swap) == 2):  # type: ignore[arg-type]
            raise ValueError(
                f"S3: topo_ops[{idx}] reroute swap_indices must be length 2"
            )
        i, j = swap[0], swap[1]  # type: ignore[index]
        if not (
            isinstance(i, int)
            and isinstance(j, int)
            and 0 <= i < layers_len
            and 0 <= j < layers_len
        ):
            raise ValueError(
                f"S3: topo_ops[{idx}] reroute swap_indices {swap!r} "
                f"out of bounds for layers of length {layers_len}"
            )


def _derive_op_seed(seed: int, episode_id: str, op_index: int) -> int:
    """Stable per-op seed from (factory seed, episode_id, op index)."""
    h = hashlib.sha256()
    h.update(seed.to_bytes(8, "little", signed=False))
    h.update(episode_id.encode("utf-8"))
    h.update(op_index.to_bytes(8, "little", signed=False))
    # Use the first 8 bytes as an unsigned int seed for mx.random.key.
    return int.from_bytes(h.digest()[:8], "little", signed=False)


def restructure_lora_handler(
    state: RestructureRealState,
    *,
    model,  # LoRAModel — typed loosely for lazy MLX import
    max_adds_per_episode: int = 1,
    seed: int = 0,
) -> Callable[[DreamEpisode], "TopologyDiff | None"]:
    """Build a LoRA-only restructure handler that emits a ``TopologyDiff``.

    Reads ``topo_ops`` from the episode (default empty list), validates
    every op before any mutation, then applies them in order: ``add``
    inserts a new :class:`LoRALinear`, ``remove`` snapshots the doomed
    layer's arrays for undo and pops it, ``reroute`` swaps two
    positions. The INSS bound is enforced as a per-episode *soft* cap on
    ``add``: beyond ``max_adds_per_episode`` extra adds are silently
    skipped (no entry in the diff). Empty input or fully-skipped → returns
    ``None`` (S1 no-op).

    Each applied op produces an entry whose payload is *executable* —
    enough to reconstruct or undo the mutation — plus a 64-hex
    ``model_sha256_post`` fingerprint (R1).

    Reference:
      docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
    """
    import mlx.core as mx

    from kiki_oniric.dream.channels import TopologyDiff
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRALinear

    def handler(episode: DreamEpisode) -> "TopologyDiff | None":
        ops = episode.input_slice.get("topo_ops", [])
        if not ops:
            # S1 no-op: nothing requested.
            state.last_compute_flops = 0
            return None

        # Validate every op against current layers length BEFORE any
        # mutation. A single defect aborts the episode with a "S3:"
        # ValueError; the model stays untouched.
        layers_len = len(model.layers)
        for idx, op_dict in enumerate(ops):
            _validate_topo_op(op_dict, layers_len, idx)

        # Reset per-episode INSS counter.
        state.adds_this_episode = 0
        applied: list[tuple[str, dict[str, object]]] = []

        for idx, op_dict in enumerate(ops):
            op = op_dict["op"]

            if op == "add":
                if state.adds_this_episode >= max_adds_per_episode:
                    # INSS soft cap: silently skip — no entry, no mutation.
                    continue
                in_features = int(op_dict["in_features"])
                out_features = int(op_dict["out_features"])
                rank = int(op_dict["rank"])
                alpha = float(op_dict["alpha"])
                op_seed = _derive_op_seed(seed, episode.episode_id, idx)
                new_layer = LoRALinear(
                    in_features=in_features,
                    out_features=out_features,
                    rank=rank,
                    alpha=alpha,
                    key=mx.random.key(op_seed),
                )
                insert_at = int(op_dict["index"])
                model.layers.insert(insert_at, new_layer)
                state.adds_this_episode += 1
                state.total_adds += 1
                state.diff_history.append("add")
                payload: dict[str, object] = {
                    "index": insert_at,
                    "in_features": in_features,
                    "out_features": out_features,
                    "rank": rank,
                    "alpha": alpha,
                    "seed": op_seed,
                    "model_sha256_post": _model_sha256(model),
                }
                applied.append(("add", payload))

            elif op == "remove":
                rm_at = int(op_dict["index"])
                layer = model.layers[rm_at]
                bias_arr: np.ndarray | None
                if layer.use_bias:
                    bias_arr = np.asarray(
                        layer.bias, dtype=np.float32,
                    ).copy()
                else:
                    bias_arr = None
                snapshot: dict[str, object] = {
                    "base_weight": np.asarray(
                        layer.base_weight, dtype=np.float32,
                    ).copy(),
                    "lora_a": np.asarray(
                        layer.lora_a, dtype=np.float32,
                    ).copy(),
                    "lora_b": np.asarray(
                        layer.lora_b, dtype=np.float32,
                    ).copy(),
                    "bias": bias_arr,
                    "in_features": int(layer.in_features),
                    "out_features": int(layer.out_features),
                    "rank": int(layer.rank),
                    "alpha": float(layer.alpha),
                }
                model.layers.pop(rm_at)
                state.total_removes += 1
                state.diff_history.append("remove")
                applied.append(
                    (
                        "remove",
                        {
                            "index": rm_at,
                            "snapshot": snapshot,
                            "model_sha256_post": _model_sha256(model),
                        },
                    )
                )

            else:  # reroute
                i, j = int(op_dict["swap_indices"][0]), int(  # type: ignore[index]
                    op_dict["swap_indices"][1]  # type: ignore[index]
                )
                model.layers[i], model.layers[j] = (
                    model.layers[j],
                    model.layers[i],
                )
                state.total_reroutes += 1
                state.diff_history.append("reroute")
                applied.append(
                    (
                        "reroute",
                        {
                            "swap_indices": (i, j),
                            "model_sha256_post": _model_sha256(model),
                        },
                    )
                )

        if not applied:
            # Every requested op was an add that hit the INSS cap.
            state.last_compute_flops = 0
            return None

        state.last_compute_flops = _flop_estimate_restructure(applied, model)
        return TopologyDiff(diff=tuple(applied))

    return handler


__all__ = [
    "RestructureRealState",
    "restructure_lora_handler",
    "restructure_real_handler",
    "_model_sha256",
]
```

- [ ] **Step 4: Run the new test file**

Run: `uv run pytest tests/unit/test_restructure_lora.py -v`
Expected: PASS — 14 tests.

- [ ] **Step 5: Full suite + mypy + ruff**

Run: `uv run pytest -q` — all pass.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/dream/operations/restructure_real.py tests/unit/test_restructure_lora.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/dream/operations/restructure_real.py tests/unit/test_restructure_lora.py
git commit -m "feat: restructure_lora_handler emits a TopologyDiff"
```

Commit body:
```
feat: restructure_lora_handler emits a TopologyDiff

B3 / issue #15. Adds the LoRA-substrate restructure handler with
the full {add, remove, reroute} vocab. Validates every op before
mutating, applies them in order with an INSS soft cap on add
(default 1), snapshots base_weight/lora_a/lora_b/bias for remove
undo, and emits a TopologyDiff with executable per-op payloads
plus a 64-hex model_sha256_post R1 fingerprint. Empty or
fully-INSS-skipped episodes return None (S1 no-op).
RestructureRealState gains 4 backwards-compatible counter fields.
```

---

## Task 3: Documentation and DualVer sync

**Files:**
- Modify: `CHANGELOG.md`, `pyproject.toml`, `uv.lock`
- Modify: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/...`

- [ ] **Step 1: Add the CHANGELOG entry**

Insert at the top of the `CHANGELOG.md` body, immediately above the existing `[C-v0.17.0+PARTIAL]` entry:

```markdown
## [C-v0.18.0+PARTIAL] — 2026-05-20 — restructure emits TopologyDiff (B3)

### Formal axis (FC) — MINOR (v0.17.0 → v0.18.0)

- **New handler** `restructure_lora_handler` in
  `kiki_oniric/dream/operations/restructure_real.py`: mutates a
  `LoRAModel`'s adapter stack with the full
  `{add, remove, reroute}` vocab and emits a channel-3
  `TopologyDiff`. Validates every op pre-mutation with `"S3:"`-
  tagged errors. INSS soft cap on `add`
  (`max_adds_per_episode`, default 1) silently skips overflowing
  adds (no entry in the diff). `remove` snapshots
  `base_weight` / `lora_a` / `lora_b` / `bias` and the
  reconstruction dims into the payload so B5 can undo or rebuild
  the layer. Each applied op carries a 64-hex
  `model_sha256_post` fingerprint (R1).
- **Hardened type** `TopologyDiff` now has a `__post_init__`
  enforcing structural S3 validity (vocab, per-op required keys,
  positive `rank`, finite snapshot arrays, 64-hex sha). B0 had
  documented the deferral; B3 fills it in.
- **State** `RestructureRealState` gains 4 backwards-compatible
  counter fields (`adds_this_episode`, `total_adds`,
  `total_removes`, `total_reroutes`).
- Sub-project B3 of issue #15. `restructure` is the third dream
  operation to emit a real channel output (after `replay` B1b
  and `downscale` B2). `recombine` (B4) remains the only
  skeleton-only op. No profile wiring — the handler is exercised
  via a direct `DreamRuntime`. CasCor partial weight-freeze
  (INSS hint, optional companion to the Add bound) is **not**
  implemented in B3 and is flagged as future work.

### Empirical axis (EC) — UNCHANGED (PARTIAL)

- No new substrate, axiom, or empirical claim. EC stays
  `+PARTIAL`.

### Packaging

- `pyproject.toml` version bumped `0.15.0 → 0.16.0`.
```

Match the formatting of the surrounding entries.

- [ ] **Step 2: Bump `pyproject.toml`**

Change `version = "0.15.0"` to `version = "0.16.0"`.

- [ ] **Step 3: Sync the lockfile**

Run: `uv lock`
Expected: `uv.lock` updates the `dreamofkiki` pin to `0.16.0`.

- [ ] **Step 4: Update the framework-C spec**

In `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §4.2, near the existing B2 `downscale` note (the same area B1b/B2 notes were added), add:

```markdown
As of B3 (issue #15), `restructure` emits a real channel-3
`TopologyDiff`: `restructure_lora_handler` mutates a `LoRAModel`'s
adapter stack with the full `{add, remove, reroute}` vocab,
applies an INSS soft cap on `add`, and returns an executable +
SHA-256-fingerprinted diff. With `replay` (B1b), `downscale`
(B2), and `restructure` (B3), three of the four dream operations
now emit real channel outputs. `recombine` still returns `None`.
```

Apply the equivalent French sentence at the matching location in `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` (code identifiers stay in original form):

```markdown
Depuis B3 (issue #15), `restructure` émet un véritable
`TopologyDiff` (canal 3) : `restructure_lora_handler` mute la
pile d'adaptateurs d'un `LoRAModel` avec le vocabulaire complet
`{add, remove, reroute}`, applique un plafond INSS doux sur
`add`, et retourne un diff exécutable avec empreinte SHA-256.
Avec `replay` (B1b), `downscale` (B2) et `restructure` (B3),
trois des quatre opérations de rêve émettent désormais des
sorties de canal réelles. `recombine` retourne encore `None`.
```

- [ ] **Step 5: Verify**

Run: `uv run pytest -q` — full suite passes (docs/version change, no code touched).
Run: `uv run mypy harness tests` — `Success`.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md pyproject.toml uv.lock docs/specs/2026-04-17-dreamofkiki-framework-C-design.md docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md
git commit -m "docs: sync spec for B3 restructure TopologyDiff"
```

Commit body:
```
docs: sync spec for B3 restructure TopologyDiff

B3 / issue #15. Add the C-v0.18.0+PARTIAL changelog entry, note
in framework-C spec section 4.2 (EN + FR) that restructure now
emits a real TopologyDiff, and bump the package version to
0.16.0.
```

If `tests/reproducibility/golden_hashes.json` shows modified (R1 metadata drift), do NOT stage it.

---

## Task 4: Final verification

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
Expected: empty (if `tests/reproducibility/golden_hashes.json` shows modified, that is unrelated R1 metadata drift — restore it with `git checkout -- tests/reproducibility/golden_hashes.json`).

- [ ] **Step 5: Update issue #15**

Add a comment on issue #15: B3 complete — `restructure` is the third dream operation emitting a real channel output (channel 3 `TopologyDiff`). B4 (recombine → `LatentSample` on channel 2) is the last remaining emitting op; B5 then rewires `consolidate()` to actually apply the four channel outputs to a target model.

---

## Self-Review

- **Spec coverage:**
  - `restructure_lora_handler` in `restructure_real.py` takes `LoRAModel`, extends `RestructureRealState` (Task 2) ✓
  - Validates every op pre-mutation with `"S3:"` errors; bad op aborts whole episode (Task 2, `_validate_topo_op` + `test_restructure_s3_vocab_rejects_unknown_op` + `test_restructure_s3_add_missing_dims_rejected`) ✓
  - `add`/`remove`/`reroute` mutate `model.layers` correctly (Task 2, three behavioural tests) ✓
  - INSS soft cap on `add` (Task 2, `test_restructure_inss_soft_cap_one_skip` + `test_restructure_inss_soft_cap_full_skip_returns_none`) ✓
  - Executable payload per op + 64-hex `model_sha256_post` (Task 2, sha256 tests + reconstruction tests) ✓
  - `TopologyDiff.__post_init__` rejects malformed diffs with `"S3:"` / `"S2:"` (Task 1, 10 negatives in `test_channels.py`) ✓
  - 14 handler tests in `tests/unit/test_restructure_lora.py` (Task 2) ✓; channels tests for the new `__post_init__` (Task 1) ✓
  - FC-MINOR bump, CHANGELOG entry, spec §4.2 EN+FR note, `pyproject.toml` 0.16.0 (Task 3) ✓
  - Final verification (Task 4) ✓
- **No profile wiring** — handler exercised via direct `DreamRuntime`, matches spec scope boundary ✓
- **Placeholder scan:** no TBD/TODO; every code step has complete code.
- **Type consistency:**
  - `_model_sha256(model: LoRAModel) -> str` — same signature in Task 2 implementation and in `test_restructure_sha256_stable_under_identity_swap` import.
  - `_flop_estimate_restructure(applied, model)` — taken as `list[tuple[str, dict]]` from the live `applied` list built in the handler; consistent.
  - `_derive_op_seed(seed, episode_id, op_index)` — int / str / int — consistent with handler invocation `_derive_op_seed(seed, episode.episode_id, idx)`.
  - `restructure_lora_handler(state, *, model, max_adds_per_episode=1, seed=0)` — same kwargs used by all tests (`max_adds`, `seed`).
  - `TopologyDiff(diff=tuple[(op, payload)])` — `_VALID_TOPO_OPS = {"add", "remove", "reroute"}` is identical between `channels/__init__.py` (Task 1) and `restructure_real.py:_LORA_TOPO_OPS` (Task 2).
  - `RestructureRealState` fields used by tests (`total_adds`, `total_removes`, `last_compute_flops`) match the dataclass definition in Task 2.
- **Inter-task ordering:** Task 1 ships the `__post_init__` first; Task 2's handler emits `TopologyDiff` instances that already pass the guard. If Tasks were swapped, Task 2's tests would still pass (no `__post_init__` enforcement yet), but the existing `test_topology_diff_holds_tuple` would be a latent bug. The chosen order avoids that.
