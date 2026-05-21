# B3 refactor — delegate to `_apply_topology_op`

**Date** : 2026-05-21
**Status** : design approved, pending spec review
**Tracking** : deferred cleanup item from B5 spec, flagged again
in B6c milestone. No GitHub issue.
**Scope** : internal refactor — extract the shared add/remove/
reroute mutation helper to a substrate-utility module, have
both `restructure_lora_handler` (B3) and
`LoRAHierarchyChangeChannel.apply_diff` (B5) delegate to it.
Zero behaviour change.

---

## Context

B3 (`kiki_oniric/dream/operations/restructure_real.py` ::
`restructure_lora_handler`) and B5
(`kiki_oniric/dream/channels/hierarchy_change.py` ::
`LoRAHierarchyChangeChannel.apply_diff`) both mutate
`model.layers` with the same three-op vocabulary
(`add`/`remove`/`reroute`). B5's spec extracted the mutation into
a module-level `_apply_topology_op(model, op, payload)` helper
*and* explicitly flagged this duplication for a future cleanup :

> *(B5 spec, §"Scope boundary")* B6 does not refactor
> `restructure_lora_handler` to delegate to
> `_apply_topology_op` — the helper is intentionally module-level
> so B3 can delegate later.

That "later" is now. After B6c shipped and the LoRA-profile
decomposition closed (`C-v0.23.0+PARTIAL`), the duplication is
ripe for cleanup with zero risk : the 14 existing tests in
`tests/unit/test_restructure_lora.py` pin behaviour, and B5's
own tests pin the helper's behaviour from the channel side.

## Problem

Two code paths mutate `LoRAModel.layers` for the same three op
kinds. Any future change to the mutation contract (new op kind,
revised reroute semantics, etc.) has to land in two places.
Single source of truth removes that drift surface.

## Approaches considered

**Helper location.** Three options were considered :

1. `kiki_oniric/dream/operations/_topology_ops.py` — a new
   operations-level utility. Dependency direction
   `channels → operations`, slightly counter-intuitive (channels
   are typically downstream of ops in the dream lifecycle).
   Rejected.
2. **`kiki_oniric/substrates/micro_kiki/lora_topology_ops.py`** —
   a substrate-level utility. Both `operations/` and
   `channels/` depend on `substrates/` for `LoRAModel` already,
   so importing the helper from there keeps a clean topological
   sort. **Chosen.**
3. Leave the helper in `channels/hierarchy_change.py` and have
   `restructure_real.py` import from there. Smallest diff but
   creates an `operations → channels` dependency for a utility
   that's substrate-level in nature. Rejected.

**Helper signature.** Two options :

1. **Keep `_apply_topology_op(model, op, payload)` unchanged.**
   B3 builds the payload first (with the derived seed for `add`,
   the snapshot for `remove`) and calls the helper. Defense-in-
   depth bounds checks inside the helper stay — they never fire
   from B3 because B3's `_validate_topo_op` runs first.
   **Chosen.**
2. Add an optional `skip_validation: bool = False` kwarg so B3
   can opt out of the redundant bounds check. Rejected — the
   bounds check is a constant-time no-op when input is already
   valid ; the kwarg adds API surface for no measurable gain.

**Naming.** Three options :

1. **Keep `_apply_topology_op` (with leading underscore)** as
   the canonical name. The module-level helper is conceptually
   "private to the topology-op kernel" — callers are
   restructure_real.py and hierarchy_change.py, both intimately
   related to the topology contract. The underscore signals
   "internal API, do not rely on this from outside the
   kiki_oniric package". The old `__all__` in
   hierarchy_change.py re-exports it unchanged. **Chosen.**
2. Rename to `apply_topology_op` (public). Slightly cleaner
   semantically but breaks the `__all__` line in
   hierarchy_change.py, which is a published symbol since B5
   shipped. Rejected unless explicitly requested.
3. Both names (alias). Rejected — YAGNI, two names for the
   same thing.

## Design

### New file `kiki_oniric/substrates/micro_kiki/lora_topology_ops.py`

Move the existing `_apply_topology_op` from
`kiki_oniric/dream/channels/hierarchy_change.py` verbatim
(including the `from ... import LoRALinear` lazy import inside,
the `import mlx.core as mx`, and the bounds checks). The body
does not change ; only the file location does.

```python
"""LoRA-substrate topology mutation kernel.

Single source of truth for ``add`` / ``remove`` / ``reroute`` on
a ``LoRAModel.layers`` stack. Imported by two call sites :

- ``kiki_oniric.dream.operations.restructure_real.restructure_lora_handler``
  (B3 + the 2026-05-21 refactor) — for the dream-side mutation +
  ``TopologyDiff`` emission.
- ``kiki_oniric.dream.channels.hierarchy_change.LoRAHierarchyChangeChannel
  .apply_diff`` (B5) — for the awake-side replay of an emitted
  ``TopologyDiff``.

The helper validates ``op`` + indices defensively even though
the dream-side caller (B3) validates upfront via
``_validate_topo_op`` — defense-in-depth ; the channel-side
caller (B5) trusts ``TopologyDiff.__post_init__`` (B3) to have
validated the diff's structural correctness, but bounds against
the current ``model.layers`` length is something only the
mutating call can check.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


def _apply_topology_op(
    model: "LoRAModel",
    op: str,
    payload: dict[str, object],
) -> None:
    """Apply a single topology operation to ``model.layers``.

    Args:
      model: a ``LoRAModel`` (``layers`` list mutated in place).
      op: one of ``"add"`` / ``"remove"`` / ``"reroute"``.
      payload: op-specific kwargs (see below).

    Payloads (mirrors ``TopologyDiff`` per-op shapes, B3 §4.2):
      - ``add`` : ``{"index", "in_features", "out_features", "rank",
        "alpha", "seed"}`` ; the new ``LoRALinear`` is constructed
        with ``mx.random.key(payload["seed"])`` so the awake clone
        ends up bit-equal to the dream-side insertion.
      - ``remove`` : ``{"index"}`` ; the ``snapshot`` field that
        B3 also carries is *not* read by the helper — the helper
        only pops.
      - ``reroute`` : ``{"swap_indices": (i, j)}``.

    Raises:
      ValueError (S3) on unknown op or out-of-bounds index.
    """
    # === body verbatim from current hierarchy_change.py:_apply_topology_op ===
```

The body itself is copied unchanged from
`kiki_oniric/dream/channels/hierarchy_change.py` lines 26-95
(the current `_apply_topology_op` function).

### Modify `kiki_oniric/dream/channels/hierarchy_change.py`

- Remove the local `_apply_topology_op` definition.
- Add an import at module top :
  ```python
  from kiki_oniric.substrates.micro_kiki.lora_topology_ops import (
      _apply_topology_op,
  )
  ```
- Keep `_apply_topology_op` in `__all__` (re-export ; B5's
  published symbol is unchanged).
- `LoRAHierarchyChangeChannel.apply_diff` body is untouched —
  it already calls `_apply_topology_op(self._target, op, payload)`.

### Modify `kiki_oniric/dream/operations/restructure_real.py`

Add an import :

```python
from kiki_oniric.substrates.micro_kiki.lora_topology_ops import (
    _apply_topology_op,
)
```

Refactor `restructure_lora_handler`'s per-op blocks :

**`add` branch** :
```python
# Before: builds new_layer + calls model.layers.insert in 12 lines.
# After: build the payload (with derived seed) and delegate.
op_seed = _derive_op_seed(seed, episode.episode_id, idx)
insert_at = int(op_dict["index"])
in_features = int(op_dict["in_features"])
out_features = int(op_dict["out_features"])
rank = int(op_dict["rank"])
alpha = float(op_dict["alpha"])
add_payload: dict[str, object] = {
    "index": insert_at,
    "in_features": in_features,
    "out_features": out_features,
    "rank": rank,
    "alpha": alpha,
    "seed": op_seed,
    "model_sha256_post": "",  # filled post-mutation
}
_apply_topology_op(model, "add", add_payload)
add_payload["model_sha256_post"] = _model_sha256(model)
state.adds_this_episode += 1
state.total_adds += 1
state.diff_history.append("add")
applied.append(("add", add_payload))
```

**`remove` branch** — capture snapshot BEFORE delegating :
```python
rm_at = int(op_dict["index"])
layer = model.layers[rm_at]
bias_arr = (
    np.asarray(layer.bias, dtype=np.float32).copy()
    if layer.use_bias else None
)
snapshot: dict[str, object] = {
    "base_weight": np.asarray(layer.base_weight, dtype=np.float32).copy(),
    "lora_a": np.asarray(layer.lora_a, dtype=np.float32).copy(),
    "lora_b": np.asarray(layer.lora_b, dtype=np.float32).copy(),
    "bias": bias_arr,
    "in_features": int(layer.in_features),
    "out_features": int(layer.out_features),
    "rank": int(layer.rank),
    "alpha": float(layer.alpha),
}
remove_payload: dict[str, object] = {
    "index": rm_at,
    "snapshot": snapshot,
    "model_sha256_post": "",
}
_apply_topology_op(model, "remove", remove_payload)
remove_payload["model_sha256_post"] = _model_sha256(model)
state.total_removes += 1
state.diff_history.append("remove")
applied.append(("remove", remove_payload))
```

**`reroute` branch** :
```python
i = int(op_dict["swap_indices"][0])
j = int(op_dict["swap_indices"][1])
reroute_payload: dict[str, object] = {
    "swap_indices": (i, j),
    "model_sha256_post": "",
}
_apply_topology_op(model, "reroute", reroute_payload)
reroute_payload["model_sha256_post"] = _model_sha256(model)
state.total_reroutes += 1
state.diff_history.append("reroute")
applied.append(("reroute", reroute_payload))
```

The `mx.core` / `LoRALinear` imports at handler-build time can
be dropped from `restructure_real.py` — the helper module owns
them now. (`np` import stays because the remove snapshot still
uses numpy at this call site.)

### Test plan

**Zero new tests required** — `tests/unit/test_restructure_lora.py`
(14 tests) MUST all pass unchanged. That's the refactor's
acceptance criterion.

**One optional new test** to pin the re-export path :
`tests/unit/test_lora_topology_ops_reexport.py` :
```python
def test_apply_topology_op_reexport_identity() -> None:
    """The helper is reachable from both the canonical
    substrate module and the legacy channels re-export."""
    from kiki_oniric.dream.channels.hierarchy_change import (
        _apply_topology_op as via_channels,
    )
    from kiki_oniric.substrates.micro_kiki.lora_topology_ops import (
        _apply_topology_op as canonical,
    )
    assert via_channels is canonical
```

This is **included** in the refactor scope (1 test, ~10 lines).

## Scope boundary

**Refactor does** :
- Create `kiki_oniric/substrates/micro_kiki/lora_topology_ops.py`
  with the verbatim helper body.
- Modify `kiki_oniric/dream/channels/hierarchy_change.py` to
  import + re-export.
- Modify `kiki_oniric/dream/operations/restructure_real.py`
  to delegate the 3 per-op mutations.
- Add `tests/unit/test_lora_topology_ops_reexport.py` (1 test).
- `pyproject.toml` version `0.22.0 → 0.22.1`.
- CHANGELOG bullet under `[Unreleased]` flagging the refactor
  (no `[C-v0.24.1]` section since FC-PATCH bumps don't get
  their own changelog block per dreamOfkiki convention — see
  the prior FC-PATCH amendment of DR-2 in 2026-04-21 which was
  documented via an amendment file, not a CHANGELOG block).

**Refactor does not** :
- Rename `_apply_topology_op` → `apply_topology_op`.
- Change the helper's signature or body logic.
- Touch `_validate_topo_op`, `_derive_op_seed`, `_model_sha256`,
  `_flop_estimate_restructure`, or `RestructureRealState`.
- Modify any existing test (no rewrites of `test_restructure_lora.py`).
- Touch `kiki_oniric/profiles/p_*_lora.py`.
- Add a framework-C spec note (the refactor is internal —
  no spec wording changes).

## Convention compliance

- `lora_topology_ops.py` lives next to `lora_model.py` in
  `kiki_oniric/substrates/micro_kiki/` — same package
  as the substrate it mutates.
- The leading underscore on `_apply_topology_op` is preserved.
- Both `kiki_oniric.dream.channels.hierarchy_change.__all__` and
  the new `kiki_oniric.substrates.micro_kiki.lora_topology_ops`
  expose the symbol ; the canonical home is the new module, the
  channels module is a backwards-compat re-export.

## DualVer

FC-**PATCH** (`C-v0.24.0 → C-v0.24.1`) :

- Internal refactor, zero behaviour change.
- No axiom, no Protocol, no primitive signature change, no
  invariant ID change.
- 14 existing tests + 1 new re-export identity test all pass.

EC unchanged `+PARTIAL`. `pyproject.toml` version `0.21.0 →
0.21.1`. No new `[C-v0.24.1+PARTIAL]` CHANGELOG section
(per FC-PATCH convention) ; one Refactored bullet under
`[Unreleased]` instead.

## Acceptance criteria

1. `kiki_oniric/substrates/micro_kiki/lora_topology_ops.py`
   exists with `_apply_topology_op(model, op, payload)`,
   body identical to the prior version in
   `hierarchy_change.py`.
2. `kiki_oniric/dream/channels/hierarchy_change.py` no longer
   defines `_apply_topology_op` locally ; imports it from
   `kiki_oniric.substrates.micro_kiki.lora_topology_ops` ;
   re-exports it via `__all__`.
3. `kiki_oniric/dream/operations/restructure_real.py` imports
   `_apply_topology_op` and delegates the three per-op
   mutations. Snapshot capture for `remove` happens before
   the helper call. Seed derivation for `add` and payload
   sha256 post-mutation happen at the same call site.
4. `tests/unit/test_lora_topology_ops_reexport.py` passes (1
   test asserting `via_channels is canonical`).
5. `tests/unit/test_restructure_lora.py` and
   `tests/unit/test_apply_channel_outputs.py` pass unchanged.
6. Full pytest suite green ; `uv run mypy harness tests` clean
   ; `uv run ruff check .` clean.
7. `pyproject.toml` bumped `0.22.0 → 0.22.1`. One Refactored
   bullet under `[Unreleased]` in `CHANGELOG.md`. No new
   versioned `[C-v0.24.1+PARTIAL]` section.
