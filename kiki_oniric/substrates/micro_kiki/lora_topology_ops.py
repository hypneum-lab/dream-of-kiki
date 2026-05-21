"""LoRA-substrate topology mutation kernel.

Single source of truth for ``add`` / ``remove`` / ``reroute`` on
a ``LoRAModel.layers`` stack. Imported by two call sites :

- ``kiki_oniric.dream.operations.restructure_real.restructure_lora_handler``
  (B3 + the 2026-05-21 delegate refactor) — for the dream-side
  mutation + ``TopologyDiff`` emission.
- ``kiki_oniric.dream.channels.hierarchy_change.LoRAHierarchyChangeChannel
  .apply_diff`` (B5) — for the awake-side replay of an emitted
  ``TopologyDiff``.

The helper validates ``op`` + indices defensively even though
the dream-side caller (B3) validates upfront via
``_validate_topo_op`` — defense-in-depth ; the channel-side
caller (B5) trusts ``TopologyDiff.__post_init__`` to have
validated the diff's structural correctness, but bounds against
the current ``model.layers`` length is something only the
mutating call can check.

The ``add`` path uses the ``seed`` field that B3 stores in the
payload to call ``mx.random.key(seed)`` — the R1 linchpin : the
reconstructed ``LoRALinear`` is bit-identical to the one created
by ``restructure_lora_handler``.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
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
    """Apply one topology op from a ``TopologyDiff`` entry onto *model*.

    ``add``    — insert a new ``LoRALinear`` at ``payload["index"]``,
                 reconstructed via ``mx.random.key(payload["seed"])``
                 for R1 bit-exactness.
    ``remove`` — pop the layer at ``payload["index"]``.  The snapshot
                 stored in the payload is not re-applied here (undo
                 logic is future work); only the layer is removed so
                 that the topology matches the dream-side post-state.
    ``reroute`` — swap the two layers at ``payload["swap_indices"]``.

    Raises ``ValueError`` with an ``"S3:"`` prefix on any structural
    defect (unknown op, index out of bounds).
    """
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRALinear

    _VALID_OPS = frozenset({"add", "remove", "reroute"})
    if op not in _VALID_OPS:
        raise ValueError(
            f"S3: _apply_topology_op unknown op {op!r}; "
            f"must be one of {sorted(_VALID_OPS)}"
        )

    if op == "add":
        index = int(payload["index"])  # type: ignore[arg-type]
        in_features = int(payload["in_features"])  # type: ignore[arg-type]
        out_features = int(payload["out_features"])  # type: ignore[arg-type]
        rank = int(payload["rank"])  # type: ignore[arg-type]
        alpha = float(payload["alpha"])  # type: ignore[arg-type]
        seed = int(payload["seed"])  # type: ignore[arg-type]
        if not (0 <= index <= len(model.layers)):
            raise ValueError(
                f"S3: add index {index} out of bounds for "
                f"{len(model.layers)} layers"
            )
        new_layer = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            key=mx.random.key(seed),
        )
        model.layers.insert(index, new_layer)

    elif op == "remove":
        rm_at = int(payload["index"])  # type: ignore[arg-type]
        if not (0 <= rm_at < len(model.layers)):
            raise ValueError(
                f"S3: remove index {rm_at} out of bounds for "
                f"layers of length {len(model.layers)}"
            )
        model.layers.pop(rm_at)

    else:  # reroute
        swap = payload["swap_indices"]
        i, j = int(swap[0]), int(swap[1])  # type: ignore[index]
        layers_len = len(model.layers)
        if not (0 <= i < layers_len and 0 <= j < layers_len):
            raise ValueError(
                f"S3: reroute swap_indices ({i}, {j}) out of bounds "
                f"for layers of length {layers_len}"
            )
        model.layers[i], model.layers[j] = (
            model.layers[j],
            model.layers[i],
        )


__all__ = ["_apply_topology_op"]
