"""LoRA-target concrete implementation of HierarchyChangeChannel (B5).

Applies a ``TopologyDiff`` (channel-3 output) onto a ``LoRAModel``
adapter stack, reconstructing or undoing each topology mutation
bit-exactly. The ``add`` path uses the ``seed`` field that B3 stores
in the payload to call ``mx.random.key(seed)`` — this is the R1
linchpin: the reconstructed ``LoRALinear`` is identical to the one
created by ``restructure_lora_handler``.

The per-op mutation kernel ``_apply_topology_op`` lives in
``kiki_oniric.substrates.micro_kiki.lora_topology_ops`` (extracted
2026-05-21). It is re-exported from this module for backwards
compatibility — callers that imported it from here continue to work.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

from kiki_oniric.substrates.micro_kiki.lora_topology_ops import (
    _apply_topology_op,
)

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


class LoRAHierarchyChangeChannel:
    """Concrete ``HierarchyChangeChannel`` for a ``LoRAModel`` target.

    ``apply_diff`` iterates the diff entries and delegates each one
    to ``_apply_topology_op``, then calls ``mx.eval`` to materialise.
    An empty diff is a no-op (S1 contract).

    The ``add`` path passes ``payload["seed"]`` to
    ``mx.random.key(seed)`` so the reconstructed layer is bit-identical
    to the one originally created by ``restructure_lora_handler`` —
    the R1 linchpin for B5.
    """

    def __init__(self, target: "LoRAModel") -> None:
        self._target = target

    def apply_diff(
        self,
        diff: list[tuple[str, dict]],
    ) -> None:
        """Apply every (op, payload) entry in *diff* onto the target model."""
        for op, payload in diff:
            _apply_topology_op(self._target, op, payload)
        if diff:
            mx.eval(self._target.parameters())


__all__ = [
    "_apply_topology_op",
    "LoRAHierarchyChangeChannel",
]
