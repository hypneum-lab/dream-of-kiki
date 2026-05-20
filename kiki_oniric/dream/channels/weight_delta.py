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
            if layer_idx < 0 or layer_idx >= len(self._target.layers):
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
