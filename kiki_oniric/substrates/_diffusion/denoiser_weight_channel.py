"""DenoiserWeightDeltaChannel — Canal 1 awake-side applier for the
diffusion substrate.

Consumes ``WeightUpdate`` outputs whose ``lora_delta`` field carries
a per-layer dense delta keyed by ``f"layer_{i}_weight"`` or
``f"layer_{i}_bias"`` (matching the emission convention used by
``replay_diffusion_handler`` and ``downscale_diffusion_handler``).
Applies the delta in-place to the matching ``MLPDenoiser``
parameter, then re-checks S2 (finite values) on the result.

The ``lora_`` prefix on the ``WeightUpdate`` field is a legacy
naming carry-over from the LoRA origin of the channel-1 protocol;
the value is a dict of dense per-layer arrays.

See ``docs/superpowers/specs/2026-05-21-m7-delta-acc-consolidation-eval-design.md``
§ Components 1.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray


class DenoiserWeightDeltaChannel:
    """Awake-side WeightDeltaChannel for an ``MLPDenoiser``.

    Implements the ``WeightDeltaChannel`` protocol (one method,
    ``apply``). Holds a reference to the denoiser whose parameters
    it mutates.
    """

    def __init__(self, denoiser: Any) -> None:
        self._denoiser = denoiser
        self.last_fisher_bump: (
            dict[str, NDArray[np.float32]] | None
        ) = None

    def apply(
        self,
        lora_delta: dict[str, NDArray[np.float32]],
        fisher_bump: dict[str, NDArray[np.float32]] | None = None,
    ) -> None:
        """Add each per-layer delta to the matching denoiser param.

        Keys follow ``f"layer_{i}_weight"`` / ``f"layer_{i}_bias"``.
        Unknown keys raise ``KeyError`` — silent skip would hide
        emission/consume mismatches.

        S2 finite values: input is already validated by
        ``WeightUpdate.__post_init__``; we re-validate the post-apply
        parameter so a numerical surprise on the denoiser side
        surfaces here rather than at the next forward pass.
        """
        layers = self._denoiser.layers
        for key, delta in lora_delta.items():
            layer_idx, attr = self._parse_key(key, n_layers=len(layers))
            current = getattr(layers[layer_idx], attr)
            new_val = current + mx.array(delta)
            new_np = np.asarray(new_val)
            if not np.isfinite(new_np).all():
                raise ValueError(
                    f"S2: denoiser {attr} {layer_idx} non-finite after "
                    f"apply"
                )
            setattr(layers[layer_idx], attr, new_val)
        self.last_fisher_bump = fisher_bump
        mx.eval(*self._all_layer_tensors())

    @staticmethod
    def _parse_key(key: str, *, n_layers: int) -> tuple[int, str]:
        for attr in ("weight", "bias"):
            prefix = "layer_"
            suffix = f"_{attr}"
            if key.startswith(prefix) and key.endswith(suffix):
                try:
                    idx = int(key[len(prefix):-len(suffix)])
                except ValueError:
                    break
                if 0 <= idx < n_layers:
                    return idx, attr
        raise KeyError(
            f"DenoiserWeightDeltaChannel: unknown layer key {key!r}"
        )

    def _all_layer_tensors(self) -> list[Any]:
        out: list[Any] = []
        for layer in self._denoiser.layers:
            for attr in ("weight", "bias"):
                t = getattr(layer, attr, None)
                if t is not None:
                    out.append(t)
        return out


__all__ = ["DenoiserWeightDeltaChannel"]
