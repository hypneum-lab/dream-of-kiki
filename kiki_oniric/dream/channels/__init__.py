"""Dream-awake channel-output value types.

The four operations of framework C publish their result on one of
four typed channels (framework-C spec §4.1). This module defines
the value types carried on each channel; the channel Protocols
that *consume* them live in `kiki_oniric/core/primitives.py`.

B0 (issue #15) defines these types and threads them through the
runtime log. The operations themselves return ``None`` until
sub-projects B1-B4 populate real values.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.1
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "WeightUpdate",
    "LatentSample",
    "HierarchyDiff",
    "AttentionPrior",
    "ChannelOutput",
]


@dataclass(frozen=True)
class WeightUpdate:
    """Channel 1 output — parametric consolidation delta.

    Consumed by ``WeightDeltaChannel.apply`` (invariants S1 + S2).
    ``lora_delta`` / ``fisher_bump`` are keyed by layer name to
    match the channel Protocol signature.
    """

    lora_delta: dict[str, NDArray[np.float32]]
    fisher_bump: dict[str, NDArray[np.float32]] | None = None

    def __post_init__(self) -> None:
        for layer, arr in self.lora_delta.items():
            if not np.isfinite(arr).all():
                raise ValueError(
                    f"S2: WeightUpdate.lora_delta[{layer!r}] non-finite"
                )
        if self.fisher_bump is not None:
            for layer, arr in self.fisher_bump.items():
                if not np.isfinite(arr).all():
                    raise ValueError(
                        f"S2: WeightUpdate.fisher_bump[{layer!r}] "
                        f"non-finite"
                    )


@dataclass(frozen=True)
class LatentSample:
    """Channel 2 output — generative-replay latent vector.

    Consumed by ``LatentSampleChannel.enqueue`` (invariant I3).
    """

    species: str
    latent_vector: NDArray[np.float32]
    provenance: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.latent_vector).all():
            raise ValueError("S2: LatentSample.latent_vector non-finite")


@dataclass(frozen=True)
class HierarchyDiff:
    """Channel 3 output — topology diff.

    Consumed by ``HierarchyChangeChannel.apply_diff`` (invariant S3).
    S3 validity is enforced by sub-project B3 when restructure
    produces a real diff.
    """

    diff: tuple[tuple[str, dict[str, object]], ...]


@dataclass(frozen=True)
class AttentionPrior:
    """Channel 4 output — meta-cognitive attention prior.

    Consumed by ``AttentionPriorChannel.set_prior`` (invariant S4).
    """

    prior: NDArray[np.float32]

    def __post_init__(self) -> None:
        if not np.isfinite(self.prior).all():
            raise ValueError("S2: AttentionPrior.prior non-finite")


ChannelOutput = WeightUpdate | LatentSample | HierarchyDiff | AttentionPrior
