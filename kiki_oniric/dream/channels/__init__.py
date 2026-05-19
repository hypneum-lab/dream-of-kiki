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
    "TopologyDiff",
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


@dataclass(frozen=True)
class AttentionPrior:
    """Channel 4 output — meta-cognitive attention prior.

    Consumed by ``AttentionPriorChannel.set_prior`` (invariant S4).
    """

    prior: NDArray[np.float32]

    def __post_init__(self) -> None:
        if not np.isfinite(self.prior).all():
            raise ValueError("S2: AttentionPrior.prior non-finite")


ChannelOutput = WeightUpdate | LatentSample | TopologyDiff | AttentionPrior
