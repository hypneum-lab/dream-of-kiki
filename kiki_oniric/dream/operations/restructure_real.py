"""Real-weight restructure op — topology mutation (reroute) with S3 guard.

Cycle-3 C3.3 counterpart to
:mod:`kiki_oniric.dream.operations.restructure` that actually mutates
``model.layers`` ordering. The narrow production use-case today is
``"reroute"`` : swap two layers' positions without resizing.

Contract :

- ``topo_op`` read from ``episode.input_slice`` ; only ``"reroute"``
  is supported in the cycle-3 real-weight op.
- Unknown ``topo_op`` → ``ValueError`` whose message contains the
  literal ``"S3"`` tag so regex-based tests can match it (test 7).
- ``swap_indices`` defaults to ``[0, 1]`` if absent ; must be a
  length-2 sequence of valid layer indices.
- ``state.diff_history`` grows by one entry per call.
- ``state.last_compute_flops`` is tagged with a rough cost estimate
  (roughly O(n_layers) for a pointer swap).

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from kiki_oniric.dream.episode import DreamEpisode


# Subset of topo_ops supported by the real-weight op in cycle-3.
# The skeleton restructure.py accepts {"add", "remove", "reroute"},
# but the real-weight op only implements "reroute" — mutating the
# MLX parameter tree for add / remove lands in a later cycle.
_SUPPORTED_TOPO_OPS: frozenset[str] = frozenset({"reroute"})


@dataclass
class RestructureRealState:
    """K1-tagged restructure state across multiple episodes."""

    diff_history: list[str] = field(default_factory=list)
    last_compute_flops: int = 0


def restructure_real_handler(
    state: RestructureRealState,
    *,
    model,
) -> Callable[[DreamEpisode], None]:
    """Build a real-weight restructure handler bound to ``state``.

    Only ``"reroute"`` is supported — unknown ops raise a
    :class:`ValueError` whose message contains the literal ``"S3"``
    tag (per cycle-3 plan §C3.3 invariant 3 / test 7).
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
        # K1 tag : pointer swap is O(1) but charge the layer count
        # so the scheduler sees a non-zero cost.
        state.last_compute_flops = max(len(model.layers), 1)

    return handler


__all__ = [
    "RestructureRealState",
    "restructure_real_handler",
]
