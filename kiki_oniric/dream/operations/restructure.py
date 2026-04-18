"""Restructure operation — D-Friston FEP source (free energy minimisation).

Skeleton version (S10.1): records topology diff events
{add, remove, reroute} with target_layer info. Real graph mutation
on the model topology lands S10.2 (topology guard) + S11.2 (P_equ
full wiring with restructure handler).

Mathematical role (per docs/proofs/op-pair-analysis.md): canonical
order places restructure AFTER replay/downscale (serial branch
A-B-D) because restructure on yet-restructured topology risks
losing episodic specificity.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from kiki_oniric.dream.episode import DreamEpisode


VALID_TOPO_OPS: frozenset[str] = frozenset({"add", "remove", "reroute"})


@dataclass
class RestructureOpState:
    """Mutable state for restructure op across episodes."""

    total_episodes_handled: int = 0
    total_diffs_emitted: int = 0
    last_diff_type: str | None = None
    diff_history: list[str] = field(default_factory=list)


def restructure_handler(
    state: RestructureOpState,
) -> Callable[[DreamEpisode], None]:
    """Build a restructure handler bound to a state instance.

    Handler reads `topo_op` from input_slice (must be in
    {add, remove, reroute}), updates state. No-op on actual graph
    structure for now (skeleton) — real D-Friston FEP gradient
    descent on hierarchy lands S11.2.
    """

    def handler(episode: DreamEpisode) -> None:
        topo_op = episode.input_slice.get("topo_op", "")
        if topo_op not in VALID_TOPO_OPS:
            raise ValueError(
                f"topo_op must be one of {sorted(VALID_TOPO_OPS)}, "
                f"got {topo_op!r}"
            )
        state.total_episodes_handled += 1
        state.total_diffs_emitted += 1
        state.last_diff_type = topo_op
        state.diff_history.append(topo_op)

    return handler


def restructure_handler_mlx(
    state: RestructureOpState,
    model,
) -> Callable[[DreamEpisode], None]:
    """Build a restructure handler with real MLX topology mutation.

    Operates on a list-based model (e.g., StackedMLP wrapper around
    a list of nn.Linear). Three operations:
    - "add": append new nn.Linear with `new_dim` from input_slice
    - "remove": detach layer at `layer_index`
    - "reroute": swap positions in `swap_indices` (length-2 list)

    Skeleton handler preserved for tests / contexts not requiring
    real topology mutation. This MLX variant is the production path
    for the G4 GO-FULL gate.

    Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
    """
    import mlx.nn as nn

    def handler(episode: DreamEpisode) -> None:
        topo_op = episode.input_slice.get("topo_op", "")
        if topo_op not in VALID_TOPO_OPS:
            raise ValueError(
                f"topo_op must be one of {sorted(VALID_TOPO_OPS)}, "
                f"got {topo_op!r}"
            )

        if topo_op == "add":
            new_dim = episode.input_slice.get("new_dim", 0)
            if new_dim <= 0:
                raise ValueError(
                    f"add requires new_dim > 0, got {new_dim}"
                )
            last_out = model.layers[-1].weight.shape[0]
            model.layers.append(nn.Linear(last_out, new_dim))

        elif topo_op == "remove":
            layer_index = episode.input_slice.get("layer_index", -1)
            if not (0 <= layer_index < len(model.layers)):
                raise ValueError(
                    f"layer_index {layer_index} out of range"
                )
            del model.layers[layer_index]

        elif topo_op == "reroute":
            swap_indices = episode.input_slice.get("swap_indices", [])
            if len(swap_indices) != 2:
                raise ValueError(
                    "reroute requires swap_indices of length 2"
                )
            i, j = swap_indices
            model.layers[i], model.layers[j] = (
                model.layers[j], model.layers[i]
            )

        state.total_episodes_handled += 1
        state.total_diffs_emitted += 1
        state.last_diff_type = topo_op
        state.diff_history.append(topo_op)

    return handler
