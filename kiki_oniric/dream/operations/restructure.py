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
