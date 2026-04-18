"""P_equ profile — balanced canonical consolidation (S11.2 fully wired).

Channels: β + δ → 1 + 3 + 4 (curated buffer + hierarchical latents
in, weight delta + hierarchy change + attention prior out).
Operations: {replay, downscale, restructure, recombine}.

S11.2 promotion from skeleton (S8.3) to fully wired profile:
- 4 op handlers registered on a fresh DreamRuntime
- 4 op states tracked as fields
- recombine_light variant uses random.Random for sampling (rng
  field available for seeded reproducibility)

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from kiki_oniric.dream.episode import Operation
from kiki_oniric.dream.operations.downscale import (
    DownscaleOpState,
    downscale_handler,
)
from kiki_oniric.dream.operations.recombine import (
    RecombineOpState,
    recombine_handler,
)
from kiki_oniric.dream.operations.replay import (
    ReplayOpState,
    replay_handler,
)
from kiki_oniric.dream.operations.restructure import (
    RestructureOpState,
    restructure_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime


@dataclass
class PEquProfile:
    """Balanced profile with 4 ops + 3 output channels wired."""

    status: str = "wired"
    unimplemented_ops: list[str] = field(default_factory=list)
    runtime: DreamRuntime = field(default_factory=DreamRuntime)
    replay_state: ReplayOpState = field(default_factory=ReplayOpState)
    downscale_state: DownscaleOpState = field(
        default_factory=DownscaleOpState
    )
    restructure_state: RestructureOpState = field(
        default_factory=RestructureOpState
    )
    recombine_state: RecombineOpState = field(
        default_factory=RecombineOpState
    )
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.runtime.register_handler(
            Operation.REPLAY, replay_handler(self.replay_state)
        )
        self.runtime.register_handler(
            Operation.DOWNSCALE, downscale_handler(self.downscale_state)
        )
        self.runtime.register_handler(
            Operation.RESTRUCTURE,
            restructure_handler(self.restructure_state),
        )
        self.runtime.register_handler(
            Operation.RECOMBINE,
            recombine_handler(self.recombine_state, rng=self.rng),
        )
