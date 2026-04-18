"""P_max profile — maximalist consolidation (C2.8 fully wired).

Channels: α + β + δ → 1 + 2 + 3 + 4 (full input + full output set).
Operations: {replay, downscale, restructure, recombine_full}.

C2.8 promotion from skeleton (cycle-1 S16.1) to fully wired
profile :
- 4 op handlers registered on a fresh DreamRuntime
- 4 op states tracked as fields
- alpha_stream ring buffer (C2.5) for canal-α input
- attention_prior canal-4 channel (C2.7) with S4 guard
- recombine_full handler skeleton variant (real MLX VAE wiring
  via recombine_handler_full_mlx C2.6 deferred to runtime
  invocation with concrete encoder/decoder)

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from kiki_oniric.dream.channels.alpha_stream import (
    AlphaStreamBuffer,
)
from kiki_oniric.dream.channels.attention_prior import (
    AttentionPriorChannel,
)
from kiki_oniric.dream.episode import Operation, OutputChannel
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


_DEFAULT_ALPHA_CAPACITY = 1024
_DEFAULT_ATTENTION_BUDGET = 1.5


@dataclass
class PMaxProfile:
    """Maximalist profile with 4 ops + 4 output channels wired."""

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
    alpha_stream: AlphaStreamBuffer = field(
        default_factory=lambda: AlphaStreamBuffer(
            capacity=_DEFAULT_ALPHA_CAPACITY, order="fifo"
        )
    )
    attention_prior: AttentionPriorChannel = field(
        default_factory=lambda: AttentionPriorChannel(
            budget_attention=_DEFAULT_ATTENTION_BUDGET
        )
    )
    rng: random.Random = field(default_factory=random.Random)
    target_ops: set[Operation] = field(
        default_factory=lambda: {
            Operation.REPLAY,
            Operation.DOWNSCALE,
            Operation.RESTRUCTURE,
            Operation.RECOMBINE,
        }
    )
    target_channels_out: set[OutputChannel] = field(
        default_factory=lambda: {
            OutputChannel.WEIGHT_DELTA,
            OutputChannel.LATENT_SAMPLE,
            OutputChannel.HIERARCHY_CHG,
            OutputChannel.ATTENTION_PRIOR,
        }
    )

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
