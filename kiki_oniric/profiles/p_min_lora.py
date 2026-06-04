"""P_min LoRA-substrate profile (B6a, issue #15 continuation).

Subclass of ``PMinProfile`` that wires the B-series LoRA-emitting
handlers (``replay_lora_handler``, ``downscale_lora_handler``)
onto a dream/awake ``LoRAModel`` pair, and exposes
``consolidate_log()`` to apply the runtime log onto the awake
model via the B5 ``LoRAWeightDeltaChannel``.

Channels out (per framework-C spec §3.1) : ``{WEIGHT_DELTA}``
only. Neither ``TopologyDiff`` nor ``LatentSample`` nor
``AttentionPrior`` is emitted by P_min's op set.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kiki_oniric.dream.channels.weight_delta import (
    LoRAWeightDeltaChannel,
)
from kiki_oniric.dream.episode import Operation
from kiki_oniric.dream.operations.downscale_real import (
    DownscaleRealState,
    downscale_lora_handler,
)
from kiki_oniric.dream.operations.replay_real import (
    ReplayRealState,
    replay_lora_handler,
)
from kiki_oniric.profiles.p_min import PMinProfile

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


@dataclass(kw_only=True)
class PMinLoRAProfile(PMinProfile):
    """P_min rewired for the B-series LoRA substrate.

    Required kwargs: ``dream_model`` and ``awake_model`` — two
    ``LoRAModel`` instances. For within-machine bit-exact
    reproducibility under ``consolidate_log()``, build both with
    the same ``seed`` so the awake model starts as a bit-clone of
    the dream model.

    The parent's ``replay_state`` / ``downscale_state`` fields
    (typed ``ReplayOpState`` / ``DownscaleOpState`` for the
    skeleton handlers) are overridden to the ``_RealState``
    variants required by the LoRA handlers.

    ``__post_init__`` intentionally does NOT call
    ``super().__post_init__()`` — the parent registers skeleton
    handlers ; we register the LoRA-emitting variants on the
    same runtime instead.
    """

    dream_model: "LoRAModel"
    awake_model: "LoRAModel"
    lr: float = 0.01
    # Override parent state types — LoRA handlers need _RealState.
    replay_state: ReplayRealState = field(  # type: ignore[assignment]
        default_factory=ReplayRealState,
    )
    downscale_state: DownscaleRealState = field(  # type: ignore[assignment]
        default_factory=DownscaleRealState,
    )
    weight_channel: LoRAWeightDeltaChannel | None = None

    def __post_init__(self) -> None:
        # Intentionally do NOT call super().__post_init__() — the
        # parent registers skeleton handlers. We register the
        # LoRA-emitting variants instead, on the dream model.
        self.runtime.register_handler(
            Operation.REPLAY,
            replay_lora_handler(
                self.replay_state,
                model=self.dream_model,
                lr=self.lr,
            ),
        )
        self.runtime.register_handler(
            Operation.DOWNSCALE,
            downscale_lora_handler(
                self.downscale_state,
                model=self.dream_model,
            ),
        )
        self.weight_channel = LoRAWeightDeltaChannel(self.awake_model)

    def consolidate_log(self) -> int:
        """Replay every ``WeightUpdate`` in the runtime log onto
        ``awake_model`` via ``weight_channel``, then clear the log.

        Returns the number of channel outputs dispatched. The log
        is cleared on success so a second call without further
        ``runtime.execute()`` returns 0 (idempotent no-op).
        """
        from kiki_oniric.consolidate import apply_channel_outputs

        count = apply_channel_outputs(
            self.runtime.log,
            weight_channel=self.weight_channel,
        )
        self.runtime.reset_log()
        return count
