"""P_equ LoRA-substrate profile (B6b, issue #15 continuation).

Subclass of ``PEquProfile`` that wires the B-series LoRA-emitting
handlers (``replay_lora_handler``, ``downscale_lora_handler``,
``restructure_lora_handler``) onto a dream/awake ``LoRAModel`` pair
and the skeleton ``recombine_handler`` for ``recombine_light``.

Channels out (per framework-C spec §3.1) :
- ch1 (``WeightUpdate``) via ``LoRAWeightDeltaChannel`` (B5).
- ch3 (``TopologyDiff``) via ``LoRAHierarchyChangeChannel`` (B5).
- ch4 (``AttentionPrior``) via ``AttentionPriorChannel`` — *state
  surface only*, populated externally via
  ``profile.attention_prior.set_prior(prior)``. No op currently
  emits ``AttentionPrior`` into the runtime log.
- ch2 (``LatentSample``) NOT in P_equ ; ``recombine_light`` returns
  ``None``.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kiki_oniric.dream.channels.attention_prior import (
    AttentionPriorChannel,
)
from kiki_oniric.dream.channels.hierarchy_change import (
    LoRAHierarchyChangeChannel,
)
from kiki_oniric.dream.channels.weight_delta import (
    LoRAWeightDeltaChannel,
)
from kiki_oniric.dream.episode import Operation
from kiki_oniric.dream.operations.downscale_real import (
    DownscaleRealState,
    downscale_lora_handler,
)
from kiki_oniric.dream.operations.recombine import (
    recombine_handler,
)
from kiki_oniric.dream.operations.replay_real import (
    ReplayRealState,
    replay_lora_handler,
)
from kiki_oniric.dream.operations.restructure_real import (
    RestructureRealState,
    restructure_lora_handler,
)
from kiki_oniric.profiles.p_equ import PEquProfile

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


_DEFAULT_ATTENTION_BUDGET = 1.5


@dataclass(kw_only=True)
class PEquLoRAProfile(PEquProfile):
    """P_equ rewired for the B-series LoRA substrate.

    Required kwargs : ``dream_model`` and ``awake_model`` —
    ``LoRAModel`` instances. For within-machine bit-exact
    reproducibility under ``consolidate_log()``, build both at the
    same ``seed`` so the awake model starts as a bit-clone of the
    dream model.

    Parent state fields (``replay_state``, ``downscale_state``,
    ``restructure_state``) are widened from their cycle-3 skeleton
    types to the ``_RealState`` variants required by the LoRA
    handlers. ``recombine_state`` keeps its ``RecombineOpState``
    type because the skeleton ``recombine_handler`` is what
    services ``recombine_light``.

    ``__post_init__`` intentionally does NOT call
    ``super().__post_init__()`` — the parent registers cycle-3
    skeleton handlers ; we register the LoRA-emitting variants
    (for replay / downscale / restructure) plus the skeleton
    handler (for recombine_light) on the same runtime instead.
    """

    dream_model: "LoRAModel"
    awake_model: "LoRAModel"
    lr: float = 0.01
    max_adds_per_episode: int = 1
    seed: int = 0
    # Override parent state types — LoRA handlers need _RealState.
    replay_state: ReplayRealState = field(  # type: ignore[assignment]
        default_factory=ReplayRealState,
    )
    downscale_state: DownscaleRealState = field(  # type: ignore[assignment]
        default_factory=DownscaleRealState,
    )
    restructure_state: RestructureRealState = field(  # type: ignore[assignment]
        default_factory=RestructureRealState,
    )
    # recombine_state STAYS at RecombineOpState — skeleton handler.
    weight_channel: LoRAWeightDeltaChannel | None = None
    hierarchy_channel: LoRAHierarchyChangeChannel | None = None
    attention_prior: AttentionPriorChannel = field(
        default_factory=lambda: AttentionPriorChannel(
            budget_attention=_DEFAULT_ATTENTION_BUDGET,
        ),
    )

    def __post_init__(self) -> None:
        # Do NOT call super().__post_init__() — parent registers
        # cycle-3 skeleton handlers; we register the LoRA variants
        # for the three emitting ops + skeleton for recombine_light.
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
        self.runtime.register_handler(
            Operation.RESTRUCTURE,
            restructure_lora_handler(
                self.restructure_state,
                model=self.dream_model,
                max_adds_per_episode=self.max_adds_per_episode,
                seed=self.seed,
            ),
        )
        self.runtime.register_handler(
            Operation.RECOMBINE,
            recombine_handler(self.recombine_state, rng=self.rng),
        )
        self.weight_channel = LoRAWeightDeltaChannel(self.awake_model)
        self.hierarchy_channel = LoRAHierarchyChangeChannel(
            self.awake_model,
        )

    def consolidate_log(self) -> int:
        """Dispatch the runtime log onto the awake model via
        ``weight_channel`` (ch1) and ``hierarchy_channel`` (ch3),
        then clear the log.

        ``latent_channel`` is ``None`` because ``recombine_light``
        returns ``None`` (no ch2 in P_equ). ``attention_channel``
        is ``None`` because no op currently emits ``AttentionPrior``
        into the runtime log ; the profile's ``attention_prior``
        field is a state surface for external callers, not a
        dispatch target.

        Returns the number of channel outputs dispatched. The log
        is cleared on success so a second call without further
        ``runtime.execute()`` returns 0 (idempotent no-op).
        """
        from kiki_oniric.consolidate import apply_channel_outputs

        count = apply_channel_outputs(
            self.runtime.log,
            weight_channel=self.weight_channel,
            hierarchy_channel=self.hierarchy_channel,
        )
        self.runtime.reset_log()
        return count
