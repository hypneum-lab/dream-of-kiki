"""P_max LoRA-substrate profile (B6c, issue #15 continuation).

Subclass of ``PMaxProfile`` that wires the full B-series :
- ``replay_lora_handler`` (B1b) → ch1
- ``downscale_lora_handler`` (B2) → ch1
- ``restructure_lora_handler`` (B3) → ch3
- ``recombine_real_handler`` (B4 VAE) → ch2

Channels out (per framework-C spec §3.1, primitives_out={1,2,3,4}):
- ch1 ``WeightUpdate`` via ``LoRAWeightDeltaChannel`` (B5).
- ch2 ``LatentSample`` via ``LatentSampleQueue`` (B5).
- ch3 ``TopologyDiff`` via ``LoRAHierarchyChangeChannel`` (B5).
- ch4 ``AttentionPrior`` via inherited ``AttentionPriorChannel`` —
  state surface only, populated externally via
  ``profile.attention_prior.set_prior(prior)``. No op emits.

Input channel α : ``AlphaStreamBuffer`` (inherited from
``PMaxProfile``), populated externally by the awake side via
``profile.alpha_stream.append(TraceRecord(...))``. Not dispatched
by ``consolidate_log()`` (α is awake → dream input, not output).

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kiki_oniric.dream.channels.hierarchy_change import (
    LoRAHierarchyChangeChannel,
)
from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue
from kiki_oniric.dream.channels.weight_delta import (
    LoRAWeightDeltaChannel,
)
from kiki_oniric.dream.episode import Operation
from kiki_oniric.dream.operations.downscale_real import (
    DownscaleRealState,
    downscale_lora_handler,
)
from kiki_oniric.dream.operations.recombine_real import (
    RecombineRealState,
    recombine_real_handler,
)
from kiki_oniric.dream.operations.replay_real import (
    ReplayRealState,
    replay_lora_handler,
)
from kiki_oniric.dream.operations.restructure_real import (
    RestructureRealState,
    restructure_lora_handler,
)
from kiki_oniric.profiles.p_max import PMaxProfile

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


@dataclass(kw_only=True)
class PMaxLoRAProfile(PMaxProfile):
    """P_max rewired for the B-series LoRA substrate + VAE recombine.

    Required kwargs : ``dream_model``, ``awake_model`` (LoRAModel
    pair), ``encoder``, ``decoder`` (MLX nn.Module VAE pair for
    ``recombine_real_handler``).

    Optional kwargs : ``lr=0.01``, ``max_adds_per_episode=1``,
    ``seed=0``, ``latent_queue_capacity=1024``.

    Parent state fields (``replay_state``, ``downscale_state``,
    ``restructure_state``, ``recombine_state``) are widened from
    cycle-3 skeleton ``OpState`` types to ``_RealState`` variants
    required by the B-series LoRA / VAE handlers.

    Inherited from ``PMaxProfile`` cycle-3 :
    - ``alpha_stream: AlphaStreamBuffer`` — awake → dream input
      channel state surface.
    - ``attention_prior: AttentionPriorChannel`` — ch4 state
      surface.
    - ``rng: random.Random`` — kept on the dataclass but unused
      by the LoRA handlers (which use MLX RNG keyed off
      ``seed``).

    ``__post_init__`` intentionally does NOT call
    ``super().__post_init__()`` — the parent registers cycle-3
    skeleton handlers ; we register the B-series LoRA / VAE
    variants on the same runtime instead.
    """

    dream_model: "LoRAModel"
    awake_model: "LoRAModel"
    encoder: Any
    decoder: Any
    lr: float = 0.01
    max_adds_per_episode: int = 1
    seed: int = 0
    latent_queue_capacity: int = 1024
    replay_state: ReplayRealState = field(  # type: ignore[assignment]
        default_factory=ReplayRealState,
    )
    downscale_state: DownscaleRealState = field(  # type: ignore[assignment]
        default_factory=DownscaleRealState,
    )
    restructure_state: RestructureRealState = field(  # type: ignore[assignment]
        default_factory=RestructureRealState,
    )
    recombine_state: RecombineRealState = field(  # type: ignore[assignment]
        default_factory=RecombineRealState,
    )
    weight_channel: LoRAWeightDeltaChannel | None = None
    hierarchy_channel: LoRAHierarchyChangeChannel | None = None
    latent_channel: LatentSampleQueue | None = None

    def __post_init__(self) -> None:
        # Do NOT call super().__post_init__() — parent registers
        # cycle-3 skeleton handlers; we register the B-series
        # variants (replay/downscale/restructure LoRA + recombine
        # VAE) on the same runtime instead.
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
            recombine_real_handler(
                self.recombine_state,
                encoder=self.encoder,
                decoder=self.decoder,
                seed=self.seed,
            ),
        )
        self.weight_channel = LoRAWeightDeltaChannel(self.awake_model)
        self.hierarchy_channel = LoRAHierarchyChangeChannel(
            self.awake_model,
        )
        self.latent_channel = LatentSampleQueue(
            maxlen=self.latent_queue_capacity,
        )

    def consolidate_log(self) -> int:
        """Dispatch the runtime log onto awake-side channels :
        ch1 (``WeightUpdate``) via ``weight_channel``, ch2
        (``LatentSample``) via ``latent_channel`` (queue), ch3
        (``TopologyDiff``) via ``hierarchy_channel``. Then clear
        the log.

        ``attention_channel`` defaults to ``None`` (apply_channel
        _outputs's relaxed kwargs since B6a) because no op
        currently emits ``AttentionPrior`` into the runtime log ;
        the profile's ``attention_prior`` field is a state
        surface for external callers.

        The α input channel is not in the apply loop : it carries
        awake → dream traces, populated by the awake side via
        ``profile.alpha_stream.append(...)``.

        Returns the number of channel outputs dispatched. The
        log is cleared on success so a second call without
        further ``runtime.execute()`` returns 0 (idempotent
        no-op).
        """
        from kiki_oniric.consolidate import apply_channel_outputs

        count = apply_channel_outputs(
            self.runtime.log,
            weight_channel=self.weight_channel,
            hierarchy_channel=self.hierarchy_channel,
            latent_channel=self.latent_channel,
        )
        self.runtime.reset_log()
        return count
