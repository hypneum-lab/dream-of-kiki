"""E-SNN spiking-CNN dream-episode wrapper for the G5-ter pilot.

Couples REPLAY+DOWNSCALE+RESTRUCTURE+RECOMBINE to the 4-layer
spiking-CNN classifier weights ``(W_c1, b_c1, W_c2, b_c2, W_fc1,
b_fc1, W_out, b_out)``. The four step functions are free
functions taking the classifier as first arg ;
``dream_episode_cnn_esnn`` orchestrates them following the same
P_min / P_equ / P_max op-set selection used in
``experiments.g5_bis_richer_esnn.esnn_dream_wrap_hier``.

DR-0 accountability is preserved : ``profile.runtime.execute(
episode)`` is called before any weight mutation so every episode
appends one ``EpisodeLogEntry`` to ``profile.runtime.log``
regardless of substrate-side outcome.

The profile is built via
``experiments.g5_cross_substrate.esnn_dream_wrap.{PROFILE_FACTORIES,
_rebind_to_esnn}`` — the same cross-substrate dispatch path
G5 / G5-bis exercised.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md sec 3.1
    docs/osf-prereg-g5-ter-spiking-cnn.md sec 2
    experiments/g5_bis_richer_esnn/esnn_dream_wrap_hier.py (sister)
"""
from __future__ import annotations

from collections import deque
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from experiments.g5_cross_substrate.esnn_dream_wrap import (
    PROFILE_FACTORIES,
    ProfileT,
    _rebind_to_esnn,
)
from experiments.g5_ter_spiking_cnn.spiking_cnn import (
    EsnnG5TerSpikingCNN,
)
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.profiles.p_min import PMinProfile

__all__ = [
    "BetaRecordCNN",
    "EsnnCNNBetaBuffer",
    "ProfileT",
    "build_esnn_cnn_profile",
    "dream_episode_cnn_esnn",
]


# Names of the 8 trainable tensors on EsnnG5TerSpikingCNN, in
# DOWNSCALE iteration order.
_DOWNSCALE_TENSOR_NAMES: tuple[str, ...] = (
    "W_c1",
    "b_c1",
    "W_c2",
    "b_c2",
    "W_fc1",
    "b_fc1",
    "W_out",
    "b_out",
)


class BetaRecordCNN(TypedDict, total=False):
    """One curated episodic exemplar for the spiking-CNN head.

    ``x`` carries the NHWC nested-list view of the original
    ``(32, 32, 3)`` float32 image. ``latent`` is the post-fc1 LIF
    rate vector at push time (or None for pre-warmup pushes).
    """

    x: list  # nested NHWC list (3-deep)
    y: int
    latent: list[float] | None


class EsnnCNNBetaBuffer:
    """Bounded curated episodic buffer with optional latents.

    FIFO eviction at capacity. Stores ``x`` as the nested-list NHWC
    view of a ``(32, 32, 3)`` float32 image — round-trips through
    ``np.asarray(...).reshape(-1, 32, 32, 3)`` exactly.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(
                f"capacity must be positive, got {capacity}"
            )
        self._capacity = capacity
        self._records: deque[BetaRecordCNN] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._records)

    def push(
        self,
        *,
        x: NDArray[np.float32],
        y: int,
        latent: NDArray[np.float32] | None,
    ) -> None:
        record: BetaRecordCNN = {
            "x": x.astype(np.float32).tolist(),
            "y": int(y),
            "latent": (
                latent.astype(np.float32).tolist()
                if latent is not None
                else None
            ),
        }
        self._records.append(record)

    def snapshot(self) -> list[BetaRecordCNN]:
        return [
            {
                "x": r["x"],
                "y": int(r["y"]),
                "latent": (
                    list(r["latent"])
                    if r["latent"] is not None
                    else None
                ),
            }
            for r in self._records
        ]

    def sample(self, n: int, seed: int) -> list[BetaRecordCNN]:
        n_avail = len(self._records)
        if n_avail == 0:
            return []
        rng = np.random.default_rng(seed)
        n_take = min(n, n_avail)
        indices = rng.choice(n_avail, size=n_take, replace=False)
        snapshot = list(self._records)
        return [
            {
                "x": snapshot[i]["x"],
                "y": int(snapshot[i]["y"]),
                "latent": (
                    list(snapshot[i]["latent"])
                    if snapshot[i]["latent"] is not None
                    else None
                ),
            }
            for i in sorted(indices.tolist())
        ]


def build_esnn_cnn_profile(name: str, seed: int) -> ProfileT:
    """Construct a profile and rebind its op handlers to E-SNN adapters.

    Reuses ``PROFILE_FACTORIES`` and ``_rebind_to_esnn`` from
    ``experiments.g5_cross_substrate.esnn_dream_wrap`` — the same
    cross-substrate dispatch path G5 / G5-bis exercised. The
    profile API is unchanged from the caller's point of view :
    ``runtime.execute(episode)`` dispatches through the rebound
    handlers.
    """
    if name not in PROFILE_FACTORIES:
        raise ValueError(
            f"unknown profile {name!r} - expected one of "
            f"{sorted(PROFILE_FACTORIES)}"
        )
    factory = PROFILE_FACTORIES[name]
    if name == "P_min":
        profile: ProfileT = factory()
    else:
        import random

        profile = factory(rng=random.Random(seed))
    _rebind_to_esnn(profile)
    return profile


# -------------------- coupling steps (free functions) --------------------


def _replay_step(
    clf: EsnnG5TerSpikingCNN,
    records: list[BetaRecordCNN],
    *,
    lr: float,
    n_steps: int,
) -> None:
    """SGD-with-STE over sampled buffer records.

    No-op if ``records`` empty, ``lr <= 0``, or ``n_steps <= 0``.
    Stacks the nested-list ``x`` records back into a
    ``(N, 32, 32, 3)`` float32 NHWC tensor.
    """
    if not records or lr <= 0.0 or n_steps <= 0:
        return
    x = np.asarray(
        [r["x"] for r in records], dtype=np.float32
    )
    y = np.asarray([r["y"] for r in records], dtype=np.int64)
    for _ in range(n_steps):
        clf._ste_backward(x, y, lr)


def _downscale_step(
    clf: EsnnG5TerSpikingCNN, *, factor: float
) -> None:
    """Multiply all 8 weight + bias tensors by ``factor``.

    Bounds : ``factor`` must lie in ``(0, 1]`` — same constraint as
    the MLX sister and the substrate's ``downscale_real_handler``.
    """
    if not (0.0 < factor <= 1.0):
        raise ValueError(
            f"shrink_factor must be in (0, 1], got {factor}"
        )
    f32 = np.float32(factor)
    for name in _DOWNSCALE_TENSOR_NAMES:
        tensor = getattr(clf, name)
        setattr(clf, name, (tensor * f32).astype(np.float32))


def _restructure_step(
    clf: EsnnG5TerSpikingCNN,
    *,
    factor: float,
    seed: int,
) -> None:
    """Add ``factor * sigma * N(0, 1)`` noise to ``W_c2`` only.

    ``sigma`` is the standard deviation of ``W_c2`` at call time.
    ``factor=0`` is a no-op (early exit) ; ``factor < 0`` raises.
    Mirrors the MLX small-CNN ``restructure_step`` (perturbs the
    second conv layer's weights only, leaving the input projection
    and output classifier intact).
    """
    if factor < 0.0:
        raise ValueError(
            f"factor must be non-negative, got {factor}"
        )
    if factor == 0.0:
        return
    sigma = float(clf.W_c2.std()) if clf.W_c2.size > 0 else 0.0
    if sigma == 0.0:
        return
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=clf.W_c2.shape).astype(np.float32)
    clf.W_c2 = (
        clf.W_c2 + np.float32(factor) * np.float32(sigma) * noise
    ).astype(np.float32)


def _recombine_step(
    clf: EsnnG5TerSpikingCNN,
    *,
    latents: list[tuple[list[float], int]],
    n_synthetic: int,
    lr: float,
    seed: int,
) -> None:
    """One CE-loss SGD step on ``(W_out, b_out)`` over per-class MoG samples.

    Empty ``latents`` -> no-op. Single-class latents -> no-op
    (degenerate MoG). For each class, estimate ``(mean, std+1e-6)``
    over the population of recorded latents at that class, sample
    ``n_synthetic // n_classes`` synthetic vectors per class, then
    run one softmax-CE SGD step on ``(W_out, b_out)``. Leaves all
    convolutional and fc1 tensors untouched.
    """
    if not latents:
        return
    classes = sorted({lbl for _, lbl in latents})
    if len(classes) < 2:
        return

    rng = np.random.default_rng(seed)
    components: dict[
        int, tuple[NDArray[np.float32], NDArray[np.float32]]
    ] = {}
    for c in classes:
        arr = np.asarray(
            [lat for lat, lbl in latents if lbl == c],
            dtype=np.float32,
        )
        mean = arr.mean(axis=0).astype(np.float32)
        std = (arr.std(axis=0) + 1e-6).astype(np.float32)
        components[c] = (mean, std)

    per_class = max(1, n_synthetic // len(classes))
    synth_x: list[NDArray[np.float32]] = []
    synth_y: list[int] = []
    for c in classes:
        mean, std = components[c]
        for _ in range(per_class):
            synth_x.append(
                (
                    mean
                    + std
                    * rng.standard_normal(size=mean.shape).astype(
                        np.float32
                    )
                ).astype(np.float32)
            )
            synth_y.append(c)

    rates = np.stack(synth_x).astype(np.float32)
    y_arr = np.asarray(synth_y, dtype=np.int64)
    logits = (rates @ clf.W_out + clf.b_out).astype(np.float32)
    logits_shift = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits_shift)
    probs = exp / exp.sum(axis=1, keepdims=True)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_arr)), y_arr] = 1.0
    d_logits = ((probs - one_hot) / max(len(y_arr), 1)).astype(
        np.float32
    )
    d_W_out = (rates.T @ d_logits).astype(np.float32)
    d_b_out = d_logits.sum(axis=0).astype(np.float32)
    clf.W_out = (
        clf.W_out - np.float32(lr) * d_W_out
    ).astype(np.float32)
    clf.b_out = (
        clf.b_out - np.float32(lr) * d_b_out
    ).astype(np.float32)


# -------------------- orchestration --------------------


def dream_episode_cnn_esnn(
    clf: EsnnG5TerSpikingCNN,
    profile: ProfileT,
    seed: int,
    *,
    beta_buffer: EsnnCNNBetaBuffer,
    replay_n_records: int,
    replay_n_steps: int,
    replay_lr: float,
    downscale_factor: float,
    restructure_factor: float,
    recombine_n_synthetic: int,
    recombine_lr: float,
) -> None:
    """Drive one DreamEpisode and mutate spiking-CNN weights.

    Coupling map :
        - Operation.REPLAY    -> ``_replay_step`` over buffer sample
        - Operation.DOWNSCALE -> ``_downscale_step`` on all 8 tensors
        - Operation.RESTRUCTURE -> ``_restructure_step`` on W_c2 only
        - Operation.RECOMBINE -> ``_recombine_step`` on (W_out, b_out)

    P_min runs REPLAY+DOWNSCALE only ; P_equ / P_max run all four
    ops. DR-0 spectator runtime path is exercised before any
    substrate-side mutation.
    """
    if isinstance(profile, PMinProfile):
        ops: tuple[Operation, ...] = (
            Operation.REPLAY,
            Operation.DOWNSCALE,
        )
        channels: tuple[OutputChannel, ...] = (
            OutputChannel.WEIGHT_DELTA,
        )
    else:
        ops = (
            Operation.REPLAY,
            Operation.DOWNSCALE,
            Operation.RESTRUCTURE,
            Operation.RECOMBINE,
        )
        channels = (
            OutputChannel.WEIGHT_DELTA,
            OutputChannel.HIERARCHY_CHG,
            OutputChannel.LATENT_SAMPLE,
        )

    # DR-0 spectator runtime path : synthetic input_slice consistent
    # with the E-SNN handler factories' expectations.
    rng = np.random.default_rng(seed + 10_000)
    synthetic_records = [
        {
            "x": rng.standard_normal(4).astype(np.float32).tolist(),
            "y": rng.standard_normal(4).astype(np.float32).tolist(),
            "input": rng.standard_normal(4).astype(np.float32).tolist(),
        }
        for _ in range(4)
    ]
    delta_latents = [
        rng.standard_normal(4).astype(np.float32).tolist()
        for _ in range(2)
    ]
    episode = DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={
            "beta_records": synthetic_records,
            "shrink_factor": 0.99,
            "topo_op": "reroute",
            "swap_indices": [0, 1],
            "delta_latents": delta_latents,
            "seed": seed,
            "n_steps": 20,
        },
        operation_set=ops,
        output_channels=channels,
        budget=BudgetCap(
            flops=10_000_000, wall_time_s=10.0, energy_j=1.0
        ),
        episode_id=f"g5ter-{type(profile).__name__}-seed{seed}",
    )
    profile.runtime.execute(episode)

    # ---- G5-ter coupling on spiking-CNN substrate ----
    if Operation.REPLAY in ops:
        sampled = beta_buffer.sample(n=replay_n_records, seed=seed)
        _replay_step(clf, sampled, lr=replay_lr, n_steps=replay_n_steps)
    if Operation.DOWNSCALE in ops:
        _downscale_step(clf, factor=downscale_factor)
    if Operation.RESTRUCTURE in ops:
        _restructure_step(
            clf, factor=restructure_factor, seed=seed + 20_000
        )
    if Operation.RECOMBINE in ops:
        latents_records: list[tuple[list[float], int]] = []
        for r in beta_buffer.snapshot():
            lat = r["latent"]
            if lat is not None:
                latents_records.append((list(lat), int(r["y"])))
        _recombine_step(
            clf,
            latents=latents_records,
            n_synthetic=recombine_n_synthetic,
            lr=recombine_lr,
            seed=seed + 30_000,
        )
