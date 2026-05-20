"""Substrate factory unifying divergent substrate ABIs behind one Protocol.

The three cycle-3 substrates have incompatible handler signatures:

- ``mlx_kiki_oniric``: operates on ``DreamEpisode`` via ``_real`` ops,
  mutates MLX weights in-place.
- ``esnn_thalamocortical``: numpy-native, ``NDArray`` in/out.
- ``micro_kiki``: OPLoRA adapter dicts (stub) or SpikingKiki-V4 real
  backend (env-gated).

This factory wraps each behind a unified ``SubstrateAdapter`` consumable
by ``DreamRuntime.execute()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, Sequence


SUBSTRATE_NAMES: tuple[str, ...] = (
    "mlx_kiki_oniric",
    "esnn_thalamocortical",
    "micro_kiki",
    "mlx_latent_diffusion",
)


SubstrateName = Literal[
    "mlx_kiki_oniric",
    "esnn_thalamocortical",
    "micro_kiki",
    "mlx_latent_diffusion",
]


@dataclass(frozen=True)
class CellRequest:
    """One ablation cell = (substrate, profile, seed, scale, model_path)."""

    substrate: SubstrateName
    profile: str
    seed: int
    scale: str
    model_path: Path
    benchmarks: Sequence[str] = field(
        default_factory=lambda: ("mmlu", "hellaswag", "mega_v2")
    )


class SubstrateAdapter(Protocol):
    """Unified Protocol for all cycle-3 substrates."""

    def execute_profile(self, request: CellRequest) -> dict:
        """Run pre-eval -> dream ops (profile-ordered) -> post-eval.

        Returns raw cell metrics dict.
        """
        ...

    def teardown(self) -> None:
        """Release MLX Metal buffers / close files."""
        ...


class ESNNAdapter:
    """ESNN (E-SNN thalamocortical) adapter.

    Wraps ``EsnnSubstrate`` handler factories, invokes each with
    synthetic numpy inputs, returns a proxy metrics dict. The real
    H1/H2/H4 metrics come from the runner, not this adapter.
    """

    def __init__(self) -> None:
        # Lazy import so test collection doesn't pull numpy/LIF code
        from kiki_oniric.substrates.esnn_thalamocortical import EsnnSubstrate
        self._substrate = EsnnSubstrate()

    def execute_profile(self, request: CellRequest) -> dict:
        import time
        import numpy as np
        t0 = time.perf_counter()

        n_units = 16
        rng = np.random.default_rng(request.seed)
        beta_records = [{"input": rng.standard_normal(n_units).tolist()}]
        weights = rng.standard_normal((n_units, n_units))
        conn = (rng.random((n_units, n_units)) > 0.5).astype(float)
        latents = rng.standard_normal((2, n_units))

        replay = self._substrate.replay_handler_factory()
        downscale = self._substrate.downscale_handler_factory()
        restructure = self._substrate.restructure_handler_factory()
        recombine = self._substrate.recombine_handler_factory()

        trace = {
            "replay_rate": float(np.mean(replay(beta_records, 20))),
            "downscale_norm": float(
                np.linalg.norm(downscale(weights, 0.5))
            ),
            "restructure_sum": float(
                restructure(conn, "reroute", 0, 1).sum()
            ),
            "recombine_rate": float(
                np.mean(recombine(latents, request.seed, 10))
            ),
        }
        return {
            **trace,
            "delta_acc": trace["replay_rate"],  # proxy
            "wall_time_s": time.perf_counter() - t0,
        }

    def teardown(self) -> None:
        from kiki_oniric.substrates.esnn_thalamocortical import EsnnSubstrate
        self._substrate = EsnnSubstrate()


def build_substrate_adapter(name: SubstrateName) -> SubstrateAdapter:
    """Return a ``SubstrateAdapter`` instance for the requested substrate.

    Thin dispatcher used by the Wave 3b M2 unit tests and by the
    M4 ablation_cycle3 integration. Lazy-imports each adapter so
    importing this module does not pull MLX / numpy / Norse code
    eagerly. Raises ``ValueError`` for unknown names.

    Currently wires:

    - ``mlx_latent_diffusion`` → ``MLXLatentDiffusionSubstrate`` (M2)

    The other substrates (``mlx_kiki_oniric``, ``esnn_thalamocortical``,
    ``micro_kiki``) are reachable via their existing adapter
    classes and are intentionally not re-dispatched here in M2 to
    keep the change additive. Future milestones may extend the
    dispatch table without altering the public signature.
    """
    if name == "mlx_latent_diffusion":
        from kiki_oniric.substrates.mlx_latent_diffusion import (
            MLXLatentDiffusionSubstrate,
        )
        return MLXLatentDiffusionSubstrate()
    if name == "esnn_thalamocortical":
        return ESNNAdapter()
    if name in ("mlx_kiki_oniric", "micro_kiki"):
        raise NotImplementedError(
            f"build_substrate_adapter({name!r}) is not wired in Wave 3b M2. "
            "Use the existing adapter classes directly until a future "
            "milestone extends the dispatcher."
        )
    raise ValueError(
        f"Unknown substrate name: {name!r}. "
        f"Choose from {SUBSTRATE_NAMES}."
    )
