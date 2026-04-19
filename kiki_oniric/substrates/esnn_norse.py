"""Norse SNN fallback wrapper (cycle-3 C3.11 — Phase 2 track b).

CPU-local fallback for Loihi-2 hardware unavailability. Wraps a
PyTorch LIF SNN via the Norse library (``pip install norse``) and
exposes the 8-primitive Protocol surface of framework C so the
DR-3 Conformance Criterion condition (1) applies (same signatures
as the MLX and E-SNN thalamocortical substrates).

Norse (and its ``torch`` parent) are **optional dependencies**.
When either is unimportable the substrate transparently falls
back to a numpy LIF path that produces equivalent spike-rate
dynamics (same pattern as ``esnn_thalamocortical.py``). This
keeps the CI wheel slim and the tests runnable on machines
without a GPU / torch toolchain.

Reference : docs/specs/2026-04-17-dreamofkiki-framework-C-design.md
§6.2 (DR-3 Conformance Criterion), K1 compute budget tagging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

# DualVer C-v0.7.0+PARTIAL — locked in step with the E-SNN
# thalamocortical substrate (cycle-3 Phase 1 bump ; see
# CHANGELOG.md [C-v0.7.0+PARTIAL] and framework-C spec §12.3).
ESNN_NORSE_SUBSTRATE_NAME = "esnn_norse"
ESNN_NORSE_SUBSTRATE_VERSION = "C-v0.7.0+PARTIAL"

# Optional-dependency probe : import torch + norse lazily ; when
# unavailable the fallback numpy LIF path is used. The probe is
# performed at module import so the flag can be introspected by
# callers (`NorseSNNSubstrate.norse_available`). Tests explicitly
# cover the False branch — the True branch only runs on machines
# where `pip install norse` has been performed.
try:  # pragma: no cover - branch depends on env
    import torch  # noqa: F401
    import norse  # noqa: F401
    _NORSE_AVAILABLE = True
except ImportError:  # pragma: no cover - branch depends on env
    _NORSE_AVAILABLE = False


@dataclass
class NorseSNNSubstrate:
    """Norse-backed (or numpy-fallback) LIF SNN substrate.

    Parameters
    ----------
    n_neurons
        Number of LIF neurons in the population. Default 128, a
        modest research-prototype size (not a production network).
    tau_mem
        Membrane time constant in seconds (Norse convention).
    seed
        Numpy RNG seed — controls input noise + any stochastic
        spike sampling in the fallback path.

    Attributes
    ----------
    last_flops
        K1 compute-budget tag : estimated FLOPs of the most recent
        ``forward`` call. Populated at each invocation.
    norse_available
        True when torch + norse imported successfully at module
        load. Informational ; the fallback is always used when
        False.
    """

    n_neurons: int = 128
    tau_mem: float = 0.02
    seed: int = 0
    last_flops: int = field(default=0, init=False)
    norse_available: bool = field(default=_NORSE_AVAILABLE, init=False)
    _net: Any = field(default=None, init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_neurons <= 0:
            raise ValueError(
                f"n_neurons must be > 0, got {self.n_neurons}"
            )
        self._rng = np.random.default_rng(self.seed)
        # If norse is present we could eagerly build a torch
        # LIFCell here ; we defer to forward() to keep import
        # side-effects minimal (and to allow tests to construct
        # the substrate without any torch wheels loaded).
        self._net = None

    # ----- core SNN forward pass -----

    def forward(
        self,
        spike_trains: NDArray[np.floating],
        n_steps: int = 20,
        dt: float = 1.0,
        threshold: float = 1.0,
    ) -> NDArray[np.floating]:
        """Run ``n_steps`` of LIF dynamics on ``spike_trains`` input.

        When Norse is installed we dispatch to
        :meth:`_forward_norse` ; otherwise we fall through to
        :meth:`_forward_numpy_lif`. Both paths return a 1-D numpy
        array of mean spike rates (``n_neurons`` long) and record
        the K1 FLOP estimate on ``last_flops``.
        """
        spike_trains = np.asarray(spike_trains, dtype=float)
        if spike_trains.ndim != 1:
            raise ValueError(
                f"spike_trains must be 1-D, got shape {spike_trains.shape}"
            )
        if spike_trains.shape[0] != self.n_neurons:
            # Broadcast scalar-like input up to population size
            if spike_trains.size == 1:
                spike_trains = np.full(
                    self.n_neurons, float(spike_trains.item()),
                    dtype=float,
                )
            else:
                raise ValueError(
                    f"spike_trains length {spike_trains.shape[0]} "
                    f"!= n_neurons {self.n_neurons}"
                )

        # K1 FLOP tag : simple LIF ≈ 4 FLOPs / neuron / step
        # (decay, add, compare, reset). Recorded BEFORE the
        # simulation so partial failures still leave the tag set.
        self.last_flops = 4 * self.n_neurons * n_steps

        if self.norse_available:  # pragma: no cover - env-gated
            return self._forward_norse(
                spike_trains, n_steps=n_steps, dt=dt,
                threshold=threshold,
            )
        return self._forward_numpy_lif(
            spike_trains, n_steps=n_steps, dt=dt,
            threshold=threshold,
        )

    def _forward_numpy_lif(
        self,
        input_current: NDArray[np.floating],
        n_steps: int,
        dt: float,
        threshold: float,
    ) -> NDArray[np.floating]:
        """Deterministic numpy LIF — fallback path.

        Implements dv/dt = (-v + I) / tau with spike + reset.
        Mirrors the closed-form in ``esnn_thalamocortical``
        so cross-substrate comparisons stay meaningful.
        """
        v = np.zeros(self.n_neurons, dtype=float)
        spike_sum = np.zeros(self.n_neurons, dtype=float)
        # tau_mem is in seconds ; the simulator here uses
        # unit-free dt (Euler step) so decay = exp(-dt/tau).
        decay = float(np.exp(-dt / max(self.tau_mem, 1e-9)))
        for _ in range(n_steps):
            v = v * decay + input_current * dt
            spikes = (v >= threshold).astype(float)
            v = np.where(spikes == 1.0, 0.0, v)
            spike_sum += spikes
        return spike_sum / n_steps

    def _forward_norse(  # pragma: no cover - env-gated
        self,
        input_current: NDArray[np.floating],
        n_steps: int,
        dt: float,
        threshold: float,
    ) -> NDArray[np.floating]:
        """Real Norse LIFCell path.

        Only reached when ``norse_available`` is True. Keeps the
        import local so the module is cheap to load when torch
        isn't installed. Produces the same shape + semantics as
        the numpy fallback for downstream compatibility.
        """
        import torch
        from norse.torch.module.lif import LIFCell

        if self._net is None:
            self._net = LIFCell()
        x = torch.from_numpy(input_current.astype(np.float32))
        state = None
        spike_sum = torch.zeros_like(x)
        for _ in range(n_steps):
            spikes, state = self._net(x, state)
            spike_sum = spike_sum + spikes
        return (spike_sum / n_steps).detach().cpu().numpy()

    # ----- Protocol-contract factories (mirror E-SNN thalamocortical) -----

    def replay_handler_factory(
        self,
    ) -> Callable[[list[dict], int], NDArray]:
        """A-Walker/Stickgold replay → spike-rate retention."""

        def handler(beta_records, n_steps: int = 20):
            if not beta_records:
                return np.zeros(1, dtype=float)
            all_inputs = [
                np.asarray(r["input"], dtype=float)
                for r in beta_records
                if "input" in r
            ]
            if not all_inputs:
                return np.zeros(1, dtype=float)
            mean_input = np.mean(all_inputs, axis=0)
            # Use a temporary substrate sized to the input for
            # replay (population size varies per episode). This
            # matches the shape contract tested in the E-SNN
            # conformance suite.
            sub = NorseSNNSubstrate(
                n_neurons=int(mean_input.shape[0]),
                tau_mem=self.tau_mem,
                seed=self.seed,
            )
            return sub.forward(mean_input, n_steps=n_steps)

        return handler

    def downscale_handler_factory(
        self,
    ) -> Callable[[NDArray, float], NDArray]:
        """B-Tononi SHY → multiplicative synaptic scaling."""

        def handler(weights: NDArray, factor: float) -> NDArray:
            if not (0.0 < factor <= 1.0):
                raise ValueError(
                    f"shrink_factor must be in (0, 1], got {factor}"
                )
            return weights * factor

        return handler

    def restructure_handler_factory(
        self,
    ) -> Callable[[NDArray, str, int, int], NDArray]:
        """D-Friston FEP restructure → topology edits."""

        def handler(
            conn: NDArray, op: str, src: int = 0, dst: int = 1
        ) -> NDArray:
            valid_ops = {"add", "remove", "reroute"}
            if op not in valid_ops:
                raise ValueError(
                    f"op must be one of {sorted(valid_ops)}, "
                    f"got {op!r}"
                )
            new_conn = conn.copy()
            if op == "add":
                new_conn[src, dst] = 1.0
            elif op == "remove":
                new_conn[src, dst] = 0.0
            elif op == "reroute":
                new_conn[[src, dst]] = new_conn[[dst, src]]
            return new_conn

        return handler

    def recombine_handler_factory(
        self,
    ) -> Callable[[NDArray, int, int], NDArray]:
        """C-Hobson recombine → Poisson spike-train mix of latents."""

        def handler(
            latents: NDArray, seed: int = 0, n_steps: int = 10
        ) -> NDArray:
            if latents.shape[0] < 2:
                raise ValueError(
                    f"recombine needs >=2 latents, got "
                    f"{latents.shape[0]}"
                )
            rng = np.random.default_rng(seed)
            alpha = rng.random()
            mixed = alpha * latents[0] + (1 - alpha) * latents[1]
            mixed = np.maximum(mixed, 0.0)
            spike_counts = rng.poisson(lam=mixed * n_steps)
            return spike_counts / n_steps

        return handler


def norse_substrate_components() -> dict[str, str]:
    """Return the canonical map of Norse substrate components.

    Mirrors ``esnn_thalamocortical.esnn_substrate_components`` so
    the DR-3 Conformance Criterion test suite parametrizes over
    the three substrates (MLX / E-SNN numpy / Norse) uniformly.
    """
    return {
        "primitives": "kiki_oniric.core.primitives",
        "replay": "kiki_oniric.substrates.esnn_norse",
        "downscale": "kiki_oniric.substrates.esnn_norse",
        "restructure": "kiki_oniric.substrates.esnn_norse",
        "recombine": "kiki_oniric.substrates.esnn_norse",
        "finite": "kiki_oniric.dream.guards.finite",
        "topology": "kiki_oniric.dream.guards.topology",
        "runtime": "kiki_oniric.dream.runtime",
        "swap": "kiki_oniric.dream.swap",
        "p_min": "kiki_oniric.profiles.p_min",
        "p_equ": "kiki_oniric.profiles.p_equ",
        "p_max": "kiki_oniric.profiles.p_max",
    }
