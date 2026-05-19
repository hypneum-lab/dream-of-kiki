"""E-SNN thalamocortical substrate (cycle-2 C2.3 — spike-rate ops).

Second substrate for dreamOfkiki, validating DR-3 substrate-
agnosticism empirically alongside MLX kiki-oniric.

Backend choice :
- `EsnnBackend.NORSE` (default) : declares Norse target ; the
  cycle-2 skeleton uses a numpy-native LIF simulator that
  produces equivalent spike-rate dynamics for validation
  purposes. Swapping to real Norse (PyTorch-based) is an
  implementation detail behind the factory methods.
- `EsnnBackend.NXNET` : Intel Loihi-2 NxSDK/NxNet runtime.
  Requires hardware access ; skeleton falls back to numpy LIF.

The numpy LIF simulator implements the canonical Leaky
Integrate-and-Fire neuron: dv/dt = (-v + I) / tau ; spike when
v >= threshold ; reset v to 0 post-spike. This is sufficient
spike-rate realism for :
- DR-3 Conformance Criterion validation (C2.4) — axiom property
  tests pass on spike-rate state
- Cross-substrate H1-H4 statistical comparison (C2.11)

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md
§6.2 (DR-3 Conformance Criterion)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray


ESNN_SUBSTRATE_NAME = "esnn_thalamocortical"
ESNN_SUBSTRATE_VERSION = "C-v0.7.0+PARTIAL"


class EsnnBackend(str, Enum):
    """Backend choice for the E-SNN substrate."""

    NORSE = "norse"
    NXNET = "nxnet"


@dataclass
class LIFState:
    """Leaky Integrate-and-Fire neuron state.

    - v : membrane potential per neuron (float array)
    - spikes : binary spike output from last step
    """

    n_neurons: int
    v: NDArray = field(init=False)
    spikes: NDArray = field(init=False)

    def __post_init__(self) -> None:
        self.v = np.zeros(self.n_neurons, dtype=float)
        self.spikes = np.zeros(self.n_neurons, dtype=int)


def simulate_lif_step(
    state: LIFState,
    input_current: NDArray,
    dt: float = 1.0,
    tau: float = 10.0,
    threshold: float = 1.0,
) -> LIFState:
    """Single Euler step of LIF dynamics.

    Returns a new LIFState with updated v and spikes arrays.
    """
    new_state = LIFState(n_neurons=state.n_neurons)
    # Leaky decay with current injection:
    # new_v = v * decay + I * dt
    # where decay = exp(-dt/tau) approximated as (1 - dt/tau)
    decay = max(0.0, 1.0 - dt / tau)
    new_v = state.v * decay + input_current * dt
    # Threshold crossing produces spike ; reset to 0
    spikes = (new_v >= threshold).astype(int)
    new_v = np.where(spikes == 1, 0.0, new_v)
    new_state.v = new_v
    new_state.spikes = spikes
    return new_state


def _simulate_population(
    input_trace: NDArray,
    n_steps: int = 20,
    n_neurons: int | None = None,
    dt: float = 1.0,
    tau: float = 10.0,
    threshold: float = 1.0,
) -> NDArray:
    """Simulate a LIF population over n_steps, return spike-rate.

    input_trace: 1-D array of constant input current to each
    neuron. Returns mean spike rate per neuron over the run.
    """
    n = n_neurons or len(input_trace)
    state = LIFState(n_neurons=n)
    spike_sum = np.zeros(n, dtype=float)
    for _ in range(n_steps):
        state = simulate_lif_step(
            state, input_trace, dt=dt, tau=tau, threshold=threshold
        )
        spike_sum += state.spikes
    return spike_sum / n_steps  # average firing rate


@dataclass
class EsnnSubstrate:
    """E-SNN thalamocortical substrate (cycle-2 C2.3)."""

    backend: EsnnBackend = EsnnBackend.NORSE

    def replay_handler_factory(
        self,
    ) -> Callable[..., NDArray]:
        """Build a replay handler using numpy LIF spike-rate sim.

        Handler signature: (beta_records: list[dict], n_steps: int)
        -> spike_rates: NDArray of shape (n_neurons,).

        Maps A-Walker/Stickgold replay to spike-rate retention :
        each record's input drives the population over n_steps,
        and the resulting mean firing rate is the retention
        signal (analogue of gradient magnitude).
        """
        def handler(beta_records, n_steps: int = 20):
            if not beta_records:
                return np.zeros(1, dtype=float)
            # Aggregate input across all records (mean drive)
            all_inputs = [
                np.asarray(r["input"], dtype=float)
                for r in beta_records
                if "input" in r
            ]
            if not all_inputs:
                # All records malformed or missing "input" key
                return np.zeros(1, dtype=float)
            mean_input = np.mean(all_inputs, axis=0)
            return _simulate_population(mean_input, n_steps=n_steps)
        return handler

    def downscale_handler_factory(
        self,
    ) -> Callable[..., NDArray]:
        """Build a downscale handler using synaptic scaling.

        Handler signature: (weights: NDArray, factor: float)
        -> NDArray. Maps B-Tononi SHY to multiplicative synaptic
        scaling. Commutative, not idempotent (shrink_f ∘ shrink_f
        = f²).
        """
        def handler(weights: NDArray, factor: float) -> NDArray:
            if not (0.0 < factor <= 1.0):
                raise ValueError(
                    f"shrink_factor must be in (0, 1], got {factor}"
                )
            return weights * factor
        return handler

    def restructure_handler_factory(
        self,
    ) -> Callable[..., NDArray]:
        """Build a restructure handler modifying connectivity.

        Handler signature: (conn: NDArray, op: str, src: int,
        dst: int) -> NDArray. Maps D-Friston FEP restructure to
        topology edits on the synaptic connectivity matrix.
        Supported ops: "add", "remove", "reroute".
        """
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
                # Swap rows src and dst (outgoing connectivity)
                new_conn[[src, dst]] = new_conn[[dst, src]]
            return new_conn
        return handler

    def recombine_handler_factory(
        self,
    ) -> Callable[..., NDArray]:
        """Build a recombine handler via Poisson spike train mix.

        Handler signature: (latents: NDArray of shape (2, n),
        seed: int, n_steps: int) -> NDArray of shape (n,).
        Maps C-Hobson to rate-coded Poisson sampling :
        interpolate two latents with a random alpha, then
        generate spike train whose mean rate is the mixed latent.
        """
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
            # Ensure non-negative (spike rates can't be negative)
            mixed = np.maximum(mixed, 0.0)
            # Poisson spike counts over n_steps, then normalize
            spike_counts = rng.poisson(lam=mixed * n_steps)
            return spike_counts / n_steps
        return handler


def esnn_substrate_components() -> dict[str, str]:
    """Return the canonical map of E-SNN substrate components.

    Mirrors `mlx_kiki_oniric.mlx_substrate_components()` keys so
    the DR-3 Conformance Criterion test suite can parametrize
    over both substrates. Cycle-2 C2.3 wires ops to numpy LIF.
    """
    return {
        # Substrate-agnostic primitives (shared with MLX)
        "primitives": "kiki_oniric.core.primitives",
        # 4 operations (numpy LIF skeleton in this module)
        "replay": "kiki_oniric.substrates.esnn_thalamocortical",
        "downscale": "kiki_oniric.substrates.esnn_thalamocortical",
        "restructure": "kiki_oniric.substrates.esnn_thalamocortical",
        "recombine": "kiki_oniric.substrates.esnn_thalamocortical",
        # 2 invariant guards (substrate-agnostic, shared)
        "finite": "kiki_oniric.dream.guards.finite",
        "topology": "kiki_oniric.dream.guards.topology",
        # Runtime + swap (substrate-agnostic, shared)
        "runtime": "kiki_oniric.dream.runtime",
        "swap": "kiki_oniric.dream.swap",
        # 3 profiles (substrate-agnostic wrappers, shared)
        "p_min": "kiki_oniric.profiles.p_min",
        "p_equ": "kiki_oniric.profiles.p_equ",
        "p_max": "kiki_oniric.profiles.p_max",
    }
