"""Dream-episode coupling for the G6 pilot.

Wires the four MicroKikiSubstrate handler factories (replay,
downscale, restructure (OPLoRA), recombine (TIES-Merge)) into a
profile-aware runner that fires one episode per subdomain transition.
baseline arm runs no episode; P_min runs replay + downscale only;
P_equ / P_max run all four.

DR-0 / DR-1 stamps land on substrate._restructure_state and
substrate._recombine_state via the handler closures. The wrapper
holds its own ``episodes_run`` + ``last_episode_id`` counters because
the replay and downscale handlers on MicroKikiSubstrate are *pure*
(they return tensors and do not mutate accumulator state — see
``micro_kiki.py`` docstrings).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate


PROFILE_OPS: dict[str, tuple[str, ...]] = {
    "baseline": (),  # baseline = no episode
    "P_min": ("replay", "downscale"),
    "P_equ": ("replay", "downscale", "restructure", "recombine"),
    "P_max": ("replay", "downscale", "restructure", "recombine"),
}


def build_episode_payload(
    *,
    seed: int,
    adapter_keys: tuple[str, ...],
    out_dim: int,
    rank: int,
) -> dict[str, Any]:
    """Build a synthetic dream-episode payload for the four handlers.

    The payload's shape mirrors the contract documented in
    ``MicroKikiSubstrate.{replay, downscale, restructure,
    recombine}_handler_factory``.

    Parameters
    ----------
    seed
        Pins the RNG so the same seed always produces the same
        payload.
    adapter_keys
        LoRA tensor keys to include in the synthetic adapter dict
        (used by restructure_handler to know which key to project).
    out_dim
        Output dimension of the LoRA B matrix (matches Qwen-35B
        hidden size at runtime — synthesised here).
    rank
        LoRA rank (matches r=16 in production; smaller in tests).

    Returns
    -------
    dict
        Keys ::
            beta_records   : list[{"input": list[float]}]  (replay)
            shrink_factor  : float (downscale)
            prior_deltas   : list[ndarray] (restructure / OPLoRA priors)
            deltas         : list[ndarray] (recombine / TIES-Merge inputs)
            adapter_keys   : tuple[str, ...]
            <each adapter_key> : ndarray of shape (out_dim, rank)
    """
    rng = np.random.default_rng(seed)
    beta_records = [
        {"input": rng.standard_normal(out_dim).astype(np.float32).tolist()}
        for _ in range(4)
    ]
    prior_deltas = [
        rng.standard_normal((out_dim, rank)).astype(np.float32)
        for _ in range(2)
    ]
    deltas = [
        rng.standard_normal((out_dim, rank)).astype(np.float32)
        for _ in range(3)
    ]
    payload: dict[str, Any] = {
        "beta_records": beta_records,
        "shrink_factor": 0.99,
        "prior_deltas": prior_deltas,
        "deltas": deltas,
        "adapter_keys": adapter_keys,
    }
    for key in adapter_keys:
        payload[key] = rng.standard_normal(
            (out_dim, rank),
        ).astype(np.float32)
    return payload


@dataclass
class G6DreamRunner:
    """Profile-aware episode runner over a MicroKikiSubstrate.

    Attributes
    ----------
    substrate
        The MicroKikiSubstrate instance under test.
    profile_name
        One of {"baseline", "P_min", "P_equ", "P_max"}.
    out_dim
        Synthetic LoRA out_dim used for replay / restructure /
        recombine payloads. 8 in tests; 4096 (Qwen-35B hidden) in
        production.
    rank
        Synthetic LoRA rank. 4 in tests; 16 in production.
    episodes_run
        Bumped on every successful run_episode call (DR-0 wrapper-
        level).
    last_episode_id
        Set on every run_episode call to
        ``f"g6-{profile}-{subdomain}-seed{seed}"``.
    """

    substrate: MicroKikiSubstrate
    profile_name: str
    out_dim: int = 8
    rank: int = 4
    episodes_run: int = field(default=0, init=False)
    last_episode_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.profile_name not in PROFILE_OPS:
            raise ValueError(
                f"unknown profile {self.profile_name!r}; expected one "
                f"of {sorted(PROFILE_OPS)}"
            )

    def run_episode(self, *, seed: int, subdomain: str) -> dict[str, Any]:
        """Run one dream episode for the given (seed, subdomain).

        Returns the payload that was constructed (useful for tests +
        debugging). For the baseline arm returns an empty dict and
        does NOT touch the substrate.
        """
        ops = PROFILE_OPS[self.profile_name]
        if not ops:
            return {}

        adapter_key = "layer_0_lora_B"
        payload = build_episode_payload(
            seed=seed,
            adapter_keys=(adapter_key,),
            out_dim=self.out_dim,
            rank=self.rank,
        )
        episode_id = (
            f"g6-{self.profile_name}-{subdomain}-seed{seed}"
        )
        payload["episode_id"] = episode_id

        # 1. replay -> vector aggregate (stub; production: feeds replay
        #    LoRA SGD).
        if "replay" in ops:
            replay = self.substrate.replay_handler_factory()
            replay(list(payload["beta_records"]), 20)

        # 2. downscale -> multiplicative shrink (stub; production:
        #    shrinks LoRA B).
        if "downscale" in ops:
            downscale = self.substrate.downscale_handler_factory()
            arr = np.asarray(payload[adapter_key])
            downscale(arr, float(payload["shrink_factor"]))

        # 3. restructure -> OPLoRA project (writes
        #    substrate._restructure_state).
        if "restructure" in ops:
            restructure = self.substrate.restructure_handler_factory()
            adapter: dict[str, Any] = {
                adapter_key: payload[adapter_key],
                "prior_deltas": payload["prior_deltas"],
                "episode_id": episode_id,
            }
            restructure(adapter, "oplora", adapter_key)

        # 4. recombine -> TIES-Merge (writes
        #    substrate._recombine_state).
        if "recombine" in ops:
            recombine = self.substrate.recombine_handler_factory()
            recombine_payload: dict[str, Any] = {
                "deltas": payload["deltas"],
                "episode_id": episode_id,
            }
            recombine(recombine_payload, "ties")

        self.episodes_run += 1
        self.last_episode_id = episode_id
        return payload


__all__ = ["G6DreamRunner", "PROFILE_OPS", "build_episode_payload"]
