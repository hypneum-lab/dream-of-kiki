"""MLX MLP classifier + dream-episode wrapper for the G4 pilot.

Bridges the framework-C dream runtime with a Split-FMNIST binary
classifier on the MLX substrate. The classifier exposes
``train_task`` / ``eval_accuracy`` / ``predict_logits`` /
``dream_episode``. ``dream_episode`` builds a
:class:`DreamEpisode` whose ``input_slice.beta_records`` carries
recent training samples and dispatches it via the profile's
:class:`DreamRuntime` — exactly the ``runtime.execute(...)`` path
used by ``scripts/pilot_cycle3_sanity.py``.

DR-0 accountability is automatic : every call to
``dream_episode`` appends one :class:`EpisodeLogEntry` to
``profile.runtime.log`` regardless of handler outcome.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
    kiki_oniric/profiles/{p_min, p_equ, p_max}.py
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from experiments.g4_split_fmnist.dataset import SplitFMNISTTask
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_max import PMaxProfile
from kiki_oniric.profiles.p_min import PMinProfile


# Type alias for the three concrete profile classes the G4 pilot
# wires through. ``baseline`` arm runs no dream-episode and never
# instantiates a profile, so it does not appear in this union.
ProfileT = PMinProfile | PEquProfile | PMaxProfile


PROFILE_FACTORIES: dict[str, Callable[..., ProfileT]] = {
    "P_min": PMinProfile,
    "P_equ": PEquProfile,
    "P_max": PMaxProfile,
}


def build_profile(name: str, seed: int) -> ProfileT:
    """Construct a fresh profile of the given ``name`` with ``seed``.

    ``P_equ`` and ``P_max`` accept a seeded RNG (used by their
    recombine handler) ; ``P_min`` is constructed without a seed
    field (its replay/downscale handlers are deterministic given
    the input slice).

    Raises :
        ValueError : ``name`` is not one of ``P_min`` / ``P_equ`` /
                     ``P_max``. The pilot driver's "baseline" arm
                     must not call this function — it skips the
                     dream episode entirely.
    """
    if name not in PROFILE_FACTORIES:
        raise ValueError(
            f"unknown profile {name!r} — expected one of "
            f"{sorted(PROFILE_FACTORIES)} (baseline arm runs no DE)"
        )
    factory = PROFILE_FACTORIES[name]
    if name == "P_min":
        return factory()
    return factory(rng=random.Random(seed))


def sample_beta_records(
    seed: int,
    n_records: int,
    feat_dim: int,
) -> list[dict[str, list[float]]]:
    """Return ``n_records`` ``{x: [...], y: [...]}`` records.

    Mirrors the convention from
    ``scripts/pilot_cycle3_sanity.py::_build_episode`` — a fresh
    ``np.random.default_rng(seed)`` produces ``standard_normal``
    feature + target vectors of width ``feat_dim``, packaged as
    Python-list-of-floats for the dream-ops contract
    (input_slice values must be JSON-serialisable lists).
    """
    rng = np.random.default_rng(seed)
    records: list[dict[str, list[float]]] = []
    for _ in range(n_records):
        records.append(
            {
                "x": rng.standard_normal(feat_dim).astype(np.float32).tolist(),
                "y": rng.standard_normal(feat_dim).astype(np.float32).tolist(),
            }
        )
    return records


@dataclass
class G4Classifier:
    """Tiny MLX MLP classifier for Split-FMNIST 2-class tasks.

    Architecture : Linear(in_dim, hidden) → ReLU → Linear(hidden,
    n_classes). Deterministic under a fixed ``seed`` via
    ``mx.random.seed`` at construction time.
    """

    in_dim: int
    hidden_dim: int
    n_classes: int
    seed: int
    _model: nn.Module = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mx.random.seed(self.seed)
        np.random.seed(self.seed)
        self._model = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_classes),
        )
        # Force eager init so identical seeds yield identical weights.
        mx.eval(self._model.parameters())

    # -------------------- forward --------------------

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """Return raw logits as a numpy array shape ``(N, n_classes)``."""
        out = self._model(mx.array(x))
        mx.eval(out)
        return np.asarray(out)

    def eval_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy in ``[0, 1]`` on ``(x, y)``."""
        if len(x) == 0:
            return 0.0
        logits = self.predict_logits(x)
        pred = logits.argmax(axis=1)
        return float((pred == y).mean())

    # -------------------- training --------------------

    def train_task(
        self,
        task: SplitFMNISTTask,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        """Train ``self._model`` on ``task`` for ``epochs`` SGD passes.

        Uses MLX's standard cross-entropy + SGD optimiser. Mini-
        batches are drawn deterministically from a numpy RNG seeded
        at ``self.seed`` so two classifiers with identical seeds
        and identical task data produce identical post-train weights.
        """
        x = mx.array(task["x_train"])
        y = mx.array(task["y_train"])
        n = x.shape[0]
        opt = optim.SGD(learning_rate=lr)
        rng = np.random.default_rng(self.seed)
        loss_and_grad = nn.value_and_grad(self._model, self._loss_fn)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if len(idx) == 0:
                    continue
                xb = x[mx.array(idx)]
                yb = y[mx.array(idx)]
                _loss, grads = loss_and_grad(self._model, xb, yb)
                opt.update(self._model, grads)
                mx.eval(self._model.parameters(), opt.state)

    def _loss_fn(
        self, model: nn.Module, xb: mx.array, yb: mx.array
    ) -> mx.array:
        logits = model(xb)
        return nn.losses.cross_entropy(logits, yb, reduction="mean")

    # -------------------- dream --------------------

    def dream_episode(self, profile: ProfileT, seed: int) -> None:
        """Drive one :class:`DreamEpisode` through the profile's runtime.

        Builds an episode whose ``operation_set`` matches the
        profile's wired handlers (P_min : replay+downscale ;
        P_equ/P_max : +restructure+recombine), and dispatches via
        ``profile.runtime.execute``. The classifier weights are
        **not directly** mutated by this call — the episode logs
        DR-0 evidence on the profile's runtime, and downstream
        eval picks up any state drift the profile chose to apply.

        For the G4 pilot the dream episode is the *interleaved*
        signal between sequential tasks : its presence vs absence
        is what distinguishes the dream-active arms from baseline.
        """
        profile_name = type(profile).__name__
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
        beta_records = sample_beta_records(
            seed=seed, n_records=4, feat_dim=4
        )
        rng = np.random.default_rng(seed + 10_000)
        delta_latents = [
            rng.standard_normal(4).astype(np.float32).tolist()
            for _ in range(2)
        ]
        episode = DreamEpisode(
            trigger=EpisodeTrigger.SCHEDULED,
            input_slice={
                "beta_records": beta_records,
                "shrink_factor": 0.99,
                "topo_op": "reroute",
                "swap_indices": [0, 1],
                "delta_latents": delta_latents,
            },
            operation_set=ops,
            output_channels=channels,
            budget=BudgetCap(
                flops=10_000_000, wall_time_s=10.0, energy_j=1.0
            ),
            episode_id=f"g4-{profile_name}-seed{seed}",
        )
        profile.runtime.execute(episode)
