"""Cross-domain RECOMBINE strategies for G4-nono (spec section 5).

Extension of ``experiments/g4_quater_test/recombine_strategies.py``;
the ladder ``mog`` / ``ae`` code paths are untouched. The return
value reuses the ``SyntheticLatentBatch`` interface (``x`` latents,
``y`` targets) but ``y`` is a soft 20-d target and two extra keys
(``lam``, ``pairs``) expose the draw for auditing.

RNG stream (one ``default_rng(seed)``, in this order):

1. ``lam`` ~ Uniform(0.2, 0.8), 16 values;
2. the 16 class pairs: ``cross`` draws the 16 domain-A indices then
   the 16 domain-B indices (``integers``); ``within`` draws pair by
   pair (8 domain-A then 8 domain-B) a 2-subset without replacement;
3. one buffer record per class, pair by pair, first class then
   second class, uniform among that class's records.

4. ``variant="full"`` only (``P_max``'s recombine_full, framework C
   section 3.1; PI decision 2026-10-24): ``eps`` ~ N(0, I) of shape
   ``(n_synthetic, d)``, drawn after step 3 so ``lam``, the pairs, the
   records and the soft targets are identical to ``light``. The latent
   becomes ``max(0, z + sigma * eps)``: a reparameterised Gaussian
   sample around the interpolant ``z`` with ``sigma`` the per-dimension
   SD (ddof 0) of all buffer latents (moment-matched encoder), decoded
   by the ``fc1`` ReLU so it stays in the post-ReLU latent space.
   ``light`` (``P_equ``'s recombine_light) is the plain interpolant.

Both ``cross`` and ``within`` require at least one buffer record in
each domain (so they are no-ops at the DEs before tasks 2-6) and
``within`` additionally needs two classes present in each domain.
"""
from __future__ import annotations

from typing import Literal

import mlx.core as mx
import numpy as np

from experiments.g4_nono_test.stream import DOMAIN_A, DOMAIN_B
from experiments.g4_quater_test.recombine_strategies import (
    SyntheticLatentBatch,
)

CrossStrategy = Literal["cross", "within", "none"]
RecombineVariant = Literal["light", "full"]
N_CLASSES = 20
LAMBDA_LO = 0.2
LAMBDA_HI = 0.8
N_WITHIN_A = 8  # within: pairs 1-8 domain A, 9-16 domain B


class CrossBatch(SyntheticLatentBatch):
    """``SyntheticLatentBatch`` with soft targets + audit fields."""

    lam: np.ndarray
    pairs: np.ndarray


def _present(labels: np.ndarray, domain: tuple[int, ...]) -> np.ndarray:
    return np.asarray(
        sorted(set(labels.tolist()) & set(domain)), dtype=np.int64
    )


def sample_cross_domain_latents(
    *,
    strategy: CrossStrategy,
    latents: np.ndarray,
    labels: np.ndarray,
    n_synthetic: int,
    seed: int,
    variant: RecombineVariant = "light",
) -> CrossBatch | None:
    """Return ``n_synthetic`` mixed latents with soft targets.

    ``latents`` ``(N, d)`` and ``labels`` ``(N,)`` are the beta
    buffer contents (latents frozen at push time). Returns ``None``
    for ``none`` or when the strategy is inactive (see module doc).
    """
    if variant not in ("light", "full"):
        raise ValueError(f"unknown recombine variant: {variant!r}")
    if strategy == "none":
        return None
    if strategy not in ("cross", "within"):
        raise ValueError(f"unknown cross-domain strategy: {strategy!r}")
    if latents.shape[0] == 0:
        return None
    pres_a = _present(labels, DOMAIN_A)
    pres_b = _present(labels, DOMAIN_B)
    if len(pres_a) == 0 or len(pres_b) == 0:
        return None
    if strategy == "within" and (len(pres_a) < 2 or len(pres_b) < 2):
        return None

    rng = np.random.default_rng(seed)
    lam = rng.uniform(LAMBDA_LO, LAMBDA_HI, size=n_synthetic)

    pairs = np.empty((n_synthetic, 2), dtype=np.int64)
    if strategy == "cross":
        ia = rng.integers(0, len(pres_a), size=n_synthetic)
        ib = rng.integers(0, len(pres_b), size=n_synthetic)
        pairs[:, 0] = pres_a[ia]
        pairs[:, 1] = pres_b[ib]
    else:
        for j in range(n_synthetic):
            pool = pres_a if j < N_WITHIN_A else pres_b
            pick = rng.choice(len(pool), size=2, replace=False)
            pairs[j] = pool[pick]

    by_class = {
        int(c): np.flatnonzero(labels == c)
        for c in set(pairs.ravel().tolist())
    }
    x = np.empty((n_synthetic, latents.shape[1]), dtype=np.float32)
    y = np.zeros((n_synthetic, N_CLASSES), dtype=np.float32)
    for j in range(n_synthetic):
        a, b = int(pairs[j, 0]), int(pairs[j, 1])
        ra = by_class[a][int(rng.integers(0, len(by_class[a])))]
        rb = by_class[b][int(rng.integers(0, len(by_class[b])))]
        lj = float(lam[j])
        x[j] = lj * latents[ra] + (1.0 - lj) * latents[rb]
        y[j, a] = lj
        y[j, b] = 1.0 - lj
    if variant == "full":
        sigma = latents.astype(np.float64).std(axis=0)
        eps = rng.standard_normal(x.shape)
        x = np.maximum(x + sigma * eps, 0.0).astype(np.float32)
    return {"x": x, "y": y, "lam": lam, "pairs": pairs}


def soft_target_fc2_step(
    cnn: object, x: np.ndarray, targets: np.ndarray, *, lr: float
) -> None:
    """One SGD step on ``cnn._fc2`` only: mean soft-target CE
    ``-sum_c t_c log softmax(fc2(z))_c`` over the batch."""
    import mlx.nn as nn
    import mlx.optimizers as optim

    fc2 = cnn._fc2  # type: ignore[attr-defined]
    xb = mx.array(x.astype(np.float32))
    tb = mx.array(targets.astype(np.float32))
    opt = optim.SGD(learning_rate=lr)

    def loss_fn(layer: nn.Linear, xv: mx.array, tv: mx.array) -> mx.array:
        logp = layer(xv) - mx.logsumexp(layer(xv), axis=1, keepdims=True)
        return -(tv * logp).sum(axis=1).mean()

    _loss, grads = nn.value_and_grad(fc2, loss_fn)(fc2, xb, tb)
    opt.update(fc2, grads)
    mx.eval(fc2.parameters(), opt.state)
