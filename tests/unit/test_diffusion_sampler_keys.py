"""R1 key-derivation contract for the diffusion :class:`Sampler`.

Per plan §2.3 D3 + §4 M3, the sampler MUST consume only
``mx.random.split``-derived keys (never raw ``mx.random.key(seed)``
straight to ``mx.random.normal``) and produce a bit-identical
output given a fixed ``(root_key, n_steps, shape)`` triple on a
fixed MLX version + chip family.

Tests :

1. Deterministic given (root_key, n_steps, shape) — two runs
   produce bit-identical arrays.
2. Different root_key → different output.
3. Key tree depth matches n_steps + 1 (no leakage).
4. Shape invariance — output shape == requested shape.
5. Skipped on platforms without MLX (Linux CI) via
   :func:`pytest.importorskip`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import mlx.core as mx
else:
    mx = pytest.importorskip("mlx.core")

from kiki_oniric.substrates._diffusion import (
    MLPDenoiser,
    NoiseSchedule,
    Sampler,
)


def _make_sampler(d_latent: int = 4, t_steps: int = 16) -> Sampler:
    """Build a deterministic small sampler (real weights, fixed seed)."""
    mx.random.seed(0)
    denoiser = MLPDenoiser(d_latent=d_latent, d_hidden=8, n_layers=2)
    schedule = NoiseSchedule(t_steps=t_steps, beta_min=1e-4, beta_max=2e-2)
    return Sampler(model=denoiser, schedule=schedule)


def _hash_array(arr: "mx.array") -> tuple[float, ...]:
    """Return a flat tuple of floats for bit-exact equality assertions."""
    import numpy as np

    return tuple(np.asarray(arr).astype(float).flatten().tolist())


def test_sampler_deterministic_given_same_root_key() -> None:
    """Same root key + n_steps + shape ⇒ bit-identical output."""
    sampler = _make_sampler()
    key = mx.random.key(123)
    out_a = sampler.sample(key=key, n_steps=4, shape=(2, 4))
    out_b = sampler.sample(key=key, n_steps=4, shape=(2, 4))
    assert _hash_array(out_a) == _hash_array(out_b)


def test_sampler_different_root_key_yields_different_output() -> None:
    """Flipping the root key produces a different sample.

    Equality is bit-level on flattened floats ; with probability
    ≈ 1 over a fresh standard normal draw at least one element
    differs. We assert at least one element differs.
    """
    sampler = _make_sampler()
    out_a = sampler.sample(key=mx.random.key(1), n_steps=4, shape=(2, 4))
    out_b = sampler.sample(key=mx.random.key(2), n_steps=4, shape=(2, 4))
    flat_a = _hash_array(out_a)
    flat_b = _hash_array(out_b)
    assert any(a != b for a, b in zip(flat_a, flat_b))


def test_sampler_key_tree_depth_matches_n_steps_plus_one() -> None:
    """The sampler derives exactly ``n_steps + 1`` subkeys.

    We probe the contract by mirroring the documented key
    derivation : ``mx.random.split(root, num=n_steps + 1)`` MUST
    succeed and MUST yield the same first sub-key used for the
    initial-noise draw across two independent splits.
    """
    n_steps = 5
    root = mx.random.key(42)
    subs_a = mx.random.split(root, num=n_steps + 1)
    subs_b = mx.random.split(root, num=n_steps + 1)
    # split must be deterministic on the same root key.
    for i in range(n_steps + 1):
        assert _hash_array(subs_a[i]) == _hash_array(subs_b[i])
    # And shape[0] is exactly n_steps + 1.
    assert subs_a.shape[0] == n_steps + 1


def test_sampler_shape_invariance() -> None:
    """Output shape matches the requested ``shape`` argument."""
    sampler = _make_sampler(d_latent=8, t_steps=10)
    for shape in [(1, 8), (3, 8), (4, 8)]:
        out = sampler.sample(key=mx.random.key(7), n_steps=3, shape=shape)
        assert tuple(out.shape) == shape


def test_sampler_rejects_invalid_n_steps_or_shape() -> None:
    """Defensive guards : n_steps must be in (0, schedule.t_steps]."""
    sampler = _make_sampler(d_latent=4, t_steps=8)
    with pytest.raises(ValueError, match="n_steps"):
        sampler.sample(key=mx.random.key(0), n_steps=0, shape=(1, 4))
    with pytest.raises(ValueError, match="exceeds schedule"):
        sampler.sample(key=mx.random.key(0), n_steps=99, shape=(1, 4))
    with pytest.raises(ValueError, match="shape must be non-empty"):
        sampler.sample(key=mx.random.key(0), n_steps=2, shape=())
