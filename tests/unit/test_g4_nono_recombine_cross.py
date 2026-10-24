"""Cross-domain RECOMBINE sampler (spec section 5 checklist 1-3)."""
from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

from experiments.g4_nono_test.recombine_cross import (
    LAMBDA_HI,
    LAMBDA_LO,
    N_CLASSES,
    CrossBatch,
    CrossStrategy,
    sample_cross_domain_latents,
)
from experiments.g4_nono_test.stream import DOMAIN_A, DOMAIN_B

D = 8


def _buffer(
    classes: list[int], per_class: int = 16, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.asarray(classes, np.int64), per_class)
    # latent of every record encodes (class, index) so provenance is
    # recoverable: column 0 = class id, column 1 = within-class index.
    lat = rng.standard_normal((len(labels), D)).astype(np.float32)
    lat[:, 0] = labels
    lat[:, 1] = np.tile(np.arange(per_class), len(classes))
    return lat, labels


FULL = list(DOMAIN_A[:3]) + list(DOMAIN_B[:3])


def _draw(
    strategy: CrossStrategy,
    seed: int = 7,
    classes: list[int] | None = None,
    n: int = 16,
) -> CrossBatch | None:
    lat, lab = _buffer(classes or FULL)
    return sample_cross_domain_latents(
        strategy=strategy, latents=lat, labels=lab, n_synthetic=n, seed=seed
    )


def _ok(b: CrossBatch | None) -> CrossBatch:
    """Narrow a draw that the test expects to be active."""
    assert b is not None
    return b


def test_none_returns_none() -> None:
    assert _draw("none") is None


def test_unknown_strategy_raises() -> None:
    lat, lab = _buffer(FULL)
    with pytest.raises(ValueError):
        sample_cross_domain_latents(
            strategy="mog",  # type: ignore[arg-type]
            latents=lat, labels=lab, n_synthetic=16, seed=1,
        )


@pytest.mark.parametrize("strategy", ["cross", "within"])
def test_shapes_and_soft_targets(strategy: CrossStrategy) -> None:
    b = _draw(strategy)
    assert b is not None
    assert b["x"].shape == (16, D) and b["x"].dtype == np.float32
    assert b["y"].shape == (16, N_CLASSES) and b["y"].dtype == np.float32
    assert np.allclose(b["y"].sum(axis=1), 1.0, atol=1e-6)
    lam = b["lam"]
    assert lam.shape == (16,)
    assert (lam >= LAMBDA_LO).all() and (lam <= LAMBDA_HI).all()
    for j in range(16):
        a, c = b["pairs"][j]
        assert b["y"][j, a] == pytest.approx(lam[j], abs=1e-6)
        assert b["y"][j, c] == pytest.approx(1 - lam[j], abs=1e-6)
        assert np.count_nonzero(b["y"][j]) == 2


@pytest.mark.parametrize("strategy", ["cross", "within"])
def test_latent_is_convex_mix_of_buffer_records(
    strategy: CrossStrategy,
) -> None:
    b = _ok(_draw(strategy))
    for j in range(16):
        a, c = b["pairs"][j]
        # recover the two source records: mix is lam*z_a + (1-lam)*z_c;
        # column 0 therefore equals lam*a + (1-lam)*c exactly.
        assert b["x"][j, 0] == pytest.approx(
            b["lam"][j] * a + (1 - b["lam"][j]) * c, abs=1e-5
        )


def test_same_seed_same_lambda_across_strategies() -> None:
    """Checklist 2: lambda drawn first and shared by cross/within."""
    c = _ok(_draw("cross", seed=11))
    w = _ok(_draw("within", seed=11))
    assert np.array_equal(c["lam"], w["lam"])
    assert c["lam"].tolist() == np.random.default_rng(11).uniform(
        0.2, 0.8, size=16
    ).tolist()


def test_deterministic_for_seed_and_differs_across_seeds() -> None:
    a = _ok(_draw("cross", seed=3))
    b = _ok(_draw("cross", seed=3))
    c = _ok(_draw("cross", seed=4))
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["pairs"], b["pairs"])
    assert not np.array_equal(a["x"], c["x"])


def test_distinct_class_and_domain_constraints_1e4_draws() -> None:
    """Checklist 3, over 10^4 draws."""
    lat, lab = _buffer(list(range(20)), per_class=2)
    setA, setB = set(DOMAIN_A), set(DOMAIN_B)
    for s in range(10_000 // 16 + 1):  # 16 pairs per draw -> >= 10^4 pairs
        c = _ok(sample_cross_domain_latents(
            strategy="cross", latents=lat, labels=lab, n_synthetic=16, seed=s
        ))
        w = _ok(sample_cross_domain_latents(
            strategy="within", latents=lat, labels=lab, n_synthetic=16, seed=s
        ))
        for a, b in c["pairs"].tolist():
            assert a in setA and b in setB
        for j, (a, b) in enumerate(w["pairs"].tolist()):
            assert a != b
            dom = setA if j < 8 else setB
            assert a in dom and b in dom


def test_within_split_is_8_domain_a_then_8_domain_b() -> None:
    w = _ok(_draw("within"))
    first = set(w["pairs"][:8].ravel().tolist())
    second = set(w["pairs"][8:].ravel().tolist())
    assert first <= set(DOMAIN_A) and second <= set(DOMAIN_B)


def test_exact_rng_stream_order() -> None:
    """lambda first, then cross pair classes (a's then b's), then
    one record per class (pair order, first class then second)."""
    lat, lab = _buffer(FULL, per_class=16)
    seed = 21
    b = _ok(sample_cross_domain_latents(
        strategy="cross", latents=lat, labels=lab, n_synthetic=16, seed=seed
    ))
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0.2, 0.8, size=16)
    pa = sorted(set(FULL) & set(DOMAIN_A))
    pb = sorted(set(FULL) & set(DOMAIN_B))
    ia = rng.integers(0, len(pa), size=16)
    ib = rng.integers(0, len(pb), size=16)
    pairs = np.stack([np.asarray(pa)[ia], np.asarray(pb)[ib]], axis=1)
    assert np.array_equal(b["pairs"], pairs)
    x = np.empty((16, D), np.float32)
    for j in range(16):
        ra = int(rng.integers(0, 16))
        rb = int(rng.integers(0, 16))
        za = lat[lab == pairs[j, 0]][ra]
        zb = lat[lab == pairs[j, 1]][rb]
        x[j] = lam[j] * za + (1 - lam[j]) * zb
    assert np.allclose(b["x"], x, atol=1e-6)


@pytest.mark.parametrize("strategy", ["cross", "within"])
def test_noop_when_only_one_domain_in_buffer(
    strategy: CrossStrategy,
) -> None:
    """Spec: active only once both domains are in the buffer."""
    assert _draw(strategy, classes=list(DOMAIN_A[:4])) is None
    assert _draw(strategy, classes=list(DOMAIN_B[:4])) is None


def test_within_needs_two_classes_per_domain() -> None:
    assert _draw("within", classes=[0, 2]) is None
    assert _draw("cross", classes=[0, 2]) is not None


def test_empty_buffer() -> None:
    lat = np.zeros((0, D), np.float32)
    lab = np.zeros((0,), np.int64)
    assert sample_cross_domain_latents(
        strategy="cross", latents=lat, labels=lab, n_synthetic=16, seed=0
    ) is None


def test_ladder_module_untouched() -> None:
    from experiments.g4_quater_test import recombine_strategies as r

    assert get_args(r.RecombineStrategy) == ("mog", "ae", "none")
