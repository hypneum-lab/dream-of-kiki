"""Unit tests for the G4HierarchicalClassifier (Plan G4-ter Task 2)."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g4_ter_hp_sweep.dream_wrap_hier import (
    BetaBufferHierFIFO,
    G4HierarchicalClassifier,
)


def test_classifier_has_three_linear_layers() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=784, hidden_1=32, hidden_2=16, n_classes=2, seed=7
    )
    # Public attributes for RESTRUCTURE site identification.
    assert clf.hidden_1 == 32
    assert clf.hidden_2 == 16
    assert clf.n_classes == 2


def test_predict_logits_shape() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=784, hidden_1=32, hidden_2=16, n_classes=2, seed=7
    )
    x = np.zeros((4, 784), dtype=np.float32)
    logits = clf.predict_logits(x)
    assert logits.shape == (4, 2)


def test_seed_determinism() -> None:
    a = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=42
    )
    b = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=42
    )
    x = np.ones((1, 10), dtype=np.float32)
    np.testing.assert_array_equal(a.predict_logits(x), b.predict_logits(x))


def _toy_task(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 64
    x = rng.standard_normal((n, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int32)
    return {
        "x_train": x[:48],
        "y_train": y[:48],
        "x_test": x[48:],
        "y_test": y[48:],
    }


def test_train_task_then_eval_accuracy() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=32, hidden_2=16, n_classes=2, seed=11
    )
    task = _toy_task(seed=11)
    clf.train_task(task, epochs=2, batch_size=16, lr=0.05)
    acc = clf.eval_accuracy(task["x_test"], task["y_test"])
    assert 0.0 <= acc <= 1.0


def test_buffer_push_pop_with_latent() -> None:
    buf = BetaBufferHierFIFO(capacity=4)
    x = np.zeros(10, dtype=np.float32)
    latent = np.ones(16, dtype=np.float32)
    buf.push(x=x, y=1, latent=latent)
    snap = buf.snapshot()
    assert len(snap) == 1
    assert snap[0]["y"] == 1
    assert snap[0]["latent"] == [1.0] * 16


def test_buffer_sample_deterministic() -> None:
    buf = BetaBufferHierFIFO(capacity=8)
    for i in range(8):
        buf.push(
            x=np.full(4, float(i), dtype=np.float32),
            y=i % 2,
            latent=np.full(2, float(i), dtype=np.float32),
        )
    a = buf.sample(n=3, seed=42)
    b = buf.sample(n=3, seed=42)
    assert [r["x"] for r in a] == [r["x"] for r in b]


def test_buffer_sample_no_latent() -> None:
    """latent=None records are still sampleable (legacy compat)."""
    buf = BetaBufferHierFIFO(capacity=2)
    buf.push(x=np.zeros(2, dtype=np.float32), y=0, latent=None)
    snap = buf.snapshot()
    assert snap[0]["latent"] is None


def test_restructure_step_modifies_only_hidden_2() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    w1_before = np.asarray(clf._l1.weight)
    w2_before = np.asarray(clf._l2.weight)
    w3_before = np.asarray(clf._l3.weight)
    clf._restructure_step(factor=0.05, seed=99)
    w1_after = np.asarray(clf._l1.weight)
    w2_after = np.asarray(clf._l2.weight)
    w3_after = np.asarray(clf._l3.weight)
    # Input + output untouched.
    np.testing.assert_array_equal(w1_before, w1_after)
    np.testing.assert_array_equal(w3_before, w3_after)
    # Hidden_2 perturbed (probabilistically: any element changed).
    assert not np.array_equal(w2_before, w2_after)


def test_restructure_step_factor_zero_is_noop() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    w2_before = np.asarray(clf._l2.weight)
    clf._restructure_step(factor=0.0, seed=99)
    w2_after = np.asarray(clf._l2.weight)
    np.testing.assert_array_equal(w2_before, w2_after)


def test_restructure_step_factor_negative_raises() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    with pytest.raises(ValueError, match="factor must be"):
        clf._restructure_step(factor=-0.01, seed=99)


def test_restructure_step_seed_determinism() -> None:
    a = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    b = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    a._restructure_step(factor=0.05, seed=99)
    b._restructure_step(factor=0.05, seed=99)
    np.testing.assert_array_equal(
        np.asarray(a._l2.weight), np.asarray(b._l2.weight)
    )


def test_recombine_step_with_empty_buffer_is_noop() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    w3_before = np.asarray(clf._l3.weight)
    clf._recombine_step(
        latents=[], n_synthetic=5, lr=0.01, seed=42
    )
    w3_after = np.asarray(clf._l3.weight)
    np.testing.assert_array_equal(w3_before, w3_after)


def test_recombine_step_modifies_l3_only() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    # Synthetic latents indexed by class: list of (latent, label).
    latents = [
        ([0.5, 0.5, 0.5], 0),
        ([0.6, 0.4, 0.5], 0),
        ([-0.3, -0.4, -0.5], 1),
        ([-0.4, -0.5, -0.6], 1),
    ]
    w1_before = np.asarray(clf._l1.weight)
    w2_before = np.asarray(clf._l2.weight)
    w3_before = np.asarray(clf._l3.weight)
    clf._recombine_step(
        latents=latents, n_synthetic=4, lr=0.1, seed=7
    )
    np.testing.assert_array_equal(w1_before, np.asarray(clf._l1.weight))
    np.testing.assert_array_equal(w2_before, np.asarray(clf._l2.weight))
    assert not np.array_equal(w3_before, np.asarray(clf._l3.weight))


def test_recombine_step_seed_determinism() -> None:
    latents = [
        ([0.5, 0.5, 0.5], 0),
        ([-0.5, -0.5, -0.5], 1),
    ] * 4
    a = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    b = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    a._recombine_step(latents=latents, n_synthetic=8, lr=0.1, seed=7)
    b._recombine_step(latents=latents, n_synthetic=8, lr=0.1, seed=7)
    np.testing.assert_array_equal(
        np.asarray(a._l3.weight), np.asarray(b._l3.weight)
    )


from experiments.g4_split_fmnist.dream_wrap import build_profile  # noqa: E402


def _fill_buffer(buf: BetaBufferHierFIFO, clf: G4HierarchicalClassifier,
                 n_per_class: int = 6) -> None:
    rng = np.random.default_rng(0)
    for cls in (0, 1):
        for _ in range(n_per_class):
            x = rng.standard_normal(10).astype(np.float32)
            latent = clf.latent(x[None, :])[0]
            buf.push(x=x, y=cls, latent=latent)


def test_dream_episode_hier_p_min_runs() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    buf = BetaBufferHierFIFO(capacity=32)
    _fill_buffer(buf, clf)
    profile = build_profile("P_min", seed=13)
    clf.dream_episode_hier(
        profile,
        seed=13,
        beta_buffer=buf,
        replay_n_records=8,
        replay_n_steps=1,
        replay_lr=0.01,
        downscale_factor=0.95,
        restructure_factor=0.05,
        recombine_n_synthetic=4,
        recombine_lr=0.01,
    )
    assert len(profile.runtime.log) == 1


def test_dream_episode_hier_p_equ_mutates_l2() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    buf = BetaBufferHierFIFO(capacity=32)
    _fill_buffer(buf, clf)
    w2_before = np.asarray(clf._l2.weight)
    profile = build_profile("P_equ", seed=13)
    clf.dream_episode_hier(
        profile,
        seed=13,
        beta_buffer=buf,
        replay_n_records=8,
        replay_n_steps=1,
        replay_lr=0.01,
        downscale_factor=0.95,
        restructure_factor=0.05,
        recombine_n_synthetic=4,
        recombine_lr=0.01,
    )
    w2_after = np.asarray(clf._l2.weight)
    # P_equ runs RESTRUCTURE -> _l2.weight changes (perturb + downscale).
    assert not np.array_equal(w2_before, w2_after)


def test_dream_episode_hier_p_min_does_not_perturb_l2_random() -> None:
    """P_min runs DOWNSCALE only on _l2 (deterministic factor *), no random
    perturbation. Two runs with same seed must be bit-identical."""
    a = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    b = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=13
    )
    buf_a = BetaBufferHierFIFO(capacity=32)
    buf_b = BetaBufferHierFIFO(capacity=32)
    _fill_buffer(buf_a, a)
    _fill_buffer(buf_b, b)
    pa = build_profile("P_min", seed=13)
    pb = build_profile("P_min", seed=13)
    a.dream_episode_hier(
        pa, seed=13, beta_buffer=buf_a,
        replay_n_records=8, replay_n_steps=1, replay_lr=0.01,
        downscale_factor=0.95, restructure_factor=0.05,
        recombine_n_synthetic=4, recombine_lr=0.01,
    )
    b.dream_episode_hier(
        pb, seed=13, beta_buffer=buf_b,
        replay_n_records=8, replay_n_steps=1, replay_lr=0.01,
        downscale_factor=0.95, restructure_factor=0.05,
        recombine_n_synthetic=4, recombine_lr=0.01,
    )
    np.testing.assert_array_equal(
        np.asarray(a._l2.weight), np.asarray(b._l2.weight)
    )
