"""DR-3 Conformance — G4-quater 5-layer deeper hierarchical substrate.

Verifies the deeper substrate (G4HierarchicalDeeperClassifier)
satisfies DR-3 conditions (1) typed Protocols, (2) executable
primitives, (3) primitives chain without raising.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md sec 6.2
    docs/osf-prereg-g4-quater-pilot.md sec 5
"""
from __future__ import annotations

import numpy as np

from experiments.g4_quater_test.deeper_classifier import (
    G4HierarchicalDeeperClassifier,
)
from experiments.g4_quater_test.run_step1_deeper import (
    _BetaBufferDeeper,
    _dream_episode_deeper,
)
from experiments.g4_split_fmnist.dream_wrap import build_profile
from experiments.g4_ter_hp_sweep.hp_grid import representative_combo


def _fill_buffer(
    buf: _BetaBufferDeeper,
    clf: G4HierarchicalDeeperClassifier,
    n_per_class: int = 4,
) -> None:
    rng = np.random.default_rng(0)
    for cls in (0, 1):
        for _ in range(n_per_class):
            x = rng.standard_normal(10).astype(np.float32)
            latent = clf.latent(x[None, :])[0]
            buf.push(x=x, y=cls, latent=latent)


def test_dr3_g4_deeper_protocols_present() -> None:
    """DR-3 (1) — substrate exposes the public coupling methods."""
    clf = G4HierarchicalDeeperClassifier(
        in_dim=10, hidden=(8, 6, 4, 2), n_classes=2, seed=0
    )
    assert hasattr(clf, "predict_logits")
    assert hasattr(clf, "latent")
    assert hasattr(clf, "train_task")
    assert hasattr(clf, "eval_accuracy")
    assert hasattr(clf, "replay_optimizer_step")
    assert hasattr(clf, "downscale_step")
    assert hasattr(clf, "restructure_step")
    assert hasattr(clf, "recombine_step")


def test_dr3_g4_deeper_primitives_executable() -> None:
    """DR-3 (2) — REPLAY/DOWNSCALE/RESTRUCTURE/RECOMBINE dispatchable."""
    clf = G4HierarchicalDeeperClassifier(
        in_dim=10, hidden=(8, 6, 4, 2), n_classes=2, seed=0
    )
    buf = _BetaBufferDeeper(capacity=16)
    _fill_buffer(buf, clf)
    profile = build_profile("P_max", seed=0)
    combo = representative_combo()
    _dream_episode_deeper(
        clf,
        profile,
        seed=0,
        beta_buffer=buf,
        combo=combo,
        restructure_factor=0.05,
        recombine_n_synthetic=4,
        recombine_lr=0.01,
    )
    assert len(profile.runtime.log) == 1


def test_dr3_g4_deeper_primitives_chain() -> None:
    """DR-3 (3) — three consecutive episodes do not raise."""
    clf = G4HierarchicalDeeperClassifier(
        in_dim=10, hidden=(8, 6, 4, 2), n_classes=2, seed=0
    )
    buf = _BetaBufferDeeper(capacity=16)
    _fill_buffer(buf, clf)
    profile = build_profile("P_max", seed=0)
    combo = representative_combo()
    for k in range(3):
        _dream_episode_deeper(
            clf,
            profile,
            seed=k,
            beta_buffer=buf,
            combo=combo,
            restructure_factor=0.05,
            recombine_n_synthetic=4,
            recombine_lr=0.01,
        )
    assert len(profile.runtime.log) == 3
