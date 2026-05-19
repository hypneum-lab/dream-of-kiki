"""Unit tests for `experiments.g5_cross_substrate.esnn_dream_wrap`."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g5_cross_substrate.esnn_classifier import EsnnG5Classifier
from experiments.g5_cross_substrate.esnn_dream_wrap import (
    build_esnn_profile,
    dream_episode,
)
from kiki_oniric.dream.episode import Operation
from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_max import PMaxProfile
from kiki_oniric.profiles.p_min import PMinProfile


def test_build_esnn_profile_returns_known_profile_types() -> None:
    """`build_esnn_profile` returns the canonical profile classes."""
    p_min = build_esnn_profile("P_min", seed=0)
    p_equ = build_esnn_profile("P_equ", seed=0)
    p_max = build_esnn_profile("P_max", seed=0)
    assert isinstance(p_min, PMinProfile)
    assert isinstance(p_equ, PEquProfile)
    assert isinstance(p_max, PMaxProfile)


def test_build_esnn_profile_rejects_unknown_name() -> None:
    """Unknown profile name -> ValueError (mirrors G4-bis)."""
    with pytest.raises(ValueError, match="unknown profile"):
        build_esnn_profile("P_unknown", seed=0)


def test_build_esnn_profile_rebinds_all_required_ops() -> None:
    """After rebind, every op in the profile is bound to an E-SNN
    adapter (introspect the runtime's private handler dict)."""
    profile = build_esnn_profile("P_max", seed=0)
    handlers = profile.runtime._handlers
    expected = {
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    }
    assert expected.issubset(set(handlers.keys()))
    # Every rebound handler must be the E-SNN adapter — not the
    # default MLX/numpy stubs. Adapters are tagged with __esnn__.
    for op in expected:
        assert getattr(handlers[op], "__esnn__", False), (
            f"op {op} not rebound to E-SNN adapter"
        )


def test_dream_episode_appends_log_entry() -> None:
    """One `dream_episode(...)` call appends one runtime log entry."""
    clf = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    profile = build_esnn_profile("P_equ", seed=0)
    n_before = len(profile.runtime.log)
    dream_episode(clf, profile, seed=42)
    n_after = len(profile.runtime.log)
    assert n_after == n_before + 1


def test_dream_episode_does_not_mutate_classifier_weights() -> None:
    """DR-0-only wrapper : weights stay bit-exact across `dream_episode`.

    This is the documented design choice — coupling lands in a
    follow-up plan ; G5 measures the same logging-only baseline as
    G4-bis 2026-05-03 to keep the comparison apples-to-apples for
    the dream-episode dispatch path. Note that G4-bis also does
    additional weight-mutating coupling outside the runtime ; G5
    leaves classifier weights spectator-only at the dream-episode
    boundary so the cross-substrate comparison isolates the
    E-SNN substrate dispatch path.
    """
    clf = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    w_in_before = clf.W_in.copy()
    w_out_before = clf.W_out.copy()
    profile = build_esnn_profile("P_min", seed=0)
    dream_episode(clf, profile, seed=42)

    np.testing.assert_array_equal(clf.W_in, w_in_before)
    np.testing.assert_array_equal(clf.W_out, w_out_before)


def test_dream_episode_deterministic_under_same_seed() -> None:
    """Same `seed` -> same episode_id appended to the log."""
    clf_a = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    clf_b = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    p_a = build_esnn_profile("P_max", seed=0)
    p_b = build_esnn_profile("P_max", seed=0)
    dream_episode(clf_a, p_a, seed=7)
    dream_episode(clf_b, p_b, seed=7)
    assert (
        p_a.runtime.log[-1].episode_id == p_b.runtime.log[-1].episode_id
    )
