"""CDPA / WDPA / retention / 2-way metrics on synthetic predictions."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g4_nono_test.metrics import (
    compute_cdpa,
    compute_final_accuracy,
    compute_retention_domain_a,
    compute_two_way_accuracy,
    compute_wdpa,
)
from experiments.g4_nono_test.stream import DOMAIN_A, DOMAIN_B


def _labels(n_per: int = 4) -> np.ndarray:
    return np.repeat(np.arange(20), n_per)


def _onehot_logits(y: np.ndarray, scale: float = 5.0) -> np.ndarray:
    z = np.zeros((len(y), 20), np.float32)
    z[np.arange(len(y)), y] = scale
    return z


def test_perfect_logits_give_one() -> None:
    y = _labels()
    z = _onehot_logits(y)
    assert compute_cdpa(z, y) == 1.0
    assert compute_wdpa(z, y) == 1.0
    assert compute_final_accuracy(z, y) == 1.0
    assert compute_retention_domain_a(z, y) == 1.0


def test_all_equal_logits_give_exactly_half() -> None:
    """Section 6 tie rule: exact tie counts 0.5."""
    y = _labels()
    z = np.zeros((len(y), 20), np.float32)
    assert compute_cdpa(z, y) == 0.5
    assert compute_wdpa(z, y) == 0.5


def test_inverted_logits_give_zero() -> None:
    y = _labels()
    z = -_onehot_logits(y)  # true class has the smallest logit
    assert compute_cdpa(z, y) == 0.0
    assert compute_wdpa(z, y) == 0.0


def test_pair_counts_are_100_and_90() -> None:
    from experiments.g4_nono_test.metrics import cdpa_pairs, wdpa_pairs

    assert len(cdpa_pairs()) == 100
    assert len(wdpa_pairs()) == 90
    assert all(a in DOMAIN_A and b in DOMAIN_B for a, b in cdpa_pairs())
    assert len(set(wdpa_pairs())) == 90


def test_hand_computed_cdpa_single_error_mode() -> None:
    """Two test images per class; the network always prefers class 0
    over every domain-B class and is otherwise perfect.

    For pairs (0, b): images of class b (2 of 4) are wrong -> 0.5;
    the 90 other pairs are perfect (1.0). CDPA = (10*0.5 + 90*1)/100.
    """
    y = _labels(2)
    z = _onehot_logits(y)
    z[:, 0] = 100.0  # class 0 logit dominates everywhere
    # for images of class 0 itself the argmax is still 0 (correct)
    expected = (10 * 0.5 + 90 * 1.0) / 100
    assert compute_cdpa(z, y) == pytest.approx(expected)


def test_tie_on_a_single_pair_counts_half_per_image() -> None:
    y = _labels(2)
    z = _onehot_logits(y)
    # Force images of class 1 (domain A) to tie between 1 and 2 (B).
    rows = np.where(y == 1)[0]
    z[rows, 1] = 3.0
    z[rows, 2] = 3.0
    # pair (1, 2): class-1 images tie -> 0.5 each; class-2 images ok.
    # accuracy for that pair = (2*0.5 + 2*1) / 4 = 0.75
    # other 99 pairs perfect.
    expected = (0.75 + 99 * 1.0) / 100
    assert compute_cdpa(z, y) == pytest.approx(expected)


def test_wdpa_ignores_cross_domain_errors() -> None:
    y = _labels(2)
    z = _onehot_logits(y)
    # Make domain A image logits always prefer a B class.
    rows = np.where(np.isin(y, DOMAIN_A))[0]
    z[rows, 2] = 50.0
    # restricted to within-domain pairs, class 2 only competes with
    # B classes: A images are not in any (b, b') pair mask, so only
    # B-vs-B pairs are scored and B images are still perfect.
    assert compute_wdpa(z, y) == 1.0
    assert compute_cdpa(z, y) < 1.0


def test_retention_domain_a_restricts_to_10_logits() -> None:
    y = _labels(2)
    z = _onehot_logits(y)
    # B logits huge: a 20-way argmax would fail, restricted must not.
    z[:, list(DOMAIN_B)] += 1000.0
    rows = np.isin(y, DOMAIN_A)
    assert compute_retention_domain_a(z, y) == 1.0
    assert compute_final_accuracy(z, y) < 1.0
    assert rows.sum() == 20


def test_two_way_accuracy_uses_only_first_two_logits() -> None:
    y = np.array([0, 0, 1, 1])
    z = np.zeros((4, 20), np.float32)
    z[:, 0] = [2, 2, 0, 2]
    z[:, 1] = [1, 1, 1, 1]
    z[:, 5] = 99.0  # untrained unit must be ignored
    assert compute_two_way_accuracy(z, y, (0, 1)) == pytest.approx(0.75)
