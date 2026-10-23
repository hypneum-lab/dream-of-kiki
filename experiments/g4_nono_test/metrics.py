"""G4-nono outcome metrics (spec section 6) on raw logits.

All functions take ``logits`` of shape ``(N, 20)`` and integer coarse
``labels`` of shape ``(N,)`` and return plain floats.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from experiments.g4_nono_test.stream import DOMAIN_A, DOMAIN_B


class NonFiniteLogitsError(ValueError):
    """Logits contain NaN / Inf: the cell is excluded (section 4)."""


def check_finite(logits: np.ndarray) -> None:
    """Raise :class:`NonFiniteLogitsError` on any non-finite logit."""
    if not np.isfinite(logits).all():
        raise NonFiniteLogitsError("non-finite logits")


def cdpa_pairs() -> list[tuple[int, int]]:
    """The 100 (a in A, b in B) class pairs."""
    return [(a, b) for a in DOMAIN_A for b in DOMAIN_B]


def wdpa_pairs() -> list[tuple[int, int]]:
    """The 45 + 45 within-domain class pairs."""
    return list(combinations(DOMAIN_A, 2)) + list(
        combinations(DOMAIN_B, 2)
    )


def pairwise_accuracy(
    logits: np.ndarray,
    labels: np.ndarray,
    pairs: list[tuple[int, int]],
) -> float:
    """Unweighted mean over ``pairs`` of the binary accuracy.

    For pair ``(a, b)``: over the test images of class ``a`` or ``b``
    predict the class with the larger of the two logits; an exact
    tie counts 0.5 (section 6).
    """
    check_finite(logits)
    per_pair: list[float] = []
    for a, b in pairs:
        mask = (labels == a) | (labels == b)
        if not mask.any():
            raise ValueError(f"no test images for pair ({a}, {b})")
        la = logits[mask, a]
        lb = logits[mask, b]
        is_a = labels[mask] == a
        score = np.where(
            la == lb, 0.5, np.where(la > lb, is_a, ~is_a)
        ).astype(np.float64)
        per_pair.append(float(score.mean()))
    return float(np.mean(per_pair))


def compute_cdpa(logits: np.ndarray, labels: np.ndarray) -> float:
    """Cross-domain pairwise accuracy (100 pairs)."""
    return pairwise_accuracy(logits, labels, cdpa_pairs())


def compute_wdpa(logits: np.ndarray, labels: np.ndarray) -> float:
    """Within-domain pairwise accuracy (90 pairs)."""
    return pairwise_accuracy(logits, labels, wdpa_pairs())


def compute_retention_domain_a(
    logits: np.ndarray, labels: np.ndarray
) -> float:
    """Accuracy on domain-A test images, restricted to the 10 A logits."""
    check_finite(logits)
    mask = np.isin(labels, DOMAIN_A)
    cols = np.asarray(DOMAIN_A)
    pred = cols[logits[mask][:, cols].argmax(axis=1)]
    return float((pred == labels[mask]).mean())


def compute_final_accuracy(
    logits: np.ndarray, labels: np.ndarray
) -> float:
    """Plain 20-way accuracy."""
    check_finite(logits)
    return float((logits.argmax(axis=1) == labels).mean())


def compute_two_way_accuracy(
    logits: np.ndarray, labels: np.ndarray, classes: tuple[int, int]
) -> float:
    """Accuracy restricted to the logits of ``classes`` (manipulation
    check of section 4: task-1 test images, the two task-1 logits)."""
    check_finite(logits)
    cols = np.asarray(classes)
    pred = cols[logits[:, cols].argmax(axis=1)]
    return float((pred == labels).mean())
