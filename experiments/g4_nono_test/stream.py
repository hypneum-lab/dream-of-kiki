"""Two-domain Split-CIFAR-100-coarse stream for G4-nono (spec section 5).

Domain A = animals (10 coarse ids), domain B = plants / objects /
scenes (10 coarse ids). 10 tasks of 2 superclasses, consecutive in id
order inside each domain: tasks 1-5 are domain A, tasks 6-10 domain B.
Labels are the coarse ids (0..19) on one shared 20-way head.
"""
from __future__ import annotations

from typing import Final, TypedDict

import numpy as np

DOMAIN_A: Final[tuple[int, ...]] = (0, 1, 7, 8, 11, 12, 13, 14, 15, 16)
DOMAIN_B: Final[tuple[int, ...]] = (2, 3, 4, 5, 6, 9, 10, 17, 18, 19)
TASK_CLASS_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1), (7, 8), (11, 12), (13, 14), (15, 16),
    (2, 3), (4, 5), (6, 9), (10, 17), (18, 19),
)
N_TASKS: Final[int] = 10


def domain_of(coarse_id: int) -> str:
    """Return ``"A"`` or ``"B"`` for a coarse id."""
    if coarse_id in DOMAIN_A:
        return "A"
    if coarse_id in DOMAIN_B:
        return "B"
    raise ValueError(f"coarse id out of range: {coarse_id}")


class DomainTask(TypedDict):
    """One 2-superclass task (NHWC float32 in [0, 1], coarse labels)."""

    x_train_nhwc: np.ndarray
    y_train: np.ndarray
    x_test_nhwc: np.ndarray
    y_test: np.ndarray


def build_domain_stream(
    x_train_raw: np.ndarray,
    coarse_train: np.ndarray,
    x_test_raw: np.ndarray,
    coarse_test: np.ndarray,
) -> list[DomainTask]:
    """Build the 10-task stream from raw uint8 NHWC + coarse labels.

    Input scaling is the G4-sexto default (``/ 255``). Task
    composition is fixed (no per-seed reshuffle); every image of the
    task's two superclasses is used, in original archive order.
    """
    x_tr = x_train_raw.astype(np.float32) / 255.0
    x_te = x_test_raw.astype(np.float32) / 255.0
    tasks: list[DomainTask] = []
    for pair in TASK_CLASS_PAIRS:
        tr = np.isin(coarse_train, pair)
        te = np.isin(coarse_test, pair)
        tasks.append(
            DomainTask(
                x_train_nhwc=x_tr[tr],
                y_train=coarse_train[tr].astype(np.int64),
                x_test_nhwc=x_te[te],
                y_test=coarse_test[te].astype(np.int64),
            )
        )
    return tasks
