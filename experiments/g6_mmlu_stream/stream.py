"""MMLU subdomain stream loader for the G6 pilot.

Builds a sequence of SubdomainSplit (train, eval_) pairs, one per
target subject, drawn deterministically from a JSONL fixture matching
the HF cais/mmlu schema. Used by the G6 pilot driver to construct a
continual-learning task stream.

The loader is network-free : it consumes a pre-materialised JSONL
fixture (see ``tests/fixtures/mmlu_sanity.jsonl`` for shape, but note
that the production pilot requires the full cais/mmlu export — the
sanity fixture has < 5 records per production subject).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from harness.real_benchmarks.mmlu import MMLURecord


def _stable_per_subject_seed(seed: int, subject: str) -> int:
    """Derive a stable per-subject int seed from (seed, subject).

    Python 3.14 ``random.Random`` only accepts hashable scalars
    (int / float / str / bytes / bytearray / None) as seed —
    tuples are rejected. Combine the cell-level seed with the
    subject name via SHA-256 to get a stable int seed unique per
    (seed, subject) pair.
    """
    raw = f"{seed}|{subject}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


@dataclass(frozen=True)
class SubdomainSplit:
    """Frozen (train, eval) split for a single MMLU subject.

    Fields
    ------
    subject
        MMLU subject (one of the 57 cais/mmlu subjects).
    train
        Training records (input features for the per-subdomain
        adaptation step). Ordered as drawn — driver re-shuffles
        if it wants epoch-style mini-batches.
    eval_
        Held-out evaluation records (disjoint from ``train``).
        Used by `evaluate_mmlu` after each subdomain step.
    """

    subject: str
    train: list[MMLURecord]
    eval_: list[MMLURecord]


def _record_from_raw(row: dict) -> MMLURecord:
    """Validate a raw JSON row and lift it to a frozen MMLURecord.

    Mirrors `MMLULoader._record_from_raw` (private) so this loader
    stays decoupled from MMLULoader's R1-pin lifecycle.
    """
    choices = row["choices"]
    if len(choices) != 4:
        raise ValueError(
            f"MMLU row has {len(choices)} choices, expected 4: {row!r}"
        )
    answer = int(row["answer"])
    if not 0 <= answer <= 3:
        raise ValueError(
            f"MMLU row has answer index {answer} outside [0,3]: {row!r}"
        )
    return MMLURecord(
        question=str(row["question"]),
        choices=(
            str(choices[0]),
            str(choices[1]),
            str(choices[2]),
            str(choices[3]),
        ),
        answer=answer,
        subject=str(row.get("subject", "unknown")),
    )


def build_subdomain_stream(
    *,
    fixture_path: Path,
    subdomains: Sequence[str],
    n_train: int,
    n_eval: int,
    seed: int,
) -> list[SubdomainSplit]:
    """Build a stream of (train, eval) splits, one per subdomain.

    Parameters
    ----------
    fixture_path
        JSONL file matching the HF cais/mmlu schema. The full
        cais/mmlu ``test`` split exported as JSONL is the production
        target ; the sanity fixture under
        ``tests/fixtures/mmlu_sanity.jsonl`` has < 5 records per
        production subject and only validates pipeline shape.
    subdomains
        Ordered tuple of MMLU subjects forming the CL stream. Order
        matters — it dictates the curriculum.
    n_train
        Number of training records per subdomain. Drawn from the
        per-subject pool via seeded shuffle.
    n_eval
        Number of held-out eval records per subdomain. Drawn from
        the per-subject pool *after* removing the train slice
        (no leakage).
    seed
        Pins the per-subject shuffle so the same (fixture, seed) pair
        always yields identical splits.

    Returns
    -------
    list[SubdomainSplit]
        Length matches ``len(subdomains)``, in the same order.

    Raises
    ------
    KeyError
        A subject in ``subdomains`` has no rows in the fixture.
    ValueError
        Per-subject pool has fewer than ``n_train + n_eval`` rows.
    """
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"MMLU stream fixture not found at {fixture_path!s}"
        )
    by_subject: dict[str, list[dict]] = {}
    with fixture_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_subject.setdefault(
                str(row.get("subject", "unknown")), [],
            ).append(row)

    splits: list[SubdomainSplit] = []
    for subject in subdomains:
        pool = by_subject.get(subject)
        if not pool:
            raise KeyError(
                f"no MMLU records for subject {subject!r} in "
                f"{fixture_path!s} (subjects present : "
                f"{sorted(by_subject)})"
            )
        if len(pool) < n_train + n_eval:
            raise ValueError(
                f"not enough records for subject {subject!r}: "
                f"need {n_train + n_eval}, got {len(pool)}"
            )
        # Stable per-subject order without colliding across
        # (seed, subject) pairs — see ``_stable_per_subject_seed``.
        rng = random.Random(_stable_per_subject_seed(seed, subject))
        order = list(range(len(pool)))
        rng.shuffle(order)
        train_idx = order[:n_train]
        eval_idx = order[n_train : n_train + n_eval]
        train = [_record_from_raw(pool[i]) for i in train_idx]
        eval_ = [_record_from_raw(pool[i]) for i in eval_idx]
        splits.append(
            SubdomainSplit(subject=subject, train=train, eval_=eval_),
        )
    return splits


__all__ = ["SubdomainSplit", "build_subdomain_stream"]
