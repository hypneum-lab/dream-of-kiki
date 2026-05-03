"""Tests for the G6 MMLU subdomain stream loader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.g6_mmlu_stream.stream import (
    SubdomainSplit,
    build_subdomain_stream,
)
from harness.real_benchmarks.mmlu import MMLURecord


def _write_fixture(tmp_path: Path) -> Path:
    """Write 8 records each across 5 target subdomains (40 rows total)."""
    rows: list[dict[str, object]] = []
    subjects = (
        "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology",
    )
    for subj in subjects:
        for i in range(8):
            rows.append({
                "question": f"{subj}-Q{i}?",
                "choices": ["A", "B", "C", "D"],
                "answer": i % 4,
                "subject": subj,
            })
    # Plus 3 distractor records that should never make it into a split.
    for i in range(3):
        rows.append({
            "question": f"distractor-{i}",
            "choices": ["A", "B", "C", "D"],
            "answer": 0,
            "subject": "elementary_mathematics",
        })
    path = tmp_path / "mmlu_subset.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def test_build_subdomain_stream_returns_5_splits(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    splits = build_subdomain_stream(
        fixture_path=fixture,
        subdomains=(
            "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology",
        ),
        n_train=4,
        n_eval=4,
        seed=0,
    )
    assert len(splits) == 5
    for split in splits:
        assert isinstance(split, SubdomainSplit)
        assert split.subject in {
            "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology",
        }
        assert len(split.train) == 4
        assert len(split.eval_) == 4
        # No leakage between train and eval inside a single subject.
        train_qs = {r.question for r in split.train}
        eval_qs = {r.question for r in split.eval_}
        assert train_qs.isdisjoint(eval_qs), (
            f"train/eval overlap in {split.subject}: "
            f"{train_qs & eval_qs}"
        )
        # No cross-subject contamination.
        for r in split.train + split.eval_:
            assert isinstance(r, MMLURecord)
            assert r.subject == split.subject


def test_build_subdomain_stream_is_deterministic(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    a = build_subdomain_stream(
        fixture_path=fixture,
        subdomains=(
            "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology",
        ),
        n_train=4, n_eval=4, seed=42,
    )
    b = build_subdomain_stream(
        fixture_path=fixture,
        subdomains=(
            "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology",
        ),
        n_train=4, n_eval=4, seed=42,
    )
    assert [s.subject for s in a] == [s.subject for s in b]
    for sa, sb in zip(a, b):
        assert [r.question for r in sa.train] == [
            r.question for r in sb.train
        ]
        assert [r.question for r in sa.eval_] == [
            r.question for r in sb.eval_
        ]


def test_build_subdomain_stream_raises_on_insufficient_rows(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="not enough records"):
        build_subdomain_stream(
            fixture_path=fixture,
            subdomains=("anatomy",),
            n_train=20, n_eval=20, seed=0,  # only 8 rows for anatomy
        )


def test_build_subdomain_stream_raises_on_unknown_subject(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    with pytest.raises(KeyError, match="no MMLU records for subject"):
        build_subdomain_stream(
            fixture_path=fixture,
            subdomains=("not_a_real_subject",),
            n_train=2, n_eval=2, seed=0,
        )
