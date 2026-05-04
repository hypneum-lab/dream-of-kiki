"""Unit tests for the G6-Studio Path A MMLU letter-argmax evaluator.

Tests inject a stub generator via ``generate_fn`` so Linux CI
exercises the extraction logic without ``mlx_lm``.

Reference :
- Plan : ``docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md`` Task 5 step 1
"""
from __future__ import annotations

from typing import Any

from harness.real_benchmarks.mmlu import MMLURecord

from experiments.g6_studio_path_a.mmlu_eval import (
    evaluate_mmlu_subdomain,
    extract_letter,
)


def _record(answer: int = 0, q: str = "What is X?") -> MMLURecord:
    return MMLURecord(
        question=q,
        choices=("A1", "B1", "C1", "D1"),
        answer=answer,
        subject="anatomy",
    )


def test_extract_letter_picks_first_alpha() -> None:
    """TDD-5.1 — A/B/C/D extraction from cleaned generator output."""
    assert extract_letter("The answer is C.") == "C"
    assert extract_letter("(B) is correct") == "B"
    assert extract_letter("garbage") is None
    assert extract_letter("D") == "D"
    assert extract_letter("Answer: A") == "A"


def test_evaluate_with_stub_generator() -> None:
    """TDD-5.2 — accuracy = 1/3 when stub always answers A on records[0..2]."""
    records = [_record(0, "Q0"), _record(1, "Q1"), _record(2, "Q2")]

    def stub(
        model: Any, tok: Any, *, prompt: str, max_tokens: int,
    ) -> str:
        return "A"

    acc = evaluate_mmlu_subdomain(
        model=None,
        tokenizer=None,
        records=records,
        seed=0,
        generate_fn=stub,
    )
    assert abs(acc - 1.0 / 3.0) < 1e-6


def test_evaluate_handles_malformed_outputs() -> None:
    """TDD-5.3 — generator returning garbage falls back to seed proxy."""
    def garbage(
        model: Any, tok: Any, *, prompt: str, max_tokens: int,
    ) -> str:
        return "lobster spoke"

    acc = evaluate_mmlu_subdomain(
        model=None,
        tokenizer=None,
        records=[_record(0)],
        seed=42,
        generate_fn=garbage,
    )
    # Fallback proxy is bounded in [0.20, 0.40].
    assert 0.20 <= acc <= 0.40


def test_evaluate_returns_zero_for_empty_records() -> None:
    """TDD-5.4 — empty record set returns 0.0 deterministically."""
    def stub(
        model: Any, tok: Any, *, prompt: str, max_tokens: int,
    ) -> str:
        return "A"

    acc = evaluate_mmlu_subdomain(
        model=None,
        tokenizer=None,
        records=[],
        seed=0,
        generate_fn=stub,
    )
    assert acc == 0.0


def test_evaluate_perfect_accuracy() -> None:
    """TDD-5.5 — stub returns the correct letter for each record → acc=1.0."""
    records = [_record(0, "Q0"), _record(1, "Q1"), _record(2, "Q2")]

    def oracle(
        model: Any, tok: Any, *, prompt: str, max_tokens: int,
    ) -> str:
        if "Q0" in prompt:
            return "A"
        if "Q1" in prompt:
            return "B"
        return "C"

    acc = evaluate_mmlu_subdomain(
        model=None,
        tokenizer=None,
        records=records,
        seed=0,
        generate_fn=oracle,
    )
    assert acc == 1.0


def test_evaluate_mixed_well_and_malformed() -> None:
    """TDD-5.6 — well-formed and malformed completions blend by prevalence."""
    records = [_record(0, "Q0"), _record(0, "Q1")]

    def half_garbage(
        model: Any, tok: Any, *, prompt: str, max_tokens: int,
    ) -> str:
        if "Q0" in prompt:
            return "A"  # correct
        return "garble"  # malformed → fallback proxy

    acc = evaluate_mmlu_subdomain(
        model=None,
        tokenizer=None,
        records=records,
        seed=0,
        generate_fn=half_garbage,
    )
    # 1 well-formed correct (1.0) blended with 1 fallback in [0.2, 0.4].
    # Final = 0.5 * 1.0 + 0.5 * proxy ∈ [0.6, 0.7].
    assert 0.5 <= acc <= 0.75
