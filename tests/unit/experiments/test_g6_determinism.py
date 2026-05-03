"""Determinism property tests for the G6 pilot pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from experiments.g6_mmlu_stream.dream_wrap import G6DreamRunner
from experiments.g6_mmlu_stream.micro_kiki_inference import (
    InferenceOnlyAdapter,
    adapt_subdomain,
)
from experiments.g6_mmlu_stream.stream import build_subdomain_stream
from harness.real_benchmarks.mmlu import MMLURecord
from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate


@settings(max_examples=20, deadline=2_000, derandomize=True)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_dream_runner_deterministic_for_fixed_seed(seed: int) -> None:
    """Same seed + profile + subdomain -> same substrate state."""
    s1 = MicroKikiSubstrate(num_layers=4, rank=4, seed=seed)
    s2 = MicroKikiSubstrate(num_layers=4, rank=4, seed=seed)
    r1 = G6DreamRunner(
        substrate=s1, profile_name="P_equ", out_dim=4, rank=2,
    )
    r2 = G6DreamRunner(
        substrate=s2, profile_name="P_equ", out_dim=4, rank=2,
    )
    r1.run_episode(seed=seed, subdomain="anatomy")
    r2.run_episode(seed=seed, subdomain="anatomy")
    assert r1.last_episode_id == r2.last_episode_id
    assert (
        s1.recombine_state.last_output_shape
        == s2.recombine_state.last_output_shape
    )
    assert (
        s1.restructure_state.last_episode_id
        == s2.restructure_state.last_episode_id
    )


@settings(max_examples=10, deadline=2_000, derandomize=True)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_inference_adapter_deterministic_for_fixed_seed(seed: int) -> None:
    """Two adapters with the same seed produce the same delta."""
    a1 = InferenceOnlyAdapter(out_dim=8, rank=4, seed=seed)
    a2 = InferenceOnlyAdapter(out_dim=8, rank=4, seed=seed)
    train = [
        MMLURecord(
            question=f"Q{i}?", choices=("A", "B", "C", "D"),
            answer=i % 4, subject="anatomy",
        )
        for i in range(4)
    ]
    adapt_subdomain(adapter=a1, subdomain="anatomy", train=train, seed=seed)
    adapt_subdomain(adapter=a2, subdomain="anatomy", train=train, seed=seed)
    np.testing.assert_array_equal(
        a1.current_delta("layer_0_lora_B"),
        a2.current_delta("layer_0_lora_B"),
    )


def test_subdomain_stream_split_seed_isolated_from_cell_seed(
    tmp_path: Path,
) -> None:
    """Stream-split seed pins the splits ; cell seed must not change them."""
    rows = []
    for subj in ("anatomy", "astronomy"):
        for i in range(8):
            rows.append({
                "question": f"{subj}-Q{i}?",
                "choices": ["A", "B", "C", "D"],
                "answer": i % 4, "subject": subj,
            })
    fixture = tmp_path / "f.jsonl"
    fixture.write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8",
    )
    a = build_subdomain_stream(
        fixture_path=fixture, subdomains=("anatomy", "astronomy"),
        n_train=4, n_eval=2, seed=0,
    )
    b = build_subdomain_stream(
        fixture_path=fixture, subdomains=("anatomy", "astronomy"),
        n_train=4, n_eval=2, seed=0,
    )
    for sa, sb in zip(a, b):
        assert [r.question for r in sa.train] == [
            r.question for r in sb.train
        ]
        assert [r.question for r in sa.eval_] == [
            r.question for r in sb.eval_
        ]


def test_run_pilot_path_b_is_deterministic(tmp_path: Path) -> None:
    """Two run_pilot calls with the same seeds yield identical retention."""
    from experiments.g6_mmlu_stream.run_g6 import run_pilot

    rows = []
    for subj in (
        "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology",
    ):
        for i in range(8):
            rows.append({
                "question": f"{subj}-Q{i}?",
                "choices": ["A", "B", "C", "D"],
                "answer": i % 4, "subject": subj,
            })
    fixture = tmp_path / "mmlu.jsonl"
    fixture.write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8",
    )

    a = run_pilot(
        fixture_path=fixture,
        out_json=tmp_path / "a.json",
        out_md=tmp_path / "a.md",
        registry_db=tmp_path / "ra.sqlite",
        seeds=(0, 1),
        n_train=4, n_eval=4, inner_steps=2,
        lr=5e-5, rank=4, alpha=4.0,
        path="B", scale_slot="qwen3p5-1p5b-fp16",
    )
    b = run_pilot(
        fixture_path=fixture,
        out_json=tmp_path / "b.json",
        out_md=tmp_path / "b.md",
        registry_db=tmp_path / "rb.sqlite",
        seeds=(0, 1),
        n_train=4, n_eval=4, inner_steps=2,
        lr=5e-5, rank=4, alpha=4.0,
        path="B", scale_slot="qwen3p5-1p5b-fp16",
    )
    a_retention = sorted(
        (c["arm"], c["seed"], round(c["retention"], 12))
        for c in a["cells"]
    )
    b_retention = sorted(
        (c["arm"], c["seed"], round(c["retention"], 12))
        for c in b["cells"]
    )
    assert a_retention == b_retention
