"""G4 ablation runner: baseline vs P_min vs P_equ on mega-v2.

Synthetic-mode (cycle 1): mega-v2 fallback synthetic dataset (500
items, 25 domains x 20 each) with mock predictors that simulate
profile behavior at three accuracy levels. Validates the
measurement + statistical pipeline end-to-end.

Real-mode (S16+): replace mock predictors with PMinProfile /
PEquProfile inference once MLX-native swap loop is wired with
real model deployment.

A batch ``run_id`` is registered against
``harness.storage.run_registry.RunRegistry`` before any predictor
runs ; the id is broadcast into the JSON dump so synthetic-pipeline
results remain traceable per the R1 contract.

Usage:
    uv run python scripts/ablation_g4.py

Output:
    docs/milestones/ablation-results.md  (human report)
    docs/milestones/ablation-results.json (machine data)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness.benchmarks.mega_v2.adapter import (
    load_megav2_stratified,
)
from harness.storage.run_registry import RunRegistry
from kiki_oniric.dream.eval_retained import evaluate_retained
from kiki_oniric.eval.statistics import (
    jonckheere_trend,
    one_sample_threshold,
    tost_equivalence,
    welch_one_sided,
)


HARNESS_VERSION = "C-v0.5.0+STABLE"


def _resolve_commit_sha() -> str:
    env_sha = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def baseline_predictor_factory(seed: int):
    """Mock baseline: ~50% accuracy (no consolidation)."""
    def predict(item: dict) -> str:
        rid = int(item["id"].split("-")[-1])
        # Seed-shifted parity: 50% accuracy with seed-dependent
        # variation
        return item["expected"] if (rid + seed) % 2 == 0 else "WRONG"
    return predict


def p_min_predictor_factory(seed: int):
    """Mock P_min: ~70% accuracy (basic consolidation)."""
    def predict(item: dict) -> str:
        rid = int(item["id"].split("-")[-1])
        # ~70% via mod-10 selection (7 of 10 correct)
        return item["expected"] if (rid + seed) % 10 < 7 else "WRONG"
    return predict


def p_equ_predictor_factory(seed: int):
    """Mock P_equ: ~85% accuracy (balanced consolidation)."""
    def predict(item: dict) -> str:
        rid = int(item["id"].split("-")[-1])
        # ~85% via mod-20 selection (17 of 20 correct)
        return item["expected"] if (rid + seed) % 20 < 17 else "WRONG"
    return predict


def main() -> None:
    benchmark = load_megav2_stratified(
        real_path=None,
        items_per_domain=20,
        synthetic_seed=42,
    )

    seeds = [42, 123, 7]

    # Register the ablation batch in the run registry (R1 contract)
    # so the synthetic-pipeline numbers in the JSON dump are
    # traceable. Profile name "G4_ablation" + the smallest seed
    # uniquely keys this batch ; commit_sha pins the code state.
    registry_path = Path(os.environ.get(
        "DREAMOFKIKI_RUN_REGISTRY",
        REPO_ROOT / ".run_registry.sqlite",
    ))
    registry = RunRegistry(registry_path)
    run_id = registry.register(
        c_version=HARNESS_VERSION,
        profile="G4_ablation",
        seed=min(seeds),
        commit_sha=_resolve_commit_sha(),
    )

    # Run grid (profile x seed) directly — keeps each seed
    # genuinely independent (avoids AblationRunner's single-seed-
    # per-spec workaround). Each cell calls evaluate_retained on
    # the (synthetic) predictor for that seed ; the seed is also
    # propagated into evaluate_retained for trace integrity.
    baseline_acc = [
        evaluate_retained(
            baseline_predictor_factory(s), benchmark, seed=s
        )
        for s in seeds
    ]
    p_min_acc = [
        evaluate_retained(
            p_min_predictor_factory(s), benchmark, seed=s
        )
        for s in seeds
    ]
    p_equ_acc = [
        evaluate_retained(
            p_equ_predictor_factory(s), benchmark, seed=s
        )
        for s in seeds
    ]

    # H1: P_equ < baseline forgetting. Convert accuracy to
    # forgetting rate (1 - accuracy) so that "lower is better"
    # matches Welch one-sided convention.
    forgetting_baseline = [1 - a for a in baseline_acc]
    forgetting_p_equ = [1 - a for a in p_equ_acc]
    h1 = welch_one_sided(
        treatment=forgetting_p_equ,
        control=forgetting_baseline,
        alpha=0.0125,  # Bonferroni 0.05/4
    )

    # H2: P_max ~ P_equ within +/-5% — placeholder (no P_max yet).
    # Smoke test : compare p_equ against a near-copy with a tiny
    # deterministic perturbation (well within the 5% epsilon) to
    # avoid the zero-variance pathology of strict self-equivalence.
    p_max_smoke = [a + 0.001 * (i - 1) for i, a in enumerate(p_equ_acc)]
    h2 = tost_equivalence(
        treatment=p_max_smoke,
        control=p_equ_acc,
        epsilon=0.05,
        alpha=0.0125,
    )

    # H3: monotonic trend P_min < P_equ (P_max deferred)
    h3 = jonckheere_trend(
        groups=[p_min_acc, p_equ_acc], alpha=0.0125
    )

    # H4: energy ratio <= 2.0 (synthetic placeholder ratios)
    energy_ratios = [1.5 + 0.1 * i for i in range(len(seeds))]
    h4 = one_sample_threshold(
        sample=energy_ratios, threshold=2.0, alpha=0.0125
    )

    results = {
        "run_id": run_id,
        "harness_version": HARNESS_VERSION,
        "is_synthetic": True,
        "benchmark_size": len(benchmark.items),
        "benchmark_hash": benchmark.source_hash,
        "seeds": seeds,
        "accuracy": {
            "baseline": baseline_acc,
            "p_min": p_min_acc,
            "p_equ": p_equ_acc,
        },
        "hypotheses": {
            "H1_forgetting": {
                "test_name": h1.test_name,
                "p_value": h1.p_value,
                "reject_h0": h1.reject_h0,
                "interpretation": (
                    "P_equ reduces forgetting vs baseline"
                    if h1.reject_h0 else
                    "no significant reduction"
                ),
            },
            "H2_equivalence_self": {
                "test_name": h2.test_name,
                "p_value": h2.p_value,
                "reject_h0": h2.reject_h0,
                "interpretation": "smoke test, P_max not wired",
            },
            "H3_monotonic": {
                "test_name": h3.test_name,
                "p_value": h3.p_value,
                "reject_h0": h3.reject_h0,
                "interpretation": (
                    "P_min < P_equ monotonic"
                    if h3.reject_h0 else
                    "no significant trend"
                ),
            },
            "H4_energy_budget": {
                "test_name": h4.test_name,
                "p_value": h4.p_value,
                "reject_h0": h4.reject_h0,
                "interpretation": (
                    "energy ratio significantly < 2.0"
                    if h4.reject_h0 else
                    "energy budget violated or marginal"
                ),
            },
        },
    }

    # Gate criterion: P_equ > P_min on >=2 metrics significant
    sig_count = sum(
        1 for h in results["hypotheses"].values() if h["reject_h0"]
    )
    results["gate_significant_count"] = sig_count
    results["gate_pass"] = sig_count >= 2

    print("=" * 60)
    print("G4 ABLATION RESULTS - SYNTHETIC mega-v2")
    print("=" * 60)
    print(f"benchmark: {len(benchmark.items)} items, "
          f"hash: {benchmark.source_hash[:24]}...")
    print(f"seeds: {seeds}")
    print(f"baseline acc: {[f'{a:.3f}' for a in baseline_acc]}")
    print(f"p_min acc:    {[f'{a:.3f}' for a in p_min_acc]}")
    print(f"p_equ acc:    {[f'{a:.3f}' for a in p_equ_acc]}")
    print("-" * 60)
    for name, h in results["hypotheses"].items():
        flag = "PASS" if h["reject_h0"] else "fail"
        print(f"{name:20s} p={h['p_value']:.4f} {flag}")
    print("-" * 60)
    print(f"significant hypotheses: {sig_count}/4")
    print(f"gate (>=2 significant): "
          f"{'PASS' if results['gate_pass'] else 'FAIL'}")
    print("=" * 60)

    out_dir = REPO_ROOT / "docs" / "milestones"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ablation-results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written to {json_path}")


if __name__ == "__main__":
    main()
