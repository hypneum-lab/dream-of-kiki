"""Ablation runner harness — (profile × seed × benchmark) matrix.

Executes the cartesian product of profile specifications and
seeds against a frozen benchmark, collecting metrics into a
pandas.DataFrame ready for S15.1 statistical tests.

Profile specs are intentionally minimal — they bind a name to a
predictor callable. Real ablation (S15.3) wraps PMin/PEqu profile
inference into predictors. Tests use mock predictors directly.

Each `run()` call registers a single batch ``run_id`` against
``harness.storage.run_registry.RunRegistry`` so every emitted row
is traceable to the registered execution (R1 contract). The seed
is propagated into ``evaluate_retained`` for trace integrity even
when the predictor itself is deterministic — this matches the
project rule that experimental claims resolve to a registered
run_id.

Reference: docs/specs/2026-04-17-dreamofkiki-master-design.md §5
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from harness.benchmarks.retained.retained import RetainedBenchmark
from harness.storage.run_registry import RunRegistry
from kiki_oniric.dream.eval_retained import evaluate_retained


ItemPredictor = Callable[[dict], str]


def _resolve_commit_sha() -> str:
    """Best-effort git HEAD lookup ; fallback to env / 'unknown'."""
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


def _default_registry_path() -> Path:
    return Path(os.environ.get(
        "DREAMOFKIKI_RUN_REGISTRY",
        Path.cwd() / ".run_registry.sqlite",
    ))


@dataclass(frozen=True)
class ProfileSpec:
    """Binding of profile name to predictor callable."""

    name: str
    predictor: ItemPredictor


@dataclass
class AblationRunner:
    """Run (profile × seed) matrix on a frozen benchmark.

    Each cell calls `evaluate_retained(spec.predictor, benchmark,
    seed=seed)` and records a row in the output DataFrame. Seeds
    are recorded alongside the result for downstream statistical
    handling (e.g., paired tests across seeds).

    Run registration: a single ``run_id`` is computed per
    ``run()`` invocation from a deterministic key derived from the
    batch (profile names + seeds + benchmark hash) and inserted
    into the registry under profile ``"ablation_batch"``. The id
    is broadcast to every output row.
    """

    profile_specs: list[ProfileSpec]
    seeds: list[int]
    benchmark: RetainedBenchmark
    c_version: str = "C-v0.5.0+STABLE"
    registry_path: Path = field(default_factory=_default_registry_path)

    def _batch_seed(self) -> int:
        """Deterministic 31-bit seed derived from the batch shape."""
        names = "|".join(spec.name for spec in self.profile_specs)
        seeds = ",".join(str(s) for s in self.seeds)
        bench = self.benchmark.source_hash or ""
        key = f"{names}::{seeds}::{bench}".encode()
        digest = hashlib.sha256(key).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

    def _register(self) -> str:
        registry = RunRegistry(self.registry_path)
        return registry.register(
            c_version=self.c_version,
            profile="ablation_batch",
            seed=self._batch_seed(),
            commit_sha=_resolve_commit_sha(),
        )

    def run(self) -> pd.DataFrame:
        """Execute the full grid and return results DataFrame."""
        run_id = self._register()
        rows: list[dict] = []
        for spec in self.profile_specs:
            for seed in self.seeds:
                acc = evaluate_retained(
                    spec.predictor, self.benchmark, seed=seed
                )
                rows.append({
                    "run_id": run_id,
                    "profile": spec.name,
                    "seed": seed,
                    "accuracy": acc,
                    "benchmark_hash": self.benchmark.source_hash,
                })
        return pd.DataFrame(rows)
