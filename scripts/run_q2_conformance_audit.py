"""Q2 — Structural-invariant conformance audit on 15 adversarial substrates.

Plan: HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md Task 9.
Pre-reg: docs/milestones/q2-conformance-negative-2026-05-10.md.

Scope: structural invariants only (S2 finite + range + nonneg + bounded
delta_acc). Substrate-specific axiom property tests (C2) are NOT
included — they are hardcoded per real substrate and cannot run
against arbitrary adversarial substrates.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from kiki_oniric.substrates._adversarial.cat_a_trivial import (
    ConstantSubstrate,
    FrozenZerosSubstrate,
    IdentitySubstrate,
    LookupSubstrate,
    RandomNoiseSubstrate,
)
from kiki_oniric.substrates._adversarial.cat_b_adversarial import (
    BoundaryCheater,
    BudgetGamingSubstrate,
    CommutativityViolator,
    OverloadRecombine,
    PermutedReplay,
    ShapePreservingNoise,
    StatelessAccident,
)
from kiki_oniric.substrates._adversarial.cat_c_statistical import (
    RandomCoinFlip,
    SeedDependentSubstrate,
    ShapeDistributionDependent,
)
from kiki_oniric.substrates._adversarial.cat_d_modular import (
    AffineModN,
    BinaryGray,
    Mod7Cycler,
    Mod13Hasher,
)
from kiki_oniric.substrates._adversarial.cat_e_replay_loops import (
    AlphaInfiniteRecycle,
    BetaPermutationLeak,
    GammaSnapshotRollback,
    ReplayHistorySpoof,
)
from kiki_oniric.substrates._adversarial.cat_f_asymmetric import (
    AsymRecombineEmpty,
    AsymReplayHeavy,
)
from kiki_oniric.substrates.factory import CellRequest


W_MAX = 1e6  # mirrors finite.DEFAULT_W_MAX


@dataclass
class TrialResult:
    seed: int
    passed: bool
    failures: list[str] = field(default_factory=list)


@dataclass
class SubstrateAudit:
    name: str
    category: str
    n_trials: int
    pass_count: int
    fail_count: int
    fp_rate: float
    sample_failures: list[str]


def check_invariants(metrics: dict[str, float]) -> list[str]:
    """Return list of failure reasons; empty list means all invariants pass."""
    failures: list[str] = []
    required = ("replay_rate", "downscale_norm", "restructure_sum",
                "recombine_rate", "delta_acc", "wall_time_s")
    # 1. S2 finite
    for key in required:
        if key not in metrics:
            failures.append(f"missing key '{key}'")
            continue
        v = metrics[key]
        if not isinstance(v, (int, float)):
            failures.append(f"{key}: non-numeric ({type(v).__name__})")
            continue
        if math.isnan(v):
            failures.append(f"{key}: NaN")
        elif math.isinf(v):
            failures.append(f"{key}: Inf")
        elif abs(v) > W_MAX:
            failures.append(f"{key}: |val|={abs(v)} > w_max={W_MAX}")
    # 2. range bounds (S4-style)
    for key in ("replay_rate", "recombine_rate"):
        if key in metrics and isinstance(metrics[key], (int, float)):
            v = metrics[key]
            if not math.isnan(v) and not math.isinf(v):
                if v < 0.0 or v > 1.0:
                    failures.append(f"{key}: {v} outside [0,1]")
    # 3. nonneg
    for key in ("restructure_sum", "wall_time_s"):
        if key in metrics and isinstance(metrics[key], (int, float)):
            v = metrics[key]
            if not math.isnan(v) and not math.isinf(v):
                if v < 0.0:
                    failures.append(f"{key}: {v} < 0")
    # 4. bounded delta_acc
    if "delta_acc" in metrics and isinstance(metrics["delta_acc"], (int, float)):
        v = metrics["delta_acc"]
        if not math.isnan(v) and not math.isinf(v):
            if abs(v) > 0.1:
                failures.append(f"delta_acc: |{v}| > 0.1")
    return failures


def run_substrate(
    substrate_cls: type,
    n_trials: int,
    seed_iter: Callable[[int], int],
) -> SubstrateAudit:
    pass_count = 0
    fail_count = 0
    sample_failures: list[str] = []
    for trial in range(n_trials):
        seed = seed_iter(trial)
        s = substrate_cls()
        req = CellRequest(
            substrate="mlx_kiki_oniric",
            profile="p_min",
            seed=seed,
            scale="small",
            model_path=Path("/tmp"),
        )
        try:
            metrics = s.execute_profile(req)
            failures = check_invariants(metrics)
        finally:
            s.teardown()
        if failures:
            fail_count += 1
            if len(sample_failures) < 3:
                sample_failures.append(f"seed={seed}: " + "; ".join(failures))
        else:
            pass_count += 1
    return SubstrateAudit(
        name=substrate_cls.__name__,
        category="",
        n_trials=n_trials,
        pass_count=pass_count,
        fail_count=fail_count,
        fp_rate=pass_count / n_trials if n_trials else 0.0,
        sample_failures=sample_failures,
    )


CAT_A = [IdentitySubstrate, RandomNoiseSubstrate, LookupSubstrate,
         FrozenZerosSubstrate, ConstantSubstrate]
CAT_B = [ShapePreservingNoise, BudgetGamingSubstrate, PermutedReplay,
         OverloadRecombine, StatelessAccident, CommutativityViolator,
         BoundaryCheater]
CAT_C = [RandomCoinFlip, ShapeDistributionDependent, SeedDependentSubstrate]
CAT_D = [Mod7Cycler, Mod13Hasher, BinaryGray, AffineModN]
CAT_E = [ReplayHistorySpoof, BetaPermutationLeak,
         AlphaInfiniteRecycle, GammaSnapshotRollback]
CAT_F = [AsymReplayHeavy, AsymRecombineEmpty]


def main() -> int:
    out_path = Path("docs/milestones/q2plus-conformance-negative-results.json")
    audits: list[SubstrateAudit] = []
    for cat_name, cat_classes, n_trials, seed_iter in [
        ("A_trivial", CAT_A, 1, lambda t: 0),
        ("B_adversarial", CAT_B, 1, lambda t: 0),
        ("C_statistical", CAT_C, 100, lambda t: t),
        ("D_modular", CAT_D, 1, lambda t: 0),
        ("E_replay_loops", CAT_E, 1, lambda t: 0),
        ("F_asymmetric", CAT_F, 1, lambda t: 0),
    ]:
        for cls in cat_classes:
            audit = run_substrate(cls, n_trials, seed_iter)
            audit.category = cat_name
            audits.append(audit)
            print(f"[{cat_name}] {audit.name}: pass={audit.pass_count}/"
                  f"{audit.n_trials} fp_rate={audit.fp_rate:.2%}")

    fp_substrates = [a for a in audits if a.pass_count > 0]
    n_fp = len(fp_substrates)
    n_total = len(audits)
    print()
    print(f"Substrates with non-zero pass rate (= FP candidates): "
          f"{n_fp} / {n_total}")
    if n_fp == 0:
        verdict = "0_FP_robust"
    elif n_fp <= 2:
        verdict = "1_to_2_FP_strengthen"
    else:
        verdict = "ge_3_FP_reformulate"
    print(f"Verdict per N8 pre-registration: {verdict}")

    # Q2+ pre-reg brackets (25-substrate pool)
    if n_fp == n_total:
        q2plus_verdict = f"q2plus_{n_fp}_{n_total}_FP_C2_required"
    elif n_total - n_fp <= 3:
        q2plus_verdict = f"q2plus_{n_fp}_{n_total}_FP_some_discrimination"
    else:
        q2plus_verdict = f"q2plus_{n_fp}_{n_total}_FP_strong_discrimination"
    print(f"Verdict per Q2+ pre-registration: {q2plus_verdict}")

    # Per-category FP counts
    by_cat: dict[str, list[SubstrateAudit]] = {}
    for a in audits:
        by_cat.setdefault(a.category, []).append(a)
    cat_fp_counts = {
        c: {
            "fp": sum(1 for a in lst if a.pass_count > 0),
            "total": len(lst),
        }
        for c, lst in by_cat.items()
    }
    print("Per-category FP counts:")
    for c, d in cat_fp_counts.items():
        print(f"  {c}: {d['fp']}/{d['total']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "verdict": verdict,
        "q2plus_verdict": q2plus_verdict,
        "n_fp_substrates": n_fp,
        "n_total": n_total,
        "per_category": cat_fp_counts,
        "audits": [asdict(a) for a in audits],
    }, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
