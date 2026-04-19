"""Cycle-3 sanity pilot — 1.5B-scale fail-fast driver (C3.7).

**Gate ID** : G10 cycle-3 Gate D pilot (fail-fast decision)
**Validates** : whether H1 (forgetting reduction, Welch pre vs
post) is detectable at the smallest scale-slot
``qwen3p5-1p5b`` before the user commits ~10 days of Studio
compute to the full 3-scale 1080-config launch (C3.8).
**Mode** : empirical claim at small scale — *pipeline-validation*
at the full 1080 scale (fail-fast only, not used to certify G10).
**Expected output** : a go/no-go verdict JSON dumped under
``docs/milestones/pilot-cycle3-sanity-1p5b.json`` (sibling to the
human-readable milestone report).

Cartesian product :

    1 scale × 3 profiles × 2 substrates × 30 seeds = 180 runs

Half the seed count of the full 60-seed launch per §7 compute
envelope : ~1 day of dedicated Studio time vs ~3-4 days per full
scale-slot. Run registry rows are tagged under the same
``harness_version = C-v0.7.0+PARTIAL`` and ``profile_tag``
convention as the full launch so the pilot's cells are *a subset*
of the full-launch's resume contract. Re-running
``scripts/ablation_cycle3.py --resume`` after the pilot therefore
skips the 180 pilot cells automatically (R1 identity).

GO / NO-GO decision rule (per user spec) :

- **GO** : H1 (Welch pre vs post) rejects the null in ≥ 4 / 6
  cells at α = 0.0125 (cycle-1 bar on 1 scale, Bonferroni-
  corrected for 4 hypotheses). Compute budget cleared, launch
  C3.8 full 3-scale matrix.
- **NO-GO** : H1 rejects in < 4 / 6 cells. Do **not** burn Studio
  compute on 7B + 35B. Open a root-cause review (`pivot-4` branch
  per spec §5.1 R3).

The actual ``evaluate_retained`` loop is **deferred** in this
commit — the script ships as a fully-wired enumeration /
decision template with a ``--dry-run`` flag that validates the
pipeline and prints the planned run count without touching any
predictor or substrate. The user decides when to launch
(decoupled from this commit per C3.7 scope).

Usage ::

    # Enumerate the 180 configs ; no dream ops.
    uv run python scripts/pilot_cycle3_sanity.py --dry-run

    # Full run (1 day Studio compute — do NOT launch from CI).
    uv run python scripts/pilot_cycle3_sanity.py

Reference :
    docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md
    §5.1 R3 (Pivot 4 if Gate D = NO-GO),
    §7 (compute budget C3.7 sanity envelope)
    docs/milestones/pilot-cycle3-sanity-1p5b.md (this script's
    milestone report, populated after run).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ablation_cycle3 import (  # noqa: E402
    AblationCycle3Runner,
    HARNESS_VERSION,
    PROFILES,
    SUBSTRATES,
)

# Sanity-pilot axis restriction (§7 compute envelope).
SANITY_SCALE = "qwen3p5-1p5b"
SANITY_SEEDS = tuple(range(30))
SANITY_PROFILES = PROFILES
SANITY_SUBSTRATES = SUBSTRATES
# 1 × 3 × 2 × 30 = 180 configs per §7.
EXPECTED_CELL_COUNT = (
    1 * len(SANITY_PROFILES) * len(SANITY_SUBSTRATES) * len(SANITY_SEEDS)
)

# GO / NO-GO decision rule (cycle-1 bar on 1 scale).
GO_BONFERRONI_ALPHA = 0.0125
GO_CELLS_REJECTED_MIN = 4  # ≥ 4 of (3 profiles × 2 substrates)


def _parse_cli(argv: list[str]) -> dict:
    """Light argv parser — no click dependency."""
    opts = {"dry_run": False}
    for token in argv:
        if token == "--dry-run":
            opts["dry_run"] = True
    return opts


def _plan() -> list[dict]:
    """Enumerate the 180-cell sanity plan.

    Wraps :class:`AblationCycle3Runner` restricted to the
    1.5B scale and 30 seeds ; preserves the full-launch's
    ``(harness_version, profile_tag, seed, commit_sha) ->
    run_id`` identity so this pilot's rows are a strict subset of
    the eventual 1080-config launch.
    """
    runner = AblationCycle3Runner(
        scales=(SANITY_SCALE,),
        profiles=SANITY_PROFILES,
        substrates=SANITY_SUBSTRATES,
        seeds=SANITY_SEEDS,
    )
    plan = []
    for cfg in runner.enumerate():
        plan.append({
            "scale": cfg.scale,
            "profile": cfg.profile,
            "substrate": cfg.substrate,
            "seed": cfg.seed,
            "run_id": runner.compute_run_id(cfg),
            "profile_tag": runner._registry_profile_tag(cfg),
        })
    return plan


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    """Entrypoint — enumerate (+ dry-run) or launch the pilot.

    This commit ships the enumeration + decision-template only ;
    per-cell ``evaluate_retained`` wiring arrives in a follow-up
    commit once the user decides to launch.
    """
    opts = _parse_cli(list(argv) if argv is not None else sys.argv[1:])
    plan = _plan()
    assert len(plan) == EXPECTED_CELL_COUNT, (
        f"Sanity plan produced {len(plan)} cells, expected "
        f"{EXPECTED_CELL_COUNT} per §7."
    )
    print("=" * 64)
    print("CYCLE-3 SANITY PILOT (1.5B-scale fail-fast)")
    print("=" * 64)
    print(f"harness_version : {HARNESS_VERSION}")
    print(f"scale           : {SANITY_SCALE}")
    print(f"profiles        : {SANITY_PROFILES}")
    print(f"substrates      : {SANITY_SUBSTRATES}")
    print(f"seeds           : {len(SANITY_SEEDS)} (0..{SANITY_SEEDS[-1]})")
    print(f"planned cells   : {len(plan)}")
    print(f"GO rule         : H1 rejected in ≥ {GO_CELLS_REJECTED_MIN}"
          f"/6 cells at α = {GO_BONFERRONI_ALPHA}")
    print("-" * 64)
    if opts["dry_run"]:
        print("[dry-run] enumeration validated ; no dream ops.")
        return
    # Non-dry-run path intentionally unimplemented in this commit :
    # C3.7 ships the plan + milestone template, not the 1-day
    # Studio run. The user launches by extending this block.
    print(
        "[deferred] per-cell execution wiring lands in a "
        "follow-up commit ; plan enumeration validated above."
    )
    out_dir = REPO_ROOT / "docs" / "milestones"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "pilot-cycle3-sanity-1p5b.json"
    with plan_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "harness_version": HARNESS_VERSION,
                "scale": SANITY_SCALE,
                "profiles": list(SANITY_PROFILES),
                "substrates": list(SANITY_SUBSTRATES),
                "seeds": list(SANITY_SEEDS),
                "planned_cell_count": len(plan),
                "go_rule": {
                    "alpha": GO_BONFERRONI_ALPHA,
                    "cells_rejected_min": GO_CELLS_REJECTED_MIN,
                    "cells_total": len(SANITY_PROFILES) * len(
                        SANITY_SUBSTRATES
                    ),
                },
                "status": "plan-only",
                "plan": plan,
            },
            fh,
            indent=2,
        )
    print(f"Plan manifest written to {plan_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
