"""B6 LoRA smoke pilot — exercise the three LoRA profile tiers
(PMinLoRAProfile, PEquLoRAProfile, PMaxLoRAProfile) end-to-end
via the B5 ``apply_channel_outputs`` loop.

Builds dream/awake ``LoRAModel`` clones at seed=42 for each tier,
runs an identical core workload (2 replay + 1 downscale) plus
per-tier extras (PEqu adds 1 restructure ; PMax adds 1
restructure + 1 recombine), calls ``consolidate_log()`` to
dispatch the channel outputs onto the awake model, then verifies
bit-equality and DR-4 chain inclusion.

Writes a dated milestone JSON + markdown to
``docs/milestones/b6-lora-smoke-<date>.{json,md}``.

Reference :
- spec ``docs/superpowers/specs/2026-05-21-b6-lora-smoke-pilot-design.md``
- framework-C ``docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`` §3.1

Usage :
    uv run python scripts/pilot_b6_lora_smoke.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from kiki_oniric.dream.episode import (  # noqa: E402
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile  # noqa: E402
from kiki_oniric.profiles.p_max_lora import PMaxLoRAProfile  # noqa: E402
from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile  # noqa: E402

from tests.reproducibility._r1_helpers import _chip_family  # noqa: E402
from tests.unit.profiles._lora_helpers import (  # noqa: E402
    assert_lora_models_equal,
    lora_clones,
)


_CANONICAL_SEED = 42
_DATE_TAG = "2026-05-21"
_FRAMEWORK_VERSION = "C-v0.23.0+PARTIAL"
_PACKAGE_VERSION = "0.21.0"
_LATENT_DIM = 4
_INPUT_DIM = 4


class _PilotEncoder(nn.Module):  # type: ignore[misc,name-defined]
    """Deterministic linear encoder x -> (mu, log_var).

    Mirrors ``_TinyEncoder`` from
    ``tests/unit/test_recombine_latent_sample.py`` but defined
    locally to keep the pilot self-contained.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mu_w = mx.ones((_LATENT_DIM, _INPUT_DIM)) * 0.1
        self.lv_w = mx.ones((_LATENT_DIM, _INPUT_DIM)) * -0.5

    def __call__(self, x):  # type: ignore[no-untyped-def]  # noqa: ANN001
        return x @ self.mu_w.T, x @ self.lv_w.T


class _PilotDecoder(nn.Module):  # type: ignore[misc,name-defined]
    """Deterministic linear decoder z -> output."""

    def __init__(self) -> None:
        super().__init__()
        self.w = mx.ones((_INPUT_DIM, _LATENT_DIM)) * 0.2

    def __call__(self, z):  # type: ignore[no-untyped-def]  # noqa: ANN001
        return z @ self.w.T


def _short_head() -> str:
    """Return the short HEAD sha or 'unknown'."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return sha or "unknown"


def _build_core_episodes() -> list[DreamEpisode]:
    """2 replay episodes + 1 downscale episode, identical content
    across all three tiers."""
    budget = BudgetCap(
        flops=10_000_000, wall_time_s=1.0, energy_j=1.0,
    )
    replay_a = DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={
            "beta_records": [
                {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
                {"x": [0.5, 0.6, 0.7, 0.8], "y": [0.0, 1.0]},
            ],
        },
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=budget,
        episode_id="pilot-replay-a",
    )
    replay_b = DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={
            "beta_records": [
                {"x": [0.2, 0.3, 0.4, 0.5], "y": [0.0, 1.0]},
                {"x": [0.6, 0.7, 0.8, 0.9], "y": [1.0, 0.0]},
            ],
        },
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=budget,
        episode_id="pilot-replay-b",
    )
    downscale = DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"shrink_factor": 0.5},
        operation_set=(Operation.DOWNSCALE,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=budget,
        episode_id="pilot-downscale",
    )
    return [replay_a, replay_b, downscale]


def _build_pequ_extras() -> list[DreamEpisode]:
    """1 restructure episode (reroute swap_indices=[0, 1])."""
    return [
        DreamEpisode(
            trigger=EpisodeTrigger.SCHEDULED,
            input_slice={
                "topo_ops": [
                    {"op": "reroute", "swap_indices": [0, 1]},
                ],
            },
            operation_set=(Operation.RESTRUCTURE,),
            output_channels=(OutputChannel.HIERARCHY_CHG,),
            budget=BudgetCap(
                flops=10_000_000, wall_time_s=1.0, energy_j=1.0,
            ),
            episode_id="pilot-restr-reroute",
        ),
    ]


def _build_pmax_extras() -> list[DreamEpisode]:
    """1 restructure (reroute) + 1 recombine (delta_latents)."""
    return _build_pequ_extras() + [
        DreamEpisode(
            trigger=EpisodeTrigger.SCHEDULED,
            input_slice={
                "delta_latents": [[0.1, 0.2, 0.3, 0.4]],
            },
            operation_set=(Operation.RECOMBINE,),
            output_channels=(OutputChannel.LATENT_SAMPLE,),
            budget=BudgetCap(
                flops=10_000_000, wall_time_s=1.0, energy_j=1.0,
            ),
            episode_id="pilot-recombine",
        ),
    ]


def _collect_total_flops(profile: Any) -> int:
    """Sum total_compute_flops across the profile's per-op
    states, falling back to last_compute_flops for states that
    don't track totals (e.g. the skeleton RecombineOpState).

    Returns an int — informational K1 tag, no gate.
    """
    state_attrs = (
        "replay_state",
        "downscale_state",
        "restructure_state",
        "recombine_state",
    )
    total = 0
    for attr in state_attrs:
        state = getattr(profile, attr, None)
        if state is None:
            continue
        value = getattr(
            state,
            "total_compute_flops",
            getattr(state, "last_compute_flops", 0),
        )
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return total


def _emitted_types(profile: Any) -> list[str]:
    """Sorted list of distinct channel-output type names seen in
    the profile's runtime log BEFORE consolidate_log() clears it."""
    seen: set[str] = set()
    for entry in profile.runtime.log:
        for output in entry.channel_outputs:
            if output is None:
                continue
            seen.add(type(output).__name__)
    return sorted(seen)


def _active_ops(profile: Any) -> set[str]:
    """Set of operation names whose handler emitted at least one
    non-None channel output in the runtime log.

    DR-4 chain inclusion tracks the monotonic growth of the active
    (emitting) operation set across tiers, not merely the set of
    registered handlers. PEquLoRA registers a skeleton RECOMBINE
    handler that emits None — excluding it here ensures the ops
    strict-subset check reflects the *emitting* op chain.

    Called BEFORE consolidate_log() clears the log.
    """
    active: set[str] = set()
    for entry in profile.runtime.log:
        for op, output in zip(
            entry.operations_executed, entry.channel_outputs
        ):
            if output is not None:
                active.add(op.value)
    return active


def _run_tier(
    name: str,
    profile: Any,
    episodes: list[DreamEpisode],
    dream: Any,
    awake: Any,
) -> dict[str, Any]:
    """Run all ``episodes`` on ``profile``, capture metrics,
    call consolidate_log(), check bit-equality."""
    wall_start = time.monotonic()
    for ep in episodes:
        profile.runtime.execute(ep)
    emitted = _emitted_types(profile)
    emitting_ops = _active_ops(profile)
    total_flops = _collect_total_flops(profile)
    dispatch_count = profile.consolidate_log()
    wall_s = time.monotonic() - wall_start

    try:
        assert_lora_models_equal(dream, awake)
        bit_equal = True
    except AssertionError:
        bit_equal = False

    return {
        "tier": name,
        "wall_s": round(wall_s, 6),
        "dispatch_count": dispatch_count,
        "bit_equal": bit_equal,
        "total_flops": total_flops,
        "emitted_types": emitted,
        "_emitting_ops": emitting_ops,
    }


def _print_table(results: dict[str, Any]) -> None:
    """Print a small terminal table summarising per-tier metrics."""
    print("=" * 72)
    print(
        "B6 LoRA SMOKE PILOT — synthetic workload, seed="
        f"{results['seed']}, chip={results['chip_family']}"
    )
    print("=" * 72)
    header = (
        f"{'tier':10} {'wall_s':>9} {'dispatch':>9} "
        f"{'bit_eq':>7} {'flops':>10} {'emitted':<40}"
    )
    print(header)
    print("-" * 72)
    for tier_name, tier in results["tiers"].items():
        emitted = ",".join(tier["emitted_types"])
        print(
            f"{tier_name:10} {tier['wall_s']:>9.6f} "
            f"{tier['dispatch_count']:>9} {str(tier['bit_equal']):>7} "
            f"{tier['total_flops']:>10} {emitted:<40}"
        )
    print("-" * 72)
    dr4 = results["dr4_chain_inclusion"]
    print(
        f"DR-4 ops_strict_subset={dr4['ops_strict_subset']} "
        f"emitters_strict_subset={dr4['emitters_strict_subset']} "
        f"verdict={dr4['verdict']}"
    )
    print(
        f"bit_equal_all_tiers={results['bit_equal_all_tiers']} "
        f"verdict={results['verdict']}"
    )
    print("=" * 72)


def _write_markdown(
    md_path: Path, results: dict[str, Any],
) -> None:
    """Write the markdown sibling next to the JSON."""
    lines: list[str] = []
    lines.append(f"# {results['milestone']}")
    lines.append("")
    lines.append(
        f"**Date** : {results['date']}  ")
    lines.append(
        f"**Framework** : {results['framework_version']}  "
    )
    lines.append(
        f"**Package** : {results['package_version']}  "
    )
    lines.append(
        f"**Commit** : `{results['trigger_commit']}`  "
    )
    lines.append(
        f"**Chip family** : `{results['chip_family']}`  "
    )
    lines.append(
        f"**Seed** : `{results['seed']}`  "
    )
    lines.append(
        "**Sibling JSON** : "
        f"`{md_path.with_suffix('.json').name}`"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "First script-level exercise of the closed awake/dream "
        "loop across the three LoRA profile tiers, post-B6c. "
        "Synthetic workload, within-machine R1 only. No new "
        "empirical claim — infra-validation milestone."
    )
    lines.append("")
    lines.append("## Per-tier metrics")
    lines.append("")
    lines.append(
        "| tier | wall_s | dispatch | bit_equal | flops | emitted |",
    )
    lines.append("|---|---|---|---|---|---|")
    for tier_name, tier in results["tiers"].items():
        emitted = ", ".join(tier["emitted_types"])
        lines.append(
            f"| `{tier_name}` | {tier['wall_s']:.6f} | "
            f"{tier['dispatch_count']} | "
            f"{tier['bit_equal']} | {tier['total_flops']} | "
            f"{emitted} |"
        )
    lines.append("")
    lines.append("## DR-4 chain inclusion")
    lines.append("")
    dr4 = results["dr4_chain_inclusion"]
    lines.append(
        f"- `ops_strict_subset` : **{dr4['ops_strict_subset']}**"
    )
    lines.append(
        f"- `emitters_strict_subset` : "
        f"**{dr4['emitters_strict_subset']}**"
    )
    lines.append(f"- Verdict : **{dr4['verdict']}**")
    lines.append("")
    lines.append("## What this does not measure")
    lines.append("")
    lines.append(
        "- Cross-machine R1 — bit-equality is within-machine "
        "only ; for cross-machine see "
        "`docs/milestones/r1-cross-machine-m5-vs-m1-2026-05-20.{md,json}`."
    )
    lines.append(
        "- Benchmark accuracy — synthetic workload, no held-out "
        "evaluation."
    )
    lines.append(
        "- Performance SLA — `wall_s` is informational."
    )
    lines.append(
        "- No empirical claim — EC stays `+PARTIAL`."
    )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(output_dir: Path | None = None) -> dict[str, Any]:
    """Run the pilot, write the milestone files, return the results dict.

    When ``output_dir`` is ``None`` the JSON + markdown are
    written to ``docs/milestones/`` on the repo. Tests pass a
    ``tmp_path`` to capture the files in a sandbox.
    """
    if output_dir is None:
        output_dir = REPO_ROOT / "docs" / "milestones"

    # PMinLoRA
    pmin_dream, pmin_awake = lora_clones(seed=_CANONICAL_SEED)
    pmin = PMinLoRAProfile(
        dream_model=pmin_dream, awake_model=pmin_awake,
    )
    pmin_result = _run_tier(
        "PMinLoRA", pmin, _build_core_episodes(),
        pmin_dream, pmin_awake,
    )

    # PEquLoRA
    pequ_dream, pequ_awake = lora_clones(seed=_CANONICAL_SEED)
    pequ = PEquLoRAProfile(
        dream_model=pequ_dream, awake_model=pequ_awake,
    )
    pequ_result = _run_tier(
        "PEquLoRA", pequ,
        _build_core_episodes() + _build_pequ_extras(),
        pequ_dream, pequ_awake,
    )

    # PMaxLoRA — required encoder + decoder.
    pmax_dream, pmax_awake = lora_clones(seed=_CANONICAL_SEED)
    pmax = PMaxLoRAProfile(
        dream_model=pmax_dream, awake_model=pmax_awake,
        encoder=_PilotEncoder(), decoder=_PilotDecoder(),
        seed=_CANONICAL_SEED,
    )
    pmax_result = _run_tier(
        "PMaxLoRA", pmax,
        _build_core_episodes() + _build_pmax_extras(),
        pmax_dream, pmax_awake,
    )

    tiers = {
        "PMinLoRA": pmin_result,
        "PEquLoRA": pequ_result,
        "PMaxLoRA": pmax_result,
    }

    # DR-4 chain inclusion checks.
    # ops_strict_subset uses _emitting_ops (operations that emitted
    # at least one non-None channel output in the workload), not the
    # full registered-handler key set. PEquLoRA registers a skeleton
    # RECOMBINE handler that never emits, so using emitting ops
    # correctly reflects the DR-4 monotonic op-set growth.
    pmin_ops = pmin_result["_emitting_ops"]
    pequ_ops = pequ_result["_emitting_ops"]
    pmax_ops = pmax_result["_emitting_ops"]
    ops_strict_subset = (
        pmin_ops < pequ_ops < pmax_ops
    )

    pmin_em = set(pmin_result["emitted_types"])
    pequ_em = set(pequ_result["emitted_types"])
    pmax_em = set(pmax_result["emitted_types"])
    emitters_strict_subset = pmin_em < pequ_em < pmax_em

    dr4_pass = ops_strict_subset and emitters_strict_subset
    bit_equal_all = all(
        t["bit_equal"] for t in tiers.values()
    )
    overall_pass = dr4_pass and bit_equal_all

    # Strip internal-only fields before writing.
    tiers_out = {
        k: {fk: fv for fk, fv in v.items() if not fk.startswith("_")}
        for k, v in tiers.items()
    }

    results: dict[str, Any] = {
        "milestone": f"b6-lora-smoke-{_DATE_TAG}",
        "date": _DATE_TAG,
        "framework_version": _FRAMEWORK_VERSION,
        "package_version": _PACKAGE_VERSION,
        "trigger_commit": _short_head(),
        "chip_family": _chip_family(),
        "seed": _CANONICAL_SEED,
        "tiers": tiers_out,
        "dr4_chain_inclusion": {
            "ops_strict_subset": ops_strict_subset,
            "emitters_strict_subset": emitters_strict_subset,
            "verdict": "PASS" if dr4_pass else "FAIL",
        },
        "bit_equal_all_tiers": bit_equal_all,
        "verdict": "PASS" if overall_pass else "FAIL",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"b6-lora-smoke-{_DATE_TAG}.json"
    md_path = output_dir / f"b6-lora-smoke-{_DATE_TAG}.md"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    _write_markdown(md_path, results)

    _print_table(results)
    print(f"\nMilestone written to {json_path}")
    print(f"Milestone written to {md_path}")
    return results


if __name__ == "__main__":
    main()
