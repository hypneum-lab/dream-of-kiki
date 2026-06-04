# B6 LoRA smoke pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a synthetic-workload pilot script that exercises the three LoRA profile tiers (PMinLoRAProfile, PEquLoRAProfile, PMaxLoRAProfile) end-to-end through the B5 `apply_channel_outputs` loop, verifies within-machine bit-equality + DR-4 chain inclusion at runtime, and writes a dated milestone JSON + markdown for the audit trail.

**Architecture:** `scripts/pilot_b6_lora_smoke.py` builds three dream/awake `LoRAModel` clone pairs at seed 42, runs an identical core workload (2 replay + 1 downscale) plus per-tier extras (PEqu adds 1 restructure ; PMax adds 1 restructure + 1 recombine VAE), then calls each profile's `consolidate_log()`, collects wall-time / FLOPs / dispatch counts / emitted-type sets per tier, and writes the result to `docs/milestones/b6-lora-smoke-2026-05-21.{json,md}`. A unit test invokes `main(output_dir=tmp_path)` and asserts ~10 properties of the produced JSON (3 tiers present, bit-equal everywhere, dispatch counts 3/4/5, emitter strict-subset chain, DR-4 verdict PASS).

**Tech Stack:** Python 3.12+, `uv`, MLX (`mlx.core`, `mlx.nn`), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-05-21-b6-lora-smoke-pilot-design.md`

**No DualVer bump :** pilot is operational tooling, not a substrate change. CHANGELOG gets an Operational bullet under `## [Unreleased]` after the first run records its verdict.

**Chip-family capture :** the milestone JSON records the chip family the pilot ran on (Apple M5 / M3 Ultra / M1 Max / …) via `_chip_family()` from `tests/reproducibility/_r1_helpers.py`. This is relevant given the 2026-05-20 cross-machine R1 finding (`mx.random.normal` divergence on M1 Max with raw keys) — replay and recombine *do* touch that path, so the milestone JSON will differ bit-for-bit across machines on those fields and the chip_family field documents which one is captured.

---

## File Structure

- **Create** `scripts/pilot_b6_lora_smoke.py` — pilot entry point + helpers.
- **Create** `tests/unit/scripts/__init__.py` — new test sub-package.
- **Create** `tests/unit/scripts/test_pilot_b6_lora_smoke.py` — 1 test, ~10 assertions.
- **Run** the pilot once on `main`, then **commit** the produced `docs/milestones/b6-lora-smoke-2026-05-21.json` + `.md` as the audit trail.
- **Modify** `CHANGELOG.md` — append one Operational bullet under `[Unreleased]` documenting the first-run verdict + chip family.

No code under `kiki_oniric/` or `harness/` is touched. No `pyproject.toml` change. No DualVer entry.

---

## Task 1: Pilot script + unit test

**Files:**
- Create: `scripts/pilot_b6_lora_smoke.py`
- Create: `tests/unit/scripts/__init__.py`
- Create: `tests/unit/scripts/test_pilot_b6_lora_smoke.py`

- [ ] **Step 1: Create the test sub-package marker**

Create `tests/unit/scripts/__init__.py` as an empty file (mirrors `tests/unit/profiles/__init__.py` pattern).

```python
```

(empty file)

- [ ] **Step 2: Write the failing test file**

Create `tests/unit/scripts/test_pilot_b6_lora_smoke.py`:

```python
"""Unit test for the B6 LoRA smoke pilot.

Invokes ``main(output_dir=tmp_path)`` and asserts the produced
milestone JSON has the expected structure + verdicts. The pilot
itself runs the same workload as the unit tests in aggregate
(~1-2 s), so this is fast.
"""
from __future__ import annotations

import json
from pathlib import Path


def test_pilot_b6_lora_smoke_writes_milestone(tmp_path: Path) -> None:
    from scripts.pilot_b6_lora_smoke import main

    results = main(output_dir=tmp_path)

    json_files = list(tmp_path.glob("b6-lora-smoke-*.json"))
    md_files = list(tmp_path.glob("b6-lora-smoke-*.md"))
    assert len(json_files) == 1
    assert len(md_files) == 1

    # Round-trip the JSON to ensure it parses and matches the
    # returned dict.
    with json_files[0].open(encoding="utf-8") as fh:
        from_disk = json.load(fh)
    assert from_disk == results

    # 3 tiers present.
    assert set(results["tiers"].keys()) == {
        "PMinLoRA", "PEquLoRA", "PMaxLoRA",
    }

    # Bit-equal across all tiers.
    assert results["bit_equal_all_tiers"] is True
    for tier in results["tiers"].values():
        assert tier["bit_equal"] is True

    # Dispatch counts match the workload (core 3 + per-tier extras).
    assert results["tiers"]["PMinLoRA"]["dispatch_count"] == 3
    assert results["tiers"]["PEquLoRA"]["dispatch_count"] == 4
    assert results["tiers"]["PMaxLoRA"]["dispatch_count"] == 5

    # Emitter strict-subset chain across tiers.
    pmin_set = set(results["tiers"]["PMinLoRA"]["emitted_types"])
    pequ_set = set(results["tiers"]["PEquLoRA"]["emitted_types"])
    pmax_set = set(results["tiers"]["PMaxLoRA"]["emitted_types"])
    assert pmin_set == {"WeightUpdate"}
    assert pequ_set == {"WeightUpdate", "TopologyDiff"}
    assert pmax_set == {
        "WeightUpdate", "TopologyDiff", "LatentSample",
    }
    assert pmin_set < pequ_set < pmax_set  # strict subset chain

    # DR-4 chain inclusion verdict at the milestone-JSON level.
    assert results["dr4_chain_inclusion"]["ops_strict_subset"] is True
    assert results["dr4_chain_inclusion"]["emitters_strict_subset"] is True
    assert results["dr4_chain_inclusion"]["verdict"] == "PASS"

    # Top-level verdict.
    assert results["verdict"] == "PASS"
```

- [ ] **Step 3: Run to verify the test fails**

Run: `uv run pytest tests/unit/scripts/test_pilot_b6_lora_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.pilot_b6_lora_smoke'`.

- [ ] **Step 4: Create the pilot script**

Create `scripts/pilot_b6_lora_smoke.py`:

```python
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

    def __call__(self, x):  # noqa: ANN001 — mlx.nn dynamic
        return x @ self.mu_w.T, x @ self.lv_w.T


class _PilotDecoder(nn.Module):  # type: ignore[misc,name-defined]
    """Deterministic linear decoder z -> output."""

    def __init__(self) -> None:
        super().__init__()
        self.w = mx.ones((_INPUT_DIM, _LATENT_DIM)) * 0.2

    def __call__(self, z):  # noqa: ANN001
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
    pmin_ops = set(pmin.runtime._handlers.keys())
    pequ_ops = set(pequ.runtime._handlers.keys())
    pmax_ops = set(pmax.runtime._handlers.keys())
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

    results: dict[str, Any] = {
        "milestone": f"b6-lora-smoke-{_DATE_TAG}",
        "date": _DATE_TAG,
        "framework_version": _FRAMEWORK_VERSION,
        "package_version": _PACKAGE_VERSION,
        "trigger_commit": _short_head(),
        "chip_family": _chip_family(),
        "seed": _CANONICAL_SEED,
        "tiers": tiers,
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
```

- [ ] **Step 5: Run to verify the test passes**

Run: `uv run pytest tests/unit/scripts/test_pilot_b6_lora_smoke.py -v`
Expected: PASS — 1 test.

- [ ] **Step 6: Full sanity (suite + mypy + ruff)**

Run: `uv run pytest -q` — full suite passes (the new pilot test adds 1 ; expect 866 passed or however many are on the current branch tip).
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check scripts/pilot_b6_lora_smoke.py tests/unit/scripts/` — clean.

- [ ] **Step 7: Commit**

```bash
git add scripts/pilot_b6_lora_smoke.py tests/unit/scripts/__init__.py tests/unit/scripts/test_pilot_b6_lora_smoke.py
git commit -m "$(cat <<'EOF'
feat(pilot): B6 LoRA smoke pilot + milestone writer

First script-level exercise of the closed awake/dream loop
across the three LoRA profile tiers (PMinLoRA / PEquLoRA /
PMaxLoRA). Synthetic workload at seed=42 — identical core (2
replay + 1 downscale) plus per-tier extras (PEqu adds 1
restructure ; PMax adds 1 restructure + 1 recombine). Calls
consolidate_log() per tier, captures wall_s + total_flops +
dispatch_count + emitted_types + within-machine bit_equal.

Verifies DR-4 chain inclusion at runtime (ops strict subset +
emitter strict subset across PMin/PEqu/PMax). Writes a dated
milestone JSON + markdown to docs/milestones/.

No DualVer bump — pilot is operational tooling, not a
substrate change. Records the chip family via the R1
_chip_family helper so a future cross-machine run produces a
distinct artifact (per the 2026-05-20 cross-machine probe).

Test invokes main(output_dir=tmp_path) with ~10 assertions on
the JSON structure and per-tier verdicts.
EOF
)"
```

---

## Task 2: Run the pilot on `main` and commit the milestone

**Files:**
- Create: `docs/milestones/b6-lora-smoke-2026-05-21.json`
- Create: `docs/milestones/b6-lora-smoke-2026-05-21.md`

- [ ] **Step 1: Run the pilot from the repo root**

Run: `uv run python scripts/pilot_b6_lora_smoke.py`
Expected: terminal table prints, `docs/milestones/b6-lora-smoke-2026-05-21.json` and `b6-lora-smoke-2026-05-21.md` are created.

- [ ] **Step 2: Inspect the JSON for correctness**

Run: `cat docs/milestones/b6-lora-smoke-2026-05-21.json | jq '.verdict, .bit_equal_all_tiers, .tiers | keys'`
Expected: `"PASS"`, `true`, `["PEquLoRA", "PMaxLoRA", "PMinLoRA"]`.

If the verdict is `"FAIL"`, **stop and investigate** — do not commit a failing milestone. The most likely cause is the M1 Max `mx.random.normal` divergence (see `docs/proofs/r1-cross-machine.md`) affecting `bit_equal` on a tier that touches the divergent kernel ; if so, document in the markdown sibling and adjust the spec rather than masking the result.

- [ ] **Step 3: Commit the milestone**

```bash
git add docs/milestones/b6-lora-smoke-2026-05-21.json docs/milestones/b6-lora-smoke-2026-05-21.md
git commit -m "$(cat <<'EOF'
milestone(pilot): B6 LoRA smoke pilot first-run audit

First run of scripts/pilot_b6_lora_smoke.py on main. Captures
the chip-family + commit + per-tier metrics for the audit
trail. Milestone files are dated immutable per
docs/CLAUDE.md milestones convention.
EOF
)"
```

---

## Task 3: CHANGELOG Operational bullet

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add an Operational bullet under `## [Unreleased]`**

Open `CHANGELOG.md`. Locate the `## [Unreleased]` section. Append the new bullet under any existing `### Operational` sub-block (or create one if absent — match the formatting of the prior Operational entries on 2026-05-04 and 2026-05-12). The bullet text :

```markdown
### Operational (B6 LoRA smoke pilot first-run, 2026-05-21)

- First run of `scripts/pilot_b6_lora_smoke.py` on `main` (post-B6c).
  Synthetic workload at seed=42 across the three LoRA profile
  tiers exercising the closed awake↔dream loop via
  `apply_channel_outputs`. Verdict captured in
  `docs/milestones/b6-lora-smoke-2026-05-21.{json,md}` ; chip
  family + commit recorded for the audit trail. Within-machine
  R1 bit-equality verified ; DR-4 chain inclusion (ops +
  emitters strict-subset across PMin / PEqu / PMax) verified
  at runtime. No FC / EC bump — pilot is infra-validation, no
  empirical claim.
```

If the file structure of `## [Unreleased]` doesn't have prior `### Operational` headers, add this block immediately under `## [Unreleased]` (search the file's top for `## [Unreleased]` and place the new sub-block on the next non-empty line).

- [ ] **Step 2: Verify**

Run: `git diff CHANGELOG.md` — confirm the bullet appears in the right place.
Run: `uv run pytest -q` — full suite still passes (CHANGELOG change, no code touched).

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs: log B6 LoRA smoke pilot first-run operational

Adds an Operational bullet under [Unreleased] in CHANGELOG.md
documenting the first run of scripts/pilot_b6_lora_smoke.py
on main. References the dated milestone JSON+md as the audit
trail.

No FC / EC bump — pilot is infra-validation, no empirical
claim and no substrate change.
EOF
)"
```

---

## Task 4: Final verification + push

**Files:** none — verification only.

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -q`
Expected: all pass, 0 failures, coverage gate met.

- [ ] **Step 2: Type check**

Run: `uv run mypy harness tests`
Expected: `Success: no issues found`.

- [ ] **Step 3: Lint**

Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 4: Confirm working tree is clean**

Run: `git status --short`
Expected: empty (the per-family golden hashes file may show modified — restore it).

- [ ] **Step 5: Push**

```bash
git push origin main
```

- [ ] **Step 6: Update issue #15 (optional)**

Comment on issue #15 (closed but still receives comments) if it adds value : "B6 LoRA smoke pilot shipped — `scripts/pilot_b6_lora_smoke.py` exercises the three LoRA profile tiers end-to-end with within-machine bit-equality + runtime DR-4 chain-inclusion verification. Audit trail in `docs/milestones/b6-lora-smoke-2026-05-21.{json,md}`. No DualVer bump."

---

## Self-Review

- **Spec coverage:**
  - `scripts/pilot_b6_lora_smoke.py` with `_PilotEncoder` / `_PilotDecoder` (Task 1 Step 4) ✓
  - `_build_core_episodes` (2 replay + 1 downscale identical across tiers) (Task 1 Step 4) ✓
  - `_build_pequ_extras` (1 restructure reroute) (Task 1 Step 4) ✓
  - `_build_pmax_extras` (restructure + recombine) (Task 1 Step 4) ✓
  - `_collect_total_flops(profile)` with `getattr` fallback (Task 1 Step 4) ✓
  - `_run_tier(name, profile, episodes, dream, awake)` collecting `wall_s` / `dispatch_count` / `bit_equal` / `total_flops` / `emitted_types` (Task 1 Step 4) ✓
  - `main(output_dir: Path | None = None)` builds 3 profiles + runs + DR-4 chain inclusion + writes JSON+md + prints table + returns dict (Task 1 Step 4) ✓
  - Uses `lora_clones` + `assert_lora_models_equal` from `_lora_helpers` (Task 1 Step 4 imports) ✓
  - Imports `_chip_family` from `tests/reproducibility/_r1_helpers` (Task 1 Step 4 imports) ✓
  - `tests/unit/scripts/__init__.py` + `tests/unit/scripts/test_pilot_b6_lora_smoke.py` with 1 test, ~10 assertions (Task 1 Steps 1-2) ✓
  - Pilot run on `main` + commit JSON + md as audit trail (Task 2) ✓
  - CHANGELOG `[Unreleased]` Operational bullet (Task 3) ✓
  - No DualVer / version bump (Task 4 verifies via clean `git status`) ✓
  - Final verification + push (Task 4) ✓
- **Placeholder scan:** no TBD/TODO; every code step has complete code.
- **Type consistency:**
  - `main(output_dir: Path | None = None) -> dict[str, Any]` — test invokes with `tmp_path` and asserts on the returned dict + JSON on disk.
  - `_PilotEncoder`, `_PilotDecoder` — same constructor / forward shape as `_TinyEncoder` / `_TinyDecoder` in B4.
  - `lora_clones(seed)` returns `tuple[LoRAModel, LoRAModel]` — same as B6c.
  - `assert_lora_models_equal(a, b)` — same as B6c.
  - `_chip_family()` returns `str` — slugified per `_r1_helpers.py`.
  - `PMinLoRAProfile(dream_model=, awake_model=)`, `PEquLoRAProfile(dream_model=, awake_model=)`, `PMaxLoRAProfile(dream_model=, awake_model=, encoder=, decoder=, seed=)` — all match the B6a/b/c profile kwargs.
  - `profile.consolidate_log() -> int` — same name across all three profiles.
  - `profile.runtime._handlers.keys()` — same access pattern used by B6c test 17.
  - JSON keys (`tiers`, `dr4_chain_inclusion.{ops_strict_subset, emitters_strict_subset, verdict}`, `bit_equal_all_tiers`, `verdict`) — same names in the implementation and the test assertions.
- **Inter-task ordering:** Task 1 (script + test) lands first ; Task 2 (run + commit milestone) depends on Task 1 ; Task 3 (CHANGELOG bullet) depends on Task 2 (it references the milestone JSON) ; Task 4 (verif + push) depends on all prior. Order 1→2→3→4 is the dependency order.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-21-b6-lora-smoke-pilot.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, inline review by me between tasks (matches the B6a / B6b / B6c pattern).

**2. Inline Execution** — execute tasks in this session.
