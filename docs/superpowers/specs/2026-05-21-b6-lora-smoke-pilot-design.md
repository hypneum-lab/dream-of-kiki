# B6 LoRA smoke pilot — design

**Date** : 2026-05-21
**Status** : design approved, pending spec review
**Tracking** : informal continuation of issue #15 (B6 closed) ;
no new issue.
**Scope** : a synthetic-workload pilot script that exercises the
three LoRA profile variants end-to-end via the B5 channel apply
loop, with a milestone JSON + markdown sibling and a unit test.

---

## Context

Approach B closed at `C-v0.23.0+PARTIAL` with B6c
(`PMaxLoRAProfile`). All three LoRA profile tiers are wired and
unit-tested in isolation. No script exists that *uses* them
together — pilot_g2.py and pilot_cycle3_sanity.py predate B0 and
use cycle-3 skeleton profiles.

This pilot is the first script-level exercise of the closed
awake↔dream loop across the full DR-4 chain. It validates the
infrastructure (no new empirical claim), produces a dated
milestone JSON for the audit trail, and gets covered by a unit
test so future regressions are caught in CI.

## Problem

No script integrates PMinLoRA / PEquLoRA / PMaxLoRA in one run.
The unit tests (`test_p_*_lora.py`) verify each profile in
isolation ; the DR-4 chain test (B6c test 17) compares the three
*statically* (op-key + emitter set inclusion). Nothing actually
runs all three end-to-end with the same seed, the same
synthetic workload, and compares wall-time / FLOPs / dispatch
counts.

The pilot fills that gap.

## Approaches considered

**Formal level.** Three options:

1. **Hybride smoke + milestone JSON+md** — script in `scripts/`,
   dated immutable milestone in `docs/milestones/`, no G-gate
   pre-reg, no go/no-go criterion that gates a release.
   Bit-equality and DR-4 chain inclusion are *verifications*
   embedded in the script, not external gates. **Chosen.**
2. G-gate G7 formal milestone — would require pre-registration
   under `docs/osf-prereg-g7-*.md` and a hypothesis with
   reject/accept criteria. Overkill for an infra-validation
   exercise that makes no new empirical claim. Rejected.
3. Pure smoke (no milestone) — same as B6a/B6b/B6c unit tests
   but as a script. Loses the dated audit trail. Rejected.

**Workload structure.** Three options:

1. **Identical core + per-tier extras**. All three profiles run
   2 replay + 1 downscale at the same seed (identical
   `beta_records`, identical `shrink_factor`). PEqu/PMax add
   1 restructure (reroute swap). PMax adds 1 recombine
   (delta_latents). Total : PMin=3 episodes, PEqu=4, PMax=5.
   Allows direct FLOP / wall-time comparison on the shared core.
   **Chosen.**
2. Per-tier specific — each profile gets a workload that
   exercises its full op set. More representative but not
   directly comparable. Rejected.
3. Single shared episode replayed N times — too minimalist.
   Rejected.

**VAE encoder/decoder injection for PMaxLoRA.** Three options:

1. **Local `_PilotEncoder` / `_PilotDecoder`** in the pilot
   script. Deterministic linear layers with fixed weights
   (same pattern as `_TinyEncoder`/`_TinyDecoder` in
   `tests/unit/test_recombine_latent_sample.py`). Self-
   contained. **Chosen.**
2. Import from `tests/unit/test_recombine_latent_sample.py` —
   couples `scripts/` to `tests/`. Rejected.
3. Extract to `kiki_oniric/substrates/micro_kiki/tiny_vae.py` —
   would promote test fixtures to production. Scope creep.
   Rejected.

**Test coverage of the pilot.** Three options:

1. **Dedicated `tests/unit/scripts/test_pilot_b6_lora_smoke.py`**
   that imports `main(output_dir=tmp_path)` and asserts on
   the produced JSON. **Chosen.**
2. No test — pilot is self-validating via internal asserts.
   Matches `pilot_g2.py` precedent but loses CI regression
   coverage. Rejected.
3. Light sanity-only test (JSON parseable, no semantic
   checks). Rejected — semantic checks are cheap and
   valuable.

## Design

### New script `scripts/pilot_b6_lora_smoke.py`

```python
"""B6 LoRA smoke pilot — exercise the three LoRA profile variants
end-to-end via the B5 apply_channel_outputs loop.

Builds dream/awake LoRAModel clones at seed=42 for each of the
three profile tiers (PMinLoRA / PEquLoRA / PMaxLoRA), runs an
identical core workload (2 replay + 1 downscale) plus per-tier
extras (PEqu/PMax add 1 restructure ; PMax adds 1 recombine),
calls consolidate_log() to dispatch the channel outputs onto the
awake model, then verifies bit-equality (within-machine R1) and
DR-4 chain inclusion (ops + emitter strict subset).

Writes a dated milestone JSON + markdown to docs/milestones/.

Usage:
    uv run python scripts/pilot_b6_lora_smoke.py
"""
```

Top-level constants : `_CANONICAL_SEED = 42`, `_DATE_TAG =
"2026-05-21"`, paths derived from `Path(__file__).resolve().
parents[1]` like in `pilot_g2.py`.

**`_PilotEncoder(nn.Module)`** : MLX `nn.Module` with two
deterministic linear weights for `mu` and `log_var`. Mirrors
`_TinyEncoder` from B4 tests but defined locally.

**`_PilotDecoder(nn.Module)`** : single deterministic linear
weight for the reconstruction.

**`_build_core_episodes() -> list[DreamEpisode]`** : returns 3
episodes (2 replay with fixed `beta_records`, 1 downscale with
`shrink_factor=0.5`). Same across all 3 tiers.

**`_build_pequ_extras() -> list[DreamEpisode]`** : 1 restructure
with `topo_ops=[{"op": "reroute", "swap_indices": [0, 1]}]`.

**`_build_pmax_extras() -> list[DreamEpisode]`** : 1
restructure (same as PEqu) + 1 recombine with `delta_latents=
[[0.1, 0.2, 0.3, 0.4]]`.

**`_collect_total_flops(profile) -> int`** : sum
`state.total_compute_flops` over the profile's 3-4 state fields,
falling back to `getattr(state, 'total_compute_flops',
state.last_compute_flops)` for states that don't track totals.

**`_run_tier(name, profile, episodes) -> dict`** :
```python
{
    "tier": name,
    "wall_s": float,
    "dispatch_count": int,            # consolidate_log() return
    "bit_equal": bool,
    "total_flops": int,
    "emitted_types": list[str],       # sorted list of channel-output type names
                                      # observed in profile.runtime.log BEFORE
                                      # consolidate_log() clears it
}
```

**`main(output_dir: Path | None = None) -> dict`** :
- Build PMinLoRA + PEquLoRA + PMaxLoRA at seed=42 each (own
  dream/awake `lora_clones(42)`).
- Run their workloads, collect per-tier dicts.
- Compute DR-4 chain inclusion verdict (`ops_strict_subset`,
  `emitters_strict_subset`).
- Compute global verdict (`bit_equal_all_tiers` AND
  `dr4_chain_inclusion.verdict == "PASS"`).
- Print a terminal table.
- Write JSON to `output_dir / "b6-lora-smoke-<date>.json"` and
  markdown sibling.
- `output_dir` defaults to `REPO_ROOT / "docs" / "milestones"`.
- Returns the results dict (also written to disk).

### Milestone JSON format

```json
{
  "milestone": "b6-lora-smoke-2026-05-21",
  "date": "2026-05-21",
  "framework_version": "C-v0.23.0+PARTIAL",
  "package_version": "0.21.0",
  "trigger_commit": "<short HEAD>",
  "chip_family": "<sysctl machdep.cpu.brand_string slugified>",
  "seed": 42,
  "tiers": {
    "PMinLoRA": {
      "wall_s": 0.012,
      "dispatch_count": 3,
      "bit_equal": true,
      "total_flops": 4096,
      "emitted_types": ["WeightUpdate"]
    },
    "PEquLoRA": {
      "wall_s": 0.018,
      "dispatch_count": 4,
      "bit_equal": true,
      "total_flops": 4128,
      "emitted_types": ["TopologyDiff", "WeightUpdate"]
    },
    "PMaxLoRA": {
      "wall_s": 0.041,
      "dispatch_count": 5,
      "bit_equal": true,
      "total_flops": 5120,
      "emitted_types": ["LatentSample", "TopologyDiff", "WeightUpdate"]
    }
  },
  "dr4_chain_inclusion": {
    "ops_strict_subset": true,
    "emitters_strict_subset": true,
    "verdict": "PASS"
  },
  "bit_equal_all_tiers": true,
  "verdict": "PASS"
}
```

`chip_family` reuses `_chip_family()` from
`tests/reproducibility/_r1_helpers.py` so the milestone
captures the hardware on which it was run (relevant given the
2026-05-20 cross-machine R1 finding).

### Markdown sibling

`docs/milestones/b6-lora-smoke-2026-05-21.md` is a short prose
summary with :

- Header (date, status, sibling JSON pointer, framework version).
- The same per-tier table rendered in markdown.
- DR-4 chain inclusion paragraph.
- Reference to the R1 cross-machine milestone for chip-family
  context.
- "What this does not measure" paragraph (no cross-machine, no
  EC claim, no benchmark accuracy).

### Test — `tests/unit/scripts/test_pilot_b6_lora_smoke.py`

A new test directory `tests/unit/scripts/` with `__init__.py`.

```python
def test_pilot_b6_lora_smoke_writes_milestone(tmp_path):
    from scripts.pilot_b6_lora_smoke import main
    results = main(output_dir=tmp_path)
    # JSON file exists with expected name
    json_files = list(tmp_path.glob("b6-lora-smoke-*.json"))
    assert len(json_files) == 1
    # Markdown sibling exists
    md_files = list(tmp_path.glob("b6-lora-smoke-*.md"))
    assert len(md_files) == 1
    # 3 tiers present
    assert set(results["tiers"].keys()) == {"PMinLoRA", "PEquLoRA", "PMaxLoRA"}
    # Bit-equal across all tiers
    assert results["bit_equal_all_tiers"] is True
    for tier in results["tiers"].values():
        assert tier["bit_equal"] is True
    # Dispatch counts match the workload
    assert results["tiers"]["PMinLoRA"]["dispatch_count"] == 3
    assert results["tiers"]["PEquLoRA"]["dispatch_count"] == 4
    assert results["tiers"]["PMaxLoRA"]["dispatch_count"] == 5
    # Emitter strict-subset chain across tiers
    pmin_set = set(results["tiers"]["PMinLoRA"]["emitted_types"])
    pequ_set = set(results["tiers"]["PEquLoRA"]["emitted_types"])
    pmax_set = set(results["tiers"]["PMaxLoRA"]["emitted_types"])
    assert pmin_set <= pequ_set <= pmax_set
    assert pmin_set == {"WeightUpdate"}
    assert pequ_set == {"WeightUpdate", "TopologyDiff"}
    assert pmax_set == {"WeightUpdate", "TopologyDiff", "LatentSample"}
    # DR-4 chain inclusion verdict
    assert results["dr4_chain_inclusion"]["verdict"] == "PASS"
    assert results["verdict"] == "PASS"
```

One test, ~10 assertions. Fast (the pilot runs the same
workload as the unit tests in aggregate, ~1-2 s).

### Invariants pinned by the pilot

- **R1 within-machine** — bit-equality per tier (3 assertions).
- **DR-4 chain inclusion** — ops + emitter strict subset
  verified at run-time (2 boolean fields in the JSON +
  asserted in the test).
- **K1** — `total_flops` populated per tier (informational ;
  no gate).

## Scope boundary

**Pilot does** :
- New `scripts/pilot_b6_lora_smoke.py`.
- New `tests/unit/scripts/test_pilot_b6_lora_smoke.py` +
  `tests/unit/scripts/__init__.py`.
- Two new files in `docs/milestones/` after first run
  (committed as the audit trail).
- No code in `kiki_oniric/` or `harness/` touched.
- No CHANGELOG / DualVer bump — pilots aren't substrate
  changes. An "Operational" CHANGELOG entry is added under the
  `## [Unreleased]` section for the milestone-run record.

**Pilot does not** :
- Real benchmark data (synthetic only).
- Cross-machine comparison (within-machine R1 only).
- Performance SLA (wall-time is informational).
- New empirical claim (no EC bump).
- G-gate pre-registration.
- Modify the 3 LoRA profiles or any B-series channel.

## Convention compliance

- Script structure follows `pilot_g2.py` (REPO_ROOT shim,
  `if __name__ == "__main__": main()`).
- Milestone files follow `docs/milestones/` conventions
  (dated immutable, md + json sibling — cf. `g2-pilot-results
  .{md,json}` + the 2026-05-20 R1 cross-machine pair).
- Test file lives under `tests/unit/scripts/` (new
  sub-directory) with its own `__init__.py`. Pattern mirrors
  `tests/unit/profiles/`.

## DualVer

No FC bump, no EC bump. Pilot is operational tooling, not a
substrate change. CHANGELOG gets an "Operational" entry under
`## [Unreleased]` documenting the milestone run (verdict + chip
family) ; no version field changes.

## Acceptance criteria

1. `scripts/pilot_b6_lora_smoke.py` exists, runs to completion
   with `uv run python scripts/pilot_b6_lora_smoke.py`, and
   exits 0.
2. The script writes `docs/milestones/b6-lora-smoke-<date>.json`
   and the markdown sibling on completion.
3. Both files commit-clean on first run for the recorder's
   machine (additional chip families re-run later append
   sections to the markdown but the JSON stays per-run).
4. `tests/unit/scripts/test_pilot_b6_lora_smoke.py` passes with
   ~10 assertions ; full pytest suite green ; mypy + ruff
   clean.
5. `[Unreleased]` block in `CHANGELOG.md` gets an Operational
   bullet noting the first-run verdict + chip family.
6. No CHANGELOG version-bump entry, no `pyproject.toml`
   change.
