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
