"""M5 runner — resume skips cells that already have an output hash."""

from __future__ import annotations

from pathlib import Path

from scripts.ablation_cycle3_diffusion import _parse_cli, _resume_skip
from harness.storage.run_registry import RunRegistry


def test_resume_skip_true_when_hash_present(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "r.sqlite")
    run_id = registry.register("C-v0.14.0+PARTIAL", "p_min", 0, "abc123")
    registry.register_output_hash(run_id, "deadbeef" * 8)
    assert _resume_skip(registry, run_id) is True


def test_resume_skip_false_when_hash_absent(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "r.sqlite")
    run_id = registry.register("C-v0.14.0+PARTIAL", "p_min", 1, "abc123")
    assert _resume_skip(registry, run_id) is False


def test_cli_num_seeds_rejects_over_30() -> None:
    try:
        _parse_cli(["--num-seeds", "60"])
    except SystemExit:
        return
    raise AssertionError("--num-seeds 60 should raise SystemExit")
