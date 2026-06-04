"""Pin the re-export path : the topology kernel is reachable from
both the canonical substrate module and the legacy channels
module. Refactor accident insurance.
"""
from __future__ import annotations


def test_apply_topology_op_reexport_identity() -> None:
    """The helper is the same object via either import path."""
    from kiki_oniric.dream.channels.hierarchy_change import (
        _apply_topology_op as via_channels,
    )
    from kiki_oniric.substrates.micro_kiki.lora_topology_ops import (
        _apply_topology_op as canonical,
    )
    assert via_channels is canonical
