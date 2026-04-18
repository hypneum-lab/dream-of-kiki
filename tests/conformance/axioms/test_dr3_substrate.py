"""DR-3 Substrate-agnosticism — Conformance Criterion condition (1) signature typing.

Verifies that the 8 primitives (4 awake→dream + 4 dream→awake) are
declared as typed Protocols, satisfying the first condition of the
DR-3 Conformance Criterion.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
"""
from typing import Protocol, runtime_checkable

from kiki_oniric.core.primitives import (
    AlphaStreamProtocol,
    AttentionPriorChannel,
    BetaBufferProtocol,
    DeltaLatentsProtocol,
    GammaSnapshotProtocol,
    HierarchyChangeChannel,
    LatentSampleChannel,
    WeightDeltaChannel,
)


def _is_protocol(cls: type) -> bool:
    return Protocol in getattr(cls, "__mro__", ())


def test_alpha_protocol_is_runtime_checkable() -> None:
    assert runtime_checkable(AlphaStreamProtocol) is AlphaStreamProtocol


def test_beta_protocol_is_runtime_checkable() -> None:
    assert runtime_checkable(BetaBufferProtocol) is BetaBufferProtocol


def test_all_8_protocols_declared() -> None:
    protocols = [
        AlphaStreamProtocol,
        BetaBufferProtocol,
        GammaSnapshotProtocol,
        DeltaLatentsProtocol,
        WeightDeltaChannel,
        LatentSampleChannel,
        HierarchyChangeChannel,
        AttentionPriorChannel,
    ]
    assert len(protocols) == 8
    for p in protocols:
        assert _is_protocol(p), f"{p.__name__} must be a typing.Protocol"
