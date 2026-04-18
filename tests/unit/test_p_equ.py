"""Skeleton P_equ tests deprecated S11.2 — see test_p_equ_wiring.py.

The S8.3 skeleton (status="skeleton", unimplemented_ops list) was
replaced in S11.2 with the fully wired profile (4 ops + 4 states +
runtime). Skeleton-era assertions are intentionally removed; the
wiring tests cover the new contract.
"""
from __future__ import annotations

from kiki_oniric.profiles.p_equ import PEquProfile


def test_p_equ_can_be_instantiated() -> None:
    """Smoke test: profile constructs without error."""
    profile = PEquProfile()
    assert profile is not None
