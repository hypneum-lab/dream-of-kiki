"""kiki_oniric — substrate-agnostic dream-consolidation framework C.

Public top-level API : `consolidate()` is the single facade
downstream consumers (e.g. `nerve-wml`'s `bridge/dream_bridge.py`)
should import. See `kiki_oniric.consolidate` for the contract.
"""
from __future__ import annotations

from kiki_oniric.consolidate import Profile, consolidate

__all__ = ["consolidate", "Profile"]
