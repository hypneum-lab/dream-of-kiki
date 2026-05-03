"""Curated HP combo grid for the G4-ter pilot.

10 combos along the qualitative gradient hypothesised most likely
to flip the sign of g_h1. C5 is the G4-bis production calibration
anchor *except* for replay_n_steps (1 -> 5).

Reference :
    docs/osf-prereg-g4-ter-pilot.md sec 3
    docs/superpowers/plans/2026-05-03-g4-ter-hp-sweep-richer-substrate.md
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HPCombo:
    """One curated point in the HP grid (R1 stable id)."""

    combo_id: str
    downscale_factor: float
    replay_batch: int
    replay_n_steps: int
    replay_lr: float


HP_COMBOS: tuple[HPCombo, ...] = (
    HPCombo("C0", 0.85, 16, 1, 0.001),
    HPCombo("C1", 0.85, 32, 5, 0.001),
    HPCombo("C2", 0.90, 32, 1, 0.001),
    HPCombo("C3", 0.90, 32, 5, 0.01),
    HPCombo("C4", 0.95, 32, 1, 0.001),
    HPCombo("C5", 0.95, 32, 5, 0.01),
    HPCombo("C6", 0.95, 64, 10, 0.001),
    HPCombo("C7", 0.99, 16, 1, 0.001),
    HPCombo("C8", 0.99, 32, 5, 0.01),
    HPCombo("C9", 0.99, 64, 10, 0.05),
)


def representative_combo() -> HPCombo:
    """Return the C5 anchor used for the richer-substrate sweep."""
    return next(c for c in HP_COMBOS if c.combo_id == "C5")
