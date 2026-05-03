"""HP grid enumeration tests (Plan G4-ter Task 7)."""
from __future__ import annotations

import dataclasses

import pytest

from experiments.g4_ter_hp_sweep.hp_grid import HP_COMBOS, HPCombo


def test_hp_grid_has_10_combos() -> None:
    assert len(HP_COMBOS) == 10


def test_hp_combo_ids_are_unique_and_sequential() -> None:
    ids = [c.combo_id for c in HP_COMBOS]
    assert ids == [f"C{i}" for i in range(10)]


def test_hp_grid_includes_g4_bis_anchor() -> None:
    """C5 must be the G4-bis-aligned combo for richer-substrate sweep."""
    c5 = next(c for c in HP_COMBOS if c.combo_id == "C5")
    assert c5.downscale_factor == 0.95
    assert c5.replay_batch == 32
    assert c5.replay_n_steps == 5
    assert c5.replay_lr == 0.01


def test_hp_combo_is_frozen() -> None:
    """HPCombo dataclass must reject mutation (R1 contract)."""
    c = HP_COMBOS[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.replay_lr = 0.5  # type: ignore[misc]
