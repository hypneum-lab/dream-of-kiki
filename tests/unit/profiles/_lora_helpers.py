"""Shared test helpers for the LoRA-substrate profile tests.

Extracted at B6c (third copy trigger). Used by tests for
PMinLoRAProfile (B6a), PEquLoRAProfile (B6b), PMaxLoRAProfile
(B6c). Lives in tests/unit/profiles/ to keep the helpers
profile-local.
"""
from __future__ import annotations

import numpy as np

from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


def lora_clones(seed: int = 0) -> tuple[LoRAModel, LoRAModel]:
    """Two bit-identical LoRAModels at the same seed.

    The dream/awake split for B6a/B6b/B6c needs an awake clone
    bit-equal to the dream model at t=0 so ``consolidate_log()``
    can be verified via bit-equality (within-machine R1).
    """
    return (
        LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=seed),
        LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=seed),
    )


def assert_lora_models_equal(a: LoRAModel, b: LoRAModel) -> None:
    """Assert bit-equality of every layer's base + adapters."""
    assert len(a.layers) == len(b.layers)
    for la, lb in zip(a.layers, b.layers):
        np.testing.assert_array_equal(
            np.asarray(la.base_weight), np.asarray(lb.base_weight),
        )
        np.testing.assert_array_equal(
            np.asarray(la.lora_a), np.asarray(lb.lora_a),
        )
        np.testing.assert_array_equal(
            np.asarray(la.lora_b), np.asarray(lb.lora_b),
        )
