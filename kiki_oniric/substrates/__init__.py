"""Substrate abstraction layer.

Cycle 1 ships a single substrate (MLX kiki-oniric).
Cycle 2 adds E-SNN thalamocortical as a second substrate to
empirically validate the DR-3 substrate-agnosticism claim.

Each substrate module re-exports or wraps the substrate-specific
implementations of the 8 typed Protocols defined in
`kiki_oniric.core.primitives`.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md
§6.2 (DR-3 Conformance Criterion)
"""
from kiki_oniric.substrates.mlx_kiki_oniric import (
    MLX_SUBSTRATE_NAME,
    MLX_SUBSTRATE_VERSION,
    mlx_substrate_components,
)
from kiki_oniric.substrates.esnn_thalamocortical import (
    ESNN_SUBSTRATE_NAME,
    ESNN_SUBSTRATE_VERSION,
    EsnnBackend,
    EsnnSubstrate,
    esnn_substrate_components,
)
from kiki_oniric.substrates.micro_kiki import (
    MICRO_KIKI_SUBSTRATE_NAME,
    MICRO_KIKI_SUBSTRATE_VERSION,
    MicroKikiSubstrate,
    micro_kiki_substrate_components,
)
from kiki_oniric.substrates.wake_sleep_cl_baseline import (
    WAKE_SLEEP_BASELINE_NAME,
    WAKE_SLEEP_BASELINE_VERSION,
    WakeSleepCLBaseline,
    wake_sleep_substrate_components,
)

# Wave 3b M2: ``mlx_latent_diffusion`` is intentionally NOT re-exported
# here. Its module-level ``import mlx.core as mx`` would break the
# libmlx-less Linux CI runner by failing this package's import chain
# (test_esnn_operations.py etc. transitively import the substrates
# package). Access the substrate via ``build_substrate_adapter
# ("mlx_latent_diffusion")`` in factory.py, which lazy-imports it.

__all__ = [
    "MLX_SUBSTRATE_NAME",
    "MLX_SUBSTRATE_VERSION",
    "mlx_substrate_components",
    "ESNN_SUBSTRATE_NAME",
    "ESNN_SUBSTRATE_VERSION",
    "EsnnBackend",
    "EsnnSubstrate",
    "esnn_substrate_components",
    "MICRO_KIKI_SUBSTRATE_NAME",
    "MICRO_KIKI_SUBSTRATE_VERSION",
    "MicroKikiSubstrate",
    "micro_kiki_substrate_components",
    "WAKE_SLEEP_BASELINE_NAME",
    "WAKE_SLEEP_BASELINE_VERSION",
    "WakeSleepCLBaseline",
    "wake_sleep_substrate_components",
]
