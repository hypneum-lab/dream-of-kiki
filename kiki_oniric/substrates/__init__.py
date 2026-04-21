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
]
