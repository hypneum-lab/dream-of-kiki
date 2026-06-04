"""Statistical tooling for the dream-of-kiki harness.

Currently exposes the powered TOST equivalence machinery (E1 of the
experiment backlog). This is *outillage* (FC-axis tooling) — it makes
no empirical claim on its own and never registers a run. The empirical
claim lands only when a registered pilot feeds real predictor outputs
into :func:`tost_equivalence_g`.
"""
from __future__ import annotations

from harness.stats.tost import (
    SESOI_HEDGES_G,
    TOST_ALPHA_BONFERRONI,
    PowerAnalysisResult,
    TostResult,
    paired_hedges_g,
    required_n_for_power,
    tost_equivalence_g,
)

__all__ = [
    "SESOI_HEDGES_G",
    "TOST_ALPHA_BONFERRONI",
    "PowerAnalysisResult",
    "TostResult",
    "paired_hedges_g",
    "required_n_for_power",
    "tost_equivalence_g",
]
