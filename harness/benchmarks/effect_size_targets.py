"""Empirical effect-size targets for the C framework.

Encodes published meta-analytic Hedges' g and 95 % CIs from
[@hu2020tmr] (TMR, k=91, N=2004) and [@javadi2024sleeprestriction]
(sleep-restriction, k=39, N=1234). Treated as immutable empirical
anchors that future pilots compare observed effect sizes against.

These targets do **not** themselves trigger an EC bump : they are
external published numbers, not registered run outputs. They are
the floor against which `P_equ` consolidation gains and `P_min`
decrement magnitudes are evaluated for plausibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProfileLabel = Literal["P_min", "P_equ", "P_max"]


@dataclass(frozen=True)
class EffectSizeTarget:
    """A single published Hedges' g target with 95% CI.

    Attributes :
        name : human-readable label (e.g. ``"hu2020_nrem2"``).
        hedges_g : point estimate of standardized mean difference.
        ci_low, ci_high : 95% confidence interval bounds (CI_low <=
            hedges_g <= CI_high enforced at construction).
        sample_size_n : aggregated participant count across studies.
        k_studies : number of independent studies in the meta.
        source_bibtex_key : key in
            ``docs/papers/paper1/references.bib`` (validated by the
            test suite, not at runtime).
        profile_target : which dream-of-kiki profile this target
            floors — ``"P_equ"`` for consolidation gains,
            ``"P_min"`` for sleep-restriction-style decrements.
        stratum : optional sleep-stage / scope label
            (``"NREM2"``, ``"SWS"``, ``None`` for overall).
    """

    name: str
    hedges_g: float
    ci_low: float
    ci_high: float
    sample_size_n: int
    k_studies: int
    source_bibtex_key: str
    profile_target: ProfileLabel
    stratum: str | None

    def __post_init__(self) -> None:
        if self.ci_low > self.ci_high:
            raise ValueError(
                f"ci_low ({self.ci_low}) must be <= ci_high "
                f"({self.ci_high}) for target {self.name!r}"
            )
        if not (self.ci_low <= self.hedges_g <= self.ci_high):
            raise ValueError(
                f"hedges_g ({self.hedges_g}) must lie within ci "
                f"[{self.ci_low}, {self.ci_high}] for target "
                f"{self.name!r}"
            )


# ----------------------------------------------------------------------
# Hu et al. 2020 TMR meta-analysis (Psychological Bulletin)
# k = 91 reports, 212 effect sizes, N = 2004 participants
# Source : docs/papers/paper1/references.bib :: hu2020tmr
# ----------------------------------------------------------------------

HU_2020_OVERALL: EffectSizeTarget = EffectSizeTarget(
    name="hu2020_overall",
    hedges_g=0.29,
    ci_low=0.21,
    ci_high=0.38,
    sample_size_n=2004,
    k_studies=91,
    source_bibtex_key="hu2020tmr",
    profile_target="P_equ",
    stratum=None,
)

HU_2020_NREM2: EffectSizeTarget = EffectSizeTarget(
    name="hu2020_nrem2",
    hedges_g=0.32,
    ci_low=0.04,
    ci_high=0.60,
    sample_size_n=2004,
    k_studies=91,
    source_bibtex_key="hu2020tmr",
    profile_target="P_equ",
    stratum="NREM2",
)

HU_2020_SWS: EffectSizeTarget = EffectSizeTarget(
    name="hu2020_sws",
    hedges_g=0.27,
    ci_low=0.20,
    ci_high=0.35,
    sample_size_n=2004,
    k_studies=91,
    source_bibtex_key="hu2020tmr",
    profile_target="P_equ",
    stratum="SWS",
)
