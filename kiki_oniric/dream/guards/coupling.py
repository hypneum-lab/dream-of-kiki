"""K-coupling guard — assert measured SO x fast-spindle coupling lies in
the empirical 95 % CI from eLife 2025 Bayesian meta-analysis.

Reference: docs/invariants/registry.md (K2), framework-C spec §5,
BibTeX `elife2025bayesian`.

Mirrors the S2/S3/S4 guard convention: a single ``check_*`` callable
plus a dedicated exception type, both re-exported here.
"""
from __future__ import annotations

import math


class CouplingGuardError(RuntimeError):
    """Raised when measured coupling falls outside the K2 CI."""


def check_coupling_in_window(
    value: float, *, ci_low: float, ci_high: float
) -> None:
    """Validate ``value`` against the empirical CI.

    Parameters
    ----------
    value : float
        Measured coupling strength (e.g. mean vector length).
    ci_low, ci_high : float
        Bounds of the 95 % CI. ``ci_low <= ci_high`` required.

    Raises
    ------
    CouplingGuardError
        If ``value`` is NaN or falls outside ``[ci_low, ci_high]``.
    ValueError
        If ``ci_low > ci_high`` (programmer error).
    """
    if ci_low > ci_high:
        raise ValueError(
            f"ci_low ({ci_low}) must be <= ci_high ({ci_high})"
        )
    if math.isnan(value):
        raise CouplingGuardError("K2: coupling value is NaN")
    if value < ci_low:
        raise CouplingGuardError(
            f"K2: coupling {value:.4f} below CI low {ci_low:.4f}"
        )
    if value > ci_high:
        raise CouplingGuardError(
            f"K2: coupling {value:.4f} above CI high {ci_high:.4f}"
        )
