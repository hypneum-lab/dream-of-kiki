"""Sharon 2025 SO-trough biomarker calibration tests for P_min / P_equ / P_max.

Reference: Sharon et al., Alzheimer's & Dementia 2025 (sharon2025alzdementia
in docs/papers/paper1/references.bib). hd-EEG, N=55 (21 healthy older /
28 aMCI / 6 AD). Cognitive performance decreases monotonically with
slow-wave trough amplitude and frontocentral synchronization.

These tests verify the qualitative calibration of the
``so_trough_amplitude_factor`` field on each profile: an informed
placeholder whose final empirical value lands at G2 / G4 pilots
(cf. ``scripts/pilot_g2.py``).
"""
from __future__ import annotations

import math

from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_max import PMaxProfile
from kiki_oniric.profiles.p_min import PMinProfile
from kiki_oniric.profiles.so_calibration import (
    SHARON_2025_AD_FLOOR,
    SHARON_2025_AMCI_MIDPOINT,
    SHARON_2025_HEALTHY_OLDER_ANCHOR,
    compute_so_amplitude_proxy,
)


def test_p_min_so_trough_amplitude_factor_default_value() -> None:
    """P_min default factor = 0.45 (aMCI midpoint, Sharon 2025)."""
    profile = PMinProfile()
    assert math.isclose(profile.so_trough_amplitude_factor, 0.45)


def test_p_equ_so_trough_amplitude_factor_default_value() -> None:
    """P_equ default factor = 1.0 (healthy-older anchor, Sharon 2025)."""
    profile = PEquProfile()
    assert math.isclose(profile.so_trough_amplitude_factor, 1.0)


def test_p_max_so_trough_amplitude_factor_default_value() -> None:
    """P_max default factor = 1.0 (intact substrate, healthy-young anchor)."""
    profile = PMaxProfile()
    assert math.isclose(profile.so_trough_amplitude_factor, 1.0)


def test_compute_so_amplitude_proxy_reads_p_min() -> None:
    """compute_so_amplitude_proxy returns the field value on P_min."""
    profile = PMinProfile()
    assert math.isclose(compute_so_amplitude_proxy(profile), 0.45)


def test_compute_so_amplitude_proxy_reads_p_equ_and_p_max() -> None:
    """compute_so_amplitude_proxy returns 1.0 on healthy anchors."""
    assert math.isclose(compute_so_amplitude_proxy(PEquProfile()), 1.0)
    assert math.isclose(compute_so_amplitude_proxy(PMaxProfile()), 1.0)


def test_compute_so_amplitude_proxy_rejects_non_profile() -> None:
    """compute_so_amplitude_proxy raises TypeError on missing attribute."""
    import pytest

    class _NotAProfile:
        pass

    with pytest.raises(TypeError, match="so_trough_amplitude_factor"):
        # Intentionally wrong type to exercise the TypeError guard.
        compute_so_amplitude_proxy(_NotAProfile())  # type: ignore[arg-type]


def test_sharon_2025_anchor_constants() -> None:
    """Module-level constants pin the Sharon 2025 anchor values."""
    assert math.isclose(SHARON_2025_HEALTHY_OLDER_ANCHOR, 1.0)
    assert math.isclose(SHARON_2025_AMCI_MIDPOINT, 0.45)
    assert math.isclose(SHARON_2025_AD_FLOOR, 0.20)


def test_so_amplitude_proxy_monotonic_p_max_p_equ_p_min() -> None:
    """Monotone ordering: proxy(P_max) >= proxy(P_equ) >= proxy(P_min).

    Aligns with DR-4 Lemma DR-4.L (capacity-monotone metric across the
    profile chain): SO-trough amplitude is a substrate-health proxy
    that is monotone in capacity. The healthy anchors P_max and P_equ
    tie at 1.0 (Sharon 2025 healthy-older arm); P_min sits below at
    0.45 (aMCI midpoint placeholder).
    """
    proxy_max = compute_so_amplitude_proxy(PMaxProfile())
    proxy_equ = compute_so_amplitude_proxy(PEquProfile())
    proxy_min = compute_so_amplitude_proxy(PMinProfile())
    assert proxy_max >= proxy_equ, (
        f"SO-trough monotonicity broken: P_max={proxy_max} < P_equ={proxy_equ}"
    )
    assert proxy_equ >= proxy_min, (
        f"SO-trough monotonicity broken: P_equ={proxy_equ} < P_min={proxy_min}"
    )


def test_so_amplitude_proxy_p_min_strictly_below_p_equ() -> None:
    """P_min sits *strictly* below the healthy anchor (degraded substrate).

    A strict-inequality assertion guards against accidental upward
    drift of the P_min default (e.g. someone copy-pasting from P_equ).
    The 1e-6 margin is paranoid — values are explicit floats not RNG.
    """
    proxy_equ = compute_so_amplitude_proxy(PEquProfile())
    proxy_min = compute_so_amplitude_proxy(PMinProfile())
    assert proxy_min < proxy_equ - 1e-6, (
        "P_min must be strictly degraded vs P_equ on SO-trough proxy"
    )


def test_so_amplitude_proxy_deterministic_across_instances() -> None:
    """Two independent P_min instances yield identical proxy values.

    Calibration must be a constant of the profile class, not RNG-driven.
    R1 reproducibility contract: same (c_version, profile, seed) ->
    same proxy value.
    """
    assert (
        compute_so_amplitude_proxy(PMinProfile())
        == compute_so_amplitude_proxy(PMinProfile())
    )


def test_so_amplitude_proxy_independent_of_p_equ_rng_seed() -> None:
    """Seeding the P_equ rng field does not perturb the SO proxy.

    Guards against accidental future coupling between the recombine_light
    RNG (P_equ.rng) and the calibration field.
    """
    import random

    rng_a = random.Random(0)
    rng_b = random.Random(424242)
    profile_a = PEquProfile(rng=rng_a)
    profile_b = PEquProfile(rng=rng_b)
    assert (
        compute_so_amplitude_proxy(profile_a)
        == compute_so_amplitude_proxy(profile_b)
    )


def test_so_amplitude_proxy_independent_of_p_max_rng_seed() -> None:
    """Seeding the P_max rng field does not perturb the SO proxy."""
    import random

    profile_a = PMaxProfile(rng=random.Random(0))
    profile_b = PMaxProfile(rng=random.Random(424242))
    assert (
        compute_so_amplitude_proxy(profile_a)
        == compute_so_amplitude_proxy(profile_b)
    )


def test_so_calibration_coexists_with_dr4_chain_inclusion() -> None:
    """SO calibration field must not perturb DR-4 ops/channels chain.

    Cross-check: instantiate all three profiles, verify their op-handler
    registries still satisfy DR-4 chain inclusion in the presence of
    the new calibration field. This is a regression guard, not the
    axiom test (cf. tests/conformance/axioms/test_dr4_profile_inclusion.py).
    """
    p_min = PMinProfile()
    p_equ = PEquProfile()
    p_max = PMaxProfile()
    ops_min = set(p_min.runtime._handlers.keys())
    ops_equ = set(p_equ.runtime._handlers.keys())
    assert ops_min <= ops_equ, "DR-4 ops chain regressed under SO calibration"
    assert ops_equ <= p_max.target_ops, "DR-4 P_equ subset P_max regressed"
    # And the SO proxy still evaluates on each — no AttributeError.
    _ = compute_so_amplitude_proxy(p_min)
    _ = compute_so_amplitude_proxy(p_equ)
    _ = compute_so_amplitude_proxy(p_max)
