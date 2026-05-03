"""Smoke check that the G6 OSF pre-reg landed and pins the right hypotheses.

Mirrors `tests/unit/test_osf_prereg_g4_present.py` (where present) and
exists primarily to lock Task 1 of the G6 plan : the pre-reg lives at
`docs/osf-prereg-g6-pilot.md`, enumerates the four hypotheses (H1', H3',
H_DR4', H_NEW), pins the five subdomains, and discloses Path A / Path B.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PREREG = REPO_ROOT / "docs" / "osf-prereg-g6-pilot.md"


def test_g6_prereg_exists() -> None:
    assert PREREG.is_file(), f"missing {PREREG}"


def test_g6_prereg_pins_hypotheses() -> None:
    text = PREREG.read_text(encoding="utf-8")
    for token in (
        "H1'", "H3'", "H_DR4'", "H_NEW",
        "Hu 2020", "Javadi 2024",
        "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology",
        "Bonferroni",
    ):
        assert token in text, f"pre-reg missing token {token!r}"


def test_g6_prereg_pins_path_branches() -> None:
    text = PREREG.read_text(encoding="utf-8")
    assert "Path A" in text and "Path B" in text, (
        "pre-reg must enumerate both Path A (full pilot) and "
        "Path B (inference-only) branches"
    )


def test_g6_prereg_carries_g4_bis_amendment() -> None:
    """H_NEW must be honestly reformulated given G4-bis g_h1 = -2.31."""
    text = PREREG.read_text(encoding="utf-8")
    # Amendment must reference G4-bis sign-reversal + reformulate H_NEW
    # as exploratory infrastructure validation, not a non-inferiority
    # margin test.
    assert "Amendment" in text or "amendment" in text
    assert "g4-bis" in text.lower() or "G4-bis" in text
    assert "sign-reversed" in text or "sign reversed" in text or "negative" in text
    assert "exploratory" in text.lower()
    # Must explicitly disclose that Path B does not promote STABLE
    assert "STABLE" in text
