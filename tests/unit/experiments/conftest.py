"""Shared fixtures for tests/unit/experiments/.

Extracted from per-file copy-paste in N2 Task 2 (commit 450f828)
per N3 Task 2 (loose-end #4 from N2 audit, 2026-05-10).
"""
from __future__ import annotations

from pathlib import Path

import pytest

# H3 fix (N2 plan, 2026-05-10): the FMNIST data directory is gitignored
# (raw IDX files; not committed). Convert "data missing" from a hard
# failure into a clean skip so a fresh clone reports green by default
# and surfaces this as a "data prerequisite" rather than a code bug.
FMNIST_DATA = (
    Path(__file__).resolve().parents[3] / "experiments/g4_split_fmnist/data"
)

skip_if_no_fmnist = pytest.mark.skipif(
    not FMNIST_DATA.exists(),
    reason=(
        f"FMNIST raw IDX files not present at {FMNIST_DATA}; "
        "download the 4 Fashion-MNIST IDX files (train/test, "
        "images/labels) from "
        "https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion "
        "into that directory before running this pilot. "
        "The loader at experiments/g4_split_fmnist/dataset.py "
        "expects the standard IDX layout."
    ),
)
