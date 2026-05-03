"""G5 pilot driver — Split-FMNIST × profile sweep on E-SNN substrate.

**Gate ID** : G5 — first cross-substrate empirical pilot.
**Validates** : whether the per-arm retention distribution observed
on the MLX substrate (G4-bis) is statistically consistent with the
distribution observed on the E-SNN thalamocortical substrate. A
"consistency" verdict (Welch two-sided test fails to reject at
α/4 = 0.0125) upgrades DR-3 evidence in
`docs/proofs/dr3-substrate-evidence.md` from "synthetic substitute"
to "real-substrate empirical evidence".

**Mode** : empirical claim at first-pilot scale (N=5 seeds per arm).
**Expected output** :
    - docs/milestones/g5-cross-substrate-2026-05-03.json
    - docs/milestones/g5-cross-substrate-2026-05-03.md

Sweep : arms × seeds = 4 × 5 = 20 cells, mirroring G4-bis :
    arms  = ["baseline", "P_min", "P_equ", "P_max"]
    seeds = [0, 1, 2, 3, 4]

Usage ::

    uv run python experiments/g5_cross_substrate/run_g5.py --smoke
    uv run python experiments/g5_cross_substrate/run_g5.py
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """G5 driver entry point — full body lands in Task 4."""
    parser = argparse.ArgumentParser(
        description="G5 cross-substrate pilot driver"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.parse_args(argv)
    raise NotImplementedError(
        "run_g5.main() body lands in Task 4 — Task 1 ships only the stub"
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
