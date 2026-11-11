"""Positive control of the C1 / C2 decision rule on a network with a known symmetry.

Pre-registration draft (L0), section 6 quater:
docs/milestones/2026-11-11-downscale-dose-response.md.

A small classifier in which every linear map is followed by an RMS
normalisation and has no bias: its output does not depend on the scale of
the weights (the construction of Lyle et al. 2024). Two variants:

- ``gain=False``: no other parameter. DOWNSCALE is an exact no-op.
- ``gain=True``: one scalar output gain, also a parameter. DOWNSCALE
  multiplies it by (1 - eps), which is exactly a temperature 1 / (1 - eps)
  on the scores and leaves every argmax unchanged.

The arms of the dose-response are run on it and the verdict is computed by
the unmodified ``scripts.e1_dose_response.analyse``. Expected: C1 supported
in both variants; C2 not applicable without the gain, supported with it.
Pure numpy, CPU, no training loop (random features and a ridge readout).

The depth was chosen after looking at five configurations: the control needs
a noise arm that breaks the network inside the dose grid, and at depth 2 it
does not (the gate then returns non_test). This is a precondition of the
control, set by hand, not one of its outcomes.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from scripts.e1_dose_response import DOSES, RT_SEEDS, analyse

N_ITEMS = 760
N_CLASSES = 4
DIM = 32
HIDDEN = 256
DEPTH = 6  # at depth 2 the noise arm never breaks the network inside the grid
N_TRAIN = 4000
CLASS_SEP = 1.4
OUTPUT_GAIN = 4.0
RIDGE = 1e-2
SEED = 20261111

Params = dict[str, np.ndarray]


def _rms(z: np.ndarray) -> np.ndarray:
    out: np.ndarray = z / np.sqrt((z**2).mean(-1, keepdims=True) + 1e-300)
    return out


def _hidden(p: Params, x: np.ndarray) -> np.ndarray:
    h = x
    for k in sorted(q for q in p if q.startswith("h")):
        h = np.maximum(_rms(h @ p[k]), 0.0)
    return h


def scores(p: Params, x: np.ndarray) -> np.ndarray:
    z = _rms(_hidden(p, x) @ p["out"])
    gain = float(p["gain"][0]) if "gain" in p else OUTPUT_GAIN
    out: np.ndarray = gain * z
    return out


def _data(rng: np.random.Generator, centres: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    y = rng.integers(0, N_CLASSES, n)
    return centres[y] + rng.normal(size=(n, DIM)), y


def build(gain: bool, seed: int = SEED) -> tuple[Params, np.ndarray, np.ndarray]:
    """Random features, ridge readout fitted on a training set; returns the test set."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(N_CLASSES, DIM)) * CLASS_SEP / np.sqrt(DIM) * np.sqrt(DIM / 8)
    p: Params = {f"h{k:02d}": rng.normal(size=(DIM if k == 0 else HIDDEN, HIDDEN))
                 for k in range(DEPTH)}
    xtr, ytr = _data(rng, centres, N_TRAIN)
    h = _hidden(p, xtr)
    target = np.eye(N_CLASSES)[ytr] - 1.0 / N_CLASSES
    p["out"] = np.linalg.solve(h.T @ h + RIDGE * N_TRAIN * np.eye(HIDDEN), h.T @ target)
    if gain:
        p["gain"] = np.array([OUTPUT_GAIN])
    x, y = _data(rng, centres, N_ITEMS)
    return p, x, y


def downscale(p: Params, eps: float) -> Params:
    return {k: v * (1 - eps) for k, v in p.items()}


def perturb_per_tensor(p: Params, eps: float, seed: int) -> Params:
    """Each tensor receives Gaussian noise of norm eps * ||tensor||."""
    rng = np.random.default_rng(1000 + seed)
    out: Params = {}
    for k in sorted(p):
        g = rng.normal(size=p[k].shape)
        out[k] = p[k] + g * (eps * np.linalg.norm(p[k]) / np.linalg.norm(g))
    return out


def passes(p: Params, x: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"arm": "base", "eps": None, "seed": None, "scores": scores(p, x)}]
    for eps in DOSES:
        out.append({"arm": "D", "eps": eps, "seed": None, "scores": scores(downscale(p, eps), x)})
        out.extend({"arm": "Rt", "eps": eps, "seed": s,
                    "scores": scores(perturb_per_tensor(p, eps, s), x)} for s in RT_SEEDS)
    return out


def relabelled(ps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Negative control: the noise of seed 0 presented as the DOWNSCALE arm."""
    keep = [q for q in ps if q["arm"] != "D"]
    fake = [{**q, "arm": "D", "seed": None} for q in ps if q["arm"] == "Rt" and q["seed"] == 0]
    return keep + fake


def control(gain: bool, n_boot: int = 10_000) -> dict[str, Any]:
    p, x, y = build(gain)
    ps = passes(p, x)
    keep = np.ones(y.size, dtype=bool)
    res = analyse(ps, y, keep, n_boot)
    neg = analyse(relabelled(ps), y, keep, n_boot)
    return {"gain": gain, "verdict": res,
            "negative_control": {"C1": neg["C1"], "C2": neg["C2"],
                                 "rank_D": neg["rank_D"], "rank_Rt": neg["rank_Rt"]}}


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps([control(False), control(True)], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
