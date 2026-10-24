"""DR-3 relational probe: per-seed statistics from representation dumps.

**Gate** : pre-filing gate of ``docs/osf-prereg-dr3-relational-probe-2026-10-23.md``
(section 9); hypotheses H13-A..D.
**Validates** : nothing by itself. **Measurement only** : consumes
exported representation matrices and writes the per-seed statistics
(``S_mKNN``, ``S_H0``, ``S_mKNN_tau``, ``S_H0_tau``) for the trained
pair and the untrained null pair, plus the calibrated differences
``D = S(trained) - S(null)``. It does NOT train substrates, does NOT
apply the learning gate (accuracy >= 0.25, done by the export helper),
and does NOT compute the section 2 states or section 7 rows.
**Mode** : pipeline code, no empirical claim.

Input format (one ``.npz`` per seed, float32 or float64)
--------------------------------------------------------
``seed``       scalar int
``a_trained``  (n, d_a)  substrate A (MlpWML) trained, pre-VQ codes
``b_trained``  (n, d_b)  substrate B (LifWML) trained, pre-VQ codes
``a_null``     (n, d_a)  substrate A, same constructor/seed, 0 steps
``b_null``     (n, d_b)  substrate B, same constructor/seed, 0 steps

Row ``i`` of all four arrays is the same input (the 1024 inputs of
that seed). An optional sidecar ``<file>.npz.sha256`` (hex digest as
first token) is verified when present; ``--require-hash`` makes it
mandatory.

Output : JSON (``--output``), per seed ``status`` ``ok`` or
``excluded`` (with reasons, section 4), ``trained`` / ``null`` /
``D`` dicts, the exploratory ``conditional_weight`` (``S^tau / S``
trained minus null) and, with ``--sensitivity``, the preprocessing /
k / tau sensitivity blocks. ``abort`` is true when more than
``--abort-threshold`` (default 6) seeds are excluded.

Usage ::

    uv run python scripts/dr3_relational_probe.py dumps/seed*.npz \\
        --output out/dr3-probe-stats.json [--sensitivity]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kiki_oniric.eval.relational_probe import (  # noqa: E402
    PREPROCESS_MODES,
    STAT_KEYS,
    calibrated_difference,
    check_representation,
    relational_statistics,
)

ARRAY_KEYS = ("a_trained", "b_trained", "a_null", "b_null")
PRIMARY_K = 10
PRIMARY_TAU = 0.1
TAU_GRID = (1.0, 0.1, 0.01, 0.001, 0.0001)
K_SENSITIVITY = (5, 20)
DEFAULT_ABORT_THRESHOLD = 6


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_seed_npz(path: str | Path, require_hash: bool = False) -> dict[str, Any]:
    """Load and validate one per-seed ``.npz`` (see module docstring)."""
    p = Path(path)
    sidecar = p.with_name(p.name + ".sha256")
    digest: str | None = None
    if sidecar.exists():
        expected = sidecar.read_text().split()[0].lower()
        digest = _sha256_of(p)
        if digest != expected:
            raise ValueError(f"{p.name}: sha256 mismatch (sidecar {expected}, file {digest})")
    elif require_hash:
        raise ValueError(f"{p.name}: required sha256 sidecar {sidecar.name} not found")
    with np.load(p, allow_pickle=False) as z:
        missing = [k for k in ("seed", *ARRAY_KEYS) if k not in z.files]
        if missing:
            raise ValueError(f"{p.name}: missing key(s) {missing}")
        seed_arr = np.asarray(z["seed"])
        if seed_arr.shape != () or not np.issubdtype(seed_arr.dtype, np.integer):
            raise ValueError(f"{p.name}: 'seed' must be a scalar integer")
        rec: dict[str, Any] = {"path": str(p), "seed": int(seed_arr), "sha256": digest}
        for key in ARRAY_KEYS:
            arr = np.asarray(z[key])
            if arr.ndim != 2:
                raise ValueError(f"{p.name}: '{key}' must be 2-D (n, d), got ndim={arr.ndim}")
            if arr.dtype.kind not in "fiu":
                raise ValueError(f"{p.name}: '{key}' must be numeric, got dtype {arr.dtype}")
            rec[key] = arr.astype(np.float64)
    rows = {rec[k].shape[0] for k in ARRAY_KEYS}
    if len(rows) != 1:
        raise ValueError(f"{p.name}: arrays must have the same number of rows, got {sorted(rows)}")
    return rec


def _pair_stats(x: Any, y: Any, k: int, tau: float, mode: str) -> dict[str, float | int]:
    return relational_statistics(x, y, k=k, tau=tau, preprocess=mode)


def _conditional_weight(stats: dict[str, float | int]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, geo, base in (("mKNN", "S_mKNN_tau", "S_mKNN"), ("H0", "S_H0_tau", "S_H0")):
        out[name] = float(stats[geo]) / float(stats[base]) if stats[base] else None
    return out


def analyse_seed(
    rec: dict[str, Any],
    k: int = PRIMARY_K,
    tau: float = PRIMARY_TAU,
    sensitivity: bool = False,
) -> dict[str, Any]:
    """Statistics for one seed, or an exclusion record (section 4)."""
    out: dict[str, Any] = {"seed": rec["seed"], "sha256": rec.get("sha256")}
    reasons = []
    for key in ARRAY_KEYS:
        reason = check_representation(rec[key], "l2")
        if reason is not None:
            reasons.append(f"{key}: {reason}")
    if reasons:
        out.update(status="excluded", reasons=reasons)
        return out
    out["status"] = "ok"
    out["n"] = int(rec["a_trained"].shape[0])
    trained = _pair_stats(rec["a_trained"], rec["b_trained"], k, tau, "l2")
    null = _pair_stats(rec["a_null"], rec["b_null"], k, tau, "l2")
    out["trained"], out["null"] = trained, null
    out["D"] = calibrated_difference(
        {s: float(trained[s]) for s in STAT_KEYS}, {s: float(null[s]) for s in STAT_KEYS}
    )
    ct, cn = _conditional_weight(trained), _conditional_weight(null)
    cd: dict[str, float | None] = {}
    for m in ct:
        a, b = ct[m], cn[m]
        cd[m] = (a - b) if a is not None and b is not None else None
    out["conditional_weight"] = {"trained": ct, "null": cn, "D": cd}
    if sensitivity:
        out["sensitivity"] = _sensitivity(rec, k, tau)
    return out


def _both(rec: dict[str, Any], k: int, tau: float, mode: str) -> dict[str, Any]:
    tr = _pair_stats(rec["a_trained"], rec["b_trained"], k, tau, mode)
    nu = _pair_stats(rec["a_null"], rec["b_null"], k, tau, mode)
    d = calibrated_difference(
        {s: float(tr[s]) for s in STAT_KEYS}, {s: float(nu[s]) for s in STAT_KEYS}
    )
    return {"trained": tr, "null": nu, "D": d}


def _sensitivity(rec: dict[str, Any], k: int, tau: float) -> dict[str, Any]:
    """Reported only; never changes a verdict (section 5)."""
    pipelines: dict[str, Any] = {}
    for mode in PREPROCESS_MODES:
        reason = [check_representation(rec[key], mode) for key in ARRAY_KEYS]
        pipelines[mode] = (
            {"excluded": [r for r in reason if r]} if any(reason) else _both(rec, k, tau, mode)
        )
    return {
        "pipelines": pipelines,
        "k": {str(kk): _both(rec, kk, tau, "l2") for kk in K_SENSITIVITY},
        "tau_grid": {f"{t:g}": _both(rec, k, t, "l2") for t in TAU_GRID},
    }


def _parse_cli(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DR-3 relational probe statistics")
    ap.add_argument("inputs", nargs="+", type=Path, help="per-seed .npz dumps")
    ap.add_argument("--output", type=Path, required=True, help="stats JSON path")
    ap.add_argument("--k", type=int, default=PRIMARY_K)
    ap.add_argument("--tau", type=float, default=PRIMARY_TAU)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--require-hash", action="store_true")
    ap.add_argument("--abort-threshold", type=int, default=DEFAULT_ABORT_THRESHOLD)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_cli(argv)
    seeds = [
        analyse_seed(
            load_seed_npz(p, require_hash=args.require_hash),
            k=args.k,
            tau=args.tau,
            sensitivity=args.sensitivity,
        )
        for p in sorted(args.inputs)
    ]
    seeds.sort(key=lambda r: r["seed"])
    n_excl = sum(r["status"] == "excluded" for r in seeds)
    doc = {
        "script": "scripts/dr3_relational_probe.py",
        "prereg": "docs/osf-prereg-dr3-relational-probe-2026-10-23.md",
        "protocol": {
            "k": args.k,
            "tau": args.tau,
            "preprocess": "l2",
            "statistics": list(STAT_KEYS),
            "calibration": "difference trained - null (paired per seed)",
        },
        "n_total": len(seeds),
        "n_retained": len(seeds) - n_excl,
        "n_excluded": n_excl,
        "abort_threshold": args.abort_threshold,
        "abort": n_excl > args.abort_threshold,
        "seeds": seeds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}: retained={doc['n_retained']} excluded={n_excl} abort={doc['abort']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
