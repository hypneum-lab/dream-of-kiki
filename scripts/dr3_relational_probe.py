"""DR-3 relational probe: per-seed statistics from representation dumps.

**Gate** : pre-filing gate of ``docs/osf-prereg-dr3-relational-probe-2026-10-23.md``
(section 9); hypotheses H13-A..D.
**Validates** : nothing by itself. **Measurement only** : consumes
exported representation matrices and writes the per-seed statistics
(``S_mKNN``, ``S_H0``, ``S_mKNN_tau``, ``S_H0_tau``) for the trained
pair and the untrained null pair, plus the calibrated differences
``D = S(trained) - S(null)``. It does NOT train substrates, does NOT
score the learning gate (accuracy >= 0.25, scored by the export helper
``scripts/dr3_export_nerve_wml.py``); it applies the gate recorded in
the dump, so a gate failure is a seed-level exclusion counted towards
the section 4 abort.
With ``--decide`` it also computes the section 2 states, the section 7
row and the "global" wording rule from the retained seeds (fixed n,
run once after all 20 seeds are dumped and hashed, section 8).
**Mode** : pipeline code, no empirical claim.

Input format (one ``.npz`` per seed, float32 or float64)
--------------------------------------------------------
``seed``       scalar int
``a_trained``  (n, d_a)  substrate A (MlpWML) trained, pre-VQ codes
``b_trained``  (n, d_b)  substrate B (LifWML) trained, pre-VQ codes
``a_null``     (n, d_a)  substrate A, same constructor/seed, 0 steps
``b_null``     (n, d_b)  substrate B, same constructor/seed, 0 steps

Row ``i`` of all four arrays is the same input (the 1024 inputs of
that seed). Optional learning-gate fields written by the export helper:
``gate_pass`` (bool) and ``acc_a_trained``, ``acc_b_trained``,
``acc_a_null``, ``acc_b_null``; ``gate_pass`` false excludes the seed. An optional sidecar ``<file>.npz.sha256`` (hex digest as
first token) is verified when present; ``--require-hash`` makes it
mandatory.

Output : JSON (``--output``), per seed ``status`` ``ok`` or
``excluded`` (with reasons, section 4), ``trained`` / ``null`` /
``D`` dicts, the exploratory ``conditional_weight`` (``S^tau / S``
trained minus null), the descriptive 1-NN ``comparator``
(``S_mKNN_k1``, ``S_H0_res``, section 5) and, with ``--sensitivity``, the preprocessing /
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

from kiki_oniric.eval.statistics import paired_tost  # noqa: E402
from kiki_oniric.eval.relational_probe import (  # noqa: E402
    PREPROCESS_MODES,
    STAT_KEYS,
    calibrated_difference,
    check_representation,
    nn_comparator,
    relational_statistics,
)

ARRAY_KEYS = ("a_trained", "b_trained", "a_null", "b_null")
NERVE_WML_COMMIT = "95ac6b78477c3ce999b31bca775769041e082b46"
PRIMARY_K = 10
PRIMARY_TAU = 0.1
TAU_GRID = (1.0, 0.1, 0.01, 0.001, 0.0001)
K_SENSITIVITY = (5, 20)
DEFAULT_ABORT_THRESHOLD = 6
DELTA = 0.025
ALPHA = 0.05 / 4
CONFIRMATORY_SEEDS = tuple(range(100, 120))
COMPARATOR_KEYS = ("S_mKNN_k1", "S_H0_res")
GATE_THRESHOLD = 0.25
GATE_ACC_KEYS = ("acc_a_trained", "acc_b_trained", "acc_a_null", "acc_b_null")
REGISTERED_SHAPE = (1024, 16)
PERM_K = 500
PERM_ALPHA = 0.05


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
        rec["gate"] = None
        if "gate_pass" in z.files:
            gate: dict[str, Any] = {"gate_pass": bool(np.asarray(z["gate_pass"]))}
            for key in GATE_ACC_KEYS:
                if key in z.files:
                    gate[key] = float(np.asarray(z[key]))
            rec["gate"] = gate
        rec["pinned"] = bool(np.asarray(z["pinned"])) if "pinned" in z.files else None
    rows = {rec[k].shape[0] for k in ARRAY_KEYS}
    if len(rows) != 1:
        raise ValueError(f"{p.name}: arrays must have the same number of rows, got {sorted(rows)}")
    return rec


def gate_failures(gate: dict[str, Any]) -> list[str]:
    """Section 4 learning gate recomputed from the exported accuracies.

    The supplied ``gate_pass`` is not trusted: a trained accuracy that
    is missing, non-finite or below 0.25 fails the gate.
    """
    out = []
    for key in ("acc_a_trained", "acc_b_trained"):
        acc = gate.get(key)
        if acc is None or not np.isfinite(acc):
            out.append(f"learning gate: {key[4:]} accuracy missing or non-finite")
        elif acc < GATE_THRESHOLD:
            out.append(f"learning gate: {key[4:]} accuracy {acc:.4f} < {GATE_THRESHOLD}")
    return out


def calibrate(t_obs: float, t_perm: Any, alpha: float = PERM_ALPHA) -> dict[str, Any]:
    """You et al. (arXiv:2609.27252, section 3.4, Eqs. 7-8) calibration.

    ``c_alpha`` is the (1 - alpha) quantile of the pool
    ``{T_obs, T_1, ..., T_K}``, taken as the order statistic of rank
    ``ceil((1 - alpha)(K + 1))``; ``T_cal = max((T_obs - c)/(1 - c), 0)``
    (0 when ``c >= 1``). With alpha = 0.05 this is the 95th percentile.
    """
    perm = np.asarray(t_perm, dtype=float)
    pool = np.sort(np.concatenate([[float(t_obs)], perm]))
    rank = int(np.ceil((1.0 - alpha) * pool.size))
    c = float(pool[rank - 1])
    t_cal = max((float(t_obs) - c) / (1.0 - c), 0.0) if c < 1.0 else 0.0
    return {
        "T_obs": float(t_obs),
        "c_alpha": c,
        "T_cal": t_cal,
        "K": int(perm.size),
        "alpha": alpha,
        "p_add_one": float((1 + np.sum(perm >= t_obs)) / (perm.size + 1)),
    }


def permutation_null(
    x: Any,
    y: Any,
    k: int = PRIMARY_K,
    tau: float = PRIMARY_TAU,
    n_perm: int = PERM_K,
    alpha: float = PERM_ALPHA,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Secondary permutation calibration (section 5, departure 1).

    Each draw permutes the sample correspondence of ``y`` (its rows)
    and recomputes the same four statistics, as in You et al.; ``x`` is
    never permuted. The stream is ``default_rng([seed, 2609])``.
    """
    rng = np.random.default_rng([seed, 2609])
    obs = _pair_stats(x, y, k, tau, "l2")
    draws: dict[str, list[float]] = {s: [] for s in STAT_KEYS}
    ya = np.asarray(y)
    for _ in range(n_perm):
        st = _pair_stats(x, ya[rng.permutation(ya.shape[0])], k, tau, "l2")
        for s in STAT_KEYS:
            draws[s].append(float(st[s]))
    return {s: calibrate(float(obs[s]), draws[s], alpha) for s in STAT_KEYS}


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
    n_perm: int = 0,
) -> dict[str, Any]:
    """Statistics for one seed, or an exclusion record (section 4)."""
    gate = rec.get("gate")
    out: dict[str, Any] = {
        "seed": rec["seed"],
        "sha256": rec.get("sha256"),
        "gate": gate,
        "pinned": rec.get("pinned"),
        "dims": {key: list(np.shape(rec[key])) for key in ARRAY_KEYS},
    }
    reasons = []
    if gate is not None:
        failed = gate_failures(gate)
        out["gate_recomputed_pass"] = not failed
        out["gate_pass_mismatch"] = bool(gate["gate_pass"]) != (not failed)
        reasons.extend(failed)
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
    comp_t = nn_comparator(rec["a_trained"], rec["b_trained"], "l2")
    comp_n = nn_comparator(rec["a_null"], rec["b_null"], "l2")
    out["comparator"] = {
        "trained": comp_t,
        "null": comp_n,
        "D": calibrated_difference(comp_t, comp_n),
    }
    if sensitivity:
        out["sensitivity"] = _sensitivity(rec, k, tau)
    if n_perm:
        out["permutation_null"] = {
            "reported_only": True,
            "trained": permutation_null(rec["a_trained"], rec["b_trained"], k, tau,
                                        n_perm, PERM_ALPHA, seed=int(rec["seed"])),
            "null": permutation_null(rec["a_null"], rec["b_null"], k, tau,
                                     n_perm, PERM_ALPHA, seed=int(rec["seed"]) + 1),
        }
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


# --- section 2 states, section 7 rows, wording rule ----------------------


def _bounds(d: list[float], alpha: float) -> dict[str, Any]:
    """Paired one-sided (1 - alpha) bounds of mean(D) (df = n - 1).

    ``delta`` only parametrises paired_tost's equivalence flag, unused
    here; identical differences give L = U = mean(D) exactly.
    """
    r = paired_tost(np.asarray(d, dtype=float), np.zeros(len(d)), delta=1.0, alpha=alpha)
    return {
        "n": r.n, "mean": r.mean_diff, "sd": r.sd_diff,
        "L": r.ci_lower, "U": r.ci_upper, "degenerate": r.degenerate,
    }


def statistic_state(d: list[float], delta: float = DELTA, alpha: float = ALPHA) -> dict[str, Any]:
    """Section 2 state of one statistic: SURVIVES / FAILS / INDETERMINATE."""
    out = _bounds(d, alpha)
    if out["L"] > 0 and out["mean"] >= delta:
        out["state"] = "SURVIVES"
    elif out["U"] < delta:
        out["state"] = "FAILS"
    else:
        out["state"] = "INDETERMINATE"
    return out


def outcome_row(states: tuple[str, ...] | list[str], *, abort: bool) -> int:
    """First matching section 7 row (0-5) from the (A, B, C, D) states."""
    if abort:
        return 0
    a, b, c, d = states
    if a == b == "SURVIVES" and c == d == "FAILS":
        return 1
    if a == b == "SURVIVES" and "SURVIVES" in (c, d):
        return 2
    if a == "SURVIVES" and b == "FAILS":
        return 3
    if a == "FAILS":
        return 4
    return 5


def eligibility_problems(doc: dict[str, Any]) -> list[str]:
    """Why the run cannot carry a confirmatory decision (empty if it can).

    Requires: the exact unique seed set 100..119; every dump hashed and
    matching a verified export manifest from a pinned v1.9.0 checkout
    (``confirmatory_eligible``); the primary protocol (k = 10,
    tau = 0.1, l2); the registered 1024 x 16 shape for every array;
    and a scored learning gate (finite trained accuracies) on every
    seed.
    """
    probs: list[str] = []
    seeds = [int(s["seed"]) for s in doc.get("seeds", [])]
    if len(seeds) != len(set(seeds)):
        probs.append("duplicate seeds")
    if tuple(sorted(set(seeds))) != CONFIRMATORY_SEEDS:
        probs.append("seed set is not exactly 100..119")
    proto = doc.get("protocol", {})
    if (proto.get("k"), proto.get("tau"), proto.get("preprocess")) != (PRIMARY_K, PRIMARY_TAU, "l2"):
        probs.append("non-primary protocol (k, tau or preprocessing)")
    prov = doc.get("provenance")
    if not prov:
        probs.append("no verified export manifest")
    else:
        if not prov.get("manifest_hash_ok"):
            probs.append("export manifest hash not verified")
        if not prov.get("confirmatory_eligible"):
            probs.append("export manifest is not confirmatory (unpinned checkout)")
        if prov.get("seed_hash_mismatch") or prov.get("seeds_not_in_manifest"):
            probs.append("dump hashes do not match the export manifest")
        nw = prov.get("nerve_wml") or {}
        if nw.get("commit") != NERVE_WML_COMMIT or nw.get("head") != NERVE_WML_COMMIT:
            probs.append(f"export manifest pin is not nerve-wml {NERVE_WML_COMMIT}")
    if doc.get("abort_threshold", DEFAULT_ABORT_THRESHOLD) != DEFAULT_ABORT_THRESHOLD:
        probs.append(
            f"abort threshold overridden ({doc.get('abort_threshold')}); registered"
            f" is more than {DEFAULT_ABORT_THRESHOLD} excluded seeds"
        )
    for s in doc.get("seeds", []):
        sid = s["seed"]
        if not s.get("sha256"):
            probs.append(f"seed {sid}: dump hash not verified")
        if s.get("pinned") is not True:
            probs.append(f"seed {sid}: exported from an unpinned checkout (pinned flag absent or false)")
        dims = s.get("dims") or {}
        if any(tuple(v) != REGISTERED_SHAPE for v in dims.values()) or len(dims) != 4:
            probs.append(f"seed {sid}: arrays are not {REGISTERED_SHAPE}")
        gate = s.get("gate")
        if not gate or any(
            gate.get(kk) is None or not np.isfinite(gate[kk])
            for kk in ("acc_a_trained", "acc_b_trained")
        ):
            probs.append(f"seed {sid}: learning gate not scored")
    return probs


def _reading(doc: dict[str, Any], delta: float, alpha: float) -> dict[str, Any]:
    ok = [s for s in doc["seeds"] if s.get("status") == "ok"]
    abort = bool(doc.get("abort"))
    if len(ok) < 2:
        return {"states": None, "row": 0 if abort else 5, "hypotheses": None,
                "comparator": None, "residual_criterion": False,
                "note": "fewer than 2 retained seeds"}
    states = {k: statistic_state([float(s["D"][k]) for s in ok], delta, alpha) for k in STAT_KEYS}
    comparator = {}
    for k in COMPARATOR_KEYS:
        b = _bounds([float(s["comparator"]["D"][k]) for s in ok], 0.025)
        comparator[k] = {"n": b["n"], "mean": b["mean"], "sd": b["sd"], "ci95": [b["L"], b["U"]]}
    res = comparator["S_H0_res"]
    return {
        "states": states,
        "hypotheses": {
            "H13-A": states["S_mKNN"]["state"] == "SURVIVES",
            "H13-B": states["S_H0"]["state"] == "SURVIVES",
            "H13-C": states["S_mKNN_tau"]["state"] == "FAILS",
            "H13-D": states["S_H0_tau"]["state"] == "FAILS",
        },
        "row": outcome_row(tuple(states[k]["state"] for k in STAT_KEYS), abort=abort),
        "comparator": comparator,
        "residual_criterion": bool(res["mean"] >= delta and res["ci95"][0] > 0),
    }


def decide(doc: dict[str, Any], delta: float = DELTA, alpha: float = ALPHA) -> dict[str, Any]:
    """Section 2 states, section 7 row and the "global" wording rule.

    Uses the seeds with ``status == "ok"`` (seed-level exclusions apply
    to all four statistics). A confirmatory decision needs
    :func:`eligibility_problems` to be empty; otherwise the same
    reading is kept only under ``exploratory`` and the confirmatory
    fields are None. "global" may be used only when the run is
    eligible and not aborted, the row is 1 or 2, and the descriptive
    S_H0_res has mean(D) >= delta with a 95 % interval excluding 0.
    """
    ok = [s for s in doc["seeds"] if s.get("status") == "ok"]
    seed_set = tuple(sorted(int(s["seed"]) for s in doc["seeds"]))
    probs = eligibility_problems(doc)
    n_excl = sum(s.get("status") == "excluded" for s in doc["seeds"])
    # the registered abort (section 4) is recomputed, never taken from a
    # possibly overridden --abort-threshold
    abort = bool(doc.get("abort")) or n_excl > DEFAULT_ABORT_THRESHOLD
    reading = _reading(dict(doc, abort=abort), delta, alpha)
    out: dict[str, Any] = {
        "delta": delta,
        "alpha": alpha,
        "n_retained": len(ok),
        "confirmatory_seed_set": seed_set == CONFIRMATORY_SEEDS,
        "all_retained_hashed": all(s.get("sha256") for s in ok),
        "eligible": not probs,
        "eligibility_problems": probs,
        "abort": abort,
    }
    if probs:
        out.update(states=None, hypotheses=None, row=None, comparator=None,
                   global_wording_allowed=False, exploratory=reading)
        return out
    out.update(
        states=reading["states"],
        hypotheses=reading["hypotheses"],
        row=reading["row"],
        comparator=reading["comparator"],
        global_wording_allowed=bool(
            not abort and reading["row"] in (1, 2) and reading["residual_criterion"]
        ),
    )
    if "note" in reading:
        out["note"] = reading["note"]
    return out


def verify_manifest(path: Path, seeds: list[dict[str, Any]]) -> dict[str, Any]:
    """Check an ``export-manifest.json`` and match it to the loaded dumps."""
    side = path.with_name(path.name + ".sha256")
    hash_ok = side.exists() and side.read_text().split()[0].lower() == _sha256_of(path)
    man = json.loads(path.read_text())
    by_seed = {int(e["seed"]): e for e in man.get("seeds", [])}
    mismatch, missing = [], []
    for s in seeds:
        e = by_seed.get(int(s["seed"]))
        if e is None:
            missing.append(s["seed"])
        elif e.get("sha256") != s.get("sha256"):
            mismatch.append(s["seed"])
    return {
        "manifest": str(path),
        "manifest_hash_ok": bool(hash_ok),
        "confirmatory_eligible": bool(man.get("confirmatory_eligible")),
        "nerve_wml": man.get("nerve_wml"),
        "seed_hash_mismatch": mismatch,
        "seeds_not_in_manifest": missing,
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
    ap.add_argument(
        "--decide", action="store_true",
        help="also compute section 2 states, section 7 row and wording rule",
    )
    ap.add_argument("--manifest", type=Path, help="export-manifest.json of the dumps")
    ap.add_argument("--permutation-null", action="store_true",
                    help="secondary You et al. calibration (K draws, alpha = 0.05)")
    ap.add_argument("--n-perm", type=int, default=PERM_K)
    args = ap.parse_args(argv)
    if args.decide and (args.k != PRIMARY_K or args.tau != PRIMARY_TAU):
        ap.error(f"--decide needs the primary protocol k = {PRIMARY_K}, tau = {PRIMARY_TAU}")
    if args.decide and args.abort_threshold != DEFAULT_ABORT_THRESHOLD:
        ap.error(f"--decide uses the registered abort threshold {DEFAULT_ABORT_THRESHOLD}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_cli(argv)
    seeds = [
        analyse_seed(
            load_seed_npz(p, require_hash=args.require_hash),
            k=args.k,
            tau=args.tau,
            sensitivity=args.sensitivity,
            n_perm=args.n_perm if args.permutation_null else 0,
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
    if args.manifest is not None:
        doc["provenance"] = verify_manifest(args.manifest, seeds)
    if args.decide:
        doc["decision"] = decide(doc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}: retained={doc['n_retained']} excluded={n_excl} abort={doc['abort']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
