#!/usr/bin/env python3
"""E1-small: calibrated paired nulls and checks next to the registered reading.

**Status** : exploratory supplement (2026-11-06). It does not replace
``scripts/analyze_e1_cuda.py``, which alone decides the section 2 states.
It reads the same confirmatory cell cache and, for each contrast, uses
the same per-seed statistic ``D_s = delta_s(X) - delta_s(Y)`` with the
same pairing and exclusions (every cell of the nine registered profiles
is retained in this cache; the script re-checks that).

Per contrast it reports:

- the paired sign-flip null of the mean difference (exchange of the two
  arm labels within each seed, N draws), its median and P99, the
  empirical p = (k + 1) / (N + 1) for the registered direction
  (superiority: H0 mu = 0; non-inferiority: H0 mu = -delta, by flipping
  ``D_s + delta``), and the same for the paired t statistic;
- the null distribution of the lower bound L, to say how often the
  registered threshold ``L > 0`` / ``L > -delta`` is passed under H0;
- a seed bootstrap interval of the mean difference;
- the item discordance p_disc between the two post checkpoints;
- the power of the registered state rules at the observed SD_D
  (non-central t, exact for the bound rules; Monte Carlo for SUPERIOR).

Also: a learning gate (per-profile delta against the item-sampling noise
of ``p_noise``), invariants, budgets, instrument fields, and the
H14-B same-norm bracketing of the declaration of 2026-11-02, section 1.3.

    python scripts/e1_small_null_supplement.py <cells.jsonl> --out supp.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

DELTA = 0.01
ALPHA = 1.0 / 60.0
ALPHA_SEC = 0.05
SEEDS = list(range(141))
REGISTERED = ["p_min", "p_equ", "p_max", "p_ctrl_sft", "p_ctrl_shuffle", "p_noise",
              "p_ctrl_template", "p_ctrl_vanilla_replay_tuned", "p_ctrl_onpolicy_replay"]
CONTRASTS = {
    "H14-A": ("p_min", "p_ctrl_template", "superiority", ALPHA),
    "H14-B": ("p_equ", "p_ctrl_vanilla_replay_tuned", "superiority", ALPHA),
    "H14-C": ("p_equ", "p_ctrl_onpolicy_replay", "noninferiority", ALPHA),
    "sec:p_min-p_ctrl_vanilla_replay_tuned": ("p_min", "p_ctrl_vanilla_replay_tuned",
                                              "superiority", ALPHA_SEC),
    "sec:p_min-p_ctrl_sft": ("p_min", "p_ctrl_sft", "superiority", ALPHA_SEC),
    "sec:p_ctrl_sft-p_ctrl_shuffle": ("p_ctrl_sft", "p_ctrl_shuffle", "superiority",
                                      ALPHA_SEC),
}


def bounds(d: np.ndarray, alpha: float) -> tuple[float, float, float, float]:
    n = d.size
    m = float(d.mean())
    s = float(d.std(ddof=1))
    se = s / math.sqrt(n)
    q = float(stats.t.ppf(1 - alpha, n - 1))
    return m, s, m - q * se, m + q * se


def signflip(d: np.ndarray, shift: float, alpha: float, n_draws: int,
             rng: np.random.Generator) -> dict[str, Any]:
    """Paired sign-flip null of (D + shift); returns null of mean, t and L (unshifted scale)."""
    x = d + shift
    n = x.size
    q = float(stats.t.ppf(1 - alpha, n - 1))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_draws, n))
    xs = signs * np.abs(x)[None, :]
    means = xs.mean(axis=1)
    sds = xs.std(axis=1, ddof=1)
    ts = np.where(sds > 0, means / (sds / math.sqrt(n)), 0.0)
    Ls = means - q * sds / math.sqrt(n) - shift  # L on the original D scale
    m_obs = float(x.mean())
    s_obs = float(x.std(ddof=1))
    t_obs = m_obs / (s_obs / math.sqrt(n)) if s_obs > 0 else 0.0
    k_mean = int((means >= m_obs).sum())
    k_t = int((ts >= t_obs).sum())
    thr = 0.0 if shift == 0 else -shift
    return {
        "n_draws": n_draws, "shift": shift,
        "null_mean_D": {"median": float(np.median(means - shift)),
                        "p99": float(np.quantile(means - shift, 0.99)),
                        "p01": float(np.quantile(means - shift, 0.01))},
        "null_t": {"median": float(np.median(ts)), "p99": float(np.quantile(ts, 0.99))},
        "null_L": {"median": float(np.median(Ls)), "p99": float(np.quantile(Ls, 0.99)),
                   "share_passing_threshold": float((Ls > thr).mean())},
        "observed_mean_D": float(d.mean()), "observed_t_shifted": t_obs,
        "p_mean": (k_mean + 1) / (n_draws + 1), "p_t": (k_t + 1) / (n_draws + 1),
        "k_mean": k_mean, "k_t": k_t,
    }


def bootstrap(d: np.ndarray, n_draws: int, rng: np.random.Generator) -> dict[str, float]:
    idx = rng.integers(0, d.size, size=(n_draws, d.size))
    m = d[idx].mean(axis=1)
    return {"lo95": float(np.quantile(m, 0.025)), "hi95": float(np.quantile(m, 0.975)),
            "lo_1m2a": float(np.quantile(m, ALPHA)), "hi_1m2a": float(np.quantile(m, 1 - ALPHA))}


def power(sd: float, n: int, alpha: float, rng: np.random.Generator,
          draws: int = 200_000) -> dict[str, float]:
    """Power of the state rules at sd, by sufficient-statistic Monte Carlo."""
    q = float(stats.t.ppf(1 - alpha, n - 1))
    out: dict[str, float] = {}
    for mu, tag in ((DELTA, "superior_at_mu_delta"), (1.5 * DELTA, "superior_at_mu_1p5delta"),
                    (0.0, "bounded_or_ni_at_mu_0")):
        means = rng.normal(mu, sd / math.sqrt(n), draws)
        s = sd * np.sqrt(rng.chisquare(n - 1, draws) / (n - 1))
        se = s / math.sqrt(n)
        L, U = means - q * se, means + q * se
        if tag.startswith("superior"):
            out[tag] = float(((L > 0) & (means >= DELTA)).mean())
        else:
            out[tag] = float((U < DELTA).mean())  # = P(L > -delta) by symmetry at mu 0
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=10_000)
    ap.add_argument("--rng-seed", type=int, default=20261106)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.rng_seed)
    cells = [json.loads(x) for x in a.cells.read_text(encoding="utf-8").splitlines() if x.strip()]
    idx = {(c["profile"], int(c["seed"])): c for c in cells}
    res: dict[str, Any] = {"draws": a.draws, "rng_seed": a.rng_seed, "n_cells": len(cells)}

    # integrity
    reg = [c for c in cells if c["profile"] in REGISTERED]
    res["integrity"] = {
        "registered_cells": len(reg),
        "missing_registered": [f"{p}|{s}" for p in REGISTERED for s in SEEDS if (p, s) not in idx],
        "invariant_violations": sum(1 for c in reg if c.get("invariant_violation")),
        "non_finite": sum(1 for c in reg if not all(
            isinstance(c.get(k), float | int) and math.isfinite(c[k]) for k in ("pre", "post"))),
        "budget_status": sorted({c["budget"]["status"] for c in reg}),
        "budget_mismatch": sum(1 for c in reg
                               if c["achieved_loss_tokens"] != c["budget"]["update_tokens"]),
        "code_commit": sorted({c["code_commit"] for c in cells}),
        "run_identity": sorted({c["run_identity"] for c in cells}),
        "amendment_sha256": sorted({c["amendment_sha256"] for c in cells}),
        "backend": sorted({c["backend"] for c in cells}),
        "linear_attn": sorted({str(c["versions"].get("linear_attn")) for c in cells}),
        "deterministic": sorted({str(c["versions"].get("determinism", {}).get("enabled"))
                                 for c in cells}),
        "n_eval": sorted({c["n_eval_items"] for c in cells}),
        "pre_identical_within_seed": all(
            len({idx[(p, s)]["pre"] for p in REGISTERED}) == 1 for s in SEEDS),
        "exploratory_lr_matched_cells": sum(1 for c in cells
                                            if c["profile"] == "p_ctrl_lr_matched"),
    }

    # learning gate: per-profile delta, and noise floor from p_noise
    noise = np.array([idx[("p_noise", s)]["post"] - idx[("p_noise", s)]["pre"] for s in SEEDS])
    lg: dict[str, Any] = {"p_noise_delta": {"mean": float(noise.mean()),
                                             "sd": float(noise.std(ddof=1)),
                                             "p01": float(np.quantile(noise, 0.01)),
                                             "p99": float(np.quantile(noise, 0.99))}}
    for p in REGISTERED:
        d = np.array([idx[(p, s)]["post"] - idx[(p, s)]["pre"] for s in SEEDS])
        sf = signflip(d, 0.0, ALPHA_SEC, a.draws, rng)
        lg[p] = {"mean_delta": float(d.mean()), "sd": float(d.std(ddof=1)),
                 "pre_mean": float(np.mean([idx[(p, s)]["pre"] for s in SEEDS])),
                 "post_mean": float(np.mean([idx[(p, s)]["post"] for s in SEEDS])),
                 "signflip_p_delta_gt_0": sf["p_mean"],
                 "share_seeds_above_noise_p99": float((d > lg["p_noise_delta"]["p99"]).mean()),
                 "collapse": sum(1 for s in SEEDS if idx[(p, s)].get("collapse"))}
    res["learning_gate"] = lg

    out: dict[str, Any] = {}
    for name, (x, y, kind, alpha) in CONTRASTS.items():
        d = np.array([(idx[(x, s)]["post"] - idx[(x, s)]["pre"])
                      - (idx[(y, s)]["post"] - idx[(y, s)]["pre"]) for s in SEEDS])
        m, s, L, U = bounds(d, alpha)
        shift = DELTA if kind == "noninferiority" else 0.0
        disc = [float(np.mean(np.array(idx[(x, k)]["post_correct"])
                              != np.array(idx[(y, k)]["post_correct"]))) for k in SEEDS]
        out[name] = {
            "x": x, "y": y, "kind": kind, "alpha": alpha, "n": int(d.size),
            "mean_D": m, "sd_D": s, "L": L, "U": U,
            "n_D_zero": int((d == 0).sum()), "n_D_pos": int((d > 0).sum()),
            "n_D_neg": int((d < 0).sum()),
            "sum_D_items": float(np.round(d.sum() * 162, 6)),
            "signflip": signflip(d, shift, alpha, a.draws, rng),
            "bootstrap": bootstrap(d, a.draws, rng),
            "p_disc_post": {"mean": float(np.mean(disc)), "min": float(np.min(disc)),
                            "max": float(np.max(disc)),
                            "seeds_identical_post": int(sum(v == 0 for v in disc))},
            "power_at_observed_sd": power(s, int(d.size), alpha, rng),
        }
    res["contrasts"] = out

    # H14-B same-norm bracketing (declaration 2026-11-02, section 1.3), exploratory
    br = []
    for s in SEEDS:
        e, v = idx[("p_equ", s)], idx[("p_ctrl_vanilla_replay_tuned", s)]
        th0 = e["norm_change"] / e["drift"] if e["drift"] else float("nan")
        th0v = v["norm_change"] / v["drift"] if v["drift"] else float("nan")
        aa = v["norm_change"]
        br.append({"seed": s, "theta0": th0, "theta0_from_vanilla": th0v, "a": aa,
                   "rho_lo": 0.01 * abs(th0 - aa), "rho_hi": 0.01 * (th0 + aa),
                   "norm_change_equ": e["norm_change"]})
    res["h14b_same_norm_bracket"] = {
        "theta0_median": float(np.median([b["theta0"] for b in br])),
        "theta0_rel_spread_between_arms": float(np.max([abs(b["theta0"] - b["theta0_from_vanilla"])
                                                        / b["theta0"] for b in br])),
        "a_median": float(np.median([b["a"] for b in br])),
        "rho_lo_median": float(np.median([b["rho_lo"] for b in br])),
        "rho_hi_median": float(np.median([b["rho_hi"] for b in br])),
        "norm_change_equ_median": float(np.median([b["norm_change_equ"] for b in br])),
        "ratio_downscale_to_replay_median": float(np.median([b["rho_lo"] / b["a"] for b in br])),
    }
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for name, r in out.items():
        sf = r["signflip"]
        print(f"{name}: D={r['mean_D']:+.4f} sd={r['sd_D']:.4f} L={r['L']:+.4f} U={r['U']:+.4f} "
              f"null med={sf['null_mean_D']['median']:+.4f} P99={sf['null_mean_D']['p99']:+.4f} "
              f"p={sf['p_mean']:.5f} pdisc={r['p_disc_post']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
