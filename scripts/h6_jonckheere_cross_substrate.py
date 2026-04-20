"""H6 profile ordering cross-substrate Jonckheere–Terpstra test.

.. note::

    **DEFERRED to Paper 2** (PLOS CB pivot, 2026-04-20) along with
    the C3.13 driver (see :mod:`scripts.pilot_phase2b_neuromorph`).
    This analyzer is preserved as the Paper 2 reactivation entry
    point for the H6 hypothesis test on cross-substrate profile
    gradient ordering.

**Gate ID** : G10e — H6 cross-substrate profile ordering
**Validates** : whether each substrate (MLX numpy-equivalent and
Norse SNN proxy) exhibits a monotonic ordering of the
(``p_min``, ``p_equ``, ``p_max``) profile sequence on the
``delta_norm`` observable, via the Jonckheere–Terpstra trend test.
**Mode** : pipeline-validation — **synthetic 4×4 matrix**, not a
real Qwen model. Production validation lives in the Phase B real
pilot (1.5B Qwen FP16). This pilot is a pipeline-validation
artifact for the H6 ordering hypothesis under DR-3 condition (3)
cross-substrate observability only.

## Hypothesis

H0_6 : the three profile-conditional distributions of ``delta_norm``
       are stochastically identical across profiles (no monotonic
       ordering).
H1_6 : there exists a strictly monotonic ordering
       (``p_min`` < ``p_equ`` < ``p_max``) or the reverse
       (``p_min`` > ``p_equ`` > ``p_max``) on ``delta_norm``.

The task brief (goal e) permits either direction provided it is
OSF pre-registration compatible. We test the **two-sided**
Jonckheere statistic normalised by its null-mean and null-std and
reject when the absolute z exceeds 1.96 (two-sided α=0.05).

## Inputs

Consumes ``docs/milestones/g10a-neuromorph.json`` produced by
:mod:`scripts.pilot_phase2b_neuromorph` (G10a driver). The JSON
must contain three profiles and the two substrates.

## Outputs

- ``docs/milestones/h6-cross-substrate-ordering.md`` (human)
- ``docs/milestones/h6-cross-substrate-ordering.json`` (machine)

Usage ::

    uv run python scripts/h6_jonckheere_cross_substrate.py

Reference :
    docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md
    §3 Phase 2 track b (H6 ordering test per substrate)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
G10A_JSON = REPO_ROOT / "docs" / "milestones" / "g10a-neuromorph.json"
MILESTONE_MD = (
    REPO_ROOT / "docs" / "milestones" / "h6-cross-substrate-ordering.md"
)
MILESTONE_JSON = (
    REPO_ROOT / "docs" / "milestones" / "h6-cross-substrate-ordering.json"
)

PROFILE_ORDER: tuple[str, ...] = ("p_min", "p_equ", "p_max")
SUBSTRATES: tuple[str, ...] = ("mlx", "norse")


def jonckheere_terpstra(
    groups: list[np.ndarray],
) -> dict:
    """Compute the Jonckheere–Terpstra ordered trend statistic.

    Parameters
    ----------
    groups :
        Ordered list of 1-D arrays ; the test checks for a monotonic
        increase (or, via two-sided p-value, decrease) across the
        sequence of groups.

    Returns
    -------
    dict with :
      - ``J``         : raw Jonckheere statistic
      - ``mean_null`` : null-distribution mean
      - ``var_null``  : null-distribution variance
      - ``z``         : standardised statistic
      - ``p_two_sided``: two-sided asymptotic normal p-value
      - ``p_one_sided_gt`` : one-sided p-value for monotone increase
      - ``p_one_sided_lt`` : one-sided p-value for monotone decrease
      - ``n_per_group``    : sizes per group
      - ``total_n``   : total sample size

    Ties handled via the standard 0.5 weighting on within-pair ties
    (same as scipy.stats.mannwhitneyu method='asymptotic').
    """
    k = len(groups)
    if k < 3:
        raise ValueError("Jonckheere–Terpstra requires at least 3 groups")

    n_per_group = [int(g.size) for g in groups]
    total_n = sum(n_per_group)

    # Raw statistic : sum over all i<j of U(groups[i], groups[j])
    # where U counts pairs (x in group_i, y in group_j) with x < y
    # plus 0.5 × number of tied pairs.
    J = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            gi = groups[i]
            gj = groups[j]
            # Broadcast comparison ; small-group-friendly (≤30 each).
            diff = gj[None, :] - gi[:, None]
            J += float(np.sum(diff > 0)) + 0.5 * float(np.sum(diff == 0))

    # Null distribution (no ties approximation ; with small number
    # of ties, still a reasonable asymptotic).
    mean_null = (
        total_n * total_n - sum(n * n for n in n_per_group)
    ) / 4.0
    var_null = (
        total_n * total_n * (2 * total_n + 3)
        - sum(n * n * (2 * n + 3) for n in n_per_group)
    ) / 72.0

    if var_null <= 0.0:
        z = 0.0
        p_two_sided = 1.0
        p_gt = 0.5
        p_lt = 0.5
    else:
        z = (J - mean_null) / math.sqrt(var_null)
        # Standard normal CDF via erf
        p_gt = 0.5 * math.erfc(z / math.sqrt(2.0))
        p_lt = 1.0 - p_gt
        p_two_sided = 2.0 * min(p_gt, p_lt)

    return {
        "J": J,
        "mean_null": mean_null,
        "var_null": var_null,
        "z": z,
        "p_two_sided": p_two_sided,
        "p_one_sided_gt": p_gt,
        "p_one_sided_lt": p_lt,
        "n_per_group": n_per_group,
        "total_n": total_n,
    }


def _extract_delta_by_profile(
    g10a_json: dict, substrate: str
) -> dict[str, np.ndarray]:
    """Collect ``delta_norm`` per profile from G10a cell dumps."""
    cells = g10a_json["cells"]
    out: dict[str, list[float]] = {p: [] for p in PROFILE_ORDER}
    for c in cells:
        if c["substrate"] != substrate:
            continue
        if c["profile"] not in out:
            continue
        out[c["profile"]].append(float(c["delta_norm"]))
    return {p: np.asarray(v, dtype=np.float64) for p, v in out.items()}


def _render_markdown(dump: dict) -> str:
    lines: list[str] = []
    lines.append(
        "# G10e / H6 — Cross-substrate profile ordering (2026-04-19)"
    )
    lines.append("")
    lines.append(f"**Status** : **{dump['verdict']}**")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "- Hypothesis H6 : each substrate exhibits a strictly "
        "monotonic ordering of ``delta_norm`` across profiles "
        "(``p_min``, ``p_equ``, ``p_max``), tested via the "
        "Jonckheere–Terpstra trend statistic."
    )
    lines.append(
        "- Data source : "
        "``docs/milestones/g10a-neuromorph.json`` (30 seeds × 3 "
        "profiles × 2 substrates = 180 cells)."
    )
    lines.append(
        "- Decision rule : reject H0_6 when "
        "``|z| > 1.96`` (two-sided α = 0.05)."
    )
    lines.append(
        "- Cross-substrate claim : H6 holds substrate-agnostically "
        "when **both** substrates reject H0_6 in the same direction."
    )
    lines.append("")
    lines.append(
        "> **Synthetic caveat** (CLAUDE.md §3) : results below are "
        "on a 4×4 synthetic weight matrix, **not a real model**. "
        "Production H6 validation lives in Paper 2 with real-scale "
        "Qwen pilots. This pilot is a pipeline-validation artifact "
        "for DR-3 condition (3) cross-substrate observability only."
    )
    lines.append("")
    lines.append("## Per-substrate group statistics")
    lines.append("")
    lines.append(
        "| Substrate | Profile | n | mean Δ | std Δ | median Δ |"
    )
    lines.append("|---|---|---|---|---|---|")
    for substrate in SUBSTRATES:
        for profile in PROFILE_ORDER:
            s = dump["group_stats"][substrate][profile]
            lines.append(
                f"| {substrate} | {profile} | {s['n']} | "
                f"{s['mean']:.4f} | {s['std']:.4f} | "
                f"{s['median']:.4f} |"
            )
    lines.append("")
    lines.append("## Jonckheere–Terpstra per substrate")
    lines.append("")
    lines.append(
        "| Substrate | J | z | p (two-sided) | p (↑) | p (↓) | "
        "Reject H0_6 | Direction |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for substrate in SUBSTRATES:
        r = dump["jonckheere"][substrate]
        direction = r["direction"] if r["reject_h0"] else "—"
        lines.append(
            f"| {substrate} | {r['J']:.1f} | {r['z']:+.3f} | "
            f"{r['p_two_sided']:.3g} | "
            f"{r['p_one_sided_gt']:.3g} | "
            f"{r['p_one_sided_lt']:.3g} | "
            f"{'YES' if r['reject_h0'] else 'NO'} | "
            f"{direction} |"
        )
    lines.append("")
    lines.append("## Cross-substrate H6 verdict")
    lines.append("")
    lines.append(
        f"- MLX rejects H0_6 : "
        f"{'YES' if dump['jonckheere']['mlx']['reject_h0'] else 'NO'}"
    )
    lines.append(
        f"- Norse rejects H0_6 : "
        f"{'YES' if dump['jonckheere']['norse']['reject_h0'] else 'NO'}"
    )
    if dump["both_reject"] and dump["same_direction"]:
        claim = (
            "**H6 HOLDS substrate-agnostically** — both substrates "
            f"exhibit a monotonic ordering in the `{dump['common_direction']}` "
            "direction."
        )
    elif dump["both_reject"] and not dump["same_direction"]:
        claim = (
            "**H6 PARTIAL** — both substrates reject H0_6 but the "
            "monotonic direction differs (confound : substrate-"
            "specific effect direction)."
        )
    elif dump["jonckheere"]["mlx"]["reject_h0"] or dump[
        "jonckheere"
    ]["norse"]["reject_h0"]:
        claim = (
            "**H6 PARTIAL** — only one substrate rejects H0_6 ; the "
            "substrate-agnostic claim does not hold on this pilot."
        )
    else:
        claim = (
            "**H6 FAILS** — no substrate exhibits a significant "
            "monotonic ordering on ``delta_norm``."
        )
    lines.append("")
    lines.append(claim)
    lines.append("")
    lines.append("## References")
    lines.append("")
    lines.append(
        "- G10a driver : ``scripts/pilot_phase2b_neuromorph.py`` "
        "(C3.13 ; produces the ``delta_norm`` distributions consumed "
        "here)."
    )
    lines.append(
        "- Jonckheere (1954), Terpstra (1952) ordered-alternatives "
        "rank test ; asymptotic normal approximation used."
    )
    lines.append(
        "- Cross-substrate data : "
        "``docs/milestones/g10a-neuromorph.json``."
    )
    lines.append(
        "- Spec : "
        "``docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-"
        "design.md`` §3 Phase 2 track b (H6 ordering)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if not G10A_JSON.exists():
        raise SystemExit(
            f"missing G10a dump at {G10A_JSON!s} — run "
            "``uv run python scripts/pilot_phase2b_neuromorph.py`` first."
        )
    g10a = json.loads(G10A_JSON.read_text())

    group_stats: dict[str, dict] = {}
    jonckheere: dict[str, dict] = {}
    for substrate in SUBSTRATES:
        by_profile = _extract_delta_by_profile(g10a, substrate)
        group_stats[substrate] = {}
        for profile in PROFILE_ORDER:
            arr = by_profile[profile]
            group_stats[substrate][profile] = {
                "n": int(arr.size),
                "mean": float(arr.mean()) if arr.size else 0.0,
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "median": float(np.median(arr)) if arr.size else 0.0,
            }
        ordered = [by_profile[p] for p in PROFILE_ORDER]
        jt = jonckheere_terpstra(ordered)
        reject = bool(abs(jt["z"]) > 1.96)
        direction = "increasing" if jt["z"] > 0 else "decreasing"
        jt["reject_h0"] = reject
        jt["direction"] = direction
        jonckheere[substrate] = jt

    both_reject = all(
        jonckheere[s]["reject_h0"] for s in SUBSTRATES
    )
    directions = {
        s: jonckheere[s]["direction"] for s in SUBSTRATES
    }
    same_direction = len(set(directions.values())) == 1
    common_direction = (
        directions["mlx"] if same_direction else "mixed"
    )

    if both_reject and same_direction:
        verdict = "SOFT-GO"
    elif both_reject and not same_direction:
        verdict = "PARTIAL"
    elif any(jonckheere[s]["reject_h0"] for s in SUBSTRATES):
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"

    dump = {
        "gate": "G10e",
        "milestone": "H6 cross-substrate profile ordering",
        "harness_version": g10a.get("harness_version", "unknown"),
        "source_g10a": str(
            G10A_JSON.relative_to(REPO_ROOT)
        ),
        "profiles": list(PROFILE_ORDER),
        "substrates": list(SUBSTRATES),
        "group_stats": group_stats,
        "jonckheere": jonckheere,
        "both_reject": both_reject,
        "same_direction": same_direction,
        "common_direction": common_direction,
        "verdict": verdict,
        "alpha_two_sided": 0.05,
        "z_threshold": 1.96,
    }

    MILESTONE_JSON.write_text(json.dumps(dump, indent=2, sort_keys=True))
    MILESTONE_MD.write_text(_render_markdown(dump))

    print(f"[G10e] verdict = {verdict}")
    for substrate in SUBSTRATES:
        r = jonckheere[substrate]
        print(
            f"[G10e] {substrate}: J={r['J']:.1f} z={r['z']:+.3f} "
            f"p2={r['p_two_sided']:.3g} reject={r['reject_h0']} "
            f"direction={r['direction']}"
        )
    print(
        f"[G10e] both_reject={both_reject} same_direction={same_direction} "
        f"common={common_direction}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
