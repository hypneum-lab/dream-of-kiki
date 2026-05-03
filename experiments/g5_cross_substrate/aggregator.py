"""Cross-substrate aggregator for the G5 pilot.

Loads a G4-bis MLX milestone and a G5 E-SNN milestone, runs four
Welch one-sided consistency tests (one per arm) at Bonferroni
α/4 = 0.0125, and emits a cross-substrate verdict. The verdict
upgrades DR-3 evidence in `docs/proofs/dr3-substrate-evidence.md`
when `dr3_cross_substrate_consistency_ok = True`.

Statistical model ::

    For each arm a in {baseline, P_min, P_equ, P_max} :
        H0 : mean(MLX retention[a]) == mean(E-SNN retention[a])
        H1 : means differ
        Test : two-sided Welch via min(2 * min(p_left, p_right), 1)
        Hedges' g = compute_hedges_g(MLX[a], E-SNN[a]).
        consistency[a] = not welch_reject_h0

    DR-3 cross-substrate verdict :
        all(consistency[a] for a in arms) -> consistency_ok = True
        any consistency[a] = False -> divergence finding (verdict
        records which arm diverged + observed g + p).

Reference :
    docs/proofs/dr3-substrate-evidence.md
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kiki_oniric.eval.statistics import compute_hedges_g, welch_one_sided


REQUIRED_ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
ALPHA_PER_ARM = 0.05 / 4  # Bonferroni across 4 arms


def _load_retention_by_arm(
    milestone_path: Path,
) -> dict[str, list[float]]:
    payload = json.loads(milestone_path.read_text())
    retention = payload.get("verdict", {}).get("retention_by_arm")
    if not isinstance(retention, dict):
        raise ValueError(
            f"milestone {milestone_path} missing verdict.retention_by_arm"
        )
    for arm in REQUIRED_ARMS:
        if arm not in retention:
            raise ValueError(
                f"milestone {milestone_path} missing arm {arm!r} in "
                f"verdict.retention_by_arm"
            )
    return {arm: list(map(float, retention[arm])) for arm in REQUIRED_ARMS}


def aggregate_cross_substrate_verdict(
    mlx_milestone: Path, esnn_milestone: Path
) -> dict[str, Any]:
    """Compute the per-arm Welch consistency tests + aggregate verdict.

    Returns a dict with :
        - per_arm : {arm: {hedges_g, welch_p, consistency, n_mlx, n_esnn}}
        - dr3_cross_substrate_consistency_ok : bool
        - alpha_per_arm : float (= 0.0125)
        - mlx_milestone : str (path)
        - esnn_milestone : str (path)
    """
    mlx = _load_retention_by_arm(mlx_milestone)
    esnn = _load_retention_by_arm(esnn_milestone)

    per_arm: dict[str, dict[str, Any]] = {}
    all_consistent = True
    for arm in REQUIRED_ARMS:
        mlx_vals = mlx[arm]
        esnn_vals = esnn[arm]
        if len(mlx_vals) < 2 or len(esnn_vals) < 2:
            per_arm[arm] = {
                "insufficient_samples": True,
                "n_mlx": len(mlx_vals),
                "n_esnn": len(esnn_vals),
            }
            all_consistent = False
            continue
        g = compute_hedges_g(mlx_vals, esnn_vals)
        # Two-sided fold : run both directions, take the smaller p
        welch_a = welch_one_sided(mlx_vals, esnn_vals, alpha=ALPHA_PER_ARM)
        welch_b = welch_one_sided(esnn_vals, mlx_vals, alpha=ALPHA_PER_ARM)
        p_two_sided = min(
            2.0 * min(welch_a.p_value, welch_b.p_value), 1.0
        )
        reject = bool(p_two_sided < ALPHA_PER_ARM)
        consistency = not reject
        per_arm[arm] = {
            "hedges_g_mlx_minus_esnn": g,
            "welch_p_two_sided": p_two_sided,
            "reject_h0": reject,
            "consistency": consistency,
            "n_mlx": len(mlx_vals),
            "n_esnn": len(esnn_vals),
        }
        if not consistency:
            all_consistent = False

    return {
        "per_arm": per_arm,
        "dr3_cross_substrate_consistency_ok": all_consistent,
        "alpha_per_arm": ALPHA_PER_ARM,
        "mlx_milestone": str(mlx_milestone),
        "esnn_milestone": str(esnn_milestone),
    }


def _render_md(verdict: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        "# G5 cross-substrate aggregate — DR-3 cross-substrate consistency"
    )
    lines.append("")
    lines.append("**Date** : 2026-05-03")
    lines.append(
        f"**MLX milestone** : `{verdict['mlx_milestone']}`"
    )
    lines.append(
        f"**E-SNN milestone** : `{verdict['esnn_milestone']}`"
    )
    lines.append(
        f"**Bonferroni alpha / 4** : {verdict['alpha_per_arm']:.4f}"
    )
    lines.append("")
    ok = verdict["dr3_cross_substrate_consistency_ok"]
    if ok:
        lines.append(
            "## Verdict : DR-3 cross-substrate consistency CONFIRMED"
        )
        lines.append("")
        lines.append(
            "All four arms (baseline, P_min, P_equ, P_max) show "
            "Welch p > alpha/4 = 0.0125 — cross-substrate distributions "
            "are statistically indistinguishable at first-pilot scale "
            "(N=5 per cell)."
        )
    else:
        lines.append(
            "## Verdict : DR-3 cross-substrate divergence DETECTED"
        )
        lines.append("")
        lines.append(
            "At least one arm shows Welch p <= alpha/4 = 0.0125 — see "
            "per-arm table below for the diverging arm(s)."
        )
    lines.append("")
    lines.append("## Per-arm Welch consistency")
    lines.append("")
    lines.append(
        "| arm | g (MLX - E-SNN) | Welch p (two-sided) | "
        "reject H0 | consistent |"
    )
    lines.append(
        "|-----|------------------|----------------------|"
        "-----------|------------|"
    )
    for arm in REQUIRED_ARMS:
        row = verdict["per_arm"][arm]
        if row.get("insufficient_samples"):
            lines.append(
                f"| {arm} | INSUFFICIENT | INSUFFICIENT | n/a | False |"
            )
            continue
        lines.append(
            f"| {arm} | {row['hedges_g_mlx_minus_esnn']:+.4f} | "
            f"{row['welch_p_two_sided']:.4f} | "
            f"{row['reject_h0']} | {row['consistency']} |"
        )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(
        "- DR-3 spec : "
        "`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §6.2"
    )
    lines.append(
        "- DR-3 evidence record : `docs/proofs/dr3-substrate-evidence.md`"
    )
    lines.append(
        "- Aggregator : `experiments/g5_cross_substrate/aggregator.py`"
    )
    lines.append("")
    return "\n".join(lines)


def write_aggregate_dump(
    *,
    mlx_milestone: Path,
    esnn_milestone: Path,
    out_json: Path,
    out_md: Path,
) -> dict[str, Any]:
    """Compute the verdict and persist `.json` + `.md` siblings."""
    verdict = aggregate_cross_substrate_verdict(
        mlx_milestone, esnn_milestone
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    out_md.write_text(_render_md(verdict))
    return verdict
