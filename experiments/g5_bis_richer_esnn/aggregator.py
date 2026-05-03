"""Cross-substrate H7-A/B/C aggregator for the G5-bis pilot.

Loads a G4-ter MLX richer-head milestone (key
``verdict.retention_richer_by_arm``) and a G5-bis E-SNN richer-head
milestone (key ``verdict.retention_by_arm``), runs four Welch
two-sided per-arm consistency tests at Bonferroni alpha/4 = 0.0125,
classifies the cross-substrate outcome into one of three
pre-registered hypotheses :

- **H7-A** : positive transfer with level-divergence — own-substrate
  ``g_h7a >= 0.5`` AND own-substrate Welch rejects, but P_equ
  cross-substrate consistency is False (means differ across MLX
  and E-SNN).
- **H7-B** : MLX-only artefact — ``|g_h7a| < 0.5`` AND own-substrate
  Welch fails to reject.
- **H7-C** : universal cross-substrate — ``g_h7a >= 0.5`` AND
  own-substrate Welch rejects AND P_equ cross-substrate
  consistency is True (level parity within Welch tolerance).
- ``ambiguous`` : everything else.

Reference :
    docs/osf-prereg-g5-bis-richer-esnn.md sec 1
    docs/proofs/dr3-substrate-evidence.md
    experiments/g5_cross_substrate/aggregator.py (sister)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kiki_oniric.eval.statistics import compute_hedges_g, welch_one_sided


REQUIRED_ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
ALPHA_PER_ARM = 0.05 / 4  # Bonferroni across 4 arms
H7B_G_THRESHOLD = 0.5  # decision knob (see pre-reg sec 1)


def _load_retention(
    milestone_path: Path, key: str
) -> dict[str, list[float]]:
    """Load ``verdict[key]`` as a {arm: [retention floats]} dict.

    Validates the dict carries every arm in ``REQUIRED_ARMS`` ;
    raises ValueError otherwise.
    """
    payload = json.loads(milestone_path.read_text())
    retention = payload.get("verdict", {}).get(key)
    if not isinstance(retention, dict):
        raise ValueError(
            f"milestone {milestone_path} missing verdict.{key}"
        )
    for arm in REQUIRED_ARMS:
        if arm not in retention:
            raise ValueError(
                f"milestone {milestone_path} missing arm {arm!r} "
                f"in verdict.{key}"
            )
    return {
        arm: [float(v) for v in retention[arm]] for arm in REQUIRED_ARMS
    }


def _welch_two_sided(
    a: list[float], b: list[float], alpha: float
) -> tuple[float, bool]:
    """Run Welch one-sided in both directions, return (p_two_sided, reject)."""
    welch_a = welch_one_sided(a, b, alpha=alpha)
    welch_b = welch_one_sided(b, a, alpha=alpha)
    p_two_sided = min(2.0 * min(welch_a.p_value, welch_b.p_value), 1.0)
    reject = bool(p_two_sided < alpha)
    return p_two_sided, reject


def aggregate_g5bis_verdict(
    mlx_milestone: Path, esnn_milestone: Path
) -> dict[str, Any]:
    """Compute per-arm cross-substrate Welch + H7-A/B/C classification."""
    mlx = _load_retention(mlx_milestone, "retention_richer_by_arm")
    esnn = _load_retention(esnn_milestone, "retention_by_arm")

    per_arm: dict[str, dict[str, Any]] = {}
    for arm in REQUIRED_ARMS:
        mlx_vals = mlx[arm]
        esnn_vals = esnn[arm]
        if len(mlx_vals) < 2 or len(esnn_vals) < 2:
            per_arm[arm] = {
                "insufficient_samples": True,
                "n_mlx": len(mlx_vals),
                "n_esnn": len(esnn_vals),
            }
            continue
        g = compute_hedges_g(mlx_vals, esnn_vals)
        p_two_sided, reject = _welch_two_sided(
            mlx_vals, esnn_vals, ALPHA_PER_ARM
        )
        per_arm[arm] = {
            "hedges_g_mlx_minus_esnn": g,
            "welch_p_two_sided": p_two_sided,
            "reject_h0": reject,
            "consistency": not reject,
            "n_mlx": len(mlx_vals),
            "n_esnn": len(esnn_vals),
        }

    # E-SNN own-substrate effect (P_equ vs baseline)
    if len(esnn["P_equ"]) < 2 or len(esnn["baseline"]) < 2:
        g_h7a = 0.0
        welch_p = 1.0
        welch_reject = False
        own_insufficient = True
    else:
        g_h7a = compute_hedges_g(esnn["P_equ"], esnn["baseline"])
        welch_h7a = welch_one_sided(
            esnn["baseline"], esnn["P_equ"], alpha=ALPHA_PER_ARM
        )
        welch_p = welch_h7a.p_value
        welch_reject = bool(welch_h7a.reject_h0)
        own_insufficient = False

    # Decision rule (locked, pre-reg sec 1)
    p_equ_row = per_arm.get("P_equ", {})
    p_equ_consistency = p_equ_row.get("consistency")

    classification: str
    if own_insufficient:
        classification = "ambiguous"
    elif (
        abs(g_h7a) < H7B_G_THRESHOLD and not welch_reject
    ):
        classification = "H7-B"
    elif (
        g_h7a >= H7B_G_THRESHOLD
        and welch_reject
        and p_equ_consistency is True
    ):
        classification = "H7-C"
    elif g_h7a >= H7B_G_THRESHOLD and welch_reject:
        classification = "H7-A"
    else:
        classification = "ambiguous"

    return {
        "per_arm": per_arm,
        "h7_classification": classification,
        "g_h7a_esnn": g_h7a,
        "g_h7a_welch_p": welch_p,
        "g_h7a_welch_reject_h0": welch_reject,
        "alpha_per_arm": ALPHA_PER_ARM,
        "h7b_g_threshold": H7B_G_THRESHOLD,
        "mlx_milestone": str(mlx_milestone),
        "esnn_milestone": str(esnn_milestone),
    }


def _render_md(verdict: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        "# G5-bis cross-substrate aggregate - H7-A/B/C verdict"
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
    lines.append(
        f"**H7-B g threshold** : {verdict['h7b_g_threshold']:.2f}"
    )
    lines.append("")
    classification = verdict["h7_classification"]
    lines.append(f"## Verdict : {classification}")
    lines.append("")
    lines.append(
        f"- Observed E-SNN g_h7a (P_equ vs baseline) : "
        f"**{verdict['g_h7a_esnn']:+.4f}**"
    )
    lines.append(
        f"- Welch one-sided p (alpha/4 = "
        f"{verdict['alpha_per_arm']:.4f}) : "
        f"{verdict['g_h7a_welch_p']:.4f}"
    )
    lines.append(
        f"- reject_h0 (own-substrate) : "
        f"{verdict['g_h7a_welch_reject_h0']}"
    )
    lines.append("")
    lines.append("## Per-arm cross-substrate Welch consistency")
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
        "- Aggregator : `experiments/g5_bis_richer_esnn/aggregator.py`"
    )
    lines.append(
        "- Pre-registration : `docs/osf-prereg-g5-bis-richer-esnn.md`"
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
    verdict = aggregate_g5bis_verdict(mlx_milestone, esnn_milestone)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    out_md.write_text(_render_md(verdict))
    return verdict
