"""G4-quater aggregator — load 3 step milestones and emit verdicts.

Per pre-reg sec 2-3 :

- **H4-A** : Jonckheere on step1 retention by arm, alpha = 0.0167.
- **H4-B** : per-factor Jonckheere on step2 retention,
  multiplicity-adjusted alpha = 0.0056.
- **H4-C** : Welch two-sided between step3 (P_max with mog) and
  (P_max with none) at alpha = 0.0167. **Failure to reject** ->
  H4-C confirmed (RECOMBINE empty).

Outputs :
    docs/milestones/g4-quater-aggregate-2026-05-03.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_STEP1 = (
    REPO_ROOT / "docs" / "milestones" / "g4-quater-step1-2026-05-03.json"
)
DEFAULT_STEP2 = (
    REPO_ROOT / "docs" / "milestones" / "g4-quater-step2-2026-05-03.json"
)
DEFAULT_STEP3 = (
    REPO_ROOT / "docs" / "milestones" / "g4-quater-step3-2026-05-03.json"
)
DEFAULT_OUT_JSON = (
    REPO_ROOT
    / "docs"
    / "milestones"
    / "g4-quater-aggregate-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT
    / "docs"
    / "milestones"
    / "g4-quater-aggregate-2026-05-03.md"
)


def aggregate_g4_quater_verdict(
    step1_path: Path,
    step2_path: Path,
    step3_path: Path,
) -> dict[str, Any]:
    """Load the three step milestones and return aggregate verdicts."""
    with step1_path.open() as fh:
        s1 = json.load(fh)
    with step2_path.open() as fh:
        s2 = json.load(fh)
    with step3_path.open() as fh:
        s3 = json.load(fh)

    h4a = s1["verdict"]["h4a_deeper_substrate"]
    h4b_per_factor = s2["verdict"]["h4b_per_factor"]
    h4b_any = s2["verdict"]["any_factor_recovers_ordering"]
    h4c = s3["verdict"]["h4c_recombine_strategy"]
    ae_obs = s3["verdict"]["ae_observation"]

    h4a_confirmed = (
        not h4a.get("insufficient_samples", False)
        and bool(h4a.get("reject_h0"))
        and bool(h4a.get("monotonic_observed"))
    )
    h4b_confirmed = bool(h4b_any)
    h4c_confirmed = (
        not h4c.get("insufficient_samples", False)
        and bool(h4c.get("h4c_recombine_empty_confirmed"))
    )

    return {
        "h4a_substrate_depth": {
            **h4a,
            "confirmed": h4a_confirmed,
        },
        "h4b_hp_calibration": {
            "per_factor": h4b_per_factor,
            "any_factor_recovers_ordering": h4b_any,
            "confirmed": h4b_confirmed,
        },
        "h4c_recombine_emptiness": {
            **h4c,
            "ae_secondary": ae_obs,
            "confirmed": h4c_confirmed,
        },
        "summary": {
            "h4a_confirmed": h4a_confirmed,
            "h4b_confirmed": h4b_confirmed,
            "h4c_confirmed": h4c_confirmed,
            "any_confirmed": (
                h4a_confirmed or h4b_confirmed or h4c_confirmed
            ),
            "all_three_confirmed": (
                h4a_confirmed and h4b_confirmed and h4c_confirmed
            ),
        },
    }


def _render_md(verdict: dict[str, Any]) -> str:
    h4a = verdict["h4a_substrate_depth"]
    h4b = verdict["h4b_hp_calibration"]
    h4c = verdict["h4c_recombine_emptiness"]
    s = verdict["summary"]

    lines: list[str] = [
        "# G4-quater aggregate verdict",
        "",
        "**Date** : 2026-05-03",
        "**Pre-registration** : "
        "[docs/osf-prereg-g4-quater-pilot.md]"
        "(../osf-prereg-g4-quater-pilot.md)",
        "",
        "## Summary",
        "",
        f"- H4-A (substrate-depth) confirmed : **{s['h4a_confirmed']}**",
        f"- H4-B (HP-calibration) confirmed : **{s['h4b_confirmed']}**",
        (
            "- H4-C (RECOMBINE empirically empty) confirmed : "
            f"**{s['h4c_confirmed']}**"
        ),
        f"- All three confirmed : {s['all_three_confirmed']}",
        "",
        "## H4-A — substrate-depth",
        "",
    ]
    if h4a.get("insufficient_samples"):
        lines.append("INSUFFICIENT SAMPLES")
    else:
        lines += [
            f"- mean P_min : {h4a['mean_p_min']:.4f}",
            f"- mean P_equ : {h4a['mean_p_equ']:.4f}",
            f"- mean P_max : {h4a['mean_p_max']:.4f}",
            f"- monotonic_observed : {h4a['monotonic_observed']}",
            f"- Jonckheere J : {h4a['j_statistic']:.4f}",
            (
                f"- one-sided p (alpha = "
                f"{h4a['alpha_per_test']:.4f}) : "
                f"{h4a['p_value']:.4f}"
            ),
            f"- reject_h0 : {h4a['reject_h0']}",
            f"- **H4-A confirmed** : {h4a['confirmed']}",
        ]
    lines += [
        "",
        "## H4-B — HP-calibration (RESTRUCTURE factor sweep)",
        "",
        f"any_factor_recovers_ordering : {h4b['any_factor_recovers_ordering']}",
        "",
    ]
    for entry in h4b["per_factor"]:
        lines.append(f"### factor = {entry['factor']}")
        if entry.get("insufficient_samples"):
            lines.append("INSUFFICIENT SAMPLES")
        else:
            lines += [
                f"- mean P_min : {entry['mean_p_min']:.4f}",
                f"- mean P_equ : {entry['mean_p_equ']:.4f}",
                f"- mean P_max : {entry['mean_p_max']:.4f}",
                f"- monotonic_observed : {entry['monotonic_observed']}",
                f"- Jonckheere J : {entry['j_statistic']:.4f}",
                (
                    f"- p (alpha = {entry['alpha_per_test']:.4f}) : "
                    f"{entry['p_value']:.4f}"
                ),
                f"- reject_h0 : {entry['reject_h0']}",
            ]
        lines.append("")
    lines += [
        f"**H4-B confirmed** : {h4b['confirmed']}",
        "",
        "## H4-C — RECOMBINE empirical-emptiness",
        "",
    ]
    if h4c.get("insufficient_samples"):
        lines.append("INSUFFICIENT SAMPLES")
    else:
        lines += [
            f"- mean P_max (mog) : {h4c['mean_p_max_mog']:.4f}",
            f"- mean P_max (none) : {h4c['mean_p_max_none']:.4f}",
            f"- Hedges' g (mog vs none) : {h4c['hedges_g_mog_vs_none']:.4f}",
            f"- Welch t : {h4c['welch_t']:.4f}",
            (
                f"- Welch p two-sided (alpha = "
                f"{h4c['alpha_per_test']:.4f}) : "
                f"{h4c['welch_p_two_sided']:.4f}"
            ),
            (
                f"- fail_to_reject_h0 : "
                f"{h4c['fail_to_reject_h0']} -> "
                f"H4-C confirmed = {h4c['confirmed']}"
            ),
            "",
            (
                "*Honest reading* : Welch fail-to-reject = no evidence "
                "at this N for a difference between mog and none ; the "
                "predicted outcome under H4-C, framed as positive "
                "empirical claim that RECOMBINE adds nothing beyond "
                "the REPLAY+DOWNSCALE coupling already provided by "
                "P_min."
            ),
        ]
    ae = h4c.get("ae_secondary", {})
    lines += ["", "### Secondary observation — AE strategy"]
    if ae.get("insufficient_samples"):
        lines.append("INSUFFICIENT SAMPLES")
    else:
        lines += [
            f"- mean P_max (ae) : {ae['mean_p_max_ae']:.4f}",
            f"- mean P_max (none) : {ae['mean_p_max_none']:.4f}",
            f"- Welch p two-sided : {ae['welch_p_two_sided']:.4f}",
        ]
    lines += [
        "",
        "## Verdict — DR-4 evidence",
        "",
        (
            "Per pre-reg §6 : EC stays PARTIAL across all outcomes ; "
            "FC stays at C-v0.12.0. If H4-C is confirmed, the partial "
            "refutation of DR-4 established by G4-ter is "
            "**strengthened**, not weakened."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G4-quater aggregator")
    parser.add_argument("--step1", type=Path, default=DEFAULT_STEP1)
    parser.add_argument("--step2", type=Path, default=DEFAULT_STEP2)
    parser.add_argument("--step3", type=Path, default=DEFAULT_STEP3)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)

    verdict = aggregate_g4_quater_verdict(
        args.step1, args.step2, args.step3
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    args.out_md.write_text(_render_md(verdict))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    s = verdict["summary"]
    print(
        f"H4-A : {s['h4a_confirmed']}  "
        f"H4-B : {s['h4b_confirmed']}  "
        f"H4-C : {s['h4c_confirmed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
