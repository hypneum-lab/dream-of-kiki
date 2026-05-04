"""G4-octavo aggregator — H7-A verdict + H7-B universality-extension flag.

Per pre-reg §2 :

- **H7-A** : Welch two-sided between (P_max with mog) and
  (P_max with none) on Split-Tiny-ImageNet with the ViT-tiny
  substrate, alpha = 0.05 (single new test ; no Bonferroni
  inheritance from the closed G4-{quater..septimo} cycle).
  **Rejecting** H0 with the predicted positive sign
  ``mean(mog) > mean(none)`` *and* Hedges' g >= 0.5 is the H7-A
  positive empirical claim. This **inverts** the directional
  reading of the closed G4-{quater..septimo} ladder, where
  fail-to-reject was the predicted positive claim. Four resolution
  states :

    - ``confirmed_positive`` — rejected with positive sign and
      g >= 0.5 (Row 1 ; framework's RECOMBINE prediction restored
      at the small-transformer tier).
    - ``confirmed_subthreshold`` — rejected with positive sign but
      g < 0.5 (Row 4 ; reported exploratory ; H7-B not resolved).
    - ``confirmed_negative`` — rejected with negative sign (Row 3 ;
      RECOMBINE *hurts* on the ViT-tiny substrate ; framework's
      claim further weakened).
    - ``null`` — fail-to-reject (Row 2 ; H7-B universality-extension
      flag fires : the closed CNN-or-MLP scope ceiling extends to
      the small-transformer tier).

- **H7-B** : derived universality-extension flag, no additional
  Welch test (logical aggregation per pre-reg §2). Three resolution
  states :

    - ``universality_extends`` — H7-A null ; the closed
      G4-{quater..septimo} four-benchmark × four-CNN-or-MLP
      universality flag extends to include the small-transformer
      tier (Tiny-ImageNet 200-class with ViT-tiny). DR-4 evidence
      v0.7 logs the scope extension.
    - ``universality_breaks`` — H7-A confirmed_positive ; the
      universality is shown to break at the transformer
      architecture and the framework's RECOMBINE prediction is
      restored at the transformer tier. DR-4 evidence v0.7
      records the architectural escape.
    - ``unresolved`` — H7-A confirmed_subthreshold or
      confirmed_negative or insufficient_samples ; H7-B is not
      resolved by this pilot.

Outputs :
    docs/milestones/g4-octavo-aggregate-2026-05-04.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_STEP1 = (
    REPO_ROOT / "docs" / "milestones" / "g4-octavo-step1-2026-05-04.json"
)
DEFAULT_OUT_JSON = (
    REPO_ROOT
    / "docs"
    / "milestones"
    / "g4-octavo-aggregate-2026-05-04.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT
    / "docs"
    / "milestones"
    / "g4-octavo-aggregate-2026-05-04.md"
)

HONEST_READING_H7A = (
    "H7-A confirmation requires Welch *rejecting* H0 with positive "
    "sign **and** Hedges' g >= 0.5 (both conditions). Sub-threshold "
    "positive (rejected, positive sign, g < 0.5) is exploratory ; "
    "rejected with negative sign is the Row 3 negative-direction "
    "break-through where RECOMBINE *hurts* on the ViT-tiny substrate ; "
    "failing-to-reject is the H7-A null which extends the closed "
    "G4-{quater..septimo} CNN-or-MLP scope ceiling to the small-"
    "transformer tier. This **inverts** the directional reading of "
    "the closed G4-{quater..septimo} ladder."
)


def _derive_h7b(h7a_resolution: str) -> dict[str, Any]:
    """Derive the H7-B universality-extension flag from H7-A resolution.

    Per pre-reg §6 :
    - ``confirmed_positive`` -> universality breaks at transformer.
    - ``null`` -> universality extends to small-transformer tier.
    - ``confirmed_subthreshold`` / ``confirmed_negative`` /
      ``insufficient_samples`` -> H7-B unresolved.
    """
    if h7a_resolution == "null":
        return {
            "state": "universality_extends",
            "universality_extends_to_small_transformer": True,
            "universality_breaks_at_transformer": False,
        }
    if h7a_resolution == "confirmed_positive":
        return {
            "state": "universality_breaks",
            "universality_extends_to_small_transformer": False,
            "universality_breaks_at_transformer": True,
        }
    return {
        "state": "unresolved",
        "universality_extends_to_small_transformer": False,
        "universality_breaks_at_transformer": False,
    }


def aggregate_g4_octavo_verdict(
    step1_path: Path,
) -> dict[str, Any]:
    """Load Step 1 milestone and return H7-A/H7-B verdicts.

    H7-A is taken verbatim from the Step 1 ``verdict`` block ; H7-B
    is derived per pre-reg §6 from the H7-A resolution state. No
    sister-pilot aggregate is consumed (H7-A is a single new test
    inverting the G4-{quater..septimo} convention).
    """
    s1 = json.loads(step1_path.read_text())
    h7a = s1["verdict"]["h7a_recombine_strategy"]
    insufficient = bool(h7a.get("insufficient_samples", False))
    if insufficient:
        resolution = "insufficient_samples"
        confirmed_positive = False
    else:
        resolution = str(h7a.get("resolution", "null"))
        confirmed_positive = resolution == "confirmed_positive"
    h7a_block: dict[str, Any] = {
        **h7a,
        "resolution": resolution,
        "confirmed_positive": confirmed_positive,
    }

    h7b_block = _derive_h7b(resolution)

    return {
        "h7a_tiny_imagenet_vit_tiny": h7a_block,
        "h7b_universality_extension": h7b_block,
        "summary": {
            "h7a_resolution": resolution,
            "h7a_confirmed_positive": confirmed_positive,
            "h7a_confirmed_subthreshold": (
                resolution == "confirmed_subthreshold"
            ),
            "h7a_confirmed_negative": (
                resolution == "confirmed_negative"
            ),
            "h7a_null": resolution == "null",
            "h7b_state": h7b_block["state"],
            "universality_extends_to_small_transformer": h7b_block[
                "universality_extends_to_small_transformer"
            ],
            "universality_breaks_at_transformer": h7b_block[
                "universality_breaks_at_transformer"
            ],
        },
    }


def _render_md(verdict: dict[str, Any]) -> str:
    h7a = verdict["h7a_tiny_imagenet_vit_tiny"]
    h7b = verdict["h7b_universality_extension"]
    s = verdict["summary"]

    lines: list[str] = [
        "# G4-octavo aggregate verdict",
        "",
        "**Date** : 2026-05-04",
        "**Pre-registration** : "
        "[docs/osf-prereg-g4-octavo-pilot.md]"
        "(../osf-prereg-g4-octavo-pilot.md)",
        "**Sister pilot** : G4-septimo aggregate "
        "[docs/milestones/g4-septimo-aggregate-2026-05-04.md]"
        "(./g4-septimo-aggregate-2026-05-04.md) (H6-C universality "
        "**confirmed** at four-benchmark × four-CNN-or-MLP scope "
        "ceiling — the closed leg this pilot extends).",
        "",
        "## Summary",
        "",
        f"- H7-A resolution : **{s['h7a_resolution']}**",
        (
            "- H7-A confirmed_positive (rejected + positive sign + "
            f"g >= 0.5) : **{s['h7a_confirmed_positive']}**"
        ),
        (
            "- H7-A confirmed_subthreshold (rejected + positive sign "
            f"+ g < 0.5) : **{s['h7a_confirmed_subthreshold']}**"
        ),
        (
            "- H7-A confirmed_negative (rejected + negative sign — "
            f"RECOMBINE hurts) : **{s['h7a_confirmed_negative']}**"
        ),
        f"- H7-A null (fail-to-reject) : **{s['h7a_null']}**",
        f"- H7-B state : **{s['h7b_state']}**",
        (
            "- Universality extends to small-transformer tier : "
            f"**{s['universality_extends_to_small_transformer']}**"
        ),
        (
            "- Universality breaks at transformer architecture : "
            f"**{s['universality_breaks_at_transformer']}**"
        ),
        "",
        "## H7-A — Tiny-ImageNet (n_classes=20 per task, G4ViTTiny)",
        "",
    ]
    if h7a.get("insufficient_samples"):
        lines.append("INSUFFICIENT SAMPLES")
    else:
        lines += [
            f"- mean P_max (mog) : {h7a['mean_p_max_mog']:.4f}",
            f"- mean P_max (none) : {h7a['mean_p_max_none']:.4f}",
            (
                f"- Hedges' g (mog vs none) : "
                f"{h7a['hedges_g_mog_vs_none']:.4f} "
                f"(threshold |g| >= {h7a['g_threshold']:.2f} : "
                f"{h7a['g_above_threshold']})"
            ),
            f"- Welch t : {h7a['welch_t']:.4f}",
            (
                f"- Welch p two-sided (alpha = "
                f"{h7a['alpha_per_test']:.4f}) : "
                f"{h7a['welch_p_two_sided']:.4f}"
            ),
            (
                f"- rejected_h0 : {h7a['rejected_h0']} ; "
                f"positive_sign : {h7a['positive_sign']} -> "
                f"resolution = `{h7a['resolution']}`"
            ),
            "",
            f"*Honest reading* : {HONEST_READING_H7A}",
        ]
    lines += [
        "",
        "## H7-B — small-transformer universality-extension flag",
        "",
        f"State : **{h7b['state']}**",
        "",
    ]
    if h7b["state"] == "universality_extends":
        lines.append(
            "H7-A null (fail-to-reject). The closed "
            "G4-{quater..septimo} four-benchmark × four-CNN-or-MLP "
            "RECOMBINE-empty universality flag extends to include "
            "the small-transformer tier (Tiny-ImageNet 200-class "
            "with ViT-tiny). The framework-C claim 'richer ops "
            "yield richer consolidation' is empirically refuted at "
            "this additional architectural scale ; DR-4 evidence "
            "v0.7 logs the scope extension."
        )
    elif h7b["state"] == "universality_breaks":
        lines.append(
            "H7-A confirmed_positive (rejected + positive sign + "
            "g >= 0.5). The closed G4-{quater..septimo} CNN-or-MLP "
            "universality is shown to **break** at the transformer "
            "architecture ; the framework-C RECOMBINE prediction is "
            "restored at the small-transformer tier. The CNN-or-MLP "
            "scope ceiling becomes a substrate-bounded refutation "
            "rather than universal. DR-4 evidence v0.7 records the "
            "architectural escape and triggers a re-analysis of the "
            "whole escalation ladder."
        )
    else:
        lines.append(
            "H7-B unresolved by this pilot. H7-A landed in "
            "`confirmed_subthreshold` (sub-threshold positive ; "
            "exploratory) or `confirmed_negative` (RECOMBINE hurts) "
            "or `insufficient_samples`. No universality-extension "
            "claim is filed ; DR-4 evidence v0.7 records the "
            "exploratory or negative-direction finding without "
            "resolving the small-transformer scope question."
        )
    lines += [
        "",
        "## Verdict — DR-4 evidence",
        "",
        (
            "Per pre-reg §6 : EC stays PARTIAL across all rows ; "
            "FC stays at C-v0.12.0 (no formal-axis bump scheduled "
            "by this pilot). Under H7-A `confirmed_positive` "
            "(Row 1), the framework's RECOMBINE prediction is "
            "restored at the transformer architectural tier and "
            "DR-4 evidence v0.7 records the architectural escape "
            "from the CNN-or-MLP scope ceiling. Under H7-A `null` "
            "(Row 2), the RECOMBINE-empty universality extends from "
            "{MLP, CNN} to {MLP, CNN, small-transformer} at <= 200 "
            "classes and DR-4 evidence v0.7 logs the scope "
            "extension. Under H7-A `confirmed_negative` (Row 3), "
            "the framework's claim is *further weakened* (RECOMBINE "
            "actively reduces transformer retention) and DR-4 "
            "evidence v0.7 records a negative-direction finding. "
            "Under H7-A `confirmed_subthreshold` (Row 4), the "
            "finding is exploratory and H7-B is not resolved."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G4-octavo aggregator")
    parser.add_argument("--step1", type=Path, default=DEFAULT_STEP1)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)

    verdict = aggregate_g4_octavo_verdict(args.step1)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    args.out_md.write_text(_render_md(verdict))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    s = verdict["summary"]
    print(
        f"H7-A : {s['h7a_resolution']}  "
        f"H7-B : {s['h7b_state']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
