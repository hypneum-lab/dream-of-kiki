"""H9-A / H9-B / H9-C verdict aggregator for G6-Studio Path A.

Implements the locked decision rules from
``docs/osf-prereg-g6-studio-path-a.md`` §6 :

- H9-A confirmed iff Welch one-sided rejects ``H0: mean(P_equ) <=
  mean(baseline)`` AND positive sign (mean(P_equ) > mean(baseline))
  AND ``g_h9a >= 0.5``. Within H9-A, the strict large-effect band is
  ``g_h9a >= 2`` ; the medium / sub-large band ``0.5 <= g_h9a < 2``
  is reported as exploratory ``H9-A*`` (per-reg §3 N=5 detection
  floor).
- H9-B confirmed iff Welch fails to reject H0 OR ``g_h9a < 0.5``.
  Mutually exclusive with H9-A on the (g threshold, Welch reject)
  pair ; mixed signals are reported as ``INDETERMINATE``.
- H9-C confirmed iff Jonckheere fails to reject monotonic
  increasing trend over ``[P_min, P_equ, P_max]`` AND
  ``mean(P_min) > mean(P_equ)`` (DR-4 inversion).

Bonferroni family-wise α = 0.05 / 3 (three confirmatory hypotheses).

Reference :
- Plan : ``docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md`` Task 7
- Pre-reg : ``docs/osf-prereg-g6-studio-path-a.md`` §5-§6
"""
from __future__ import annotations

from typing import Any

from kiki_oniric.eval.statistics import (
    compute_hedges_g,
    jonckheere_trend,
    welch_one_sided,
)


# Family-wise α correction over the three confirmatory hypotheses
# {H9-A, H9-B, H9-C}. Single source of truth — cited verbatim in
# pre-reg §5, paper2 §7.1.10, and the H9-A test suite.
H9_BONFERRONI_ALPHA: float = 0.05 / 3

# Strict large-effect band threshold for the H9-A confirmation row
# (pre-reg §3 / §6 row 1). Below this threshold but above 0.5 is the
# H9-A* exploratory band (row 1*).
H9A_STRICT_LARGE_G_THRESHOLD: float = 2.0
H9A_MEDIUM_G_THRESHOLD: float = 0.5


def _mean(xs: list[float]) -> float:
    """Mean of a sample, NaN for empty lists."""
    return float(sum(xs) / len(xs)) if xs else float("nan")


def classify_h9(retention: dict[str, list[float]]) -> dict[str, Any]:
    """Run H9-A / H9-B / H9-C decision rules on a retention-by-arm dict.

    Parameters
    ----------
    retention
        Mapping ``{arm -> list[retention_per_seed]}`` keyed by
        ``baseline`` / ``P_min`` / ``P_equ`` / ``P_max``. Sample
        size < 2 in any required arm yields the
        ``INSUFFICIENT`` classification for that hypothesis.

    Returns
    -------
    dict[str, Any]
        Verdict dict with keys ::

            bonferroni_alpha          : float (0.05 / 3)
            h9a_classification        : "H9-A" | "H9-A*" | "H9-B"
                                         | "INDETERMINATE" | "INSUFFICIENT"
            h9a_g                     : float (Hedges' g, P_equ vs baseline)
            h9a_welch_p               : float
            h9a_welch_reject          : bool
            h9a_positive_sign         : bool (mean(P_equ) > mean(baseline))
            h9a_strict_large_effect   : bool (g >= 2)
            h9c_classification        : "H9-C" | "NOT_H9C" | "INSUFFICIENT"
            h9c_jonckheere_p          : float
            h9c_jonckheere_reject     : bool
            h9c_mean_p_min            : float
            h9c_mean_p_equ            : float
            h9c_mean_p_max            : float
    """
    out: dict[str, Any] = {"bonferroni_alpha": H9_BONFERRONI_ALPHA}

    base = retention.get("baseline", [])
    p_equ = retention.get("P_equ", [])
    p_min = retention.get("P_min", [])
    p_max = retention.get("P_max", [])

    # ----- H9-A : real-LLM transfer of REPLAY+DOWNSCALE coupling -----
    if len(base) < 2 or len(p_equ) < 2:
        out.update(
            {
                "h9a_classification": "INSUFFICIENT",
                "h9a_g": float("nan"),
                "h9a_welch_p": float("nan"),
                "h9a_welch_reject": False,
                "h9a_positive_sign": False,
                "h9a_strict_large_effect": False,
            },
        )
    else:
        try:
            g_h9a = compute_hedges_g(p_equ, base)
        except ValueError:
            # Both samples constant with non-equal means : Cohen's d
            # undefined. Treat as zero-effect fallback so the
            # downstream classification stays deterministic ; the
            # Welch test below still drives reject/fail.
            g_h9a = 0.0
        # welch_one_sided rejects when mean(treatment) < mean(control).
        # We test "P_equ improves over baseline" so the H1 we want is
        # mean(baseline) < mean(P_equ) — pass arms accordingly.
        welch = welch_one_sided(base, p_equ, alpha=H9_BONFERRONI_ALPHA)
        positive_sign = _mean(p_equ) > _mean(base)
        strict_large = g_h9a >= H9A_STRICT_LARGE_G_THRESHOLD

        out["h9a_g"] = float(g_h9a)
        out["h9a_welch_p"] = float(welch.p_value)
        out["h9a_welch_reject"] = bool(welch.reject_h0)
        out["h9a_positive_sign"] = bool(positive_sign)
        out["h9a_strict_large_effect"] = bool(strict_large)

        if g_h9a < H9A_MEDIUM_G_THRESHOLD and not welch.reject_h0:
            # H9-B : washout / spectator pattern persists at 35B.
            out["h9a_classification"] = "H9-B"
        elif (
            welch.reject_h0
            and positive_sign
            and g_h9a >= H9A_MEDIUM_G_THRESHOLD
        ):
            # H9-A confirmed. Distinguish strict (g >= 2) vs the
            # exploratory medium-effect band (0.5 <= g < 2).
            out["h9a_classification"] = (
                "H9-A" if strict_large else "H9-A*"
            )
        elif welch.reject_h0 and not positive_sign:
            # Welch rejected with negative sign — H9-C territory
            # (handled via the dedicated H9-C block below for the
            # canonical Jonckheere-based classification, but flag
            # here as well so callers see both signals).
            out["h9a_classification"] = "H9-C"
        else:
            # Mixed signals (e.g., Welch reject + tiny g, or large g
            # + Welch fail). Reported honestly as INDETERMINATE.
            out["h9a_classification"] = "INDETERMINATE"

    # ----- H9-C : universal DR-4 inversion (P_min > P_equ) -----
    if any(len(g) < 2 for g in (p_min, p_equ, p_max)):
        out.update(
            {
                "h9c_classification": "INSUFFICIENT",
                "h9c_jonckheere_p": float("nan"),
                "h9c_jonckheere_reject": False,
                "h9c_mean_p_min": _mean(p_min),
                "h9c_mean_p_equ": _mean(p_equ),
                "h9c_mean_p_max": _mean(p_max),
            },
        )
    else:
        jt = jonckheere_trend(
            [p_min, p_equ, p_max], alpha=H9_BONFERRONI_ALPHA,
        )
        m_min = _mean(p_min)
        m_equ = _mean(p_equ)
        m_max = _mean(p_max)
        out["h9c_jonckheere_p"] = float(jt.p_value)
        out["h9c_jonckheere_reject"] = bool(jt.reject_h0)
        out["h9c_mean_p_min"] = m_min
        out["h9c_mean_p_equ"] = m_equ
        out["h9c_mean_p_max"] = m_max
        out["h9c_classification"] = (
            "H9-C" if (not jt.reject_h0 and m_min > m_equ) else "NOT_H9C"
        )

    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint — load a milestone JSON, print the H9 verdict.

    Usage ::

        uv run python -m experiments.g6_studio_path_a.aggregator_h9 \\
            --milestone docs/milestones/g6-studio-path-a-2026-05-04.json
    """
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description=(
            "G6-Studio Path A H9-A/B/C verdict aggregator (decision "
            "rules locked in pre-reg §5)."
        ),
    )
    parser.add_argument(
        "--milestone",
        type=Path,
        required=True,
        help=(
            "Path to a milestone JSON whose top-level has a "
            "'retention_by_arm' or 'cells' field."
        ),
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.milestone.read_text(encoding="utf-8"))
    retention = payload.get("retention_by_arm")
    if retention is None:
        cells = payload.get("cells", [])
        retention = {}
        for cell in cells:
            arm = cell.get("arm")
            if arm is None:
                continue
            if cell.get("excluded_underperforming_baseline"):
                continue
            retention.setdefault(arm, []).append(
                float(cell["retention"]),
            )
    verdict = classify_h9(retention)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


__all__ = [
    "H9A_MEDIUM_G_THRESHOLD",
    "H9A_STRICT_LARGE_G_THRESHOLD",
    "H9_BONFERRONI_ALPHA",
    "classify_h9",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
