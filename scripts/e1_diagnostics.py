"""E1 drift, collapse, frozen-reference and MoE diagnostics (task 5).

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` section 6.
**Mode** : stdlib only. The harness measures the raw quantities (norm
sums, choice scores, routed-expert counts) and these helpers turn them
into the per-cell fields; :func:`collapse_rate` feeds the 20 % validity
gate of the analysis.

Operationalisation of the descriptive measures (no gate, no state):

- *Frozen reference.* The pre-replay checkpoint of the same cell. The
  probe set is the cell's own eval items (drawn from the confirmatory
  eval partition; tuning cells use their tuning eval items). For each
  item the 4 choice scores of ``_score_choice`` (length-normalised
  log-probabilities) are turned into a distribution over the 4 choices
  by a softmax; the measure is KL(reference || post), averaged over
  items. No gold answer is used.
- *Previously correct, now lost.* Among the probe items the reference
  answered correctly, the fraction the post-replay model answers
  wrongly (None when the reference answered none correctly).
- *Router drift.* ||g_post - g_pre|| / ||g_pre|| over the router (gate)
  weights. *Expert usage.* Normalised histogram of top-k routed tokens
  per expert, summed over layers, on the probe prompts; L1 distance
  between pre and post.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

from scipy import stats

COLLAPSE_DROP = 0.10
CHANCE_FLOOR = 0.25
COLLAPSE_GATE = 0.20


def relative_drift(sq_diff_sum: float, sq_pre_sum: float) -> float:
    """||w_post - w_pre|| / ||w_pre|| from sums of squares."""
    if sq_pre_sum <= 0:
        raise ValueError("pre-replay norm must be positive")
    return math.sqrt(sq_diff_sum) / math.sqrt(sq_pre_sum)


def collapse_flag(pre: float, post: float) -> bool:
    """Section 6: post acc < pre acc - 0.10, or post acc below 0.25 (4-way chance)."""
    return post < pre - COLLAPSE_DROP or post < CHANCE_FLOOR


def collapse_rate(flags: Sequence[bool]) -> float:
    return sum(bool(f) for f in flags) / len(flags) if flags else 0.0


def collapse_gate_fails(flags: Sequence[bool]) -> bool:
    """True when the collapse rate exceeds 20 % (the contrast is UNINTERPRETABLE)."""
    return collapse_rate(flags) > COLLAPSE_GATE


def _softmax(xs: Sequence[float]) -> list[float]:
    mx = max(xs)
    e = [math.exp(x - mx) for x in xs]
    z = sum(e)
    return [v / z for v in e]


def choice_kl(ref_scores: Sequence[Sequence[float]], post_scores: Sequence[Sequence[float]]) -> float:
    """Mean over items of KL(softmax(ref) || softmax(post)) over the choices."""
    if len(ref_scores) != len(post_scores) or not ref_scores:
        raise ValueError("need the same non-empty probe items for both checkpoints")
    total = 0.0
    for r, p in zip(ref_scores, post_scores, strict=True):
        pr, pp = _softmax(r), _softmax(p)
        total += sum(a * (math.log(a) - math.log(b)) for a, b in zip(pr, pp, strict=True)
                     if a > 0)
    return total / len(ref_scores)


def lost_correct(pre_correct: Sequence[bool], post_correct: Sequence[bool]) -> float | None:
    if len(pre_correct) != len(post_correct):
        raise ValueError("pre and post correctness must cover the same items")
    pre = [i for i, c in enumerate(pre_correct) if c]
    if not pre:
        return None
    return sum(1 for i in pre if not post_correct[i]) / len(pre)


def usage_l1(pre: Mapping[str, int], post: Mapping[str, int]) -> float | None:
    """L1 distance of two normalised expert-usage histograms (None if either is empty)."""
    sp, sq = sum(pre.values()), sum(post.values())
    if sp == 0 or sq == 0:
        return None
    keys = set(pre) | set(post)
    return sum(abs(pre.get(k, 0) / sp - post.get(k, 0) / sq) for k in keys)


def mean_ci95(values: Sequence[float | None]) -> dict[str, float | int | None]:
    """Mean and two-sided 95 % t interval over the defined values (descriptive)."""
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return {"n": 1, "mean": m, "lo": None, "hi": None}
    half = float(stats.t.ppf(0.975, len(xs) - 1)) * statistics.stdev(xs) / math.sqrt(len(xs))
    return {"n": len(xs), "mean": m, "lo": m - half, "hi": m + half}
