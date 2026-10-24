"""E1 matched-compute accounting with scoring passes charged.

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` section 5.4
(PI decision 2026-10-24: offline scoring passes are counted in the
compute budget).
**Mode** : pure accounting helpers, no model code (stdlib only).

Unit: one training token (forward + backward, about 6 N FLOPs per token
for N active parameters). A forward-only scoring token costs about 2 N,
i.e. ``FORWARD_TOKEN_COST = 1/3`` of a training token. Pre / post
evaluation passes are identical for every profile, so they are counted
but cancel in the match; scoring passes that only some profiles run
(OPR's 4 choices x 40 prompts) are charged against that profile's
update tokens: ``U = T_s - ceil(S / 3)``, floored at 0.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

FORWARD_TOKEN_COST = 1.0 / 3.0


def scoring_pass_tokens(passes: Iterable[tuple[int, Sequence[int]]]) -> int:
    """Tokens read by choice-scoring passes.

    ``passes`` yields ``(prompt_tokens, [choice_tokens, ...])``; each
    choice is one forward pass over prompt + choice.
    """
    total = 0
    for prompt, choices in passes:
        total += sum(prompt + c for c in choices)
    return total


def charged_scoring_tokens(scoring_tokens: int) -> int:
    """Training-token equivalent of ``scoring_tokens``, rounded up."""
    if scoring_tokens < 0:
        raise ValueError(f"scoring_tokens must be >= 0, got {scoring_tokens}")
    return math.ceil(scoring_tokens * FORWARD_TOKEN_COST - 1e-9)


def update_token_budget(t_s: int, arm_scoring_tokens: int) -> int:
    """Update tokens left for a profile after its own scoring passes.

    ``t_s`` is the section 5.4 budget ``5 x tok(p_min corpus)``;
    ``arm_scoring_tokens`` are the forward tokens of scoring passes the
    profile runs beyond the common pre / post evaluation (0 for every
    profile except OPR).
    """
    if t_s < 0:
        raise ValueError(f"t_s must be >= 0, got {t_s}")
    return max(t_s - charged_scoring_tokens(arm_scoring_tokens), 0)
