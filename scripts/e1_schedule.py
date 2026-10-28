"""E1 token-budget update schedule (task 3 of section 9).

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` section
5.4 ("Token budget T", "Update schedule", "Scoring passes are
counted").
**Mode** : stdlib only; works on token counts, so the harness passes
the pinned tokenizer's lengths and replays the returned updates.

- ``tok(record)`` counts loss tokens: sequence length - 1 (0 for a
  record of fewer than 2 tokens).
- ``T_s = 5 x tok(p_min corpus of seed s)``; 5 is the fixed harness
  default ``inner``, never the tuned m.
- Schedule: records in corpus order, each repeated m times
  consecutively, cycling from the start, records with < 2 tokens
  skipped, until exactly the budget of loss tokens is reached. A
  terminal record that would overshoot with r tokens remaining is
  truncated to its first r + 1 input tokens (r loss tokens).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

BUDGET_INNER = 5  # fixed constant of section 5.4, not the tuned m


def loss_tokens(n_tokens: int) -> int:
    return n_tokens - 1 if n_tokens >= 2 else 0


def token_budget(p_min_lengths: Sequence[int]) -> int:
    """T_s from the token lengths of the seed's ``p_min`` corpus."""
    return BUDGET_INNER * sum(loss_tokens(n) for n in p_min_lengths)


@dataclass(frozen=True)
class Update:
    record: int  # index into the profile corpus
    n_input: int  # input tokens fed (the full record, or r + 1 if terminal)

    @property
    def n_loss(self) -> int:
        return self.n_input - 1


def schedule(lengths: Sequence[int], m: int, budget: int) -> list[Update]:
    """Updates whose loss tokens sum exactly to ``budget``.

    Returns ``[]`` if ``budget`` is 0. Raises ValueError if ``m < 1`` or
    the budget is positive but no record has >= 2 tokens (no update can
    spend it).
    """
    if m < 1:
        raise ValueError(f"m must be >= 1 (got {m})")
    if budget < 0:
        raise ValueError(f"budget must be >= 0 (got {budget})")
    if budget == 0:
        return []
    usable = [i for i, n in enumerate(lengths) if n >= 2]
    if not usable:
        raise ValueError("positive budget but no record has >= 2 tokens")
    out: list[Update] = []
    spent = 0
    while True:
        for i in usable:
            for _ in range(m):
                left = budget - spent
                full = lengths[i] - 1
                if full <= left:
                    out.append(Update(i, lengths[i]))
                    spent += full
                else:
                    out.append(Update(i, left + 1))
                    spent += left
                if spent == budget:
                    return out


def summary(updates: Sequence[Update], lengths: Sequence[int]) -> dict[str, int]:
    """Achieved loss tokens, optimizer steps and truncated (terminal) updates."""
    return {"achieved_loss_tokens": sum(u.n_loss for u in updates),
            "n_updates": len(updates),
            "n_truncated": sum(1 for u in updates if u.n_input < lengths[u.record])}
