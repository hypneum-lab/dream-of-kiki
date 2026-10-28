"""E1 token-budget update schedule (task 3, section 5.4)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts.e1_schedule import Update, loss_tokens, schedule, summary, token_budget


def test_budget_uses_fixed_five() -> None:
    assert loss_tokens(1) == 0 and loss_tokens(0) == 0 and loss_tokens(7) == 6
    assert token_budget([10, 1, 5]) == 5 * (9 + 0 + 4)


def test_schedule_repeats_m_times_then_cycles_and_truncates() -> None:
    lengths = [4, 1, 3]  # 3 and 2 loss tokens; record 1 is skipped
    ups = schedule(lengths, m=2, budget=12)
    assert [(u.record, u.n_input) for u in ups] == [
        (0, 4), (0, 4), (2, 3), (2, 3), (0, 3)]  # last: r = 2 -> 3 inputs
    assert summary(ups, lengths) == {"achieved_loss_tokens": 12, "n_updates": 5,
                                     "n_truncated": 1}


def test_r_equals_one_and_exact_fit() -> None:
    assert schedule([5], m=1, budget=5)[-1] == Update(0, 2)
    assert [u.n_input for u in schedule([5], m=1, budget=8)] == [5, 5]


def test_zero_budget_and_errors() -> None:
    assert schedule([5], m=3, budget=0) == []
    with pytest.raises(ValueError):
        schedule([1, 0], m=1, budget=3)
    with pytest.raises(ValueError):
        schedule([5], m=0, budget=3)


@given(st.lists(st.integers(0, 40), min_size=1, max_size=12).filter(lambda x: max(x) >= 2),
       st.integers(1, 5), st.integers(0, 600))
def test_achieved_tokens_equal_budget(lengths: list[int], m: int, budget: int) -> None:
    ups = schedule(lengths, m, budget)
    assert sum(u.n_loss for u in ups) == budget
    assert all(1 <= u.n_loss <= lengths[u.record] - 1 for u in ups)
    assert all(u.n_input == lengths[u.record] for u in ups[:-1])
