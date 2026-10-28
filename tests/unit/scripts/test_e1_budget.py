"""E1 compute accounting: scoring passes are charged (PI 2026-10-24)."""

from __future__ import annotations

import pytest

from scripts.e1_budget import (
    budget_status,
    FORWARD_TOKEN_COST,
    charged_scoring_tokens,
    scoring_pass_tokens,
    update_token_budget,
)


def test_forward_cost_is_one_third_of_a_training_token() -> None:
    # forward ~ 2 N FLOPs per token, forward + backward ~ 6 N
    assert FORWARD_TOKEN_COST == pytest.approx(1 / 3)


def test_scoring_pass_tokens_sums_prompt_plus_choice_lengths() -> None:
    # 2 prompts x 4 choices: each pass reads prompt + choice tokens
    passes = [(10, [1, 2, 3, 4]), (7, [2, 2, 2, 2])]
    assert scoring_pass_tokens(passes) == (4 * 10 + 10) + (4 * 7 + 8)


def test_no_arm_specific_scoring_keeps_full_budget() -> None:
    assert update_token_budget(12_000, 0) == 12_000


def test_opr_scoring_is_charged_against_its_updates() -> None:
    # 3000 forward tokens = 1000 training-token equivalents
    assert charged_scoring_tokens(3000) == 1000
    assert update_token_budget(12_000, 3000) == 11_000


def test_charge_rounds_up_so_budget_never_exceeds_match() -> None:
    assert charged_scoring_tokens(1) == 1
    assert update_token_budget(10, 2) == 9


def test_budget_floors_at_zero() -> None:
    assert update_token_budget(100, 10_000) == 0


@pytest.mark.parametrize(("t", "s"), [(-1, 0), (10, -5)])
def test_negative_inputs_raise(t: int, s: int) -> None:
    with pytest.raises(ValueError):
        update_token_budget(t, s)


# --- review round 3: infeasible cells are classified, not silently floored -


def test_overbudget_cell_is_classified() -> None:
    # T = 100, S = 10000: scoring alone costs 3334 token-equivalents
    st = budget_status(100, 10_000)
    assert st == {"t_s": 100, "charge": 3334, "update_tokens": 0,
                  "status": "scoring-overbudget", "matched": False}


def test_feasible_cell_is_matched() -> None:
    st = budget_status(12_000, 3000)
    assert st["status"] == "matched" and st["matched"] is True
    assert st["update_tokens"] == 11_000


def test_exact_boundary_is_feasible() -> None:
    st = budget_status(1000, 3000)  # charge 1000 == T
    assert st["matched"] is True and st["update_tokens"] == 0
