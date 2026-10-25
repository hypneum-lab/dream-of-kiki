"""DR-3 Conformance Criterion — TTT fast-weight substrate (NH-03).

Validates the three conditions of the Conformance Criterion
(framework-C spec §6.2, ``conforms(S)``) on the TTT substrate :

1. Signature typing : identity constants, 4 callable op factories,
   component registry sharing the MLX/E-SNN core keys.
2. Axiom property tests : DR-0 (every write is metered and traced to
   an episode id ; the write budget is finite and capped), DR-1
   (replayed beta records are consumed through DreamRuntime episodes
   that all log), DR-2 op-pair facts (downscale commutative and not
   idempotent), R1 determinism (same seed -> same fast weights).
3. BLOCKING invariants enforceable : S2 finite guard aborts a
   diverging write ; S1 retained non-regression via ``swap_atomic``
   on the fast-weight tensor ; S3 topology guard on the canonical
   chain.

Reference : docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from kiki_oniric.dream.episode import (  # noqa: E402
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.guards.finite import (  # noqa: E402
    FiniteGuardError,
    check_finite,
)
from kiki_oniric.dream.guards.topology import (  # noqa: E402
    TopologyGuardError,
    validate_topology,
)
from kiki_oniric.dream.runtime import DreamRuntime  # noqa: E402
from kiki_oniric.dream.swap import SwapAborted, swap_atomic  # noqa: E402
from kiki_oniric.substrates.ttt_fast_weights import (  # noqa: E402
    TTT_SUBSTRATE_NAME,
    FastWriteAborted,
    TTTFastState,
    TTTSubstrate,
    WriteBudgetExceeded,
    WriteLedger,
    apply_fast,
    ttt_substrate_components,
)

DIM = 6
N_CLASSES = 2


def _toy_problem(seed: int) -> tuple[mx.array, mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((16, DIM)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int32)
    head = rng.standard_normal((N_CLASSES, DIM)).astype(np.float32)
    return mx.array(x), mx.array(y), mx.array(head)


def _forward_fn(head: mx.array) -> Callable[[mx.array, mx.array], mx.array]:
    def fwd(w: mx.array, x: mx.array) -> mx.array:
        return apply_fast(w, x) @ head.T

    return fwd


def _ce(state: TTTFastState, x: mx.array, y: mx.array, head: mx.array) -> float:
    loss = nn.losses.cross_entropy(
        apply_fast(state.weights, x) @ head.T, y, reduction="mean"
    )
    mx.eval(loss)
    return float(loss.item())


# ===== Condition 1 : signature typing =====


def test_c1_ttt_identity_and_factories() -> None:
    assert TTT_SUBSTRATE_NAME == "ttt_fast_weights"
    sub = TTTSubstrate()
    assert callable(sub.replay_handler_factory())
    assert callable(sub.downscale_handler_factory())
    assert callable(sub.restructure_handler_factory())
    assert callable(sub.recombine_handler_factory())


def test_c1_ttt_registry_mirrors_core_keys() -> None:
    from kiki_oniric.substrates.mlx_kiki_oniric import (
        mlx_substrate_components,
    )

    core = {
        "primitives", "replay", "downscale", "restructure", "recombine",
        "finite", "topology", "runtime", "swap", "p_min", "p_equ", "p_max",
    }
    assert core <= set(ttt_substrate_components())
    assert core <= set(mlx_substrate_components())


def test_c1_zero_fast_weights_are_identity() -> None:
    x, _, _ = _toy_problem(0)
    out = TTTFastState.zeros(DIM).apply(x)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))


# ===== Condition 2 : axiom property tests =====


def test_c2_replay_is_ttt_descent_and_metered() -> None:
    """One inner step lowers the inner loss and logs exactly 1 write."""
    x, y, head = _toy_problem(1)
    sub = TTTSubstrate()
    replay = sub.replay_handler_factory()
    s0 = TTTFastState.zeros(DIM)
    s1 = replay(s0, x, y, _forward_fn(head), 0.1, "de-0")
    assert _ce(s1, x, y, head) < _ce(s0, x, y, head)
    assert sub.ledger.total == 1
    assert sub.ledger.records[0].op == "replay"
    assert sub.ledger.records[0].episode_id == "de-0"
    assert sub.ledger.sample_gradients == 16


def test_c2_empty_batch_is_not_a_write() -> None:
    sub = TTTSubstrate()
    replay = sub.replay_handler_factory()
    s0 = TTTFastState.zeros(DIM)
    empty_x = mx.zeros((0, DIM))
    empty_y = mx.zeros((0,), dtype=mx.int32)
    s1 = replay(s0, empty_x, empty_y, _forward_fn(mx.ones((2, DIM))), 0.1, "e")
    assert s1 is s0
    assert sub.ledger.total == 0


@given(cap=st.integers(min_value=0, max_value=8))
@settings(max_examples=20, deadline=None, derandomize=True)
def test_c2_dr0_write_budget_cap_enforced(cap: int) -> None:
    """DR-0 : writes beyond the finite cap raise ; count == cap."""
    x, y, head = _toy_problem(2)
    sub = TTTSubstrate(ledger=WriteLedger(cap=cap))
    replay = sub.replay_handler_factory()
    state = TTTFastState.zeros(DIM)
    for i in range(cap):
        state = replay(state, x, y, _forward_fn(head), 0.05, f"de-{i}")
    with pytest.raises(WriteBudgetExceeded):
        replay(state, x, y, _forward_fn(head), 0.05, "de-over")
    assert sub.ledger.total == cap


def test_c2_dr0_every_write_traced_to_logged_episode() -> None:
    """Writes executed through DreamRuntime : every metered write's
    episode id appears in the runtime log (DR-0), and every replayed
    beta mini-batch is consumed by a logged DE (DR-1 analogue)."""
    x, y, head = _toy_problem(3)
    sub = TTTSubstrate()
    replay = sub.replay_handler_factory()
    downscale = sub.downscale_handler_factory()
    box = {"state": TTTFastState.zeros(DIM)}

    def h_replay(ep: DreamEpisode) -> None:
        box["state"] = replay(
            box["state"], x, y, _forward_fn(head), 0.1, ep.episode_id
        )

    def h_down(ep: DreamEpisode) -> None:
        box["state"] = downscale(box["state"], 0.9, ep.episode_id)

    rt = DreamRuntime()
    rt.register_handler(Operation.REPLAY, h_replay)
    rt.register_handler(Operation.DOWNSCALE, h_down)
    for i in range(5):
        op = Operation.DOWNSCALE if i == 4 else Operation.REPLAY
        rt.execute(
            DreamEpisode(
                trigger=EpisodeTrigger.SCHEDULED,
                input_slice={},
                operation_set=(op,),
                output_channels=(OutputChannel.WEIGHT_DELTA,),
                budget=BudgetCap(flops=1, wall_time_s=1.0, energy_j=0.0),
                episode_id=f"micro-{i}",
            )
        )
    logged = {e.episode_id for e in rt.log if e.completed}
    assert set(sub.ledger.episode_ids()) <= logged
    assert sub.ledger.by_op() == {"replay": 4, "downscale": 1}


def test_c2_downscale_commutative_not_idempotent() -> None:
    sub = TTTSubstrate()
    down = sub.downscale_handler_factory()
    w = mx.array(np.arange(DIM * DIM, dtype=np.float32).reshape(DIM, DIM))
    s = TTTFastState(weights=w)
    once = down(s, 0.5, "a")
    twice = down(once, 0.5, "b")
    np.testing.assert_allclose(twice.numpy(), np.asarray(w) * 0.25)
    assert not np.allclose(twice.numpy(), once.numpy())
    ab = down(down(s, 0.5, "c"), 0.8, "d")
    ba = down(down(s, 0.8, "e"), 0.5, "f")
    np.testing.assert_allclose(ab.numpy(), ba.numpy(), rtol=1e-6)
    with pytest.raises(ValueError):
        down(s, 1.5, "g")


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=15, deadline=None, derandomize=True)
def test_c2_r1_determinism_per_seed(seed: int) -> None:
    """Same seed -> bit-identical fast weights after the same writes."""

    def run() -> np.ndarray:
        x, y, head = _toy_problem(seed)
        sub = TTTSubstrate()
        replay = sub.replay_handler_factory()
        down = sub.downscale_handler_factory()
        state = TTTFastState.zeros(DIM)
        for i in range(3):
            state = replay(state, x, y, _forward_fn(head), 0.2, f"d{i}")
        state = down(state, 0.95, "dd")
        return state.numpy()

    np.testing.assert_array_equal(run(), run())



def test_c2_recombine_deterministic_with_seed() -> None:
    rec = TTTSubstrate().recombine_handler_factory()
    lat = np.array([[0.8, 0.2, 0.5], [0.1, 0.9, 0.3]], dtype=np.float32)
    np.testing.assert_array_equal(rec(lat, seed=42), rec(lat, seed=42))
    with pytest.raises(ValueError):
        rec(lat[:1], seed=0)


def test_c2_restructure_ops_and_metering() -> None:
    sub = TTTSubstrate()
    restr = sub.restructure_handler_factory()
    s = TTTFastState.zeros(DIM)
    s = restr(s, "add", 0, 1, "r0")
    assert s.numpy()[0, 1] == 1.0
    s = restr(s, "reroute", 0, 2, "r1")
    assert s.numpy()[2, 1] == 1.0 and s.numpy()[0, 1] == 0.0
    s = restr(s, "remove", 2, 1, "r2")
    assert not s.numpy().any()
    assert sub.ledger.by_op() == {"restructure": 3}
    with pytest.raises(ValueError):
        restr(s, "bogus", 0, 1, "r3")


# ===== Condition 3 : BLOCKING invariants enforceable =====


def test_c3_s2_guard_aborts_diverging_write() -> None:
    """A write producing NaN is rejected and NOT metered (S2)."""
    x, y, head = _toy_problem(4)
    sub = TTTSubstrate()
    replay = sub.replay_handler_factory()

    def exploding(w: mx.array, xb: mx.array) -> mx.array:
        return (apply_fast(w, xb) @ head.T) * mx.array(float("nan"))

    with pytest.raises(FastWriteAborted):
        replay(TTTFastState.zeros(DIM), x, y, exploding, 0.1, "boom")
    assert sub.ledger.total == 0


def test_c3_s2_finite_on_fast_state() -> None:
    check_finite(TTTFastState.zeros(DIM).numpy())
    bad = TTTFastState.zeros(DIM).numpy()
    bad[1, 1] = np.nan
    with pytest.raises(FiniteGuardError):
        check_finite(bad)


def test_c3_s1_swap_guard_on_fast_weights() -> None:
    """S1 : a fast-weight candidate that regresses the retained
    metric is refused by swap_atomic ; an improving one commits."""
    x, y, head = _toy_problem(5)
    sub = TTTSubstrate()
    replay = sub.replay_handler_factory()
    s0 = TTTFastState.zeros(DIM)
    good = replay(s0, x, y, _forward_fn(head), 0.2, "g")

    def retained(w: np.ndarray) -> float:
        return -_ce(TTTFastState(weights=mx.array(w)), x, y, head)

    pre = retained(s0.numpy())
    res = swap_atomic(s0.numpy(), good.numpy(), retained, pre, 0.0)
    assert res.committed
    bad = s0.numpy() + 5.0 * np.sign(good.numpy() - s0.numpy())
    with pytest.raises(SwapAborted):
        swap_atomic(s0.numpy(), bad, retained, pre, 0.0)


def test_c3_s3_topology_guard() -> None:
    canonical = {
        "rho_phono": ["rho_lex"],
        "rho_lex": ["rho_syntax"],
        "rho_syntax": ["rho_sem"],
        "rho_sem": [],
    }
    validate_topology(canonical)
    bad = dict(canonical, rho_phono=["rho_phono"])
    with pytest.raises(TopologyGuardError):
        validate_topology(bad)
