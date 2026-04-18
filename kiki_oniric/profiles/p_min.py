"""P_min profile — minimal publishable consolidation.

Channels: β → 1 (curated buffer in, weight delta out).
Operations: {replay, downscale}.

S9.4: model field + swap_now() method wire P_min to the swap
protocol with retained-benchmark gating (S1 invariant). Pragmatic
skeleton: model parameters flattened to np.array for the
substrate-agnostic swap_atomic API. Full MLX-native swap lands
S13+ alongside the concurrent dream worker.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from harness.benchmarks.retained.retained import RetainedBenchmark
from kiki_oniric.dream.episode import Operation
from kiki_oniric.dream.eval_retained import evaluate_retained
from kiki_oniric.dream.operations.downscale import (
    DownscaleOpState,
    downscale_handler,
)
from kiki_oniric.dream.operations.replay import (
    ReplayOpState,
    replay_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime
from kiki_oniric.dream.swap import SwapResult, swap_atomic


@dataclass
class PMinProfile:
    """Minimal profile: replay + downscale handlers + swap protocol."""

    model: Any = None
    runtime: DreamRuntime = field(default_factory=DreamRuntime)
    replay_state: ReplayOpState = field(default_factory=ReplayOpState)
    downscale_state: DownscaleOpState = field(
        default_factory=DownscaleOpState
    )

    def __post_init__(self) -> None:
        self.runtime.register_handler(
            Operation.REPLAY, replay_handler(self.replay_state)
        )
        self.runtime.register_handler(
            Operation.DOWNSCALE, downscale_handler(self.downscale_state)
        )

    def swap_now(
        self,
        retained_pre_acc: float,
        benchmark: RetainedBenchmark,
        model_predictor: Callable[[dict], str],
        delta_regression: float = 0.02,
    ) -> SwapResult:
        """Trigger swap_atomic with retained-benchmark gating.

        Pragmatic skeleton: w_awake / w_scratch are placeholder
        np.arrays — substantive content here is the retained eval
        closure that bridges the model to S1 guard. Full MLX-native
        swap lands S13+.

        Raises SwapAborted (from swap_atomic) when S1 (retained
        non-regression) or S2 (finite weights) fails.
        """
        w_awake = np.array([0.0])
        w_scratch = np.array([0.0])

        def retained_eval(_w):
            return evaluate_retained(model_predictor, benchmark)

        return swap_atomic(
            _w_awake=w_awake,
            w_scratch=w_scratch,
            retained_eval=retained_eval,
            retained_pre_acc=retained_pre_acc,
            delta_regression=delta_regression,
        )
