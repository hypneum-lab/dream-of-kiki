# M7 delta_acc consolidation→eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the `mlx_latent_diffusion` substrate's `delta_acc` to the dream consolidation so it stops being a structural zero, via the B5 awake/dream loop with WeightUpdate-only scope.

**Architecture:** Two diffusion-substrate-local emitting handlers (`replay_diffusion_handler`, `downscale_diffusion_handler`) wrap the existing non-emitting `_real.py` logic and additionally capture per-layer dense deltas on the `MLPDenoiser`. A new `DenoiserWeightDeltaChannel` consumes those `WeightUpdate`s via `apply_channel_outputs`. The CL head probes a denoiser-derived feature `denoiser(z, t_fixed)` before and after the apply, so `delta_acc = post_acc − baseline_acc` reflects the actual consolidation effect.

**Tech Stack:** MLX (Apple Silicon), Python ≥ 3.12, `uv`, `pytest`. No new third-party deps.

**Source spec:** `docs/superpowers/specs/2026-05-21-m7-delta-acc-consolidation-eval-design.md` (sections normative).

**Branch:** `feat/m7-substrate-dr3` (already created; do not branch from this plan).

**Project constraints to respect at every commit:** subject ≤ 50 chars, scope ≥ 3 chars (no underscore), body lines ≤ 72 chars, English, no AI attribution, no `--no-verify`. `uv run ruff check .` and `uv run mypy harness tests` must stay green.

---

## Files to create

- `kiki_oniric/substrates/_diffusion/denoiser_weight_channel.py` — `DenoiserWeightDeltaChannel`.
- `kiki_oniric/substrates/_diffusion/handlers_emit.py` — diffusion-substrate-local emitting wrappers for `replay` and `downscale`.
- `tests/unit/test_denoiser_weight_channel.py`
- `tests/unit/test_diffusion_handlers_emit.py`

## Files to modify

- `kiki_oniric/substrates/_diffusion/__init__.py` — re-export new symbols.
- `kiki_oniric/substrates/_diffusion/cl_eval_head.py` — add `denoiser_feature` helper.
- `kiki_oniric/substrates/_diffusion/dream_ops_adapter.py` — switch to emitting handlers; populate `output_channels`.
- `kiki_oniric/substrates/mlx_latent_diffusion.py` — head probes denoiser-feature; call `apply_channel_outputs` between baseline and post.
- `tests/unit/test_mlx_latent_diffusion_adapter.py` — replace the degenerate order-independence test with a real regression.
- `tests/reproducibility/golden_hashes_apple_m5.json` — regenerate the 3 diffusion entries.
- `tests/reproducibility/REBASELINE_NOTE.md` — append entry.
- `CHANGELOG.md` — correct the M7 (C-v0.25.0) `delta_acc` wording.
- `harness/diffusion_eval/milestone.py` — fix the hardcoded c_version on line 40.
- `docs/milestones/wave3b-bench-pending.{md,json,cells.jsonl}` — regenerate.

---

## Task 1: `DenoiserWeightDeltaChannel`

**Files:**
- Create: `kiki_oniric/substrates/_diffusion/denoiser_weight_channel.py`
- Test: `tests/unit/test_denoiser_weight_channel.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/unit/test_denoiser_weight_channel.py`:

```python
"""Unit tests for DenoiserWeightDeltaChannel (M7 delta_acc wiring)."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from kiki_oniric.dream.channels import WeightUpdate
from kiki_oniric.substrates._diffusion.model import MLPDenoiser
from kiki_oniric.substrates._diffusion.denoiser_weight_channel import (
    DenoiserWeightDeltaChannel,
)


def _denoiser() -> MLPDenoiser:
    return MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)


def test_apply_adds_delta_to_named_layer_weight() -> None:
    """apply() must add the per-layer delta to the matching weight."""
    d = _denoiser()
    layer0_w_before = np.asarray(d.layers[0].weight).copy()
    delta = np.ones_like(layer0_w_before, dtype=np.float32)
    channel = DenoiserWeightDeltaChannel(d)

    channel.apply({"layer_0_weight": delta}, fisher_bump=None)

    layer0_w_after = np.asarray(d.layers[0].weight)
    np.testing.assert_allclose(
        layer0_w_after, layer0_w_before + delta, atol=1e-6,
    )


def test_apply_adds_delta_to_named_layer_bias() -> None:
    """apply() must add the per-layer delta to the matching bias."""
    d = _denoiser()
    layer0_b_before = np.asarray(d.layers[0].bias).copy()
    delta = np.ones_like(layer0_b_before, dtype=np.float32) * 0.5
    channel = DenoiserWeightDeltaChannel(d)

    channel.apply({"layer_0_bias": delta}, fisher_bump=None)

    layer0_b_after = np.asarray(d.layers[0].bias)
    np.testing.assert_allclose(
        layer0_b_after, layer0_b_before + delta, atol=1e-6,
    )


def test_apply_unknown_key_raises() -> None:
    """An unknown layer key must raise — silent skip would hide bugs."""
    d = _denoiser()
    delta = np.zeros((2, 2), dtype=np.float32)
    channel = DenoiserWeightDeltaChannel(d)

    with pytest.raises(KeyError, match="bogus_key"):
        channel.apply({"bogus_key": delta}, fisher_bump=None)


def test_apply_finite_guard_raises_on_non_finite_post() -> None:
    """If the apply would produce NaN/inf, raise with S2 in the message."""
    d = _denoiser()
    layer0_w = np.asarray(d.layers[0].weight)
    bad = np.full_like(layer0_w, np.inf, dtype=np.float32)
    channel = DenoiserWeightDeltaChannel(d)

    # WeightUpdate.__post_init__ already rejects non-finite input,
    # so we bypass it by feeding the raw dict directly to apply.
    with pytest.raises(ValueError, match="S2"):
        channel.apply({"layer_0_weight": bad}, fisher_bump=None)


def test_apply_accepts_fisher_bump_without_using_it() -> None:
    """fisher_bump is recorded for traceability but does not gate apply."""
    d = _denoiser()
    delta = np.zeros_like(np.asarray(d.layers[0].weight), dtype=np.float32)
    fisher = {"layer_0_weight": np.ones_like(delta)}
    channel = DenoiserWeightDeltaChannel(d)

    channel.apply({"layer_0_weight": delta}, fisher_bump=fisher)
    # No exception; fisher_bump captured on the channel for traceability.
    assert channel.last_fisher_bump is fisher
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_denoiser_weight_channel.py -v`
Expected: ImportError / ModuleNotFoundError for `denoiser_weight_channel`.

- [ ] **Step 1.3: Implement the channel**

Create `kiki_oniric/substrates/_diffusion/denoiser_weight_channel.py`:

```python
"""DenoiserWeightDeltaChannel — Canal 1 awake-side applier for the
diffusion substrate.

Consumes ``WeightUpdate`` outputs whose ``lora_delta`` field carries
a per-layer dense delta keyed by ``f"layer_{i}_weight"`` or
``f"layer_{i}_bias"`` (matching the emission convention used by
``replay_diffusion_handler`` and ``downscale_diffusion_handler``).
Applies the delta in-place to the matching ``MLPDenoiser``
parameter, then re-checks S2 (finite values) on the result.

The ``lora_`` prefix on the ``WeightUpdate`` field is a legacy
naming carry-over from the LoRA origin of the channel-1 protocol;
the value is a dict of dense per-layer arrays.

See ``docs/superpowers/specs/2026-05-21-m7-delta-acc-consolidation-eval-design.md``
§ Components 1.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray


class DenoiserWeightDeltaChannel:
    """Awake-side WeightDeltaChannel for an ``MLPDenoiser``.

    Implements the ``WeightDeltaChannel`` protocol (one method,
    ``apply``). Holds a reference to the denoiser whose parameters
    it mutates.
    """

    def __init__(self, denoiser: Any) -> None:
        self._denoiser = denoiser
        self.last_fisher_bump: (
            dict[str, NDArray[np.float32]] | None
        ) = None

    def apply(
        self,
        lora_delta: dict[str, NDArray[np.float32]],
        fisher_bump: dict[str, NDArray[np.float32]] | None = None,
    ) -> None:
        """Add each per-layer delta to the matching denoiser param.

        Keys follow ``f"layer_{i}_weight"`` / ``f"layer_{i}_bias"``.
        Unknown keys raise ``KeyError`` — silent skip would hide
        emission/consume mismatches.

        S2 finite values: input is already validated by
        ``WeightUpdate.__post_init__``; we re-validate the post-apply
        parameter so a numerical surprise on the denoiser side
        surfaces here rather than at the next forward pass.
        """
        layers = self._denoiser.layers
        for key, delta in lora_delta.items():
            layer_idx, attr = self._parse_key(key, n_layers=len(layers))
            current = getattr(layers[layer_idx], attr)
            new_val = current + mx.array(delta)
            new_np = np.asarray(new_val)
            if not np.isfinite(new_np).all():
                raise ValueError(
                    f"S2: denoiser {attr} {layer_idx} non-finite after "
                    f"apply"
                )
            setattr(layers[layer_idx], attr, new_val)
        self.last_fisher_bump = fisher_bump
        mx.eval(*self._all_layer_tensors())

    @staticmethod
    def _parse_key(key: str, *, n_layers: int) -> tuple[int, str]:
        for attr in ("weight", "bias"):
            prefix = "layer_"
            suffix = f"_{attr}"
            if key.startswith(prefix) and key.endswith(suffix):
                try:
                    idx = int(key[len(prefix):-len(suffix)])
                except ValueError:
                    break
                if 0 <= idx < n_layers:
                    return idx, attr
        raise KeyError(
            f"DenoiserWeightDeltaChannel: unknown layer key {key!r}"
        )

    def _all_layer_tensors(self) -> list[Any]:
        out: list[Any] = []
        for layer in self._denoiser.layers:
            for attr in ("weight", "bias"):
                t = getattr(layer, attr, None)
                if t is not None:
                    out.append(t)
        return out


__all__ = ["DenoiserWeightDeltaChannel"]
```

- [ ] **Step 1.4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_denoiser_weight_channel.py -v`
Expected: 5 passed.

- [ ] **Step 1.5: Lint + type-check**

Run: `uv run ruff check kiki_oniric/substrates/_diffusion/denoiser_weight_channel.py tests/unit/test_denoiser_weight_channel.py`
Run: `uv run mypy kiki_oniric/substrates/_diffusion/denoiser_weight_channel.py`
Expected: both green.

- [ ] **Step 1.6: Commit**

```bash
git add kiki_oniric/substrates/_diffusion/denoiser_weight_channel.py \
        tests/unit/test_denoiser_weight_channel.py
git commit -m "feat(substrate): denoiser weight delta channel"
```

---

## Task 2: `replay_diffusion_handler` (emitting)

**Files:**
- Create: `kiki_oniric/substrates/_diffusion/handlers_emit.py`
- Test: `tests/unit/test_diffusion_handlers_emit.py`

**Why substrate-local:** `kiki_oniric/dream/operations/CLAUDE.md` forbids adding a 4th variant under `dream/operations/`; new substrates dispatch via `kiki_oniric/substrates/`. The emitting wrappers therefore live in `_diffusion/`.

- [ ] **Step 2.1: Write the failing test**

Create `tests/unit/test_diffusion_handlers_emit.py`:

```python
"""Unit tests for the diffusion-substrate emitting handlers."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from kiki_oniric.dream.channels import WeightUpdate
from kiki_oniric.dream.episode import (
    BudgetCap, DreamEpisode, EpisodeTrigger, Operation,
)
from kiki_oniric.dream.operations.replay_real import ReplayRealState
from kiki_oniric.substrates._diffusion.handlers_emit import (
    replay_diffusion_handler,
)
from kiki_oniric.substrates._diffusion.model import MLPDenoiser


def _model_adapter(denoiser: MLPDenoiser):
    # Thin holder exposing .layers (mirroring _DenoiserSingleArgAdapter
    # surface that bind_real_handlers binds to). The handler computes
    # SGD on `model_inner(x)`; we wrap denoiser to a single-arg
    # callable by closing over a fixed t.
    class _A:
        def __init__(self, d):
            self._d = d
            self.layers = list(d.layers)

        def __call__(self, x):
            t = mx.zeros((x.shape[0],), dtype=mx.int32)
            return self._d(x, t)

        def parameters(self):
            return self._d.parameters()

        def update(self, params):
            self._d.update(params)

        def trainable_parameters(self):
            return self._d.trainable_parameters()

    return _A(denoiser)


def _episode_with_records(records):
    return DreamEpisode(
        trigger=EpisodeTrigger.PROFILE,
        input_slice={"beta_records": records},
        operation_set=(Operation.REPLAY,),
        output_channels=("weight_update",),
        budget=BudgetCap(flops=10**9, wall_time_s=10.0, energy_j=1.0),
        episode_id="t/2/diff",
    )


def test_replay_handler_emits_weight_update_on_records() -> None:
    """Non-empty beta_records must yield a WeightUpdate with all denoiser
    layer keys populated."""
    d = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    adapter = _model_adapter(d)
    state = ReplayRealState()
    handler = replay_diffusion_handler(state, model=adapter, lr=1e-2)

    records = [
        {"x": mx.zeros((5,)), "y": mx.zeros((4,))} for _ in range(3)
    ]
    out = handler(_episode_with_records(records))

    assert isinstance(out, WeightUpdate)
    # Three layers (n_layers=2 hidden + 1 output) → 6 keys
    # (weight + bias each), all present.
    expected = {f"layer_{i}_{a}" for i in range(3) for a in ("weight", "bias")}
    assert set(out.lora_delta.keys()) == expected


def test_replay_handler_returns_none_on_empty_records() -> None:
    """Empty beta_records is the S1 no-op branch — no emission."""
    d = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    adapter = _model_adapter(d)
    state = ReplayRealState()
    handler = replay_diffusion_handler(state, model=adapter, lr=1e-2)

    out = handler(_episode_with_records([]))

    assert out is None
    assert state.last_loss is None


def test_replay_handler_delta_matches_post_minus_pre() -> None:
    """The emitted delta must equal (post_param − pre_param) per layer."""
    d = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    pre = {
        f"layer_{i}_{a}": np.asarray(getattr(d.layers[i], a)).copy()
        for i in range(3) for a in ("weight", "bias")
    }
    adapter = _model_adapter(d)
    state = ReplayRealState()
    handler = replay_diffusion_handler(state, model=adapter, lr=1e-2)

    records = [
        {"x": mx.ones((5,)), "y": mx.ones((4,))} for _ in range(3)
    ]
    out = handler(_episode_with_records(records))
    assert out is not None

    for key, pre_arr in pre.items():
        # _i / attr inferred from key for assertions
        i = int(key.split("_")[1])
        a = key.split("_")[2]
        post = np.asarray(getattr(d.layers[i], a))
        np.testing.assert_allclose(
            out.lora_delta[key], post - pre_arr, atol=1e-6,
        )
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_diffusion_handlers_emit.py -v`
Expected: ImportError on `handlers_emit`.

- [ ] **Step 2.3: Implement the handler**

Create `kiki_oniric/substrates/_diffusion/handlers_emit.py`:

```python
"""Diffusion-substrate emitting wrappers for replay and downscale.

These wrappers mirror ``kiki_oniric.dream.operations.{replay,downscale}_real``
in behaviour but additionally capture the dense per-layer delta on
the bound ``MLPDenoiser`` and emit a channel-1 ``WeightUpdate`` so
the B5 awake/dream loop can apply the consolidation to the denoiser
via ``DenoiserWeightDeltaChannel``.

Substrate-local per ``kiki_oniric/dream/operations/CLAUDE.md`` (no
4th variant under ``dream/operations/``; new substrates dispatch
via ``kiki_oniric/substrates/``).

See ``docs/superpowers/specs/2026-05-21-m7-delta-acc-consolidation-eval-design.md``
§ Components 2 + § Components 3.
"""
from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx
import mlx.nn as _nn
import mlx.optimizers as optim
import numpy as np

from kiki_oniric.dream.channels import WeightUpdate
from kiki_oniric.dream.episode import DreamEpisode
from kiki_oniric.dream.operations.downscale_real import DownscaleRealState
from kiki_oniric.dream.operations.replay_real import ReplayRealState

nn: Any = _nn


def _snapshot_layers(layers: list[Any]) -> dict[str, np.ndarray]:
    """Capture a numpy snapshot of every layer.weight / layer.bias."""
    snap: dict[str, np.ndarray] = {}
    for i, layer in enumerate(layers):
        for attr in ("weight", "bias"):
            t = getattr(layer, attr, None)
            if t is not None:
                snap[f"layer_{i}_{attr}"] = np.asarray(t).copy()
    return snap


def _diff(
    pre: dict[str, np.ndarray], post: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return per-key (post − pre) deltas as float32 arrays."""
    return {
        k: (post[k] - pre[k]).astype(np.float32, copy=False)
        for k in pre
    }


def replay_diffusion_handler(
    state: ReplayRealState,
    *,
    model: Any,
    lr: float = 1e-2,
) -> Callable[[DreamEpisode], "WeightUpdate | None"]:
    """Build a diffusion-substrate replay handler that emits.

    Mirrors ``replay_real_handler``'s SGD step on the denoiser but
    snapshots all ``layer_i.{weight,bias}`` before / after and emits
    a ``WeightUpdate`` carrying the dense delta. Empty
    ``beta_records`` → no-op, returns ``None`` (S1 contract).
    """
    optimizer = optim.SGD(learning_rate=lr)

    def loss_fn(model_inner: Any, x: mx.array, y: mx.array) -> mx.array:
        pred = model_inner(x)
        return mx.mean((pred - y) ** 2)

    grad_fn = nn.value_and_grad(model, loss_fn)

    def handler(episode: DreamEpisode) -> "WeightUpdate | None":
        records = episode.input_slice.get("beta_records", [])
        if not records:
            state.last_loss = None
            state.last_compute_flops = 0
            return None

        for idx, r in enumerate(records):
            if "x" not in r or "y" not in r:
                raise ValueError(
                    f"record {idx} missing 'x' or 'y' key: {r!r}"
                )

        pre = _snapshot_layers(model.layers)

        xs = mx.array([r["x"] for r in records])
        ys = mx.array([r["y"] for r in records])
        loss, grads = grad_fn(model, xs, ys)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        post = _snapshot_layers(model.layers)
        delta = _diff(pre, post)

        state.total_records_consumed += len(records)
        state.last_loss = float(loss.item())
        # K1 tag: a crude record-count proxy is enough for non-LoRA.
        state.last_compute_flops = max(len(records), 1)
        state.total_compute_flops += state.last_compute_flops

        return WeightUpdate(lora_delta=delta, fisher_bump=None)

    return handler


def downscale_diffusion_handler(
    state: DownscaleRealState,
    *,
    model: Any,
) -> Callable[[DreamEpisode], "WeightUpdate | None"]:
    """Build a diffusion-substrate downscale handler that emits.

    Mirrors ``downscale_real_handler`` (multiply weight + bias by
    ``shrink_factor`` per layer) but snapshots all
    ``layer_i.{weight,bias}`` before / after and emits a
    ``WeightUpdate`` carrying the dense (factor − 1) · W delta.
    """

    def handler(episode: DreamEpisode) -> "WeightUpdate | None":
        factor = episode.input_slice.get("shrink_factor", 1.0)
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )

        pre = _snapshot_layers(model.layers)

        for layer in model.layers:
            w = getattr(layer, "weight", None)
            b = getattr(layer, "bias", None)
            if w is not None:
                layer.weight = w * factor
            if b is not None:
                layer.bias = b * factor

        tensors_to_eval: list[Any] = []
        for layer in model.layers:
            for attr in ("weight", "bias"):
                t = getattr(layer, attr, None)
                if t is not None:
                    tensors_to_eval.append(t)
        if tensors_to_eval:
            mx.eval(*tensors_to_eval)

        post = _snapshot_layers(model.layers)
        delta = _diff(pre, post)

        state.compound_factor *= factor
        state.last_compute_flops = max(
            sum(arr.size for arr in pre.values()), 1
        )

        return WeightUpdate(lora_delta=delta, fisher_bump=None)

    return handler


__all__ = [
    "replay_diffusion_handler",
    "downscale_diffusion_handler",
]
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_diffusion_handlers_emit.py -v`
Expected: 3 passed.

- [ ] **Step 2.5: Lint + type-check**

Run: `uv run ruff check kiki_oniric/substrates/_diffusion/handlers_emit.py tests/unit/test_diffusion_handlers_emit.py`
Run: `uv run mypy kiki_oniric/substrates/_diffusion/handlers_emit.py`
Expected: both green.

- [ ] **Step 2.6: Commit**

```bash
git add kiki_oniric/substrates/_diffusion/handlers_emit.py \
        tests/unit/test_diffusion_handlers_emit.py
git commit -m "feat(substrate): emitting diffusion handlers"
```

---

## Task 3: `denoiser_feature` helper

**Files:**
- Modify: `kiki_oniric/substrates/_diffusion/cl_eval_head.py`
- Test: extend `tests/unit/test_diffusion_cl_eval_head.py`

- [ ] **Step 3.1: Append the failing test**

Add to the end of `tests/unit/test_diffusion_cl_eval_head.py`:

```python
def test_denoiser_feature_shape_and_determinism() -> None:
    """denoiser_feature returns (batch, d_latent) and is deterministic."""
    from kiki_oniric.substrates._diffusion.cl_eval_head import (
        denoiser_feature,
    )
    from kiki_oniric.substrates._diffusion.model import MLPDenoiser

    d = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    z = mx.zeros((5, 4))
    out_a = denoiser_feature(d, z, t_fixed=5)
    out_b = denoiser_feature(d, z, t_fixed=5)

    assert out_a.shape == (5, 4)
    import numpy as np
    np.testing.assert_allclose(
        np.asarray(out_a), np.asarray(out_b), atol=0.0,
    )
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_diffusion_cl_eval_head.py::test_denoiser_feature_shape_and_determinism -v`
Expected: ImportError on `denoiser_feature`.

- [ ] **Step 3.3: Add the helper to `cl_eval_head.py`**

Append to `kiki_oniric/substrates/_diffusion/cl_eval_head.py`:

```python
def denoiser_feature(
    denoiser: Any, z: mx.array, *, t_fixed: int = -1
) -> mx.array:
    """Probe feature: ``denoiser(z, t_fixed)`` for the CL head.

    ``t_fixed`` is a deterministic scalar timestep used for both the
    baseline (pre-dream) and post (post-dream) eval, so the only
    thing moving between the two evals is the denoiser's weights —
    which is exactly what the B5 apply mutates. The default
    ``t_fixed=-1`` is replaced by ``denoiser.config-equivalent /
    2`` resolution at the substrate call site; pass an explicit
    non-negative ``t_fixed`` here.

    Returns an MLX array of shape ``(batch, d_latent)``.
    """
    if t_fixed < 0:
        raise ValueError(
            f"t_fixed must be non-negative, got {t_fixed}"
        )
    t = mx.array([t_fixed], dtype=mx.int32)
    return denoiser(z, t)
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_diffusion_cl_eval_head.py -v`
Expected: 3 passed (the new test plus the original two).

- [ ] **Step 3.5: Commit**

```bash
git add kiki_oniric/substrates/_diffusion/cl_eval_head.py \
        tests/unit/test_diffusion_cl_eval_head.py
git commit -m "feat(substrate): denoiser_feature probe helper"
```

---

## Task 4: Re-export new symbols

**Files:**
- Modify: `kiki_oniric/substrates/_diffusion/__init__.py`

- [ ] **Step 4.1: Read current re-exports**

Run: `cat kiki_oniric/substrates/_diffusion/__init__.py`
Expected: a small module re-exporting `Encoder`, `Decoder`, `ClEvalHead`, `bind_real_handlers`.

- [ ] **Step 4.2: Add the new exports**

Edit `kiki_oniric/substrates/_diffusion/__init__.py` to also re-export `DenoiserWeightDeltaChannel`, `replay_diffusion_handler`, `downscale_diffusion_handler`, `denoiser_feature`. The exact append (the rest of the file is unchanged):

```python
from kiki_oniric.substrates._diffusion.denoiser_weight_channel import (
    DenoiserWeightDeltaChannel,
)
from kiki_oniric.substrates._diffusion.handlers_emit import (
    downscale_diffusion_handler,
    replay_diffusion_handler,
)
from kiki_oniric.substrates._diffusion.cl_eval_head import (
    denoiser_feature,
)
```

And extend `__all__` to include those four names.

- [ ] **Step 4.3: Smoke-import**

Run: `uv run python -c "from kiki_oniric.substrates._diffusion import DenoiserWeightDeltaChannel, replay_diffusion_handler, downscale_diffusion_handler, denoiser_feature; print('ok')"`
Expected: `ok`.

- [ ] **Step 4.4: Commit**

```bash
git add kiki_oniric/substrates/_diffusion/__init__.py
git commit -m "feat(substrate): re-export delta_acc symbols"
```

---

## Task 5: Switch `bind_real_handlers` to emitting handlers

**Files:**
- Modify: `kiki_oniric/substrates/_diffusion/dream_ops_adapter.py`

- [ ] **Step 5.1: Read the current adapter**

Run: `cat kiki_oniric/substrates/_diffusion/dream_ops_adapter.py`
Expected: imports `replay_real_handler` + `downscale_real_handler` (non-emitting); the `bind_real_handlers` function returns a `set[Operation]` and registers the four ops.

- [ ] **Step 5.2: Replace the replay + downscale handler imports**

Replace:

```python
from kiki_oniric.dream.operations.replay_real import (
    ReplayRealState,
    replay_real_handler,
)
```

with:

```python
from kiki_oniric.dream.operations.replay_real import ReplayRealState
from kiki_oniric.substrates._diffusion.handlers_emit import (
    downscale_diffusion_handler,
    replay_diffusion_handler,
)
```

And replace:

```python
from kiki_oniric.dream.operations.downscale_real import (
    DownscaleRealState,
    downscale_real_handler,
)
```

with:

```python
from kiki_oniric.dream.operations.downscale_real import (
    DownscaleRealState,
)
```

- [ ] **Step 5.3: Switch the two registrations**

Replace `replay_real_handler(...)` with `replay_diffusion_handler(...)` and `downscale_real_handler(...)` with `downscale_diffusion_handler(...)` in the two `register_handler` calls inside `bind_real_handlers`.

`restructure_real_handler` and `recombine_real_handler` are unchanged (out-of-scope per spec).

- [ ] **Step 5.4: Run the DR-3 conformance test to verify wiring**

Run: `uv run pytest tests/conformance/axioms/test_dr3_diffusion_profile.py -v --no-cov`
Expected: 2 passed (activation-set wiring intact; the test does not execute handlers).

- [ ] **Step 5.5: Lint + type-check**

Run: `uv run ruff check kiki_oniric/substrates/_diffusion/dream_ops_adapter.py`
Run: `uv run mypy kiki_oniric/substrates/_diffusion/dream_ops_adapter.py`
Expected: both green.

- [ ] **Step 5.6: Commit**

```bash
git add kiki_oniric/substrates/_diffusion/dream_ops_adapter.py
git commit -m "feat(substrate): bind emitting diffusion handlers"
```

---

## Task 6: Wire `execute_profile` — head probe + apply loop

**Files:**
- Modify: `kiki_oniric/substrates/mlx_latent_diffusion.py`

This is the central wiring step. It rewires the baseline / dream-loop / post sequence in `execute_profile`.

- [ ] **Step 6.1: Open the file and locate the baseline / post block**

Run: `grep -n "delta_acc baseline\|delta_acc post\|return.*\"delta_acc\":" kiki_oniric/substrates/mlx_latent_diffusion.py`
Expected: three lines — the baseline-comment block, the post-comment block, and the metrics dict that returns `delta_acc`.

- [ ] **Step 6.2: Import the new symbols at the top of `execute_profile`**

Locate the imports block inside `execute_profile` (the one that imports `ClEvalHead, train_head_inplace, eval_head_accuracy` from `_diffusion.cl_eval_head`). Replace it with:

```python
from kiki_oniric.substrates._diffusion.cl_eval_head import (
    ClEvalHead,
    denoiser_feature,
    eval_head_accuracy,
    train_head_inplace,
)
from kiki_oniric.substrates._diffusion.denoiser_weight_channel import (
    DenoiserWeightDeltaChannel,
)
from kiki_oniric.consolidate import apply_channel_outputs
```

- [ ] **Step 6.3: Pin the fixed probe timestep**

Just after `train_root, _sample_root, data_root, _head_root = mx.random.split(root, num=4)` add:

```python
# Fixed timestep for the CL head probe — used for BOTH baseline
# and post eval. Only the denoiser's weights change between the
# two evals (via apply_channel_outputs), which is what makes
# delta_acc a real measurement (M7 delta_acc design § Architecture).
t_probe = int(self.config.t_steps) // 2
```

- [ ] **Step 6.4: Replace the baseline block**

Replace:

```python
# delta_acc baseline: train + eval the head BEFORE dream cycle.
# All cells use the first batch as the eval slice for simplicity.
baseline_acc = 0.0
if dataset:
    train_head_inplace(
        cl_head, dataset[0], labels_per_batch[0],
    )
    baseline_acc = eval_head_accuracy(
        cl_head, dataset[0], labels_per_batch[0],
    )
```

with:

```python
# delta_acc baseline: train + eval the head BEFORE dream cycle
# on the denoiser-derived probe feature. The head is frozen
# from here until post_acc — only the denoiser will move (via
# apply_channel_outputs after the dream loop), so delta_acc
# reflects the consolidation's effect on the denoiser feature.
baseline_acc = 0.0
if dataset:
    pre_feat = denoiser_feature(
        self.denoiser, dataset[0], t_fixed=t_probe
    )
    train_head_inplace(cl_head, pre_feat, labels_per_batch[0])
    baseline_acc = eval_head_accuracy(
        cl_head, pre_feat, labels_per_batch[0],
    )
```

- [ ] **Step 6.5: Add the apply step after the dream loop, before post_acc**

Locate the line `# delta_acc post-cycle: same (per-cell) head, eval again after`. Just *before* that comment block, insert:

```python
# B5 awake/dream apply — consume the WeightUpdate emissions from
# replay + downscale and apply them to the denoiser in place. The
# log accumulates from `profile.runtime.execute(episode)` calls
# inside the dream loop above.
weight_channel = DenoiserWeightDeltaChannel(self.denoiser)
apply_channel_outputs(
    profile.runtime.log, weight_channel=weight_channel
)
profile.runtime.reset_log()
```

- [ ] **Step 6.6: Replace the post block**

Replace:

```python
# delta_acc post-cycle: same (per-cell) head, eval again after
# the dream.
post_acc = 0.0
if dataset:
    post_acc = eval_head_accuracy(
        cl_head, dataset[0], labels_per_batch[0],
    )
```

with:

```python
# delta_acc post-cycle: same head, recompute the denoiser probe
# feature against the now-consolidated denoiser, eval again.
post_acc = 0.0
if dataset:
    post_feat = denoiser_feature(
        self.denoiser, dataset[0], t_fixed=t_probe
    )
    post_acc = eval_head_accuracy(
        cl_head, post_feat, labels_per_batch[0],
    )
```

- [ ] **Step 6.7: Run the smoke test**

Run: `uv run pytest tests/unit/test_mlx_latent_diffusion_adapter.py -k execute_profile -v --no-cov`
Expected: existing `test_execute_profile_runs_minimal_smoke_cycle` + variants pass; `test_execute_profile_delta_acc_is_order_independent` still passes (delta is still order-independent — but no longer trivially zero).

- [ ] **Step 6.8: Lint + type-check**

Run: `uv run ruff check kiki_oniric/substrates/mlx_latent_diffusion.py`
Run: `uv run mypy kiki_oniric/substrates/mlx_latent_diffusion.py`
Expected: both green.

- [ ] **Step 6.9: Commit**

```bash
git add kiki_oniric/substrates/mlx_latent_diffusion.py
git commit -m "feat(substrate): apply WeightUpdate, probe denoiser"
```

---

## Task 7: Strengthen the regression test (`delta_acc ≠ 0` + order-independence)

**Files:**
- Modify: `tests/unit/test_mlx_latent_diffusion_adapter.py`

The previous order-independence test is now a degenerate `0 == 0` (passing trivially). Replace it with a strengthened version: assert `delta_acc != 0` AND order-independence.

- [ ] **Step 7.1: Replace the test body**

Locate `def test_execute_profile_delta_acc_is_order_independent` in `tests/unit/test_mlx_latent_diffusion_adapter.py`. Replace its body with:

```python
def test_execute_profile_delta_acc_is_real_and_order_independent() -> None:
    """delta_acc must be (a) non-zero (real measurement, M7 delta_acc
    design) AND (b) identical across call orders on the same
    substrate (R1 — the per-cell head + per-call apply loop must
    not leak between cells).
    """
    from dataclasses import dataclass

    @dataclass
    class _Req:
        seed: int
        profile: str

    # Cell A alone on a pristine substrate.
    solo = MLXLatentDiffusionSubstrate().execute_profile(
        _Req(seed=1, profile="p_min")
    )
    acc_a_solo = solo["delta_acc"]

    # Cell A run after cell B on the same substrate.
    shared = MLXLatentDiffusionSubstrate()
    shared.execute_profile(_Req(seed=2, profile="p_max"))
    after = shared.execute_profile(_Req(seed=1, profile="p_min"))
    acc_a_after_b = after["delta_acc"]

    # Order-independence
    assert acc_a_after_b == acc_a_solo, (
        "delta_acc changed across orders — head/apply state is "
        "leaking between cells"
    )
    # delta_acc is bounded by construction (head frozen)
    assert -1.0 <= acc_a_solo <= 1.0, (
        f"delta_acc out of [-1, 1]: {acc_a_solo}"
    )
    # delta_acc is a real measurement, not the structural zero
    # that motivated the M7 wiring fix
    assert acc_a_solo != 0.0, (
        "delta_acc is exactly zero — consolidation→eval is not "
        "wired (regression to pre-fix M7 state)"
    )
```

- [ ] **Step 7.2: Run the strengthened test**

Run: `uv run pytest tests/unit/test_mlx_latent_diffusion_adapter.py::test_execute_profile_delta_acc_is_real_and_order_independent -v --no-cov`
Expected: PASS.

- [ ] **Step 7.3: Lint**

Run: `uv run ruff check tests/unit/test_mlx_latent_diffusion_adapter.py`
Expected: green.

- [ ] **Step 7.4: Commit**

```bash
git add tests/unit/test_mlx_latent_diffusion_adapter.py
git commit -m "test(substrate): delta_acc is real and order-indep"
```

---

## Task 8: Fix `milestone.py:40` hardcoded c_version

**Files:**
- Modify: `harness/diffusion_eval/milestone.py`

- [ ] **Step 8.1: Read the milestone module head**

Run: `sed -n '1,80p' harness/diffusion_eval/milestone.py`
Expected: see line ~40 hardcoding `"c_version": "C-v0.14.0+PARTIAL"` inside `aggregate_cells` (or sibling).

- [ ] **Step 8.2: Derive c_version from the cells**

Replace the hardcoded line:

```python
"c_version": "C-v0.14.0+PARTIAL",
```

with:

```python
"c_version": _c_version_of(cells),
```

And add at module top-level (above `aggregate_cells`):

```python
def _c_version_of(cells: list[dict[str, Any]]) -> str:
    """Return the most common c_version across cells.

    Heterogeneous c_versions appear when a registry has not been
    purged between runs of different substrate versions; emitting
    the dominant value avoids silently stamping a stale version
    onto the milestone summary.
    """
    if not cells:
        return "unknown"
    from collections import Counter
    counts = Counter(c.get("c_version", "unknown") for c in cells)
    return counts.most_common(1)[0][0]
```

- [ ] **Step 8.3: Spot-check on the existing pending file**

Run: `uv run python -c "
import json
from harness.diffusion_eval.milestone import _c_version_of
cells = [json.loads(l) for l in open('docs/milestones/wave3b-bench-pending.cells.jsonl')]
print(_c_version_of(cells))
"`
Expected: `C-v0.15.0+PARTIAL` (the fresh cells dominate at 465 / 490).

- [ ] **Step 8.4: Lint + type-check**

Run: `uv run ruff check harness/diffusion_eval/milestone.py`
Run: `uv run mypy harness/diffusion_eval/milestone.py`
Expected: both green.

- [ ] **Step 8.5: Commit**

```bash
git add harness/diffusion_eval/milestone.py
git commit -m "fix(harness): derive milestone c_version from cells"
```

---

## Task 9: Purge stale entries from `.run_registry.sqlite`

**Files:**
- Modify: `.run_registry.sqlite` (data file, not source)

`.run_registry.sqlite` is uncommitted (per `.gitignore`); the purge is a local maintenance step before re-running the bench.

- [ ] **Step 9.1: Inspect what's there**

Run:

```bash
uv run python -c "
import sqlite3
con = sqlite3.connect('.run_registry.sqlite')
for row in con.execute(
    'SELECT c_version, COUNT(*) FROM runs GROUP BY c_version'
):
    print(row)
"
```

Expected: at least two rows — a stale `C-v0.14.0+PARTIAL` (or earlier) count and the `C-v0.15.0+PARTIAL` count.

- [ ] **Step 9.2: Delete the stale rows**

Run:

```bash
uv run python -c "
import sqlite3
con = sqlite3.connect('.run_registry.sqlite')
con.execute(\"DELETE FROM runs WHERE c_version != 'C-v0.15.0+PARTIAL'\")
con.commit()
print('purged stale rows')
"
```

Expected: `purged stale rows`. (If the table name is not `runs`, replace with the actual name discovered in step 9.1.)

- [ ] **Step 9.3: Re-inspect to confirm**

Same command as 9.1.
Expected: a single row, `C-v0.15.0+PARTIAL` only.

- [ ] **Step 9.4: No commit**

The registry is not tracked. Move to Task 10.

---

## Task 10: Regenerate R1 golden hashes (3 diffusion entries)

**Files:**
- Modify: `tests/reproducibility/golden_hashes_apple_m5.json`
- Modify: `tests/reproducibility/REBASELINE_NOTE.md`

The denoiser is now mutated by `apply_channel_outputs`, and the head probes the denoiser. The three diffusion R1 entries (`test_r1_diffusion_*`, or the names that exist on this branch) will hash differently.

- [ ] **Step 10.1: Run the diffusion R1 tests in regenerate mode**

Run: `uv run pytest tests/reproducibility/ -k diffusion -v --no-cov`
Expected: the three diffusion tests rewrite their entries (the suite's behaviour is to update `pending_review` entries' hashes on M5 when the recorded hash mismatches; verify by inspecting the JSON diff).

If they instead FAIL (because the suite is strict-compare on M5), the regeneration path is:

```bash
DREAMOFKIKI_REBASELINE_DIFFUSION=1 uv run pytest \
    tests/reproducibility/ -k diffusion -v --no-cov
```

Inspect the suite's `conftest.py` or `tests/reproducibility/CLAUDE.md` for the precise rebaseline knob if neither path applies; the existing M7 rebaseline (`d5c427a test(r1): regenerate diffusion hashes for M7`) used a matching mechanism — mirror it.

- [ ] **Step 10.2: Verify the JSON diff matches expectation**

Run: `git diff tests/reproducibility/golden_hashes_apple_m5.json`
Expected: only the 3 diffusion entries' `hash` and `commit` fields change; their `status` stays `pending_review`; no other entry is touched.

- [ ] **Step 10.3: Append a REBASELINE_NOTE entry**

Append to `tests/reproducibility/REBASELINE_NOTE.md`:

```markdown
## 2026-05-21 — M7 delta_acc consolidation→eval wiring

The three diffusion R1 entries (`test_r1_diffusion_*`) are
regenerated under `C-v0.15.0+PARTIAL` after `execute_profile` now
applies `apply_channel_outputs` to the denoiser and the CL head
probes `denoiser(z, t_fixed)` instead of raw encoder latents. The
hash drift is expected; entries stay `pending_review` pending
cross-machine confirmation on m3_ultra and m1_max.
```

- [ ] **Step 10.4: Commit**

```bash
git add tests/reproducibility/golden_hashes_apple_m5.json \
        tests/reproducibility/REBASELINE_NOTE.md
git commit -m "test(r1): rebaseline diffusion for delta_acc wiring"
```

---

## Task 11: Regenerate `wave3b-bench-pending.*`

**Files:**
- Modify: `docs/milestones/wave3b-bench-pending.md`
- Modify: `docs/milestones/wave3b-bench-pending.json`
- Modify: `docs/milestones/wave3b-bench-pending.cells.jsonl`

- [ ] **Step 11.1: Run the full diffusion ablation bench**

Run: `uv run python scripts/ablation_cycle3_diffusion.py 2>&1 | tail -8`
Expected: `[m5] complete: 450/450 cells`.

- [ ] **Step 11.2: Verify the regenerated milestone**

Run:

```bash
uv run python -c "
import json, math
md = open('docs/milestones/wave3b-bench-pending.md').read()
js = json.load(open('docs/milestones/wave3b-bench-pending.json'))
cells = [json.loads(l) for l in open('docs/milestones/wave3b-bench-pending.cells.jsonl')]
print('total cells:', len(cells))
print('c_versions :', {c['c_version'] for c in cells})
print('commit_sha :', js['commit_sha'])
da = [c['delta_acc'] for c in cells]
print('delta_acc  : min=%.4f max=%.4f nan=%s zeros=%d/%d' % (
    min(da), max(da), any(math.isnan(v) for v in da),
    sum(1 for v in da if v == 0.0), len(da),
))
"
```

Expected:
- `total cells: 450`
- `c_versions : {'C-v0.15.0+PARTIAL'}` (single)
- `commit_sha` matches `git rev-parse HEAD`
- `delta_acc min/max` inside `[-1, 1]`, **no NaN**, and not all zeros (the zero count is well below 450 — non-trivial movement).

- [ ] **Step 11.3: Commit**

```bash
git add docs/milestones/wave3b-bench-pending.md \
        docs/milestones/wave3b-bench-pending.json \
        docs/milestones/wave3b-bench-pending.cells.jsonl
git commit -m "milestone(bench): regenerate M7 with real delta_acc"
```

---

## Task 12: Correct the CHANGELOG M7 entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 12.1: Locate the M7 entry**

Run: `grep -n "C-v0.25.0\|delta_acc is now a real" CHANGELOG.md | head -5`
Expected: line numbers for the `## [C-v0.25.0+PARTIAL]` header and the existing `delta_acc` description.

- [ ] **Step 12.2: Replace the overclaim wording**

Replace:

```
  contract is satisfied via a new random-init `Decoder` MLP
  (`_diffusion.decoder`). `delta_acc` is now a real CL head
  measurement (`_diffusion.cl_eval_head`). Closes issue #36.
```

with:

```
  contract is satisfied via a new random-init `Decoder` MLP
  (`_diffusion.decoder`). `delta_acc` is a real CL head measurement
  wired via the B5 awake/dream loop: replay + downscale emit
  `WeightUpdate`s, `apply_channel_outputs` applies them to the
  denoiser through `DenoiserWeightDeltaChannel`, and the head
  probes `denoiser(z, t_fixed)` before and after — so `delta_acc`
  reflects the consolidation's effect on the denoiser feature
  (bounded in `[-1, 1]` by construction). Closes issue #36.
```

And inside the same entry's `### Added` list, append:

```
- `kiki_oniric/substrates/_diffusion/denoiser_weight_channel.py`
  (`DenoiserWeightDeltaChannel`).
- `kiki_oniric/substrates/_diffusion/handlers_emit.py`
  (`replay_diffusion_handler`, `downscale_diffusion_handler`).
- `denoiser_feature` helper in `_diffusion.cl_eval_head`.
- `tests/unit/test_denoiser_weight_channel.py`,
  `tests/unit/test_diffusion_handlers_emit.py`.
```

- [ ] **Step 12.3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): M7 delta_acc real measurement"
```

---

## Task 13: Final verification

- [ ] **Step 13.1: Full suite, ruff, mypy**

Run:

```bash
uv run ruff check . && \
uv run mypy harness tests && \
uv run pytest -q
```

Expected: ruff green, mypy green, pytest `… passed` with no failures and coverage ≥ 40% (local gate; macOS nightly enforces 90%).

- [ ] **Step 13.2: Working tree audit**

Run: `git status --short`
Expected: clean (or only the self-mutating golden-hash `commit` field, which is the known defect tracked separately).

- [ ] **Step 13.3: One-line PR-readiness summary in conversation**

Print a single sentence summarizing: `delta_acc` is real (non-zero, bounded), bench regenerated, R1 rebased, suite green. Hand off to the M7 PR opener.

---

## Out of scope (do not implement here)

- Routing `restructure` (`TopologyDiff`) and `recombine` (`LatentSample`)
  through diffusion-native channels — follow-up plan.
- LoRA-wrapping the denoiser — explicitly rejected (Approach B in spec).
- Any change to the encoder, sampler, or trainer.
- Project-level CHANGELOG entries for versions other than `C-v0.25.0`.
- Touching the macOS nightly 90 % coverage gate — handled separately.
