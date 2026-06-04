# M7 — Substrate DR-3 conformance fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `kiki_oniric/substrates/mlx_latent_diffusion.execute_profile` honour the framework-C §3.1 normative profile contract — each profile activates the exact dream-side primitives declared in `kiki_oniric/profiles/p_{min,equ,max}.py`, the real handlers run against the diffusion's denoiser, and the substrate emits real per-op metrics off the registered `*RealState` dataclasses. Closes [issue #36](https://github.com/hypneum-lab/kiki-of-dream/issues/36).

**Architecture:** Reuse — do not re-implement — the existing `_real.py` handler factories (`replay_real_handler`, `downscale_real_handler`, `restructure_real_handler`, `recombine_real_handler`). The diffusion substrate constructs the right `PMinProfile / PEquProfile / PMaxProfile` for each cell, then **overrides** the profile's skeleton-handler registrations on its `runtime: DreamRuntime` with `_real.py` factories bound to the substrate's `denoiser`, `_cifar_encoder`, and a new `Decoder` (D7 of the spec). One canonical `DreamEpisode` per loader batch drives the runtime ; metrics are snapshotted off the profile's state fields. A small 1-layer classifier head produces the real `delta_acc` (D2 of the spec). FC MINOR : `C-v0.14.0 → C-v0.15.0+PARTIAL`.

**Tech Stack:** MLX (Apple Silicon), Python 3.12+, `uv`, `pytest`. No new third-party deps.

**Source spec:** `docs/superpowers/specs/2026-05-21-m7-substrate-dr3-design.md` (post-audit revision authoritative, sections §2 / §3 D1-D7 normative).

**Branch:** `feat/m7-substrate-dr3` (already created off `main`).

---

## Pre-flight knowledge (do not re-discover — audit-verified 2026-05-21)

- **Profile dataclasses** are independent (no Protocol base) ; `__post_init__` registers **skeleton** handlers (e.g. `replay_handler`, not `replay_real_handler`). M7 must *override* the registrations with the `_real.py` factories.
- **Handler factory signatures** (exact) :
  - `replay_real_handler(state: ReplayRealState, *, model, lr: float = 0.01) -> Callable[[DreamEpisode], None]`
  - `downscale_real_handler(state: DownscaleRealState, *, model) -> Callable[[DreamEpisode], None]`
  - `restructure_real_handler(state: RestructureRealState, *, model) -> Callable[[DreamEpisode], None]` (legacy : only `topo_op="reroute"` works)
  - `recombine_real_handler(state: RecombineRealState, *, encoder: VAEEncoder, decoder: VAEDecoder, seed: int) -> Callable[[DreamEpisode], LatentSample | None]`
- **`DreamEpisode`** is `@dataclass(frozen=True)` with `trigger`, `input_slice` (MappingProxyType — immutable post-init), `operation_set: tuple[Operation, ...]`, `output_channels`, `budget`, `episode_id`.
- **`DreamRuntime.execute(episode)`** executes `episode.operation_set` in order via `_handlers[op]` ; missing handler raises `NotImplementedError` ; DR-0 guarantees a log entry per call.
- **Activation per §3.1 (NORMATIVE)** : `p_min ⊃ {replay, downscale}`, `p_equ ⊃ {replay, downscale, restructure, recombine}`, `p_max ⊃ {replay, downscale, restructure, recombine}`.
- **Existing conformance tests** : `tests/conformance/axioms/test_dr{0,1,3}_diffusion*.py` — must remain green.
- **R1 tests** : `tests/reproducibility/test_r1_diffusion.py` (3 tests : train / sample / full_pipeline), compare against `golden_hashes_apple_{m5,m3_ultra,m1_max}.json`. These hashes **will change** under M7 — regenerate.

---

## File structure

| File | Role | Task |
|---|---|---|
| `kiki_oniric/substrates/_diffusion/decoder.py` | New : random-init MLP `d_latent → 3072` for the recombine VAE-shape contract (D7) | 1 |
| `kiki_oniric/substrates/_diffusion/cl_eval_head.py` | New : 1-layer classifier head + before/after eval helpers for `delta_acc` (D2) | 2 |
| `kiki_oniric/substrates/_diffusion/dream_ops_adapter.py` | New : bind real handlers onto a profile's runtime | 3 |
| `kiki_oniric/substrates/_diffusion/__init__.py` | Re-export `Decoder`, `ClEvalHead`, `bind_real_handlers` | 1, 2, 3 |
| `kiki_oniric/substrates/mlx_latent_diffusion.py` | Rewrite `execute_profile`, add `_decoder` / `_cl_head` to `__init__`, bump version constant to `C-v0.15.0+PARTIAL` | 4 |
| `tests/conformance/axioms/test_dr3_diffusion_profile.py` | New : prove the substrate honours the §3.1 activation surface | 5 |
| `tests/reproducibility/golden_hashes_apple_{m5,m3_ultra,m1_max}.json` | Regenerate (3 entries change under M7) | 6 |
| `tests/reproducibility/REBASELINE_NOTE.md` | Append M7 rebaseline entry | 6 |
| `pyproject.toml` | SemVer `0.22.2 → 0.23.0` (FC MINOR alias) | 7 |
| `CHANGELOG.md` | New section `[C-v0.15.0+PARTIAL] — 2026-05-2X` | 7 |

(`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §3 is unchanged — the spec already pins the normative table. No FR mirror change.)

---

### Task 1: Decoder MLP module

**Files:**
- Create: `kiki_oniric/substrates/_diffusion/decoder.py`
- Test: `tests/unit/test_diffusion_decoder.py`

- [ ] **Step 1: Read the existing Encoder for style match**

Run: `cat kiki_oniric/substrates/_diffusion/encoder.py` (or `model.py` if Encoder lives there). Note the class shape, constructor parameter names, `__call__` signature, MLX import style. Match it.

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_diffusion_decoder.py
"""Decoder MLP for the recombine VAE-shape contract (M7 D7)."""

from __future__ import annotations

import mlx.core as mx

from kiki_oniric.substrates._diffusion.decoder import Decoder


def test_decoder_maps_d_latent_to_3072() -> None:
    decoder = Decoder(d_latent=64, d_out=3072)
    z = mx.zeros((4, 64))
    out = decoder(z)
    assert out.shape == (4, 3072)


def test_decoder_is_seed_deterministic() -> None:
    decoder_a = Decoder(d_latent=64, d_out=3072)
    decoder_b = Decoder(d_latent=64, d_out=3072)
    # Two random inits produce different outputs (sanity that the
    # random init is actually firing).
    z = mx.zeros((1, 64))
    assert not mx.allclose(decoder_a(z), decoder_b(z))
```

- [ ] **Step 3: Run, expect FAIL (module missing)**

Run: `uv run pytest tests/unit/test_diffusion_decoder.py -v --no-cov`

- [ ] **Step 4: Implement `Decoder`**

```python
# kiki_oniric/substrates/_diffusion/decoder.py
"""Random-init MLP decoder for the recombine VAE-shape contract.

The diffusion substrate's denoiser is *not* a VAE decoder — it
predicts noise. The recombine handler's Protocol (defined in
``kiki_oniric/dream/operations/recombine_real.py``) expects a
``VAEDecoder.__call__(z) -> mx.array`` mapping latents back to
raw features. This module provides a minimal random-init MLP that
satisfies the Protocol *shape* without claiming any reconstruction
fidelity — see ``docs/superpowers/specs/2026-05-21-m7-substrate-
dr3-design.md`` §3 D7.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class Decoder(nn.Module):
    """``d_latent -> d_out`` MLP. Two hidden layers, GELU."""

    def __init__(
        self, d_latent: int, d_out: int, d_hidden: int = 256
    ) -> None:
        super().__init__()
        self.up = nn.Linear(d_latent, d_hidden)
        self.mid = nn.Linear(d_hidden, d_hidden)
        self.out = nn.Linear(d_hidden, d_out)

    def __call__(self, z: "mx.array") -> "mx.array":
        h = nn.gelu(self.up(z))
        h = nn.gelu(self.mid(h))
        return self.out(h)
```

If the existing `Encoder` uses a different activation or layer count, match it for consistency — the goal is style parity with the sibling building block, not a research claim about decoder choice.

- [ ] **Step 5: Verify test passes**

Run: `uv run pytest tests/unit/test_diffusion_decoder.py -v --no-cov`

- [ ] **Step 6: Re-export from package**

In `kiki_oniric/substrates/_diffusion/__init__.py`, add `Decoder` to the `__all__` list (or equivalent) alongside `Encoder`. Confirm the import works:

Run: `uv run python -c "from kiki_oniric.substrates._diffusion import Decoder; print(Decoder)"`

- [ ] **Step 7: Commit**

```bash
git add kiki_oniric/substrates/_diffusion/decoder.py kiki_oniric/substrates/_diffusion/__init__.py tests/unit/test_diffusion_decoder.py
git commit -m "$(cat <<'EOF'
feat(substrate): diffusion Decoder for recombine

M7 D7. Random-init MLP d_latent -> 3072 to satisfy the
VAEDecoder Protocol of recombine_real_handler. Not a learned
decoder claim; provides the Protocol shape so recombine is
well-defined on the diffusion substrate.
EOF
)"
```

---

### Task 2: Classifier eval head

**Files:**
- Create: `kiki_oniric/substrates/_diffusion/cl_eval_head.py`
- Test: `tests/unit/test_diffusion_cl_eval_head.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_diffusion_cl_eval_head.py
"""Small CL classifier head for diffusion-substrate delta_acc (M7 D2)."""

from __future__ import annotations

import mlx.core as mx

from kiki_oniric.substrates._diffusion.cl_eval_head import (
    ClEvalHead,
    eval_head_accuracy,
)


def test_cl_eval_head_shapes() -> None:
    head = ClEvalHead(d_latent=64, n_classes=20)
    z = mx.zeros((8, 64))
    logits = head(z)
    assert logits.shape == (8, 20)


def test_eval_head_accuracy_in_unit_range() -> None:
    head = ClEvalHead(d_latent=64, n_classes=20)
    z = mx.zeros((16, 64))
    y = mx.zeros((16,), dtype=mx.int32)
    acc = eval_head_accuracy(head, z, y)
    assert 0.0 <= acc <= 1.0
```

- [ ] **Step 2: Implement**

```python
# kiki_oniric/substrates/_diffusion/cl_eval_head.py
"""1-layer classifier head + train/eval helpers for delta_acc.

The M7 substrate measures per-cell `delta_acc` = accuracy on the
task's val split AFTER the dream cycle minus accuracy BEFORE the
dream cycle, on a tiny 1-layer Linear head trained on the
substrate's latent space. Hyper-parameters pinned here so the
bench is byte-deterministic.

See docs/superpowers/specs/2026-05-21-m7-substrate-dr3-design.md
§3 D2.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# Pinned hyper-parameters — do not vary per cell.
_HEAD_STEPS = 50
_HEAD_LR = 1e-2


class ClEvalHead(nn.Module):
    """One ``nn.Linear`` from latent to class logits."""

    def __init__(self, d_latent: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_latent, n_classes)

    def __call__(self, z: "mx.array") -> "mx.array":
        return self.linear(z)


def train_head_inplace(
    head: ClEvalHead,
    latents: "mx.array",
    labels: "mx.array",
    *,
    steps: int = _HEAD_STEPS,
    lr: float = _HEAD_LR,
) -> None:
    """Train the head in-place on ``(latents, labels)``.

    Mutates ``head``; deterministic given a fixed MLX seed upstream.
    """
    import mlx.optimizers as optim

    optimizer = optim.SGD(learning_rate=lr)

    def loss_fn(h: ClEvalHead, z: "mx.array", y: "mx.array") -> "mx.array":
        logits = h(z)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(head, loss_fn)
    for _ in range(steps):
        _, grads = loss_and_grad(head, latents, labels)
        optimizer.update(head, grads)
        mx.eval(head.parameters())


def eval_head_accuracy(
    head: ClEvalHead, latents: "mx.array", labels: "mx.array"
) -> float:
    logits = head(latents)
    preds = mx.argmax(logits, axis=-1)
    correct = (preds == labels).astype(mx.float32)
    return float(mx.mean(correct).item())
```

If the MLX-LM / MLX optimizer API on the installed version differs (e.g. `optim.SGD` lives elsewhere, or `nn.value_and_grad` takes different args), adapt to the real surface — the goal is a deterministic 50-step SGD on cross-entropy, however it spells.

- [ ] **Step 3: Verify**

Run: `uv run pytest tests/unit/test_diffusion_cl_eval_head.py -v --no-cov`

- [ ] **Step 4: Commit**

```bash
git add kiki_oniric/substrates/_diffusion/cl_eval_head.py tests/unit/test_diffusion_cl_eval_head.py
git commit -m "$(cat <<'EOF'
feat(substrate): diffusion CL eval head

M7 D2. One Linear layer, fixed 50-step SGD, used to measure
delta_acc per cell (acc_after_dream - acc_before_dream on the
task val split). Hyper-parameters pinned in-module.
EOF
)"
```

---

### Task 3: Dream-ops adapter

**Files:**
- Create: `kiki_oniric/substrates/_diffusion/dream_ops_adapter.py`
- Test: `tests/unit/test_dream_ops_adapter.py`

- [ ] **Step 1: Read `dream/runtime.py` and a profile `__post_init__`**

Run: `cat kiki_oniric/dream/runtime.py | head -100` and `cat kiki_oniric/profiles/p_equ.py`. Confirm:
- `DreamRuntime.register_handler(op, handler)` replaces an existing registration (no error).
- The skeleton `replay_handler` / `downscale_handler` / `restructure_handler` / `recombine_handler` factories take `state` positionally.
- `Operation` enum members : `REPLAY`, `DOWNSCALE`, `RESTRUCTURE`, `RECOMBINE`.

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_dream_ops_adapter.py
"""dream_ops_adapter: bind real handlers onto a profile runtime (M7 D3)."""

from __future__ import annotations

import mlx.core as mx

from kiki_oniric.dream.episode import Operation
from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_min import PMinProfile
from kiki_oniric.substrates._diffusion.decoder import Decoder
from kiki_oniric.substrates._diffusion.dream_ops_adapter import (
    bind_real_handlers,
)
from kiki_oniric.substrates._diffusion import Encoder


def test_bind_real_handlers_overrides_p_min_replay_downscale() -> None:
    profile = PMinProfile()
    model = type("Stub", (), {"layers": [], "parameters": lambda self: {}})()
    encoder = Encoder(d_in=3072, d_latent=64)
    decoder = Decoder(d_latent=64, d_out=3072)

    bound = bind_real_handlers(
        profile, model=model, encoder=encoder, decoder=decoder, seed=0
    )

    assert Operation.REPLAY in bound
    assert Operation.DOWNSCALE in bound
    assert Operation.RESTRUCTURE not in bound  # not activated for p_min
    assert Operation.RECOMBINE not in bound    # not activated for p_min


def test_bind_real_handlers_overrides_p_equ_all_four() -> None:
    profile = PEquProfile()
    model = type("Stub", (), {"layers": [], "parameters": lambda self: {}})()
    encoder = Encoder(d_in=3072, d_latent=64)
    decoder = Decoder(d_latent=64, d_out=3072)

    bound = bind_real_handlers(
        profile, model=model, encoder=encoder, decoder=decoder, seed=0
    )

    assert {
        Operation.REPLAY, Operation.DOWNSCALE,
        Operation.RESTRUCTURE, Operation.RECOMBINE,
    } <= bound
```

- [ ] **Step 3: Implement**

```python
# kiki_oniric/substrates/_diffusion/dream_ops_adapter.py
"""Bind ``_real.py`` handlers onto a profile's ``DreamRuntime``.

Each profile (``PMinProfile`` / ``PEquProfile`` / ``PMaxProfile``)
auto-registers *skeleton* handlers in ``__post_init__``. M7 takes
that profile, inspects which states it carries, and **overrides**
those registrations with the real MLX-backed handlers bound to the
diffusion substrate's denoiser + encoder + decoder.

This is the cleanest way to honour framework-C §3.1 normative
activation without duplicating the activation logic — the profile
stays the source of truth for "which ops are active".

See docs/superpowers/specs/2026-05-21-m7-substrate-dr3-design.md
§3 D3.
"""
from __future__ import annotations

from typing import Any

from kiki_oniric.dream.episode import Operation
from kiki_oniric.dream.operations.downscale_real import (
    DownscaleRealState,
    downscale_real_handler,
)
from kiki_oniric.dream.operations.recombine_real import (
    RecombineRealState,
    recombine_real_handler,
)
from kiki_oniric.dream.operations.replay_real import (
    ReplayRealState,
    replay_real_handler,
)
from kiki_oniric.dream.operations.restructure_real import (
    RestructureRealState,
    restructure_real_handler,
)


def bind_real_handlers(
    profile: Any,
    *,
    model: Any,
    encoder: Any,
    decoder: Any,
    seed: int,
) -> set[Operation]:
    """Override skeleton handlers on ``profile.runtime`` with real ones.

    Returns the set of Operations actually re-registered, derived
    from which ``*_state`` attributes the profile carries (the
    profile's own __post_init__ already picked the activation set
    per framework-C §3.1).

    Mutates ``profile.runtime`` in-place ; also swaps the profile's
    state fields to the ``*RealState`` variants so the handlers can
    write their K1 / metrics to the same address the substrate will
    snapshot later.
    """
    overridden: set[Operation] = set()

    if hasattr(profile, "replay_state"):
        profile.replay_state = ReplayRealState()
        profile.runtime.register_handler(
            Operation.REPLAY,
            replay_real_handler(profile.replay_state, model=model),
        )
        overridden.add(Operation.REPLAY)

    if hasattr(profile, "downscale_state"):
        profile.downscale_state = DownscaleRealState()
        profile.runtime.register_handler(
            Operation.DOWNSCALE,
            downscale_real_handler(profile.downscale_state, model=model),
        )
        overridden.add(Operation.DOWNSCALE)

    if hasattr(profile, "restructure_state"):
        profile.restructure_state = RestructureRealState()
        profile.runtime.register_handler(
            Operation.RESTRUCTURE,
            restructure_real_handler(
                profile.restructure_state, model=model
            ),
        )
        overridden.add(Operation.RESTRUCTURE)

    if hasattr(profile, "recombine_state"):
        profile.recombine_state = RecombineRealState()
        profile.runtime.register_handler(
            Operation.RECOMBINE,
            recombine_real_handler(
                profile.recombine_state,
                encoder=encoder, decoder=decoder, seed=seed,
            ),
        )
        overridden.add(Operation.RECOMBINE)

    return overridden
```

**If the profile's state field type is the *non-real* `ReplayOpState` (not `ReplayRealState`)**, swapping the field as above is required so the real handler writes to the same object the substrate reads after. If the state classes are actually identical or subclass-related, the swap is still safe (a fresh instance per cell mirrors the M5 per-cell fresh-substrate posture).

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/unit/test_dream_ops_adapter.py -v --no-cov`

- [ ] **Step 5: Commit**

```bash
git add kiki_oniric/substrates/_diffusion/dream_ops_adapter.py tests/unit/test_dream_ops_adapter.py
git commit -m "$(cat <<'EOF'
feat(substrate): bind real handlers onto profile

M7 D3. bind_real_handlers inspects which *_state fields a
profile carries, swaps them to RealState variants, and overrides
the profile.runtime registrations with the _real.py factories
bound to the diffusion substrate's denoiser / encoder / decoder.
EOF
)"
```

---

### Task 4: Rewrite `execute_profile`

**Files:**
- Modify: `kiki_oniric/substrates/mlx_latent_diffusion.py` (`__init__`, `execute_profile`, version constant)
- Test: `tests/unit/test_mlx_latent_diffusion_adapter.py` (existing — extend)

- [ ] **Step 1: Bump the substrate version constant**

In `mlx_latent_diffusion.py`, replace :
```python
MLX_LATENT_DIFFUSION_SUBSTRATE_VERSION = "C-v0.14.0+PARTIAL"
```
with :
```python
MLX_LATENT_DIFFUSION_SUBSTRATE_VERSION = "C-v0.15.0+PARTIAL"
```

- [ ] **Step 2: Add `_decoder` and `_cl_head` to `__init__`**

After the existing `self._cifar_encoder` and `self.denoiser` construction, add :

```python
        from kiki_oniric.substrates._diffusion.decoder import Decoder
        from kiki_oniric.substrates._diffusion.cl_eval_head import (
            ClEvalHead,
        )
        self._diffusion_decoder: Decoder = Decoder(
            d_latent=config.d_latent, d_out=self._CIFAR_D_IN,
        )
        self._cl_head: ClEvalHead = ClEvalHead(
            d_latent=config.d_latent,
            n_classes=20,  # CIFAR-100 task-window size; see plan note
        )
```

(`self._CIFAR_D_IN = 3072` already exists.)

- [ ] **Step 3: Write the failing test**

Append to `tests/unit/test_mlx_latent_diffusion_adapter.py` :

```python
def test_execute_profile_p_min_only_activates_replay_downscale() -> None:
    """Framework-C §3.1: p_min activates {replay, downscale}, no more."""
    import mlx.core as mx

    from kiki_oniric.substrates.mlx_latent_diffusion import (
        MLXLatentDiffusionSubstrate,
    )

    class _Req:
        seed = 0
        profile = "p_min"
        task_idx = 0
        loader_batches = (
            type("B", (), {
                "features": mx.zeros((8, 3072)),
                "labels": mx.zeros((8,), dtype=mx.int32),
                "task_idx": 0,
            })(),
        )

    metrics = MLXLatentDiffusionSubstrate().execute_profile(_Req())
    # restructure inactive for p_min -> total_reroutes stays 0
    assert metrics["restructure_sum"] == 0
    # recombine inactive for p_min -> _episode_count stays 0
    assert metrics["recombine_rate"] == 0
    # replay / downscale active -> compound_factor changed (downscale ran)
    assert metrics["downscale_norm"] != 1.0 or metrics["replay_rate"] != 0.0


def test_execute_profile_p_max_activates_all_four() -> None:
    import mlx.core as mx

    from kiki_oniric.substrates.mlx_latent_diffusion import (
        MLXLatentDiffusionSubstrate,
    )

    class _Req:
        seed = 0
        profile = "p_max"
        task_idx = 0
        loader_batches = (
            type("B", (), {
                "features": mx.zeros((8, 3072)),
                "labels": mx.zeros((8,), dtype=mx.int32),
                "task_idx": 0,
            })(),
        )

    metrics = MLXLatentDiffusionSubstrate().execute_profile(_Req())
    # All 4 ops active -> recombine ran at least once
    assert metrics["recombine_rate"] >= 1
```

- [ ] **Step 4: Run, expect FAIL**

Run: `uv run pytest tests/unit/test_mlx_latent_diffusion_adapter.py -v --no-cov`

- [ ] **Step 5: Rewrite the `execute_profile` body**

Replace the current `execute_profile` body (currently lines ~163-241 — read the actual range before editing) with this profile-aware version. Keep the docstring header, update it to cite M7. Keep the `t0 = time.perf_counter()` ; keep the existing synthetic-only fallback ; replace the loader-batch branch :

```python
    def execute_profile(self, request: "CellRequest | object") -> dict[str, object]:
        """Run one ablation cell honouring framework-C §3.1.

        M7 wiring: instantiate the profile dataclass, override its
        skeleton handlers with the diffusion-bound _real.py handlers
        via dream_ops_adapter.bind_real_handlers, drive one
        DreamEpisode per loader batch through the profile's runtime,
        and snapshot the per-op metrics off the profile's *RealState
        fields.

        See docs/superpowers/specs/2026-05-21-m7-substrate-dr3-design.md
        §3 D3 + D7.
        """
        import mlx.core as mx
        from kiki_oniric.dream.episode import (
            BudgetCap, DreamEpisode, EpisodeTrigger, Operation,
            OutputChannel,
        )
        from kiki_oniric.profiles.p_min import PMinProfile
        from kiki_oniric.profiles.p_equ import PEquProfile
        from kiki_oniric.profiles.p_max import PMaxProfile
        from kiki_oniric.substrates._diffusion.dream_ops_adapter import (
            bind_real_handlers,
        )
        from kiki_oniric.substrates._diffusion.cl_eval_head import (
            train_head_inplace, eval_head_accuracy,
        )

        seed = int(getattr(request, "seed", 0))
        profile_tag = str(getattr(request, "profile", "p_equ"))
        loader_batches = getattr(request, "loader_batches", ())

        root = mx.random.key(seed)
        train_root, sample_root, data_root, head_root = mx.random.split(
            root, num=4,
        )

        # Build the dataset: encoded latents (one per loader batch),
        # or a synthetic fallback identical to the M3 skeleton so
        # existing R1 hashes for the synthetic path stay aligned
        # (post-regeneration; see plan Task 6).
        if loader_batches:
            dataset = [
                self._encode_features(batch.features)
                for batch in loader_batches
            ]
            labels_per_batch = [batch.labels for batch in loader_batches]
            synthetic = False
        else:
            d_latent = self.config.d_latent
            n_batches = 4
            batch_size = 8
            data_keys = mx.random.split(data_root, num=n_batches)
            dataset = [
                mx.random.normal(shape=(batch_size, d_latent), key=k)
                for k in data_keys
            ]
            labels_per_batch = [
                mx.zeros((batch_size,), dtype=mx.int32)
                for _ in range(n_batches)
            ]
            synthetic = True

        # delta_acc baseline: train + eval the head BEFORE dream cycle.
        # All cells use the first batch as the eval slice for simplicity.
        baseline_acc = 0.0
        if dataset:
            train_head_inplace(
                self._cl_head, dataset[0], labels_per_batch[0],
            )
            baseline_acc = eval_head_accuracy(
                self._cl_head, dataset[0], labels_per_batch[0],
            )

        # Instantiate the profile and bind real handlers.
        profile_ctor = {
            "p_min": PMinProfile,
            "p_equ": PEquProfile,
            "p_max": PMaxProfile,
        }[profile_tag]
        profile = profile_ctor()
        activated = bind_real_handlers(
            profile, model=self.denoiser,
            encoder=self._cifar_encoder,
            decoder=self._diffusion_decoder, seed=seed,
        )

        t0 = time.perf_counter()

        # Drive one DreamEpisode per loader batch through the
        # profile's runtime. Episode input_slice carries every key
        # any handler might need; inactive ops are simply not in
        # operation_set.
        op_order = tuple(
            op for op in (
                Operation.REPLAY, Operation.DOWNSCALE,
                Operation.RESTRUCTURE, Operation.RECOMBINE,
            ) if op in activated
        )
        for batch_idx, latents in enumerate(dataset):
            # encode latents to numpy-shaped records for replay /
            # recombine that expect Python-side payloads
            records = [
                {"x": latents[i], "y": labels_per_batch[batch_idx][i]}
                for i in range(latents.shape[0])
            ]
            episode = DreamEpisode(
                trigger=EpisodeTrigger.PERIODIC,
                input_slice={
                    "beta_records": records,
                    "shrink_factor": 0.95,
                    "topo_op": "reroute",
                    "swap_indices": (0, min(1, self.config.n_layers - 1)),
                    "delta_latents": [latents[i] for i in range(latents.shape[0])],
                    "species": "diffusion",
                },
                operation_set=op_order,
                output_channels=(),
                budget=BudgetCap(),
                episode_id=f"diff/{profile_tag}/seed={seed}/b={batch_idx}",
            )
            profile.runtime.execute(episode)

        # delta_acc post-cycle: same head, eval again after the dream.
        post_acc = 0.0
        if dataset:
            post_acc = eval_head_accuracy(
                self._cl_head, dataset[0], labels_per_batch[0],
            )

        wall = time.perf_counter() - t0

        # Read metrics off the profile's state fields. Inactive ops
        # return field defaults (legitimate zeros — that is the no-op
        # semantic per framework-C §3.1).
        replay_rate = float(profile.replay_state.last_loss or 0.0) \
            if hasattr(profile, "replay_state") else 0.0
        downscale_norm = float(profile.downscale_state.compound_factor) \
            if hasattr(profile, "downscale_state") else 1.0
        restructure_sum = int(profile.restructure_state.total_reroutes) \
            if hasattr(profile, "restructure_state") else 0
        recombine_rate = int(profile.recombine_state._episode_count) \
            if hasattr(profile, "recombine_state") else 0
        op_flops_total = sum(
            getattr(getattr(profile, f"{name}_state"), "last_compute_flops", 0)
            for name in ("replay", "downscale", "restructure", "recombine")
            if hasattr(profile, f"{name}_state")
        )

        return {
            "replay_rate": replay_rate,
            "downscale_norm": downscale_norm,
            "restructure_sum": restructure_sum,
            "recombine_rate": recombine_rate,
            "delta_acc": post_acc - baseline_acc,
            "op_flops_total": int(op_flops_total),
            "wall_time_s": wall,
            "synthetic": synthetic,
            "profile": profile_tag,
            "seed": seed,
            "substrate": MLX_LATENT_DIFFUSION_SUBSTRATE_NAME,
            "substrate_version": MLX_LATENT_DIFFUSION_SUBSTRATE_VERSION,
        }
```

Notes for the implementer :
- **`BudgetCap` constructor**. Read `kiki_oniric/dream/episode.py` for the signature ; if it requires non-trivial args, use sensible defaults that pass the existing DR-0 tests. If the project has a `BudgetCap.default()` or similar, use it.
- **`EpisodeTrigger.PERIODIC` vs `MANUAL`** : pick whichever is canonical (read `episode.py` enum). If `PERIODIC` doesn't exist, use the first enum member.
- **`profile.runtime.execute(episode)`** may emit a `LatentSample` (recombine) or `None` (other ops) ; the return value is ignored in M7 (channels infra not in scope per spec §2).
- If `replay_real_handler` raises on the `records` shape (e.g. expects flat python floats, not `mx.array`), convert the latents to numpy/python on the fly and re-test.

- [ ] **Step 6: Verify all adapter tests pass**

Run: `uv run pytest tests/unit/test_mlx_latent_diffusion_adapter.py -v --no-cov`

Expected: PASS — the two new p_min / p_max activation tests + all pre-existing adapter tests (they may need adapting if the metric shape changed — e.g. `test_execute_profile_runs_minimal_smoke_cycle` asserts on specific metric keys ; update its assertions to the M7 schema).

- [ ] **Step 7: Verify diffusion conformance tests still pass**

Run: `uv run pytest tests/conformance/axioms/test_dr0_diffusion_de_budget.py tests/conformance/axioms/test_dr1_diffusion_finite.py tests/conformance/axioms/test_dr3_diffusion_substrate.py -v --no-cov`

Expected: PASS. If any test asserts on the old hard-coded `restructure_sum == 0` or the M5 proxy metric values, update the test to the M7 schema (the test was asserting on the M3-era proxy ; the M7 schema is the truth source post-spec).

- [ ] **Step 8: Commit**

```bash
git add kiki_oniric/substrates/mlx_latent_diffusion.py tests/unit/test_mlx_latent_diffusion_adapter.py tests/conformance/axioms/test_dr3_diffusion_substrate.py
git commit -m "$(cat <<'EOF'
feat(substrate): diffusion honours framework-C 3.1

M7 D3. execute_profile instantiates the profile dataclass,
overrides skeleton handlers with the diffusion-bound _real.py
factories, and drives one DreamEpisode per batch through the
profile.runtime. Metrics read off the *RealState fields; inactive
ops return field defaults (the no-op semantic). delta_acc is now
a real CL head before/after measurement. Substrate version bumped
to C-v0.15.0+PARTIAL.
EOF
)"
```

---

### Task 5: New conformance test

**Files:**
- Create: `tests/conformance/axioms/test_dr3_diffusion_profile.py`

- [ ] **Step 1: Write the test (this IS the conformance assertion)**

```python
# tests/conformance/axioms/test_dr3_diffusion_profile.py
"""DR-3 Conformance: mlx_latent_diffusion activates exactly the
framework-C §3.1 primitive set per profile.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md
§3.1 (normative table) + §3.2 (monotonic inclusion).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from kiki_oniric.dream.episode import Operation
from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_max import PMaxProfile
from kiki_oniric.profiles.p_min import PMinProfile
from kiki_oniric.substrates._diffusion.decoder import Decoder
from kiki_oniric.substrates._diffusion import Encoder
from kiki_oniric.substrates._diffusion.dream_ops_adapter import (
    bind_real_handlers,
)


def _model_stub():
    return type("Stub", (), {"layers": [], "parameters": lambda self: {}})()


def _enc_dec():
    return Encoder(d_in=3072, d_latent=64), Decoder(d_latent=64, d_out=3072)


@pytest.mark.parametrize(
    "profile_ctor, expected",
    [
        (PMinProfile, {Operation.REPLAY, Operation.DOWNSCALE}),
        (PEquProfile, {
            Operation.REPLAY, Operation.DOWNSCALE,
            Operation.RESTRUCTURE, Operation.RECOMBINE,
        }),
        (PMaxProfile, {
            Operation.REPLAY, Operation.DOWNSCALE,
            Operation.RESTRUCTURE, Operation.RECOMBINE,
        }),
    ],
)
def test_diffusion_profile_activates_exact_set(
    profile_ctor, expected,
) -> None:
    """Each profile activates EXACTLY the §3.1 op set on the substrate."""
    profile = profile_ctor()
    encoder, decoder = _enc_dec()
    activated = bind_real_handlers(
        profile, model=_model_stub(), encoder=encoder, decoder=decoder,
        seed=0,
    )
    assert activated == expected


def test_diffusion_profile_inclusion_chain_is_monotonic() -> None:
    """§3.2: ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)."""
    enc, dec = _enc_dec()
    a = bind_real_handlers(
        PMinProfile(), model=_model_stub(),
        encoder=enc, decoder=dec, seed=0,
    )
    enc, dec = _enc_dec()
    b = bind_real_handlers(
        PEquProfile(), model=_model_stub(),
        encoder=enc, decoder=dec, seed=0,
    )
    enc, dec = _enc_dec()
    c = bind_real_handlers(
        PMaxProfile(), model=_model_stub(),
        encoder=enc, decoder=dec, seed=0,
    )
    assert a <= b <= c
```

- [ ] **Step 2: Run, expect PASS**

Run: `uv run pytest tests/conformance/axioms/test_dr3_diffusion_profile.py -v --no-cov`

Expected: PASS for both tests. If a profile activates a different set than the §3.1 table, the test fails and the profile (or the adapter) is wrong — investigate before patching the test.

- [ ] **Step 3: Commit**

```bash
git add tests/conformance/axioms/test_dr3_diffusion_profile.py
git commit -m "$(cat <<'EOF'
test(conformance): DR-3 diffusion profile activation

Assert the diffusion substrate's activation surface matches
framework-C 3.1 exactly per profile, and that the monotonic
inclusion of 3.2 holds. Closes issue #36.
EOF
)"
```

---

### Task 6: Regenerate R1 hashes (3 chip families)

**Files:**
- Modify: `tests/reproducibility/golden_hashes_apple_{m5,m3_ultra,m1_max}.json` (regenerate)
- Modify: `tests/reproducibility/REBASELINE_NOTE.md` (append M7 entry)

- [ ] **Step 1: Run on local Apple Silicon machine to regenerate**

On the local Apple Silicon machine the implementer is running on (M5 / M3 Ultra / M1 Max — `sysctl machdep.cpu.brand_string` to confirm), run :

```bash
uv run pytest tests/reproducibility/test_r1_diffusion.py -v --no-cov
```

Expected: 3 tests fail with `R1 hash mismatch` (the M7 substrate rewrite changes the substrate output bytes). The test harness `compare_or_bootstrap` writes the new hash to the chip's `golden_hashes_apple_<family>.json` with `status: "pending_review"`.

If the test exits 0 with `pending_review` instead of failing — it means the entry was deleted before regeneration. Either way, inspect the JSON to confirm the 3 entries are present with the new hashes.

- [ ] **Step 2: Promote the bootstrapped entries**

Edit the chip's `golden_hashes_apple_<family>.json` : flip `"status": "pending_review"` → `"status": "accepted"` only after verifying the substrate change is intentional. The test re-run with promoted hashes should be GREEN.

Re-run: `uv run pytest tests/reproducibility/test_r1_diffusion.py -v --no-cov` → GREEN.

For the OTHER two chip families : ship the JSON with `"status": "pending_review"` ; cross-machine R1 regeneration happens on those machines in a follow-up (consistent with the existing 2026-05-20 cross-machine R1 posture documented in `STATUS.md`).

- [ ] **Step 3: Append a REBASELINE_NOTE entry**

In `tests/reproducibility/REBASELINE_NOTE.md`, append the existing dated-entry format with a 2026-05-2X entry for M7 :

```markdown

## 2026-05-2X — M7 substrate DR-3 conformance fix

FC MINOR `C-v0.14.0 → C-v0.15.0+PARTIAL`. The
`mlx_latent_diffusion` substrate rewrites `execute_profile` to
honour framework-C §3.1 profile activation. This changes the
substrate's R1-hashable byte trace.

Regenerated entries (3) :
- `test_r1_diffusion_train`
- `test_r1_diffusion_sample`
- `test_r1_diffusion_full_pipeline`

Promotion : the locally-regenerated chip family is `accepted` ;
the other two stay `pending_review` until cross-machine re-runs.

Reference : `docs/superpowers/specs/2026-05-21-m7-substrate-dr3-design.md`.
```

Use the actual date the rebaseline ran ; replace `2026-05-2X`.

- [ ] **Step 4: Commit**

```bash
git add tests/reproducibility/golden_hashes_apple_*.json tests/reproducibility/REBASELINE_NOTE.md
git commit -m "$(cat <<'EOF'
test(r1): regenerate diffusion hashes for M7

3 R1 entries change under the substrate rewrite. Local chip
family accepted; the other two stay pending_review per the
existing cross-machine R1 posture.
EOF
)"
```

---

### Task 7: SemVer bump + CHANGELOG + full suite

**Files:**
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump SemVer**

In `pyproject.toml`, change `version = "0.22.2"` to `version = "0.23.0"` (FC MINOR alias).

- [ ] **Step 2: Add CHANGELOG section**

Add a new section above `[Unreleased]` (or replace `[Unreleased]` if its bullets are M7-only) :

```markdown
## [C-v0.15.0+PARTIAL] — 2026-05-2X

### Changed
- M7 substrate DR-3 conformance fix. `mlx_latent_diffusion.execute_profile`
  now honours framework-C §3.1 normative activation per profile :
  p_min activates {replay, downscale}, p_equ + p_max activate all
  four ops. Skeleton handlers in `__post_init__` are overridden by
  the `_real.py` factories bound to the diffusion denoiser via
  `_diffusion.dream_ops_adapter.bind_real_handlers`. The recombine
  contract is satisfied via a new random-init `Decoder` MLP
  (`_diffusion.decoder`). `delta_acc` is now a real CL head
  measurement (`_diffusion.cl_eval_head`). Closes issue #36.

### Added
- `kiki_oniric/substrates/_diffusion/decoder.py` (Decoder MLP).
- `kiki_oniric/substrates/_diffusion/cl_eval_head.py` (ClEvalHead +
  train/eval helpers).
- `kiki_oniric/substrates/_diffusion/dream_ops_adapter.py`
  (`bind_real_handlers`).
- `tests/conformance/axioms/test_dr3_diffusion_profile.py`.

### DualVer
- FC MINOR `C-v0.14.0 → C-v0.15.0`. EC stays `+PARTIAL` ; M6
  ship-critic decides STABLE / PARTIAL.
- 3 R1 entries regenerated ; see
  `tests/reproducibility/REBASELINE_NOTE.md`.
```

- [ ] **Step 3: Full suite**

```bash
uv run pytest
uv run ruff check .
uv run mypy harness tests kiki_oniric
```

Expected: all GREEN. Coverage gate satisfied. M7 only changed substrate + adapter code ; pre-existing unrelated mypy errors (2 in `harness/real_benchmarks/`) stay untouched.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "$(cat <<'EOF'
chore: bump to C-v0.15.0+PARTIAL for M7

FC MINOR. CHANGELOG documents the profile-activation fix, the
new decoder + cl_eval_head + dream_ops_adapter modules, and the
3 R1 regenerations.
EOF
)"
```

---

## Post-implementation: bench v3 on macM1 (separate runbook)

The plan ends here for code. After Task 7, the post-M7 450-cell bench
v3 is the M6 dependency (per the sequencing spec §5). The runbook :

1. `git push origin feat/m7-substrate-dr3` ; open PR.
2. PR review + critic pass.
3. Merge to `main`.
4. On macM1, `~/dream-of-kiki` clone, pull main, `uv sync --all-extras`.
5. Same M5 pre-flight (`HF_HUB_DISABLE_XET=1`, fresh registry).
6. `uv run python scripts/ablation_cycle3_diffusion.py --num-seeds 30 --resume --output docs/milestones/wave3b-bench-2026-05-2X.json`
7. Commit milestone, push, open follow-up PR.
8. Feed the new milestone into M6's spec §7.9 draft.

---

## Self-Review

**Spec coverage** : D1 boundary → Task 3 ; D2 metrics → Task 4 + 2 ; D3 dispatch → Task 4 (drives `profile.runtime.execute`) + Task 3 (`bind_real_handlers`) ; D4 FC MINOR → Task 7 ; D5 R1 regeneration → Task 6 ; D6 acceptance → re-bench (out of plan, runbook documented above) ; D7 VAE-shape decoder → Task 1 + Task 4 wiring.

**Placeholder scan** : `2026-05-2X` is the deliberately unresolved date for the rebaseline + CHANGELOG section + bench v3 (resolved at execution time). No "TBD" / "implement later" / "add error handling".

**Type consistency** : `bind_real_handlers(profile, *, model, encoder, decoder, seed) -> set[Operation]` (Task 3) ↔ called with those exact args from Task 4 ↔ tested in Tasks 3 + 5. `ClEvalHead(d_latent, n_classes)` (Task 2) ↔ instantiated in Task 4 with `n_classes=20`. `Decoder(d_latent, d_out, d_hidden=256)` (Task 1) ↔ instantiated in Task 4 as `Decoder(d_latent=config.d_latent, d_out=self._CIFAR_D_IN)`.

**Known soft spots requiring implementer judgement** :

1. **`BudgetCap` and `EpisodeTrigger` constructors** — read `episode.py` first ; if either takes non-trivial args, use the project's canonical default factory.
2. **MLX optimizer API on the installed version** — `optim.SGD` and `nn.value_and_grad` may differ between MLX minor versions ; adapt while preserving determinism.
3. **`replay_real_handler` records shape** — the audit shows `input_slice["beta_records"]` is `list[{"x": ..., "y": ...}]` ; the dict values may need to be python-side floats, not `mx.array`. If the handler raises, convert via `latents[i].tolist()` / `.item()`.
4. **The existing diffusion adapter tests** (`test_execute_profile_runs_minimal_smoke_cycle` etc.) may assert on M5-era metric values that M7 changes. Update those tests in Task 4 Step 6 to the M7 schema rather than weakening the M7 assertions.
5. **Stub `_model_stub()` in Task 5** has no `.parameters()` returning real weights. The conformance tests check *activation surface*, not behaviour ; this is intentional. If `bind_real_handlers` calls a handler factory that touches the stub during binding (e.g. tries to count `_param_count(model)` at factory-build), the stub will need to be expanded to return enough shape for `_param_count` not to crash. Read `downscale_real.py:_param_count` and adapt the stub accordingly.

Mark these soft spots with `DONE_WITH_CONCERNS` if you make non-trivial deviations during implementation.
