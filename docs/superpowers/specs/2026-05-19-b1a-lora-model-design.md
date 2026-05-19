# B1a — LoRA-adapter model abstraction

**Date** : 2026-05-19
**Status** : design approved, pending spec review
**Tracking issue** : #15 (`hypneum-lab/dream-of-kiki`)
**Scope** : sub-project B1a — first half of B1, itself sub-project B1 of the issue-#15 "approach B" decomposition

---

## Context

Issue #15 / approach B makes the four dream operations produce real
channel outputs. B0 (done, `C-v0.14.0+PARTIAL`) froze the
channel-output contract. B1 makes `replay` emit a real channel-1
`WeightUpdate`. The maintainer chose the **true low-rank** variant:
`replay` must do a LoRA gradient update and emit the low-rank
update as `WeightUpdate.lora_delta`.

Exploration established that the repo has LoRA *utilities*
(`micro_kiki/oplora.py` orthogonal projector, `micro_kiki/ties.py`
TIES-merge, a safetensors loader) but **no LoRA-adapter model
abstraction**: no class with a frozen base, trainable A/B adapters,
and a forward pass. The `replay` MLX handler does SGD on *all*
parameters of a plain `TinyMLP`; nothing targets adapters.

B1 therefore decomposes into two sub-projects:

| # | Sub-project | Depends on |
|---|-------------|-----------|
| **B1a** | LoRA-adapter model abstraction | B0 |
| B1b | `replay` does LoRA-only SGD on it, emits `WeightUpdate` | B1a |

**This document specifies B1a only.**

## Problem

`replay` (B1b) needs a model whose forward pass is differentiable
and whose *only* trainable parameters are named, separable low-rank
adapters — so a gradient step touches the adapters and nothing
else, and the per-layer low-rank update can be read back as a
`dict[str, NDArray]` for `WeightUpdate.lora_delta`. No such model
exists. B1a builds it.

## Approaches considered

1. **MLX, self-contained `LoRALinear` + `LoRAModel`.** A LoRA
   linear layer (MLX `nn.Module`: frozen base + trainable A/B) and
   a small stack of named `LoRALinear` layers. MLX gives free
   autodiff for B1b's SGD; the existing `replay_handler_mlx`
   already uses `nn.value_and_grad`; repo policy mandates MLX on
   Apple Silicon. **Chosen.**
2. **numpy with hand-derived gradients.** Consistent with the
   pure-numpy OPLoRA/TIES utilities, but B1b would need a
   hand-written LoRA backward pass. Rejected: fragile, no autodiff.
3. **Wrap an arbitrary MLX model**, injecting adapters on its
   `Linear` layers. More general but requires walking the module
   tree. YAGNI at the substrate's skeleton stage. Rejected for
   B1a; a generic wrapper can come later.

## Design

### New module `kiki_oniric/substrates/micro_kiki/lora_model.py`

Co-located with `oplora.py` / `ties.py` (the existing LoRA
utilities).

### `LoRALinear(nn.Module)`

A LoRA-adapted linear layer.

- `weight` — base weight `W0`, shape `(out, in)`, **frozen**
  (not in the trainable-parameter set).
- `bias` — optional base bias, shape `(out,)`, frozen.
- `lora_a` — adapter matrix `A`, shape `(rank, in)`, trainable,
  initialised with small random values (scaled normal).
- `lora_b` — adapter matrix `B`, shape `(out, rank)`, trainable,
  **initialised to zeros** — standard LoRA init, so the initial
  effective weight equals the base weight.
- `rank: int`, `alpha: float`. Scale factor `alpha / rank`.
- Forward: `y = x @ (W0 + (alpha/rank) * (B @ A)).T` (`+ bias`).

Construction: `LoRALinear(in_features, out_features, rank, alpha,
*, bias=True, seed=...)`. Seeding is explicit (research-repo
determinism rule); the `A` init draws from a seeded generator.

### `LoRAModel(nn.Module)`

A small feed-forward stack of named `LoRALinear` layers.

- Construction: `LoRAModel(layer_sizes, rank, alpha, *, seed=...)`
  where `layer_sizes` is e.g. `(in, hidden, out)` → two
  `LoRALinear` layers named `layer0`, `layer1`.
- `__call__(x)` — sequential forward with a fixed non-linearity
  (ReLU) between layers, none after the last.
- `adapter_parameters() -> dict[str, mx.array]` — returns ONLY the
  trainable adapter arrays, keyed by a stable name per adapter
  matrix (e.g. `layer0.lora_a`, `layer0.lora_b`, …). These keys
  are what B1b maps into `WeightUpdate.lora_delta`.

### What B1a hands to B1b

A model that (1) has a differentiable forward pass, (2) whose only
trainable parameters are the named A/B adapters reachable via
`adapter_parameters()`, and (3) where the base weights are frozen.
B1b can then call `nn.value_and_grad` / an optimizer scoped to the
adapters and read the per-layer low-rank delta.

## Scope boundary

**B1a does** : `LoRALinear`, `LoRAModel`, `adapter_parameters()`,
standard LoRA init (`A` random / `B` zero), the `alpha/rank`
scaling, base-weight freezing, explicit seeding, and unit tests.

**B1a does not** : any SGD or training loop, any `WeightUpdate`
emission, any change to `replay` / the dream operations / the
profiles, any runtime wiring. All of that is B1b. B1a is a
standalone, independently testable model component.

## Test plan

New `tests/unit/test_lora_model.py`:

- `LoRALinear`: `lora_a` shape `(rank, in)`, `lora_b` shape
  `(out, rank)`; `lora_b` is all-zeros at construction; the
  initial forward equals the base-only forward (`B=0` ⇒ ΔW=0);
  a non-zero `lora_b` changes the output; the `alpha/rank` scale
  is applied; the base `weight` is not among the trainable params.
- `LoRAModel`: forward output shape matches the last layer size;
  `adapter_parameters()` returns exactly the A/B arrays (two per
  layer), correctly named; base weights are excluded; the model
  is deterministic under a fixed seed.

## DualVer

FC-**MINOR** (`C-v0.14.0 → C-v0.15.0`): a new substrate component,
no axiom, invariant, or primitive-signature change. EC unchanged.
`pyproject.toml` version `0.12.0 → 0.13.0`.

## Acceptance criteria

1. `lora_model.py` defines `LoRALinear` and `LoRAModel`.
2. Standard LoRA init: `A` seeded-random, `B` zeros ⇒ initial
   forward equals the base-only forward.
3. `adapter_parameters()` returns exactly the named A/B arrays;
   base weights are frozen and excluded.
4. The `alpha/rank` scaling is applied in the forward pass.
5. `tests/unit/test_lora_model.py` covers the test plan; full
   suite green, `uv run mypy harness tests` clean.
6. FC-MINOR DualVer bump recorded in `CHANGELOG.md`; framework-C
   spec + FR mirror note the new substrate component.
