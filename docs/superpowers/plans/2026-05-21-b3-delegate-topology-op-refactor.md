# B3 delegate refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the `_apply_topology_op(model, op, payload)` mutation kernel from `kiki_oniric/dream/channels/hierarchy_change.py` to a new substrate utility `kiki_oniric/substrates/micro_kiki/lora_topology_ops.py`, then have both `restructure_lora_handler` (B3) and `LoRAHierarchyChangeChannel.apply_diff` (B5) delegate to it. Zero behaviour change ; existing 14 B3 tests + B5 tests must all pass unchanged.

**Architecture:** A new utility module `kiki_oniric/substrates/micro_kiki/lora_topology_ops.py` hosts `_apply_topology_op` (body verbatim from `hierarchy_change.py`). The channel file becomes a one-line re-import (drops local definition, keeps the symbol in `__all__` for backwards compat — B5's published surface stays identical). The handler file (`restructure_real.py`) imports the kernel and rewrites its three per-op mutation blocks to build a payload then delegate, while keeping the dream-side responsibilities (seed derivation, snapshot capture for remove, sha256 post-mutation, state counters) at the call site.

**Tech Stack:** Python 3.12+, `uv`, MLX (`mlx.core`), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-05-21-b3-delegate-topology-op-refactor-design.md`

**Version skew note:** the spec was written against `C-v0.23.0+PARTIAL` / pyproject `0.21.0`. The repo HEAD has since advanced to `C-v0.24.0+PARTIAL` / pyproject `0.22.0` (Wave 3b M3 bump). This plan targets the CURRENT versions : FC-PATCH `C-v0.24.0 → C-v0.24.1`, pyproject `0.22.0 → 0.22.1`. Task 4 amends the spec inline alongside the version bump.

**Critical not-a-bug:** the helper's bounds checks (`0 <= index <= len(model.layers)` etc.) remain. B3 will never trigger them — its `_validate_topo_op` pre-mutation pass already rules them out — but B5's path through the helper can still hit them when a malformed `TopologyDiff` lands. Defense-in-depth, not dead code.

---

## File Structure

- **Create** `kiki_oniric/substrates/micro_kiki/lora_topology_ops.py` — canonical home for `_apply_topology_op`.
- **Modify** `kiki_oniric/dream/channels/hierarchy_change.py` — drop local def, add import + re-export.
- **Modify** `kiki_oniric/dream/operations/restructure_real.py` — delegate three per-op mutations.
- **Create** `tests/unit/test_lora_topology_ops_reexport.py` — 1 test pinning the re-export path.
- **Modify** `pyproject.toml` — version `0.22.0 → 0.22.1`.
- **Modify** `CHANGELOG.md` — one Refactored bullet under `[Unreleased]`.
- **Modify** `docs/superpowers/specs/2026-05-21-b3-delegate-topology-op-refactor-design.md` — amend the DualVer + version numbers to match the actual HEAD versions.

No new framework-C spec note ; the refactor is internal.

---

## Task 1: Extract `_apply_topology_op` to the substrate utility module

**Files:**
- Create: `kiki_oniric/substrates/micro_kiki/lora_topology_ops.py`
- Modify: `kiki_oniric/dream/channels/hierarchy_change.py`

- [ ] **Step 1: Create the new module**

Create `kiki_oniric/substrates/micro_kiki/lora_topology_ops.py`:

```python
"""LoRA-substrate topology mutation kernel.

Single source of truth for ``add`` / ``remove`` / ``reroute`` on
a ``LoRAModel.layers`` stack. Imported by two call sites :

- ``kiki_oniric.dream.operations.restructure_real.restructure_lora_handler``
  (B3 + the 2026-05-21 delegate refactor) — for the dream-side
  mutation + ``TopologyDiff`` emission.
- ``kiki_oniric.dream.channels.hierarchy_change.LoRAHierarchyChangeChannel
  .apply_diff`` (B5) — for the awake-side replay of an emitted
  ``TopologyDiff``.

The helper validates ``op`` + indices defensively even though
the dream-side caller (B3) validates upfront via
``_validate_topo_op`` — defense-in-depth ; the channel-side
caller (B5) trusts ``TopologyDiff.__post_init__`` to have
validated the diff's structural correctness, but bounds against
the current ``model.layers`` length is something only the
mutating call can check.

The ``add`` path uses the ``seed`` field that B3 stores in the
payload to call ``mx.random.key(seed)`` — the R1 linchpin : the
reconstructed ``LoRALinear`` is bit-identical to the one created
by ``restructure_lora_handler``.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


def _apply_topology_op(
    model: "LoRAModel",
    op: str,
    payload: dict[str, object],
) -> None:
    """Apply one topology op from a ``TopologyDiff`` entry onto *model*.

    ``add``    — insert a new ``LoRALinear`` at ``payload["index"]``,
                 reconstructed via ``mx.random.key(payload["seed"])``
                 for R1 bit-exactness.
    ``remove`` — pop the layer at ``payload["index"]``.  The snapshot
                 stored in the payload is not re-applied here (undo
                 logic is future work); only the layer is removed so
                 that the topology matches the dream-side post-state.
    ``reroute`` — swap the two layers at ``payload["swap_indices"]``.

    Raises ``ValueError`` with an ``"S3:"`` prefix on any structural
    defect (unknown op, index out of bounds).
    """
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRALinear

    _VALID_OPS = frozenset({"add", "remove", "reroute"})
    if op not in _VALID_OPS:
        raise ValueError(
            f"S3: _apply_topology_op unknown op {op!r}; "
            f"must be one of {sorted(_VALID_OPS)}"
        )

    if op == "add":
        index = int(payload["index"])  # type: ignore[arg-type]
        in_features = int(payload["in_features"])  # type: ignore[arg-type]
        out_features = int(payload["out_features"])  # type: ignore[arg-type]
        rank = int(payload["rank"])  # type: ignore[arg-type]
        alpha = float(payload["alpha"])  # type: ignore[arg-type]
        seed = int(payload["seed"])  # type: ignore[arg-type]
        if not (0 <= index <= len(model.layers)):
            raise ValueError(
                f"S3: add index {index} out of bounds for "
                f"{len(model.layers)} layers"
            )
        new_layer = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            key=mx.random.key(seed),
        )
        model.layers.insert(index, new_layer)

    elif op == "remove":
        rm_at = int(payload["index"])  # type: ignore[arg-type]
        if not (0 <= rm_at < len(model.layers)):
            raise ValueError(
                f"S3: remove index {rm_at} out of bounds for "
                f"layers of length {len(model.layers)}"
            )
        model.layers.pop(rm_at)

    else:  # reroute
        swap = payload["swap_indices"]
        i, j = int(swap[0]), int(swap[1])  # type: ignore[index]
        layers_len = len(model.layers)
        if not (0 <= i < layers_len and 0 <= j < layers_len):
            raise ValueError(
                f"S3: reroute swap_indices ({i}, {j}) out of bounds "
                f"for layers of length {layers_len}"
            )
        model.layers[i], model.layers[j] = (
            model.layers[j],
            model.layers[i],
        )


__all__ = ["_apply_topology_op"]
```

(Body lines 26-96 are verbatim from `hierarchy_change.py`. The only additions are the new docstring header explaining the two call sites + a local `__all__`.)

- [ ] **Step 2: Modify `hierarchy_change.py` to re-export from the new module**

Open `kiki_oniric/dream/channels/hierarchy_change.py`. Locate the local `_apply_topology_op` definition at lines 26-96 and the surrounding imports. Apply two changes :

**(2a)** Replace the imports at the top + drop the local function definition. The new top block reads :

```python
"""LoRA-target concrete implementation of HierarchyChangeChannel (B5).

Applies a ``TopologyDiff`` (channel-3 output) onto a ``LoRAModel``
adapter stack, reconstructing or undoing each topology mutation
bit-exactly. The ``add`` path uses the ``seed`` field that B3 stores
in the payload to call ``mx.random.key(seed)`` — this is the R1
linchpin: the reconstructed ``LoRALinear`` is identical to the one
created by ``restructure_lora_handler``.

The per-op mutation kernel ``_apply_topology_op`` lives in
``kiki_oniric.substrates.micro_kiki.lora_topology_ops`` (extracted
2026-05-21). It is re-exported from this module for backwards
compatibility — callers that imported it from here continue to work.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

from kiki_oniric.substrates.micro_kiki.lora_topology_ops import (
    _apply_topology_op,
)

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel
```

Then **delete** lines 26-96 of the original file (the local `def _apply_topology_op(...)` and its body).

**(2b)** Keep the rest of the file (the `LoRAHierarchyChangeChannel` class + `__all__`) unchanged. In particular `apply_diff` still calls `_apply_topology_op(self._target, op, payload)` — the symbol now resolves to the imported one. `__all__` keeps `"_apply_topology_op"` for the re-export.

- [ ] **Step 3: Verify the existing B5 tests still pass**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py -v`
Expected: all B5 tests (including the ones that exercise `LoRAHierarchyChangeChannel.apply_diff`) pass unchanged.

- [ ] **Step 4: Verify the helper is reachable from both paths**

Run from the repo root :
```bash
uv run python -c "
from kiki_oniric.dream.channels.hierarchy_change import _apply_topology_op as a
from kiki_oniric.substrates.micro_kiki.lora_topology_ops import _apply_topology_op as b
assert a is b
print('OK: re-export identity holds')
"
```
Expected: `OK: re-export identity holds`.

- [ ] **Step 5: Full sanity — pytest + mypy + ruff**

Run: `uv run pytest -q` — all pass.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/substrates/micro_kiki/lora_topology_ops.py kiki_oniric/dream/channels/hierarchy_change.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/substrates/micro_kiki/lora_topology_ops.py kiki_oniric/dream/channels/hierarchy_change.py
git commit -m "$(cat <<'EOF'
refactor(substrate): extract _apply_topology_op kernel

B3 delegate refactor / step 1. Moves _apply_topology_op from
kiki_oniric/dream/channels/hierarchy_change.py to a new
substrate-level module kiki_oniric/substrates/micro_kiki/
lora_topology_ops.py. The body is verbatim ; only the file
location changes.

hierarchy_change.py now imports the symbol and re-exports it
through __all__, so B5's published surface is unchanged.
LoRAHierarchyChangeChannel.apply_diff calls the imported name
without modification.

Zero behaviour change. Step 2 (delegate restructure_lora_handler)
follows in the next commit.
EOF
)"
```

---

## Task 2: Delegate `restructure_lora_handler` per-op mutations

**Files:**
- Modify: `kiki_oniric/dream/operations/restructure_real.py`

- [ ] **Step 1: Read the current per-op blocks**

Open `kiki_oniric/dream/operations/restructure_real.py` and locate `restructure_lora_handler` (~line 195 onwards) and its three per-op branches (`add` ~lines 249-279, `remove` ~lines 281-?, `reroute` ~last branch). Note the structure :

- The function takes `(state, *, model, max_adds_per_episode=1, seed=0)`.
- At handler-build time it does `import mlx.core as mx` and `from kiki_oniric.substrates.micro_kiki.lora_model import LoRALinear` and `from kiki_oniric.dream.channels import TopologyDiff`.
- Inside the closure, each per-op branch builds a payload and mutates `model.layers`.

- [ ] **Step 2: Add the new import**

Add at the module-level imports of `kiki_oniric/dream/operations/restructure_real.py` (top of file, after the existing imports) :

```python
from kiki_oniric.substrates.micro_kiki.lora_topology_ops import (
    _apply_topology_op,
)
```

- [ ] **Step 3: Drop the handler-build-time imports that are no longer needed**

Inside `restructure_lora_handler`, locate the lazy imports (right under the docstring, lines ~223-226). They currently look like :

```python
    import mlx.core as mx

    from kiki_oniric.dream.channels import TopologyDiff
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRALinear
```

Replace this block with only the `TopologyDiff` import (the handler still needs it for the return value) :

```python
    from kiki_oniric.dream.channels import TopologyDiff
```

The `mlx.core` and `LoRALinear` imports are no longer needed at the handler-build site because the kernel module owns them.

- [ ] **Step 4: Refactor the `add` branch**

Locate the `if op == "add":` branch inside the handler's for-loop. Replace its body with :

```python
            if op == "add":
                if state.adds_this_episode >= max_adds_per_episode:
                    # INSS soft cap: silently skip — no entry, no mutation.
                    continue
                op_seed = _derive_op_seed(seed, episode.episode_id, idx)
                insert_at = int(op_dict["index"])
                in_features = int(op_dict["in_features"])
                out_features = int(op_dict["out_features"])
                rank = int(op_dict["rank"])
                alpha = float(op_dict["alpha"])
                add_payload: dict[str, object] = {
                    "index": insert_at,
                    "in_features": in_features,
                    "out_features": out_features,
                    "rank": rank,
                    "alpha": alpha,
                    "seed": op_seed,
                    "model_sha256_post": "",
                }
                _apply_topology_op(model, "add", add_payload)
                add_payload["model_sha256_post"] = _model_sha256(model)
                state.adds_this_episode += 1
                state.total_adds += 1
                state.diff_history.append("add")
                applied.append(("add", add_payload))
```

Compared to the pre-refactor code : the `LoRALinear(...)` construction + `model.layers.insert(...)` are replaced by `_apply_topology_op(model, "add", add_payload)`. Seed derivation moves *above* the payload build so the helper sees the final seed in the payload. The `sha256_post` field is filled *after* the helper returns (so the hash reflects the post-mutation state).

- [ ] **Step 5: Refactor the `remove` branch**

Replace the body of `elif op == "remove":` :

```python
            elif op == "remove":
                rm_at = int(op_dict["index"])
                layer = model.layers[rm_at]
                bias_arr: np.ndarray | None
                if layer.use_bias:
                    bias_arr = np.asarray(
                        layer.bias, dtype=np.float32,
                    ).copy()
                else:
                    bias_arr = None
                snapshot: dict[str, object] = {
                    "base_weight": np.asarray(
                        layer.base_weight, dtype=np.float32,
                    ).copy(),
                    "lora_a": np.asarray(
                        layer.lora_a, dtype=np.float32,
                    ).copy(),
                    "lora_b": np.asarray(
                        layer.lora_b, dtype=np.float32,
                    ).copy(),
                    "bias": bias_arr,
                    "in_features": int(layer.in_features),
                    "out_features": int(layer.out_features),
                    "rank": int(layer.rank),
                    "alpha": float(layer.alpha),
                }
                remove_payload: dict[str, object] = {
                    "index": rm_at,
                    "snapshot": snapshot,
                    "model_sha256_post": "",
                }
                _apply_topology_op(model, "remove", remove_payload)
                remove_payload["model_sha256_post"] = _model_sha256(model)
                state.total_removes += 1
                state.diff_history.append("remove")
                applied.append(("remove", remove_payload))
```

The snapshot capture is unchanged ; only the `model.layers.pop(rm_at)` line is replaced by the helper call. The order matters : snapshot first (still inspecting the layer about to be removed), then helper (which pops), then `sha256_post`.

- [ ] **Step 6: Refactor the `reroute` branch**

Replace the body of the `else:` (or `elif op == "reroute":` depending on the current branch label) :

```python
            else:  # reroute
                i = int(op_dict["swap_indices"][0])
                j = int(op_dict["swap_indices"][1])
                reroute_payload: dict[str, object] = {
                    "swap_indices": (i, j),
                    "model_sha256_post": "",
                }
                _apply_topology_op(model, "reroute", reroute_payload)
                reroute_payload["model_sha256_post"] = _model_sha256(model)
                state.total_reroutes += 1
                state.diff_history.append("reroute")
                applied.append(("reroute", reroute_payload))
```

The two-line swap is replaced by the helper. `swap_indices` is built as a `tuple[int, int]` (matches the existing `TopologyDiff.__post_init__` expectation).

- [ ] **Step 7: Run the B3 test suite**

Run: `uv run pytest tests/unit/test_restructure_lora.py -v`
Expected: all 14 tests pass unchanged.

- [ ] **Step 8: Full sanity**

Run: `uv run pytest -q` — full suite green.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/dream/operations/restructure_real.py` — clean.

- [ ] **Step 9: Commit**

```bash
git add kiki_oniric/dream/operations/restructure_real.py
git commit -m "$(cat <<'EOF'
refactor(op): delegate restructure_lora to topology kernel

B3 delegate refactor / step 2. restructure_lora_handler's three
per-op mutation branches (add / remove / reroute) now build a
payload and delegate the actual mutation to the substrate
utility _apply_topology_op(model, op, payload). The dream-side
responsibilities (seed derivation, snapshot capture for remove,
sha256 post-mutation, state counter updates, applied.append)
remain at the call site.

Drops the handler-build-time mx.core + LoRALinear imports —
they're now owned by the kernel module. TopologyDiff is still
imported at handler-build time because the closure returns it.

Zero behaviour change : the existing 14 tests in
tests/unit/test_restructure_lora.py all pass unchanged. The
helper's defense-in-depth bounds checks never fire here because
_validate_topo_op already runs first.
EOF
)"
```

---

## Task 3: Add the re-export identity test

**Files:**
- Create: `tests/unit/test_lora_topology_ops_reexport.py`

- [ ] **Step 1: Write the test**

Create `tests/unit/test_lora_topology_ops_reexport.py`:

```python
"""Pin the re-export path : the topology kernel is reachable from
both the canonical substrate module and the legacy channels
module. Refactor accident insurance.
"""
from __future__ import annotations


def test_apply_topology_op_reexport_identity() -> None:
    """The helper is the same object via either import path."""
    from kiki_oniric.dream.channels.hierarchy_change import (
        _apply_topology_op as via_channels,
    )
    from kiki_oniric.substrates.micro_kiki.lora_topology_ops import (
        _apply_topology_op as canonical,
    )
    assert via_channels is canonical
```

- [ ] **Step 2: Run to verify it passes**

Run: `uv run pytest tests/unit/test_lora_topology_ops_reexport.py -v`
Expected: PASS — 1 test.

- [ ] **Step 3: Full sanity**

Run: `uv run pytest -q` — full suite green.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check tests/unit/test_lora_topology_ops_reexport.py` — clean.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_lora_topology_ops_reexport.py
git commit -m "$(cat <<'EOF'
test: pin _apply_topology_op re-export identity

B3 delegate refactor / step 3. One-test file that asserts the
topology kernel is the same Python object whether imported from
kiki_oniric.dream.channels.hierarchy_change (legacy re-export)
or kiki_oniric.substrates.micro_kiki.lora_topology_ops
(canonical). Catches a future accidental local re-definition
in either module.
EOF
)"
```

---

## Task 4: DualVer + CHANGELOG + spec amendment

**Files:**
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`
- Modify: `uv.lock`
- Modify: `docs/superpowers/specs/2026-05-21-b3-delegate-topology-op-refactor-design.md`

- [ ] **Step 1: Bump `pyproject.toml`**

Change the `version = "0.22.0"` line to `version = "0.22.1"`.

- [ ] **Step 2: Sync the lockfile**

Run: `uv lock`
Expected: `uv.lock` updates the `dreamofkiki` pin to `0.22.1`.

- [ ] **Step 3: Add the Refactored bullet to `CHANGELOG.md`**

Open `CHANGELOG.md`. Locate the `## [Unreleased]` block (around line 13). Append a new `### Refactored` sub-block (or extend an existing one if present) :

```markdown
### Refactored (B3 delegate to topology kernel, 2026-05-21)

- `_apply_topology_op(model, op, payload)` — the LoRA topology
  mutation kernel — moved from
  `kiki_oniric/dream/channels/hierarchy_change.py` to a new
  substrate utility
  `kiki_oniric/substrates/micro_kiki/lora_topology_ops.py`.
  Both `LoRAHierarchyChangeChannel.apply_diff` (B5, awake-side
  replay of a `TopologyDiff`) and `restructure_lora_handler`
  (B3, dream-side mutation + diff emission) now delegate to
  the substrate-level kernel. The dream-side handler keeps its
  responsibilities (seed derivation via `_derive_op_seed`,
  snapshot capture before `remove`, `model_sha256_post` after
  each op, state counters) ; only the three `model.layers`
  mutation primitives are delegated.
- `hierarchy_change.py` re-exports `_apply_topology_op` through
  its `__all__` so B5's published symbol is unchanged.
- Zero behaviour change. The 14 existing B3 tests
  (`tests/unit/test_restructure_lora.py`) + B5 tests
  (`tests/unit/test_apply_channel_outputs.py`) all pass
  unchanged. One new test
  (`tests/unit/test_lora_topology_ops_reexport.py`) pins the
  re-export identity.
- DualVer FC-PATCH (`C-v0.24.0 → C-v0.24.1`). `pyproject.toml`
  version `0.22.0 → 0.22.1`. No new versioned `[C-v0.24.1+PARTIAL]`
  section per dreamOfkiki PATCH convention (matches the 2026-04-21
  DR-2 PATCH precedent — change-tracked via this `[Unreleased]`
  bullet + spec amendment).
```

- [ ] **Step 4: Amend the spec's DualVer + version numbers**

Open `docs/superpowers/specs/2026-05-21-b3-delegate-topology-op-refactor-design.md`. The spec body uses the stale `C-v0.23.0 → C-v0.23.1` and `0.21.0 → 0.21.1` strings (written before the Wave 3b M3 bump). Replace them with the current values :

Search-and-replace in that file :
- `C-v0.23.0 → C-v0.23.1` → `C-v0.24.0 → C-v0.24.1`
- `[C-v0.23.1+PARTIAL]` → `[C-v0.24.1+PARTIAL]`
- `0.21.0 → 0.21.1` → `0.22.0 → 0.22.1`

The substance of the spec is unchanged.

- [ ] **Step 5: Verify**

Run: `uv run pytest -q` — full suite green (no code change).
Run: `uv run mypy harness tests` — `Success`.

If `tests/reproducibility/golden_hashes_apple_*.json` shows modified, restore with `git checkout -- tests/reproducibility/golden_hashes_apple_*.json` before committing.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock CHANGELOG.md docs/superpowers/specs/2026-05-21-b3-delegate-topology-op-refactor-design.md
git commit -m "$(cat <<'EOF'
docs: log B3 delegate refactor + bump version 0.22.1

B3 delegate refactor / step 4. Bumps pyproject.toml SemVer
from 0.22.0 to 0.22.1 (FC-PATCH C-v0.24.0 to C-v0.24.1),
adds the Refactored bullet under [Unreleased] in CHANGELOG.md
documenting the kernel extract + delegate pattern, and amends
the design spec's DualVer numbers to match the actual HEAD
versions (the spec was authored against C-v0.23.0 / 0.21.0
before the Wave 3b M3 bump landed on main).

No new versioned changelog section per dreamOfkiki FC-PATCH
convention (precedent: 2026-04-21 DR-2 PATCH).
EOF
)"
```

---

## Task 5: Final verification + push

**Files:** none — verification only.

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -q`
Expected: all pass, 0 failures, coverage gate met. Expect ≥ 915 tests passed (914 from current HEAD + 1 new re-export test).

- [ ] **Step 2: Type check**

Run: `uv run mypy harness tests`
Expected: `Success: no issues found in N source files` (N = previous + 1 for the new substrate module).

- [ ] **Step 3: Lint**

Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 4: Confirm working tree is clean**

Run: `git status --short`
Expected: empty (per-family golden hashes file may show modified — restore it).

- [ ] **Step 5: Push**

```bash
git push origin main
```

- [ ] **Step 6: Sanity ping (optional)**

Inspect a single B3 test that exercises both `add` and `reroute` to confirm post-refactor :

Run: `uv run pytest tests/unit/test_restructure_lora.py::test_restructure_lora_handler_emits_multi_op_diff -v`

(Adapt the test name if different — the goal is one targeted sanity check on top of the suite-wide pass.)

---

## Self-Review

- **Spec coverage:**
  - Create `lora_topology_ops.py` (Task 1 Step 1) ✓
  - Modify `hierarchy_change.py` import + re-export (Task 1 Step 2) ✓
  - Modify `restructure_real.py` to delegate (Task 2 Steps 2-6) ✓
  - Snapshot capture for `remove` preserved at the B3 call site (Task 2 Step 5) ✓
  - Seed derivation for `add` preserved at the B3 call site (Task 2 Step 4) ✓
  - State counter updates + sha256 post-mutation preserved (Task 2 Steps 4-6) ✓
  - `_apply_topology_op` name unchanged ; re-export keeps B5 published surface (Task 1 Step 2) ✓
  - Re-export identity test (Task 3) ✓
  - pyproject 0.22.1 + uv lock (Task 4 Steps 1-2) ✓
  - CHANGELOG `[Unreleased]` Refactored bullet (Task 4 Step 3) ✓
  - Spec amend for stale version numbers (Task 4 Step 4) ✓
  - No new framework-C spec note — refactor is internal (no task) ✓
  - No new versioned changelog section per FC-PATCH convention (Task 4 Step 3 documents the convention) ✓
  - Final verification + push (Task 5) ✓
- **Placeholder scan:** no TBD/TODO; every code step has complete code.
- **Type consistency:**
  - `_apply_topology_op(model, op, payload)` — same signature in the new module, in `hierarchy_change.py` re-export, and at every B3 call site in `restructure_real.py`.
  - `payload` keys (`index`, `in_features`, `out_features`, `rank`, `alpha`, `seed`, `model_sha256_post`, `snapshot`, `swap_indices`) — identical across the kernel implementation, the B3 add/remove/reroute builders, and `TopologyDiff.__post_init__` (B3 spec).
  - `_derive_op_seed(seed, episode_id, op_index) -> int` — unchanged ; B3 still calls it.
  - `_model_sha256(model) -> str` — unchanged ; B3 still calls it.
  - `_validate_topo_op(op_dict, layers_len, idx)` — unchanged ; B3 still validates upfront.
  - `RestructureRealState` fields (`adds_this_episode`, `total_adds`, `total_removes`, `total_reroutes`, `diff_history`, `last_compute_flops`) — same names referenced in Task 2's per-op blocks.
- **Inter-task ordering:** Task 1 (kernel extract) must precede Task 2 (B3 delegate, which imports the kernel) ; Task 3 (re-export test) needs Task 1 done ; Task 4 (docs + version) cleans up after the code lands ; Task 5 (verif + push) is last. Chosen 1→2→3→4→5 is the dependency order.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-21-b3-delegate-topology-op-refactor.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, inline review by me between tasks (matches the B6a/b/c + B6 LoRA smoke pilot pattern from earlier in the session).

**2. Inline Execution** — execute tasks in this session.
