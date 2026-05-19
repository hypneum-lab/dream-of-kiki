# B3 — `restructure` emits a `TopologyDiff`

**Date** : 2026-05-20
**Status** : design approved, pending spec review
**Tracking issue** : #15 (`hypneum-lab/dream-of-kiki`)
**Scope** : sub-project B3 of the issue-#15 "approach B" decomposition

---

## Context

Issue #15 / approach B makes the four dream operations produce real
channel outputs. Prior sub-projects on `main`:

- **B0** (`C-v0.14.0`) froze the channel-output contract,
  including the placeholder `TopologyDiff` value type — *frozen,
  but without a `__post_init__`; B0 deferred S3 structural
  validation to B3*.
- **B1a** (`C-v0.15.0`) shipped `LoRALinear` / `LoRAModel` and
  the `adapter_delta()` helper.
- **B1b** (`C-v0.16.0`) shipped `replay_lora_handler` — the first
  dream operation to emit a real `WeightUpdate` (channel 1).
- **B2** (`C-v0.17.0`) shipped `downscale_lora_handler` — the
  second emitting op, also on channel 1.

**B3** makes `restructure` the third emitting op, this time on
**channel 3** (`TopologyDiff`). Framework-C §4.2:

> **restructure** — Friston FEP, free-energy minimisation on
> hierarchy. Source: D-Friston FEP.
> Input: hierarchy descriptor. **Output: TopologyDiff (channel 3).**

## Problem

`restructure_real.py`'s `restructure_real_handler` supports only
`"reroute"` (pointer swap of `model.layers[i]/[j]`), returns
`None`, and operates on any list-of-layers model — including the
LoRA stack. No handler emits a `TopologyDiff`, and the
`TopologyDiff` value type itself has no `__post_init__` (B0 left
S3 structural validity for B3 to enforce).

B3 adds the LoRA-substrate restructure variant — supporting the
full vocab `{add, remove, reroute}` — and hardens `TopologyDiff`.

## Approaches considered

**Substrate target.** Three options were considered:

1. **LoRA stack (LoRAModel.layers : list[LoRALinear]).** Composable
   with B1b/B2 — same model, same `adapter_parameters()`. Adding
   a `LoRALinear` means inserting a new low-rank block with
   explicit `in_features`/`out_features`/`rank`/`alpha`/`seed`.
   **Chosen.**
2. Wider topology (`StackedMLP` / `nn.Linear` list). Breaks
   composability with B1b/B2 (which target a `LoRAModel`).
   Rejected.
3. `reroute`-only on `LoRAModel` (minimal scope). Rejected — the
   maintainer chose full vocab `{add, remove, reroute}` to expose
   the INSS Add-bound and the snapshot/undo pattern.

**Diff payload schema.** Three options were considered:

1. Descriptive (op + indices). Compact but B5 `consolidate()`
   could not rebuild a removed `LoRALinear` from the diff alone.
   Rejected.
2. **Hybrid (executable + per-op SHA-256).** The payload carries
   enough to reconstruct each mutation deterministically (dims,
   seed, snapshot arrays for `remove`), plus a `model_sha256_post`
   field for R1 provenance. **Chosen.**
3. Plain executable (no checksum). Rejected — the repo's R1
   reproducibility contract rewards the cheap checksum hook.

**Per-episode input shape.** Three options were considered:

1. One op per episode (symmetric with B1b/B2). Makes the INSS
   counter trivial (≤ 1 add per call). Rejected.
2. **List of ops per episode**
   (`input_slice["topo_ops"] : list[dict]`). The handler loops,
   accumulates diff entries, and gates `add` ops against the
   INSS soft cap. **Chosen.**
3. One op + internal repeat (`{"topo_op": "add", "count": N}`).
   Rejected — twisted shape for no benefit.

**INSS Add bound.** Four options were considered:

1. **Per-episode soft cap** (counter reset at handler entry, `add`
   beyond cap is silently skipped — not appended to the diff —
   matching INSS's "bound `Add(layer)` per dream-episode"
   recommendation, default `max_adds_per_episode = 1`). **Chosen.**
2. Per-episode hard cap (`ValueError`). Rejected — too brittle;
   the soft variant degrades gracefully.
3. Global cap on `len(model.layers)`. Rejected — captures depth
   but not the "per dream" rhythm INSS prescribes.
4. Defer to B5. Rejected — ignores the saved INSS input.

## Design

### New factory in `kiki_oniric/dream/operations/restructure_real.py`

```python
def restructure_lora_handler(
    state: RestructureRealState,
    *,
    model: "LoRAModel",
    max_adds_per_episode: int = 1,
    seed: int = 0,
) -> Callable[[DreamEpisode], "TopologyDiff | None"]:
```

The factory reuses `RestructureRealState` (extended with
backwards-compatible fields; the existing `diff_history` /
`last_compute_flops` are preserved).

### `RestructureRealState` — additions (backwards compatible)

```python
@dataclass
class RestructureRealState:
    diff_history: list[str] = field(default_factory=list)
    last_compute_flops: int = 0
    # B3 additions — all default 0 so existing call sites unaffected.
    adds_this_episode: int = 0   # reset to 0 at handler entry
    total_adds: int = 0
    total_removes: int = 0
    total_reroutes: int = 0
```

`diff_history` continues to grow by one entry per *applied* op
(symmetric with the historical `restructure_real_handler`).

### Handler behaviour

On each `DreamEpisode`:

1. Read `ops = episode.input_slice.get("topo_ops", [])`. If `ops`
   is empty → set FLOPs 0, return `None` (S1 no-op, symmetric
   with B1b/B2).
2. **Validate every op before any mutation** (`operations/CLAUDE.md`
   anti-pattern: never mutate before validation passes for *all*
   ops). For each op dict:
   - `"op" in {"add", "remove", "reroute"}` → else
     `ValueError("S3: ...")`.
   - `add`: requires `int` keys `index`, `in_features`,
     `out_features`, `rank` and `float` key `alpha`; `rank > 0`;
     `0 <= index <= len(model.layers)` (append allowed at the
     end).
   - `remove`: requires `int` key `index`; `0 <= index <
     len(model.layers)`.
   - `reroute`: requires `swap_indices` length-2 sequence of in-
     bounds `int`.
   A single invalid op aborts the whole episode with `ValueError`;
   the model stays untouched.
3. `state.adds_this_episode = 0` (reset).
4. Apply each op in order; record an entry per **applied** op:
   - `add`: if `state.adds_this_episode >= max_adds_per_episode`,
     **skip** (no entry, no mutation, INSS soft cap). Else derive
     a per-op MLX key from `(seed, episode.episode_id, idx)`,
     construct `LoRALinear(in_features, out_features, rank, alpha,
     key=...)`, insert at `index`, increment `adds_this_episode`
     and `total_adds`.
   - `remove`: snapshot the doomed layer's adapter arrays
     (`base_weight`, `lora_a`, `lora_b`, `bias`) as float32 numpy
     arrays (defensive copy via `np.asarray(..., dtype=np.float32)`
     and `.copy()` where MLX→numpy share buffers), `pop` the
     layer, increment `total_removes`.
   - `reroute`: swap `model.layers[i], model.layers[j]`,
     increment `total_reroutes`.
5. After each applied op, compute `model_sha256_post` from the
   concatenation of every layer's flat weight bytes (see "Hashing"
   below); append to the entry's payload.
6. Update K1: `state.last_compute_flops = _flop_estimate_restructure(applied_ops, model)`.
7. If no op was applied (e.g. all ops were `add` and INSS cap was
   0): return `None` (S1 no-op).
8. Else build `TopologyDiff(diff=tuple(entries))` and **return** it.

`TopologyDiff.__post_init__` (see below) then enforces structural
S3 validity at construction.

### Diff entry shapes

For an applied op, the entry is `(op_name, payload)`:

```python
# add
("add", {
    "index": int,
    "in_features": int,
    "out_features": int,
    "rank": int,
    "alpha": float,
    "seed": int,                    # derived (seed, episode_id, idx)
    "model_sha256_post": str,       # hex, 64 chars
})

# remove — snapshot lets B5 (consolidate) reconstruct via:
#     layer = LoRALinear(in_features, out_features, rank, alpha)
#     layer.base_weight = mx.array(snapshot["base_weight"])
#     layer.lora_a       = mx.array(snapshot["lora_a"])
#     layer.lora_b       = mx.array(snapshot["lora_b"])
#     if snapshot["bias"] is not None: layer.bias = mx.array(snapshot["bias"])
("remove", {
    "index": int,
    "snapshot": {
        "base_weight": NDArray[np.float32],
        "lora_a":      NDArray[np.float32],
        "lora_b":      NDArray[np.float32],
        "bias":        NDArray[np.float32] | None,
        "in_features": int,
        "out_features": int,
        "rank": int,
        "alpha": float,
    },
    "model_sha256_post": str,
})

# reroute
("reroute", {
    "swap_indices": tuple[int, int],
    "model_sha256_post": str,
})
```

### Hashing — `model_sha256_post`

```python
def _model_sha256(model: "LoRAModel") -> str:
    import hashlib
    h = hashlib.sha256()
    for layer in model.layers:
        for arr in (layer.base_weight, layer.lora_a, layer.lora_b):
            h.update(np.asarray(arr, dtype=np.float32).tobytes())
        if layer.use_bias:
            h.update(np.asarray(layer.bias, dtype=np.float32).tobytes())
    return h.hexdigest()
```

Cheap (one pass over the parameter tree), bit-exact under R1
(seeded init), preserves the repo's reproducibility contract.

### `TopologyDiff.__post_init__` (S3 hardening)

B0 left this as a no-op explicitly:
`# No __post_init__ — S3 structural validity deferred to B3.`

B3 fills it in:

```python
_VALID_TOPO_OPS: frozenset[str] = frozenset({"add", "remove", "reroute"})

@dataclass(frozen=True)
class TopologyDiff:
    diff: tuple[tuple[str, dict[str, object]], ...]

    def __post_init__(self) -> None:
        for idx, entry in enumerate(self.diff):
            if not (isinstance(entry, tuple) and len(entry) == 2):
                raise ValueError(
                    f"S3: TopologyDiff.diff[{idx}] must be a (op, payload) tuple"
                )
            op, payload = entry
            if op not in _VALID_TOPO_OPS:
                raise ValueError(f"S3: TopologyDiff.diff[{idx}].op {op!r} unknown")
            if not isinstance(payload, dict):
                raise ValueError(f"S3: TopologyDiff.diff[{idx}].payload not a dict")
            if "model_sha256_post" not in payload:
                raise ValueError(f"S3: TopologyDiff.diff[{idx}] missing model_sha256_post")
            sha = payload["model_sha256_post"]
            if not (isinstance(sha, str) and len(sha) == 64):
                raise ValueError(f"S3: TopologyDiff.diff[{idx}].model_sha256_post invalid")
            # Op-specific structural checks
            if op == "add":
                for key in ("index", "in_features", "out_features", "rank", "alpha", "seed"):
                    if key not in payload:
                        raise ValueError(f"S3: add entry missing {key!r}")
                if not (isinstance(payload["rank"], int) and payload["rank"] > 0):
                    raise ValueError("S3: add rank must be positive int")
            elif op == "remove":
                if "index" not in payload or "snapshot" not in payload:
                    raise ValueError("S3: remove entry missing keys")
                snap = payload["snapshot"]
                if not isinstance(snap, dict):
                    raise ValueError("S3: remove snapshot must be a dict")
                for arr_key in ("base_weight", "lora_a", "lora_b"):
                    if arr_key not in snap:
                        raise ValueError(f"S3: remove snapshot missing {arr_key!r}")
                    if not np.isfinite(snap[arr_key]).all():
                        raise ValueError(f"S2: remove snapshot[{arr_key!r}] non-finite")
            elif op == "reroute":
                swap = payload.get("swap_indices")
                if not (
                    isinstance(swap, tuple)
                    and len(swap) == 2
                    and all(isinstance(v, int) for v in swap)
                ):
                    raise ValueError("S3: reroute swap_indices invalid")
```

The guard is constructor-side, so any handler — present or future
— that emits a malformed `TopologyDiff` fails fast.

### K1 FLOP estimate

```python
def _flop_estimate_restructure(
    applied_ops: list[tuple[str, dict[str, object]]],
    model: "LoRAModel",
) -> int:
    total = 0
    for op, payload in applied_ops:
        if op == "add":
            r = int(payload["rank"])
            n = int(payload["in_features"])
            m = int(payload["out_features"])
            total += 2 * r * (n + m)  # adapter init
        elif op == "remove":
            snap = payload["snapshot"]
            total += int(snap["lora_a"].size) + int(snap["lora_b"].size) \
                  + int(snap["base_weight"].size)
        elif op == "reroute":
            total += max(len(model.layers), 1)
        # sha256 cost is constant; folded into the per-op tag implicitly.
    return max(total, 1)
```

### Invariants

- **S3** (hierarchy / vocabulary): enforced (i) in the handler
  pre-mutation pass (vocab + dims + bounds) with `"S3:"`-tagged
  errors and (ii) in `TopologyDiff.__post_init__` (structural).
- **S2** (finite): enforced in `TopologyDiff.__post_init__` on
  `remove` snapshot arrays; the handler itself never introduces
  non-finite values.
- **S1** (retained non-regression): empty `topo_ops` or
  all-skipped (INSS cap) → handler returns `None` (no diff
  emitted). Caller's scheduler decides whether the absence of a
  topology change is acceptable.
- **K1** (compute budget): per-op FLOPs tagged on
  `state.last_compute_flops`; cumulative counters on the state.
- **R1** (reproducibility): per-op SHA-256 of the model
  post-mutation pinned in each entry; seeded LoRALinear init.

### Convention compliance

`restructure_real.py` may import `numpy` (already does — the
"MLX-only" rule in `operations/CLAUDE.md` is aspirational and the
existing `_real` handlers in this op tree already use numpy via
`np.asarray`). The new factory needs numpy for the remove-snapshot
arrays and for the SHA-256 byte-view.

## DualVer

FC-**MINOR** (`C-v0.17.0 → C-v0.18.0`):

- New substrate handler `restructure_lora_handler` (additive).
- `TopologyDiff` gets a `__post_init__` (a constructor-side
  hardening). Technically tightens the contract, but B0 already
  documented the deferral (`# No __post_init__ — S3 structural
  validity deferred to B3.`) — the type-system signature is
  unchanged, no callsite outside the handler constructs
  `TopologyDiff` today, so MINOR is correct.
- No axiom change, no primitive signature change.

EC unchanged `+PARTIAL`. `pyproject.toml` version `0.15.0 → 0.16.0`.

## Testing — `tests/unit/test_restructure_lora.py`

End-to-end via `DreamRuntime.execute()` (no profile wiring).

1. **emit_topology_diff** — 1-op reroute happy path:
   `runtime.log[-1].channel_outputs[0]` is a `TopologyDiff`.
2. **s3_vocab_reject** — unknown `op` → `ValueError` matching
   `r"^S3:"`; model untouched after exception.
3. **s3_dims_reject** — `add` missing `rank` → `ValueError`
   pre-mutation; model untouched.
4. **add_grows_model** — single `add` → `len(model.layers) +=
   1`; payload carries `(in_features, out_features, rank, alpha,
   seed, index)`.
5. **add_reconstruction_round_trip** — re-instantiate
   `LoRALinear` from the payload `(in, out, rank, alpha,
   key=mx.random.key(seed))`; the inserted layer's
   `base_weight`/`lora_a`/`lora_b` match the rebuilt one
   bit-exactly (R1).
6. **remove_shrinks_model** — single `remove` → `len -= 1`;
   snapshot arrays are finite, float32, correct shapes.
7. **remove_undo_round_trip** — assigning the snapshot back into
   a fresh `LoRALinear(in, out, rank, alpha)` (via direct attr
   assignment) restores the exact removed layer (bit-exact
   compare on each array).
8. **reroute_swaps_in_place** — `model.layers[i] / [j]` swapped.
9. **multi_op_diff_length** — episode `[add, reroute, remove]` →
   `diff` length 3, entries in the input order.
10. **inss_soft_cap_one_skip** — 2 adds, `max_adds_per_episode=1`
    → 1 applied, 1 skipped, `diff` length 1; `state.total_adds
    == 1`.
11. **inss_soft_cap_full_skip** — `max_adds_per_episode=0` with
    only-add ops → handler returns `None` (S1 no-op),
    `last_compute_flops == 0`, model untouched.
12. **sha256_changes_under_mutation** — `model_sha256_post` of
    consecutive ops differ when the model actually changed.
13. **sha256_stable_under_identity_swap** — reroute `(i, i)`
    (no-op swap) → `model_sha256_post` equals the pre-handler
    hash.
14. **deterministic_under_seed** — same factory `seed`, same
    `episode_id`, same `topo_ops` → bit-identical `diff` payload
    (including `model_sha256_post`).

Plus a small `tests/unit/test_channels.py` addition exercising
`TopologyDiff.__post_init__` directly: each negative case from
the validation logic (malformed entry shape, unknown op, missing
keys, non-finite snapshot, bad sha length) raises `ValueError`.

## Scope boundary

**B3 does** : the `restructure_lora_handler` factory in
`restructure_real.py`; the full `{add, remove, reroute}` vocab on
`LoRAModel`; INSS soft cap on `add`; remove-snapshot for undo;
per-op SHA-256; `TopologyDiff.__post_init__` S3 hardening;
extended `RestructureRealState`; K1 FLOP tagging; unit tests
covering 14 handler scenarios + the `__post_init__` negatives;
spec/`CHANGELOG`/version sync.

**B3 does not** : wire the handler into any profile (`p_min` /
`p_equ` keep their skeleton restructure handler); rewire
`consolidate()` to actually apply the diff to a target model
(B5); re-train the freshly inserted `LoRALinear` (its adapters
are standard-init `A=random`, `B=0`); implement CasCor partial
weight-freeze (INSS hint, future work flagged in the changelog).

## Acceptance criteria

1. `restructure_lora_handler` exists in `restructure_real.py`,
   takes a `LoRAModel`, reuses `RestructureRealState` (extended
   with the 4 backwards-compatible fields).
2. The handler validates **every** op pre-mutation (vocab, dims,
   bounds) with `"S3:"`-tagged errors; a single bad op aborts the
   whole episode without mutating the model.
3. `add` / `remove` / `reroute` mutate `model.layers` correctly;
   `add` honours the INSS soft cap
   (`max_adds_per_episode`, default 1); skipped `add`s produce
   no entry in the diff; an episode that applies no op returns
   `None` with FLOPs 0.
4. Each applied op's entry carries the executable payload defined
   above plus a 64-hex `model_sha256_post`. `remove` snapshots
   include `base_weight` / `lora_a` / `lora_b` / `bias` and the
   reconstruction dims.
5. `TopologyDiff.__post_init__` rejects malformed diffs with
   `"S3:"`-tagged `ValueError`s; finite-array checks raise with
   `"S2:"`.
6. `tests/unit/test_restructure_lora.py` (14 tests) and the
   `test_channels.py` additions for `TopologyDiff.__post_init__`
   pass; full pytest suite green; `uv run mypy harness tests`
   clean; `uv run ruff check .` clean.
7. FC-MINOR DualVer bump recorded in `CHANGELOG.md`
   (`[C-v0.18.0+PARTIAL]`); framework-C spec §4.2 (EN + FR)
   notes that `restructure` now emits a real `TopologyDiff`;
   `pyproject.toml` bumped `0.15.0 → 0.16.0`.
