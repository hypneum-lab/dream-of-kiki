"""Real-weight restructure op — topology mutation with S3 guard.

Cycle-3 C3.3 introduced ``restructure_real_handler`` for the
``reroute``-only swap on a list-of-layers model. Sub-project B3
(issue #15) adds ``restructure_lora_handler``: the full
``{add, remove, reroute}`` vocab on a ``LoRAModel`` adapter stack,
emitting a channel-3 ``TopologyDiff`` whose entries carry executable
payloads and a per-op SHA-256 model fingerprint.

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from kiki_oniric.dream.episode import DreamEpisode

if TYPE_CHECKING:
    from kiki_oniric.dream.channels import TopologyDiff
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


# Subset of topo_ops supported by the legacy real-weight reroute op.
_SUPPORTED_TOPO_OPS: frozenset[str] = frozenset({"reroute"})

# Full vocab supported by the LoRA variant (B3).
_LORA_TOPO_OPS: frozenset[str] = frozenset({"add", "remove", "reroute"})


@dataclass
class RestructureRealState:
    """K1-tagged restructure state across multiple episodes."""

    diff_history: list[str] = field(default_factory=list)
    last_compute_flops: int = 0
    # B3 additions — defaulted so existing call sites are unaffected.
    adds_this_episode: int = 0
    total_adds: int = 0
    total_removes: int = 0
    total_reroutes: int = 0


def restructure_real_handler(
    state: RestructureRealState,
    *,
    model,
) -> Callable[[DreamEpisode], None]:
    """Build a real-weight restructure handler bound to ``state``.

    Legacy cycle-3 handler — only ``"reroute"`` is supported; unknown
    ops raise a :class:`ValueError` whose message contains the literal
    ``"S3"`` tag (cycle-3 plan §C3.3 invariant 3 / test 7).
    """

    def handler(episode: DreamEpisode) -> None:
        topo_op = episode.input_slice.get("topo_op", "")
        if topo_op not in _SUPPORTED_TOPO_OPS:
            raise ValueError(
                f"S3: DE {episode.episode_id!r}: unknown topo_op "
                f"{topo_op!r} ; real-weight op supports "
                f"{sorted(_SUPPORTED_TOPO_OPS)}"
            )

        swap_indices = episode.input_slice.get("swap_indices", [0, 1])
        if len(swap_indices) != 2:
            raise ValueError(
                "S3: reroute requires swap_indices of length 2"
            )
        i, j = swap_indices
        if not (
            isinstance(i, int)
            and isinstance(j, int)
            and 0 <= i < len(model.layers)
            and 0 <= j < len(model.layers)
        ):
            raise ValueError(
                f"S3: reroute swap_indices {swap_indices!r} out of "
                f"bounds for layers of length {len(model.layers)}"
            )

        model.layers[i], model.layers[j] = (
            model.layers[j],
            model.layers[i],
        )

        state.diff_history.append(topo_op)
        state.last_compute_flops = max(len(model.layers), 1)

    return handler


def _model_sha256(model: "LoRAModel") -> str:
    """SHA-256 of the full LoRA parameter tree (R1 fingerprint)."""
    h = hashlib.sha256()
    for layer in model.layers:
        h.update(np.asarray(layer.base_weight, dtype=np.float32).tobytes())
        h.update(np.asarray(layer.lora_a, dtype=np.float32).tobytes())
        h.update(np.asarray(layer.lora_b, dtype=np.float32).tobytes())
        if layer.use_bias:
            h.update(np.asarray(layer.bias, dtype=np.float32).tobytes())
    return h.hexdigest()


def _flop_estimate_restructure(
    applied: list[tuple[str, dict[str, object]]],
    model: "LoRAModel",
) -> int:
    """Rough FLOP cost summed across applied restructure ops."""
    total = 0
    for op, payload in applied:
        if op == "add":
            r = int(payload["rank"])
            n = int(payload["in_features"])
            m = int(payload["out_features"])
            total += 2 * r * (n + m)
        elif op == "remove":
            snap = payload["snapshot"]  # type: ignore[index]
            total += (
                int(snap["base_weight"].size)
                + int(snap["lora_a"].size)
                + int(snap["lora_b"].size)
            )
        else:  # reroute
            total += max(len(model.layers), 1)
    return max(total, 1)


def _validate_topo_op(
    op_dict: dict[str, object],
    layers_len: int,
    idx: int,
) -> None:
    """Raise S3 ValueError on any structural defect; no mutation."""
    op = op_dict.get("op")
    if op not in _LORA_TOPO_OPS:
        raise ValueError(
            f"S3: topo_ops[{idx}].op {op!r} unknown; "
            f"must be one of {sorted(_LORA_TOPO_OPS)}"
        )
    if op == "add":
        for key in ("index", "in_features", "out_features", "rank", "alpha"):
            if key not in op_dict:
                raise ValueError(
                    f"S3: topo_ops[{idx}] add missing key {key!r}"
                )
        if not (
            isinstance(op_dict["rank"], int) and int(op_dict["rank"]) > 0
        ):
            raise ValueError(
                f"S3: topo_ops[{idx}] add rank must be positive int"
            )
        ins_at = op_dict["index"]
        if not (isinstance(ins_at, int) and ins_at >= 0):
            raise ValueError(
                f"S3: topo_ops[{idx}] add index {ins_at!r} must be"
                " a non-negative int"
            )
    elif op == "remove":
        rm_at = op_dict.get("index")
        if not (isinstance(rm_at, int) and 0 <= rm_at < layers_len):
            raise ValueError(
                f"S3: topo_ops[{idx}] remove index {rm_at!r} out of bounds"
            )
    else:  # reroute
        swap = op_dict.get("swap_indices")
        if not (hasattr(swap, "__len__") and len(swap) == 2):  # type: ignore[arg-type]
            raise ValueError(
                f"S3: topo_ops[{idx}] reroute swap_indices must be length 2"
            )
        i, j = swap[0], swap[1]  # type: ignore[index]
        if not (
            isinstance(i, int)
            and isinstance(j, int)
            and 0 <= i < layers_len
            and 0 <= j < layers_len
        ):
            raise ValueError(
                f"S3: topo_ops[{idx}] reroute swap_indices {swap!r} "
                f"out of bounds for layers of length {layers_len}"
            )


def _derive_op_seed(seed: int, episode_id: str, op_index: int) -> int:
    """Stable per-op seed from (factory seed, episode_id, op index)."""
    h = hashlib.sha256()
    h.update(seed.to_bytes(8, "little", signed=False))
    h.update(episode_id.encode("utf-8"))
    h.update(op_index.to_bytes(8, "little", signed=False))
    # Use the first 8 bytes as an unsigned int seed for mx.random.key.
    return int.from_bytes(h.digest()[:8], "little", signed=False)


def restructure_lora_handler(
    state: RestructureRealState,
    *,
    model,  # LoRAModel — typed loosely for lazy MLX import
    max_adds_per_episode: int = 1,
    seed: int = 0,
) -> Callable[[DreamEpisode], "TopologyDiff | None"]:
    """Build a LoRA-only restructure handler that emits a ``TopologyDiff``.

    Reads ``topo_ops`` from the episode (default empty list), validates
    every op before any mutation, then applies them in order: ``add``
    inserts a new :class:`LoRALinear`, ``remove`` snapshots the doomed
    layer's arrays for undo and pops it, ``reroute`` swaps two
    positions. The INSS bound is enforced as a per-episode *soft* cap on
    ``add``: beyond ``max_adds_per_episode`` extra adds are silently
    skipped (no entry in the diff). Empty input or fully-skipped → returns
    ``None`` (S1 no-op).

    Each applied op produces an entry whose payload is *executable* —
    enough to reconstruct or undo the mutation — plus a 64-hex
    ``model_sha256_post`` fingerprint (R1).

    Reference:
      docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
    """
    import mlx.core as mx

    from kiki_oniric.dream.channels import TopologyDiff
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRALinear

    def handler(episode: DreamEpisode) -> "TopologyDiff | None":
        ops = episode.input_slice.get("topo_ops", [])
        if not ops:
            # S1 no-op: nothing requested.
            state.last_compute_flops = 0
            return None

        # Validate every op against current layers length BEFORE any
        # mutation. A single defect aborts the episode with a "S3:"
        # ValueError; the model stays untouched.
        layers_len = len(model.layers)
        for idx, op_dict in enumerate(ops):
            _validate_topo_op(op_dict, layers_len, idx)

        # Reset per-episode INSS counter.
        state.adds_this_episode = 0
        applied: list[tuple[str, dict[str, object]]] = []

        for idx, op_dict in enumerate(ops):
            op = op_dict["op"]

            if op == "add":
                if state.adds_this_episode >= max_adds_per_episode:
                    # INSS soft cap: silently skip — no entry, no mutation.
                    continue
                in_features = int(op_dict["in_features"])
                out_features = int(op_dict["out_features"])
                rank = int(op_dict["rank"])
                alpha = float(op_dict["alpha"])
                op_seed = _derive_op_seed(seed, episode.episode_id, idx)
                new_layer = LoRALinear(
                    in_features=in_features,
                    out_features=out_features,
                    rank=rank,
                    alpha=alpha,
                    key=mx.random.key(op_seed),
                )
                insert_at = int(op_dict["index"])
                model.layers.insert(insert_at, new_layer)
                state.adds_this_episode += 1
                state.total_adds += 1
                state.diff_history.append("add")
                payload: dict[str, object] = {
                    "index": insert_at,
                    "in_features": in_features,
                    "out_features": out_features,
                    "rank": rank,
                    "alpha": alpha,
                    "seed": op_seed,
                    "model_sha256_post": _model_sha256(model),
                }
                applied.append(("add", payload))

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
                model.layers.pop(rm_at)
                state.total_removes += 1
                state.diff_history.append("remove")
                applied.append(
                    (
                        "remove",
                        {
                            "index": rm_at,
                            "snapshot": snapshot,
                            "model_sha256_post": _model_sha256(model),
                        },
                    )
                )

            else:  # reroute
                i, j = int(op_dict["swap_indices"][0]), int(  # type: ignore[index]
                    op_dict["swap_indices"][1]  # type: ignore[index]
                )
                model.layers[i], model.layers[j] = (
                    model.layers[j],
                    model.layers[i],
                )
                state.total_reroutes += 1
                state.diff_history.append("reroute")
                applied.append(
                    (
                        "reroute",
                        {
                            "swap_indices": (i, j),
                            "model_sha256_post": _model_sha256(model),
                        },
                    )
                )

        if not applied:
            # Every requested op was an add that hit the INSS cap.
            state.last_compute_flops = 0
            return None

        state.last_compute_flops = _flop_estimate_restructure(applied, model)
        return TopologyDiff(diff=tuple(applied))

    return handler


__all__ = [
    "RestructureRealState",
    "restructure_lora_handler",
    "restructure_real_handler",
    "_model_sha256",
]
