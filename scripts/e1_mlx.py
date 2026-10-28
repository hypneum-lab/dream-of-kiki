#!/usr/bin/env python3
"""E1 35B replay controls — Apple MLX backend (PI decision 2026-10-25).

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` (sections
4, 5, 6, 9; the MLX deviations are listed in section 9). The modes, the
cell and the run identity are those of ``scripts/e1_harness.py``; this
file supplies the MLX operations with the same semantics as the PyTorch
backend (``scripts/pilot_cycle3_cuda.py``):

- scoring: length-normalised log-probability of the choice tokens after
  the prompt, log-softmax in float32, argmax ties to the lowest index;
- replay: full-parameter, momentumless SGD (direct update computed in
  float32 and rounded once, no optimizer state), batch 1, mean
  next-token cross-entropy in float32 over the sequence, the token-budget
  schedule of section 5.4;
- DOWNSCALE multiplies every floating parameter by 0.99; RESTRUCTURE
  swaps two adjacent decoder blocks; RECOMBINE is a no-op;
- restore: MLX arrays are immutable, so the snapshot keeps references to
  the base arrays and restoring re-binds them (no copy, exact);
- item draws come from a split file written with the torch generators
  (``scripts/e1_splits.py``), so both backends draw the same items.

The weights are the HF checkpoint of the pinned revision, loaded
directly by ``mlx_lm`` (no conversion); every file is checked against
``scripts/e1_model_pins.json`` (size and SHA-256 / git blob) at start.

    python scripts/e1_mlx.py --mode smoke --weights-dir W --splits S [--smoke-profiles p_min]
    python scripts/e1_mlx.py --mode tuning --corpus C --manifest M --weights-dir W --splits S
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e1_harness as hn
from scripts.e1_splits import FileSplits

PINS = ROOT / "scripts" / "e1_model_pins.json"
ROUTER_SUFFIX = "mlp.gate.weight"


def _file_digest(path: Path, entry: dict[str, Any]) -> bool:
    if path.stat().st_size != int(entry["size"]):
        return False
    if "sha256" in entry:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest() == entry["sha256"]
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest() == entry["git_blob"]


def verify_weights(weights_dir: Path, pins_path: Path = PINS) -> dict[str, Any]:
    """Every file of the pinned revision is present and identical (no extra file)."""
    pins = json.loads(pins_path.read_text())
    files = pins["files"]
    local = {p.name for p in weights_dir.iterdir() if p.is_file()}
    if local != set(files):
        raise SystemExit(f"{weights_dir} does not hold exactly the files of the pinned revision")
    with ThreadPoolExecutor(max_workers=8) as ex:
        ok = dict(zip(files, ex.map(lambda f: _file_digest(weights_dir / f, files[f]), files),
                      strict=True))
    bad = sorted(f for f, v in ok.items() if not v)
    if bad:
        raise SystemExit(f"files differ from {pins['repo']}@{pins['revision']}: {bad}")
    return {"repo": pins["repo"], "revision": pins["revision"], "n_files": len(files),
            "pins_sha256": hashlib.sha256(pins_path.read_bytes()).hexdigest()}


def _layers(model: Any) -> list[Any]:
    for path in ("language_model.model.layers", "model.layers"):
        obj = model
        for a in path.split("."):
            if not hasattr(obj, a):
                break
            obj = getattr(obj, a)
        else:
            return obj  # type: ignore[no-any-return]
    raise SystemExit("decoder layers not found")


def _aligned_moe_call(self: Any, x: mx.array) -> mx.array:
    """Qwen3.5 sparse-MoE forward with the HF router arithmetic.

    transformers (``Qwen3_5MoeTopKRouter``) takes the softmax of the router
    logits in float32, selects the top-k experts on those float32
    probabilities, renormalises them in float32 and casts the weights to the
    logits dtype. ``mlx_lm`` selects on probabilities rounded to the
    activation dtype (bf16), where distinct logits can tie. This forward
    restores the HF selection and records the selected experts when a
    counter is attached (the section 6 expert-usage histogram counts actual
    routing decisions).
    """
    logits = self.gate(x)
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    k = self.top_k
    inds = mx.argpartition(probs, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(probs, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    scores = scores.astype(logits.dtype)
    counts = getattr(self, "_e1_counts", None)
    if counts is not None:
        idx = np.asarray(inds).reshape(-1)
        for e, c in zip(*np.unique(idx, return_counts=True), strict=True):
            key = f"expert:{int(e)}"  # section 6: summed over layers
            counts[key] = counts.get(key, 0) + int(c)
    y = self.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    shared_y = self.shared_expert(x)
    shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
    return y + shared_y


def _l2norm_f32(x: mx.array, eps: float = 1e-6) -> mx.array:
    """transformers / FLA ``l2norm``: ``x * rsqrt(sum(x^2) + eps)``, in float32."""
    xf = x.astype(mx.float32)
    return xf * mx.rsqrt(mx.sum(xf * xf, axis=-1, keepdims=True) + eps)


def _aligned_gdn_call(self: Any, inputs: mx.array, mask: mx.array | None = None,
                      cache: Any | None = None) -> mx.array:
    """Gated DeltaNet forward of ``mlx_lm`` with the transformers arithmetic for
    q / k normalisation (D11) and the decay gate (D12), PI decision 2026-10-28.

    - D11: q and k are L2-normalised with ε added to the sum of squares, in
      float32 (``mlx_lm`` uses an RMS norm, ε on the mean square, i.e. an ε
      larger by the head dimension in sum space); q is then scaled by
      ``d_k^-0.5`` as in transformers, and both are cast back. The formula
      is the transformers / FLA one; the float32 order follows FLA's
      kernels, not the transformers CPU fallback (which normalises in the
      incoming dtype before promoting).
    - D12: ``g = exp(-exp(A_log) * softplus(a + dt_bias))`` with ``a`` promoted
      to float32 before the softplus (``mlx_lm`` adds and applies softplus in
      the activation dtype).

    Everything else (projections, causal convolution, recurrence, gated
    norm, cache handling) is the ``mlx_lm`` code path unchanged.
    """
    from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops

    B, S, _ = inputs.shape
    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)
    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    if cache is not None:
        n_keep = self.conv_kernel_size - 1
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, S)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(self.conv1d(conv_input))
    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            strict=True,
        )
    ]
    state = cache[1] if cache else None
    q = (_l2norm_f32(q) * (self.head_k_dim ** -0.5)).astype(inputs.dtype)
    k = _l2norm_f32(k).astype(inputs.dtype)
    beta = mx.sigmoid(b)
    g = mx.exp(-mx.exp(self.A_log.astype(mx.float32))
               * nn.softplus(a.astype(mx.float32) + self.dt_bias.astype(mx.float32)))
    if state is None:
        state = mx.zeros((B, self.num_v_heads, self.head_v_dim, self.head_k_dim),
                         dtype=mx.float32)
    if self.training or mx.default_device() != mx.gpu or not mx.metal.is_available():
        out, state = gated_delta_ops(q, k, v, g, beta, state, mask)
    else:
        out, state = gated_delta_kernel(q, k, v, g, beta, state, mask)
    if cache is not None:
        cache[1] = state
        cache.advance(S)
    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1))  # type: ignore[no-any-return]


def align_linear_attention(model: Any) -> int:
    """Give every Gated DeltaNet layer the transformers q / k normalisation and
    decay-gate arithmetic (D11, D12); returns the count."""
    n = 0
    for layer in _layers(model):
        gdn = getattr(layer, "linear_attn", None)
        if gdn is None:
            continue
        if getattr(gdn, "sharding_group", None) is not None:
            raise SystemExit("sharded linear attention is not supported")
        cls = gdn.__class__
        gdn.__class__ = type(f"E1Aligned{cls.__name__}", (cls,), {"__call__": _aligned_gdn_call})
        n += 1
    return n


def align_moe(model: Any) -> int:
    """Give every sparse-MoE block the HF router arithmetic; returns the count."""
    n = 0
    for i, layer in enumerate(_layers(model)):
        blk = getattr(layer, "mlp", None)
        if blk is None or not hasattr(blk, "switch_mlp"):
            continue
        if getattr(blk, "sharding_group", None) is not None:
            raise SystemExit("sharded MoE blocks are not supported")
        cls = blk.__class__
        aligned = type(f"E1Aligned{cls.__name__}", (cls,), {"__call__": _aligned_moe_call})
        blk.__class__ = aligned
        blk._e1_key = f"layers.{i}.mlp"
        blk._e1_counts = None
        n += 1
    return n


class _RouterCounter:
    """Attaches a shared counter to the aligned MoE blocks while active."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.counts: dict[str, int] = {}
        self.blocks: list[Any] = []

    def __enter__(self) -> _RouterCounter:
        for layer in _layers(self.model):
            blk = getattr(layer, "mlp", None)
            if blk is not None and hasattr(blk, "_e1_key"):
                blk._e1_counts = self.counts
                self.blocks.append(blk)
        return self

    def __exit__(self, *exc: Any) -> None:
        for blk in self.blocks:
            blk._e1_counts = None


def sgd_step(p: mx.array, g: mx.array, lr: float) -> mx.array:
    """Momentumless SGD as in torch (``p.add_(g, alpha=-lr)``): the update is
    computed in float32 and rounded once to the parameter dtype; no
    optimizer state (``mlx.optimizers.SGD`` would cast lr to the gradient
    dtype and keep a zero momentum buffer the size of the model)."""
    return (p.astype(mx.float32) - lr * g.astype(mx.float32)).astype(p.dtype)


def downscale_array(p: mx.array, factor: float) -> mx.array:
    """DOWNSCALE as in torch (``p.mul_(0.99)``): multiply in float32, round once
    (an MLX bf16 array times a Python float would round 0.99 to 0.98828125)."""
    if not mx.issubdtype(p.dtype, mx.floating):
        return p
    return (p.astype(mx.float32) * factor).astype(p.dtype)


class MLXBackend:
    """MLX operations of one E1 cell."""

    name = "mlx"

    def __init__(self, model: Any, tokenizer: Any, splits: FileSplits | None,
                 experts_per_tok: int = 0) -> None:
        self.model, self.tok, self.splits = model, tokenizer, splits
        self.k = experts_per_tok
        self.base: dict[str, mx.array] = {}

    def versions(self) -> dict[str, Any]:
        import mlx_lm
        import transformers

        info: dict[str, Any] = {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
                                "transformers": transformers.__version__,
                                "device": str(mx.default_device()), **self.tok.info()}
        try:
            info["device_info"] = {k: v for k, v in mx.metal.device_info().items()
                                   if isinstance(v, (str, int, float))}
        except Exception:  # noqa: BLE001
            pass
        return info

    def ids(self, text: str) -> list[int]:
        return self.tok.encode_ids(text)

    def snapshot(self) -> None:
        params = self.model.parameters()
        mx.eval(params)
        self.base = dict(tree_flatten(params))

    def restore(self) -> None:
        self.model.update(tree_unflatten(list(self.base.items())))
        mx.eval(self.model.parameters())

    def n_tokens(self, text: str) -> int:
        return len(self.ids(text))

    def _score_choice(self, prompt: str, choice: str) -> float:
        p = len(self.ids(prompt))
        full = self.ids(prompt + choice)
        logits = self.model(mx.array(full)[None])
        logp = nn.log_softmax(logits[0, :-1].astype(mx.float32), axis=-1)
        tgt = mx.array(full[1:])
        tok_lp = mx.take_along_axis(logp, tgt[:, None], axis=-1)[:, 0][p - 1:]
        if tok_lp.size == 0:
            return -1e9
        return float(mx.mean(tok_lp).item())

    def score_items(self, items: Sequence[dict[str, Any]]) -> tuple[list[list[float]], list[int]]:
        scores, argmax = [], []
        for ex in items:
            s = [self._score_choice(ex["prompt"], c) for c in ex["choices"]]
            scores.append(s)
            argmax.append(int(np.argmax(np.asarray(s))))  # ties -> lowest index
        return scores, argmax

    @staticmethod
    def _loss(model: Any, x: mx.array) -> mx.array:
        logits = model(x).astype(mx.float32)
        return nn.losses.cross_entropy(logits[:, :-1], x[:, 1:], reduction="mean")

    def replay(self, texts: Sequence[str], lengths: Sequence[int], updates: Sequence[Any],
               lr: float) -> dict[str, Any]:
        ids = [self.ids(t) for t in texts]
        loss_and_grad = nn.value_and_grad(self.model, self._loss)
        self.model.train()
        losses: list[float] = []
        violation = None
        n = 0
        for u in updates:
            x = mx.array(ids[u.record][: u.n_input])[None]
            loss, grads = loss_and_grad(self.model, x)
            mx.eval(loss)
            if not bool(mx.isfinite(loss).item()):
                violation = "S2"
                break
            self.model.update(tree_map(lambda p, g: sgd_step(p, g, lr),
                                       self.model.trainable_parameters(), grads))
            mx.eval(self.model.parameters())
            losses.append(float(loss.item()))
            n += 1
        self.model.eval()
        return {"n_executed": n, "losses": losses, "invariant_violation": violation}

    def downscale(self, factor: float) -> None:
        self.model.update(tree_map(lambda p: downscale_array(p, factor), self.model.parameters()))
        mx.eval(self.model.parameters())

    def swap_layers(self, i: int) -> None:
        layers = _layers(self.model)
        layers[i], layers[i + 1] = layers[i + 1], layers[i]

    def restructure(self, seed: int) -> int | None:
        layers = _layers(self.model)
        if len(layers) < 2:
            return None
        i = seed % (len(layers) - 1)
        self.swap_layers(i)
        return i

    def weights_finite(self) -> bool:
        return all(bool(mx.all(mx.isfinite(v)).item())
                   for _, v in tree_flatten(self.model.parameters())
                   if mx.issubdtype(v.dtype, mx.floating))

    def drift(self, router_only: bool) -> tuple[float, float]:
        sq_d = sq_p = 0.0
        for k, v in tree_flatten(self.model.parameters()):
            if not mx.issubdtype(v.dtype, mx.floating) or (router_only
                                                            and not k.endswith(ROUTER_SUFFIX)):
                continue
            b = self.base[k].astype(mx.float32)
            d = mx.sum((v.astype(mx.float32) - b) ** 2)
            pp = mx.sum(b ** 2)
            mx.eval(d, pp)
            sq_d += float(d.item())
            sq_p += float(pp.item())
        return sq_d, sq_p

    def router_counter(self) -> _RouterCounter:
        return _RouterCounter(self.model)

    def reset_peak_memory(self) -> None:
        mx.reset_peak_memory()

    def peak_memory(self) -> dict[str, Any]:
        return {"mlx_peak_memory": int(mx.get_peak_memory()),
                "mlx_active_memory": int(mx.get_active_memory())}

    def cell_split(self, benches: hn.Benches, seed: int, n_replay: int,
                   n_eval: int) -> tuple[hn.Benches, list[dict[str, Any]]]:
        if self.splits is None:
            raise SystemExit("the MLX backend needs --splits (scripts/e1_splits.py)")
        return self.splits.cell_split(benches, seed, n_replay, n_eval)

    def shuffle(self, text: str, seed: int) -> str:
        if self.splits is None:
            raise SystemExit("the MLX backend needs --splits (scripts/e1_splits.py)")
        return self.splits.shuffle(text, seed)


class ZeroCenteredRMSNorm(nn.Module):
    """The HF Qwen3.5 RMSNorm: parameter w, output ``norm(x) * (1 + w)`` in
    float32, cast back to the input dtype. ``mlx_lm`` instead stores
    ``1 + w`` in the checkpoint dtype, which rounds the base model in bf16
    and changes what SGD, DOWNSCALE and the drift measures act on; the
    harness restores the HF parameterisation."""

    def __init__(self, weight: mx.array, eps: float) -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.fast.rms_norm(x.astype(mx.float32), 1.0 + self.weight.astype(mx.float32),
                             self.eps)
        return y.astype(x.dtype)


_NORM_RE = re.compile(r"^(?:model\.language_model\.|model\.)((?:layers\.\d+\."
                      r"(?:input_layernorm|post_attention_layernorm|self_attn\.q_norm|"
                      r"self_attn\.k_norm))|norm)\.weight$")


def _weight_files(weights_dir: Path) -> list[Path]:
    idx = weights_dir / "model.safetensors.index.json"
    if idx.exists():
        names = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        return [weights_dir / n for n in names]
    return sorted(weights_dir.glob("*.safetensors"))


def zero_centered_norms(model: Any, weights_dir: Path) -> int:
    """Replace the shifted ``nn.RMSNorm`` modules by :class:`ZeroCenteredRMSNorm`
    holding the original checkpoint values of w; returns the count."""
    n = 0
    for f in _weight_files(weights_dir):
        arrays = mx.load(str(f))
        for key, w in arrays.items():
            m = _NORM_RE.match(key)
            if not m:
                continue
            path = ("language_model.model." + m.group(1)).split(".")
            parent: Any = model
            for a in path[:-1]:
                parent = parent[int(a)] if a.isdigit() else getattr(parent, a)
            old = getattr(parent, path[-1])
            if not isinstance(old, nn.RMSNorm):
                raise SystemExit(f"unexpected module at {'.'.join(path)}")
            setattr(parent, path[-1], ZeroCenteredRMSNorm(w, old.eps))
            n += 1
    mx.eval(model.parameters())
    return n


def load(weights_dir: Path, tokenizer_sha256: str | None = None) -> tuple[Any, Any, int]:
    """mlx_lm model (HF RMSNorm parameterisation and router arithmetic
    restored), the raw ``tokenizer.json`` of the same files (S8), experts
    per token."""
    from mlx_lm import load as mlx_load

    model, wrapper = mlx_load(str(weights_dir))
    cfg0 = json.loads((weights_dir / "config.json").read_text())
    n_layers = int(cfg0.get("text_config", cfg0)["num_hidden_layers"])
    interval = int(cfg0.get("text_config", cfg0).get("full_attention_interval", 4))
    expected = 2 * n_layers + 2 * (n_layers // interval) + 1
    got = zero_centered_norms(model, weights_dir)
    if got != expected:
        raise SystemExit(f"replaced {got} RMSNorm modules, expected {expected}")
    n_moe = align_moe(model)
    if n_moe != n_layers:
        raise SystemExit(f"aligned {n_moe} MoE blocks, expected {n_layers}")
    n_lin = align_linear_attention(model)
    if n_lin != n_layers - n_layers // interval:
        raise SystemExit(f"aligned {n_lin} linear-attention layers, expected "
                         f"{n_layers - n_layers // interval}")
    model.eval()
    cfg = json.loads((weights_dir / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    tok = hn.RawTokenizer(weights_dir / "tokenizer.json", tokenizer_sha256)
    return model, tok, int(text.get("num_experts_per_tok", 0) or 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E1 harness, MLX backend")
    hn.add_args(ap)
    ap.add_argument("--weights-dir", type=Path, required=True,
                    help="HF checkpoint of the pinned revision (checked against the pins)")
    ap.add_argument("--splits", type=Path, help="split file (scripts/e1_splits.py)")
    ap.add_argument("--no-verify-weights", action="store_true",
                    help="tiny test models only; refused for the pinned model")
    args = ap.parse_args(argv)

    def factory(a: argparse.Namespace, benches: hn.Benches,
                common: dict[str, Any]) -> MLXBackend:
        if a.no_verify_weights:
            if a.model == hn.MODEL_REPO:
                raise SystemExit("the pinned model is always verified")
            common["weights"] = {"verified": False, "dir": str(a.weights_dir)}
        else:
            common["weights"] = verify_weights(a.weights_dir)
            if (common["weights"]["repo"], common["weights"]["revision"]) != (a.model, a.revision):
                raise SystemExit("--weights-dir is not the requested model revision")
        splits = None
        if a.splits:
            doc = json.loads(a.splits.read_text())
            splits = FileSplits(doc, benches, a.mode, common.get("corpus"))
            common["splits_sha256"] = hashlib.sha256(a.splits.read_bytes()).hexdigest()
        model, tok, k = load(a.weights_dir,
                             None if a.no_verify_weights else hn.TOKENIZER_SHA256)
        return MLXBackend(model, tok, splits, k)

    return hn.run(args, factory)


if __name__ == "__main__":
    raise SystemExit(main())
