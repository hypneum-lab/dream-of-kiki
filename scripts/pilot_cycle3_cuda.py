#!/usr/bin/env python3
"""E1 35B replay controls — PyTorch / HF transformers backend (CUDA or CPU).

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` (sections
4, 5, 6, 9). The modes, the cell and the run identity live in
``scripts/e1_harness.py``; this file supplies the PyTorch operations.
The Apple MLX backend is ``scripts/e1_mlx.py``. R1: numbers from
different backends are never pooled.

Load-once + restore: the model is loaded once, a CPU copy of the base
weights is restored before every cell. ``_cell_split`` and
``_shuffle_text`` (torch generators) define the item draws of every
backend: the MLX backend reads them from a split file written with these
functions (``scripts/e1_splits.py``).

    python pilot_cycle3_cuda.py --mode smoke --model <id> [--revision R]
    python pilot_cycle3_cuda.py --mode headroom --corpus C --manifest M
    python pilot_cycle3_cuda.py --mode tuning --corpus C --manifest M
    python pilot_cycle3_cuda.py --mode calibrate --corpus C --manifest M --tuning-cells T
    python pilot_cycle3_cuda.py --mode amend --manifest M --tuning-cells T --calib-cells K
    python pilot_cycle3_cuda.py --mode confirmatory --corpus C --manifest M --amendment A
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e1_harness as hn
from scripts import e1_profiles as pr

MODEL_REPO = hn.MODEL_REPO
MODEL_REVISION = hn.MODEL_REVISION
SEEDS = hn.SEEDS
N_REPLAY = hn.N_REPLAY
NOISE_SEED_OFFSET = hn.NOISE_SEED_OFFSET
check_provenance = hn.check_provenance
_sha_file = hn.sha_file
_selection_i = hn.selection_i


def _shuffle_text(text: str, seed: int) -> str:
    """Word-shuffle: keeps the tokens (same updates) but destroys meaning."""
    w = text.split()
    if len(w) < 2:
        return text
    g = torch.Generator().manual_seed(seed + len(w))
    return " ".join(w[i] for i in torch.randperm(len(w), generator=g).tolist())


def _cell_split(benches: hn.Benches, seed: int, n_replay: int,
                n_eval: int) -> tuple[hn.Benches, list[dict[str, Any]]]:
    """Disjoint per-(bench, seed) split: replay items vs eval items."""
    eval_sets, replay_items = {}, []
    for name, items in benches.items():
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(items), generator=g).tolist()
        replay_items += [items[i] for i in order[:n_replay]]
        ev = order[n_replay:]
        eval_sets[name] = [items[i] for i in (ev[:n_eval] if n_eval > 0 else ev)]
    return eval_sets, replay_items


# --- model -------------------------------------------------------------------


def _device() -> str:
    """CUDA when available, else CPU; ``HL_TORCH_DEVICE=mps`` selects Apple MPS
    (the E1-small macM1 smoke; recorded as backend ``torch-mps``)."""
    if os.environ.get("HL_TORCH_DEVICE", "auto") == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("HL_TORCH_DEVICE=mps but the MPS device is not available")
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class DiskSnapshot:
    """Base weights kept on disk, one ``torch.save`` file per state-dict tensor
    (``HL_SNAPSHOT=disk``, for unified-memory hosts). Restore copies each
    tensor back in place: the same values as the in-RAM snapshot."""

    kind = "disk"

    def __init__(self, model: Any, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, Path] = {}
        for i, (k, v) in enumerate(model.state_dict().items()):
            path = root / f"{i:05d}.pt"
            torch.save(v.detach().to("cpu", copy=True), path)
            self.files[k] = path

    def __getitem__(self, key: str) -> torch.Tensor:
        return torch.load(self.files[key], map_location="cpu", mmap=True,  # type: ignore[no-any-return]
                          weights_only=True)

    @torch.no_grad()
    def restore(self, model: Any) -> None:
        sd = model.state_dict()
        if set(sd) != set(self.files):
            raise RuntimeError("state-dict keys differ from the disk snapshot")
        for k, t in sd.items():
            t.copy_(self[k])


class RawTokenizer:
    """The pinned ``tokenizer.json`` through the ``tokenizers`` package, no
    special tokens: the counter that built the corpus (E1 section 5.1b step 3;
    E1-small PI decision S8). Called like the transformers tokenizer."""

    kind = "raw"

    def __init__(self, path: str | Path) -> None:
        from tokenizers import Tokenizer

        got = hn.sha_file(Path(path))
        if got != hn.src.TOKENIZER_SHA256:
            raise SystemExit(f"tokenizer.json SHA-256 {got} != pinned {hn.src.TOKENIZER_SHA256}")
        self.tok = Tokenizer.from_file(str(path))

    def __call__(self, text: str, return_tensors: str = "pt") -> Any:
        from types import SimpleNamespace

        ids = self.tok.encode(text, add_special_tokens=False).ids
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


# transformers module globals that select the linear-attention kernels
# (fla + causal-conv1d when installed, else the torch fallback).
_LINEAR_ATTN_GLOBALS = ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                        "fused_recurrent_gated_delta_rule", "FusedRMSNormGated")


def force_torch_linear_attention() -> None:
    """Make models instantiated afterwards use the transformers torch fallback
    for Gated DeltaNet even when fla / causal-conv1d are installed
    (``HL_LINEAR_ATTN=torch``; E1-small S6 keeps the fallback selectable)."""
    import importlib

    for name in ("qwen3_5", "qwen3_5_moe", "qwen3_next"):
        try:
            mod = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        except ImportError:
            continue
        for attr in _LINEAR_ATTN_GLOBALS:
            if hasattr(mod, attr):
                setattr(mod, attr, None)
        if hasattr(mod, "is_fast_path_available"):
            mod.is_fast_path_available = False


def _dist_version(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def _from(obj: Any, pkg: str) -> bool:
    mod = getattr(obj, "__module__", None) or type(obj).__module__
    return str(mod).split(".")[0] == pkg


def linear_attn_impl(model: Any) -> str:
    """'fla+causal_conv1d' (every fast-path component of every linear-attention
    layer: chunked and recurrent delta rule, gated RMSNorm from fla, causal
    conv and its update from causal_conv1d), 'torch' (none of them), 'mixed'
    or 'none' (no linear-attention layer)."""
    kinds = set()
    for m in model.modules():
        if getattr(m, "chunk_gated_delta_rule", None) is None:
            continue
        parts = [_from(getattr(m, "chunk_gated_delta_rule", None), "fla"),
                 _from(getattr(m, "recurrent_gated_delta_rule", None), "fla"),
                 _from(getattr(m, "norm", None), "fla"),
                 getattr(m, "causal_conv1d_fn", None) is not None
                 and _from(m.causal_conv1d_fn, "causal_conv1d"),
                 _from(getattr(m, "causal_conv1d_update", None), "causal_conv1d")]
        kinds.add("fla+causal_conv1d" if all(parts) else "torch" if not any(parts)
                  else "mixed")
    return "none" if not kinds else kinds.pop() if len(kinds) == 1 else "mixed"


def _load_model(model_id: str, revision: str | None) -> tuple[Any, Any]:
    if model_id == MODEL_REPO and revision != MODEL_REVISION:
        raise SystemExit(f"{MODEL_REPO} must be loaded at the pinned revision {MODEL_REVISION}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = _device()
    dtype = torch.bfloat16 if dev in ("cuda", "mps") else torch.float32
    trc = os.environ.get("HL_TRUST_REMOTE_CODE", "0") == "1"
    if hn.TOKENIZER == "raw":
        from huggingface_hub import hf_hub_download

        tok: Any = RawTokenizer(hf_hub_download(model_id, "tokenizer.json", revision=revision))
    else:
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trc)
    impl = os.environ.get("HL_LINEAR_ATTN", "default")
    if impl not in ("default", "torch"):
        raise SystemExit("HL_LINEAR_ATTN must be 'default' or 'torch'")
    if impl == "torch":
        force_torch_linear_attention()
    default_map = "auto" if (dev == "cuda" and torch.cuda.device_count() > 1) else ""
    use_map = os.environ.get("HL_DEVICE_MAP", default_map)
    kw: dict[str, Any] = {"dtype": dtype, "trust_remote_code": trc, "revision": revision}
    if use_map:
        kw["device_map"] = use_map
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if not use_map:
        model = model.to(dev)
    model.eval()
    return model, tok


def _snapshot_cpu(model: Any) -> dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _restore(model: Any, base: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(base, strict=True)


def _input_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001
        return next(model.parameters()).device  # type: ignore[no-any-return]


def _ids(tok: Any, text: str) -> torch.Tensor:
    return tok(text, return_tensors="pt").input_ids  # type: ignore[no-any-return]


@torch.no_grad()
def _score_choice(model: Any, tok: Any, prompt: str, choice: str) -> float:
    """Length-normalised log-prob of ``choice`` after ``prompt`` (cloze score)."""
    dev = _input_device(model)
    p_ids = _ids(tok, prompt)
    full = _ids(tok, prompt + choice).to(dev)
    out = model(full)
    logp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
    tgt = full[0, 1:]
    cont = logp[range(tgt.numel()), tgt][p_ids.shape[1] - 1:]
    return cont.mean().item() if cont.numel() else -1e9


def _router_modules(model: Any) -> list[tuple[str, Any]]:
    """Router (gate) modules of the MoE blocks: ``nn.Linear`` or the
    transformers top-k router, whatever its class."""
    return [(n, m) for n, m in model.named_modules() if n.endswith("mlp.gate")]


class _RouterCounter:
    """Counts top-k routed tokens per (layer, expert) while attached."""

    def __init__(self, model: Any) -> None:
        self.k = int(getattr(getattr(model, "config", None), "num_experts_per_tok", 0) or 0)
        if not self.k:
            text = getattr(getattr(model, "config", None), "text_config", None)
            self.k = int(getattr(text, "num_experts_per_tok", 0) or 0)
        self.mods = _router_modules(model) if self.k else []
        self.counts: dict[str, int] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> _RouterCounter:
        for name, mod in self.mods:
            self.handles.append(mod.register_forward_hook(self._hook(name)))
        return self

    def _hook(self, name: str) -> Any:
        def fn(_m: Any, _i: Any, out: Any) -> None:
            logits = out[0] if isinstance(out, tuple) else out
            top = torch.topk(logits.reshape(-1, logits.shape[-1]).float(), self.k, dim=-1).indices
            for e, c in zip(*torch.unique(top, return_counts=True), strict=True):
                key = f"{name}:{int(e)}"
                self.counts[key] = self.counts.get(key, 0) + int(c)
        return fn

    def __exit__(self, *exc: Any) -> None:
        for h in self.handles:
            h.remove()


def _get_layers(model: Any) -> Any:
    for path in ("model.layers", "model.model.layers", "model.language_model.layers",
                 "transformer.h", "gpt_neox.layers"):
        obj = model
        for a in path.split("."):
            if not hasattr(obj, a):
                break
            obj = getattr(obj, a)
        else:
            return obj
    return None


class TorchBackend:
    """PyTorch / HF transformers operations of one E1 cell."""

    name = "torch"

    def __init__(self, model: Any, tok: Any, base: dict[str, torch.Tensor] | None = None) -> None:
        self.model, self.tok = model, tok
        self.base: Any = base if base is not None else {}
        if model is not None and any(p.device.type == "mps" for p in model.parameters()):
            self.name = "torch-mps"  # its own instrument: never pooled with CUDA cells
        # Memory strategies for unified-memory hosts; the update arithmetic is
        # unchanged (tests check identical weights).
        self.snapshot_kind = os.environ.get("HL_SNAPSHOT", "ram")
        self.sgd_kind = "fused" if os.environ.get("HL_FUSED_SGD", "0") == "1" else "optim"
        if self.snapshot_kind not in ("ram", "disk"):
            raise SystemExit("HL_SNAPSHOT must be 'ram' or 'disk'")
        self._mps_peak = 0

    def versions(self) -> dict[str, Any]:
        import tokenizers
        import transformers

        return {"torch": torch.__version__, "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__, "device": _device(),
                "placement": sorted({str(p.device) for p in self.model.parameters()}),
                "tokenizer": getattr(self.tok, "kind", "transformers"),
                "linear_attn": linear_attn_impl(self.model),
                "kernels": {d: _dist_version(d) for d in ("fla-core", "causal-conv1d")},
                "snapshot": self.snapshot_kind, "sgd": self.sgd_kind}

    def snapshot(self) -> None:
        if self.snapshot_kind == "disk":
            root = os.environ.get("HL_SNAPSHOT_DIR")
            if not root:
                raise SystemExit("HL_SNAPSHOT=disk needs HL_SNAPSHOT_DIR")
            self.base = DiskSnapshot(self.model, Path(root))
        else:
            self.base = _snapshot_cpu(self.model)

    def restore(self) -> None:
        if isinstance(self.base, DiskSnapshot):
            self.base.restore(self.model)
        else:
            _restore(self.model, self.base)

    def _sample_mps(self) -> None:
        if self.name == "torch-mps":
            self._mps_peak = max(self._mps_peak, int(torch.mps.driver_allocated_memory()))

    def n_tokens(self, text: str) -> int:
        return int(_ids(self.tok, text).shape[1])

    @torch.no_grad()
    def score_items(self, items: Sequence[dict[str, Any]]) -> tuple[list[list[float]], list[int]]:
        scores, argmax = [], []
        for ex in items:
            s = [_score_choice(self.model, self.tok, ex["prompt"], c) for c in ex["choices"]]
            scores.append(s)
            argmax.append(int(torch.tensor(s).argmax()))  # ties -> lowest index
        return scores, argmax

    def loss(self, ids: torch.Tensor) -> torch.Tensor:
        out = self.model(ids, labels=ids)
        return out.loss  # type: ignore[no-any-return]

    def replay(self, texts: Sequence[str], lengths: Sequence[int], updates: Sequence[Any],
               lr: float) -> dict[str, Any]:
        ids = [_ids(self.tok, t) for t in texts]
        dev = _input_device(self.model)
        opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        hooks = []
        if self.sgd_kind == "fused":
            # Momentumless SGD applied per parameter as soon as its gradient is
            # complete (p -= lr * g, as torch.optim.SGD), then the gradient is
            # freed: no full-model gradient buffer is ever held.
            def step(p: torch.Tensor) -> None:
                with torch.no_grad():
                    p.add_(p.grad, alpha=-lr)
                p.grad = None

            hooks = [p.register_post_accumulate_grad_hook(step)
                     for p in self.model.parameters() if p.requires_grad]
        self.model.train()
        losses: list[float] = []
        violation = None
        n = 0
        try:
            for u in updates:
                x = ids[u.record][:, : u.n_input].to(dev)
                opt.zero_grad(set_to_none=True)
                loss = self.loss(x)
                if not torch.isfinite(loss):
                    violation = "S2"
                    break
                loss.backward()
                self._sample_mps()
                if not hooks:
                    opt.step()
                losses.append(float(loss.detach()))
                n += 1
        finally:
            for hk in hooks:
                hk.remove()
        opt.zero_grad(set_to_none=True)
        self.model.eval()
        return {"n_executed": n, "losses": losses, "invariant_violation": violation}

    @torch.no_grad()
    def downscale(self, factor: float) -> None:
        for p in self.model.parameters():
            if p.is_floating_point():
                p.mul_(factor)

    def swap_layers(self, i: int) -> None:
        layers = _get_layers(self.model)
        layers[i], layers[i + 1] = layers[i + 1], layers[i]

    def restructure(self, seed: int) -> int | None:
        layers = _get_layers(self.model)
        if layers is None or len(layers) < 2:
            return None
        i = seed % (len(layers) - 1)
        try:
            if next(layers[i].parameters()).device != next(layers[i + 1].parameters()).device:
                return None
            self.swap_layers(i)
            return i
        except Exception:  # noqa: BLE001
            return None

    @torch.no_grad()
    def weights_finite(self) -> bool:
        return all(bool(torch.isfinite(p).all())
                   for p in self.model.parameters() if p.is_floating_point())

    @torch.no_grad()
    def drift(self, router_only: bool) -> tuple[float, float]:
        sq_d = sq_p = 0.0
        for k, v in self.model.state_dict().items():
            if not v.is_floating_point() or (router_only and not k.endswith("mlp.gate.weight")):
                continue
            b = self.base[k].to(v.device, dtype=torch.float32)
            sq_d += float(((v.float() - b) ** 2).sum())
            sq_p += float((b ** 2).sum())
        return sq_d, sq_p

    def router_counter(self) -> _RouterCounter:
        return _RouterCounter(self.model)

    def reset_peak_memory(self) -> None:
        self._mps_peak = 0
        if torch.cuda.is_available():
            for d in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(d)

    def peak_memory(self) -> dict[str, Any]:
        out: dict[str, Any] = {"cuda_max_memory_allocated": (
            [torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [])}
        if self.name == "torch-mps":  # sampled after every backward and at cell end
            self._sample_mps()
            out["mps_driver_allocated_max_sampled"] = self._mps_peak
        return out

    def cell_split(self, benches: hn.Benches, seed: int, n_replay: int,
                   n_eval: int) -> tuple[hn.Benches, list[dict[str, Any]]]:
        return _cell_split(benches, seed, n_replay, n_eval)

    def shuffle(self, text: str, seed: int) -> str:
        return _shuffle_text(text, seed)


# --- compatibility wrappers (one model, one base snapshot) --------------------


def run_cell(model: Any, tok: Any, base: dict[str, torch.Tensor], benches: hn.Benches,
             **kw: Any) -> dict[str, Any]:
    return hn.run_cell(TorchBackend(model, tok, base), benches, **kw)


def profile_cell(model: Any, tok: Any, base: dict[str, torch.Tensor], benches: hn.Benches,
                 profile: str, seed: int, amendment: pr.Amendment, n_eval: int) -> dict[str, Any]:
    return hn.profile_cell(TorchBackend(model, tok, base), benches, profile, seed, amendment,
                           n_eval)


def _factory(args: argparse.Namespace, benches: hn.Benches,
             common: dict[str, Any]) -> TorchBackend:
    model, tok = _load_model(args.model, args.revision or None)
    return TorchBackend(model, tok)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E1 harness, PyTorch backend")
    hn.add_args(ap)
    return hn.run(ap.parse_args(argv), _factory)


if __name__ == "__main__":
    raise SystemExit(main())
