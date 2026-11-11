"""One-hour test (PI decision of 2026-11-11): how much of a DOWNSCALE or
per-tensor noise step survives the bf16 rounding of the stored weights,
dose by dose, on the real Qwen3.5-4B checkpoint. CPU only, no scoring,
no model forward. Exploratory, not a prereg run.

For each tensor theta (bf16 on disk) and dose eps:
  D(eps):  step_nominal = (1-eps)*theta - theta  (float32)
           step_applied = bf16((1-eps)*theta) - theta
           rounding     = bf16((1-eps)*theta) - (1-eps)*theta
  Rt(eps): xi ~ N(0,1) scaled to ||xi|| = eps*||theta|| (float32, seed 0)
           step_applied = bf16(theta + xi) - theta ; rounding = bf16(theta+xi) - (theta+xi)
Reports Frobenius norms summed over tensors, in total and by tensor group
(norm gains / biases / embeddings+head / matrices), plus the fraction of
elements of D(eps) left unchanged by the rounding.
"""
import argparse
import glob
import json
import os
import struct
import sys
import time

import torch

_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
       "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


class _SafeFile:
    """Minimal safetensors reader (no dependency): 8-byte little-endian header
    length, JSON header, then raw little-endian buffers."""
    def __init__(self, path):
        self.f = open(path, "rb")  # noqa: SIM115  (closed in __exit__)
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = json.loads(self.f.read(n))
        self.h.pop("__metadata__", None)
        self.base = 8 + n
    def keys(self):
        return sorted(self.h)
    def get_tensor(self, k):
        m = self.h[k]
        a, b = m["data_offsets"]
        self.f.seek(self.base + a)
        buf = bytearray(self.f.read(b - a))
        if b == a:
            return torch.empty(m["shape"], dtype=_DT[m["dtype"]])
        return torch.frombuffer(buf, dtype=_DT[m["dtype"]]).reshape(m["shape"]).clone()
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.f.close()


def safe_open(path, framework="pt", device="cpu"):
    try:
        from safetensors import safe_open as _so
        return _so(path, framework=framework, device=device)
    except ImportError:
        return _SafeFile(path)

DOSES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


def group_of(name: str, t: torch.Tensor) -> str:
    n = name.lower()
    if "embed" in n or "lm_head" in n:
        return "embed_head"
    if t.ndim <= 1:
        return "norm_gain" if "norm" in n else "bias_1d"
    return "matrix"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N tensors")
    a = ap.parse_args()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    files = sorted(glob.glob(os.path.join(a.snapshot, "*.safetensors")))
    if not files:
        sys.exit("no safetensors in " + a.snapshot)
    keys = ["theta_sq", "nominal_sq", "applied_sq", "rounding_sq", "unchanged", "n"]
    acc = {f"{arm}:{d}": {} for arm in ("D", "Rt") for d in DOSES}
    def bump(cell, grp, k, v):
        g = cell.setdefault(grp, dict.fromkeys(keys, 0.0))
        g[k] += float(v)
    dtypes, n_t, t0 = {}, 0, time.time()
    g = torch.Generator().manual_seed(a.seed)
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as sf:
            for name in sf.keys():  # noqa: SIM118  (safetensors handle, not a dict)
                t = sf.get_tensor(name)
                dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
                if not t.is_floating_point():
                    continue
                grp = group_of(name, t)
                stored_dtype = t.dtype
                th = t.float()
                th_sq = float((th ** 2).sum())
                xi_unit = torch.randn(th.shape, generator=g, dtype=torch.float32)
                xi_unit /= xi_unit.norm().clamp_min(1e-30)
                for d in DOSES:
                    # D arm
                    target = th * (1.0 - d)
                    stored = target.to(stored_dtype).float()
                    c = acc[f"D:{d}"]
                    bump(c, grp, "theta_sq", th_sq)
                    bump(c, grp, "nominal_sq", ((target - th) ** 2).sum())
                    bump(c, grp, "applied_sq", ((stored - th) ** 2).sum())
                    bump(c, grp, "rounding_sq", ((stored - target) ** 2).sum())
                    bump(c, grp, "unchanged", (stored == th).sum())
                    bump(c, grp, "n", th.numel())
                    # Rt arm (one seed)
                    xi = xi_unit * (d * th_sq ** 0.5)
                    target = th + xi
                    stored = target.to(stored_dtype).float()
                    c = acc[f"Rt:{d}"]
                    bump(c, grp, "theta_sq", th_sq)
                    bump(c, grp, "nominal_sq", (xi ** 2).sum())
                    bump(c, grp, "applied_sq", ((stored - th) ** 2).sum())
                    bump(c, grp, "rounding_sq", ((stored - target) ** 2).sum())
                    bump(c, grp, "unchanged", (stored == th).sum())
                    bump(c, grp, "n", th.numel())
                n_t += 1
                if a.limit and n_t >= a.limit:
                    break
        if a.limit and n_t >= a.limit:
            break
    cfg_path = os.path.join(a.snapshot, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)
    cfg_keys = {k: cfg.get(k) for k in ("architectures", "model_type", "torch_dtype", "dtype", "tie_word_embeddings",
                "attention_bias", "use_qk_norm", "qk_norm", "num_hidden_layers", "hidden_size", "rms_norm_eps")}
    out = {"snapshot": a.snapshot, "config": cfg_keys, "files": [os.path.basename(x) for x in files],
           "tensors": n_t, "dtypes": dtypes, "seed": a.seed, "doses": DOSES,
           "seconds": round(time.time() - t0, 1), "cells": {}}
    for cell, groups in acc.items():
        tot = dict.fromkeys(keys, 0.0)
        for gv in groups.values():
            for k in keys:
                tot[k] += gv[k]
        def summ(v):
            nom, app, rnd = v["nominal_sq"] ** 0.5, v["applied_sq"] ** 0.5, v["rounding_sq"] ** 0.5
            return {"theta_norm": v["theta_sq"] ** 0.5, "nominal_norm": nom,
                    "applied_norm": app, "rounding_norm": rnd,
                    "rounding_over_nominal": rnd / nom if nom else None,
                    "applied_over_nominal": app / nom if nom else None,
                    "frac_unchanged": v["unchanged"] / v["n"] if v["n"] else None}
        out["cells"][cell] = {"total": summ(tot), "groups": {k: summ(v) for k, v in groups.items()}}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for cell, v in out["cells"].items():
        t = v["total"]
        print(f"{cell:8s} rounding/nominal={t['rounding_over_nominal']:.3f} "
              f"applied/nominal={t['applied_over_nominal']:.3f} unchanged={t['frac_unchanged']:.3f}")
    print("wrote", a.out, "in", out["seconds"], "s")


if __name__ == "__main__":
    main()
