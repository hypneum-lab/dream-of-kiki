"""DR-3 export helper: per-seed representation dumps from nerve-wml v1.9.0.

**Gate** : pre-filing gate of ``docs/osf-prereg-dr3-relational-probe-2026-10-23.md``
(section 9); training and export of section 5, learning gate and
representation inputs of section 4.
**Validates** : nothing by itself. **Measurement only** : trains the
v1.9.0 ``MlpWML`` / ``LifWML`` pair (800 steps) and the untrained null
pair (0 steps) per seed, exports the pre-VQ codes on the seed's 1024
inputs, scores the learning gate and writes one ``.npz`` per seed plus
a ``.sha256`` sidecar and a manifest. It computes no relational
statistic (``scripts/dr3_relational_probe.py`` does).
**Mode** : pipeline code, no empirical claim.

Standalone by design (PI decision 2026-10-24): dream-of-kiki does not
depend on nerve-wml or torch. The script is run with the Python
environment of a nerve-wml checkout at tag ``v1.9.0`` (commit
``95ac6b78477c3ce999b31bca775769041e082b46``) and imports the v1.9.0
training functions ``_train_mlp`` / ``_train_lif`` of
``scripts/platonic_rh_alignment.py`` unchanged. Only the standard
library and numpy are imported at module level; torch and nerve-wml are
imported when the export runs.

Invocation ::

    git clone https://github.com/hypneum-lab/nerve-wml.git nwml-v1.9.0
    cd nwml-v1.9.0 && git checkout --detach v1.9.0
    git rev-parse HEAD   # 95ac6b78477c3ce999b31bca775769041e082b46
    uv sync --frozen
    uv run python /path/to/dream-of-kiki/scripts/dr3_export_nerve_wml.py \\
        --seeds 100-119 --out /path/to/dumps

Outputs in ``--out`` (existing files are never overwritten):

* ``dr3-seed<seed>.npz``: ``seed``, ``a_trained``, ``b_trained``,
  ``a_null``, ``b_null`` (float32, 1024 x 16, row ``i`` = input ``i``),
  the learning-gate accuracies ``acc_a_trained``, ``acc_b_trained``,
  ``acc_a_null``, ``acc_b_null`` and ``gate_pass``;
* ``dr3-seed<seed>.npz.sha256``: ``<hex>  <file name>``;
* ``export-manifest.json`` (+ ``.sha256``): pin, versions, protocol
  constants and, per seed, the file hash, per-array content hashes
  (``sha256`` of the C-order float32 bytes, comparable across reruns
  independently of the ``.npz`` container), accuracies and gate.

Seeds failing the gate are dumped like the others and are excluded by
``scripts/dr3_relational_probe.py`` from ``gate_pass``, so the section 4
abort count sees every exclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

NERVE_WML_TAG = "v1.9.0"
NERVE_WML_COMMIT = "95ac6b78477c3ce999b31bca775769041e082b46"
STEPS = 800
N_EVAL = 1024
DIM = 16
N_CLASSES = 12
GATE_THRESHOLD = 0.25
GATE_SEED_OFFSET = 5000
ARRAY_KEYS = ("a_trained", "b_trained", "a_null", "b_null")
MANIFEST = "export-manifest.json"

GitRunner = Callable[[Sequence[str], Path], str]


class SubstrateAPI(Protocol):
    """What the export needs from nerve-wml (torch adapter or test fake)."""

    def train_mlp(self, steps: int, seed: int) -> Any: ...

    def train_lif(self, steps: int, seed: int) -> Any: ...

    def eval_inputs(self, seed: int, n: int) -> Any: ...

    def gate_batch(self, seed: int, n: int) -> tuple[Any, Any]: ...

    def embed_mlp(self, mlp: Any, x: Any) -> np.ndarray: ...

    def embed_lif(self, lif: Any, x: Any) -> np.ndarray: ...

    def accuracy_mlp(self, mlp: Any, x: Any, y: Any) -> float: ...

    def accuracy_lif(self, lif: Any, x: Any, y: Any) -> float: ...

    def versions(self) -> dict[str, str]: ...


def _git(args: Sequence[str], root: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def check_checkout(root: Path, run: GitRunner = _git, allow_unpinned: bool = False) -> dict[str, Any]:
    """Check that ``root`` is a clean nerve-wml checkout at the v1.9.0 commit.

    Tracked files must be unmodified (untracked files are allowed, for
    example ``.venv``). With ``allow_unpinned`` a mismatch is recorded
    (``pinned: false``, non-confirmatory) instead of raising.
    """
    head = run(["rev-parse", "HEAD"], root)
    dirty = run(["status", "--porcelain", "--untracked-files=no"], root)
    problems = []
    if head != NERVE_WML_COMMIT:
        problems.append(f"HEAD is {head}, expected {NERVE_WML_COMMIT} ({NERVE_WML_TAG})")
    if dirty:
        problems.append("tracked files are modified")
    if problems and not allow_unpinned:
        raise RuntimeError("nerve-wml checkout not pinned: " + "; ".join(problems))
    return {"head": head, "clean": not dirty, "pinned": not problems, "problems": problems}


def parse_seeds(spec: str) -> list[int]:
    """``"100-119"``, ``"900"`` or ``"0,1,2"`` (ranges inclusive)."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"empty seed range {part!r}")
            seeds.extend(range(lo, hi + 1))
        elif part:
            seeds.append(int(part))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seed list {spec!r} is empty or has duplicates")
    return seeds


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_sidecar(path: Path) -> str:
    digest = _sha256_bytes(path.read_bytes())
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def _as_f32(arr: np.ndarray, name: str) -> np.ndarray:
    out = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    if out.shape != (N_EVAL, DIM):
        raise ValueError(f"{name}: expected shape {(N_EVAL, DIM)}, got {out.shape}")
    return out


def dump_path(out_dir: Path, seed: int) -> Path:
    return out_dir / f"dr3-seed{seed}.npz"


def export_seed(api: SubstrateAPI, seed: int, out_dir: Path) -> dict[str, Any]:
    """Train, export, gate and dump one seed; return its manifest entry."""
    path = dump_path(out_dir, seed)
    if path.exists():
        raise FileExistsError(f"{path} exists; dumps are never overwritten")
    mlp_t = api.train_mlp(STEPS, seed)
    lif_t = api.train_lif(STEPS, seed)
    mlp_0 = api.train_mlp(0, seed)
    lif_0 = api.train_lif(0, seed)
    x = api.eval_inputs(seed, N_EVAL)
    arrays = {
        "a_trained": _as_f32(api.embed_mlp(mlp_t, x), "a_trained"),
        "b_trained": _as_f32(api.embed_lif(lif_t, x), "b_trained"),
        "a_null": _as_f32(api.embed_mlp(mlp_0, x), "a_null"),
        "b_null": _as_f32(api.embed_lif(lif_0, x), "b_null"),
    }
    xg, yg = api.gate_batch(seed, N_EVAL)
    acc = {
        "a_trained": float(api.accuracy_mlp(mlp_t, xg, yg)),
        "b_trained": float(api.accuracy_lif(lif_t, xg, yg)),
        "a_null": float(api.accuracy_mlp(mlp_0, xg, yg)),
        "b_null": float(api.accuracy_lif(lif_0, xg, yg)),
    }
    reasons = [
        f"learning gate: {k} accuracy {acc[k]:.4f} < {GATE_THRESHOLD}"
        for k in ("a_trained", "b_trained")
        if acc[k] < GATE_THRESHOLD
    ]
    gate_pass = not reasons
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"seed": np.int64(seed), **arrays}
    payload.update({f"acc_{k}": np.float64(v) for k, v in acc.items()})
    payload["gate_pass"] = np.bool_(gate_pass)
    np.savez(path, **payload)
    digest = _write_sidecar(path)
    return {
        "seed": seed,
        "file": path.name,
        "sha256": digest,
        "array_sha256": {k: _sha256_bytes(v.tobytes()) for k, v in arrays.items()},
        "accuracy": acc,
        "gate_pass": gate_pass,
        "gate_reasons": reasons,
    }


def run_export(
    api: SubstrateAPI, seeds: Sequence[int], out_dir: Path, checkout: dict[str, Any]
) -> dict[str, Any]:
    """Export every seed, then write and hash the manifest."""
    manifest_path = out_dir / MANIFEST
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists; dumps are never overwritten")
    for s in seeds:
        if dump_path(out_dir, s).exists():
            raise FileExistsError(f"{dump_path(out_dir, s)} exists; dumps are never overwritten")
    entries = [export_seed(api, s, out_dir) for s in seeds]
    doc = {
        "script": "scripts/dr3_export_nerve_wml.py",
        "prereg": "docs/osf-prereg-dr3-relational-probe-2026-10-23.md",
        "nerve_wml": {"tag": NERVE_WML_TAG, "commit": NERVE_WML_COMMIT, **checkout},
        "confirmatory_eligible": bool(checkout.get("pinned")),
        "versions": api.versions(),
        "protocol": {
            "steps": STEPS,
            "n_eval": N_EVAL,
            "dim": DIM,
            "n_classes": N_CLASSES,
            "layer": "pre-VQ: MlpWML core(x); LifWML input_proj(encoder(x))",
            "null": "same constructors and seeds, 0 steps",
            "gate_threshold": GATE_THRESHOLD,
            "gate_inputs": f"task._gen reseeded with seed + {GATE_SEED_OFFSET}",
        },
        "seeds": entries,
        "n_gate_fail": sum(not e["gate_pass"] for e in entries),
    }
    manifest_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    _write_sidecar(manifest_path)
    return doc


class TorchNerveWmlAPI:
    """Adapter over the v1.9.0 code; imports torch and nerve-wml lazily."""

    def __init__(self, root: Path) -> None:
        root = root.resolve()
        for p in (root / "scripts", root):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        # nerve-wml v1.9.0 and torch live in the checkout's environment
        import platonic_rh_alignment as pra  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        import track_w  # type: ignore[import-not-found]
        from track_w._surrogate import (  # type: ignore[import-not-found]
            spike_with_surrogate,
        )
        from track_w.tasks.hard_flow_proxy import (  # type: ignore[import-not-found]
            HardFlowProxyTask,
        )

        for mod in (pra, track_w):
            mod_file = Path(str(mod.__file__)).resolve()
            if root not in mod_file.parents:
                raise RuntimeError(f"{mod.__name__} imported from {mod_file}, not from {root}")
        self._torch = torch
        self._pra = pra
        self._spike = spike_with_surrogate
        self._task_cls = HardFlowProxyTask

    def train_mlp(self, steps: int, seed: int) -> Any:
        return self._pra._train_mlp(steps, seed)

    def train_lif(self, steps: int, seed: int) -> Any:
        return self._pra._train_lif(steps, seed)

    def eval_inputs(self, seed: int, n: int) -> Any:
        task = self._task_cls(dim=DIM, n_classes=N_CLASSES, seed=seed)
        x, _ = task.sample(batch=n)
        return x

    def gate_batch(self, seed: int, n: int) -> tuple[Any, Any]:
        task = self._task_cls(dim=DIM, n_classes=N_CLASSES, seed=seed)
        task._gen = self._torch.Generator().manual_seed(seed + GATE_SEED_OFFSET)
        x, y = task.sample(batch=n)
        return x, y

    def embed_mlp(self, mlp: Any, x: Any) -> np.ndarray:
        with self._torch.no_grad():
            out: np.ndarray = mlp.core(x).cpu().numpy()
        return out

    def embed_lif(self, lif: Any, x: Any) -> np.ndarray:
        model, encoder = lif
        with self._torch.no_grad():
            out: np.ndarray = model.input_proj(encoder(x)).cpu().numpy()
        return out

    def accuracy_mlp(self, mlp: Any, x: Any, y: Any) -> float:
        with self._torch.no_grad():
            logits = mlp.emit_head_pi(mlp.core(x))[:, :N_CLASSES]
            return float((logits.argmax(-1) == y).float().mean())

    def accuracy_lif(self, lif: Any, x: Any, y: Any) -> float:
        model, encoder = lif
        with self._torch.no_grad():
            spikes = self._spike(model.input_proj(encoder(x)), v_thr=model.v_thr)
            logits = model.emit_head_pi(spikes)[:, :N_CLASSES]
            return float((logits.argmax(-1) == y).float().mean())

    def versions(self) -> dict[str, str]:
        return {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(self._torch.__version__),
            "platform": platform.platform(),
        }


def main(
    argv: list[str] | None = None,
    api_factory: Callable[[Path], SubstrateAPI] = TorchNerveWmlAPI,
    git: GitRunner = _git,
) -> int:
    ap = argparse.ArgumentParser(description="DR-3 export from nerve-wml v1.9.0")
    ap.add_argument("--nerve-wml-root", type=Path, default=Path.cwd(),
                    help="nerve-wml checkout at v1.9.0 (default: current directory)")
    ap.add_argument("--seeds", required=True, help="e.g. 100-119, 900 or 0,1,2")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="testing only: record a non-pinned checkout instead of failing")
    args = ap.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    checkout = check_checkout(args.nerve_wml_root, run=git, allow_unpinned=args.allow_unpinned)
    api = api_factory(args.nerve_wml_root)
    doc = run_export(api, seeds, args.out, checkout)
    print(
        f"wrote {len(seeds)} dump(s) to {args.out}: gate failures={doc['n_gate_fail']}"
        f" pinned={checkout['pinned']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
