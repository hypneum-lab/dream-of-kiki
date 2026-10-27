#!/usr/bin/env python3
"""E1-small S10 check (PI decision 2026-10-27): is a cell reproducible in
deterministic mode, and do identical computations give identical results?

**Gate** : ``docs/osf-prereg-e1-small-2026-10-25.md`` section 9.7c. Run with
``HL_STUDY=e1-small HL_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8`` and
the registered instrument (``HL_LINEAR_ATTN=torch``) on the CUDA node; smoke
settings (pilot fixtures, confirmatory pilot partition, ``n_eval`` 50,
placeholder amendment). Never part of the study's cells.

One model load, four cells in this order: ``p_min`` seed 2000,
``p_ctrl_vanilla_replay_tuned`` seed 2000, ``p_ctrl_lr_matched`` seed 2000
(with the placeholder amendment all three are the same computation: default
LR and m, no DOWNSCALE, c = 1), and ``p_min`` seed 2000 again. For each cell:
pre / post accuracy, every per-item choice score (pre and post), and the
SHA-256 of all parameters after the cell. PASS iff the four cells agree
exactly on all of them; this also exercises the restore between cells.
An op without a deterministic implementation raises and is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

JOBS = ("p_min", "p_ctrl_vanilla_replay_tuned", "p_ctrl_lr_matched", "p_min")


def param_sha256(model: Any) -> str:
    import torch

    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def compare(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact agreement of every run with the first (pure, tested without torch)."""
    keys = ("pre", "post", "scores", "param_sha256")
    same = {k: all(r[k] == runs[0][k] for r in runs) for k in keys}
    return {"identical": same, "verdict": "PASS" if all(same.values()) else "FAIL"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import torch

    from scripts import e1_corpus_io as cio
    from scripts import e1_harness as hn
    from scripts import e1_profiles as pr
    from scripts import pilot_cycle3_cuda as h

    if hn.STUDY != "e1-small" or not h.deterministic() or not torch.cuda.is_available():
        raise SystemExit("run with HL_STUDY=e1-small HL_DETERMINISTIC=1 on the CUDA node")
    benches, _ = cio.load_pilot(hn.FIX, hn.FIX / "e1_partition_2026-10.json", "confirmatory")
    model, tok = h._load_model(hn.MODEL_REPO, hn.MODEL_REVISION)

    class Recording(h.TorchBackend):  # keeps every per-item choice score
        def __init__(self, *a: Any) -> None:
            super().__init__(*a)
            self.scores: list[list[list[float]]] = []

        def score_items(self, items: Any) -> Any:
            sc, am = super().score_items(items)
            self.scores.append(sc)
            return sc, am

    b = Recording(model, tok)
    versions = b.versions()
    if not hn._study_backend_ok(b.name, versions, hn.STUDY, "smoke"):
        raise SystemExit(f"not the registered instrument: {versions}")
    b.snapshot()
    am = pr.Amendment(pr.DEFAULT, pr.DEFAULT, 1.0, 50)
    runs: list[dict[str, Any]] = []
    error = None
    for profile in JOBS:
        b.scores = []
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            cell = hn.profile_cell(b, benches, profile, 2000, am, 50)
        except RuntimeError as exc:  # e.g. no deterministic implementation of an op
            error = {"profile": profile, "error": str(exc)}
            break
        runs.append({"profile": profile, "pre": cell["pre"], "post": cell["post"],
                     "scores": b.scores, "param_sha256": param_sha256(model),
                     "wall_s": round(time.time() - t0, 1), "n_updates": cell["n_updates"],
                     "peak_bytes": torch.cuda.max_memory_allocated(),
                     "invariant_violation": cell["invariant_violation"]})
        print(f"[det] {profile} pre={cell['pre']:.4f} post={cell['post']:.4f} "
              f"sha={runs[-1]['param_sha256'][:12]} wall={runs[-1]['wall_s']}s", flush=True)
    doc: dict[str, Any] = {"error": error, "versions": versions, "jobs": list(JOBS),
                           "model": hn.MODEL_REPO, "revision": hn.MODEL_REVISION}
    doc.update(compare(runs) if error is None else {"verdict": "ERROR"})
    doc["runs"] = [{k: v for k, v in r.items() if k != "scores"} for r in runs]
    doc["scores_sha256"] = [hashlib.sha256(json.dumps(r["scores"]).encode()).hexdigest()
                            for r in runs]
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: doc[k] for k in ("verdict", "identical", "error") if k in doc}))
    return 0 if doc["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
