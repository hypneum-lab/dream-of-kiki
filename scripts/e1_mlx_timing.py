#!/usr/bin/env python3
"""E1 MLX component timings on the real model (pilot fixtures only, no corpus item).

Loads the verified checkpoint once and times 20 pilot items' choice scoring,
one replay pass over their fact texts (m = 1), the drift measures and a
restore, for the section 9 runtime projection.

    python scripts/e1_mlx_timing.py <weights-dir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx

from scripts import e1_corpus_io as cio
from scripts import e1_harness as hn
from scripts import e1_profiles as pr
from scripts.e1_mlx import MLXBackend, load, verify_weights
from scripts.e1_schedule import schedule


def main(argv: list[str] | None = None) -> int:
    weights = Path((argv or sys.argv[1:])[0])
    verify_weights(weights)
    t = time.time()
    model, tok, k = load(weights, hn.TOKENIZER_SHA256)
    t_load = time.time() - t
    b = MLXBackend(model, tok, None, k)
    b.snapshot()
    benches, _ = cio.load_pilot(hn.FIX, hn.FIX / "e1_partition_2026-10.json", "tuning")
    items = benches["mmlu"][:10] + benches["hellaswag"][:10]
    mx.reset_peak_memory()
    t = time.time()
    b.score_items(items)
    t_score = time.time() - t
    n_fwd = sum(len(it["choices"]) for it in items)
    tok_fwd = sum(b.n_tokens(it["prompt"] + c) for it in items for c in it["choices"])
    texts = [pr.fact_text(it) for it in items]
    lengths = [b.n_tokens(x) for x in texts]
    ups = schedule(lengths, 1, sum(n - 1 for n in lengths))
    t = time.time()
    r = b.replay(texts, lengths, ups, 1e-4)
    t_rep = time.time() - t
    t = time.time()
    b.drift(False)
    b.drift(True)
    t_drift = time.time() - t
    t = time.time()
    b.restore()
    t_restore = time.time() - t
    n_up, up_tok = int(r["n_executed"]), sum(u.n_input for u in ups)
    print(json.dumps({
        "load_s": t_load, "score_s": t_score, "n_forward": n_fwd, "forward_tokens": tok_fwd,
        "s_per_forward": t_score / n_fwd, "s_per_forward_token": t_score / tok_fwd,
        "replay_s": t_rep, "n_updates": n_up, "update_tokens": up_tok,
        "s_per_update": t_rep / n_up, "s_per_update_token": t_rep / up_tok,
        "drift_s": t_drift, "restore_s": t_restore, "peak_bytes": mx.get_peak_memory()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
