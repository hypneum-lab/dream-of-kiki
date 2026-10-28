#!/usr/bin/env python3
"""Project the E1-small measurement smoke onto the 1,620-cell workload.

**Gate** : ``docs/osf-prereg-e1-small-2026-10-25.md`` section 9.7b (E1 draft
method: scale each component by the confirmatory workload, not by cell count).

1. Fit, over the 30 smoke cells, wall = a * scoring forwards + b * updates
   + c * update loss tokens + d (restore, diagnostics, overhead), by least
   squares. Scoring forwards per cell = 2 * 4 * n_eval_items (pre + post) +
   4 * n_replay for OPR.
2. Workload per cell kind: confirmatory at ``n_eval`` (130 by default; the
   section 5.2 rule can raise it up to 255), tuning and calibration on the
   rest of the TUNING partition (``n_tuning - 40`` eval items); T_s from the
   corpus (5 x the mean loss tokens of a fact text x 40, raw pinned
   tokenizer); updates per profile scaled from the smoke by T_s.

    python e1_small_projection.py --cells C.jsonl --corpus e1-corpus.jsonl \
        --tokenizer tokenizer.json --n-tuning 472
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

N_REPLAY = 40
CONF_PROFILES = ("p_min", "p_equ", "p_max", "p_ctrl_sft", "p_ctrl_shuffle", "p_noise",
                 "p_ctrl_template", "p_ctrl_vanilla_replay_tuned", "p_ctrl_onpolicy_replay",
                 "p_ctrl_lr_matched")


def forwards(cell: dict[str, Any], n_eval: int | None = None) -> int:
    n = cell["n_eval_items"] if n_eval is None else n_eval
    opr = 4 * int(cell.get("n_replay", N_REPLAY)) if "s_opr" in cell else 0
    return 8 * n + opr


def fit(cells: list[dict[str, Any]]) -> list[float]:
    import numpy as np

    x = np.array([[forwards(c), c["n_updates"], c["achieved_loss_tokens"], 1.0] for c in cells])
    y = np.array([c["wall_s"] for c in cells])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return [float(v) for v in coef]


def mean_fact_loss_tokens(corpus: Path, tokenizer: Path) -> float:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer))
    n = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            text = it["prompt"] + "\nAnswer:" + " " + it["choices"][int(it["answer"])]
            n.append(len(tok.encode(text, add_special_tokens=False).ids) - 1)
    return sum(n) / len(n)


def project(cells: list[dict[str, Any]], coef: list[float], t_s: float, n_eval: int,
            n_tuning: int) -> dict[str, Any]:
    a, b, c, d = coef
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by[cell["profile"]].append(cell)

    def cost(profile: str, ev: int) -> float:
        cs = by[profile]
        upd = sum(x["n_updates"] for x in cs) / len(cs)
        tok = sum(x["achieved_loss_tokens"] for x in cs) / len(cs)
        ts = sum(x["t_s"] for x in cs) / len(cs)
        scale = t_s / ts if ts else 0.0
        return a * forwards(cs[0], ev) + b * upd * scale + c * tok * scale + d

    conf = sum(cost(p, n_eval) for p in CONF_PROFILES) * 141
    tune_ev = n_tuning - N_REPLAY
    tuning = 60 * (cost("p_min", tune_ev) + cost("p_ctrl_onpolicy_replay", tune_ev))
    calib = 60 * cost("p_min", tune_ev)
    smoke = sum(x["wall_s"] for x in cells)
    total = conf + tuning + calib + smoke
    return {"coef_s": {"per_forward": a, "per_update": b, "per_loss_token": c,
                       "per_cell": d},
            "t_s_confirmatory": t_s, "n_eval": n_eval, "tuning_eval_items": tune_ev,
            "hours": {"confirmatory_1410": conf / 3600, "tuning_120": tuning / 3600,
                      "calibration_60": calib / 3600, "smoke_30_measured": smoke / 3600,
                      "total_1620": total / 3600},
            "per_confirmatory_cell_s": conf / 1410}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--n-tuning", type=int, required=True)
    args = ap.parse_args(argv)
    cells = [json.loads(x) for x in args.cells.read_text().splitlines() if x.strip()]
    coef = fit(cells)
    t_s = 5 * N_REPLAY * mean_fact_loss_tokens(args.corpus, args.tokenizer)
    out = {n: project(cells, coef, t_s, n, args.n_tuning) for n in (130, 255)}
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
