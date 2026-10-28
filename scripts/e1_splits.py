#!/usr/bin/env python3
"""E1 item draws as a file, for backends without torch (the MLX port).

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` section 5.2
(``_cell_split``) and the ``p_ctrl_shuffle`` word shuffle. The draws are
defined by the torch generators of ``scripts/pilot_cycle3_cuda.py``
(``_cell_split``: ``randperm`` seeded with the cell seed; ``_shuffle_text``:
``randperm`` seeded with seed + word count). This script evaluates those
exact functions (torch needed, macM1 or the CUDA node) and writes, per
seed, the item order of every bench (cell seed and seed + 9973 for
``p_noise``) and the word permutations the shuffle control needs. The MLX
backend replays them and checks the item hashes, so both backends draw
the same items. No model is loaded and no item is scored.

    python scripts/e1_splits.py --mode tuning --corpus C --manifest M --out splits.json
    python scripts/e1_splits.py --mode smoke --out splits.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e1_corpus_io as cio
from scripts import e1_harness as hn
from scripts import e1_profiles as pr

SCHEMA = "e1-splits/1"
SPLIT_MODES = ("smoke", "tuning", "calibrate", "confirmatory")


def _torch_order(n: int, seed: int) -> list[int]:
    import torch

    g = torch.Generator().manual_seed(seed)
    return [int(i) for i in torch.randperm(n, generator=g).tolist()]


def _torch_perm(n: int, seed: int) -> list[int]:
    import torch

    g = torch.Generator().manual_seed(seed + n)
    return [int(i) for i in torch.randperm(n, generator=g).tolist()]


def build(benches: hn.Benches, seeds: list[int], prov: dict[str, Any]) -> dict[str, Any]:
    import torch

    orders: dict[str, dict[str, list[int]]] = {}
    shuffle: dict[str, list[int]] = {}
    for s in seeds:
        for seed in (s, s + hn.NOISE_SEED_OFFSET):
            orders[str(seed)] = {b: _torch_order(len(items), seed) for b, items in benches.items()}
        replay = [benches[b][i] for b in benches for i in orders[str(s)][b][: hn.N_REPLAY]]
        for text in pr.profile_texts("facts", replay, s):
            n = len(text.split())
            if n >= 2:
                shuffle[f"{s}:{n}"] = _torch_perm(n, s)
    return {"schema": SCHEMA, "torch": torch.__version__, "provenance": prov,
            "seeds": seeds, "n_replay": hn.N_REPLAY, "noise_seed_offset": hn.NOISE_SEED_OFFSET,
            "items": {b: [it["item_sha256"] for it in items] for b, items in benches.items()},
            "orders": orders, "shuffle": shuffle}


class FileSplits:
    """``cell_split`` / ``shuffle`` replayed from a split file."""

    def __init__(self, doc: dict[str, Any], benches: hn.Benches, mode: str | None = None,
                 prov: dict[str, Any] | None = None) -> None:
        if doc.get("schema") != SCHEMA:
            raise SystemExit("not an E1 split file")
        got = {b: [it["item_sha256"] for it in items] for b, items in benches.items()}
        if got != doc["items"]:
            raise SystemExit("the split file was written for other items")
        if doc["n_replay"] != hn.N_REPLAY or doc["noise_seed_offset"] != hn.NOISE_SEED_OFFSET:
            raise SystemExit("the split file uses other split constants")
        if mode is not None:
            dprov = doc.get("provenance") or {}
            if dprov.get("mode") != mode:
                raise SystemExit(f"the split file was written for mode {dprov.get('mode')}")
            for k in ("corpus_sha256", "partition", "partition_sha256", "pilot_manifest_sha256"):
                if prov is not None and k in prov and dprov.get(k) != prov[k]:
                    raise SystemExit(f"the split file does not match the loaded items ({k})")
            want = set(hn.SEEDS[mode])
            if set(doc["seeds"]) != want:
                raise SystemExit("the split file does not cover exactly the mode's seeds")
        for s in doc["seeds"]:
            for seed in (s, s + hn.NOISE_SEED_OFFSET):
                orders = doc["orders"].get(str(seed))
                if orders is None or set(orders) != set(benches):
                    raise SystemExit(f"split file: no order for seed {seed}")
                for b, items in benches.items():
                    if sorted(orders[b]) != list(range(len(items))):
                        raise SystemExit(f"split file: order of {b}, seed {seed} is not a "
                                         "permutation of the items")
        for key, perm in doc["shuffle"].items():
            n = int(key.split(":")[1])
            if sorted(perm) != list(range(n)):
                raise SystemExit(f"split file: shuffle {key} is not a permutation")
        self.doc = doc

    def cell_split(self, benches: hn.Benches, seed: int, n_replay: int,
                   n_eval: int) -> tuple[hn.Benches, list[dict[str, Any]]]:
        orders = self.doc["orders"].get(str(seed))
        if orders is None:
            raise SystemExit(f"seed {seed} is not in the split file")
        eval_sets, replay_items = {}, []
        for name, items in benches.items():
            order = orders[name]
            replay_items += [items[i] for i in order[:n_replay]]
            ev = order[n_replay:]
            eval_sets[name] = [items[i] for i in (ev[:n_eval] if n_eval > 0 else ev)]
        return eval_sets, replay_items

    def shuffle(self, text: str, seed: int) -> str:
        w = text.split()
        if len(w) < 2:
            return text
        perm = self.doc["shuffle"].get(f"{seed}:{len(w)}")
        if perm is None:
            raise SystemExit(f"no shuffle permutation for seed {seed}, {len(w)} words")
        return " ".join(w[i] for i in perm)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True, choices=SPLIT_MODES)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--pilot-partition", default="confirmatory", choices=cio.PARTITIONS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.mode == "smoke":
        benches, prov = cio.load_pilot(hn.FIX, hn.FIX / "e1_partition_2026-10.json",
                                       args.pilot_partition)
    else:
        if not (args.corpus and args.manifest):
            raise SystemExit(f"{args.mode} needs --corpus and --manifest")
        items, prov = cio.load_corpus(args.corpus, args.manifest, args.mode)
        benches = {"facts": items}
    doc = build(benches, list(hn.SEEDS[args.mode]), {"mode": args.mode, **prov})
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    text = json.dumps(doc, sort_keys=True) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
