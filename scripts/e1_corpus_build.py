#!/usr/bin/env python3
"""Build and check the E1 fact corpus (section 5.1b steps 1-8).

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` sections
5.1, 5.1a, 5.1b, task 7 of section 9.
**Mode** : stdlib + numpy (through ``scripts.e1_corpus_core``). No
model is loaded and no item is scored: step 9 (headroom, base
checkpoint on the TUNING partition only) runs on the CUDA node with
``pilot_cycle3_cuda.py --mode headroom`` and is applied with
``scripts.e1_corpus_core.headroom_rebalance``.

Outputs in ``--out``:

- ``e1-corpus.jsonl``: one frozen item per line, the canonical JSON of
  ``{source, source_id, date, prompt, choices, answer, src_hash}``, so
  the SHA-256 of a line (UTF-8, no newline) is its ``item_sha256``.
  It holds ``realtimeqa`` text and is **not** committed (rule 9); it is
  rebuilt from the pinned sources.
- ``e1-corpus-manifest.json``: raw-file pins and SHA-256, per-step and
  per-source counts, item IDs, hashes and partitions, partition hashes,
  the corpus file SHA-256 and the build status. No item text.

Token rule of step 3 (question + gold <= 256 tokens under the pinned
tokenizer): ``--tokenizer PATH`` counts with the pinned tokenizer files
(needs the ``tokenizers`` package); without it the guaranteed byte
bound of ``scripts.e1_sources.byte_bound_counter`` decides the items it
can, and ``--allow-undecided`` keeps the others, flagged, in a build
whose status is ``PROVISIONAL-TOKENIZER`` (never to be frozen).

Step 9: ``--freeze HEADROOM_JSON`` (``{"correct": {src_hash: bool}}`` for
every TUNING item, written by ``pilot_cycle3_cuda.py --mode headroom``)
rebuilds, applies ``headroom_rebalance`` and writes
``e1-corpus-frozen.jsonl`` + ``e1-corpus-frozen-manifest.json`` with
status FROZEN (or REJECTED, no items). Only a FROZEN corpus feeds
tuning and confirmatory cells.

    python scripts/e1_corpus_build.py --sources <dir> --out <dir> [--allow-undecided]
    python scripts/e1_corpus_build.py --sources <dir> --out <dir> --check
    python scripts/e1_corpus_build.py --sources <dir> --out <dir> --freeze headroom.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e1_sources as src
from scripts.e1_corpus_core import (
    ACCEPT,
    CUTOFF,
    MIN_CONFIRMATORY,
    MIN_TUNING,
    cap_and_split,
    canonical_json,
    contamination_drop,
    dedupe,
    final_item,
    headroom_rebalance,
    skeleton_cap,
    src_hash,
)

Item = dict[str, Any]
SOURCES = ("realtimeqa", "daily_oracle", "mmlu_cf")
CORPUS_FILE = "e1-corpus.jsonl"
MANIFEST_FILE = "e1-corpus-manifest.json"
SCHEMA = "e1-corpus/1"
PROVISIONAL = "PROVISIONAL-TOKENIZER"
PENDING = "PENDING-HEADROOM"
FROZEN = "FROZEN"
REJECTED = "REJECTED"
FROZEN_CORPUS_FILE = "e1-corpus-frozen.jsonl"
FROZEN_MANIFEST_FILE = "e1-corpus-frozen-manifest.json"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _counts(items: Iterable[Item]) -> dict[str, int]:
    c = Counter(it["source"] for it in items)
    return {s: c.get(s, 0) for s in SOURCES}


def build(
    raw: list[Item],
    reference: list[Item],
    count: src.TokenCounter,
    allow_undecided: bool = False,
) -> tuple[list[Item], dict[str, Any]]:
    """Steps 2-8 on loaded rows. Returns (frozen items with partition, report)."""
    parts, report = build_parts(raw, reference, count, allow_undecided)
    return freeze_items(parts, report), report


def build_parts(
    raw: list[Item],
    reference: list[Item],
    count: src.TokenCounter,
    allow_undecided: bool = False,
) -> tuple[dict[str, dict[str, list[Item]]], dict[str, Any]]:
    """Steps 2-7: source items (pre-permutation) per source and partition."""
    report: dict[str, Any] = {"steps": {}}
    steps = report["steps"]
    steps["1_load"] = _counts(raw)
    rows = [src.normalise(r) for r in raw]  # step 2
    steps["2_normalise"] = _counts(rows)

    kept: list[Item] = []
    reasons: dict[str, Counter[str]] = {s: Counter() for s in SOURCES}
    undecided: list[Item] = []
    for r in rows:  # step 3
        why = src.exclusion_reason(r, count)
        if why == "undecided_tokens":
            undecided.append(r)
            if not allow_undecided:
                continue
            why = None
        if why is None:
            kept.append(r)
        else:
            reasons[r["source"]][why] += 1
    if undecided and not allow_undecided:
        raise SystemExit(f"{len(undecided)} item(s) undecided by the token rule; pass "
                         "--tokenizer or --allow-undecided (provisional build)")
    steps["3_exclude"] = _counts(kept)
    report["exclusions"] = {s: dict(sorted(c.items())) for s, c in reasons.items()}
    report["undecided_tokens"] = sorted(
        ({"source": r["source"], "source_id": r["source_id"], "src_hash": src_hash(r)}
         for r in undecided), key=lambda d: (d["source"], d["src_hash"]))

    keep, _, _ = src.split_by_date(kept)  # step 4
    _, pre_ref, no_date = src.split_by_date(rows)
    steps["4a_date"] = _counts(keep)
    dated_kept = [it for it in keep if it["source"] != "mmlu_cf"]
    undated = [it for it in keep if it["source"] == "mmlu_cf"]
    clean = contamination_drop(dated_kept, pre_ref) + undated
    steps["4b_contamination"] = _counts(clean)
    report["pre_cutoff_reference_rows"] = _counts(pre_ref)
    report["dated_rows_without_date"] = _counts(no_date)

    deduped, dstats = dedupe(clean, reference)  # step 5
    steps["5a_dedupe"] = _counts(deduped)
    report["dedupe"] = dstats
    capped = skeleton_cap(deduped)
    steps["5b_skeleton_cap"] = _counts(capped)

    bank = {s: [it for it in capped if it["source"] == s] for s in SOURCES}  # steps 6-7
    parts = cap_and_split(bank)
    steps["6_cap"] = {s: len(p["tuning"]) + len(p["confirmatory"]) for s, p in parts.items()}
    report["cap_k"] = max(len(bank[s]) for s in ("realtimeqa", "daily_oracle"))
    steps["7_split"] = {s: {k: len(v) for k, v in p.items()} for s, p in parts.items()}
    return parts, report


def freeze_items(parts: Mapping[str, Mapping[str, list[Item]]],
                 report: dict[str, Any]) -> list[Item]:
    """Step 8: permute, prefix, hash; adds the partition and the token flag."""
    undecided_hashes = {d["src_hash"] for d in report["undecided_tokens"]}
    frozen: list[Item] = []  # step 8
    longest: dict[str, Counter[str]] = {s: Counter() for s in SOURCES}
    for s in SOURCES:
        for label in ("tuning", "confirmatory"):
            for it in parts.get(s, {}).get(label, []):
                f = final_item(it)
                f["partition"] = label
                f["undecided_tokens"] = f["src_hash"] in undecided_hashes
                frozen.append(f)
                lens = [len(c) for c in it["choices"]]
                g = int(it["answer"])
                longest[s]["gold_longest" if lens[g] == max(lens) else "other"] += 1
    frozen.sort(key=lambda f: (f["partition"], f["source"], f["src_hash"]))
    report["gold_longest"] = {s: dict(c) for s, c in longest.items()}
    return frozen


def corpus_text(frozen: Iterable[Mapping[str, Any]]) -> str:
    keys = ("source", "source_id", "date", "prompt", "choices", "answer", "src_hash")
    return "".join(canonical_json({k: f[k] for k in keys}) + "\n" for f in frozen)


def manifest(frozen: list[Item], report: dict[str, Any], pins: dict[str, Any],
             corpus: str, token_counter: str, reference_info: dict[str, Any],
             headroom: dict[str, Any] | None = None) -> dict[str, Any]:
    parts: dict[str, Any] = {}
    for label in ("confirmatory", "tuning"):
        mem = [f for f in frozen if f["partition"] == label]
        hs = sorted(f["item_sha256"] for f in mem)
        parts[label] = {"n_items": len(mem), "by_source": _counts(mem),
                        "sha256": _sha("\n".join(hs))}
    n_conf = parts["confirmatory"]["n_items"]
    status = PROVISIONAL if report["undecided_tokens"] else PENDING
    note = ("step 9 (headroom on TUNING only) not run; partitions are pre-headroom"
            + ("; token rule undecided for flagged items" if status == PROVISIONAL else ""))
    if headroom is not None:
        status = headroom["status_label"]
        note = f"step 9 applied: {headroom['status']} {headroom.get('reason', '')}".strip()
    minima_ok = n_conf >= MIN_CONFIRMATORY and parts["tuning"]["n_items"] >= MIN_TUNING
    return {
        "headroom": headroom,
        "schema": SCHEMA,
        "prereg": "docs/osf-prereg-e1-replay-controls-2026-10-23.md",
        "script": "scripts/e1_corpus_build.py",
        "status": status,
        "status_note": note,
        "cutoff": CUTOFF,
        "token_counter": token_counter,
        "sources": pins,
        "reference_rows": reference_info,
        "row_mapping": {
            "realtimeqa": "question_sentence; choices; answer = one-element list of an "
                          "integer index; question_date YYYY/MM/DD -> YYYY-MM-DD; "
                          "source_id <path>:<0-based line>",
            "daily_oracle": "question; choice_a..choice_d; answer letter a-d; date; "
                            "source_id <0-based csv data row>",
            "mmlu_cf": "Question; A..D; Answer letter A-D; no date; "
                       "source_id val.csv:<0-based csv data row>",
        },
        "report": report,
        "minima_met_pre_headroom": minima_ok,
        "weights_pre_headroom": {s: (c / n_conf if n_conf else 0.0)
                                 for s, c in parts["confirmatory"]["by_source"].items()},
        "partitions": parts,
        "corpus_file": FROZEN_CORPUS_FILE if headroom is not None else CORPUS_FILE,
        "corpus_sha256": _sha(corpus),
        "items": [{"item_sha256": f["item_sha256"], "src_hash": f["src_hash"],
                   "source": f["source"], "source_id": f["source_id"], "date": f["date"],
                   "partition": f["partition"], "undecided_tokens": f["undecided_tokens"]}
                  for f in frozen],
    }


def mega_v2_facts() -> list[dict[str, Any]]:
    from harness.real_benchmarks.mega_v2_eval import _mega_v2_default_fallback_records

    return _mega_v2_default_fallback_records()


def run(sources_dir: Path, count: src.TokenCounter, token_counter: str,
        allow_undecided: bool,
        tree_blobs: Callable[[Path, str], dict[str, str]] = src.git_tree_blobs,
        fixtures_dir: Path = ROOT / "tests" / "fixtures",
        mega: list[dict[str, Any]] | None = None,
        headroom: Mapping[str, bool] | None = None,
        headroom_sha256: str | None = None) -> tuple[str, str]:
    """Return (corpus text, manifest text).

    With ``headroom`` (the base checkpoint's correctness on every TUNING
    item, keyed by src_hash, from ``pilot_cycle3_cuda.py --mode
    headroom``) step 9 is applied with ``headroom_rebalance``: the
    result is the FROZEN corpus (or REJECTED, with no items).
    """
    pins = src.verify_pins(sources_dir, tree_blobs)
    raw = src.load_all(sources_dir)
    facts = mega if mega is not None else mega_v2_facts()
    reference = src.reference_rows(fixtures_dir, facts)
    ref_info = {
        "mmlu_real_sha256": src.sha256_file(fixtures_dir / "mmlu_real.jsonl"),
        "hellaswag_sanity_sha256": src.sha256_file(fixtures_dir / "hellaswag_sanity.jsonl"),
        "mega_v2_n": len(facts),
        "mega_v2_sha256": _sha(canonical_json({"facts": facts})),
        "n_rows": dict(Counter(r["source"] for r in reference)),
    }
    parts, report = build_parts(raw, reference, count, allow_undecided)
    hr: dict[str, Any] | None = None
    if headroom is None:
        frozen = freeze_items(parts, report)
    else:
        if report["undecided_tokens"]:
            raise SystemExit("refusing to freeze a PROVISIONAL-TOKENIZER build")
        tuning = {src_hash(x) for p in parts.values() for x in p["tuning"]}
        foreign = set(headroom) - tuning
        if foreign:  # confirmatory items are never scored before the first seed
            raise SystemExit(f"headroom input has {len(foreign)} non-TUNING item(s)")
        res = headroom_rebalance(parts, headroom)
        hr = {k: res[k] for k in ("status", "reason", "accuracy", "dropped")}
        hr["pooled_accuracy"] = res.get("pooled_accuracy")
        hr["weights"] = res.get("weights")
        hr["input_sha256"] = headroom_sha256
        hr["status_label"] = FROZEN if res["status"] == ACCEPT else REJECTED
        frozen = freeze_items(res["partitions"], report) if res["status"] == ACCEPT else []
    corpus = corpus_text(frozen)
    man = manifest(frozen, report, pins, corpus, token_counter, ref_info, hr)
    return corpus, json.dumps(man, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def _tokenizer_counter(path: Path) -> src.TokenCounter:
    from tokenizers import Tokenizer  # pinned tokenizer files; optional dependency

    tok = Tokenizer.from_file(str(path))

    def count(question: str, gold: str) -> int | None:
        return len(tok.encode(question, add_special_tokens=False).ids) + len(
            tok.encode(gold, add_special_tokens=False).ids)

    return count


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, help="pinned tokenizer.json")
    ap.add_argument("--allow-undecided", action="store_true")
    ap.add_argument("--check", action="store_true", help="rebuild and compare, write nothing")
    ap.add_argument("--freeze", type=Path, metavar="HEADROOM_JSON",
                    help="apply step 9 from the TUNING-only headroom scores")
    args = ap.parse_args(argv)
    if args.tokenizer:
        count, name = _tokenizer_counter(args.tokenizer), f"tokenizer:{src.sha256_file(args.tokenizer)}"
    else:
        count, name = src.byte_bound_counter, "byte-bound (NFC UTF-8 bytes, q + 1 + gold)"
    hr, hr_sha = None, None
    if args.freeze:
        hr = {str(k): bool(v) for k, v in
              json.loads(args.freeze.read_text(encoding="utf-8"))["correct"].items()}
        hr_sha = src.sha256_file(args.freeze)
    corpus, man = run(args.sources, count, name, args.allow_undecided,
                      headroom=hr, headroom_sha256=hr_sha)
    cpath, mpath = ((args.out / FROZEN_CORPUS_FILE, args.out / FROZEN_MANIFEST_FILE) if args.freeze
                    else (args.out / CORPUS_FILE, args.out / MANIFEST_FILE))
    if args.check:
        ok = (cpath.is_file() and mpath.is_file()
              and cpath.read_text(encoding="utf-8") == corpus
              and mpath.read_text(encoding="utf-8") == man)
        print("corpus and manifest OK" if ok else "corpus or manifest missing or stale",
              file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1
    args.out.mkdir(parents=True, exist_ok=True)
    for p in (cpath, mpath):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    cpath.write_text(corpus, encoding="utf-8")
    mpath.write_text(man, encoding="utf-8")
    m = json.loads(man)
    print(f"status={m['status']} confirmatory={m['partitions']['confirmatory']['n_items']} "
          f"tuning={m['partitions']['tuning']['n_items']} corpus_sha256={m['corpus_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
