#!/usr/bin/env python3
"""Digest of the raw pinned-tokenizer ids of every harness text of the corpus.

**Gate** : ``docs/osf-prereg-e1-small-2026-10-25.md`` (PI decision S8). The
harness tokenizes with the raw pinned ``tokenizer.json`` through the
``tokenizers`` package, the counter that built the corpus (``tokenizers``
0.23.3 on macM1). The run node pins ``tokenizers`` 0.22.2 (transformers
5.7.0 requires <= 0.23.0). Running this script under both versions and
comparing the digests shows that both give the same ids on every text the
harness scores or replays from the corpus.

Texts per corpus item (harness format, ``scripts/e1_corpus_io.harness_item``):
the prompt ``P = prompt + "\\nAnswer:"`` and ``P + " " + choice`` for each of
the four choices. No special tokens.

    python e1_small_token_ids.py --corpus e1-corpus.jsonl --tokenizer tokenizer.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"


def harness_texts(line: str) -> list[str]:
    item = json.loads(line)
    p = item["prompt"] + "\nAnswer:"
    return [p] + [p + " " + c for c in item["choices"]]


def digest(corpus: Path, encode: Any) -> dict[str, Any]:
    h = hashlib.sha256()
    n = longest = 0
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for t in harness_texts(line):
            ids = encode(t)
            h.update(json.dumps(ids, separators=(",", ":")).encode() + b"\n")
            n += 1
            longest = max(longest, len(ids))
    return {"sha256": h.hexdigest(), "n_texts": n, "max_tokens": longest,
            "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    args = ap.parse_args(argv)
    got = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    if got != TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer.json SHA-256 {got} != pinned {TOKENIZER_SHA256}")
    import tokenizers
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.tokenizer))
    doc = digest(args.corpus, lambda t: tok.encode(t, add_special_tokens=False).ids)
    doc["tokenizers"] = tokenizers.__version__
    print(json.dumps(doc, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
