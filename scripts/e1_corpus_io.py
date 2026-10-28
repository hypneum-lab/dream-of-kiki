"""E1 corpus loader for the harness (task 2 of section 9).

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` sections
5.1, 5.2, 5.3, 9 (task 2).
**Mode** : stdlib only. Reads a corpus built by
``scripts/e1_corpus_build.py`` and its manifest, verifies every hash,
and returns the items of **one** partition in the harness format.

Status rules (tuning / confirmatory isolation):

- ``confirmatory`` and ``tuning`` cells need a ``FROZEN`` manifest
  (the corpus must exist, frozen and hashed, before tuning starts).
- ``headroom`` scoring needs the ``PENDING-HEADROOM`` manifest and may
  read the ``tuning`` partition only.
- The pilot fixtures (smoke, section 5.3) are loaded with
  :func:`load_pilot`, which checks the fixture partition manifest.

Harness item: ``{prompt, choices, answer, source, item_sha256,
src_hash, question_prompt}`` where ``prompt`` is the frozen prompt
(date prefix included) plus ``"\\nAnswer:"``, each choice has a leading
space (the ``_score_choice`` convention of the pilot loaders), and
``question_prompt`` is the prompt-only replay text of
``p_ctrl_template`` (identical to ``prompt``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

Item = dict[str, Any]
PARTITIONS = ("confirmatory", "tuning")
FROZEN = "FROZEN"
PENDING = "PENDING-HEADROOM"
MODE_STATUS = {"confirmatory": FROZEN, "tuning": FROZEN, "calibrate": FROZEN,
               "headroom": PENDING}
MODE_PARTITION = {"confirmatory": "confirmatory", "tuning": "tuning",
                  "calibrate": "tuning", "headroom": "tuning"}


class CorpusError(RuntimeError):
    """The corpus or its manifest fails verification or isolation."""


def _sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode("utf-8") if isinstance(b, str) else b).hexdigest()


def harness_item(frozen: Item, item_sha256: str) -> Item:
    prompt = frozen["prompt"] + "\nAnswer:"
    return {"prompt": prompt, "question_prompt": prompt,
            "choices": [" " + c for c in frozen["choices"]],
            "answer": int(frozen["answer"]), "source": frozen["source"],
            "item_sha256": item_sha256, "src_hash": frozen["src_hash"]}


def load_corpus(corpus_path: Path, manifest_path: Path, mode: str) -> tuple[list[Item], Item]:
    """Verified items of the partition ``mode`` may read, and a provenance record."""
    if mode not in MODE_STATUS:
        raise CorpusError(f"unknown mode {mode!r}")
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    want = MODE_STATUS[mode]
    if man.get("status") != want:
        raise CorpusError(f"mode {mode} needs a {want} manifest (got {man.get('status')})")
    raw = corpus_path.read_bytes()
    if _sha(raw) != man["corpus_sha256"]:
        raise CorpusError("corpus file SHA-256 does not match the manifest")
    lines = raw.decode("utf-8").splitlines()
    entries = man["items"]
    if len(lines) != len(entries):
        raise CorpusError(f"{len(lines)} corpus lines vs {len(entries)} manifest items")
    by_part: dict[str, list[str]] = {p: [] for p in PARTITIONS}
    partition = MODE_PARTITION[mode]
    out: list[Item] = []
    for line, ent in zip(lines, entries, strict=True):
        h = _sha(line)
        if h != ent["item_sha256"]:
            raise CorpusError(f"item hash mismatch for {ent.get('source_id')}")
        frozen = json.loads(line)
        if frozen["src_hash"] != ent["src_hash"] or frozen["source"] != ent["source"]:
            raise CorpusError(f"manifest entry does not describe line {ent.get('source_id')}")
        by_part[ent["partition"]].append(h)
        if ent["partition"] == partition:
            out.append(harness_item(frozen, h))
    for p in PARTITIONS:
        got = _sha("\n".join(sorted(by_part[p])))
        if got != man["partitions"][p]["sha256"]:
            raise CorpusError(f"{p} partition hash mismatch")
    prov = {"corpus_sha256": man["corpus_sha256"], "status": man["status"],
            "partition": partition, "partition_sha256": man["partitions"][partition]["sha256"],
            "n_items": len(out), "manifest_sha256": _sha(manifest_path.read_bytes())}
    return out, prov


def load_pilot(fixtures_dir: Path, partition_manifest: Path, partition: str) -> tuple[
        dict[str, list[Item]], Item]:
    """Pilot fixtures restricted to one partition of the fixture manifest (smoke only)."""
    if partition not in PARTITIONS:
        raise CorpusError(f"unknown partition {partition!r}")
    man = json.loads(partition_manifest.read_text(encoding="utf-8"))
    benches: dict[str, list[Item]] = {}
    for name, b in man["benchmarks"].items():
        path = fixtures_dir / Path(b["fixture"]).name
        data = path.read_bytes()
        if _sha(data) != b["fixture_sha256"]:
            raise CorpusError(f"{name}: fixture SHA-256 does not match the manifest")
        lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()]
        rows = {it["row"]: it for it in b["items"]}
        items: list[Item] = []
        for i, line in enumerate(lines):
            if _sha(line) != rows[i]["sha256"]:
                raise CorpusError(f"{name}: row {i} hash mismatch")
            if rows[i]["partition"] != partition:
                continue
            r = json.loads(line)
            if name == "mmlu":
                prompt, choices, ans = (str(r["question"]).strip() + "\nAnswer:",
                                        list(r["choices"]), int(r["answer"]))
            else:
                prompt, choices, ans = str(r["ctx"]).strip(), list(r["endings"]), int(r["label"])
            items.append({"prompt": prompt, "question_prompt": prompt,
                          "choices": [" " + c for c in choices], "answer": ans,
                          "source": name, "item_sha256": rows[i]["sha256"]})
        benches[name] = items
    prov = {"pilot_manifest_sha256": _sha(partition_manifest.read_bytes()),
            "partition": partition, "n_items": {k: len(v) for k, v in benches.items()}}
    return benches, prov
