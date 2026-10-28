"""E1 fact-corpus construction core (deterministic parts of section 5.1b).

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` sections
5.1, 5.1a, 5.1b (study gate; task 7).
**Mode** : reference implementation of the deterministic steps (hashes,
near-duplicate grouping, skeleton cap, date contamination check, cap,
quotas, split, headroom rebalancing) on already-loaded items. Loading
the pinned sources, tokenisation and model scoring stay in task 7.

An item is ``{source, source_id, date, question, choices, answer}``
after whitespace normalisation (rule 2): ``date`` ``YYYY-MM-DD`` or
None; ``answer`` the index of the gold option in source order.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

CUTOFF = "2026-04-16"
DATED = ("realtimeqa", "daily_oracle")
UNDATED = "mmlu_cf"
Q_THRESHOLD = 0.6
ANSWER_THRESHOLD = 0.4
SKELETON_CAP = 5
TUNING_TENTHS = 3  # ceil(0.30 n) items per source go to tuning
HEADROOM = 0.80
MIN_CONFIRMATORY = 300
MIN_TUNING = 120
REJECT = "REJECT"
ACCEPT = "ACCEPT"

Item = dict[str, Any]


def canonical_json(obj: Mapping[str, Any]) -> str:
    """UTF-8 canonical form: sorted keys, compact separators, no escaping."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def src_hash(item: Mapping[str, Any]) -> str:
    """Source-item hash: normalised, pre-permutation; drives every choice."""
    keys = ("source", "source_id", "date", "question", "choices", "answer")
    return _sha(canonical_json({k: item[k] for k in keys}))


def final_item(item: Mapping[str, Any]) -> Item:
    """Frozen item: options permuted from src_hash, date prefix, final hash."""
    sh = src_hash(item)
    perm = [int(j) for j in np.random.default_rng(int(sh[:16], 16)).permutation(4)]
    choices = [item["choices"][j] for j in perm]
    prompt = f"Date: {item['date']}\n{item['question']}" if item["date"] else item["question"]
    frozen: Item = {
        "source": item["source"], "source_id": item["source_id"], "date": item["date"],
        "prompt": prompt, "choices": choices, "answer": perm.index(int(item["answer"])),
        "src_hash": sh,
    }
    frozen["item_sha256"] = _sha(canonical_json(frozen))
    return frozen


# --- near-duplicates ---------------------------------------------------------


def norm_key(text: str) -> str:
    """Dedup / skeleton key: NFKC, lower-case, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def grams(text: str) -> set[str]:
    """Character 5-grams; a non-empty string shorter than 5 is one gram."""
    if not text:
        return set()
    if len(text) < 5:
        return {text}
    return {text[i : i + 5] for i in range(len(text) - 4)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard index; two empty sets give 0 (never a duplicate)."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _gold(item: Mapping[str, Any]) -> str | None:
    """Normalised gold option, or None when the gold is missing or
    invalid (choices not a list, answer not an int in range). Such a
    row (possible only among pre-cutoff reference rows) can match by
    question similarity alone, never by the equal-gold rule."""
    choices, ans = item.get("choices"), item.get("answer")
    if not isinstance(choices, list) or isinstance(ans, bool) or not isinstance(ans, int):
        return None
    if not 0 <= ans < len(choices):
        return None
    return norm_key(str(choices[ans]))


Key = tuple[set[str], str | None]


def _key(item: Mapping[str, Any]) -> Key:
    return grams(norm_key(str(item.get("question") or ""))), _gold(item)


def _near_dup_keys(ka: Key, kb: Key, q_t: float, a_t: float) -> bool:
    j = jaccard(ka[0], kb[0])
    if j >= q_t:
        return True
    # the gold is consulted only once the question similarity reaches a_t
    return j >= a_t and ka[1] is not None and ka[1] == kb[1]



def dedupe(
    items: list[Item],
    reference: list[Item],
    q_threshold: float = Q_THRESHOLD,
    answer_threshold: float = ANSWER_THRESHOLD,
) -> tuple[list[Item], dict[str, int]]:
    """Section 5.1 property 3 by union-find over all pairs.

    Nodes are the items plus the reference rows (pilot fixtures,
    mega_v2 facts). Every pair that is a near-duplicate is joined; a
    component containing a reference row is removed entirely, every
    other component keeps its item with the lowest src_hash.
    """
    nodes = list(items) + list(reference)
    keys = [_key(n) for n in nodes]  # computed once per row
    parent = list(range(len(nodes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if _near_dup_keys(keys[i], keys[j], q_threshold, answer_threshold):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    comps: dict[int, list[int]] = {}
    for i in range(len(nodes)):
        comps.setdefault(find(i), []).append(i)
    kept: list[Item] = []
    removed_ref = 0
    for members in comps.values():
        own = [m for m in members if m < len(items)]
        if not own:
            continue
        if any(m >= len(items) for m in members):
            removed_ref += len(own)
            continue
        kept.append(min((items[m] for m in own), key=src_hash))
    kept.sort(key=src_hash)
    return kept, {"n_in": len(items), "n_groups": len(kept), "n_removed_reference": removed_ref}


def contamination_drop(
    items: list[Item],
    pre_cutoff: list[Item],
    q_threshold: float = Q_THRESHOLD,
    answer_threshold: float = ANSWER_THRESHOLD,
) -> list[Item]:
    """Drop kept items that near-duplicate a pre-cutoff row of their source."""
    ref = [(p["source"], _key(p)) for p in pre_cutoff]
    out = []
    for it in items:
        k = _key(it)
        if not any(src == it["source"] and _near_dup_keys(k, kp, q_threshold, answer_threshold)
                   for src, kp in ref):
            out.append(it)
    return out


# --- skeletons ---------------------------------------------------------------

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
          r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_DATE_RES = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(_MONTH + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b"),
)
_NUM_RE = re.compile(r"\d(?:[\d,.]*\d)?")
_CAP_RE = re.compile(r"[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*")
_D, _N, _E = "\x00d\x00", "\x00n\x00", "\x00e\x00"


def skeleton(question: str) -> str:
    """Stem skeleton (section 5.1 property 2), dependency-free.

    In order: dates -> <d> (ISO dates; month-name dates with optional
    day suffix and year), numbers -> <n>, lower-case the first
    character, runs of capitalised tokens -> <e>; then NFKC,
    lower-case and whitespace collapse.
    """
    s = unicodedata.normalize("NFKC", question)
    for rx in _DATE_RES:
        s = rx.sub(_D, s)
    s = _NUM_RE.sub(_N, s)
    s = s[:1].lower() + s[1:]
    s = _CAP_RE.sub(_E, s)
    s = s.replace(_D, "<d>").replace(_N, "<n>").replace(_E, "<e>")
    return " ".join(s.lower().split())


def skeleton_cap(items: Iterable[Item], cap: int = SKELETON_CAP) -> list[Item]:
    """Keep at most ``cap`` items per skeleton, the lowest src_hash."""
    groups: dict[str, list[Item]] = {}
    for it in items:
        groups.setdefault(skeleton(it["question"]), []).append(it)
    kept = [x for g in groups.values() for x in sorted(g, key=src_hash)[:cap]]
    return sorted(kept, key=src_hash)


# --- cap, quotas, split ------------------------------------------------------


def _tuning_quota(n: int) -> int:
    return -(-TUNING_TENTHS * n // 10)


def cap_and_split(bank: Mapping[str, list[Item]]) -> dict[str, dict[str, list[Item]]]:
    """Step 6 (cap mmlu_cf at the largest dated source) and step 7 (split)."""
    k = max(len(bank.get(s, [])) for s in DATED)
    parts: dict[str, dict[str, list[Item]]] = {}
    for source, items in bank.items():
        ordered = sorted(items, key=src_hash)
        if source == UNDATED:
            ordered = ordered[:k]
        t = _tuning_quota(len(ordered))
        parts[source] = {"tuning": ordered[:t], "confirmatory": ordered[t:]}
    return parts


def headroom_rebalance(
    parts: Mapping[str, Mapping[str, list[Item]]], correct: Mapping[str, bool]
) -> dict[str, Any]:
    """Step 9: per-source headroom drop, cap re-application, re-check.

    ``correct`` maps the src_hash of every tuning item to the base
    checkpoint's correctness. Outcomes are ACCEPT or REJECT (the corpus
    is then not used before filing); tuning items are never moved.
    """
    acc: dict[str, float] = {}
    for s, p in parts.items():
        try:
            vals = [bool(correct[src_hash(x)]) for x in p["tuning"]]
        except KeyError as exc:
            raise ValueError(f"tuning item of {s} not scored: {exc}") from exc
        acc[s] = sum(vals) / len(vals) if vals else 0.0
    dropped = sorted(s for s, a in acc.items() if a > HEADROOM)
    out: dict[str, Any] = {"accuracy": acc, "dropped": dropped}
    if all(s in dropped for s in DATED):
        return {**out, "status": REJECT, "reason": "both dated sources fail headroom",
                "partitions": None}
    new = {s: {"tuning": list(p["tuning"]), "confirmatory": list(p["confirmatory"])}
           for s, p in parts.items() if s not in dropped}
    if UNDATED in new and any(s in dropped for s in DATED):
        k_new = max(len(new[s]["tuning"]) + len(new[s]["confirmatory"])
                    for s in DATED if s in new)
        cf = new[UNDATED]
        if len(cf["tuning"]) > k_new:
            return {**out, "status": REJECT, "partitions": None,
                    "reason": "mmlu_cf cap cannot be met without moving scored tuning items"}
        excess = len(cf["tuning"]) + len(cf["confirmatory"]) - k_new
        if excess > 0:
            cf["confirmatory"] = sorted(cf["confirmatory"], key=src_hash)[: len(cf["confirmatory"]) - excess]
    tuning = [x for p in new.values() for x in p["tuning"]]
    pooled = sum(bool(correct[src_hash(x)]) for x in tuning) / len(tuning) if tuning else 0.0
    out["pooled_accuracy"] = pooled
    if pooled > HEADROOM:
        return {**out, "status": REJECT, "reason": "pooled headroom", "partitions": None}
    n_conf = sum(len(p["confirmatory"]) for p in new.values())
    n_tune = len(tuning)
    if n_conf < MIN_CONFIRMATORY or n_tune < MIN_TUNING:
        return {**out, "status": REJECT, "partitions": None,
                "reason": f"minima: confirmatory {n_conf}, tuning {n_tune}"}
    weights = {s: len(p["confirmatory"]) / n_conf for s, p in new.items()}
    return {**out, "status": ACCEPT, "reason": "", "partitions": new, "weights": weights}
