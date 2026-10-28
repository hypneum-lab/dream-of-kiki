"""E1 corpus construction core (section 5.1b): synthetic items only."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from scripts.e1_corpus_core import (
    CUTOFF,
    REJECT,
    canonical_json,
    cap_and_split,
    contamination_drop,
    dedupe,
    final_item,
    grams,
    headroom_rebalance,
    jaccard,
    norm_key,
    skeleton,
    skeleton_cap,
    src_hash,
)


def _item(source: str, sid: str, q: str, gold: str = "Paris", date: str | None = "2026-05-01",
          choices: list[str] | None = None) -> dict[str, Any]:
    ch = choices or [gold, "Lyon", "Nice", "Lille"]
    return {"source": source, "source_id": sid, "date": date, "question": q,
            "choices": ch, "answer": ch.index(gold)}


# --- hashes ------------------------------------------------------------------


def test_src_hash_is_canonical_and_pre_permutation() -> None:
    it = _item("realtimeqa", "past/2026/20260501_qa.jsonl:3", "Which city hosted it?")
    want = hashlib.sha256(canonical_json(it).encode("utf-8")).hexdigest()
    assert src_hash(it) == want
    assert canonical_json(it) == json.dumps(it, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=False)
    fr = final_item(it)
    # the permutation is seeded from src_hash, and the final hash covers it
    assert fr["src_hash"] == src_hash(it)
    assert sorted(fr["choices"]) == sorted(it["choices"])
    assert fr["choices"][fr["answer"]] == "Paris"
    assert fr["prompt"].startswith("Date: 2026-05-01\n")
    assert fr["item_sha256"] == hashlib.sha256(
        canonical_json({k: v for k, v in fr.items() if k != "item_sha256"}).encode()
    ).hexdigest()
    assert final_item(it) == fr  # deterministic


def test_undated_item_has_no_prefix() -> None:
    fr = final_item(_item("mmlu_cf", "val.csv:7", "What is 2+2?", gold="4",
                          date=None, choices=["4", "3", "5", "6"]))
    assert fr["prompt"] == "What is 2+2?"


# --- near-duplicate grouping -------------------------------------------------


def test_norm_key_and_short_string_grams() -> None:
    assert norm_key("  Ｐaris\tIS  big ") == "paris is big"
    assert grams("abc") == {"abc"}  # shorter than 5: the whole string
    assert grams("") == set()
    assert jaccard(set(), set()) == 0.0


def test_dedupe_uses_union_find_and_min_src_hash() -> None:
    a = _item("daily_oracle", "1", "Who won the 2026 Tour de France cycling race?")
    b = _item("daily_oracle", "2", "Who won the 2026 Tour de France cycling race ?")
    c = _item("daily_oracle", "3", "Who won the 2026 Tour de France cycling race??")
    d = _item("daily_oracle", "4", "Completely unrelated question about volcanoes", gold="Etna",
              choices=["Etna", "Fuji", "Hekla", "Teide"])
    kept, log = dedupe([a, b, c, d], reference=[])
    ids = sorted(x["source_id"] for x in kept)
    rep = min((a, b, c), key=src_hash)["source_id"]
    assert ids == sorted([rep, "4"])
    assert log["n_groups"] == 2


def test_dedupe_same_gold_lower_threshold() -> None:
    a = _item("daily_oracle", "1", "Which city will host the summit in June?")
    b = _item("daily_oracle", "2", "Where is the June summit going to be held?")
    assert jaccard(grams(norm_key(a["question"])), grams(norm_key(b["question"]))) < 0.6
    kept, _ = dedupe([a, b], reference=[], answer_threshold=0.0)
    assert len(kept) == 1  # identical gold "Paris" + Jaccard >= threshold


def test_dedupe_removes_whole_component_touching_reference() -> None:
    a = _item("mmlu_cf", "val.csv:1", "What is the capital of France?", date=None)
    ref = _item("pilot", "mmlu_real.jsonl:0", "What is the capital of France ?", date=None)
    kept, log = dedupe([a], reference=[ref])
    assert kept == [] and log["n_removed_reference"] == 1


# --- skeleton ----------------------------------------------------------------


def test_skeleton_masks_dates_numbers_and_capitalised_runs() -> None:
    q = "What did Emmanuel Macron announce on May 3, 2026 about 1,200 New Jobs?"
    assert skeleton(q) == "what did <e> announce on <d> about <n> <e>?"
    assert skeleton("Which country won?") == "which country won?"
    assert skeleton("On 2026-05-01 who led?") == "on <d> who led?"


def test_skeleton_cap_keeps_five_lowest_src_hash() -> None:
    items = [_item("daily_oracle", str(i), f"What will Company{chr(65 + i)} report in Q{i}?")
             for i in range(8)]
    kept = skeleton_cap(items, cap=5)
    assert len(kept) == 5
    assert {src_hash(x) for x in kept} == set(sorted(src_hash(x) for x in items)[:5])


# --- date-based contamination ------------------------------------------------


def test_post_cutoff_duplicate_of_pre_cutoff_row_is_dropped() -> None:
    old = _item("realtimeqa", "past/2025/20250704_qa.jsonl:0",
                "Which document was adopted on July 4?", date="2025-07-04")
    new = _item("realtimeqa", "past/2026/20260704_qa.jsonl:0",
                "Which document was adopted on July 4 ?", date="2026-07-04")
    other = _item("realtimeqa", "past/2026/20260705_qa.jsonl:0",
                  "Which team won the cup final this week?", date="2026-07-05")
    kept = contamination_drop([new, other], pre_cutoff=[old])
    assert [x["source_id"] for x in kept] == [other["source_id"]]
    assert CUTOFF == "2026-04-16"


# --- cap, quotas, split ------------------------------------------------------


def _bank(n_rt: int, n_do: int, n_cf: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "realtimeqa": [_item("realtimeqa", f"past/2026/x.jsonl:{i}", f"rt {i}") for i in range(n_rt)],
        "daily_oracle": [_item("daily_oracle", str(i), f"do {i}") for i in range(n_do)],
        "mmlu_cf": [_item("mmlu_cf", f"val.csv:{i}", f"cf {i}", date=None) for i in range(n_cf)],
    }


def test_cap_and_split_quotas_and_order() -> None:
    parts = cap_and_split(_bank(545, 492, 5400))
    assert len(parts["mmlu_cf"]["tuning"]) + len(parts["mmlu_cf"]["confirmatory"]) == 545
    for s, n in (("realtimeqa", 545), ("daily_oracle", 492), ("mmlu_cf", 545)):
        t = parts[s]["tuning"]
        c = parts[s]["confirmatory"]
        assert len(t) == -(-3 * n // 10)  # ceil(0.30 n)
        assert len(t) + len(c) == n
        assert max(src_hash(x) for x in t) < min(src_hash(x) for x in c)
    bank = _bank(545, 492, 5400)
    kept_cf = sorted(bank["mmlu_cf"], key=src_hash)[:545]
    got = parts["mmlu_cf"]["tuning"] + parts["mmlu_cf"]["confirmatory"]
    assert {src_hash(x) for x in got} == {src_hash(x) for x in kept_cf}


# --- headroom drop and rebalancing -------------------------------------------


def _correct(parts: dict[str, Any], acc: dict[str, float]) -> dict[str, bool]:
    out = {}
    for s, p in parts.items():
        t = sorted(p["tuning"], key=src_hash)
        k = round(acc[s] * len(t))
        for i, x in enumerate(t):
            out[src_hash(x)] = i < k
    return out


def test_no_drop_when_all_sources_have_headroom() -> None:
    parts = cap_and_split(_bank(545, 492, 5400))
    res = headroom_rebalance(parts, _correct(parts, {"realtimeqa": 0.5, "daily_oracle": 0.6,
                                                     "mmlu_cf": 0.7}))
    assert res["status"] == "ACCEPT" and res["dropped"] == []


def test_mmlu_cf_drop_only_removes_it() -> None:
    parts = cap_and_split(_bank(545, 492, 5400))
    res = headroom_rebalance(parts, _correct(parts, {"realtimeqa": 0.5, "daily_oracle": 0.6,
                                                     "mmlu_cf": 0.9}))
    assert res["status"] == "ACCEPT" and res["dropped"] == ["mmlu_cf"]
    assert "mmlu_cf" not in res["partitions"]


def test_dated_drop_recaps_mmlu_cf_from_confirmatory_only() -> None:
    parts = cap_and_split(_bank(545, 492, 5400))
    tuning_before = {src_hash(x) for x in parts["mmlu_cf"]["tuning"]}
    res = headroom_rebalance(parts, _correct(parts, {"realtimeqa": 0.9, "daily_oracle": 0.6,
                                                     "mmlu_cf": 0.7}))
    assert res["status"] == "ACCEPT" and res["dropped"] == ["realtimeqa"]
    cf = res["partitions"]["mmlu_cf"]
    assert len(cf["tuning"]) + len(cf["confirmatory"]) == 492
    assert {src_hash(x) for x in cf["tuning"]} == tuning_before  # never moved


def test_impossible_cap_is_rejected() -> None:
    # k would fall to 100, but 164 mmlu_cf items are already scored in tuning
    parts = cap_and_split(_bank(545, 100, 5400))
    res = headroom_rebalance(parts, _correct(parts, {"realtimeqa": 0.9, "daily_oracle": 0.6,
                                                     "mmlu_cf": 0.7}))
    assert res["status"] == REJECT and "cap" in res["reason"]


def test_both_dated_sources_failing_is_rejected() -> None:
    parts = cap_and_split(_bank(545, 492, 5400))
    res = headroom_rebalance(parts, _correct(parts, {"realtimeqa": 0.9, "daily_oracle": 0.85,
                                                     "mmlu_cf": 0.7}))
    assert res["status"] == REJECT and "both dated" in res["reason"]


def test_minima_failure_is_rejected() -> None:
    parts = cap_and_split(_bank(150, 120, 5400))
    res = headroom_rebalance(parts, _correct(parts, {"realtimeqa": 0.5, "daily_oracle": 0.6,
                                                     "mmlu_cf": 0.7}))
    assert res["status"] == REJECT and "minima" in res["reason"]


def test_pooled_recheck_after_drop() -> None:
    parts = cap_and_split(_bank(545, 492, 5400))
    res = headroom_rebalance(parts, _correct(parts, {"realtimeqa": 0.79, "daily_oracle": 0.79,
                                                     "mmlu_cf": 0.95}))
    assert res["status"] == "ACCEPT"  # pooled after the mmlu_cf drop is 0.79
    parts = cap_and_split(_bank(545, 492, 5400))
    with pytest.raises(ValueError):
        headroom_rebalance(parts, {})  # every tuning item must be scored


# --- review final pass: malformed reference rows -----------------------------


@pytest.mark.parametrize(
    ("choices", "answer"),
    [([], 0), (["a", "b", "c", "d"], 7), (["a", "b", "c", "d"], -1), (["a", "b", "c", "d"], "x")],
)
def test_malformed_reference_row_matches_by_question_only(choices: list[str], answer: Any) -> None:
    new = _item("realtimeqa", "past/2026/a.jsonl:0", "Which document was adopted on July 4?",
                date="2026-07-04")
    bad_dup = {"source": "realtimeqa", "source_id": "past/2025/b.jsonl:0", "date": "2025-07-04",
               "question": "Which document was adopted on July 4 ?", "choices": choices,
               "answer": answer}
    bad_other = dict(bad_dup, question="Which team won the cup final this week?")
    # question Jaccard >= 0.6: still a duplicate despite the invalid gold
    assert contamination_drop([new], pre_cutoff=[bad_dup]) == []
    # below 0.6 an invalid gold can never trigger the equal-gold rule
    assert contamination_drop([new], pre_cutoff=[bad_other]) == [new]
    kept, _ = dedupe([new], reference=[bad_other], answer_threshold=0.0)
    assert kept == [new]
