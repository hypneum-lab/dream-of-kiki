"""E1 corpus build (task 7): loading, pins, steps 1-8 on synthetic sources."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import e1_corpus_build as build
from scripts import e1_sources as src

FIX = Path(__file__).resolve().parents[2] / "fixtures"


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _rtqa(q: str, date: str, ans: Any = None, choices: list[str] | None = None) -> str:
    return json.dumps({"question_id": "x", "question_date": date, "question_source": "S",
                       "question_url": "u", "question_sentence": q,
                       "choices": choices if choices is not None else ["A1", "B1", "C1", "D1"],
                       "answer": ans if ans is not None else ["1"], "evidence": ""})


@pytest.fixture()
def sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "src"
    rt = d / "realtimeqa" / "past" / "2026"
    rt.mkdir(parents=True)
    (rt / "20260301_qa.jsonl").write_text("\n".join([
        _rtqa("Which city hosted the old spring flower show this year?", "2026/03/01"),
        _rtqa("Old malformed row about a famous bridge collapse event", "2026/03/01",
              choices=[]),
    ]) + "\n", encoding="utf-8")
    (rt / "20260501_qa.jsonl").write_text("\n".join([
        _rtqa("Which city hosted the old spring flower show this year?!", "2026/05/01"),
        "",
        _rtqa("Old malformed row about a famous bridge collapse event again", "2026/05/01"),
        _rtqa("What is the name of the new national park opened on Monday?", "2026/05/01",
              ans=["2"], choices=["Alpha Park", "Beta Park", "Gamma Park", "Delta Park"]),
        _rtqa("Who won the regional chess title held in the capital last week?", "2026/05/01",
              ans=["0", "1"]),
    ]) + "\n", encoding="utf-8")
    (rt / "20260501_qa_nota.jsonl").write_text(_rtqa("ignored", "2026/05/01") + "\n")
    _write_csv(d / src.DAILY_ORACLE_FILE,
               ["question", "choice_a", "choice_b", "choice_c", "choice_d", "answer", "date"],
               [["Will the central bank raise rates in its June meeting?", "Yes, by 25bp ",
                 "No change", "Cut", "Yes, by 50bp ", "a", "2026-05-02"],
                ["Will an older event happen before the spring?", "a1", "b1", "c1", "d1",
                 "b", "2020-01-01"],
                ["What will the mayor announce about the downtown tram line?", "Extension",
                 "Closure", "Delay", "New fares", "z", "2026-06-01"]])
    _write_csv(d / src.MMLU_CF_FILE, ["Question", "A", "B", "C", "D", "Answer"],
               [["Which gas do plants absorb during photosynthesis?", " oxygen", " carbon dioxide",
                 " nitrogen", " helium", "B"],
                ["Which planet is known as the red one in our solar system?", "Mars", "Venus",
                 "Earth", src.NOTA, "A"],
                ["What is the boiling point of water at sea level in Celsius?", "90", "100",
                 "110", "120", "B"]])
    for name, rel in (("DAILY_ORACLE", src.DAILY_ORACLE_FILE), ("MMLU_CF", src.MMLU_CF_FILE)):
        p = d / rel
        monkeypatch.setattr(src, f"{name}_SHA256", src.sha256_file(p))
        monkeypatch.setattr(src, f"{name}_SIZE", p.stat().st_size)
    return d


def _blobs(root: Path, commit: str) -> dict[str, str]:
    assert commit == src.REALTIMEQA_COMMIT
    return {f.relative_to(root).as_posix(): src.git_blob_sha1(f)
            for f in sorted(root.glob(src.REALTIMEQA_GLOB))}


MEGA = [{"id": "m1", "context": "The capital city of Japan is", "expected": " Tokyo."}]


def test_loaders_map_rows(sources: Path) -> None:
    rt = src.load_realtimeqa(sources / "realtimeqa")
    ids = [r["source_id"] for r in rt]
    # the _qa_nota file is not matched; blank line keeps its physical index
    assert ids == ["past/2026/20260301_qa.jsonl:0", "past/2026/20260301_qa.jsonl:1",
                   "past/2026/20260501_qa.jsonl:0", "past/2026/20260501_qa.jsonl:2",
                   "past/2026/20260501_qa.jsonl:3", "past/2026/20260501_qa.jsonl:4"]
    assert rt[0]["date"] == "2026-03-01" and rt[0]["answer"] == 1
    assert rt[5]["answer"] is None  # two gold indices -> no gold
    do = src.load_daily_oracle(sources / src.DAILY_ORACLE_FILE)
    assert [r["answer"] for r in do] == [0, 1, None] and do[0]["source_id"] == "0"
    cf = src.load_mmlu_cf(sources / src.MMLU_CF_FILE)
    assert cf[0]["answer"] == 1 and cf[0]["date"] is None and cf[2]["source_id"] == "val.csv:2"


def test_normalise_strips_whitespace_only() -> None:
    it = {"source": "daily_oracle", "source_id": "0", "date": "2026-05-02",
          "question": "  Q? ", "choices": ["a ", " b", "c", "d  "], "answer": 0}
    n = src.normalise(it)
    assert n["question"] == "Q?" and n["choices"] == ["a", "b", "c", "d"]


def test_verify_pins_accepts_and_rejects(sources: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = src.verify_pins(sources, _blobs)
    assert set(rec["realtimeqa"]["files"]) == {"past/2026/20260301_qa.jsonl",
                                               "past/2026/20260501_qa.jsonl"}
    monkeypatch.setattr(src, "MMLU_CF_SHA256", "0" * 64)
    with pytest.raises(src.PinError):
        src.verify_pins(sources, _blobs)


def test_verify_pins_rejects_realtimeqa_drift(sources: Path) -> None:
    def wrong(root: Path, commit: str) -> dict[str, str]:
        b = _blobs(root, commit)
        b["past/2026/20260501_qa.jsonl"] = "f" * 40
        return b

    with pytest.raises(src.PinError, match="differs"):
        src.verify_pins(sources, wrong)

    def extra(root: Path, commit: str) -> dict[str, str]:
        return {**_blobs(root, commit), "past/2026/20260601_qa.jsonl": "e" * 40}

    with pytest.raises(src.PinError, match="file set"):
        src.verify_pins(sources, extra)


def test_git_blob_sha1_matches_git(tmp_path: Path) -> None:
    p = tmp_path / "f"
    p.write_bytes(b"hello\n")
    assert src.git_blob_sha1(p) == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_byte_bound_counter() -> None:
    assert src.byte_bound_counter("a" * 200, "b" * 55) == 0  # 200 + 1 + 55 = 256
    assert src.byte_bound_counter("a" * 200, "b" * 56) is None
    assert src.byte_bound_counter("é" * 128, "") is None  # 256 bytes + 1


@pytest.mark.parametrize(("change", "reason"), [
    ({"choices": ["a", "b", "c"]}, "not_4_choices"),
    ({"question": ""}, "empty_field"),
    ({"choices": ["a", "", "c", "d"]}, "empty_field"),
    ({"date": None}, "empty_field"),
    ({"answer": None}, "invalid_gold"),
    ({"answer": 4}, "invalid_gold"),
    ({"choices": ["a", "b", "c", src.NOTA]}, "nota_option"),
    ({"question": "x" * 300}, "undecided_tokens"),
])
def test_exclusion_reasons(change: dict[str, Any], reason: str) -> None:
    it = {"source": "realtimeqa", "source_id": "p:0", "date": "2026-05-01",
          "question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 0, **change}
    assert src.exclusion_reason(it, src.byte_bound_counter) == reason


def test_exclusion_token_counter_over_limit() -> None:
    it = {"source": "mmlu_cf", "source_id": "val.csv:0", "date": None,
          "question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 0}
    assert src.exclusion_reason(it, lambda q, g: 257) == "over_256_tokens"
    assert src.exclusion_reason(it, lambda q, g: 256) is None


def test_build_end_to_end(sources: Path, tmp_path: Path) -> None:
    corpus, man_text = build.run(sources, src.byte_bound_counter, "byte-bound", False,
                                 tree_blobs=_blobs, fixtures_dir=FIX, mega=MEGA)
    man = json.loads(man_text)
    rep = man["report"]
    # malformed pre-cutoff row (choices = []) is a reference, and its
    # post-cutoff near-duplicate is dropped (question rule) without a crash
    assert rep["pre_cutoff_reference_rows"] == {"realtimeqa": 2, "daily_oracle": 1, "mmlu_cf": 0}
    kept_ids = {i["source_id"] for i in man["items"]}
    assert "past/2026/20260501_qa.jsonl:0" not in kept_ids  # recurring quiz question
    assert "past/2026/20260501_qa.jsonl:2" not in kept_ids  # dup of malformed pre row
    assert "past/2026/20260501_qa.jsonl:3" in kept_ids
    assert "past/2026/20260501_qa.jsonl:4" not in kept_ids  # two golds
    assert rep["exclusions"]["realtimeqa"] == {"invalid_gold": 1, "not_4_choices": 1}
    assert rep["exclusions"]["mmlu_cf"] == {"nota_option": 1}
    assert rep["exclusions"]["daily_oracle"] == {"invalid_gold": 1}
    assert man["status"] == build.PENDING and not rep["undecided_tokens"]
    # one frozen item per line; its SHA-256 is item_sha256
    lines = corpus.splitlines()
    assert [hashlib.sha256(ln.encode()).hexdigest() for ln in lines] == [
        i["item_sha256"] for i in man["items"]]
    assert man["corpus_sha256"] == hashlib.sha256(corpus.encode()).hexdigest()
    for label, part in man["partitions"].items():
        hs = sorted(i["item_sha256"] for i in man["items"] if i["partition"] == label)
        assert part["sha256"] == hashlib.sha256("\n".join(hs).encode()).hexdigest()
    first = json.loads(lines[0])
    assert "partition" not in first and first["prompt"]
    dated = [json.loads(ln) for ln in lines if json.loads(ln)["date"]]
    assert all(d["prompt"].startswith(f"Date: {d['date']}\n") for d in dated)
    # no item text in the manifest (rule 9)
    assert "national park" not in man_text
    # deterministic
    assert build.run(sources, src.byte_bound_counter, "byte-bound", False,
                     tree_blobs=_blobs, fixtures_dir=FIX, mega=MEGA) == (corpus, man_text)


def test_reference_rows_remove_component(sources: Path) -> None:
    mega = [{"id": "m", "context": "What is the name of the new national park opened on Monday?",
             "expected": "Gamma Park"}]
    _, man_text = build.run(sources, src.byte_bound_counter, "b", False,
                            tree_blobs=_blobs, fixtures_dir=FIX, mega=mega)
    man = json.loads(man_text)
    assert "past/2026/20260501_qa.jsonl:3" not in {i["source_id"] for i in man["items"]}
    assert man["report"]["dedupe"]["n_removed_reference"] == 1


def test_undecided_items_refuse_or_flag(sources: Path) -> None:
    def count(q: str, g: str) -> int | None:
        return None if "boiling" in q else 0

    with pytest.raises(SystemExit, match="undecided"):
        build.run(sources, count, "c", False, tree_blobs=_blobs, fixtures_dir=FIX, mega=MEGA)
    _, man_text = build.run(sources, count, "c", True, tree_blobs=_blobs, fixtures_dir=FIX,
                            mega=MEGA)
    man = json.loads(man_text)
    assert man["status"] == build.PROVISIONAL
    assert [d["source_id"] for d in man["report"]["undecided_tokens"]] == ["val.csv:2"]
    flagged = {i["source_id"]: i["undecided_tokens"] for i in man["items"]}
    assert all(v == (k == "val.csv:2") for k, v in flagged.items())


def test_cli_write_refuses_overwrite_and_check(sources: Path, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(src, "git_tree_blobs", _blobs)
    monkeypatch.setattr(build, "mega_v2_facts", lambda: MEGA)
    real_run = build.run

    def run(*a: Any, **k: Any) -> tuple[str, str]:
        k["tree_blobs"] = _blobs
        return real_run(*a, **k)

    monkeypatch.setattr(build, "run", run)
    out = tmp_path / "out"
    assert build.main(["--sources", str(sources), "--out", str(out)]) == 0
    assert build.main(["--sources", str(sources), "--out", str(out), "--check"]) == 0
    with pytest.raises(SystemExit, match="overwrite"):
        build.main(["--sources", str(sources), "--out", str(out)])
    (out / build.CORPUS_FILE).write_text("tampered\n", encoding="utf-8")
    assert build.main(["--sources", str(sources), "--out", str(out), "--check"]) == 1


def test_freeze_rejects_below_minima_and_foreign_items(sources: Path) -> None:
    _, man_text = build.run(sources, src.byte_bound_counter, "b", False,
                            tree_blobs=_blobs, fixtures_dir=FIX, mega=MEGA)
    man = json.loads(man_text)
    tuning = {i["src_hash"]: False for i in man["items"] if i["partition"] == "tuning"}
    corpus, frozen_text = build.run(sources, src.byte_bound_counter, "b", False,
                                    tree_blobs=_blobs, fixtures_dir=FIX, mega=MEGA,
                                    headroom=tuning, headroom_sha256="x")
    frozen = json.loads(frozen_text)
    assert frozen["status"] == build.REJECTED and frozen["items"] == [] and corpus == ""
    assert frozen["headroom"]["reason"].startswith("minima")
    conf = next(i["src_hash"] for i in man["items"] if i["partition"] == "confirmatory") \
        if any(i["partition"] == "confirmatory" for i in man["items"]) else "f" * 64
    with pytest.raises(SystemExit, match="non-TUNING"):
        build.run(sources, src.byte_bound_counter, "b", False, tree_blobs=_blobs,
                  fixtures_dir=FIX, mega=MEGA, headroom={**tuning, conf: True})
    with pytest.raises(ValueError, match="not scored"):
        build.run(sources, src.byte_bound_counter, "b", False, tree_blobs=_blobs,
                  fixtures_dir=FIX, mega=MEGA, headroom={})


def test_freeze_refuses_provisional(sources: Path) -> None:
    with pytest.raises(SystemExit, match="PROVISIONAL"):
        build.run(sources, lambda q, g: None, "c", True, tree_blobs=_blobs,
                  fixtures_dir=FIX, mega=MEGA, headroom={})
