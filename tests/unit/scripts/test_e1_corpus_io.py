"""E1 corpus loader (task 2): hash checks, status rules, partition isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import e1_corpus_build as build
from scripts import e1_corpus_io as cio

FIX = Path(__file__).resolve().parents[2] / "fixtures"


def _src(source: str, i: int, date: str | None) -> dict[str, Any]:
    return {"source": source, "source_id": f"{source}:{i}", "date": date,
            "question": f"Synthetic question {source} number {i}?",
            "choices": [f"opt{j}-{i}" for j in range(4)], "answer": i % 4}


def _write(tmp: Path, status_label: str | None) -> tuple[Path, Path]:
    parts = {
        "realtimeqa": {"tuning": [_src("realtimeqa", 0, "2026-05-01")],
                       "confirmatory": [_src("realtimeqa", 1, "2026-05-02"),
                                        _src("realtimeqa", 2, "2026-05-03")]},
        "mmlu_cf": {"tuning": [_src("mmlu_cf", 3, None)],
                    "confirmatory": [_src("mmlu_cf", 4, None)]},
    }
    report: dict[str, Any] = {"undecided_tokens": []}
    frozen = build.freeze_items(parts, report)
    corpus = build.corpus_text(frozen)
    hr = None if status_label is None else {"status_label": status_label, "status": "ACCEPT",
                                            "reason": ""}
    man = build.manifest(frozen, report, {}, corpus, "test", {}, hr, tokenizer_pinned=True)
    c, m = tmp / "c.jsonl", tmp / "m.json"
    c.write_text(corpus, encoding="utf-8")
    m.write_text(json.dumps(man), encoding="utf-8")
    return c, m


def test_load_confirmatory_and_tuning(tmp_path: Path) -> None:
    c, m = _write(tmp_path, build.FROZEN)
    conf, prov = cio.load_corpus(c, m, "confirmatory")
    tune, _ = cio.load_corpus(c, m, "tuning")
    assert len(conf) == 3 and len(tune) == 2 and prov["partition"] == "confirmatory"
    assert not {x["item_sha256"] for x in conf} & {x["item_sha256"] for x in tune}
    it = conf[0]
    assert it["prompt"].endswith("\nAnswer:") and all(ch.startswith(" ") for ch in it["choices"])
    assert it["question_prompt"] == it["prompt"]
    dated = [x for x in conf if x["source"] == "realtimeqa"]
    assert all(x["prompt"].startswith("Date: 2026-05-0") for x in dated)


@pytest.mark.parametrize("mode", ["tuning", "calibrate", "headroom"])
def test_non_confirmatory_modes_never_read_confirmatory(tmp_path: Path, mode: str) -> None:
    c, m = _write(tmp_path, build.FROZEN if mode != "headroom" else None)
    items, prov = cio.load_corpus(c, m, mode)
    assert prov["partition"] == "tuning"
    man = json.loads(m.read_text())
    tuning = {e["item_sha256"] for e in man["items"] if e["partition"] == "tuning"}
    assert {x["item_sha256"] for x in items} == tuning


def test_status_rules(tmp_path: Path) -> None:
    c, m = _write(tmp_path, None)  # PENDING-HEADROOM
    with pytest.raises(cio.CorpusError, match="FROZEN"):
        cio.load_corpus(c, m, "confirmatory")
    with pytest.raises(cio.CorpusError, match="FROZEN"):
        cio.load_corpus(c, m, "tuning")
    (tmp_path / "b").mkdir()
    c2, m2 = _write(tmp_path / "b", build.FROZEN)
    with pytest.raises(cio.CorpusError, match="PENDING"):
        cio.load_corpus(c2, m2, "headroom")
    with pytest.raises(cio.CorpusError, match="unknown"):
        cio.load_corpus(c2, m2, "smoke")


def test_tamper_detection(tmp_path: Path) -> None:
    c, m = _write(tmp_path, build.FROZEN)
    text = c.read_text()
    c.write_text(text.replace("opt0-1", "optX-1"), encoding="utf-8")
    with pytest.raises(cio.CorpusError, match="corpus file"):
        cio.load_corpus(c, m, "confirmatory")
    c, m = _write(tmp_path, build.FROZEN)
    man = json.loads(m.read_text())
    lines = c.read_text().splitlines()
    lines[0] = lines[0].replace("opt", "OPT")
    c.write_text("\n".join(lines) + "\n", encoding="utf-8")
    man["corpus_sha256"] = cio._sha(c.read_bytes())
    m.write_text(json.dumps(man))
    with pytest.raises(cio.CorpusError, match="item hash"):
        cio.load_corpus(c, m, "confirmatory")
    c, m = _write(tmp_path, build.FROZEN)
    man = json.loads(m.read_text())
    tune = [e for e in man["items"] if e["partition"] == "tuning"]
    tune[0]["partition"] = "confirmatory"  # move a tuning item
    m.write_text(json.dumps(man))
    with pytest.raises(cio.CorpusError, match="partition hash"):
        cio.load_corpus(c, m, "confirmatory")


def test_load_pilot_matches_partition_manifest() -> None:
    man = json.loads((FIX / "e1_partition_2026-10.json").read_text())
    for part in cio.PARTITIONS:
        benches, prov = cio.load_pilot(FIX, FIX / "e1_partition_2026-10.json", part)
        for name, items in benches.items():
            assert len(items) == man["benchmarks"][name]["partitions"][part]["n_rows"]
            assert all(0 <= x["answer"] < len(x["choices"]) for x in items)
    with pytest.raises(cio.CorpusError):
        cio.load_pilot(FIX, FIX / "e1_partition_2026-10.json", "all")
