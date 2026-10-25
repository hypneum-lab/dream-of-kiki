"""Study selection of the E1 harness (HL_STUDY): 35B E1 vs 4B E1-small."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from scripts import e1_harness as hn
from scripts import e1_sources as src

SMALL = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")


def _reload(monkeypatch: pytest.MonkeyPatch, study: str | None) -> Any:
    if study is None:
        monkeypatch.delenv("HL_STUDY", raising=False)
    else:
        monkeypatch.setenv("HL_STUDY", study)
    return importlib.reload(hn)


@pytest.fixture(autouse=True)
def _restore_module() -> Iterator[None]:
    yield
    importlib.reload(hn)  # after monkeypatch has restored the environment


@pytest.fixture
def small(monkeypatch: pytest.MonkeyPatch) -> Any:
    return _reload(monkeypatch, "e1-small")


def _cell(model: str, revision: str) -> dict[str, Any]:
    return {"mode": "tuning", "model": model, "revision": revision, "run_identity": "r",
            "corpus": {"corpus_sha256": "c" * 64, "partition": "tuning"}}


def test_default_study_is_the_35b_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    h = _reload(monkeypatch, None)
    assert h.STUDY == "e1"
    assert (h.MODEL_REPO, h.MODEL_REVISION) == src.STUDY_MODELS["e1"]
    assert src.STUDY_MODELS["e1"] == (src.MODEL_REPO, src.MODEL_REVISION)
    assert src.STUDY_MODELS["e1-small"] == SMALL


def test_small_study_pins_the_4b(small: Any, tmp_path: Path) -> None:
    assert (small.MODEL_REPO, small.MODEL_REVISION) == SMALL
    small.check_provenance([_cell(*SMALL)], "tuning", "c" * 64)
    with pytest.raises(SystemExit, match="does not come"):
        small.check_provenance([_cell(*src.STUDY_MODELS["e1"])], "tuning", "c" * 64)
    ap = argparse.ArgumentParser()
    small.add_args(ap)
    for extra in ([], ["--model", src.MODEL_REPO, "--revision", src.MODEL_REVISION],
                  ["--revision", "main"]):
        args = ap.parse_args(["--mode", "tuning", "--out", str(tmp_path), *extra])
        if extra:
            with pytest.raises(SystemExit, match="runs only on Qwen/Qwen3.5-4B@851bf6e"):
                small.run(args, lambda *a: None)
        else:
            assert (args.model, args.revision) == SMALL
            with pytest.raises(SystemExit, match="needs --corpus"):
                small.run(args, lambda *a: None)


def test_unknown_study_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="HL_STUDY"):
        _reload(monkeypatch, "e1-tiny")
