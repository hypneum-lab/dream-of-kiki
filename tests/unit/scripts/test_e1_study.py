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


DET = {"enabled": True, "use_deterministic_algorithms": True, "warn_only": False,
       "cublas_workspace_config": ":4096:8", "cudnn_deterministic": True,
       "cudnn_benchmark": False, "seed_per_cell": 0}
REG: dict[str, Any] = {"tokenizer": "raw", "linear_attn": "torch", "snapshot": "ram",
                       "sgd": "optim", "determinism": DET}


def _cell(model: str, revision: str, backend: str = "torch", device: str = "cuda",
          placement: tuple[str, ...] = ("cuda:0",), **over: Any) -> dict[str, Any]:
    return {"mode": "tuning", "model": model, "revision": revision, "run_identity": "r",
            "corpus": {"corpus_sha256": "c" * 64, "partition": "tuning"},
            "backend": backend, "study": "e1-small",
            "versions": {"device": device, "placement": list(placement), **REG, **over}}


class _FakeBackend:
    def __init__(self, name: str, device: str, placement: tuple[str, ...],
                 **over: Any) -> None:
        self.name, self.device, self.placement, self.over = name, device, placement, over

    def snapshot(self) -> None:
        pass

    def versions(self) -> dict[str, Any]:
        return {"device": self.device, "placement": list(self.placement), **REG, **self.over}


Instrument = tuple[str, str, tuple[str, ...], dict[str, Any]]
BAD_INSTRUMENTS: tuple[Instrument, ...] = (
                   ("torch", "cpu", ("cpu",), {}), ("mlx", "Device(gpu, 0)", (), {}),
                   ("torch", "cuda", ("cpu",), {}), ("torch", "cuda", ("cpu", "cuda:0"), {}),
                   ("torch", "cuda", ("cuda:0", "cuda:1"), {}),
                   ("torch", "cuda", ("cuda:0",), {"tokenizer": "transformers"}),
                   ("torch", "cuda", ("cuda:0",), {"linear_attn": "fla+causal_conv1d"}),
                   ("torch", "cuda", ("cuda:0",), {"linear_attn": "mixed"}),
                   ("torch", "cuda", ("cuda:0",), {"sgd": "fused"}),
                   ("torch", "cuda", ("cuda:0",), {"snapshot": "disk"}),
                   ("torch", "cuda", ("cuda:0",), {"determinism": {**DET, "enabled": False}}),
                   ("torch", "cuda", ("cuda:0",), {"determinism": {**DET, "warn_only": True}}),
                   ("torch", "cuda", ("cuda:0",), {"determinism": {}}))


@pytest.mark.parametrize(("backend", "device", "placement", "over"), BAD_INSTRUMENTS)
def test_small_study_enforces_the_registered_instrument(
        small: Any, tmp_path: Path, backend: str, device: str, placement: tuple[str, ...],
        over: dict[str, Any]) -> None:
    with pytest.raises(SystemExit, match="does not come"):
        small.check_provenance([_cell(*SMALL, backend, device, placement, **over)], "tuning",
                               "c" * 64)
    with pytest.raises(SystemExit, match="runs only on"):
        small.run(argparse.Namespace(
            mode="smoke", model=SMALL[0], revision=SMALL[1], out=tmp_path,
            pilot_partition="confirmatory", seeds=None),
            lambda *a: _FakeBackend(backend, device, placement, **over))


def test_small_study_tokenizes_raw(small: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    assert small.TOKENIZER == "raw"
    assert _reload(monkeypatch, None).TOKENIZER == "transformers"


def test_small_study_cells_must_carry_their_study(small: Any) -> None:
    for change in ({"backend": None}, {"study": None}, {"study": "e1"}):
        with pytest.raises(SystemExit, match="does not come"):
            small.check_provenance([{**_cell(*SMALL), **change}], "tuning", "c" * 64)


def test_frozen_manifest_must_carry_the_study_headroom(
        small: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def man(model: str | None, revision: str | None) -> dict[str, Any]:
        hr = {} if model is None else {"model": model, "revision": revision}
        return {"status": "FROZEN", "headroom": hr}

    small.check_manifest_study(man(*SMALL))
    for bad in (man(None, None), man(*src.STUDY_MODELS["e1"])):
        with pytest.raises(SystemExit, match="not frozen with the e1-small"):
            small.check_manifest_study(bad)
    big = _reload(monkeypatch, None)
    big.check_manifest_study(man(None, None))  # legacy manifest, default study
    big.check_manifest_study(man(*src.STUDY_MODELS["e1"]))
    with pytest.raises(SystemExit, match="not frozen with the e1 headroom"):
        big.check_manifest_study(man(*SMALL))


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


def test_small_study_mps_instrument_is_smoke_only(small: Any) -> None:
    mps = {"device": "mps", "placement": ["mps:0"], "tokenizer": "raw", "linear_attn": "torch",
           "snapshot": "disk", "sgd": "fused"}
    assert small._study_backend_ok("torch-mps", mps, "e1-small", "smoke")
    for mode in ("headroom", "tuning", "calibrate", "confirmatory"):
        assert not small._study_backend_ok("torch-mps", mps, "e1-small", mode)
    assert not small._study_backend_ok("torch-mps", {**mps, "linear_attn": "fla+causal_conv1d"},
                                       "e1-small", "smoke")
    assert not small._study_backend_ok("torch", mps, "e1-small", "smoke")
    with pytest.raises(SystemExit, match="does not come"):
        small.check_provenance([{**_cell(*SMALL), "backend": "torch-mps", "versions": mps}],
                               "tuning", "c" * 64)
