"""Harness mode ``dose-response`` and per-tensor noise on the tiny CPU model.

docs/milestones/2026-11-11-downscale-dose-response.md (L0 draft).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts import e1_dose_response as dr
from scripts import e1_harness as hn
from scripts import pilot_cycle3_cuda as h
from tests.unit.scripts.test_e1_controls import _args, _backend, _mode_setup
from tests.unit.scripts.test_pilot_cycle3_cuda import _TinyLM, _Tok


def test_per_tensor_noise_matches_each_tensor_norm() -> None:
    b = _backend()
    b.restore()
    theta = {k: v.detach().clone() for k, v in b.model.named_parameters()}
    info = b.perturb_per_tensor(0.1, 3)
    assert info["xi_norm"] == pytest.approx(0.1 * info["theta_norm"], rel=1e-5)
    assert info["applied_norm"] == pytest.approx(info["xi_norm"], rel=1e-4)
    for k, v in b.model.named_parameters():
        step = float((v.detach().float() - theta[k].float()).norm())
        assert step == pytest.approx(0.1 * float(theta[k].float().norm()), rel=1e-4)
    after = {k: v.detach().clone() for k, v in b.model.named_parameters()}
    b.restore()
    b.perturb_per_tensor(0.1, 3)  # same seed: same perturbation
    assert all(torch.equal(v, after[k]) for k, v in b.model.named_parameters())
    b.restore()
    b.perturb_per_tensor(0.1, 4)
    assert any(not torch.equal(v, after[k]) for k, v in b.model.named_parameters())


def test_dose_response_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                            capsys: pytest.CaptureFixture[str]) -> None:
    man, amd = _mode_setup(tmp_path, monkeypatch)
    fac = lambda *a: h.TorchBackend(_TinyLM(), _Tok())  # noqa: E731
    assert hn.run(_args(tmp_path, man, amd, "dose-response", None), fac) == 0
    path = tmp_path / "out" / "e1-dose-response.passes.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert lines[0]["kind"] == "header" and len(lines[0]["items"]) == 60
    got = [(x["arm"], x["eps"], x["seed"]) for x in lines[1:]]
    assert got == dr.plan()
    assert all(set(x["scores"]) == set(lines[0]["items"]) for x in lines[1:])
    passes, answers, sources = dr.load(path)
    assert len(passes) == 28 and answers.shape == (60,) and len(sources) == 60
    out = capsys.readouterr().out
    assert out.count("dose-response: pass") == 28 and "acc" not in out
    with pytest.raises(SystemExit, match="overwrite"):
        hn.run(_args(tmp_path, man, amd, "dose-response", None), fac)
