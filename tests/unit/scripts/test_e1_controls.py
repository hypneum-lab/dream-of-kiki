"""E1-small controls addendum (2026-11-06): same-norm arm and options-only mode.

Tiny CPU model (skipped without torch). docs/osf-addendum-e1-small-controls-2026-11-06.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from scripts import e1_harness as hn
from scripts import e1_profiles as pr
from scripts import pilot_cycle3_cuda as h
from tests.unit.scripts.test_pilot_cycle3_cuda import _bench, _TinyLM, _Tok

AM = pr.Amendment((3e-4, 2), (3e-5, 5), 2.0, 12)


def _backend() -> h.TorchBackend:
    model = _TinyLM()
    b = h.TorchBackend(model, _Tok())
    b.snapshot()
    return b


def test_control_profiles_are_outside_the_registered_set() -> None:
    assert not set(pr.SAMENORM) & set(pr.ALL_PROFILES)
    assert pr.CONTROL_PROFILES == ("p_equ", "p_ctrl_vanilla_replay_tuned",
                                   "p_ctrl_samenorm_r", "p_ctrl_samenorm_u")
    for p, v in pr.SAMENORM.items():
        cfg = pr.profile_config(p, AM)
        ref = pr.profile_config("p_ctrl_vanilla_replay_tuned", AM)
        assert (cfg.corpus, cfg.lr, cfg.m) == (ref.corpus, ref.lr, ref.m) == ("facts", 3e-4, 2)
        assert not cfg.downscale and not cfg.restructure and cfg.samenorm == v
        with pytest.raises(ValueError, match="amendment"):
            pr.profile_config(p, None)
    assert pr.profile_config("p_equ", AM).samenorm is None


@pytest.mark.parametrize("variant", ["R", "U"])
def test_same_norm_perturbation(variant: str) -> None:
    b = _backend()
    b.restore()
    theta = {k: v.clone() for k, v in b.model.state_dict().items()}
    rho = b.downscale_norm(pr.SHRINK)
    # rho is the norm of the step downscale applies
    b.downscale(pr.SHRINK)
    ds = sum(float(((v.float() - theta[k]) ** 2).sum()) for k, v in b.model.state_dict().items())
    assert rho == pytest.approx(ds ** 0.5, rel=1e-6)
    b.model.load_state_dict(theta)
    info = b.perturb_same_norm(rho, variant, 7)
    assert info["xi_norm"] == pytest.approx(rho, rel=1e-5)
    assert info["applied_norm"] == pytest.approx(rho, rel=1e-4)  # float32 model: no rounding
    diff = {k: v - theta[k] for k, v in b.model.state_dict().items()}
    if variant == "U":  # toward zero, equal magnitude on non-zero elements
        for k, d in diff.items():
            nz = theta[k] != 0
            assert torch.all(torch.sign(d[nz]) == -torch.sign(theta[k][nz]))
        mags = torch.cat([d[theta[k] != 0].abs().flatten() for k, d in diff.items()])
        assert float(mags.max() - mags.min()) < 1e-6
    # same seed, same perturbation; the global RNG is not touched
    b.model.load_state_dict(theta)
    state = torch.get_rng_state()
    b.perturb_same_norm(rho, variant, 7)
    assert torch.equal(state, torch.get_rng_state())
    again = {k: v - theta[k] for k, v in b.model.state_dict().items()}
    assert all(torch.equal(diff[k], again[k]) for k in diff)
    with pytest.raises(ValueError, match="variant"):
        b.perturb_same_norm(rho, "X", 7)


def test_same_norm_bf16_rounding_is_measured() -> None:
    b = _backend()
    b.model.to(torch.bfloat16)
    b.snapshot()
    b.restore()
    rho = b.downscale_norm(pr.SHRINK)
    info = b.perturb_same_norm(rho, "R", 3)
    assert info["xi_norm"] == pytest.approx(rho, rel=1e-5)
    assert info["applied_norm"] > 0 and info["applied_norm"] != info["xi_norm"]


def test_same_norm_cells_pair_with_p_equ() -> None:
    b = _backend()
    cells = {p: hn.profile_cell(b, _bench(), p, 3, AM, AM.n_eval) for p in pr.CONTROL_PROFILES}
    v, e = cells["p_ctrl_vanilla_replay_tuned"], cells["p_equ"]
    for p in pr.SAMENORM:
        c = cells[p]
        assert c["ops"]["downscale"] is None and c["ops"]["samenorm"]["variant"] == pr.SAMENORM[p]
        assert c["achieved_loss_tokens"] == v["achieved_loss_tokens"] == e["achieved_loss_tokens"]
        assert c["pre_correct"] == v["pre_correct"]
        assert c["ops"]["samenorm"]["rho"] > 0 and c["kl_ref"] >= 0
    assert "samenorm" not in e["ops"] and e["ops"]["downscale"] == pr.SHRINK
    with pytest.raises(ValueError, match="replaces DOWNSCALE"):
        hn.apply_dream_ops(b, 0, True, False, "R")


def _mode_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    items = _bench()["facts"]
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"status": "FROZEN", "corpus_sha256": "c" * 64,
                               "partitions": {"confirmatory": {"n_items": len(items)}}}))
    amd = tmp_path / "a.json"
    amd.write_text(json.dumps({"selection_vanilla": [3e-4, 2], "selection_opr": [3e-5, 5],
                               "c": 2.0, "n_eval": 20, "p_disc": 0.0, "backend": "torch",
                               "manifest_sha256": hn.sha_file(man)}))
    prov = {"corpus_sha256": "c" * 64, "status": "FROZEN", "partition": "confirmatory"}
    monkeypatch.setattr(hn.cio, "load_corpus", lambda c, m, mode: (items, prov))
    return man, amd


def _args(tmp_path: Path, man: Path, amd: Path, mode: str, seeds: str | None) -> argparse.Namespace:
    ns = argparse.ArgumentParser()
    hn.add_args(ns)
    argv = ["--mode", mode, "--corpus", str(man), "--manifest", str(man),
            "--amendment", str(amd), "--out", str(tmp_path / "out")]
    if seeds:
        argv += ["--seeds", seeds]
    return ns.parse_args(argv)


def test_controls_mode_seed_major_quiet_and_stoppable(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    man, amd = _mode_setup(tmp_path, monkeypatch)
    fac = lambda *a: h.TorchBackend(_TinyLM(), _Tok())  # noqa: E731
    stop = tmp_path / "STOP"
    monkeypatch.setenv("HL_STOP_FILE", str(stop))
    stop.write_text("")
    assert hn.run(_args(tmp_path, man, amd, "controls", "0-1"), fac) == 3
    stop.unlink()
    assert hn.run(_args(tmp_path, man, amd, "controls", "0-1"), fac) == 0
    out = capsys.readouterr().out
    assert "pre=" not in out and "post=" not in out
    cache = next((tmp_path / "out").glob("e1-controls-*.cells.jsonl"))
    keys = [json.loads(x)["key"] for x in cache.read_text().splitlines()]
    assert keys == [f"{p}|{s}" for s in (0, 1) for p in pr.CONTROL_PROFILES]
    assert all(json.loads(x)["mode"] == "controls" for x in cache.read_text().splitlines())


def test_options_only_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                           capsys: pytest.CaptureFixture[str]) -> None:
    man, amd = _mode_setup(tmp_path, monkeypatch)
    seen: list[str] = []
    real = h.TorchBackend.score_items

    def spy(self: Any, items: Any) -> Any:
        seen.extend(it["prompt"] for it in items)
        return real(self, items)

    monkeypatch.setattr(h.TorchBackend, "score_items", spy)
    fac = lambda *a: h.TorchBackend(_TinyLM(), _Tok())  # noqa: E731
    assert hn.run(_args(tmp_path, man, amd, "options-only", None), fac) == 0
    assert set(seen) == {hn.OPTIONS_ONLY_PROMPT} and len(seen) == 60
    doc = json.loads((tmp_path / "out" / "e1-options-only.json").read_text())
    assert len(doc["items"]) == 60 and doc["prompt"] == "Answer:"
    assert all(len(r["scores"]) == 4 for r in doc["items"].values())
    assert "options-only: 60" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="overwrite"):
        hn.run(_args(tmp_path, man, amd, "options-only", None), fac)
