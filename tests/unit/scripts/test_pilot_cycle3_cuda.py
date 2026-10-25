"""E1 CUDA harness wiring on a tiny CPU model (skipped without torch)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from scripts import e1_profiles as pr
from scripts import pilot_cycle3_cuda as h

V = 97


class _Tok:
    def __call__(self, text: str, return_tensors: str = "pt") -> SimpleNamespace:
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) % V for c in text]]))


class _Block(torch.nn.Module):  # type: ignore[name-defined,misc]
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)
        self.mlp = torch.nn.Module()
        self.mlp.gate = torch.nn.Linear(8, 4)  # router: 4 experts

    def forward(self, x: Any) -> Any:
        w = torch.softmax(self.mlp.gate(x), dim=-1).sum(-1, keepdim=True)
        return x + torch.tanh(self.lin(x)) * w


class _TinyLM(torch.nn.Module):  # type: ignore[name-defined,misc]
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(V, 8)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Block(), _Block(), _Block()])
        self.head = torch.nn.Linear(8, V)
        self.config = SimpleNamespace(num_experts_per_tok=2)

    def get_input_embeddings(self) -> Any:
        return self.emb

    def forward(self, ids: Any, labels: Any = None) -> SimpleNamespace:
        x = self.emb(ids)
        for b in self.model.layers:
            x = b(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits[0, :-1], labels[0, 1:])
        return SimpleNamespace(logits=logits, loss=loss)


def _bench(n: int = 60) -> dict[str, list[dict[str, Any]]]:
    items = []
    for i in range(n):
        p = f"Date: 2026-05-0{i % 9 + 1}\nQuestion {i} about item {i * 7}?\nAnswer:"
        items.append({"prompt": p, "question_prompt": p,
                      "choices": [f" opt{j}{i}" for j in range(4)], "answer": i % 4,
                      "source": ("realtimeqa", "daily_oracle", "mmlu_cf")[i % 3],
                      "item_sha256": f"{i:064x}", "src_hash": f"{i:064x}"})
    return {"facts": items}


@pytest.fixture(scope="module")
def setup() -> tuple[Any, Any, Any]:
    model, tok = _TinyLM(), _Tok()
    return model, tok, h._snapshot_cpu(model)


AM = pr.Amendment((3e-4, 2), (3e-5, 5), 2.0, 12)


@pytest.mark.parametrize("profile", pr.ALL_PROFILES)
def test_every_profile_cell(setup: tuple[Any, Any, Any], profile: str) -> None:
    model, tok, base = setup
    cell = h.profile_cell(model, tok, base, _bench(), profile, 3, AM, AM.n_eval)
    assert cell["n_replay"] == 40 and cell["n_eval_items"] == 12
    if profile == "p_noise":
        assert cell["n_updates"] == 0
        return
    b = cell["budget"]
    assert cell["achieved_loss_tokens"] == b["update_tokens"] == cell["planned_loss_tokens"]
    if profile == "p_ctrl_onpolicy_replay":
        assert b["charge"] == -(-cell["s_opr"] // 3) and b["update_tokens"] == cell["t_s"] - b["charge"]
    else:
        assert b["charge"] == 0 and b["update_tokens"] == cell["t_s"]
    if cell["n_updates"]:
        assert cell["drift"] > 0 and cell["kl_ref"] >= 0 and cell["usage_l1"] is not None
    assert (cell["ops"]["downscale"] is not None) == (profile in ("p_equ", "p_max"))
    assert (cell["ops"]["restructure_idx"] is not None) == (profile == "p_max")
    assert len(cell["pre_correct"]) == len(cell["eval_sources"]) == 12


def test_restore_is_exact_and_split_is_disjoint(setup: tuple[Any, Any, Any]) -> None:
    model, tok, base = setup
    h.profile_cell(model, tok, base, _bench(), "p_max", 4, AM, AM.n_eval)
    h._restore(model, base)
    assert all(torch.equal(v, base[k]) for k, v in model.state_dict().items())
    ev, rep = h._cell_split(_bench(), 4, 40, 12)
    assert not {x["item_sha256"] for x in ev["facts"]} & {x["item_sha256"] for x in rep}


def test_template_corpus_has_no_answers(setup: tuple[Any, Any, Any]) -> None:
    _, rep = h._cell_split(_bench(), 5, 40, 12)
    texts = pr.profile_texts("prompts", rep, 5)
    assert all(t.endswith("\nAnswer:") for t in texts)


def test_calibration_cells_report_norm_change(setup: tuple[Any, Any, Any]) -> None:
    model, tok, base = setup
    a = h.run_cell(model, tok, base, _bench(), seed=1000, n_eval=0, corpus_kind="facts",
                   lr=3e-4, m=2, downscale=False, restructure=False, charge_opr=False)
    b = h.run_cell(model, tok, base, _bench(), seed=1000, n_eval=0, corpus_kind="facts",
                   lr=3e-4, m=2, downscale=True, restructure=False, charge_opr=False)
    assert a["n_eval_items"] == 20  # tuning: the rest of the partition
    assert b["norm_change"] > a["norm_change"] > 0


def test_pinned_revision_enforced(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="pinned revision"):
        h._load_model(h.MODEL_REPO, "main")
    for mode in ("headroom", "tuning", "calibrate", "confirmatory"):
        for extra in (["--model", "other/model"], ["--revision", "main"]):
            with pytest.raises(SystemExit, match="runs only on"):
                h.main(["--mode", mode, "--out", str(tmp_path), *extra])


def test_amend_mode(tmp_path: Path) -> None:
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"status": "FROZEN", "corpus_sha256": "c" * 64,
                               "partitions": {"confirmatory": {"n_items": 500}}}))
    tune, cal = tmp_path / "t.jsonl", tmp_path / "c.jsonl"
    rows_t, rows_c = [], []
    for s in range(1000, 1010):
        for lr, m in pr.GRID:
            rows_t.append({"kind": "tune_vanilla", "seed": s, "lr": lr, "m": m,
                           "delta": 0.01 if (lr, m) == (1e-4, 2) else 0.0,
                           "post_correct": [True, False]})
            rows_t.append({"kind": "tune_opr", "seed": s, "lr": lr, "m": m, "delta": 0.0,
                           "opr_corpus_size": 5, "post_correct": [True, True]})
        for c in pr.C_GRID:
            rows_c.append({"kind": "calib_lr", "seed": s, "c": c, "lr": c * 1e-4, "m": 2,
                           "norm_change": c})
        rows_c.append({"kind": "calib_ref", "seed": s, "lr": 1e-4, "m": 2, "norm_change": 1.4})
    prov = {"model": h.MODEL_REPO, "revision": h.MODEL_REVISION,
            "corpus": {"corpus_sha256": "c" * 64, "partition": "tuning"}}
    for r in rows_t:
        r.update(prov, mode="tuning", run_identity="t1")
    tune.write_text("".join(json.dumps(r) + "\n" for r in rows_t))
    for r in rows_c:
        r.update(prov, mode="calibrate", run_identity="c1", selection_vanilla=[1e-4, 2],
                 tuning_cells_sha256=h._sha_file(tune))
    cal.write_text("".join(json.dumps(r) + "\n" for r in rows_c))
    assert h._selection_i(tune, "c" * 64) == (1e-4, 2)
    with pytest.raises(SystemExit, match="does not come"):
        h._selection_i(tune, "d" * 64)
    assert h.main(["--mode", "amend", "--manifest", str(man), "--tuning-cells", str(tune),
                   "--calib-cells", str(cal), "--out", str(tmp_path)]) == 0
    am = json.loads((tmp_path / "e1-tuning-amendment.json").read_text())
    assert am["selection_vanilla"] == [1e-4, 2] and am["c"] == 1.5
    assert am["n_eval_rule"] == 636 and am["n_eval"] == 460 and am["capped"]


def test_check_provenance_refusals() -> None:
    base = {"mode": "tuning", "model": h.MODEL_REPO, "revision": h.MODEL_REVISION,
            "corpus": {"corpus_sha256": "c" * 64, "partition": "tuning"}, "run_identity": "r"}
    h.check_provenance([base], "tuning", "c" * 64)
    for change in ({"revision": "main"}, {"mode": "calibrate"}, {"run_identity": None},
                   {"corpus": {"corpus_sha256": "c" * 64, "partition": "confirmatory"}}):
        with pytest.raises(SystemExit):
            h.check_provenance([{**base, **change}], "tuning", "c" * 64)
    with pytest.raises(SystemExit, match="more than one"):
        h.check_provenance([base, {**base, "run_identity": "s"}], "tuning", "c" * 64)
    with pytest.raises(SystemExit):
        h.check_provenance([{**base, "mode": "calibrate", "tuning_cells_sha256": "x"}],
                           "calibrate", "c" * 64, "y")
