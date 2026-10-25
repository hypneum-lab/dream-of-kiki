"""E1-small instrument pieces of the PyTorch backend (skipped without torch):
raw tokenizer (PI decision S8) and linear-attention selection (S6)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")

from scripts import e1_harness as hn
from scripts import e1_sources as src
from scripts import pilot_cycle3_cuda as h


def test_raw_tokenizer_refuses_another_file(tmp_path: Path) -> None:
    fake = tmp_path / "tokenizer.json"
    fake.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="pinned"):
        h.RawTokenizer(fake)


def test_raw_tokenizer_encodes_without_special_tokens(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers  # type: ignore[import-untyped]

    vocab = {"[UNK]": 0, "a": 1, "b": 2}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    t.save(str(path))
    monkeypatch.setattr(src, "TOKENIZER_SHA256", hn.sha_file(path))
    tok = h.RawTokenizer(path)
    assert tok.kind == "raw"
    assert tok("a b a").input_ids.tolist() == [[1, 2, 1]]
    assert tok("").input_ids.shape == (1, 0)


def _fla_fn() -> Any:
    def f() -> None:
        return None

    f.__module__ = "fla.ops.gated_delta_rule"
    return f


@pytest.mark.parametrize(("delta", "conv", "want"), [
    (_fla_fn(), object(), "fla+causal_conv1d"), (len, None, "torch"),
    (_fla_fn(), None, "mixed")])
def test_linear_attn_impl(delta: Any, conv: Any, want: str) -> None:
    layer = torch.nn.Module()
    layer.chunk_gated_delta_rule = delta
    layer.causal_conv1d_fn = conv
    model = torch.nn.Sequential(layer, torch.nn.Linear(2, 2))
    assert h.linear_attn_impl(model) == want
    assert h.linear_attn_impl(torch.nn.Linear(2, 2)) == "none"


def test_force_torch_linear_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    for attr in (*h._LINEAR_ATTN_GLOBALS, "is_fast_path_available"):
        if hasattr(mod, attr):
            monkeypatch.setattr(mod, attr, getattr(mod, attr) or SimpleNamespace())
    h.force_torch_linear_attention()
    assert all(getattr(mod, a) is None for a in h._LINEAR_ATTN_GLOBALS if hasattr(mod, a))
    assert mod.is_fast_path_available is False


def _tiny() -> tuple[Any, Any]:
    from tests.unit.scripts.test_pilot_cycle3_cuda import _TinyLM, _Tok

    return _TinyLM(), _Tok()


def test_fused_sgd_gives_identical_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    texts = ["alpha beta gamma delta", "epsilon zeta", "eta theta iota kappa lambda"]
    out = {}
    for fused in ("0", "1"):
        monkeypatch.setenv("HL_FUSED_SGD", fused)
        model, tok = _tiny()
        b = h.TorchBackend(model, tok)
        assert b.sgd_kind == ("fused" if fused == "1" else "optim")
        res = hn.replay(b, texts, 3e-2, 2, 60)
        assert res["invariant_violation"] is None
        out[fused] = {k: v.clone() for k, v in model.state_dict().items()}
    assert all(torch.equal(out["0"][k], out["1"][k]) for k in out["0"])
    assert any(not torch.equal(out["0"][k], _tiny()[0].state_dict()[k]) for k in out["0"])


def test_disk_snapshot_restores_exactly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HL_SNAPSHOT", "disk")
    monkeypatch.setenv("HL_SNAPSHOT_DIR", str(tmp_path / "snap"))
    model, tok = _tiny()
    ref = {k: v.clone() for k, v in model.state_dict().items()}
    b = h.TorchBackend(model, tok)
    b.snapshot()
    assert isinstance(b.base, h.DiskSnapshot) and b.versions()["snapshot"] == "disk"
    b.downscale(0.5)
    d_disk = b.drift(router_only=False)
    ram = h.TorchBackend(model, tok, {k: v.clone() for k, v in ref.items()})
    assert ram.drift(router_only=False) == d_disk and d_disk[0] > 0
    b.restore()
    assert all(torch.equal(v, ref[k]) for k, v in model.state_dict().items())
    monkeypatch.setenv("HL_SNAPSHOT", "bad")
    with pytest.raises(SystemExit, match="HL_SNAPSHOT"):
        h.TorchBackend(model, tok)
