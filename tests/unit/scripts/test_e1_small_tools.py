"""Torch-free parts of the E1-small S6 / S8 checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import e1_small_kernel_parity as kp
from scripts import e1_small_token_ids as ti
from scripts import e1_sources as src

GOOD = {"logprob_max_abs": 0.1, "logprob_mean_abs": 0.005, "choice_argmax_agree": 1.0,
        "choice_score_max_abs": 0.01, "loss_rel": 0.001, "grad_cos": 0.999,
        "grad_norm_ratio": 1.005, "grad_rel_err": 0.05, "grad_keys_equal": 1.0}


def test_verdict_pass_and_each_failure() -> None:
    assert kp.verdict(GOOD)["verdict"] == "PASS"
    bad = {"logprob_max_abs": 0.3, "logprob_mean_abs": 0.03, "choice_argmax_agree": 0.9,
           "choice_score_max_abs": 0.06, "loss_rel": 0.02, "grad_cos": 0.98,
           "grad_norm_ratio": 1.03, "grad_rel_err": 0.2, "grad_keys_equal": 0.0}
    for k, v in bad.items():
        out = kp.verdict({**GOOD, k: v})
        assert out["verdict"] == "FAIL" and not out["checks"][k.removesuffix("_equal")]
    assert kp.verdict({**GOOD, "grad_cos": float("nan")})["verdict"] == "FAIL"


def test_token_digest_texts_and_pin(tmp_path: Path) -> None:
    assert ti.TOKENIZER_SHA256 == src.TOKENIZER_SHA256
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(json.dumps({"prompt": "Q?", "choices": ["a", "b", "c", "d"]}) + "\n\n",
                      encoding="utf-8")
    assert ti.harness_texts(corpus.read_text().splitlines()[0]) == [
        "Q?\nAnswer:", "Q?\nAnswer: a", "Q?\nAnswer: b", "Q?\nAnswer: c", "Q?\nAnswer: d"]
    d1 = ti.digest(corpus, lambda t: [ord(x) for x in t])
    d2 = ti.digest(corpus, lambda t: [ord(x) for x in t][:-1])
    assert d1["n_texts"] == 5 and d1["max_tokens"] == 12 and d1["sha256"] != d2["sha256"]
    fake = tmp_path / "tokenizer.json"
    fake.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="pinned"):
        ti.main(["--corpus", str(corpus), "--tokenizer", str(fake)])


def test_diag_decision_rule() -> None:
    from scripts import e1_small_kernel_diag as kd

    def m(mean: float, agree: float) -> dict[str, float]:
        return {"logprob_mean_abs": mean, "argmax_agree": agree}

    assert kd.decide(m(0.01, 1.0), m(0.02, 0.96)) == "fla+causal_conv1d"
    assert kd.decide(m(0.02, 1.0), m(0.02, 1.0)) == "fla+causal_conv1d"  # ties: at least as close
    assert kd.decide(m(0.03, 1.0), m(0.02, 0.96)) == "torch"
    assert kd.decide(m(0.01, 0.92), m(0.02, 0.96)) == "torch"


def test_determinism_compare() -> None:
    from scripts import e1_small_determinism_check as dc

    r = {"pre": 0.5, "post": 0.6, "scores": [[[0.1, 0.2]]], "param_sha256": "a",
         "invariant_violation": None, "achieved_loss_tokens": 10, "planned_loss_tokens": 10,
         "n_updates": 3}
    assert dc.compare([r, dict(r), dict(r)], "base")["verdict"] == "PASS"
    for k, v in (("post", 0.61), ("scores", [[[0.1, 0.3]]]), ("param_sha256", "b")):
        out = dc.compare([r, {**r, k: v}], "base")
        assert out["verdict"] == "FAIL" and not out["identical"][k]
    for change, check in (({"invariant_violation": "S2"}, "no_invariant_violation"),
                          ({"achieved_loss_tokens": 4}, "budget_achieved"),
                          ({"n_updates": 0}, "updated")):
        out = dc.compare([{**r, **change}, {**r, **change}], "base")
        assert out["verdict"] == "FAIL" and not out["valid_replay"][check]
    out = dc.compare([r, r], "a")
    assert out["verdict"] == "FAIL" and not out["valid_replay"]["parameters_changed"]
    assert dc.compare([], "base")["verdict"] == "FAIL"
