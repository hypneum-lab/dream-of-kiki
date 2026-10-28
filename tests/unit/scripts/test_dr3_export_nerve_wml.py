"""DR-3 export helper: dump contract, learning gate, pin check, CLI.

nerve-wml and torch are not dependencies of this repository, so the
substrate API is replaced by a numpy fake that records its calls. The
torch adapter itself is exercised by running the helper inside a
nerve-wml v1.9.0 checkout (see the script docstring).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from scripts.dr3_export_nerve_wml import (
    N_EVAL,
    NERVE_WML_COMMIT,
    STEPS,
    check_checkout,
    export_seed,
    main,
    parse_seeds,
    run_export,
)
from scripts.dr3_relational_probe import analyse_seed, load_seed_npz


class FakeAPI:
    """Numpy stand-in for the v1.9.0 substrates."""

    def __init__(self, acc: dict[str, float] | None = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.acc = acc or {"a_trained": 0.6, "b_trained": 0.4, "a_null": 0.08, "b_null": 0.1}
        self.eval_x: Any = None

    def train_mlp(self, steps: int, seed: int) -> Any:
        self.calls.append(("mlp", steps, seed))
        return ("a", "trained" if steps else "null", seed)

    def train_lif(self, steps: int, seed: int) -> Any:
        self.calls.append(("lif", steps, seed))
        return ("b", "trained" if steps else "null", seed)

    def eval_inputs(self, seed: int, n: int) -> Any:
        self.calls.append(("eval", seed, n))
        self.eval_x = np.random.default_rng(seed).normal(size=(n, 16))
        return self.eval_x

    def gate_batch(self, seed: int, n: int) -> tuple[Any, Any]:
        self.calls.append(("gate", seed, n))
        return "xg", "yg"

    def _embed(self, model: Any, x: Any) -> np.ndarray:
        assert x is self.eval_x  # the same inputs feed all four models
        sub, state, seed = model
        rng = np.random.default_rng([seed, ord(sub), state == "trained"])
        out: np.ndarray = x @ rng.normal(size=(16, 16))
        return out

    def embed_mlp(self, mlp: Any, x: Any) -> np.ndarray:
        return self._embed(mlp, x)

    def embed_lif(self, lif: Any, x: Any) -> np.ndarray:
        return self._embed(lif, x)

    def _acc(self, model: Any, x: Any, y: Any) -> float:
        assert (x, y) == ("xg", "yg")
        return self.acc[f"{model[0]}_{model[1]}"]

    def accuracy_mlp(self, mlp: Any, x: Any, y: Any) -> float:
        return self._acc(mlp, x, y)

    def accuracy_lif(self, lif: Any, x: Any, y: Any) -> float:
        return self._acc(lif, x, y)

    def versions(self) -> dict[str, str]:
        return {"torch": "fake"}


def _git_ok(args: Sequence[str], root: Path) -> str:
    return {"rev-parse": NERVE_WML_COMMIT, "status": ""}[args[0]]


def test_dump_matches_probe_input_contract(tmp_path: Path) -> None:
    api = FakeAPI()
    entry = export_seed(api, 900, tmp_path)
    path = tmp_path / "dr3-seed900.npz"
    rec = load_seed_npz(path, require_hash=True)  # sidecar verified
    assert rec["seed"] == 900 and rec["sha256"] == entry["sha256"]
    with np.load(path) as z:
        for key in ("a_trained", "b_trained", "a_null", "b_null"):
            assert z[key].dtype == np.float32 and z[key].shape == (N_EVAL, 16)
    assert analyse_seed(rec, k=10, tau=0.1)["status"] == "ok"


def test_trained_and_null_pairs_use_same_seed_and_steps(tmp_path: Path) -> None:
    api = FakeAPI()
    export_seed(api, 105, tmp_path)
    assert ("mlp", STEPS, 105) in api.calls and ("lif", STEPS, 105) in api.calls
    assert ("mlp", 0, 105) in api.calls and ("lif", 0, 105) in api.calls
    assert api.calls.count(("eval", 105, N_EVAL)) == 1
    assert api.calls.count(("gate", 105, N_EVAL)) == 1


def test_sidecar_format_and_array_hashes(tmp_path: Path) -> None:
    entry = export_seed(FakeAPI(), 900, tmp_path)
    path = tmp_path / "dr3-seed900.npz"
    side = (tmp_path / "dr3-seed900.npz.sha256").read_text()
    assert side == f"{hashlib.sha256(path.read_bytes()).hexdigest()}  dr3-seed900.npz\n"
    with np.load(path) as z:
        assert entry["array_sha256"]["b_null"] == hashlib.sha256(z["b_null"].tobytes()).hexdigest()


def test_gate_pass_recorded(tmp_path: Path) -> None:
    entry = export_seed(FakeAPI(), 900, tmp_path)
    assert entry["gate_pass"] is True and entry["gate_reasons"] == []
    with np.load(tmp_path / "dr3-seed900.npz") as z:
        assert bool(z["gate_pass"]) is True
        assert float(z["acc_b_trained"]) == pytest.approx(0.4)


def test_gate_failure_on_trained_substrate_excludes_in_probe(tmp_path: Path) -> None:
    api = FakeAPI({"a_trained": 0.6, "b_trained": 0.2, "a_null": 0.08, "b_null": 0.1})
    entry = export_seed(api, 101, tmp_path)
    assert entry["gate_pass"] is False
    assert entry["gate_reasons"] == ["learning gate: b_trained accuracy 0.2000 < 0.25"]
    out = analyse_seed(load_seed_npz(tmp_path / "dr3-seed101.npz"), k=10, tau=0.1)
    assert out["status"] == "excluded"


def test_gate_threshold_is_inclusive_and_ignores_null_accuracy(tmp_path: Path) -> None:
    api = FakeAPI({"a_trained": 0.25, "b_trained": 0.25, "a_null": 0.0, "b_null": 0.0})
    assert export_seed(api, 102, tmp_path)["gate_pass"] is True


def test_never_overwrites(tmp_path: Path) -> None:
    export_seed(FakeAPI(), 900, tmp_path)
    with pytest.raises(FileExistsError):
        export_seed(FakeAPI(), 900, tmp_path)
    with pytest.raises(FileExistsError):
        run_export(FakeAPI(), [900], tmp_path, {"pinned": True})


def test_run_export_manifest(tmp_path: Path) -> None:
    api = FakeAPI()
    checkout = check_checkout(tmp_path, run=_git_ok)
    doc = run_export(api, [100, 101], tmp_path / "out", checkout)
    man = tmp_path / "out" / "export-manifest.json"
    assert json.loads(man.read_text()) == doc
    side = (tmp_path / "out" / "export-manifest.json.sha256").read_text().split()[0]
    assert side == hashlib.sha256(man.read_bytes()).hexdigest()
    assert doc["nerve_wml"]["commit"] == NERVE_WML_COMMIT and doc["nerve_wml"]["tag"] == "v1.9.0"
    assert doc["confirmatory_eligible"] is True and doc["n_gate_fail"] == 0
    for e in doc["seeds"]:
        f = tmp_path / "out" / e["file"]
        assert e["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()


def test_check_checkout_rejects_other_commit_and_dirty_tree(tmp_path: Path) -> None:
    def other(args: Sequence[str], root: Path) -> str:
        return {"rev-parse": "0" * 40, "status": ""}[args[0]]

    def dirty(args: Sequence[str], root: Path) -> str:
        return {"rev-parse": NERVE_WML_COMMIT, "status": " M track_w/training.py"}[args[0]]

    with pytest.raises(RuntimeError, match="expected 95ac6b7"):
        check_checkout(tmp_path, run=other)
    with pytest.raises(RuntimeError, match="modified"):
        check_checkout(tmp_path, run=dirty)
    rec = check_checkout(tmp_path, run=other, allow_unpinned=True)
    assert rec["pinned"] is False and rec["problems"]
    assert check_checkout(tmp_path, run=_git_ok)["pinned"] is True


def test_parse_seeds() -> None:
    assert parse_seeds("100-119") == list(range(100, 120))
    assert parse_seeds("900") == [900]
    assert parse_seeds("0,1,2") == [0, 1, 2]
    with pytest.raises(ValueError):
        parse_seeds("5,5")
    with pytest.raises(ValueError):
        parse_seeds("10-9")


def test_cli_end_to_end_with_probe(tmp_path: Path) -> None:
    out = tmp_path / "dumps"
    rc = main(
        ["--nerve-wml-root", str(tmp_path), "--seeds", "100-101", "--out", str(out)],
        api_factory=lambda root: FakeAPI(),
        git=_git_ok,
    )
    assert rc == 0
    assert sorted(p.name for p in out.glob("*.npz")) == ["dr3-seed100.npz", "dr3-seed101.npz"]
    from scripts.dr3_relational_probe import main as probe_main

    stats = tmp_path / "stats.json"
    assert probe_main([*map(str, sorted(out.glob("*.npz"))), "--require-hash",
                       "--output", str(stats)]) == 0
    assert json.loads(stats.read_text())["n_retained"] == 2


def test_cli_refuses_unpinned_checkout(tmp_path: Path) -> None:
    def other(args: Sequence[str], root: Path) -> str:
        return {"rev-parse": "0" * 40, "status": ""}[args[0]]

    with pytest.raises(RuntimeError, match="not pinned"):
        main(["--seeds", "900", "--out", str(tmp_path)], api_factory=lambda r: FakeAPI(), git=other)


# --- export -> probe --decide integration (review fix 1) -------------------


def _git_other(args: Sequence[str], root: Path) -> str:
    return {"rev-parse": "0" * 40, "status": ""}[args[0]]


def _export(tmp_path: Path, seeds: str, git: Any = _git_ok, unpinned: bool = False) -> Path:
    out = tmp_path / "dumps"
    argv = ["--nerve-wml-root", str(tmp_path), "--seeds", seeds, "--out", str(out)]
    if unpinned:
        argv.append("--allow-unpinned")
    assert main(argv, api_factory=lambda root: FakeAPI(), git=git) == 0
    return out


def _decide(dumps: Path, tmp_path: Path, files: list[Path] | None = None) -> dict[str, Any]:
    from scripts.dr3_relational_probe import main as probe_main

    stats = tmp_path / "stats.json"
    files = files if files is not None else sorted(dumps.glob("*.npz"))
    assert probe_main([*map(str, files), "--require-hash", "--manifest",
                       str(dumps / "export-manifest.json"), "--decide",
                       "--output", str(stats)]) == 0
    decision: dict[str, Any] = json.loads(stats.read_text())["decision"]
    return decision


@pytest.fixture(scope="module")
def pinned_dumps(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _export(tmp_path_factory.mktemp("pinned"), "100-119")


def test_pinned_complete_export_is_eligible(pinned_dumps: Path, tmp_path: Path) -> None:
    out = _decide(pinned_dumps, tmp_path)
    assert out["eligible"] is True, out["eligibility_problems"]
    assert out["row"] in range(1, 6)


def test_incomplete_seed_set_is_not_eligible(pinned_dumps: Path, tmp_path: Path) -> None:
    files = sorted(pinned_dumps.glob("*.npz"))[:-1]
    out = _decide(pinned_dumps, tmp_path, files)
    assert out["eligible"] is False and out["row"] is None
    assert any("100..119" in p for p in out["eligibility_problems"])


def test_unpinned_export_is_not_eligible(tmp_path: Path) -> None:
    dumps = _export(tmp_path, "100-119", git=_git_other, unpinned=True)
    with np.load(dumps / "dr3-seed100.npz") as z:
        assert bool(z["pinned"]) is False
    out = _decide(dumps, tmp_path)
    assert out["eligible"] is False and out["row"] is None
    assert any("unpinned" in p for p in out["eligibility_problems"])


def test_missing_gate_is_not_eligible(pinned_dumps: Path, tmp_path: Path) -> None:
    import shutil

    dumps = tmp_path / "copy"
    shutil.copytree(pinned_dumps, dumps)
    f = dumps / "dr3-seed105.npz"
    with np.load(f) as z:
        kept = {k: z[k] for k in z.files if not k.startswith("acc_") and k != "gate_pass"}
    np.savez(f, **kept)
    digest = hashlib.sha256(f.read_bytes()).hexdigest()
    (dumps / "dr3-seed105.npz.sha256").write_text(f"{digest}  {f.name}\n")
    out = _decide(dumps, tmp_path)
    assert out["eligible"] is False
    assert any("gate not scored" in p for p in out["eligibility_problems"])
    assert any("do not match" in p for p in out["eligibility_problems"])
