"""Build the pinned G4-nono QD-CIFAR10 corpus (CIFAR-10 + Quick, Draw!).

**Gate** : ``docs/osf-prereg-g4-nono-cross-domain-2026-10-23.md`` section 5
(domain corpus, PI decision 2026-10-24) and checklist item 5.
**Mode** : data preparation, no empirical claim.

Steps: verify the CIFAR-10 parquet SHA-256 pins; download any missing
Quick, Draw! ``.npy`` (only the 10 registered categories) and verify the
pinned SHA-256 and byte size of each; subsample 2 500 / 500 per class
with the pinned seed; write ``qd-cifar10-v1.npz`` and a manifest; write
the index lists to ``experiments/g4_nono_test/qd_cifar10_indices.json``;
with ``--delete-raw`` remove the raw ``.npy`` files afterwards.

Usage ::

    uv run python scripts/g4_build_qd_cifar10.py \\
        --cifar-dir experiments/g4_quinto_test/data \\
        --raw-dir /tmp/qd-raw --out experiments/g4_nono_test/data --delete-raw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from experiments.g4_nono_test import qd_cifar10 as qd  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(name: str, raw_dir: Path) -> Path:
    path = raw_dir / f"{name}.npy"
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(qd.QD_URL.format(name=name), path)  # noqa: S310
    pin = qd.QD_PINS[name]
    size, digest = path.stat().st_size, _sha256(path)
    if size != pin["bytes"] or digest != pin["sha256"]:
        raise ValueError(f"{name}.npy: got {size} B / {digest}, pinned {pin['bytes']} B / {pin['sha256']}")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build the QD-CIFAR10 corpus")
    ap.add_argument("--cifar-dir", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--delete-raw", action="store_true")
    args = ap.parse_args(argv)

    from experiments.g4_quinto_test.cifar10_dataset import _decode_parquet_shard

    tr_pq, te_pq = args.cifar_dir / "cifar10-train.parquet", args.cifar_dir / "cifar10-test.parquet"
    for path, want in ((tr_pq, qd.CIFAR10_TRAIN_SHA256), (te_pq, qd.CIFAR10_TEST_SHA256)):
        if _sha256(path) != want:
            raise ValueError(f"{path}: SHA-256 does not match the pin {want}")
    cifar_train = _decode_parquet_shard(tr_pq)
    cifar_test = _decode_parquet_shard(te_pq)
    raw = {name: _fetch(name, args.raw_dir) for name in qd.QD_CLASSES}
    drawings = {name: np.load(p, allow_pickle=False) for name, p in raw.items()}

    corpus = qd.build_corpus(cifar_train, cifar_test, drawings)
    npz = qd.save_corpus(corpus, args.out)
    idx_text = qd.indices_json(corpus["indices"])
    qd.INDEX_FILE.write_text(idx_text)
    manifest = {
        "corpus_file": npz.name,
        "corpus_sha256": _sha256(npz),
        "index_file": qd.INDEX_FILE.name,
        "index_sha256": hashlib.sha256(idx_text.encode()).hexdigest(),
        "cifar10": {"hf_repo": "uoft-cs/cifar10", "revision": qd.CIFAR10_HF_REVISION,
                    "train_sha256": qd.CIFAR10_TRAIN_SHA256, "test_sha256": qd.CIFAR10_TEST_SHA256,
                    "classes": list(qd.CIFAR10_CLASSES), "preprocess": "BT.601 gray x3"},
        "quickdraw": {"url": qd.QD_URL, "licence": "CC BY 4.0 (Google Creative Lab)",
                      "classes": list(qd.QD_CLASSES), "pins": qd.QD_PINS,
                      "preprocess": "28 -> 32 zero pad, gray x3"},
        "subsample": {"seed": qd.SUBSAMPLE_SEED, "n_train": qd.N_TRAIN, "n_test": qd.N_TEST},
        "label_of": {f"{c}/{k}": v for (c, k), v in qd.LABEL_OF.items()},
    }
    (args.out / "qd-cifar10-v1.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.delete_raw:
        for p in raw.values():
            p.unlink()
    print(json.dumps({k: manifest[k] for k in ("corpus_sha256", "index_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
