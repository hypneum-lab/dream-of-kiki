"""HellaSwag zero-shot loader (cycle-3 C3.1).

Binds the ``hellaswag`` :class:`DatasetPin` to a local JSONL fixture
whose rows follow the HF ``Rowan/hellaswag`` schema (``ctx`` +
``endings`` + ``label`` + ``activity_label``).

HellaSwag is evaluated **zero-shot** : the evaluation surface is
simply an in-order stream of records, so the loader's iterator
*is* the protocol. A seeded sample helper is provided for subset
tests that need fewer rows while preserving R1 determinism.

Reference :
  docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md §C3.1
  docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md §5
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from harness.real_benchmarks import MissingLocalDatasetError
from harness.real_benchmarks.dataset_registry import DatasetPin

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class HellaSwagRecord:
    """Frozen HellaSwag record — schema matches HF ``Rowan/hellaswag``.

    Fields
    ------
    ctx
        Context string (scene description).
    endings
        Four candidate continuations.
    label
        Integer index in ``endings`` identifying the correct
        ending.
    activity_label
        ActivityNet-style scene label (e.g. ``cooking``).
    """

    ctx: str
    endings: tuple[str, str, str, str]
    label: int
    activity_label: str


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class HellaSwagLoader:
    """Read-only HellaSwag loader bound to a :class:`DatasetPin`.

    See :class:`harness.real_benchmarks.mmlu.MMLULoader` for the
    SHA-256 + ``MissingLocalDatasetError`` contract ; behaviour is
    identical modulo the schema.
    """

    def __init__(
        self,
        registry_pin: DatasetPin,
        *,
        local_path: Path,
        expected_sha256: str | None = None,
    ) -> None:
        if not local_path.exists():
            raise MissingLocalDatasetError(
                f"HellaSwag fixture not found at {local_path!s} ; "
                "pass a pre-materialised JSONL export of "
                f"{registry_pin.hf_repo_id} (rev "
                f"{registry_pin.revision_sha}) — network fetch is "
                "disabled by design (R1 reproducibility)."
            )
        self._pin = registry_pin
        self._path = local_path
        self._actual_sha256 = _hash_file(local_path)
        if (
            expected_sha256 is not None
            and expected_sha256 != self._actual_sha256
        ):
            raise ValueError(
                f"sha256 mismatch on {local_path!s}: expected "
                f"{expected_sha256!r}, got {self._actual_sha256!r}"
            )
        self._hash_verified = expected_sha256 is not None

    @property
    def pin(self) -> DatasetPin:
        return self._pin

    @property
    def local_path(self) -> Path:
        return self._path

    @property
    def hash_verified(self) -> bool:
        return self._hash_verified

    def local_file_sha256(self) -> str:
        return self._actual_sha256

    def _iter_raw(self) -> Iterator[dict]:
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _record_from_raw(self, row: dict) -> HellaSwagRecord:
        endings = row["endings"]
        if len(endings) != 4:
            raise ValueError(
                f"HellaSwag row has {len(endings)} endings, "
                f"expected 4: {row!r}"
            )
        return HellaSwagRecord(
            ctx=str(row["ctx"]),
            endings=(
                str(endings[0]),
                str(endings[1]),
                str(endings[2]),
                str(endings[3]),
            ),
            label=int(row["label"]),
            activity_label=str(row.get("activity_label", "unknown")),
        )

    def iter_records(self) -> Iterator[HellaSwagRecord]:
        """Yield records in fixture order (zero-shot surface)."""
        for row in self._iter_raw():
            yield self._record_from_raw(row)

    def get_seeded_sample(
        self, seed: int, n: int
    ) -> list[HellaSwagRecord]:
        """Return ``n`` records drawn via a seeded permutation.

        Used by unit tests + pilots that want a deterministic
        subset (e.g. sanity pilot C3.7) while preserving R1.
        """
        raws = list(self._iter_raw())
        if n > len(raws):
            raise ValueError(
                f"requested {n} records but fixture has {len(raws)}"
            )
        rng = random.Random(seed)
        return [self._record_from_raw(r) for r in rng.sample(raws, n)]


# --------------------------------------------------------------------------
# Evaluator (cycle-3 C3.8 Phase A) — HellaSwag 4-choice continuation
# log-probability scoring. Network-free with a committed fallback fixture.
# --------------------------------------------------------------------------


_DEFAULT_HELLASWAG_FALLBACK = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "hellaswag_sanity.jsonl"
)


def _hellaswag_default_fallback_records() -> list[dict]:
    """Tiny hand-authored fallback — 8 commonsense-style rows.

    The full HellaSwag eval needs the real ``Rowan/hellaswag`` HF
    export for scientific claims. This fallback lets the pipeline
    run end-to-end on hosts without the HF cache, matching the
    same zero-shot schema (``ctx`` / ``endings`` / ``label`` /
    ``activity_label``). 8 rows cycled seeded up to ``n_samples``.
    """
    return [
        {
            "ctx": "The chef cracked three eggs into the bowl. They",
            "endings": [
                " started cleaning the windows.",
                " began singing a song loudly.",
                " whisked them together with a fork.",
                " watched television quietly.",
            ],
            "label": 2,
            "activity_label": "cooking",
        },
        {
            "ctx": "The dog saw the mailman approach the gate. It",
            "endings": [
                " started painting the fence.",
                " ran over and barked excitedly.",
                " read a book on the porch.",
                " baked a cake in the oven.",
            ],
            "label": 1,
            "activity_label": "animal_behavior",
        },
        {
            "ctx": "She was tired after the long run, so she",
            "endings": [
                " jumped up and did more push-ups.",
                " started another marathon immediately.",
                " sat down and drank some water.",
                " climbed the roof to sing.",
            ],
            "label": 2,
            "activity_label": "fitness",
        },
        {
            "ctx": "The rain started pouring heavily, so the cyclists",
            "endings": [
                " continued without concern.",
                " took shelter under the bridge.",
                " removed their helmets to feel the drops.",
                " raced faster through the storm.",
            ],
            "label": 1,
            "activity_label": "outdoor",
        },
        {
            "ctx": "He lit the candle on the birthday cake and",
            "endings": [
                " everyone sang happy birthday.",
                " the cat solved a puzzle.",
                " a ship arrived at the harbor.",
                " the tree grew five inches.",
            ],
            "label": 0,
            "activity_label": "celebration",
        },
        {
            "ctx": "The teacher handed out the exam papers. The students",
            "endings": [
                " left for the beach.",
                " started painting murals.",
                " picked up their pencils and began to write.",
                " planted tomatoes in the classroom.",
            ],
            "label": 2,
            "activity_label": "school",
        },
        {
            "ctx": "She plugged the guitar into the amplifier and",
            "endings": [
                " the cows started mooing louder.",
                " strummed a chord that filled the room.",
                " the oven preheated itself.",
                " the clock stopped ticking.",
            ],
            "label": 1,
            "activity_label": "music",
        },
        {
            "ctx": "The pilot announced turbulence, so the passengers",
            "endings": [
                " stood up to dance.",
                " opened the emergency doors.",
                " fastened their seat belts.",
                " started a cooking class.",
            ],
            "label": 2,
            "activity_label": "travel",
        },
    ]


def _load_hellaswag_records(
    n_samples: int,
    seed: int,
    *,
    fixture_path: Path | None = None,
) -> list[HellaSwagRecord]:
    """Materialise ``n_samples`` HellaSwag records with R1 discipline.

    Search order :

    1. Caller-supplied ``fixture_path`` (if set and existing).
    2. The committed fallback fixture at
       ``tests/fixtures/hellaswag_sanity.jsonl`` (if present).
    3. Offline HF cache (``Rowan/hellaswag`` validation split).
    4. The 8-row hand-authored in-module fallback —
       :func:`_hellaswag_default_fallback_records`.
    """
    def _materialise(raws: list[dict]) -> list[HellaSwagRecord]:
        rng = random.Random(seed)
        rng.shuffle(raws)
        if len(raws) >= n_samples:
            selected = raws[:n_samples]
        else:
            selected = [
                raws[i % len(raws)] for i in range(n_samples)
            ]
        records: list[HellaSwagRecord] = []
        for row in selected:
            endings = row["endings"]
            if len(endings) != 4:
                raise ValueError(
                    f"HellaSwag row has {len(endings)} endings: "
                    f"{row!r}"
                )
            records.append(
                HellaSwagRecord(
                    ctx=str(row["ctx"]),
                    endings=(
                        str(endings[0]),
                        str(endings[1]),
                        str(endings[2]),
                        str(endings[3]),
                    ),
                    label=int(row["label"]),
                    activity_label=str(
                        row.get("activity_label", "unknown")
                    ),
                )
            )
        return records

    # 1. Caller-supplied path — if the caller explicitly passed
    #    ``fixture_path`` but the file is missing, fail loudly
    #    rather than silently falling back to the HF cache or the
    #    in-module mini-fixture (which would mask a typo or a
    #    misplaced artefact and corrupt the run-registry signature).
    target = fixture_path or _DEFAULT_HELLASWAG_FALLBACK
    if fixture_path is not None and not target.exists():
        raise FileNotFoundError(
            f"caller supplied fixture_path={fixture_path!s} but no "
            f"file exists at {target!s} — refusing to fall back to "
            "HF cache or in-module fixture (R1 reproducibility)"
        )
    if target.exists():
        raws = []
        with target.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raws.append(json.loads(line))
        return _materialise(raws)

    # 2. Offline HF cache. Catch only the specific exceptions that
    #    can plausibly surface from an absent or unusable dataset
    #    package so unrelated failures (KeyboardInterrupt,
    #    MemoryError, …) propagate.
    try:  # pragma: no cover - optional cache path
        import os

        from datasets import load_dataset  # type: ignore[import-not-found]

        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        ds = load_dataset("Rowan/hellaswag", split="validation")
        raws = [dict(ex) for ex in ds.select(range(min(len(ds), n_samples * 2)))]
        return _materialise(raws)
    except (ImportError, ModuleNotFoundError, FileNotFoundError, OSError, ValueError):
        pass

    # 3. In-module hand-authored fallback — pipeline validation only.
    return _materialise(_hellaswag_default_fallback_records())


def _continuation_logprob(
    model_callable,
    tokenizer,
    ctx_ids: list[int],
    ending_ids: list[int],
) -> float:
    """Return the summed log-probability of ``ending_ids`` given ``ctx_ids``.

    Runs a single forward pass on ``ctx + ending``, then for each
    ending-position reads the predicted token's log-softmax entry.
    Sum (not mean) matches the lm-evaluation-harness ``loglikelihood``
    convention.
    """
    import mlx.core as mx
    import numpy as np

    full = ctx_ids + ending_ids
    tokens = mx.array([full])
    mx.random.seed(0)
    logits = model_callable(tokens)
    # Cast to fp32 on the MLX side before numpy conversion :
    # numpy has no bf16 dtype, so bf16 tensors must be widened.
    logits_fp32 = logits[0].astype(mx.float32)
    mx.eval(logits_fp32)
    logits_np = np.asarray(logits_fp32).astype(np.float32)
    # Position i predicts token at i+1. The ending spans tokens
    # indices [len(ctx), len(ctx)+len(ending)-1] ; use positions
    # [len(ctx)-1, …, len(ctx)+len(ending)-2] to read the logits.
    total = 0.0
    start = len(ctx_ids) - 1
    for i, tok_id in enumerate(ending_ids):
        pos = start + i
        if pos < 0 or pos >= logits_np.shape[0]:
            # Degenerate : ctx empty ; treat as -inf so ending
            # loses the tie.
            total += -1e9
            continue
        row = logits_np[pos]
        m = float(np.max(row))
        logsumexp = m + float(np.log(np.sum(np.exp(row - m)) + 1e-30))
        total += float(row[int(tok_id)]) - logsumexp
    return total


def evaluate_hellaswag(
    model,
    tokenizer,
    *,
    n_samples: int = 100,
    seed: int = 0,
    fixture_path: Path | None = None,
) -> dict[str, float]:
    """Run HellaSwag 4-choice continuation-scoring against ``model``.

    For each record, tokenise ``ctx + ending`` for each of the 4
    candidate endings, compute the summed log-probability of the
    ending tokens, and pick the argmax. Accuracy = fraction of
    records whose argmax ending matches ``label``.

    Returns ``{"accuracy": float, "n": int}``. Deterministic under
    ``(model_weights, tokenizer, fixture, seed)``.
    """
    records = _load_hellaswag_records(
        n_samples, seed, fixture_path=fixture_path
    )
    forward = model.model if hasattr(model, "model") else model

    correct = 0
    # Neutral fallback for whitespace-only endings : prefer
    # ``eos_token_id`` then ``pad_token_id`` ; falling back to ``0``
    # would map to a real vocab token and silently bias the score.
    fallback_token = getattr(tokenizer, "eos_token_id", None)
    if fallback_token is None:
        fallback_token = getattr(tokenizer, "pad_token_id", None)
    for record in records:
        try:
            ctx_ids = tokenizer.encode(record.ctx)
        except TypeError:
            # Both branches must use ``add_special_tokens=False`` —
            # injecting BOS/EOS in the fallback skews the
            # continuation log-prob relative to the primary path.
            ctx_ids = tokenizer.encode(record.ctx, add_special_tokens=False)
        scores: list[float] = []
        for ending in record.endings:
            try:
                end_ids = tokenizer.encode(ending)
            except TypeError:
                end_ids = tokenizer.encode(
                    ending, add_special_tokens=False
                )
            # Ensure the ending contributes at least one token —
            # a tokenizer that collapses whitespace could yield an
            # empty list, which defeats the loglikelihood signal.
            if not end_ids:
                if fallback_token is None:
                    _LOG.warning(
                        "hellaswag: empty ending tokens and no "
                        "eos/pad fallback on tokenizer ; "
                        "defaulting to token id 0 (may bias score)"
                    )
                    end_ids = [0]
                else:
                    _LOG.warning(
                        "hellaswag: ending %r tokenised to empty "
                        "list ; using fallback token id %d",
                        ending, fallback_token,
                    )
                    end_ids = [int(fallback_token)]
            scores.append(
                _continuation_logprob(
                    forward, tokenizer, ctx_ids, end_ids
                )
            )
        pred = int(max(range(4), key=lambda i: scores[i]))
        if pred == record.label:
            correct += 1
    n = len(records)
    return {"accuracy": correct / n if n else 0.0, "n": n}


__all__ = [
    "HellaSwagLoader",
    "HellaSwagRecord",
    "evaluate_hellaswag",
]
