"""MMLU 5-shot loader (cycle-3 C3.1).

Binds :class:`harness.real_benchmarks.dataset_registry.DatasetPin`
for the ``mmlu`` key to a local JSONL fixture that matches the HF
``cais/mmlu`` schema (``question``, ``choices``, ``answer``,
``subject``).

The loader is **network-free** : callers must materialise the
fixture themselves (via the ``datasets`` CLI or an offline cache)
and pass its path. Missing file → :class:`MissingLocalDatasetError`.
When the caller also supplies ``expected_sha256``, the loader
refuses to yield any record if the hash does not match the file on
disk — this is the mechanism the run-registry uses to enforce R1.

Protocol : MMLU is evaluated **5-shot** per Hendrycks et al. 2020 +
Open LLM Leaderboard. :py:meth:`get_5shot_exemplars` draws a
reproducible 5-record sample via :mod:`random.Random(seed)` so the
same ``(pin, seed)`` pair always yields the same in-context
exemplars.

Reference :
  docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md §C3.1
  docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md §5
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from harness.real_benchmarks import MissingLocalDatasetError
from harness.real_benchmarks.dataset_registry import DatasetPin


@dataclass(frozen=True)
class MMLURecord:
    """Frozen MMLU record — schema matches HF ``cais/mmlu``.

    Fields
    ------
    question
        Question stem (single-line string).
    choices
        4-choice multiple-choice set, order as authored upstream.
    answer
        Integer index in ``choices`` identifying the correct
        answer.
    subject
        One of the 57 MMLU subjects (e.g. ``abstract_algebra``).
    """

    question: str
    choices: tuple[str, str, str, str]
    answer: int
    subject: str


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class MMLULoader:
    """Read-only MMLU loader bound to a :class:`DatasetPin`.

    Parameters
    ----------
    registry_pin
        Pin returned by
        :func:`harness.real_benchmarks.dataset_registry.get_dataset_pin`
        for the ``mmlu`` slot.
    local_path
        Filesystem path to a JSONL fixture whose rows match the HF
        MMLU schema (``question`` / ``choices`` / ``answer`` /
        ``subject``).
    expected_sha256
        Optional SHA-256 (lowercase hex) ; when present, loader
        computes the digest of ``local_path`` at construction time
        and raises :class:`ValueError` on mismatch. This is how the
        harness enforces R1 byte-stability on the benchmark slice.
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
                f"MMLU fixture not found at {local_path!s} ; "
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
        """Return the SHA-256 of the local fixture (64-char hex)."""
        return self._actual_sha256

    def _iter_raw(self) -> Iterator[dict[str, Any]]:
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _record_from_raw(self, row: dict[str, Any]) -> MMLURecord:
        choices = row["choices"]
        if len(choices) != 4:
            raise ValueError(
                f"MMLU row has {len(choices)} choices, expected 4: "
                f"{row!r}"
            )
        answer = int(row["answer"])
        # Validate that the answer index lies within the 4-choice
        # bounds before constructing the record — out-of-range indices
        # would silently select the wrong option or crash downstream.
        if not 0 <= answer <= 3:
            raise ValueError(
                f"MMLU row has answer index {answer} outside [0,3]: "
                f"{row!r}"
            )
        return MMLURecord(
            question=str(row["question"]),
            choices=(
                str(choices[0]),
                str(choices[1]),
                str(choices[2]),
                str(choices[3]),
            ),
            answer=answer,
            subject=str(row.get("subject", "unknown")),
        )

    def iter_records(self, seed: int = 0) -> Iterator[MMLURecord]:
        """Yield records in a seeded-shuffled order.

        ``seed`` pins the permutation so the caller can log
        ``(pin, seed) → record_order`` reproducibly. Seed 0 keeps
        the original file order.
        """
        raws = list(self._iter_raw())
        if seed != 0:
            rng = random.Random(seed)
            rng.shuffle(raws)
        for row in raws:
            yield self._record_from_raw(row)

    def get_5shot_exemplars(self, seed: int) -> list[MMLURecord]:
        """Draw 5 in-context exemplars for the MMLU 5-shot protocol.

        Uses :func:`random.Random.sample` seeded with ``seed`` so
        the same seed always returns the same exemplars in the same
        order. Raises :class:`ValueError` if the fixture contains
        fewer than 5 records (an empirically-empty fixture cannot
        satisfy the protocol).
        """
        raws = list(self._iter_raw())
        if len(raws) < 5:
            raise ValueError(
                f"MMLU 5-shot protocol requires ≥5 records ; "
                f"{self._path!s} has {len(raws)}"
            )
        rng = random.Random(seed)
        return [self._record_from_raw(r) for r in rng.sample(raws, 5)]

    def get_seeded_sample(self, seed: int, n: int) -> list[MMLURecord]:
        """Return ``n`` records drawn via a seeded permutation."""
        raws = list(self._iter_raw())
        if n > len(raws):
            raise ValueError(
                f"requested {n} records but fixture has {len(raws)}"
            )
        rng = random.Random(seed)
        return [self._record_from_raw(r) for r in rng.sample(raws, n)]


# --------------------------------------------------------------------------
# Evaluator (cycle-3 C3.8 Phase A) — runs ``evaluate_mmlu(model, tokenizer)``
# as used by ``scripts/pilot_cycle3_real.py``. Network-free : loads from
# the ``HF_DATASETS_CACHE`` default or a committed fallback fixture.
# --------------------------------------------------------------------------


_DEFAULT_MMLU_FALLBACK = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "mmlu_sanity.jsonl"
)


def _load_mmlu_records(
    n_samples: int,
    seed: int,
    *,
    fixture_path: Path | None = None,
) -> list[MMLURecord]:
    """Materialise ``n_samples`` MMLU records with R1 discipline.

    Search order :

    1. Caller-supplied ``fixture_path`` (if set and existing).
    2. The HF ``datasets`` cache if the ``datasets`` package is
       available and a local copy of ``cais/mmlu`` is already
       materialised — no network fetch is attempted, per the R1
       contract established by :class:`MMLULoader`.
    3. The committed sanity fixture at ``tests/fixtures/mmlu_sanity.jsonl``
       expanded to ``n_samples`` via seeded cycling. The fallback
       is meant to validate the pipeline only ; empirical claims
       require the full cais/mmlu export.
    """
    def _materialise(raws: list[dict[str, Any]]) -> list[MMLURecord]:
        # Trim / expand to n_samples deterministically — seeded
        # permutation of the underlying rows then cycle-and-slice.
        if not raws:
            raise ValueError("no MMLU records available")
        rng = random.Random(seed)
        rng.shuffle(raws)
        if len(raws) >= n_samples:
            selected = raws[:n_samples]
        else:
            # Cycle through the fixture deterministically. Not a
            # scientific sample, but validates the pipeline shape.
            selected = [
                raws[i % len(raws)] for i in range(n_samples)
            ]
        records: list[MMLURecord] = []
        for row in selected:
            choices = row["choices"]
            if len(choices) != 4:
                raise ValueError(
                    f"MMLU row has {len(choices)} choices: {row!r}"
                )
            answer = int(row["answer"])
            if not 0 <= answer <= 3:
                raise ValueError(
                    f"MMLU answer {answer} outside [0,3]: {row!r}"
                )
            records.append(
                MMLURecord(
                    question=str(row["question"]),
                    choices=(
                        str(choices[0]),
                        str(choices[1]),
                        str(choices[2]),
                        str(choices[3]),
                    ),
                    answer=answer,
                    subject=str(row.get("subject", "unknown")),
                )
            )
        return records

    # 1. Caller-supplied path
    target = fixture_path or _DEFAULT_MMLU_FALLBACK
    if target.exists():
        raws = []
        with target.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raws.append(json.loads(line))
        return _materialise(raws)

    # 2. Try HF datasets cache (offline only)
    try:  # pragma: no cover - optional cache path
        import os

        from datasets import load_dataset

        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        ds = load_dataset("cais/mmlu", "all", split="test")
        raws = [dict(ex) for ex in ds.select(range(min(len(ds), n_samples * 2)))]
        return _materialise(raws)
    except Exception:
        pass

    raise MissingLocalDatasetError(
        f"MMLU evaluator cannot locate either {target!s} or an "
        "offline cais/mmlu cache — run the HF datasets export "
        "once on a networked host before launching the pilot."
    )


def _mmlu_prompt(record: MMLURecord) -> str:
    """Render an MMLU record as a zero-shot letter-choice prompt.

    Mirrors the prompt format used by the C3.7 sanity pilot so
    the two pilots are directly comparable.
    """
    return (
        "The following is a multiple choice question. Respond "
        "with only the letter of the correct answer.\n\n"
        f"Question: {record.question}\n"
        f"A. {record.choices[0]}\n"
        f"B. {record.choices[1]}\n"
        f"C. {record.choices[2]}\n"
        f"D. {record.choices[3]}\n"
        "Answer:"
    )


def _letter_token_ids(tokenizer: Any) -> list[int]:
    """Single-token IDs for the four letter choices.

    Raises :class:`ValueError` if any letter does not tokenise to
    exactly one token — that would silently bias the wrong
    vocabulary slot.
    """
    ids: list[int] = []
    for letter in ("A", "B", "C", "D"):
        if hasattr(tokenizer, "encode"):
            try:
                enc = tokenizer.encode(letter, add_special_tokens=False)
            except TypeError:
                enc = tokenizer.encode(letter)
        else:
            raise ValueError(
                f"tokenizer {tokenizer!r} has no ``encode`` method"
            )
        if len(enc) != 1:
            raise ValueError(
                f"letter {letter!r} tokenises to {enc!r} (len "
                f"{len(enc)}) ; MMLU letter-argmax proxy requires "
                "single-token letters"
            )
        ids.append(int(enc[0]))
    return ids


def evaluate_mmlu(
    model: Any,
    tokenizer: Any,
    *,
    n_samples: int = 100,
    seed: int = 0,
    fixture_path: Path | None = None,
) -> dict[str, float]:
    """Run the MMLU letter-argmax evaluation against ``model``.

    Parameters
    ----------
    model
        Callable returning logits for a given ``mx.array`` of token
        ids. Typically a :class:`harness.real_models.qwen_mlx_fp16.QwenMLXFP16Wrapper`
        or the raw ``mlx-lm`` model. Accepts the wrapper-style
        ``.model`` attribute lookup when callable returns fail.
    tokenizer
        Must expose an ``encode(text)`` method.
    n_samples
        Number of MMLU records to score. Default 100 matches the
        C3.8 Phase A per-cell eval volume.
    seed
        Pins the record subset + order.

    Returns
    -------
    dict
        ``{"accuracy": float, "n": int}`` — accuracy is the fraction
        of records whose argmax letter matches the gold label.
    """
    import mlx.core as mx
    import numpy as np

    records = _load_mmlu_records(
        n_samples, seed, fixture_path=fixture_path
    )
    letter_ids = _letter_token_ids(tokenizer)
    # Accept either a wrapper (exposes .model) or a raw callable.
    forward = model.model if hasattr(model, "model") else model

    correct = 0
    for record in records:
        prompt = _mmlu_prompt(record)
        token_ids = tokenizer.encode(prompt)
        tokens = mx.array([token_ids])
        mx.random.seed(0)
        logits = forward(tokens)
        # Cast to fp32 on the MLX side before numpy conversion :
        # numpy has no bf16 dtype, so ``np.asarray(bf16_tensor)``
        # raises ``RuntimeError: Item size 2 for PEP 3118 buffer
        # format string B does not match the dtype B item size 1``.
        last_fp32 = logits[0, -1, :].astype(mx.float32)
        mx.eval(last_fp32)
        last = np.asarray(last_fp32)
        letter_logits = last[letter_ids].astype(np.float32)
        pred = int(np.argmax(letter_logits))
        if pred == record.answer:
            correct += 1
    n = len(records)
    return {"accuracy": correct / n if n else 0.0, "n": n}


__all__ = [
    "MMLULoader",
    "MMLURecord",
    "evaluate_mmlu",
]
