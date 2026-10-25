"""E1 fact-corpus sources: pins, verification, loading, steps 1-4.

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` sections
5.1, 5.1a, 5.1b (task 7: loading the pinned sources).
**Mode** : stdlib only. Reads the three pinned source files from a
local directory (they are downloaded once, outside this module) and
maps every row to a source item ``{source, source_id, date, question,
choices, answer}``.

Directory layout expected under ``sources_dir``::

    realtimeqa/                 git checkout of realtimeqa_public at the pin
    realtimeqa/past/2026/*_qa.jsonl
    daily_oracle/mc_questions_2020-01-01_2026-07-18.csv
    mmlu_cf/val.csv

Row mapping (implementation choices, recorded in the manifest):

- ``realtimeqa``: ``question_sentence`` -> question, ``choices`` as
  given, ``answer`` must be a one-element list holding an integer index
  (a string such as ``"2"`` or an int); anything else has no gold.
  ``question_date`` ``YYYY/MM/DD`` -> ``YYYY-MM-DD``. ``source_id`` is
  the repository-relative path, a colon and the 0-based physical line
  index (blank lines keep their index).
- ``daily_oracle``: ``question``, ``choice_a`` .. ``choice_d``,
  ``answer`` a letter ``a`` .. ``d``; ``source_id`` the 0-based data-row
  index of the CSV as parsed by :mod:`csv`.
- ``mmlu_cf``: ``Question``, ``A`` .. ``D``, ``Answer`` a letter;
  ``source_id`` ``val.csv:<0-based data-row index>``; no date.

A row whose gold cannot be read keeps ``answer = None``; it is excluded
at step 3 (``invalid_gold``) but, if pre-cutoff, still serves as a
contamination reference (question rule only, section 5.1b).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from scripts.e1_corpus_core import CUTOFF, DATED

Item = dict[str, Any]

NOTA = "None of the other choices"
# Rule 3 as widened by the PI on 2026-10-25: the frozen exact list of
# "none of the above"-type options of the pinned MMLU-CF validation set.
# Case-sensitive exact match on the option after rule-2 stripping; no
# other normalisation, no fuzzy match. Applied to mmlu_cf only (rule 3);
# matches in the dated sources are counted, not removed.
NOTA_OPTIONS = frozenset({
    "None of the other choices",
    "None of the mentioned", "None of the Mentioned", "none of the mentioned",
    "None of the above mentioned", "none of the above mentioned",
    "None of the above", "None of the above.", "None of the Above.",
    "none of the above", "none of the above.", "None of above",
    "None of these", "None of These", "None of these.", "none of these", "none of these.",
    "None of the choices", "none of the choices.", "none of the options",
    "None of the above is correct.", "None of the above answers is correct.",
    "None of the above are true", "None of these alternatives is correct.",
    "None of these answer choices are correct.", "None of these answers is correct.",
    "None is applicable",
})
MAX_TOKENS = 256

REALTIMEQA_COMMIT = "e288dcfe1d7818bf48299249ae892939637775db"
REALTIMEQA_GLOB = "past/2026/*_qa.jsonl"
DAILY_ORACLE_FILE = "daily_oracle/mc_questions_2020-01-01_2026-07-18.csv"
DAILY_ORACLE_REVISION = "455b35b2a2415034b05ef91e56c80191fef4a737"
DAILY_ORACLE_SHA256 = "c8c0b5adaa266fa5e9c1ec59b9cd7d9182c89607d39eb0ef2b269cbb12e58178"
DAILY_ORACLE_SIZE = 119_302_323
MMLU_CF_FILE = "mmlu_cf/val.csv"
MMLU_CF_REVISION = "c25b89a968a2062e422dd96ddf0fe3387507f0dd"
MMLU_CF_SHA256 = "28816045197decd701b1a0e51923b43684d708d8c06ec62d20541f90939ac49f"
MMLU_CF_SIZE = 2_247_391
# Section 5.1 property 4 (PI decision 2026-10-25): MMLU-CF items carry no
# date; their provenance date is the release date of the pinned revision
# (commit c25b89a, 2025-01-08T08:01:02Z), labelled as a source date, not an
# item creation date.
MMLU_CF_RELEASE_DATE = "2025-01-08"

# Step-3 tokenizer, pinned by the PI on 2026-10-25 (E1 draft section 10 item 1).
MODEL_REPO = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"

_DATE_RE = re.compile(r"^(\d{4})[-/](\d{2})[-/](\d{2})$")


class PinError(RuntimeError):
    """A source file does not match its registered pin."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_date(raw: Any) -> str | None:
    m = _DATE_RE.match(str(raw or "").strip())
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _letter(raw: Any, letters: str) -> int | None:
    s = str(raw or "").strip()
    return letters.index(s) if len(s) == 1 and s in letters else None


def _rtqa_answer(raw: Any) -> int | None:
    if not isinstance(raw, list) or len(raw) != 1:
        return None
    v = raw[0]
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


# --- step 1: pins and loading -------------------------------------------------


def git_blob_sha1(path: Path) -> str:
    """Git object id of a file's bytes (``git hash-object``)."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def git_tree_blobs(root: Path, commit: str) -> dict[str, str]:
    """``{path: blob id}`` of the ``*_qa.jsonl`` files of ``past/2026`` at ``commit``."""
    import subprocess

    out = subprocess.run(["git", "-C", str(root), "ls-tree", "-r", commit, "past/2026"],
                         check=True, capture_output=True, text=True).stdout
    blobs = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        if Path(path).match(REALTIMEQA_GLOB):
            blobs[path] = meta.split()[2]
    return blobs


def verify_pins(
    sources_dir: Path,
    tree_blobs: Callable[[Path, str], dict[str, str]] = git_tree_blobs,
) -> dict[str, Any]:
    """Check every pin; return the raw-file record for the manifest.

    RealTime QA is verified against the pinned commit itself: the set
    of ``past/2026/*_qa.jsonl`` files and each file's git blob id must
    equal those of commit ``REALTIMEQA_COMMIT`` (``tree_blobs`` is
    injected for tests). The two CSV files are verified by SHA-256 and
    size. Every raw file's SHA-256 is recorded.
    """
    rec: dict[str, Any] = {"realtimeqa": {}, "daily_oracle": {}, "mmlu_cf": {}}
    for name, rel, sha, size, rev in (
        ("daily_oracle", DAILY_ORACLE_FILE, DAILY_ORACLE_SHA256, DAILY_ORACLE_SIZE,
         DAILY_ORACLE_REVISION),
        ("mmlu_cf", MMLU_CF_FILE, MMLU_CF_SHA256, MMLU_CF_SIZE, MMLU_CF_REVISION),
    ):
        p = sources_dir / rel
        if not p.is_file():
            raise PinError(f"{name}: missing {p}")
        got = sha256_file(p)
        if got != sha or p.stat().st_size != size:
            raise PinError(f"{name}: {rel} sha256 {got} / size {p.stat().st_size} "
                           f"!= pin {sha} / {size}")
        rec[name] = {"revision": rev, "file": rel, "sha256": got, "size": size}
    root = sources_dir / "realtimeqa"
    files = sorted(root.glob(REALTIMEQA_GLOB))
    if not files:
        raise PinError(f"realtimeqa: no {REALTIMEQA_GLOB} under {root}")
    local = {f.relative_to(root).as_posix(): f for f in files}
    pinned = tree_blobs(root, REALTIMEQA_COMMIT)
    if set(local) != set(pinned):
        raise PinError(f"realtimeqa: file set differs from commit {REALTIMEQA_COMMIT}: "
                       f"extra {sorted(set(local) - set(pinned))}, "
                       f"missing {sorted(set(pinned) - set(local))}")
    for rel, f in local.items():
        if git_blob_sha1(f) != pinned[rel]:
            raise PinError(f"realtimeqa: {rel} differs from commit {REALTIMEQA_COMMIT}")
    rec["realtimeqa"] = {
        "commit": REALTIMEQA_COMMIT,
        "files": {rel: {"sha256": sha256_file(f), "git_blob": pinned[rel],
                        "size": f.stat().st_size} for rel, f in local.items()},
    }
    return rec


def load_realtimeqa(root: Path) -> list[Item]:
    out: list[Item] = []
    for f in sorted(root.glob(REALTIMEQA_GLOB)):
        rel = f.relative_to(root).as_posix()
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            out.append({
                "source": "realtimeqa", "source_id": f"{rel}:{i}",
                "date": _norm_date(r.get("question_date")),
                "question": str(r.get("question_sentence") or ""),
                "choices": [str(c) for c in r.get("choices") or []],
                "answer": _rtqa_answer(r.get("answer")),
            })
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(1 << 30)
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_daily_oracle(path: Path) -> list[Item]:
    out: list[Item] = []
    for i, r in enumerate(_read_csv(path)):
        out.append({
            "source": "daily_oracle", "source_id": str(i),
            "date": _norm_date(r.get("date")),
            "question": r.get("question") or "",
            "choices": [r.get(f"choice_{c}") or "" for c in "abcd"],
            "answer": _letter(r.get("answer"), "abcd"),
        })
    return out


def load_mmlu_cf(path: Path) -> list[Item]:
    out: list[Item] = []
    for i, r in enumerate(_read_csv(path)):
        out.append({
            "source": "mmlu_cf", "source_id": f"val.csv:{i}", "date": None,
            "question": r.get("Question") or "",
            "choices": [r.get(c) or "" for c in "ABCD"],
            "answer": _letter(r.get("Answer"), "ABCD"),
        })
    return out


def load_all(sources_dir: Path) -> list[Item]:
    return (load_realtimeqa(sources_dir / "realtimeqa")
            + load_daily_oracle(sources_dir / DAILY_ORACLE_FILE)
            + load_mmlu_cf(sources_dir / MMLU_CF_FILE))


# --- step 2: normalisation ----------------------------------------------------


def normalise(item: Item) -> Item:
    """Rule 2: strip the question and each option (no other change)."""
    return {**item, "question": item["question"].strip(),
            "choices": [c.strip() for c in item["choices"]]}


# --- step 3: exclusion --------------------------------------------------------

TokenCounter = Callable[[str, str], "int | None"]


def byte_bound_counter(question: str, gold: str) -> int | None:
    """Exact-or-unknown count from a guaranteed bound.

    A byte-level BPE emits at least one UTF-8 byte per token, so the
    token count of ``question`` and ``gold`` (separately, or joined by
    one space) never exceeds their UTF-8 byte length plus one, taken as
    the larger of the raw text and its NFC form (the tokenizer may
    normalise). Returns 0 ("certainly within") when that bound is <= 256,
    else None (undecided). A build counted this way is always
    PROVISIONAL-TOKENIZER: step 3 needs the pinned tokenizer.
    """
    def nbytes(t: str) -> int:
        return max(len(t.encode("utf-8")), len(unicodedata.normalize("NFC", t).encode("utf-8")))

    nb = nbytes(question) + 1 + nbytes(gold)
    return 0 if nb <= MAX_TOKENS else None


def nota_variants(item: Item) -> list[str]:
    """Distinct NOTA_OPTIONS strings among an item's options, in option order."""
    out: list[str] = []
    for c in item["choices"]:
        if c in NOTA_OPTIONS and c not in out:
            out.append(c)
    return out


def gold_of(item: Item) -> str | None:
    ch, a = item.get("choices"), item.get("answer")
    if not isinstance(ch, list) or isinstance(a, bool) or not isinstance(a, int):
        return None
    return str(ch[a]) if 0 <= a < len(ch) else None


def exclusion_reason(item: Item, count: TokenCounter) -> str | None:
    """Step 3 reason (first that applies), ``"undecided_tokens"`` or None."""
    ch = item["choices"]
    if len(ch) != 4:
        return "not_4_choices"
    if not item["question"] or any(not c for c in ch):
        return "empty_field"
    if item["source"] in DATED and item["date"] is None:
        return "empty_field"
    gold = gold_of(item)
    if gold is None:
        return "invalid_gold"
    if item["source"] == "mmlu_cf" and any(c in NOTA_OPTIONS for c in ch):
        return "nota_option"
    n = count(item["question"], gold)
    if n is None:
        return "undecided_tokens"
    if n > MAX_TOKENS:
        return "over_256_tokens"
    return None


# --- step 4: date -------------------------------------------------------------


def split_by_date(items: Iterable[Item]) -> tuple[list[Item], list[Item], list[Item]]:
    """(post-cutoff or undated, pre-cutoff dated, dated without a date)."""
    keep, pre, nodate = [], [], []
    for it in items:
        if it["source"] not in DATED:
            keep.append(it)
        elif it["date"] is None:
            nodate.append(it)
        elif it["date"] >= CUTOFF:
            keep.append(it)
        else:
            pre.append(it)
    return keep, pre, nodate


# --- reference rows (pilot fixtures, mega_v2 facts) ---------------------------


def reference_rows(fixtures_dir: Path, mega_v2: Iterable[dict[str, Any]]) -> list[Item]:
    """Pilot fixtures and the hand-authored mega_v2 facts as items.

    MMLU: question / choices / answer. HellaSwag: ``ctx`` as question,
    ``endings`` / ``label``. mega_v2: ``context`` as question and the
    stripped ``expected`` as the only choice (gold index 0).
    """
    out: list[Item] = []
    for line in (fixtures_dir / "mmlu_real.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.append({"source": "ref:mmlu_real", "question": str(r["question"]).strip(),
                        "choices": [str(c).strip() for c in r["choices"]],
                        "answer": int(r["answer"])})
    for line in (fixtures_dir / "hellaswag_sanity.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.append({"source": "ref:hellaswag_sanity", "question": str(r["ctx"]).strip(),
                        "choices": [str(c).strip() for c in r["endings"]],
                        "answer": int(r["label"])})
    for r in mega_v2:
        out.append({"source": "ref:mega_v2", "question": str(r["context"]).strip(),
                    "choices": [str(r["expected"]).strip()], "answer": 0})
    return out
