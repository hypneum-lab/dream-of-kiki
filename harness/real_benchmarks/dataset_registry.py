"""Real-benchmark dataset registry — pre-cycle-3 lock #3.

Pins MMLU + HellaSwag at HuggingFace revision SHA granularity so
that every cycle-3 run recording
``(c_version, profile, seed, commit_sha) -> run_id`` maps to a
byte-identical question set per R1.

SHA values obtained from HuggingFace metadata at 2026-04-19 via
``GET /api/datasets/{repo_id}``. License fields follow the HF
``cardData.license`` value when present ; if the HF card is
silent, the upstream author's license (arXiv + GitHub release)
is used and annotated in ``notes``.

Usage parity with ``harness.real_models.base_model_registry`` :
``get_dataset_pin(name)`` returns a frozen ``DatasetPin`` and
``verify_all_datasets()`` runs local self-consistency checks
(network-free by default).

Cycle-3 spec §4 H5 treats benchmark outputs as the primary
empirical axis ; an unpinned dataset would let the HF curator
silently reshape the question set between Phase 1 Studio runs
and any downstream re-run, invalidating R1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SHA40_RE = re.compile(r"^[a-f0-9]{40}$")


@dataclass(frozen=True)
class DatasetPin:
    """Immutable pin for a real-benchmark dataset.

    Attributes
    ----------
    name
        Canonical registry key (e.g. ``mmlu``, ``hellaswag``).
    hf_repo_id
        HuggingFace ``org/repo`` identifier (datasets namespace).
    revision_sha
        40-char HuggingFace dataset revision (git) SHA.
    subset
        Sub-configuration name when the dataset ships multiple
        configs (e.g. MMLU ``all`` vs per-subject). ``None`` =
        default config.
    n_examples
        Count of examples in the active split(s) as of pin time.
        Used for sanity checks during loader integration (cycle-3
        C3.3 deliverable).
    protocol
        Evaluation protocol string (e.g. ``5-shot``, ``zero-shot``).
    license_id
        SPDX-style license identifier (e.g. ``MIT``,
        ``CC-BY-SA-4.0``). Kept as a string for freedom to
        encode composite licenses.
    notes
        Free-text provenance + protocol notes.
    """

    name: str
    hf_repo_id: str
    revision_sha: str
    subset: str | None
    n_examples: int
    protocol: str
    license_id: str
    notes: str = ""


# Pin values recorded 2026-04-19 from HuggingFace API.
# Method :
#   curl -s https://huggingface.co/api/datasets/{repo_id}
#       -> .sha, .cardData.license, .cardData.dataset_info
# Re-run verify_all_datasets() after any pin update.

DATASET_REGISTRY: dict[str, DatasetPin] = {
    "mmlu": DatasetPin(
        name="mmlu",
        hf_repo_id="cais/mmlu",
        revision_sha="c30699e8356da336a370243923dbaf21066bb9fe",
        subset="all",
        n_examples=14042,
        protocol="5-shot",
        license_id="MIT",
        notes=(
            "Massive Multitask Language Understanding, Hendrycks "
            "et al. 2020 (arXiv:2009.03300). 'all' config "
            "aggregates 57 subjects, test split 14042 examples. "
            "5-shot standard per original paper + Open LLM "
            "Leaderboard protocol ; dev split (285 rows) "
            "supplies the canonical few-shot exemplars."
        ),
    ),
    "hellaswag": DatasetPin(
        name="hellaswag",
        hf_repo_id="Rowan/hellaswag",
        revision_sha="218ec52e09a7e7462a5400043bb9a69a41d06b76",
        subset=None,
        n_examples=10042,
        protocol="zero-shot",
        license_id="MIT",
        notes=(
            "HellaSwag commonsense NLI, Zellers et al. 2019 "
            "(arXiv:1905.07830). HF card does not set a license ; "
            "upstream repository (rowanz/hellaswag) ships MIT. "
            "Protocol is zero-shot per Open LLM Leaderboard + "
            "lm-evaluation-harness defaults ; scored on the "
            "'validation' split (10042 rows) because the 'test' "
            "labels are held-out for the public leaderboard."
        ),
    ),
}


def get_dataset_pin(name: str) -> DatasetPin:
    """Return the :class:`DatasetPin` for registry key ``name``.

    Raises :class:`KeyError` with an available-keys hint if the
    key is not registered.
    """
    if name not in DATASET_REGISTRY:
        available = sorted(DATASET_REGISTRY.keys())
        raise KeyError(
            f"no dataset pinned for key {name!r} ; "
            f"available : {available}"
        )
    return DATASET_REGISTRY[name]


def verify_all_datasets(live: bool = False) -> dict[str, bool]:
    """Validate pin self-consistency for every registered dataset.

    Local-only checks (always run, network-free) :

    - ``revision_sha`` is a 40-char lowercase hex string ;
    - ``n_examples`` is strictly positive ;
    - ``hf_repo_id`` is non-empty and contains exactly one ``/`` ;
    - ``protocol`` is non-empty ;
    - ``license_id`` is non-empty.

    Live HTTP check (opt-in via ``live=True``) : fetches
    ``https://huggingface.co/api/datasets/{repo_id}`` and compares
    the returned ``sha`` field against the recorded
    ``revision_sha``. Kept opt-in so test runs stay deterministic
    and offline-friendly per R1.

    Returns a mapping ``name -> bool``.
    """
    results: dict[str, bool] = {}
    for name, pin in DATASET_REGISTRY.items():
        ok = True
        if not _SHA40_RE.match(pin.revision_sha):
            ok = False
        if pin.n_examples <= 0:
            ok = False
        if not pin.hf_repo_id or pin.hf_repo_id.count("/") != 1:
            ok = False
        if not pin.protocol:
            ok = False
        if not pin.license_id:
            ok = False
        if ok and live:
            ok = _verify_live_dataset(pin)
        results[name] = ok
    return results


def _verify_live_dataset(pin: DatasetPin) -> bool:
    """Fetch HF API and confirm ``revision_sha`` still matches.

    Lazy import of :mod:`urllib` keeps the module offline-safe at
    import time. Any network or parse failure returns ``False`` so
    the caller treats the pin as "needs re-pin".
    """
    try:  # pragma: no cover - network path
        from urllib.request import urlopen
        import json

        url = f"https://huggingface.co/api/datasets/{pin.hf_repo_id}"
        with urlopen(url, timeout=10) as resp:  # noqa: S310
            payload = json.load(resp)
        return bool(payload.get("sha") == pin.revision_sha)
    except Exception:  # pragma: no cover - network path
        return False
