"""Real benchmark dataset registry + loaders for cycle-3.

Exposes :

- :mod:`harness.real_benchmarks.dataset_registry` — SHA-pinned
  dataset entries (MMLU full + HellaSwag) per cycle-3 spec §5
  pre-cycle-3 lock #3.
- :mod:`harness.real_benchmarks.mmlu` — MMLU 5-shot loader.
- :mod:`harness.real_benchmarks.hellaswag` — HellaSwag zero-shot
  loader.
- :mod:`harness.real_benchmarks.mega_v2_eval` — mega-v2 80/20
  self-eval split loader.

Design principles (cycle-3 C3.1) :

- **Network-free by default** : each loader reads from an explicit
  ``local_path`` ; missing path raises
  :class:`MissingLocalDatasetError` rather than fetching from HF.
- **SHA-256 integrity** : every loader exposes the local-file
  digest and enforces ``expected_sha256`` when the caller supplies
  one — R1 contract (``(c_version, profile, seed, commit_sha)
  → run_id``) guarantees byte-identical benchmark input.
- **Typed records** : loaders emit frozen dataclasses so downstream
  consumers (dream-ops, run registry, real-weight ops) pin the
  schema at import time.
"""
from __future__ import annotations


class MissingLocalDatasetError(FileNotFoundError):
    """Raised when a loader is pointed at a non-existent local path.

    Explicit subclass of :class:`FileNotFoundError` so callers can
    catch the narrow case without accidentally swallowing generic IO
    errors from elsewhere in the harness.
    """


from harness.real_benchmarks.dataset_registry import (  # noqa: E402
    DATASET_REGISTRY,
    DatasetPin,
    get_dataset_pin,
    verify_all_datasets,
)

__all__ = [
    "DATASET_REGISTRY",
    "DatasetPin",
    "MissingLocalDatasetError",
    "get_dataset_pin",
    "verify_all_datasets",
]
