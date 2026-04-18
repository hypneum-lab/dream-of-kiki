"""mega-v2 dataset loader bridge.

Adapter that produces a `RetainedBenchmark` (compatible with
`evaluate_retained`) from either:
- the real mega-v2 dataset (498K examples, 25 domains) at a given
  filesystem path
- a synthetic placeholder (20 items/domain × 25 domains = 500)
  when real path is unavailable

The synthetic fallback ensures cycle-1 tests can run without
external dataset access. Real path integration arrives once
dataset access is finalized (S13+ in calendar terms).

Reference: docs/specs/2026-04-17-dreamofkiki-master-design.md §5.2
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from harness.benchmarks.retained.retained import RetainedBenchmark


# 25 mega-v2 domain names (synthetic placeholder; real domain
# taxonomy may differ slightly when real dataset is integrated).
SYNTHETIC_DOMAINS: tuple[str, ...] = (
    "arithmetic", "syntax", "lexical", "phonology", "semantic",
    "pragmatic", "discourse", "narrative", "dialog", "qa",
    "summarization", "translation", "reasoning", "math", "code",
    "physics", "biology", "chemistry", "history", "geography",
    "philosophy", "law", "medicine", "engineering", "art",
)


class MegaV2NotAvailable(Exception):
    """Raised when real mega-v2 path is missing and fallback is
    explicitly disabled."""


def _generate_synthetic_items(
    items_per_domain: int, seed: int
) -> list[dict]:
    """Generate stratified synthetic items.

    items_per_domain × len(SYNTHETIC_DOMAINS) total. Each item:
    {id, context, expected, domain}. Deterministic with seed.
    """
    rng = random.Random(seed)
    items: list[dict] = []
    for domain_idx, domain in enumerate(SYNTHETIC_DOMAINS):
        for i in range(items_per_domain):
            global_idx = domain_idx * items_per_domain + i
            items.append({
                "id": f"mv2-syn-{global_idx:05d}",
                "context": (
                    f"synthetic context {global_idx} domain {domain} "
                    f"variant {rng.randint(0, 999)}"
                ),
                "expected": f"expected-{global_idx:05d}",
                "domain": domain,
            })
    return items


def _load_real(path: Path, items_per_domain: int) -> list[dict]:
    """Load and stratify real mega-v2.

    Real schema assumed: JSONL with at least {id, context,
    expected, domain} fields per line. Stratification: keep first
    N items per encountered domain.
    """
    by_domain: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            domain = row.get("domain", "unknown")
            bucket = by_domain.setdefault(domain, [])
            if len(bucket) < items_per_domain:
                bucket.append(row)
    items: list[dict] = []
    for domain in sorted(by_domain.keys()):
        items.extend(by_domain[domain])
    return items


def load_megav2_stratified(
    real_path: Path | None = None,
    items_per_domain: int = 20,
    synthetic_seed: int = 42,
    explicit_fallback: bool = True,
) -> RetainedBenchmark:
    """Load mega-v2 stratified subset as RetainedBenchmark.

    Args:
        real_path: filesystem path to real mega-v2 JSONL. If None
            or missing AND explicit_fallback=True, generate
            synthetic placeholder. If missing AND
            explicit_fallback=False, raise MegaV2NotAvailable.
        items_per_domain: stratification count per domain.
        synthetic_seed: RNG seed for synthetic generation.
        explicit_fallback: opt-in fallback behavior.

    Returns:
        RetainedBenchmark with stratified items, hash_verified=True
        (computed over the stringified JSON).
    """
    if real_path is not None and real_path.exists():
        items = _load_real(real_path, items_per_domain)
        provenance = "real"
    elif explicit_fallback:
        items = _generate_synthetic_items(
            items_per_domain, synthetic_seed
        )
        provenance = "synthetic"
    else:
        raise MegaV2NotAvailable(
            f"real mega-v2 path missing: {real_path!r} and "
            f"explicit_fallback=False"
        )

    payload = json.dumps(items, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    return RetainedBenchmark(
        items=items,
        hash_verified=True,
        source_hash=f"{provenance}:{digest}",
    )
