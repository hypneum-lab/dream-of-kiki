"""Eval matrix configuration loader.

Parses docs/interfaces/eval-matrix.yaml and exposes the stratification
rules as a typed EvalMatrix dataclass for the harness dispatcher.

Reference: docs/interfaces/eval-matrix.yaml
         : docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §8.2
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvalMatrix:
    """Parsed eval-matrix.yaml with typed accessors."""

    version: str
    bump_rules: dict[str, dict[str, Any]]
    publication_ready_gate: dict[str, Any]
    metrics: dict[str, dict[str, Any]]


def load_eval_matrix(path: Path) -> EvalMatrix:
    """Load and validate eval-matrix.yaml."""
    if not path.exists():
        raise FileNotFoundError(f"eval-matrix.yaml not found at {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(
            f"eval-matrix.yaml must be a YAML mapping, "
            f"got {type(raw).__name__}"
        )

    required_top_keys = {
        "version",
        "bump_rules",
        "publication_ready_gate",
        "metrics",
    }
    missing = required_top_keys - raw.keys()
    if missing:
        raise ValueError(
            f"eval-matrix.yaml missing top-level keys: {missing}"
        )

    return EvalMatrix(
        version=raw["version"],
        bump_rules=raw["bump_rules"],
        publication_ready_gate=raw["publication_ready_gate"],
        metrics=raw["metrics"],
    )
