"""mega-v2 → DreamEpisode adapter (cycle-3 C3.3).

Bridges the mega-v2 80/20 eval loader (C3.1) with the dream-episode
lifecycle : a :class:`MegaV2EvalRecord` from the mega-v2 eval shard
becomes a typed :class:`DreamEpisode` carrying both α-channel raw
tokens (when requested) and β-channel curated embedding records
ready for the real-weight replay op (C3.3).

Design principles :

- **DR-0 accountability** : ``episode_id = f"de-{record.id}"`` so
  the run registry can correlate every episode back to its source
  mega-v2 record.
- **R1 reproducibility** : same ``(record, encoder, tokenizer)`` →
  byte-identical ``DreamEpisode`` content. The encoder fixture used
  in tests is deterministic ; production callers inject encoders
  whose outputs are R1-stable under the harness seeds.
- **Lossless round-trip** : the adapter stashes the original record
  fields in ``input_slice["_record"]`` so :func:`dream_episode_to_record`
  can reconstruct a :class:`MegaV2EvalRecord` without ambiguity.

Reference :
  docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md §C3.3
  docs/superpowers/specs/2026-04-17-dreamofkiki-framework-C-design.md §2.1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from harness.real_benchmarks.mega_v2_eval import MegaV2EvalRecord
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)


# Default budget for mega-v2-sourced episodes. Small flop envelope
# since each β-record drives a single MSE gradient step at 4-dim.
# Tests do not assert on these numbers — they are the K1 accounting
# seed for downstream scheduler work.
_DEFAULT_FLOPS = 10_000
_DEFAULT_WALL = 0.05
_DEFAULT_ENERGY = 0.001


def _to_float_list(values: Iterable[Any]) -> list[float]:
    """Coerce encoder output (numpy / list / tuple) to list[float]."""
    try:
        tolist = values.tolist  # numpy / mlx surface
    except AttributeError:
        return [float(v) for v in values]
    materialised = tolist()
    if isinstance(materialised, list):
        return [float(v) for v in materialised]
    return [float(materialised)]


def _pad_expected(tokens: list[int], width: int = 2) -> list[float]:
    """Pad / truncate tokenized `expected` to a fixed-width target.

    The adapter emits a tiny ``y`` label vector so downstream replay
    ops can compute MSE without reasoning about variable-length
    sequences. Width 2 matches the _TinyMLP test fixture.
    """
    if not tokens:
        return [0.0] * width
    trimmed = tokens[:width]
    while len(trimmed) < width:
        trimmed.append(0)
    return [float(t) for t in trimmed]


@dataclass
class MegaV2Adapter:
    """Convert :class:`MegaV2EvalRecord` → :class:`DreamEpisode`.

    Parameters
    ----------
    tokenizer
        Callable ``str -> list[int]``. Used for both α-channel raw
        tokens (when ``emit_alpha=True``) and ``y`` label tokens.
    encoder
        Callable ``list[int] -> array-like`` of 4 floats. Drives the
        β-channel ``x`` embedding for each replay record.
    emit_alpha
        If True the produced :class:`DreamEpisode` carries the
        tokenized context under ``input_slice["alpha_tokens"]``
        (tuple[int, ...]) — α-channel raw-trace primitive per
        framework-C §2.1.
    """

    tokenizer: Callable[[str], list[int]]
    encoder: Callable[[list[int]], Any]
    emit_alpha: bool = False

    def to_episode(self, record: MegaV2EvalRecord) -> DreamEpisode:
        """Build a deterministic :class:`DreamEpisode` from ``record``.

        DR-0 accountability : ``episode_id = f"de-{record.id}"``.
        The ``input_slice`` carries :

        - ``beta_records`` : single dict with ``x`` (encoder output as
          list[float] of dim 4) and ``y`` (padded tokenized expected).
        - ``alpha_tokens`` : tuple of tokenized context (only when
          ``emit_alpha=True``).
        - ``_record`` : dict snapshot of the source record fields so
          :func:`dream_episode_to_record` can round-trip.
        """
        context_tokens = self.tokenizer(record.context)
        expected_tokens = self.tokenizer(record.expected)

        embedding = _to_float_list(self.encoder(context_tokens))
        y_vector = _pad_expected(expected_tokens, width=2)

        input_slice: dict[str, Any] = {
            "beta_records": [
                {"x": embedding, "y": y_vector},
            ],
            "_record": {
                "id": record.id,
                "context": record.context,
                "expected": record.expected,
                "domain": record.domain,
            },
        }
        if self.emit_alpha:
            input_slice["alpha_tokens"] = tuple(context_tokens)

        return DreamEpisode(
            trigger=EpisodeTrigger.SCHEDULED,
            input_slice=input_slice,
            operation_set=(Operation.REPLAY,),
            output_channels=(OutputChannel.WEIGHT_DELTA,),
            budget=BudgetCap(
                flops=_DEFAULT_FLOPS,
                wall_time_s=_DEFAULT_WALL,
                energy_j=_DEFAULT_ENERGY,
            ),
            episode_id=f"de-{record.id}",
        )


def dream_episode_to_record(de: DreamEpisode) -> MegaV2EvalRecord:
    """Round-trip a :class:`DreamEpisode` → :class:`MegaV2EvalRecord`.

    Reads the ``_record`` snapshot stashed by
    :meth:`MegaV2Adapter.to_episode`. Raises :class:`KeyError` if
    the episode was not built by the adapter (defensive — callers
    should not invoke this on arbitrary episodes).
    """
    snapshot = de.input_slice.get("_record")
    if snapshot is None:
        raise KeyError(
            f"DE {de.episode_id!r} has no '_record' snapshot ; "
            "only adapter-produced episodes can round-trip"
        )
    return MegaV2EvalRecord(
        id=str(snapshot["id"]),
        context=str(snapshot["context"]),
        expected=str(snapshot["expected"]),
        domain=str(snapshot["domain"]),
    )


__all__ = [
    "MegaV2Adapter",
    "dream_episode_to_record",
]
