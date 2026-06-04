"""Handler factories for the micro_kiki substrate.

Extracted from monolithic ``micro_kiki.py`` (1188 LOC) by N5 Task
4 (2026-05-10). Tasks 5-7 then extracted OPLoRA (:mod:`.oplora`),
TIES-Merge (:mod:`.ties`), and the safetensors loaders +
env-var gating (:mod:`.loaders`). The substrate class itself now
lives in :mod:`.substrate` (renamed from ``_legacy.py`` by Task 7).

The factories are bundled as a mixin so they continue to read
``self._restructure_state`` / ``self._recombine_state`` without
plumbing changes ; :class:`MicroKikiSubstrate` (in
:mod:`.substrate`) inherits from :class:`MicroKikiHandlersMixin`.
The 4 method names (``replay_handler_factory``,
``downscale_handler_factory``, ``restructure_handler_factory``,
``recombine_handler_factory``) are part of the DR-3 Conformance
Criterion contract and must not be renamed.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


class MicroKikiHandlersMixin:
    """Provide the 4 op-handler factories on :class:`MicroKikiSubstrate`.

    Mixin contract — the concrete substrate class must expose the
    instance attributes ``_restructure_state`` (a
    ``MicroKikiRestructureState``) and ``_recombine_state`` (a
    ``MicroKikiRecombineState``). Helper math (``_oplora_projector``,
    ``_ties_merge``) lives in :mod:`.oplora` and :mod:`.ties` and is
    imported lazily inside the factories so this module stays free
    of cross-module import cycles.
    """

    # ----- Protocol-contract factories (mirror esnn_* substrates) -----

    def replay_handler_factory(
        self,
    ) -> Callable[..., NDArray]:
        """A-Walker/Stickgold replay → LoRA gradient proxy.

        Signature matches
        ``esnn_thalamocortical.EsnnSubstrate.replay_handler_factory``
        for DR-3 condition-1 uniformity : the handler takes a
        ``beta_records: list[dict]`` + ``n_steps: int`` and
        returns a 1-D numpy array. The stub aggregates each
        record's ``"input"`` vector and returns the mean drive —
        sufficient to exercise the swap + S1 retained-benchmark
        path without an MLX device.
        """

        def handler(
            beta_records: list[dict], n_steps: int = 20,
        ) -> NDArray:
            if not beta_records:
                return np.zeros(1, dtype=np.float32)
            vectors: list[NDArray] = [
                np.asarray(r["input"], dtype=np.float32)
                for r in beta_records
                if "input" in r
            ]
            if not vectors:
                return np.zeros(1, dtype=np.float32)
            return np.asarray(
                np.mean(np.stack(vectors), axis=0), dtype=np.float32,
            )

        return handler

    def downscale_handler_factory(
        self,
    ) -> Callable[..., NDArray]:
        """B-Tononi SHY → LoRA B-matrix multiplicative shrink.

        Preserves DR-1 on the adapter state : the caller stamps
        the returned tensor with ``episode_id`` via
        ``kiki_oniric.dream.swap`` ; the handler itself only
        performs the arithmetic. Commutative, not idempotent
        (``f(f(w)) = w * factor²``). Matches the signature of
        ``esnn_*`` substrate downscale handlers.
        """

        def handler(weights: NDArray, factor: float) -> NDArray:
            if not (0.0 < factor <= 1.0):
                raise ValueError(
                    f"shrink_factor must be in (0, 1], got {factor}"
                )
            return (weights * factor).astype(weights.dtype, copy=False)

        return handler

    def restructure_handler_factory(
        self,
        rank_thresh: float = 1e-4,
    ) -> Callable[..., dict[str, NDArray]]:
        """D-Friston FEP restructure → **OPLoRA projection (phase 2)**.

        Wires the Orthogonal-Projection LoRA algorithm of Du et
        al. (arXiv 2510.13003) : given a list of prior-stack
        adapter deltas carried on ``adapter["prior_deltas"]``, the
        handler builds the projector ``P = I - U U^T`` onto the
        orthogonal complement of the priors' range and replaces
        the new adapter's B-matrix by ``P @ B_new``. The
        contribution ``P · B_new · A_new · x`` is then orthogonal
        to every prior range, preserving the S1 retained-benchmark
        invariant across sequential stack additions.

        Handler contract
        ----------------
        ``adapter`` is a mutable dict keyed by LoRA tensor name. It
        *must* carry :

        - ``"prior_deltas"`` : ``list[ndarray]`` of prior
          ``B_i @ A_i`` products (shape ``(out_dim, in_dim_i)``).
          Empty list means no priors to project away — handler
          returns the adapter unchanged (no-op).
        - ``key`` (the third positional arg) : name of the
          B-matrix entry to project. Its shape must be
          ``(out_dim, rank)`` with ``out_dim`` matching the
          priors.
        - Optional ``"episode_id"`` : DR-0 stamp propagated into
          :attr:`_restructure_state`.

        ``op`` is accepted for signature-compat with the phase-1
        stub but currently only ``"oplora"`` is honoured (the
        default). Any other value raises ``ValueError`` — keeps
        the gate explicit per DR-3 condition-1 (no silent
        no-ops).

        Returns the same ``adapter`` dict (with the entry at
        ``key`` replaced by ``P @ adapter[key]``).

        Reference : Du et al., arXiv 2510.13003 §3.2-§3.3. The
        local micro-kiki training pipeline at
        ``src/stacks/oplora.py`` (torch) mirrors this algebra ;
        this numpy port lives substrate-side for the dream
        runtime's no-torch constraint.
        """
        # Imported lazily to avoid an import cycle.
        # Lives in :mod:`.oplora` since N5 Task 5.
        from kiki_oniric.substrates.micro_kiki.oplora import (
            _oplora_projector,
        )

        def handler(
            adapter: dict[str, NDArray], op: str, key: str,
        ) -> dict[str, NDArray]:
            if op not in {"oplora", "oplora_project", "project"}:
                raise ValueError(
                    f"micro_kiki.restructure_handler: unsupported "
                    f"op {op!r} ; expected one of "
                    f"{{'oplora', 'oplora_project', 'project'}}"
                )
            if key not in adapter:
                raise KeyError(
                    f"micro_kiki.restructure_handler: adapter "
                    f"missing entry for key {key!r}"
                )
            new_B = np.asarray(adapter[key])
            if new_B.ndim != 2:
                raise ValueError(
                    f"micro_kiki.restructure_handler: adapter[{key!r}] "
                    f"must be 2-D (out_dim, rank), got shape "
                    f"{new_B.shape}"
                )

            priors = list(adapter.get("prior_deltas", []))
            episode_id = adapter.get("episode_id")

            if priors:
                P = _oplora_projector(priors, rank_thresh=rank_thresh)
                if P.shape[0] != new_B.shape[0]:
                    raise ValueError(
                        f"micro_kiki.restructure_handler: projector "
                        f"out_dim {P.shape[0]} != adapter[{key!r}] "
                        f"out_dim {new_B.shape[0]}"
                    )
                projected = (P @ new_B.astype(np.float32)).astype(
                    new_B.dtype, copy=False,
                )
                adapter[key] = projected
                self._restructure_state.total_projections_applied += 1

            # DR-0 bookkeeping : always bump the counter + record
            # the episode (even when priors is empty — the op was
            # invoked, and DR-0 credits every handler call, not
            # just those with a non-trivial effect).
            self._restructure_state.total_episodes_handled += 1
            self._restructure_state.last_completed = True
            self._restructure_state.last_operation = "restructure"
            if isinstance(episode_id, str):
                self._restructure_state.last_episode_id = episode_id
                self._restructure_state.episode_ids.append(episode_id)
            return adapter

        return handler

    def recombine_handler_factory(
        self,
        trim_fraction: float = 0.2,
        alpha: float = 1.0,
    ) -> Callable[..., NDArray]:
        """C-Hobson recombine → **TIES-Merge (phase 2)**.

        Wires the TIES-Merging algorithm of Yadav et al. (arXiv
        2306.01708) : given a list of task-specific delta tensors
        carried on ``payload["deltas"]``, the handler returns the
        merged delta via trim → elect-sign → disjoint-mean. Scale
        coefficient ``alpha`` amplifies the final merged
        contribution (default ``1.0``, paper default).

        Handler contract
        ----------------
        ``payload`` is a dict carrying at least ``"deltas"`` — a
        list of numpy delta tensors (shape-consistent across the
        list). Pragmatically matches ``DreamEpisode`` where the
        canonical access path is ``episode.payload["deltas"]``.

        - ``"deltas"`` : ``list[ndarray]`` — per-task / per-stack
          ``τ_i = W_ft_i - W_base``. Empty list raises (caller
          handles the no-op leg explicitly).
        - Optional ``"episode_id"`` : DR-0 stamp propagated into
          :attr:`_recombine_state`.

        ``op`` is accepted for signature-compat with the sibling
        :meth:`restructure_handler_factory` ; honours ``"ties"``
        (default), ``"ties_merge"``, ``"merge"``. Any other op
        raises ``ValueError`` — no silent no-ops per DR-3
        condition 1.

        Returns the merged delta tensor (dtype of the first input
        delta). The substrate's ``_recombine_state`` is bumped on
        every call (DR-0) ; the episode-id stamp (DR-1) is
        appended when present.

        Reference : Yadav et al., arXiv 2306.01708 §3.
        """
        from kiki_oniric.substrates.micro_kiki.ties import _ties_merge

        def handler(
            payload: dict[str, Any], op: str = "ties",
        ) -> NDArray:
            if op not in {"ties", "ties_merge", "merge"}:
                raise ValueError(
                    f"micro_kiki.recombine_handler: unsupported "
                    f"op {op!r} ; expected one of "
                    f"{{'ties', 'ties_merge', 'merge'}}"
                )
            if "deltas" not in payload:
                raise KeyError(
                    "micro_kiki.recombine_handler: payload missing "
                    "'deltas' entry (expected list[ndarray])"
                )
            deltas = list(payload["deltas"])
            episode_id = payload.get("episode_id")

            # _ties_merge guards empty / single / shape-mismatch.
            merged = _ties_merge(
                deltas, trim_fraction=trim_fraction, alpha=alpha,
            )

            # DR-0 bookkeeping.
            self._recombine_state.total_episodes_handled += 1
            self._recombine_state.total_merges_applied += 1
            self._recombine_state.last_completed = True
            self._recombine_state.last_operation = "recombine"
            self._recombine_state.last_k_deltas = len(deltas)
            first_shape = tuple(np.asarray(deltas[0]).shape)
            self._recombine_state.last_input_shape = first_shape
            self._recombine_state.last_output_shape = tuple(merged.shape)
            if isinstance(episode_id, str):
                self._recombine_state.last_episode_id = episode_id
                self._recombine_state.episode_ids.append(episode_id)
            return merged

        return handler
