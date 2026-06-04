"""micro-kiki substrate class — Qwen MoE + LoRA (cycle-3 Phase 2).

Renamed from ``_legacy.py`` by N5 Task 7 (2026-05-10), the final
task of the M2 audit split. Now the canonical home of
:class:`MicroKikiSubstrate` + the two DR-0 state dataclasses
(``MicroKikiRestructureState`` / ``MicroKikiRecombineState``) and
``micro_kiki_substrate_components``. Loaders + env gating live in
:mod:`.loaders`, handlers in :mod:`.handlers`, OPLoRA / TIES in
:mod:`.oplora` / :mod:`.ties`.

Third substrate for dreamOfkiki, wrapping the micro-kiki project's
adapter-training output. The intended production base is
``Qwen/Qwen3.5-35B-A3B`` (native 256-expert MoE, 3 B active per
token) with a standard LoRA adapter trained on 32 domain experts.
The substrate is however base-model agnostic ; any MLX-loadable
checkpoint with a companion LoRA ``adapters.safetensors`` is
acceptable.

Backend choice :
- ``mlx_lm`` (default) : declared target. When importable, the
  substrate loads the model + adapter lazily at :meth:`load`
  time and ``awake`` dispatches to ``mlx_lm.generate``.
- **Stub fallback** : when ``mlx_lm`` (or its ``mlx`` parent) is
  unavailable — the default on Linux CI — the substrate builds
  cleanly and exposes the 4 op-handler factories over numpy
  tensors only. This matches the pattern from
  ``esnn_norse`` (env-gated real backend + numpy fallback) so
  the DR-3 condition-1 test surface is exercised on every host.

Reference : ``docs/specs/2026-04-17-dreamofkiki-framework-C-design.md``
§6.2 (DR-3 Conformance Criterion). Scope : DR-0 / DR-1 / DR-3
condition-1 surface ; DR-2 / DR-4 defer to phase 4 (conformance
harness over the 3-substrate matrix).

Phase boundaries (explicit) :
- **Phase 1** : replay + downscale operational on LoRA adapter
  tensors, restructure + recombine stubbed with an explicit
  ``NotImplementedError`` citing the blocker.
- **Phase 2 (this file)** : OPLoRA projection wired in
  (restructure ; arXiv 2510.13003 Du et al.) + TIES-Merge wired
  in (recombine ; arXiv 2306.01708 Yadav et al.). All 4 handlers
  now backed ; +PARTIAL retained until the Phase-4 conformance
  harness lands (no downgrade on the EC axis while Phase-3
  cross-substrate ablation is in flight).
- **Phase 3** : swap / eval_retained bindings + cross-substrate
  ablation (cycle-3 G10 Gate D).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


_LOG = logging.getLogger(__name__)


# DualVer C-v0.9.1+PARTIAL — TIES-Merge recombine (arXiv
# 2306.01708) wired in ; all 4 handlers now backed. Adds
# SpikingKiki-V4 real-backend shim behind DREAM_MICRO_KIKI_REAL=1
# env flag (additive, env-unset behaviour unchanged). Retained
# ``+PARTIAL`` until Phase-4 conformance harness confirms DR-2
# / DR-4 across the 3-substrate matrix.
MICRO_KIKI_SUBSTRATE_NAME = "micro_kiki"
MICRO_KIKI_SUBSTRATE_VERSION = "C-v0.9.1+PARTIAL"


# Env-var gating + safetensors loader extracted to :mod:`.loaders`
# by N5 Task 7 (2026-05-10). Re-exported below for back-compat.
from kiki_oniric.substrates.micro_kiki.loaders import (  # noqa: E402, F401
    _REAL_BACKEND_ENV_VAR,
    _REAL_BACKEND_PATH_ENV_VAR,
    _real_backend_enabled,
    _real_backend_path_from_env,
    _try_load_safetensors,
)


# OPLoRA projector helper extracted to :mod:`.oplora` (N5 Task 5,
# 2026-05-10). Re-exported via the package ``__init__`` ; not used
# directly inside this module since handlers.py imports it lazily.
from kiki_oniric.substrates.micro_kiki.oplora import (  # noqa: E402, F401
    _oplora_projector,
)


@dataclass
class MicroKikiRestructureState:
    """DR-0 accountability record for the OPLoRA restructure op.

    Each invocation of the :meth:`restructure_handler_factory`
    closure bumps ``total_episodes_handled`` and appends the
    episode id (if supplied via ``adapter["episode_id"]``) to
    ``episode_ids``. ``completed`` + ``operation`` are mirrored
    on the last-handled record so DR-0 traceability is preserved
    (``completed=True`` + ``operation='restructure'``). The state
    is read by the conformance harness to verify DR-1 (episodic
    stamp consistency) across the 3-substrate matrix.
    """

    total_episodes_handled: int = 0
    total_projections_applied: int = 0
    last_episode_id: str | None = None
    last_operation: str = "restructure"
    last_completed: bool = False
    episode_ids: list[str] = field(default_factory=list)


# -----------------------------------------------------------------
# TIES-Merge (Yadav et al., arXiv 2306.01708) — extracted to
# :mod:`.ties` by N5 Task 6 (2026-05-10). Re-exported below for
# back-compat with any remaining call site under ``._legacy``.
# -----------------------------------------------------------------
from kiki_oniric.substrates.micro_kiki.ties import (  # noqa: E402, F401
    _ties_merge,
)


@dataclass
class MicroKikiRecombineState:
    """DR-0 accountability record for the TIES-Merge recombine op.

    Mirrors :class:`MicroKikiRestructureState` so the DR-3
    conformance harness parametrises the 4 handlers uniformly.
    Each invocation of the :meth:`recombine_handler_factory`
    closure bumps ``total_episodes_handled`` and — when the
    handler actually merged a non-empty delta list — records the
    merged-tensor shape stamp on ``last_output_shape``. ``DR-1``
    episode-id stamps land on ``last_episode_id`` + ``episode_ids``.
    """

    total_episodes_handled: int = 0
    total_merges_applied: int = 0
    last_episode_id: str | None = None
    last_operation: str = "recombine"
    last_completed: bool = False
    last_k_deltas: int = 0
    last_input_shape: tuple[int, ...] | None = None
    last_output_shape: tuple[int, ...] | None = None
    episode_ids: list[str] = field(default_factory=list)

# Optional-dependency probe : ``mlx_lm`` (Apple Silicon MLX wheel
# + LoRA adapters) is imported lazily inside the method that
# actually needs it (:meth:`MicroKikiSubstrate.load`). We record
# only a boolean flag at module-import so callers can introspect
# availability without a second try-import. Tests cover the False
# branch ; the True branch is env-gated on Apple Silicon.
try:  # pragma: no cover - branch depends on env
    import mlx_lm  # noqa: F401

    _MLX_LM_AVAILABLE = True
except ImportError:  # pragma: no cover - branch depends on env
    _MLX_LM_AVAILABLE = False


from kiki_oniric.substrates.micro_kiki.handlers import (  # noqa: E402
    MicroKikiHandlersMixin,
)
from kiki_oniric.substrates.micro_kiki.spike_loader import (  # noqa: E402
    SpikeLoaderMixin,
)


@dataclass
class MicroKikiSubstrate(MicroKikiHandlersMixin, SpikeLoaderMixin):
    """micro-kiki framework-C substrate (Qwen MoE + LoRA).

    Parameters
    ----------
    base_model_path
        Optional path (local dir or HF repo id) to an
        ``mlx_lm``-loadable model. When ``None`` the substrate
        runs in pure-stub mode : handler factories operate on
        numpy tensors only, :meth:`awake` returns a canned string.
        This keeps the module importable + testable on hosts
        without Apple Silicon / the MLX wheel.
    adapter_path
        Optional path to a LoRA ``adapters.safetensors`` file (or
        directory containing one). Loaded by :meth:`load` via
        ``mlx_lm.load_adapters`` when present.
    num_layers
        Number of transformer layers the LoRA adapter targets.
        Default 20 — matches the micro-kiki v4 default shape. Only
        used to validate adapter-tensor shapes in the numpy
        fallback path ; real MLX loading ignores this.
    rank
        LoRA rank. Default 16 (micro-kiki v4 SOTA spec). Used to
        size stub adapter tensors in the numpy fallback path.
    seed
        Numpy RNG seed — controls any stochastic handler (recombine).

    Attributes
    ----------
    mlx_lm_available
        Informational bool mirroring the module-level probe.
    """

    base_model_path: str | None = None
    adapter_path: str | None = None
    real_backend_path: str | Path | None = None
    num_layers: int = 20
    rank: int = 16
    seed: int = 0
    mlx_lm_available: bool = field(default=_MLX_LM_AVAILABLE, init=False)
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    # Real SpikingKiki backend state. Populated by :meth:`load`
    # only when DREAM_MICRO_KIKI_REAL=1 + a valid real_backend_path
    # are set. Shape ::= ``{"lif_metadata": dict, "module_count": int,
    # "sample_weights": {module_name -> NDArray}, "adapter_weights":
    # Optional[{key -> NDArray}]}``. ``None`` elsewhere keeps stub
    # semantics wire-compatible with phase-1 callers.
    _real_state: dict[str, Any] | None = field(
        default=None, init=False, repr=False,
    )
    # Accumulator for the in-flight weight delta produced by the
    # replay / downscale handlers. Stored as a plain ``dict`` keyed
    # by the adapter weight-path (matches the shape emitted by
    # ``mlx_lm.tuner.trainable_parameters``). Round-tripped by
    # :meth:`snapshot` / :meth:`load_snapshot` as a numpy ``.npz``.
    _current_delta: dict[str, NDArray] = field(
        default_factory=dict, init=False, repr=False,
    )
    # DR-0 accountability state for the OPLoRA restructure handler
    # (arXiv 2510.13003). Exposed read-only via :meth:`restructure_state`
    # so the conformance harness can assert DR-1 (episode_id stamp
    # consistency) without poking private attributes.
    _restructure_state: MicroKikiRestructureState = field(
        default_factory=MicroKikiRestructureState,
        init=False,
        repr=False,
    )
    # DR-0 accountability state for the TIES-Merge recombine handler
    # (arXiv 2306.01708). Same accessor pattern as
    # ``_restructure_state`` — read-only via :meth:`recombine_state`.
    _recombine_state: MicroKikiRecombineState = field(
        default_factory=MicroKikiRecombineState,
        init=False,
        repr=False,
    )
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError(
                f"num_layers must be > 0, got {self.num_layers}"
            )
        if self.rank <= 0:
            raise ValueError(f"rank must be > 0, got {self.rank}")
        self._rng = np.random.default_rng(self.seed)
        # Fallback : when the constructor did not receive a
        # ``real_backend_path``, fall back to the env-provided path
        # (``DREAM_MICRO_KIKI_REAL_BACKEND_PATH``). Explicit arg wins.
        if self.real_backend_path is None:
            env_path = _real_backend_path_from_env()
            if env_path is not None:
                self.real_backend_path = env_path

    # ----- lazy model / adapter load -----

    def load(self) -> None:
        """Load the base model + LoRA adapter via ``mlx_lm``.

        Two code paths, both opt-in :

        1. **mlx_lm path** (``base_model_path`` set + ``mlx_lm``
           importable). Runs on Mac Studio M3 Ultra.
        2. **SpikingKiki path** (``DREAM_MICRO_KIKI_REAL=1`` +
           ``real_backend_path`` set). Reads
           ``lif_metadata.json`` + a 3-module ``.npz`` sample +
           (optionally) a sidecar ``adapters.safetensors``. The
           outcome is a minimal ``_real_state`` dict that
           :meth:`awake` rate-codes — the full 35B forward pass
           still requires MLX ; this shim is the minimum viable
           real-artifact ingestion needed by the conformance
           harness.

        Both paths are independent. When neither is eligible, the
        method is a no-op and the substrate stays in stub mode.
        """
        # ---- Path 1 : mlx_lm base-model load (existing) ----
        if self.base_model_path is not None and self.mlx_lm_available:
            # pragma: no cover - env-gated (Apple Silicon only)
            from mlx_lm import load as mlx_load  # type: ignore[import-not-found]

            self._model, self._tokenizer = mlx_load(self.base_model_path)
            if self.adapter_path is not None:
                from mlx_lm.tuner.utils import (  # type: ignore[import-not-found]
                    load_adapters,
                )

                self._model = load_adapters(
                    self._model, self.adapter_path,
                )

        # ---- Path 2 : SpikingKiki-V4 real-backend shim ----
        if _real_backend_enabled() and self.real_backend_path is not None:
            try:
                self._real_state = self._load_spiking_backend()
            except Exception as exc:  # noqa: BLE001 — ingestion guard
                _LOG.warning(
                    "micro_kiki real-backend load failed (%s) ; "
                    "falling back to stub mode",
                    exc,
                )
                self._real_state = None

        # ---- Path 2b : safetensors adapter sidecar (best-effort) ----
        # Independent of real_backend — may also fire in stub mode
        # to pre-populate the accumulator delta for subsequent
        # replay / downscale handler calls on a real adapter.
        if self.adapter_path is not None:
            weights = _try_load_safetensors(self.adapter_path)
            if weights is not None:
                # Merge into _current_delta so the snapshot path
                # sees the real tensors (small ones — LoRA
                # adapters are megabyte-scale on 35B bases).
                self._current_delta.update(weights)

    # ``_load_spiking_backend`` lives on :class:`SpikeLoaderMixin`
    # (extracted by N6 Task 4, 2026-05-10). Still exposed here as
    # ``self._load_spiking_backend()`` thanks to MRO composition.

    # ----- awake-side generation -----

    def awake(self, prompt: str, max_tokens: int = 32) -> str:
        """Awake forward pass — returns generated text.

        Three code paths, selected in priority order :

        1. **mlx_lm path** — model + tokenizer loaded, dispatches
           to ``mlx_lm.generate`` (Apple Silicon only).
        2. **SpikingKiki rate-coded path** — ``_real_state`` is
           populated (env flag + artifact present), returns a
           synthetic spike-count string threshold-crossed on the
           first real weight matrix. Not a full SNN forward pass
           (that requires MLX) ; this is the minimum-viable
           signal proving the real weights reach the handler
           surface. See :meth:`awake_spike_payload` for the raw
           array form that the conformance harness consumes.
        3. **Stub path** — returns ``f"[stub awake] {prompt}"``.
        """
        # Path 1 : mlx_lm
        if self._model is not None and self._tokenizer is not None:
            # pragma: no cover - env-gated (Apple Silicon only)
            from mlx_lm import generate  # type: ignore[import-not-found]

            return str(
                generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False,
                )
            )

        # Path 2 : SpikingKiki rate-coded synthesis
        if self._real_state is not None:
            payload = self.awake_spike_payload(prompt)
            n_spikes = int(payload["output_channels"]["spikes"].sum())
            T = payload["metadata"]["T"]
            module = payload["metadata"]["module"]
            return (
                f"[spiking awake T={T} module={module} spikes={n_spikes}] "
                f"{prompt}"
            )

        # Path 3 : stub
        return f"[stub awake] {prompt}"

    # ``awake_spike_payload`` lives on :class:`SpikeLoaderMixin`
    # (extracted by N6 Task 4, 2026-05-10). Still exposed here as
    # ``self.awake_spike_payload(prompt)`` thanks to MRO composition.

    # ----- Protocol-contract factories -----
    # The 4 handler factories live on
    # :class:`MicroKikiHandlersMixin` (see ``handlers.py``). Only
    # the read-only state accessors stay here (they expose private
    # attributes initialised by ``__init__``). Tasks 5-7 will keep
    # peeling concerns away from this holding file.

    @property
    def restructure_state(self) -> MicroKikiRestructureState:
        """Read-only accessor for the OPLoRA DR-0 record.

        Exposed so the conformance harness (and unit tests) can
        assert DR-0 (completed flag, operation label) + DR-1
        (episode_id stamp propagation) without poking private
        attributes. The returned object is the live state dataclass
        — do not mutate ; it is refreshed by each handler call.
        """
        return self._restructure_state

    @property
    def recombine_state(self) -> MicroKikiRecombineState:
        """Read-only accessor for the TIES-Merge DR-0 record.

        Mirrors :meth:`restructure_state`. The returned object is
        the live state dataclass — do not mutate ; it is refreshed
        by each handler call.
        """
        return self._recombine_state

    # ----- γ-snapshot : adapter round-trip -----

    def snapshot(self, path: str | Path) -> Path:
        """Persist the current accumulator delta to a ``.npz`` file.

        Returns the written path. Round-trips cleanly via
        :meth:`load_snapshot`. In a full MLX run the swap protocol
        would instead serialise the LoRA adapter via
        ``mlx_lm.utils.save_adapters`` to a ``.safetensors`` —
        here we use numpy ``.npz`` for portability so the same
        artifact format works on Linux CI.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix != ".npz":
            target = target.with_suffix(".npz")
        np.savez(target, **self._current_delta)
        return target

    def load_snapshot(self, path: str | Path) -> None:
        """Restore the accumulator delta from a :meth:`snapshot` file."""
        target = Path(path)
        if not target.exists() and target.with_suffix(".npz").exists():
            target = target.with_suffix(".npz")
        data = np.load(target, allow_pickle=False)
        self._current_delta = {k: np.asarray(data[k]) for k in data.files}


def micro_kiki_substrate_components() -> dict[str, str]:
    """Return the canonical map of micro-kiki substrate components.

    Mirrors ``esnn_substrate_components`` + ``norse_substrate_components``
    so the DR-3 Conformance Criterion test suite parametrizes over
    the three substrates uniformly. Phase 2 lands the real
    restructure + recombine backends ; the dotted paths stay
    stable across that transition (the same module hosts both
    the stub + the real impl).
    """
    return {
        # 8 typed Protocols (substrate-agnostic, shared)
        "primitives": "kiki_oniric.core.primitives",
        # 4 operations — factory methods on this substrate class
        "replay": "kiki_oniric.substrates.micro_kiki",
        "downscale": "kiki_oniric.substrates.micro_kiki",
        # phase-2 stubs, path stable across bump
        "restructure": "kiki_oniric.substrates.micro_kiki",
        "recombine": "kiki_oniric.substrates.micro_kiki",
        # 2 invariant guards (substrate-agnostic, shared)
        "finite": "kiki_oniric.dream.guards.finite",
        "topology": "kiki_oniric.dream.guards.topology",
        # Runtime + swap (substrate-agnostic, shared)
        "runtime": "kiki_oniric.dream.runtime",
        "swap": "kiki_oniric.dream.swap",
        # 3 profiles (substrate-agnostic wrappers, shared)
        "p_min": "kiki_oniric.profiles.p_min",
        "p_equ": "kiki_oniric.profiles.p_equ",
        "p_max": "kiki_oniric.profiles.p_max",
    }
