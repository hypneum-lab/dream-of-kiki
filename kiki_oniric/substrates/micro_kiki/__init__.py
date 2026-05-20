"""micro_kiki substrate package.

Split from monolithic ``micro_kiki.py`` (1188 LOC) by N5 Tasks
4-7 (M2 audit). Public API preserved via the re-exports below ;
existing call sites
``from kiki_oniric.substrates.micro_kiki import X`` keep working.

- Task 4 (2026-05-10) — :mod:`.handlers` (4 op-handler factories
  as a mixin so ``MicroKikiSubstrate`` still exposes them as
  methods, per the DR-3 Conformance Criterion contract).
- Task 5 (2026-05-10) — :mod:`.oplora` (OPLoRA projector helper).
- Task 6 (2026-05-10) — :mod:`.ties` (TIES-Merge recombine).
- Task 7 (2026-05-10) — :mod:`.loaders` (safetensors + env-var
  gating) and :mod:`.substrate` (the substrate class itself,
  renamed from ``_legacy.py``).
- N6 Task 4 (2026-05-10) — :mod:`.spike_loader` (SpikingKiki
  ingestion + rate-coded spike payload synthesis as a mixin).
"""
from __future__ import annotations

# Loaders + env gating (Task 7 — :mod:`.loaders`).
from kiki_oniric.substrates.micro_kiki.loaders import (  # noqa: F401
    _REAL_BACKEND_ENV_VAR,
    _REAL_BACKEND_PATH_ENV_VAR,
    _real_backend_enabled,
    _real_backend_path_from_env,
    _try_load_safetensors,
)

# OPLoRA projector (Task 5 — :mod:`.oplora`) and TIES-Merge
# (Task 6 — :mod:`.ties`).
from kiki_oniric.substrates.micro_kiki.oplora import (  # noqa: F401
    _oplora_projector,
)
from kiki_oniric.substrates.micro_kiki.ties import (  # noqa: F401
    _ties_merge,
)

# Substrate class + DR-0 state dataclasses + components map
# (Task 7 — :mod:`.substrate`, renamed from ``_legacy.py``).
from kiki_oniric.substrates.micro_kiki.substrate import (  # noqa: F401
    MICRO_KIKI_SUBSTRATE_NAME,
    MICRO_KIKI_SUBSTRATE_VERSION,
    MicroKikiRecombineState,
    MicroKikiRestructureState,
    MicroKikiSubstrate,
    micro_kiki_substrate_components,
)

# Handler factories (Task 4 — :mod:`.handlers`).
from kiki_oniric.substrates.micro_kiki.handlers import (  # noqa: F401
    MicroKikiHandlersMixin,
)

# Spike loader mixin (N6 Task 4 — :mod:`.spike_loader`).
from kiki_oniric.substrates.micro_kiki.spike_loader import (  # noqa: F401
    SpikeLoaderMixin,
)

# LoRA-adapter model abstraction (B1a Task 1 — :mod:`.lora_model`).
# Imported lazily : `lora_model` depends on MLX, which only ships
# wheels for arm64-darwin. On non-Apple-Silicon platforms (e.g. the
# Linux CI runners), MLX is missing and importing the package would
# raise `ImportError: libmlx.so: cannot open shared object file`.
# Tests that exercise LoRA paths import `lora_model` directly and
# are gated by `conftest.py::collect_ignore_glob`.
try:
    from kiki_oniric.substrates.micro_kiki.lora_model import (  # noqa: F401
        LoRALinear,
        LoRAModel,
    )
except ImportError:  # pragma: no cover - MLX-less platforms (Linux CI)
    LoRALinear = None  # type: ignore[assignment,misc]
    LoRAModel = None  # type: ignore[assignment,misc]

__all__ = [
    "MICRO_KIKI_SUBSTRATE_NAME",
    "MICRO_KIKI_SUBSTRATE_VERSION",
    "LoRALinear",
    "LoRAModel",
    "MicroKikiHandlersMixin",
    "MicroKikiRecombineState",
    "MicroKikiRestructureState",
    "MicroKikiSubstrate",
    "SpikeLoaderMixin",
    "_REAL_BACKEND_ENV_VAR",
    "_REAL_BACKEND_PATH_ENV_VAR",
    "_oplora_projector",
    "_real_backend_enabled",
    "_real_backend_path_from_env",
    "_ties_merge",
    "_try_load_safetensors",
    "micro_kiki_substrate_components",
]
