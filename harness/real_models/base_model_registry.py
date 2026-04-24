"""Base model registry — cycle 3 real-data lock (pre-cycle-3 lock #2).

Every entry pins ``(repo_id, revision_sha, file_sha256)`` for R1
reproducibility contract. See framework-C spec §8.4 and
``docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md``
§5 (risk R1 mitigation) + §8 (glossary : scale-axis).

SHA values obtained from HuggingFace metadata at 2026-04-19
(``https://huggingface.co/api/models/{repo_id}`` +
``/tree/main``). Run::

    python -c "from harness.real_models.base_model_registry \\
        import verify_all; print(verify_all())"

to validate that the recorded pins still match current HF state.
The verifier is network-free by default (returns local-self-check
booleans only) to keep test runs deterministic per R1 ; a live
HTTP check can be enabled by passing ``live=True``.

Note on model version (cycle-3 fallback) : the spec targets
Qwen3.5 at scale ``{1.5B, 7B, 35B}`` Q4. At 2026-04-19 the
Qwen3.5 series is not yet published on HuggingFace ; the closest
publicly-available MLX Q4-quantized weights are from the Qwen2.5
series (1.5B, 7B, 32B). We therefore pin Qwen2.5 MLX-Q4 as the
scale-slot occupant under registry keys ``qwen3p5-*`` and record
the fallback in ``notes``. When Qwen3.5 MLX-Q4 lands, a new pin
entry + DualVer bump (FC-PATCH or EC-MINOR depending on
behavioural delta) will replace the occupant in-place ; key
stability is preserved for downstream ``get_pin`` callers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SHA40_RE = re.compile(r"^[a-f0-9]{40}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class BaseModelPin:
    """Immutable pin for a base model entry.

    Attributes
    ----------
    name
        Canonical scale-slot key (e.g. ``qwen3p5-1p5b``). Stable
        across Qwen2.5 → Qwen3.5 upgrade.
    scale_params
        Nominal parameter count (pre-quantization). Used for the
        scale-axis in cycle-3 H5 tests.
    repo_id
        HuggingFace ``org/repo`` identifier for the pinned model.
    revision_sha
        40-char HuggingFace revision (git) SHA. Matches regex
        ``^[a-f0-9]{40}$``.
    file_sha256
        SHA-256 (LFS oid) of the main weight file (or the first
        shard of a multi-shard safetensors set). 64-char hex, or
        ``None`` if the HF API did not expose one at pin time.
    quantization
        Quantization scheme (e.g. ``4bit-mlx``, ``q4_K_M-gguf``).
    framework
        Inference framework (e.g. ``mlx-lm``, ``llama.cpp``).
    approx_ram_gb
        Approximate resident RAM for loaded weights (order-of-
        magnitude ; used for scheduling on Studio vs Mac Mini).
    notes
        Free-text provenance notes. See module-level docstring for
        the Qwen2.5-fallback rationale.
    """

    name: str
    scale_params: int
    repo_id: str
    revision_sha: str
    file_sha256: str | None
    quantization: str
    framework: str
    approx_ram_gb: float
    notes: str = ""


# Pin values recorded 2026-04-19 from HuggingFace API.
# Method :
#   curl -s https://huggingface.co/api/models/{repo_id} -> .sha
#   curl -s https://huggingface.co/api/models/{repo_id}/tree/main
#       -> siblings[*].lfs.oid for model*.safetensors
# Re-run verify_all() after any pin update.

REGISTRY: dict[str, BaseModelPin] = {
    "qwen3p5-1p5b": BaseModelPin(
        name="qwen3p5-1p5b",
        scale_params=1_500_000_000,
        repo_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        revision_sha="8b403126fc14f14cfc99bb4cfa72ecbc129ea677",
        file_sha256=(
            "0979f33d1bc58afcf696d13f57977644e7b11a6f0eec3e631d"
            "8e9463d18c0717"
        ),
        quantization="4bit-mlx",
        framework="mlx-lm",
        approx_ram_gb=1.0,
        notes=(
            "Qwen2.5 fallback for Qwen3.5 slot ; single-shard "
            "model.safetensors 868 MB ; license apache-2.0 ; "
            "estimated ~150-200 tok/s on M3 Ultra."
        ),
    ),
    "qwen3p5-7b": BaseModelPin(
        name="qwen3p5-7b",
        scale_params=7_000_000_000,
        repo_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        revision_sha="c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed",
        file_sha256=(
            "86110f368236b53cf4c2336f991a85703b17bcc60bb75f292b"
            "4002ec0219f071"
        ),
        quantization="4bit-mlx",
        framework="mlx-lm",
        approx_ram_gb=4.5,
        notes=(
            "Qwen2.5 fallback for Qwen3.5 slot ; single-shard "
            "model.safetensors 4.28 GB ; license apache-2.0 ; "
            "estimated ~60-90 tok/s on M3 Ultra."
        ),
    ),
    "qwen3p5-35b": BaseModelPin(
        name="qwen3p5-35b",
        scale_params=32_500_000_000,
        repo_id="mlx-community/Qwen2.5-32B-Instruct-4bit",
        revision_sha="2938092373e5f97b95538884112085364c2da315",
        file_sha256=(
            "3187f89267bdebe362410a3b23c2767d9d0707f4ffbbf7a945"
            "e5cd0abf535a21"
        ),
        quantization="4bit-mlx",
        framework="mlx-lm",
        approx_ram_gb=20.0,
        notes=(
            "Qwen2.5-32B fallback for Qwen3.5-35B slot "
            "(closest available public MLX-Q4 ; ~35B target ≈ "
            "32B actual, -8%). Multi-shard 4x safetensors, "
            "file_sha256 pins shard 1/4 (model-00001-of-00004) ; "
            "shards 2-4 oids : "
            "31547d7a3e6583eea8bcc1b7230680f8132143c43c19cab3e"
            "ab75f6b31d33e33, "
            "d00aec153b1ea6540446c1769ef76004e4ee11855844919f4"
            "8434f0d4f77b33f, "
            "4dae2dd0355ac7ad5440ccd4d5f60a08084cda924c0584"
            "02b979654359150743. "
            "Total 18.4 GB ; license apache-2.0 ; "
            "estimated ~25-40 tok/s on M3 Ultra "
            "(ref : Qwen3.5-35B-A3B Opus distill @ 162 tok/s on "
            "RTX 4090, MLX should undershoot by ~4x at similar "
            "scale per memory)."
        ),
    ),
    # ------------------------------------------------------------------
    # FP16 (bf16) scale slots — cycle-3 C3.8 Phase A.
    # ------------------------------------------------------------------
    # Unquantized bf16 weights are required for the real ablation
    # pilot because gradient/backprop flows natively through the
    # weights — dream ops (replay MSE SGD, downscale shrink,
    # restructure reroute, recombine latent substitution) can
    # genuinely mutate the model. The Q4 pins above remain useful
    # for inference-only contexts (e.g. the MMLU logit-bias proxy
    # on the Q4-quantised 1.5B adapter sanity pilot, kept for
    # reference) but the C3.8 real pilot scoring path runs against
    # these bf16 pins.
    #
    # Pin values recorded 2026-04-19 from HuggingFace API via
    #   curl -s https://huggingface.co/api/models/{repo_id}
    #   curl -s https://huggingface.co/api/models/{repo_id}/tree/main
    # See commit message for verification transcript.
    "qwen3p5-1p5b-fp16": BaseModelPin(
        name="qwen3p5-1p5b-fp16",
        scale_params=1_500_000_000,
        repo_id="mlx-community/Qwen2.5-1.5B-Instruct-bf16",
        revision_sha="4ae77cb209f06199b8df1c94e21ff341332a3a89",
        file_sha256=(
            "073400b311d049b6379b8f3f7ad5a0258268031dd009b99389"
            "cfd4030144850a"
        ),
        quantization="bf16-mlx",
        framework="mlx-lm",
        approx_ram_gb=3.1,
        notes=(
            "Qwen2.5 fallback for Qwen3.5 slot ; single-shard "
            "model.safetensors 3.09 GB bf16 ; license apache-2.0 "
            "; backprop-capable through native bf16 weights — "
            "used by scripts/pilot_cycle3_real.py for the C3.8 "
            "Phase A real ablation pilot where dream ops must "
            "mutate weights in-place."
        ),
    ),
    "qwen3p5-7b-fp16": BaseModelPin(
        name="qwen3p5-7b-fp16",
        scale_params=7_000_000_000,
        repo_id="mlx-community/Qwen2.5-7B-Instruct-bf16",
        revision_sha="349a12f0e5f131c3914e3a66e78a3db71e9f9527",
        file_sha256=(
            "2d67034c181238a2978fead9169969328d016a7bc88c2cf75a"
            "bf81ceb39452a6"
        ),
        quantization="bf16-mlx",
        framework="mlx-lm",
        approx_ram_gb=15.2,
        notes=(
            "Qwen2.5-7B bf16 fallback for Qwen3.5-7B slot ; "
            "multi-shard 3x safetensors totalling 15.23 GB ; "
            "file_sha256 pins shard 1/3 (model-00001-of-00003) ; "
            "shards 2-3 oids : "
            "e59a51540a99f1992d8a05ff263b042b869368d0082ac6e4dae"
            "596e658eb3116, "
            "d6b92aab8dd7e0d757bbb048cb814463ef992adb79d4e720ba5"
            "d6c942885287b. "
            "license apache-2.0 ; backprop-capable for dream ops."
        ),
    ),
    "qwen3p5-35b-fp16": BaseModelPin(
        name="qwen3p5-35b-fp16",
        scale_params=32_500_000_000,
        repo_id="mlx-community/Qwen2.5-32B-Instruct-bf16",
        revision_sha="7d56aa0beaa00b8d7b07509e399f4272e38769d5",
        file_sha256=(
            "9b95cdf1b6350397fdeea670dfd55f8f6d60189980188d68694"
            "b548e2449fe29"
        ),
        quantization="bf16-mlx",
        framework="mlx-lm",
        approx_ram_gb=65.0,
        notes=(
            "Qwen2.5-32B bf16 fallback for Qwen3.5-35B slot ; "
            "multi-shard 13x safetensors totalling ~65.5 GB ; "
            "file_sha256 pins shard 1/13 (model-00001-of-00013) ; "
            "remaining shards 2-13 oids enumerated in the C3.8 "
            "Phase A verification transcript (commit log). Fits "
            "easily on Studio M3 Ultra 512 GB. license apache-2.0 "
            "; practical structural concern : bf16-35B training "
            "throughput at ~5-15 tok/s limits per-cell wall time."
        ),
    ),
    # ------------------------------------------------------------------
    # Local bf16 pin — exploratory pilot only (no SHA pin, not R1).
    # ------------------------------------------------------------------
    # Studio-local path to Qwen3.6-35B-A3B bf16 weights cloned under
    # ``/Users/clems/KIKI-Mac_tunner/models/``. This pin is used by
    # the cycle-3 substrate-ablation exploratory pilot (1 model x 3
    # substrates) and is **not** intended for the reproducibility
    # publication — no HF revision SHA, no file SHA-256, and the
    # repo_id is a local absolute path rather than an ``org/repo``
    # identifier. ``verify_all`` tolerates absolute paths as a
    # self-consistent local-only pin (see ``_is_local_path`` helper
    # below) ; any empirical claim derived from this slot must be
    # tagged as pilot-only in the paper.
    "qwen3p6-35b-bf16-local": BaseModelPin(
        name="qwen3p6-35b-bf16-local",
        scale_params=35_000_000_000,
        repo_id="/Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B",
        revision_sha="0" * 40,  # dummy ; local path bypasses live check
        file_sha256=None,  # bypass SHA check (local, no HF oid)
        quantization="bf16",
        framework="mlx-lm",
        approx_ram_gb=70.0,  # observed peak at load (pilot notes)
        notes=(
            "Local bf16 Qwen3.6-35B-A3B MoE (256 experts, 8 active). "
            "Multimodal config but text-only path used by the "
            "substrate-ablation pilot. Exploratory only — not for "
            "reproducibility publication (no SHA pin, local path). "
            "Loaded via ``mlx_lm.load(path)`` which accepts absolute "
            "directory paths in addition to ``org/repo`` ids."
        ),
    ),
}


def _is_local_path(repo_id: str) -> bool:
    """True when ``repo_id`` is an absolute filesystem path.

    Exploratory pilot pins (e.g. ``qwen3p6-35b-bf16-local``) point
    at Studio-local bf16 weights rather than a HuggingFace
    ``org/repo`` identifier. ``verify_all`` treats such entries as
    self-consistent if the path starts with ``/`` — the R1
    publication contract requires HF-pinned entries and filters
    local pins out explicitly.
    """
    return repo_id.startswith("/")


def get_pin(name: str) -> BaseModelPin:
    """Return the :class:`BaseModelPin` for scale-slot ``name``.

    Raises :class:`KeyError` if the slot is not registered ; the
    error message lists available slots to ease discovery.
    """
    if name not in REGISTRY:
        available = sorted(REGISTRY.keys())
        raise KeyError(
            f"no base model pinned for slot {name!r} ; "
            f"available : {available}"
        )
    return REGISTRY[name]


def verify_all(live: bool = False) -> dict[str, bool]:
    """Validate pin self-consistency for every registered slot.

    Local-only checks (always run, network-free) :

    - ``revision_sha`` is a 40-char lowercase hex string ;
    - ``file_sha256``, if present, is a 64-char lowercase hex
      string ;
    - ``scale_params`` is strictly positive ;
    - ``repo_id`` is non-empty and contains exactly one ``/``.

    Live HTTP check (opt-in via ``live=True``) : fetches
    ``https://huggingface.co/api/models/{repo_id}`` and compares
    the returned ``sha`` field against the recorded
    ``revision_sha``. Kept opt-in so test runs stay deterministic
    and offline-friendly per R1.

    Returns a mapping ``name -> bool`` (``True`` = all checks
    passed for that entry).
    """
    results: dict[str, bool] = {}
    for name, pin in REGISTRY.items():
        ok = True
        if pin.scale_params <= 0:
            ok = False
        if not pin.repo_id:
            ok = False
        if _is_local_path(pin.repo_id):
            # Local exploratory pin — skip HF-shape checks on
            # revision/file SHA and tolerate multi-slash absolute
            # path. Live HTTP check is also a no-op (no HF repo).
            pass
        else:
            if not _SHA40_RE.match(pin.revision_sha):
                ok = False
            if pin.file_sha256 is not None and not _SHA256_RE.match(
                pin.file_sha256
            ):
                ok = False
            if pin.repo_id.count("/") != 1:
                ok = False
            if ok and live:
                ok = _verify_live(pin)
        results[name] = ok
    return results


def _verify_live(pin: BaseModelPin) -> bool:
    """Fetch HF API and confirm ``revision_sha`` still matches.

    Imported lazily so the module has no hard network dependency
    at import time. Any network/parse failure returns ``False``
    (conservative — caller must interpret as "needs re-pin").
    """
    try:  # pragma: no cover - network path
        from urllib.request import urlopen
        import json

        url = f"https://huggingface.co/api/models/{pin.repo_id}"
        with urlopen(url, timeout=10) as resp:  # noqa: S310
            payload = json.load(resp)
        return payload.get("sha") == pin.revision_sha
    except Exception:  # pragma: no cover - network path
        return False
