"""Cross-substrate neuromorph validation pilot (C3.13, Phase 2 track b).

.. note::

    **DEFERRED to Paper 2** (PLOS CB pivot, 2026-04-20). This driver
    is preserved as the Paper 2 reactivation entry point. Cycle 3
    sem-3 quota interrupted execution before completion ; Paper 1
    v0.2 retargeted PLOS Computational Biology and §8.3 moved
    H1–H4 (and the C3.13 Norse vs MLX cross-substrate cell) out
    of Paper 1 scope. See
    ``docs/milestones/cycle3-plan-adaptation-2026-04-20.md`` for
    the full adaptation matrix and Paper 2 backlog. Do **not**
    re-run this script as part of the Paper 1 v0.2 critical path —
    its outputs are not consumed by the Paper 1 narrative.

**Gate ID** : G10a — cross-substrate neuromorphic validation
**Validates** : whether Norse SNN proxy ops (`*_snn.py`) produce
weight-trajectory effects correlated with the MLX real ops
(`*_real.py`) on a shared synthetic tiny-model fixture.
**Mode** : pipeline-validation — **synthetic 4×4 matrix**, not a
real Qwen model. Production validation lives in the Phase B real
pilot (1.5B Qwen FP16) currently holding the Studio.
**Expected output** :
  - ``docs/milestones/g10a-neuromorph.md`` (human-readable table)
  - ``docs/milestones/g10a-neuromorph.json`` (deterministic dump
    for R1 provenance)

## Why a pure-numpy MLX path

The real ops in ``kiki_oniric/dream/operations/*_real.py`` call
MLX directly (``mlx.core``, ``mlx.nn.Module``-bound models). Running
them here would require either a tiny MLX model fixture (pulls in
GPU init) or a mock. Neither is necessary for the C3.13 goal : we
only need to compare the **mathematical semantics** of each op
across the two substrates on a shared synthetic tiny weight matrix.

So this pilot uses a **pure-numpy MLX equivalent** of each real op :

- ``_apply_replay_real_np``  : SGD step on raw weights towards
  target (mirrors ``replay_real_handler`` MSE + SGD update).
- ``_apply_downscale_real_np`` : multiplicative shrink
  (mirrors ``downscale_real_handler``).
- ``_apply_recombine_real_np`` : VAE-style interpolation between
  two weight vectors with seeded epsilon, matching the
  ``recombine_real_handler`` reparameterization mean-path.

The Norse side calls the actual SNN handlers from
``kiki_oniric/dream/operations/*_snn.py`` — same numpy substrate,
spike-rate proxy semantics.

This "apples-to-apples numpy" comparison answers the C3.13
question directly : do the two op families produce correlated
weight trajectories under identical seeds ? If yes, DR-3
Conformance Criterion condition (3) (observability equivalence of
effects) holds for both substrates under the same framework-C
primitive contract.

## Cells

3 profiles × 30 seeds × 2 substrates = 180 cells (C3.13b
extension 2026-04-19 added ``p_max``).

- Profile ``p_min`` : 3 × replay on the adapter weights.
- Profile ``p_equ`` : 2 × (replay → downscale → recombine).
- Profile ``p_max`` : 2 × (replay → downscale → recombine →
  restructure). The restructure proxy is the axis-0 row swap
  supported by ``restructure_snn`` ; the MLX-equivalent is a
  numpy row swap on the same 4×4 weight matrix.

## GO / NO-GO rule

- **SOFT-GO** : Pearson rho ≥ 0.7 on p_min and p_equ **and** TOST
  equivalence (±0.2 SMD on Cohen's d of ``delta_norm``) holds on
  at least 2 of 3 profiles.
- **NO-GO** : Pearson rho < 0.3 on p_min or p_equ (substrate
  divergence → investigate SNN proxy fidelity before C3.14).
- Intermediate : report as CONDITIONAL-GO and defer final verdict
  to the Phase B real pilot.

Wall-clock : ~60 seconds on GrosMac (no GPU, pure numpy).

Usage ::

    uv run python scripts/pilot_phase2b_neuromorph.py

Reference :
    docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md
    §3 Phase 2 track b (neuromorph cross-substrate validation)
    docs/milestones/g10a-neuromorph.md (this script's milestone).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kiki_oniric.dream.operations.downscale_snn import (  # noqa: E402
    DownscaleSNNState,
    downscale_snn_handler,
)
from kiki_oniric.dream.operations.recombine_snn import (  # noqa: E402
    RecombineSNNState,
    recombine_snn_handler,
)
from kiki_oniric.dream.operations.replay_snn import (  # noqa: E402
    ReplaySNNState,
    replay_snn_handler,
)
from kiki_oniric.dream.operations.restructure_snn import (  # noqa: E402
    RestructureSNNState,
    restructure_snn_handler,
)
from kiki_oniric.dream.episode import (  # noqa: E402
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)


# --- Pilot configuration -------------------------------------------------
#
# C3.13b extension (2026-04-19) : added p_max profile (restructure
# included via axis-0 row swap on the 4×4 fixture). Three profiles
# are required so the H6 Jonckheere–Terpstra trend test (Goal e) can
# be applied per substrate. The SNN restructure proxy supports
# ``"reroute"`` with axis-0 swap_indices (cf restructure_snn.py
# §_SUPPORTED_TOPO_OPS) — the MLX-equivalent here is a numpy row
# swap matching that semantics.
#
# The original docstring (top of file) flags the symmetric-ρ
# limitation : restructure is a permutation, not a continuous flow,
# so cross-substrate Pearson rho on p_max is expected to be lower
# than on p_min / p_equ. Goal (a) therefore now reports both rho
# AND TOST equivalence on Cohen's d effect sizes (within ±0.2 SMD)
# per profile so the equivalence claim is robust to permutation
# noise.

PROFILES: tuple[str, ...] = ("p_min", "p_equ", "p_max")
SUBSTRATES: tuple[str, ...] = ("mlx", "norse")
SEEDS: tuple[int, ...] = tuple(range(30))

WEIGHT_SHAPE = (4, 4)
LR = 0.01
SHRINK_FACTOR = 0.98

# G10a verdict thresholds (Pearson rho on aligned (MLX, Norse)
# trajectories across seeds, per profile).
SOFT_GO_RHO = 0.7
NO_GO_RHO = 0.3

# TOST equivalence bounds on standardised Cohen's d (Goal a).
# Two substrates are deemed equivalent on a profile if the absolute
# Cohen's d between (MLX delta_norm, Norse delta_norm) lies within
# ±0.2 SMD (Cohen's "small" effect convention used as the smallest
# effect size of interest, SESOI).
TOST_SESOI_D = 0.2

MILESTONE_MD = REPO_ROOT / "docs" / "milestones" / "g10a-neuromorph.md"
MILESTONE_JSON = REPO_ROOT / "docs" / "milestones" / "g10a-neuromorph.json"


# --- Numpy-equivalent MLX real ops --------------------------------------


def _apply_replay_real_np(
    weights: np.ndarray, target: np.ndarray, lr: float
) -> None:
    """Numpy-equivalent of ``replay_real_handler`` SGD step.

    The real op builds a tiny MLX MLP whose loss is MSE(pred, y)
    and performs one SGD step on the parameters. For a 1-layer
    linear model with identity features, the gradient of MSE wrt
    weights reduces to ``2 * (weights - target)`` ; the SGD update
    is ``weights -= lr * grad = weights - 2 * lr * (weights - target)``.

    To mirror the MLX variant's in-place mutation contract, we
    update ``weights`` in place.
    """
    grad = 2.0 * (weights - target)
    weights[...] = weights - lr * grad


def _apply_downscale_real_np(
    weights: np.ndarray, factor: float
) -> None:
    """Numpy-equivalent of ``downscale_real_handler`` shrink."""
    if not (0.0 < factor <= 1.0):
        raise ValueError(
            f"shrink_factor must be in (0, 1], got {factor}"
        )
    weights[...] = weights * factor


def _apply_restructure_real_np(
    weights: np.ndarray, swap_indices: tuple[int, int]
) -> None:
    """Numpy-equivalent of ``restructure_real_handler`` row swap.

    The real op swaps two ``model.layers`` entries by reference.
    On the synthetic 4×4 weight fixture (no layer abstraction) we
    swap the corresponding rows along axis-0, which matches the
    SNN-proxy ``restructure_snn`` rate-channel swap semantics.
    """
    i, j = swap_indices
    n = weights.shape[0]
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError(
            f"S3: reroute swap_indices ({i}, {j}) out of bounds "
            f"for weights axis-0 of length {n}"
        )
    tmp = weights[i].copy()
    weights[i] = weights[j]
    weights[j] = tmp


def _apply_recombine_real_np(
    weights: np.ndarray, partner: np.ndarray, seed: int, episode_count: int
) -> None:
    """Numpy-equivalent of ``recombine_real_handler`` VAE interp.

    The real op draws epsilon from an isolated per-episode RNG
    (``seed + episode_count``) and emits ``z = mu + sigma * eps``.
    For the synthetic fixture we interpret ``weights`` as ``mu``
    and ``partner`` as ``log_var``, use ``alpha`` from the
    per-episode RNG to interpolate mean-path between the two
    operands (matching the ``recombine_snn`` semantics), then
    write the result back into ``weights``.
    """
    rng = np.random.default_rng(seed + episode_count)
    alpha = float(rng.random())
    weights[...] = alpha * weights + (1.0 - alpha) * partner


# --- Norse SNN pipeline wrappers ----------------------------------------


def _make_episode(
    input_slice: dict, operation: Operation, channel: OutputChannel
) -> DreamEpisode:
    """Helper : build a DreamEpisode with a generous BudgetCap."""
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice=input_slice,
        operation_set=(operation,),
        output_channels=(channel,),
        budget=BudgetCap(
            flops=10**9, wall_time_s=60.0, energy_j=100.0
        ),
        episode_id=f"g10a-{operation.value}",
    )


def _norse_replay(
    weights: np.ndarray, target: np.ndarray, lr: float
) -> None:
    """One SNN-proxy replay step — mutates ``weights`` in place."""
    state = ReplaySNNState()
    handler = replay_snn_handler(state, weights=weights, lr=lr)
    # ``replay_snn_handler`` expects target in the rate domain
    # directly — caller has to give it target_rates, not target
    # weights. Convert via the same sigmoid mapping used inside
    # the op so the MLX and Norse sides see a comparable target.
    from kiki_oniric.dream.operations.replay_snn import (
        weights_to_spike_rates,
    )
    target_rates = weights_to_spike_rates(target)
    episode = _make_episode(
        {"target_rates": target_rates},
        Operation.REPLAY,
        OutputChannel.WEIGHT_DELTA,
    )
    handler(episode)


def _norse_downscale(weights: np.ndarray, factor: float) -> None:
    """One SNN-proxy downscale step — mutates ``weights`` in place."""
    state = DownscaleSNNState()
    handler = downscale_snn_handler(state, weights=weights)
    episode = _make_episode(
        {"shrink_factor": factor},
        Operation.DOWNSCALE,
        OutputChannel.WEIGHT_DELTA,
    )
    handler(episode)


def _norse_restructure(
    weights: np.ndarray, swap_indices: tuple[int, int]
) -> None:
    """One SNN-proxy restructure step — mutates ``weights`` in place."""
    state = RestructureSNNState()
    handler = restructure_snn_handler(state, weights=weights)
    episode = _make_episode(
        {"topo_op": "reroute", "swap_indices": list(swap_indices)},
        Operation.RESTRUCTURE,
        OutputChannel.WEIGHT_DELTA,
    )
    handler(episode)


def _norse_recombine(
    weights: np.ndarray, partner: np.ndarray, seed: int
) -> None:
    """One SNN-proxy recombine step — mutates ``weights`` in place.

    The SNN recombine returns its result through
    ``state.last_sample`` rather than mutating weights. We extract
    the sample and reshape it back onto the weights tensor to
    preserve the cross-substrate in-place mutation contract on
    the synthetic fixture.
    """
    state = RecombineSNNState()
    handler = recombine_snn_handler(state, seed=seed)
    latents = [weights.ravel().tolist(), partner.ravel().tolist()]
    episode = _make_episode(
        {"delta_latents": latents},
        Operation.RECOMBINE,
        OutputChannel.LATENT_SAMPLE,
    )
    handler(episode)
    assert state.last_sample is not None
    weights[...] = np.asarray(
        state.last_sample, dtype=weights.dtype
    ).reshape(weights.shape)


# --- Profile orchestrators ----------------------------------------------


def apply_profile_real(
    profile: str,
    initial: np.ndarray,
    target: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Apply the MLX-equivalent numpy op sequence for ``profile``."""
    w = initial.copy()
    if profile == "p_min":
        # 3× replay (mirrors p_min's replay-dominated schedule)
        for _ in range(3):
            _apply_replay_real_np(w, target, LR)
    elif profile == "p_equ":
        # 2× (replay → downscale → recombine)
        rng_partner = np.random.default_rng(seed ^ 0xA5A5A5)
        partner = rng_partner.normal(size=w.shape).astype(np.float32)
        for ep in range(2):
            _apply_replay_real_np(w, target, LR)
            _apply_downscale_real_np(w, SHRINK_FACTOR)
            _apply_recombine_real_np(w, partner, seed, ep)
    elif profile == "p_max":
        # 2× (replay → downscale → recombine → restructure)
        # restructure swap_indices are seeded over (0..n_chan-1)
        # so different seeds exercise different row pairs while
        # remaining in-bounds for the 4×4 fixture.
        rng_partner = np.random.default_rng(seed ^ 0xA5A5A5)
        partner = rng_partner.normal(size=w.shape).astype(np.float32)
        rng_swap = np.random.default_rng(seed ^ 0x5A5A5A)
        n = w.shape[0]
        for ep in range(2):
            _apply_replay_real_np(w, target, LR)
            _apply_downscale_real_np(w, SHRINK_FACTOR)
            _apply_recombine_real_np(w, partner, seed, ep)
            i, j = sorted(rng_swap.choice(n, size=2, replace=False).tolist())
            _apply_restructure_real_np(w, (int(i), int(j)))
    else:
        raise ValueError(f"unknown profile: {profile!r}")
    return w


def apply_profile_snn(
    profile: str,
    initial: np.ndarray,
    target: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Apply the Norse SNN-proxy op sequence for ``profile``."""
    # Cast to float64 for the SNN sigmoid round-trip (the real
    # handlers use float64 internally via np.asarray(..., dtype=float)).
    w = initial.astype(np.float64).copy()
    target64 = target.astype(np.float64)
    if profile == "p_min":
        for _ in range(3):
            _norse_replay(w, target64, LR)
    elif profile == "p_equ":
        rng_partner = np.random.default_rng(seed ^ 0xA5A5A5)
        partner = rng_partner.normal(size=w.shape).astype(np.float64)
        for _ep in range(2):
            _norse_replay(w, target64, LR)
            _norse_downscale(w, SHRINK_FACTOR)
            _norse_recombine(w, partner, seed)
    elif profile == "p_max":
        rng_partner = np.random.default_rng(seed ^ 0xA5A5A5)
        partner = rng_partner.normal(size=w.shape).astype(np.float64)
        rng_swap = np.random.default_rng(seed ^ 0x5A5A5A)
        n = w.shape[0]
        for _ep in range(2):
            _norse_replay(w, target64, LR)
            _norse_downscale(w, SHRINK_FACTOR)
            _norse_recombine(w, partner, seed)
            i, j = sorted(rng_swap.choice(n, size=2, replace=False).tolist())
            _norse_restructure(w, (int(i), int(j)))
    else:
        raise ValueError(f"unknown profile: {profile!r}")
    return w.astype(np.float32)


# --- Cell pipeline ------------------------------------------------------


def run_cell(profile: str, substrate: str, seed: int) -> dict:
    """Execute one (profile, substrate, seed) cell of the pilot.

    Returns a dict with :
      - delta_norm       : ‖final − initial‖₂
      - target_distance  : ‖final − target‖₂
      - convergence      : fraction of initial→target distance
                           collapsed (1.0 = reached target).
      - final_flat       : weight vector (for pearson rho).
    """
    rng = np.random.default_rng(seed)
    target = rng.normal(size=WEIGHT_SHAPE).astype(np.float32)
    initial = rng.normal(size=WEIGHT_SHAPE).astype(np.float32)

    if substrate == "mlx":
        final = apply_profile_real(profile, initial, target, seed)
    elif substrate == "norse":
        final = apply_profile_snn(profile, initial, target, seed)
    else:
        raise ValueError(f"unknown substrate: {substrate!r}")

    delta = float(np.linalg.norm(final - initial))
    target_distance = float(np.linalg.norm(final - target))
    init_to_target = float(np.linalg.norm(initial - target))
    convergence = (
        (init_to_target - target_distance) / init_to_target
        if init_to_target > 1e-12
        else 0.0
    )

    return {
        "profile": profile,
        "substrate": substrate,
        "seed": seed,
        "delta_norm": delta,
        "target_distance": target_distance,
        "convergence": float(convergence),
        "final_flat": final.astype(np.float64).ravel().tolist(),
    }


# --- Aggregation --------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Pearson rho + two-sided p-value, guarded for degeneracy."""
    from scipy.stats import pearsonr  # lazy import

    if a.size != b.size or a.size < 2:
        return 0.0, 1.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0, 1.0
    r, p = pearsonr(a, b)
    return float(r), float(p)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (between-sample, pooled std, ddof=1).

    Sign convention : positive d ⇒ ``a`` > ``b`` on average.
    Returns 0.0 on degeneracy (zero pooled variance, n<2 etc).
    """
    if a.size < 2 or b.size < 2:
        return 0.0
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = ((a.size - 1) * var_a + (b.size - 1) * var_b) / (
        a.size + b.size - 2
    )
    if pooled <= 1e-24:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def _tost_equivalence(
    a: np.ndarray, b: np.ndarray, sesoi_d: float
) -> dict:
    """Two One-Sided Tests (TOST) equivalence on standardised scale.

    Bounds are expressed in Cohen's d (SESOI = ``sesoi_d``). We
    compute the lower-bound t-test (H0_lower : d ≤ -sesoi) and
    upper-bound t-test (H0_upper : d ≥ +sesoi) ; equivalence is
    declared when **both** one-sided p-values are below 0.05
    (Schuirmann, 1987).

    Returns a dict with ``cohens_d``, ``p_lower``, ``p_upper``,
    ``p_tost`` (= max of the two), and ``equivalent`` boolean.
    """
    from scipy.stats import ttest_ind  # lazy import

    if a.size < 2 or b.size < 2:
        return {
            "cohens_d": 0.0,
            "p_lower": 1.0,
            "p_upper": 1.0,
            "p_tost": 1.0,
            "equivalent": False,
        }
    # Convert SESOI from Cohen's d to raw mean-difference units.
    pooled_sd = float(
        np.sqrt(
            ((a.size - 1) * np.var(a, ddof=1)
             + (b.size - 1) * np.var(b, ddof=1))
            / (a.size + b.size - 2)
        )
    )
    if pooled_sd <= 1e-24:
        # Degenerate variance — declare equivalent if means match.
        equiv = abs(float(np.mean(a) - np.mean(b))) <= 1e-9
        return {
            "cohens_d": 0.0,
            "p_lower": 0.0 if equiv else 1.0,
            "p_upper": 0.0 if equiv else 1.0,
            "p_tost": 0.0 if equiv else 1.0,
            "equivalent": equiv,
        }
    delta = sesoi_d * pooled_sd
    # H0_lower : mean(a) - mean(b) ≤ -delta  → one-sided test on
    # shifted samples (a + delta) vs b ; reject if t large positive.
    t_lower, p_two_lower = ttest_ind(
        a + delta, b, equal_var=True
    )
    p_lower = (
        p_two_lower / 2.0 if t_lower > 0 else 1.0 - p_two_lower / 2.0
    )
    # H0_upper : mean(a) - mean(b) ≥ +delta  → reject if t small.
    t_upper, p_two_upper = ttest_ind(
        a - delta, b, equal_var=True
    )
    p_upper = (
        p_two_upper / 2.0 if t_upper < 0 else 1.0 - p_two_upper / 2.0
    )
    p_tost = max(p_lower, p_upper)
    return {
        "cohens_d": _cohens_d(a, b),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_tost": float(p_tost),
        "equivalent": bool(p_tost < 0.05),
    }


def aggregate(cells: list[dict]) -> dict:
    """Build per-profile × substrate stats + cross-substrate rho."""
    per_cell: dict[tuple[str, str, int], dict] = {
        (c["profile"], c["substrate"], c["seed"]): c for c in cells
    }

    by_profile_substrate: dict[str, dict] = {}
    for profile in PROFILES:
        by_profile_substrate[profile] = {}
        for substrate in SUBSTRATES:
            subset = [
                c
                for c in cells
                if c["profile"] == profile and c["substrate"] == substrate
            ]
            deltas = np.array([c["delta_norm"] for c in subset])
            convs = np.array([c["convergence"] for c in subset])
            by_profile_substrate[profile][substrate] = {
                "n": len(subset),
                "mean_delta": float(deltas.mean()),
                "std_delta": float(deltas.std(ddof=1)) if len(subset) > 1 else 0.0,
                "mean_convergence": float(convs.mean()),
                "std_convergence": (
                    float(convs.std(ddof=1)) if len(subset) > 1 else 0.0
                ),
            }

    cross: dict[str, dict] = {}
    for profile in PROFILES:
        mlx_vec = []
        norse_vec = []
        mlx_delta = []
        norse_delta = []
        for seed in SEEDS:
            mlx_cell = per_cell.get((profile, "mlx", seed))
            norse_cell = per_cell.get((profile, "norse", seed))
            if mlx_cell is None or norse_cell is None:
                continue
            mlx_vec.extend(mlx_cell["final_flat"])
            norse_vec.extend(norse_cell["final_flat"])
            mlx_delta.append(mlx_cell["delta_norm"])
            norse_delta.append(norse_cell["delta_norm"])
        rho, p_val = _pearson(np.array(mlx_vec), np.array(norse_vec))
        # Convergence ratio : Norse / MLX (>1 = Norse over-
        # converges, <1 = Norse under-converges).
        mlx_mean_conv = by_profile_substrate[profile]["mlx"]["mean_convergence"]
        norse_mean_conv = by_profile_substrate[profile]["norse"][
            "mean_convergence"
        ]
        ratio = (
            norse_mean_conv / mlx_mean_conv
            if abs(mlx_mean_conv) > 1e-12
            else 0.0
        )
        # TOST equivalence on delta_norm distributions.
        tost = _tost_equivalence(
            np.array(mlx_delta), np.array(norse_delta), TOST_SESOI_D
        )
        cross[profile] = {
            "pearson_rho": rho,
            "p_value": p_val,
            "convergence_ratio_norse_over_mlx": float(ratio),
            "tost": tost,
        }

    # G10a verdict — combine Pearson rho and TOST equivalence.
    # SOFT-GO requires :
    #   - rho ≥ 0.7 on p_min and p_equ (continuous-flow profiles), AND
    #   - TOST equivalence (within ±0.2 SMD) on at least 2/3 profiles.
    # NO-GO if any continuous-flow profile rho < 0.3.
    # p_max permutation noise expected to depress rho ; the TOST
    # check rescues equivalence claim under permutation noise.
    rho_min_equ = [cross["p_min"]["pearson_rho"], cross["p_equ"]["pearson_rho"]]
    tost_pass_count = sum(
        1 for p in PROFILES if cross[p]["tost"]["equivalent"]
    )
    if any(r < NO_GO_RHO for r in rho_min_equ):
        verdict = "NO-GO"
    elif all(r >= SOFT_GO_RHO for r in rho_min_equ) and tost_pass_count >= 2:
        verdict = "SOFT-GO"
    else:
        verdict = "CONDITIONAL-GO"

    return {
        "by_profile_substrate": by_profile_substrate,
        "cross_substrate": cross,
        "verdict": verdict,
        "tost_pass_count": tost_pass_count,
        "tost_sesoi_d": TOST_SESOI_D,
    }


# --- Main driver --------------------------------------------------------


def main() -> int:
    t0 = time.time()
    cells: list[dict] = []
    for profile in PROFILES:
        for substrate in SUBSTRATES:
            for seed in SEEDS:
                cells.append(run_cell(profile, substrate, seed))
    wall_s = time.time() - t0

    agg = aggregate(cells)

    # JSON dump — drop heavy `final_flat` fields, keep stats only.
    cells_slim = [
        {k: v for k, v in c.items() if k != "final_flat"} for c in cells
    ]
    dump = {
        "gate": "G10a",
        "milestone": "cross-substrate neuromorph validation (C3.13)",
        "harness_version": "C-v0.7.0+PARTIAL",
        "profiles": list(PROFILES),
        "substrates": list(SUBSTRATES),
        "seeds": list(SEEDS),
        "weight_shape": list(WEIGHT_SHAPE),
        "cells": cells_slim,
        "aggregate": agg,
        "wall_time_s": wall_s,
        "thresholds": {
            "soft_go_rho": SOFT_GO_RHO,
            "no_go_rho": NO_GO_RHO,
        },
    }

    MILESTONE_JSON.write_text(json.dumps(dump, indent=2, sort_keys=True))

    # Render milestone markdown.
    md = _render_markdown(dump)
    MILESTONE_MD.write_text(md)

    print(f"[G10a] verdict = {agg['verdict']}")
    for p in PROFILES:
        c = agg["cross_substrate"][p]
        print(
            f"[G10a] {p}: rho={c['pearson_rho']:.4f} "
            f"p={c['p_value']:.4g} "
            f"ratio={c['convergence_ratio_norse_over_mlx']:.4f}"
        )
    print(f"[G10a] wall-clock = {wall_s:.1f}s ({len(cells)} cells)")
    return 0


def _render_markdown(dump: dict) -> str:
    agg = dump["aggregate"]
    verdict = agg["verdict"]
    bps = agg["by_profile_substrate"]
    cross = agg["cross_substrate"]

    lines: list[str] = []
    lines.append(
        "# G10a — Cross-substrate neuromorphic validation (2026-04-19)"
    )
    lines.append("")
    lines.append(f"**Status** : **{verdict}**")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "- Profiles tested : `p_min`, `p_equ`, `p_max` (C3.13b "
        "extension — restructure axis-0 row swap added 2026-04-19)"
    )
    lines.append(
        "- Substrates : MLX (kiki-oniric real-op numpy equivalent) "
        "vs Norse (SNN-proxy ops, pure numpy)"
    )
    lines.append(
        f"- {len(SEEDS)} seeds per cell, {WEIGHT_SHAPE[0]}×"
        f"{WEIGHT_SHAPE[1]} synthetic tiny model"
    )
    lines.append(
        "- Pure-numpy implementation, no GPU / Norse / PyTorch runtime "
        "required"
    )
    lines.append(
        f"- Wall-clock : **{dump['wall_time_s']:.1f}s** on local "
        f"CPU (GrosMac)"
    )
    lines.append("")
    lines.append(
        "> **Synthetic caveat** (CLAUDE.md §3) : results below are "
        f"on a {WEIGHT_SHAPE[0]}×{WEIGHT_SHAPE[1]} synthetic weight "
        "matrix, **not a real model**. Production validation lives "
        "in the Phase B real pilot (1.5B Qwen FP16) currently "
        "executing on Studio. This pilot is a pipeline-validation "
        "artifact for DR-3 condition (3) cross-substrate "
        "observability only."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Profile | Substrate | mean Δ | std Δ | mean conv | std conv |"
    )
    lines.append(
        "|---|---|---|---|---|---|"
    )
    for profile in PROFILES:
        for substrate in SUBSTRATES:
            s = bps[profile][substrate]
            lines.append(
                f"| {profile} | {substrate} | "
                f"{s['mean_delta']:.4f} | {s['std_delta']:.4f} | "
                f"{s['mean_convergence']:.4f} | "
                f"{s['std_convergence']:.4f} |"
            )
    lines.append("")
    lines.append("## Cross-substrate correlation")
    lines.append("")
    lines.append(
        "| Profile | Pearson ρ(MLX, Norse) | p-value | "
        "conv ratio (Norse/MLX) |"
    )
    lines.append("|---|---|---|---|")
    for profile in PROFILES:
        c = cross[profile]
        lines.append(
            f"| {profile} | {c['pearson_rho']:.4f} | "
            f"{c['p_value']:.3g} | "
            f"{c['convergence_ratio_norse_over_mlx']:.4f} |"
        )
    lines.append("")
    lines.append(
        "## TOST equivalence (Goal a — Cohen's d ±0.2 SMD)"
    )
    lines.append("")
    lines.append(
        "| Profile | Cohen's d | p_lower | p_upper | p_TOST | "
        "Equivalent (α=0.05) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for profile in PROFILES:
        t = cross[profile]["tost"]
        lines.append(
            f"| {profile} | {t['cohens_d']:+.4f} | "
            f"{t['p_lower']:.3g} | {t['p_upper']:.3g} | "
            f"{t['p_tost']:.3g} | "
            f"{'YES' if t['equivalent'] else 'NO'} |"
        )
    lines.append("")
    lines.append(
        f"TOST pass count : **{agg['tost_pass_count']}/{len(PROFILES)}** "
        f"(SESOI = ±{TOST_SESOI_D} SMD ; Schuirmann 1987 two "
        f"one-sided t-tests on `delta_norm` distributions)."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"- **SOFT-GO** threshold : ρ ≥ {SOFT_GO_RHO} on p_min and "
        f"p_equ AND ≥ 2/3 profiles TOST-equivalent at ±"
        f"{TOST_SESOI_D} SMD"
    )
    lines.append(
        f"- **NO-GO** threshold : ρ < {NO_GO_RHO} on p_min or p_equ"
    )
    lines.append("")
    lines.append(
        "- High ρ (> 0.7) ⇒ MLX and Norse substrates produce "
        "correlated dream-op effects on the shared synthetic "
        "fixture → **DR-3 Conformance Criterion condition (3)** "
        "(observability equivalence of effects) holds for both "
        "substrates under the same primitive contract."
    )
    lines.append(
        "- Low ρ (< 0.3) ⇒ substrates diverge on the synthetic "
        "fixture → investigate SNN proxy fidelity (sigmoid "
        "round-trip saturation, spike-rate interpolation, etc.)."
    )
    lines.append(
        "- Intermediate ρ ⇒ CONDITIONAL-GO : defer final verdict "
        "to the Phase B real pilot."
    )
    lines.append("")
    lines.append("## References")
    lines.append("")
    lines.append(
        "- `kiki_oniric/dream/operations/*_real.py` (MLX variants ; "
        "this pilot uses a numpy equivalent of the same math — see "
        "`scripts/pilot_phase2b_neuromorph.py` module docstring)"
    )
    lines.append(
        "- `kiki_oniric/dream/operations/*_snn.py` (Norse SNN-proxy "
        "variants, invoked directly)"
    )
    lines.append(
        "- `docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-"
        "design.md` §3 Phase 2 track b"
    )
    lines.append(
        "- JSON dump : `docs/milestones/g10a-neuromorph.json` (R1 "
        "provenance artifact)"
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
