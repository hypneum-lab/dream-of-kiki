# G10a — Cross-substrate neuromorphic validation (2026-04-19)

**Status** : **CONDITIONAL-GO**

## Summary

- Profiles tested : `p_min`, `p_equ`, `p_max` (C3.13b extension — restructure axis-0 row swap added 2026-04-19)
- Substrates : MLX (kiki-oniric real-op numpy equivalent) vs Norse (SNN-proxy ops, pure numpy)
- 30 seeds per cell, 4×4 synthetic tiny model
- Pure-numpy implementation, no GPU / Norse / PyTorch runtime required
- Wall-clock : **0.1s** on local CPU (GrosMac)

> **Synthetic caveat** (CLAUDE.md §3) : results below are on a 4×4 synthetic weight matrix, **not a real model**. Production validation lives in the Phase B real pilot (1.5B Qwen FP16) currently executing on Studio. This pilot is a pipeline-validation artifact for DR-3 condition (3) cross-substrate observability only.

## Results

| Profile | Substrate | mean Δ | std Δ | mean conv | std conv |
|---|---|---|---|---|---|
| p_min | mlx | 0.3251 | 0.0553 | 0.0588 | 0.0000 |
| p_min | norse | 0.2204 | 0.0831 | 0.0374 | 0.0087 |
| p_equ | mlx | 4.0228 | 1.4735 | 0.0528 | 0.1837 |
| p_equ | norse | 3.6046 | 1.6881 | 0.0808 | 0.1832 |
| p_max | mlx | 4.7568 | 1.2500 | 0.0572 | 0.2055 |
| p_max | norse | 4.6090 | 1.2319 | 0.0777 | 0.1955 |

## Cross-substrate correlation

| Profile | Pearson ρ(MLX, Norse) | p-value | conv ratio (Norse/MLX) |
|---|---|---|---|
| p_min | 0.9994 | 0 | 0.6368 |
| p_equ | 0.9169 | 7.35e-193 | 1.5301 |
| p_max | 0.8363 | 7.38e-127 | 1.3589 |

## TOST equivalence (Goal a — Cohen's d ±0.2 SMD)

| Profile | Cohen's d | p_lower | p_upper | p_TOST | Equivalent (α=0.05) |
|---|---|---|---|---|---|
| p_min | +1.4833 | 9.36e-09 | 1 | 1 | NO |
| p_equ | +0.2640 | 0.0388 | 0.597 | 0.597 | NO |
| p_max | +0.1191 | 0.111 | 0.378 | 0.378 | NO |

TOST pass count : **0/3** (SESOI = ±0.2 SMD ; Schuirmann 1987 two one-sided t-tests on `delta_norm` distributions).

## Interpretation

- **SOFT-GO** threshold : ρ ≥ 0.7 on p_min and p_equ AND ≥ 2/3 profiles TOST-equivalent at ±0.2 SMD
- **NO-GO** threshold : ρ < 0.3 on p_min or p_equ

- High ρ (> 0.7) ⇒ MLX and Norse substrates produce correlated dream-op effects on the shared synthetic fixture → **DR-3 Conformance Criterion condition (3)** (observability equivalence of effects) holds for both substrates under the same primitive contract.
- Low ρ (< 0.3) ⇒ substrates diverge on the synthetic fixture → investigate SNN proxy fidelity (sigmoid round-trip saturation, spike-rate interpolation, etc.).
- Intermediate ρ ⇒ CONDITIONAL-GO : defer final verdict to the Phase B real pilot.

## References

- `kiki_oniric/dream/operations/*_real.py` (MLX variants ; this pilot uses a numpy equivalent of the same math — see `scripts/pilot_phase2b_neuromorph.py` module docstring)
- `kiki_oniric/dream/operations/*_snn.py` (Norse SNN-proxy variants, invoked directly)
- `docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md` §3 Phase 2 track b
- JSON dump : `docs/milestones/g10a-neuromorph.json` (R1 provenance artifact)
