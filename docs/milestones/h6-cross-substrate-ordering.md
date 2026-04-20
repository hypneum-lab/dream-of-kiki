# G10e / H6 — Cross-substrate profile ordering (2026-04-19)

**Status** : **SOFT-GO**

## Summary

- Hypothesis H6 : each substrate exhibits a strictly monotonic ordering of ``delta_norm`` across profiles (``p_min``, ``p_equ``, ``p_max``), tested via the Jonckheere–Terpstra trend statistic.
- Data source : ``docs/milestones/g10a-neuromorph.json`` (30 seeds × 3 profiles × 2 substrates = 180 cells).
- Decision rule : reject H0_6 when ``|z| > 1.96`` (two-sided α = 0.05).
- Cross-substrate claim : H6 holds substrate-agnostically when **both** substrates reject H0_6 in the same direction.

> **Synthetic caveat** (CLAUDE.md §3) : results below are on a 4×4 synthetic weight matrix, **not a real model**. Production H6 validation lives in Paper 2 with real-scale Qwen pilots. This pilot is a pipeline-validation artifact for DR-3 condition (3) cross-substrate observability only.

## Per-substrate group statistics

| Substrate | Profile | n | mean Δ | std Δ | median Δ |
|---|---|---|---|---|---|
| mlx | p_min | 30 | 0.3251 | 0.0553 | 0.3355 |
| mlx | p_equ | 30 | 4.0228 | 1.4735 | 4.3432 |
| mlx | p_max | 30 | 4.7568 | 1.2500 | 4.8269 |
| norse | p_min | 30 | 0.2204 | 0.0831 | 0.2063 |
| norse | p_equ | 30 | 3.6046 | 1.6881 | 3.6774 |
| norse | p_max | 30 | 4.6090 | 1.2319 | 4.8729 |

## Jonckheere–Terpstra per substrate

| Substrate | J | z | p (two-sided) | p (↑) | p (↓) | Reject H0_6 | Direction |
|---|---|---|---|---|---|---|---|
| mlx | 2377.0 | +7.607 | 2.8e-14 | 1.4e-14 | 1 | YES | increasing |
| norse | 2417.0 | +7.904 | 2.71e-15 | 1.35e-15 | 1 | YES | increasing |

## Cross-substrate H6 verdict

- MLX rejects H0_6 : YES
- Norse rejects H0_6 : YES

**H6 HOLDS substrate-agnostically** — both substrates exhibit a monotonic ordering in the `increasing` direction.

## References

- G10a driver : ``scripts/pilot_phase2b_neuromorph.py`` (C3.13 ; produces the ``delta_norm`` distributions consumed here).
- Jonckheere (1954), Terpstra (1952) ordered-alternatives rank test ; asymptotic normal approximation used.
- Cross-substrate data : ``docs/milestones/g10a-neuromorph.json``.
- Spec : ``docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md`` §3 Phase 2 track b (H6 ordering).
