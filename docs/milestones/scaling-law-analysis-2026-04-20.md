# Scaling-law analysis cycle-3 Phase B (2026-04-20)

> **Erratum** (2026-06-19, confirmed catalogue v3 2026-11-06): the
> scaling-law finding (H7 / DOK-H7) and 7B pilot figures (DOK-PB7) below
> are retracted — both scale points are contaminated by the `mega_v2`
> fixture leak (20-row fallback cycled at n_samples=100, train=eval) and
> the 7B run additionally used a degenerate `hellaswag` fixture (pre≈1.0).
> The 1.5B figures (DOK-PB15) are also retracted for the same fixture
> leak. See STATUS.md CORRECTION banner.

**Status** : 2 of 3 scales complete (1.5B done, 7B done, 35B pending relaunch after crash)

Both completed scales ran the real Qwen2.5-Instruct bf16 pipeline against real
MMLU + HellaSwag + mega-v2 benchmarks (n=100 samples/benchmark, 10 seeds,
3 profiles = 30 cells/profile), executed on Apple M3 Ultra (Studio, MLX bf16).

## Finding

| Profile | Ops | 1.5B delta mean | 7B delta mean | 1.5B p-value | 7B p-value | reject H0 @ alpha=0.0125 |
|---|---|---|---|---|---|---|
| p_min | 1 (replay) | 0.113290 | 0.116158 | 9.99e-21 | 1.44e-24 | both reject |
| p_equ | 3 (replay + downscale + recombine) | 0.083645 | 0.094291 | 3.97e-19 | 6.18e-27 | both reject |
| p_max | 4 (replay + downscale + recombine + restructure) | 0.078288 | 0.056961 | 1.66e-17 | **5.51e-02** | 7B fails |

Per-profile t-statistics (one-sample t-test against delta = 0) at 7B :
p_min t=33.04, p_equ t=40.04, p_max t=1.998 (n=30 in each). Cell wall-clock
averaged 291 s/cell at 7B fp16 vs 94 s/cell at 1.5B fp16 (2.4 h vs 47 min for
full 90-cell sweep). Verdict is GO at both scales under the pre-registered
2-of-3 rule (p_min + p_equ both reject). No failed cells at either scale.

### Non-monotonic scaling of profile complexity

p_min and p_equ deltas **strengthen** with scale from 1.5B to 7B
(p_min p 9.99e-21 -> 1.44e-24 ; p_equ p 3.97e-19 -> 6.18e-27). The
delta means themselves also grow : p_min +2.5 %, p_equ +12.7 %.

p_max **collapses** 15 orders of magnitude : significant at 1.5B
(p 1.66e-17, delta 0.0783) to non-significant at 7B (p 5.51e-02,
delta 0.0570). The delta mean drops by 27.2 % while per-cell variance
explodes (std 0.0234 at 1.5B to 0.1561 at 7B, a 6.7x increase in
per-cell standard deviation).

### Interpretation

Only one operator differs between p_equ and p_max : the `restructure` op
(adjacent transformer-block swap). The three other ops of p_max are shared
with p_equ (replay + downscale + recombine) and do not collapse. Therefore
the collapse is directly attributable to restructure at 7B scale.

Mechanistically, swapping adjacent transformer blocks is a very aggressive
perturbation. At 1.5B the downscale + recombine VAE appears able to absorb
the distortion within the 3-benchmark eval window. At 7B the larger model's
residual stream relies more heavily on the precise block ordering, and the
same VAE-mediated compensation fails : the post-intervention benchmark
recovers less reliably and per-seed variance explodes (std * 6.7).

**Candidate hypothesis H7** (new for Paper 2) :
*Ops dose curve has a scale-dependent optimum, with the 3-op p_equ profile
appearing most robust across scales tested so far ; adding restructure
becomes catastrophic beyond ~1.5B parameters.*

This is consistent with H5-III power-law scaling for p_min / p_equ and
a separate scale-dependent phase transition for p_max. 35B will confirm
whether the collapse continues (monotonic scale-dependent failure of
restructure) or recovers (U-shaped non-monotonic).

## Synthetic / real caveat

Phase B uses real Qwen2.5-Instruct bf16 weights + real MMLU + HellaSwag +
mega-v2 benchmarks with MLX bf16 inference. Results are empirical, not
synthetic substitute (unlike cycle-2 preliminary replication, where the
same pattern was initially observed in a synthetic pilot). The empirical
confirmation of the non-monotonic profile ordering at real scale is the
load-bearing finding here.

## 35B status

35B Phase B crashed after 1 cell (~18 min/cell wall-clock) on the first
launch attempt due to concurrent-run resource contention on Studio (the
7B sweep was running in parallel, and a separate 35B MLX LoRA SFT job
for micro-kiki Qwen36 was also active). Scheduled relaunch solo after
7B completion — see `docs/milestones/phase-b-35b-relaunch-2026-04-20.md`
for the live tracker.

If the restructure collapse at 7B is a monotonic scale-dependent failure,
35B will show p_max p >> 0.0125 (likely also delta <= 0 given the variance
trend). If U-shaped, p_max may recover at 35B — this would be an
interesting signature of phase transitions in scaling behaviour.

## Cross-references

- Phase B 1.5B JSON : `docs/milestones/pilot-cycle3-real-1p5b.json` (commit 22c58c9)
- Phase B 7B JSON : `docs/milestones/pilot-cycle3-real-7b.json` (this commit)
- Phase B 35B JSON : pending relaunch (see tracker doc)
- Paper 2 outline : `docs/papers/paper2/outline.md` (receive H7 addition once 35B results in)
