# Cycle-3 Sanity Pilot Report — 1.5B-Scale Fail-Fast

**Gate** : G10 cycle-3 Gate D *pilot* (fail-fast decision)
**Scale** : `qwen3p5-1p5b` only (Qwen2.5-1.5B MLX-Q4 fallback)
**Launched on** : `{RESULTS TBD — populated after run}`
**Studio session ID** : `{RESULTS TBD}`
**Harness version** : `C-v0.7.0+PARTIAL`
**Script** : [`scripts/pilot_cycle3_sanity.py`](../../scripts/pilot_cycle3_sanity.py)
**Decision dump** : [`pilot-cycle3-sanity-1p5b.json`](./pilot-cycle3-sanity-1p5b.json)

---

## Purpose

Before burning **~10 days** of Studio compute on the full 3-scale
1080-config launch (C3.8), exercise the full cycle-3 pipeline at
the **smallest** scale-slot and verify that H1 (forgetting
reduction, Welch pre vs post on retained accuracy) is detectable.
If H1 is dead at 1.5B, the 7B and 35B launches are almost
certainly wasted compute — abort before the commitment.

## Plan

```
scale       : qwen3p5-1p5b   (1 scale)
profiles    : p_min, p_equ, p_max  (3 profiles)
substrates  : mlx_kiki_oniric, esnn_thalamocortical  (2 substrates)
seeds       : 0..29  (30 seeds — half the 60-seed full launch)
total cells : 1 × 3 × 2 × 30 = 180 runs
compute     : ~1 day dedicated Studio M3 Ultra 512GB
```

Every cell's `run_id` is a strict subset of the full 1080-config
launch's resume contract (same
`(harness_version, profile_tag, seed, commit_sha)` tuple, same
composite `profile_tag` scheme). Re-running `ablation_cycle3.py
--resume` after the pilot therefore **skips** the 180 pilot cells
automatically (R1 identity).

## GO / NO-GO Rule

- **GO** : H1 (Welch pre vs post) rejects the null in **≥ 4 / 6**
  cells at **α = 0.0125** (cycle-1 Bonferroni-corrected bar,
  family size 4 = {H1, H2, H3, H4}). Clear to launch C3.8 full
  3-scale matrix.
- **NO-GO** : H1 rejects in **< 4 / 6** cells → **abort**. Do
  not launch C3.8. Open a root-cause review (`pivot-4` branch
  per cycle-3 spec §5.1 R3).

The 6 cells are `{p_min, p_equ, p_max} × {mlx_kiki_oniric,
esnn_thalamocortical}` at scale `qwen3p5-1p5b`.

## Results

### Runs completed

`{RESULTS TBD — populated after run}` / 180

### Wall-clock

`{RESULTS TBD}` hh:mm:ss — used to extrapolate 7B (≈ 4×) and
35B (≈ 20× per §7 scale-cost model).

### H1 per cell

| cell (profile × substrate) | p-value | reject H0 (α = 0.0125) |
|----------------------------|---------|------------------------|
| p_min × mlx_kiki_oniric | `{TBD}` | `{TBD}` |
| p_min × esnn_thalamocortical | `{TBD}` | `{TBD}` |
| p_equ × mlx_kiki_oniric | `{TBD}` | `{TBD}` |
| p_equ × esnn_thalamocortical | `{TBD}` | `{TBD}` |
| p_max × mlx_kiki_oniric | `{TBD}` | `{TBD}` |
| p_max × esnn_thalamocortical | `{TBD}` | `{TBD}` |

### Verdict

`{RESULTS TBD — GO / NO-GO populated after run}`

### 7B + 35B extrapolation (if GO)

Wall-clock × 4 (7B) + × 20 (35B) → ~`{TBD}` total Studio days
to close C3.8.

## Notes

- Script ships in *plan-only* mode at commit time (`C3.7`). The
  per-cell `evaluate_retained` wiring is deferred to a follow-up
  commit so the user can decide when to launch ; the template +
  decision rule + run-registry identity are all locked in now.
- `--dry-run` validates the enumeration without touching any
  predictor / substrate — safe from CI.
- If GO fires, update the `{RESULTS TBD}` markers with the
  measured values and keep this report immutable per the
  `scripts/CLAUDE.md` "don't edit pilots after gate decision"
  rule (create `pilot_cycle3_sanity_v2.py` if methodology shifts).

## References

- `scripts/pilot_cycle3_sanity.py` — driver.
- `scripts/ablation_cycle3.py` — parent 1080-config runner.
- `kiki_oniric/eval/statistics.py` — `welch_one_sided` H1 test.
- `docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md`
  §5.1 R3 (Pivot 4 if NO-GO), §7 (compute budget).
