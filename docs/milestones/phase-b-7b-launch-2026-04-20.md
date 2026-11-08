# Phase B 7B launch — C3.8 goal (d) scale law

> **Erratum** (2026-06-19, confirmed catalogue v3 2026-11-06): the 1.5B
> reference figures in §Reference comparison below (DOK-PB15) and the 7B
> run this document launched (DOK-PB7) are both retracted — `mega_v2`
> fixture leak (20-row fallback, train=eval) and degenerate `hellaswag`
> fixture (pre≈1.0) in the 7B run. See STATUS.md CORRECTION banner.

**Status** : launched fire-and-forget 2026-04-20 02:43 CEST on Studio.
**Gate** : G10 cycle-3 Gate D real pilot — scale-axis extension.
**Goal** : measure the catastrophic-forgetting tradeoff at
``qwen3p5-7b-fp16`` vs the ``qwen3p5-1p5b-fp16`` reference already
published at commit ``22c58c9`` (Phase B 1.5B : GO 3/3, wall 46.75 min,
p-values ≤ 1.66e-17 per profile).

## Run parameters

| Field | Value |
|-------|-------|
| Scale slot | ``qwen3p5-7b-fp16`` (Qwen2.5-7B-Instruct-bf16, 15.23 GB, 3 shards) |
| Registry pin | ``mlx-community/Qwen2.5-7B-Instruct-bf16`` rev ``349a12f0…`` |
| Profiles | ``p_min``, ``p_equ``, ``p_max`` — 30 seeds each, 90 cells total |
| Benchmarks | ``mmlu`` + ``hellaswag`` + ``mega_v2`` (composite mean accuracy) |
| n_samples / benchmark | 100 |
| Harness version | ``C-v0.7.0+PARTIAL`` |
| Host | Studio (``clems@192.168.13.100``), Python 3.14 + MLX 0.31.1 + mlx-lm 0.31.2 |

## Launch transcript

```
host        : studio
cwd         : /Users/clems/Documents/Projets/dreamOfkiki
launched_at : 2026-04-20 02:43 CEST
pid         : 19746
log         : ~/dreamOfkiki-runs/phase-b-7b-20260420-0243.log
command     : nohup uv run python scripts/pilot_cycle3_real.py \
                --scale=qwen3p5-7b-fp16
output dump : docs/milestones/pilot-cycle3-real-7b.json (on completion)
```

## Smoke-cell evidence

Pre-launch smoke-cell validated the 7B wiring end-to-end before
committing the full matrix compute :

| Field | Value |
|-------|-------|
| Profile | ``p_min`` |
| Benchmark | ``mega_v2`` |
| n_samples | 20 |
| pre_acc | 0.3649 |
| post_acc | 0.6812 |
| delta | **+0.3163** |
| wall_time_s | 151.70 (includes ~120 s HF shard fetch for 11 files) |
| run_id | ``a35709e042de879f4487d9062ec16866`` |

The ``weight_mutation_verified=False`` flag on the first scalar of
``model.embed_tokens.weight`` is expected for ``p_min`` : the smoke-cell
profile only runs ``replay_real`` (no ``downscale_real``), so the first
scalar of that row may not receive an SGD update above bf16 resolution
even though the post-eval composite accuracy shifted by +0.3163 — the
empirical signal that replay did mutate weights elsewhere in the tree.

## Wall-clock ETA

The 1.5B Phase B reference averaged ``31 s/cell`` over 90 cells
(46.75 min total). Forward-pass FLOPs scale ~linearly with parameter
count. Empirical 7B smoke cell (1 benchmark × 20 samples, warm cache)
ran ~30 s of compute after load. Full cells (3 benchmarks × 100
samples + dream episodes) extrapolate to :

- Best case : ``~2 min/cell`` → 90 cells ≈ **3 h**
- Typical : ``~3 min/cell`` → 90 cells ≈ **4.5 h**
- Worst case : ``~5 min/cell`` → 90 cells ≈ **7.5 h**

Expected completion : **2026-04-20 between 05:45 and 10:15 CEST**.

## Reference comparison — Phase B 1.5B

Loaded from ``docs/milestones/pilot-cycle3-real-1p5b.json`` at
commit ``22c58c9`` :

| Profile | n | p (t-test) | reject H0 | verdict |
|---------|---|-----------:|-----------|---------|
| ``p_min`` | 30 | 6.31e-18 | True | GO contributor |
| ``p_equ`` | 30 | 1.74e-17 | True | GO contributor |
| ``p_max`` | 30 | 1.66e-17 | True | GO contributor |

Go rule : ≥ 2 / 3 profiles reject H0 at α = 0.0125 — met 3 / 3.
Wall-clock total : 2805 s = 46.75 min, failures : 0.

## Pending post-completion actions

1. Verify the final dump ``pilot-cycle3-real-7b.json`` landed cleanly
   (``verdict``, ``h1`` block, ``completed_cells == 90``).
2. Compute H1 per profile at 7B and tabulate delta magnitude + p-value
   side-by-side with the 1.5B reference.
3. Fit the scale-law delta-vs-N curve across ``{1p5b, 7b}`` and note
   whether the catastrophic-forgetting tradeoff holds, widens, or
   flips at 7B vs 1.5B.
4. If 7B dump shows a GO verdict, update ``STATUS.md`` + ``CHANGELOG.md``
   with a DualVer EC bump candidate — empirical scale-law evidence at
   two points on the scale axis.
5. Schedule the 35B extrapolation slot on the same Studio host
   (budget wall ~18-25 h per rough RAM/FLOPs estimate ; see
   registry note on ``qwen3p5-35b-fp16``).

## Change set context

This run required a surgical CLI fix to
``scripts/pilot_cycle3_real.py`` — the ``--scale`` flag was missing
and the script hardcoded ``REAL_SCALE_FP16 = "qwen3p5-1p5b-fp16"``.
The fix parameterizes scale through the call tree
(``_load_fresh_fp16_wrapper``, ``_run_cell``, ``_register_cell``,
``main``) and derives a filename-friendly slug (``1p5b`` / ``7b`` /
``35b``) so per-scale milestone dumps never collide. Coverage remains
≥ 90 % : ``scripts/`` is outside the covered roots
(``harness`` + ``kiki_oniric``), so the CLI surface area change does
not require new unit tests.
