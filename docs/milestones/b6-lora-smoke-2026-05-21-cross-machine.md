# b6-lora-smoke-2026-05-21 — cross-machine summary

**Date** : 2026-05-21
**Framework** : C-v0.24.1+PARTIAL
**Package** : 0.22.1
**Trigger commit** : `216700b` (post-B3 delegate refactor)
**Seed** : `42`
**Sibling JSONs** :
- `b6-lora-smoke-2026-05-21-apple_m5.json`
- `b6-lora-smoke-2026-05-21-apple_m3_ultra.json`
- `b6-lora-smoke-2026-05-21-apple_m1_max.json`

---

First three-machine run of `scripts/pilot_b6_lora_smoke.py` —
Apple M5 (grosmac), Apple M3 Ultra (Studio), Apple M1 Max
(macM1) — same code, same MLX 0.31.1, same seed=42, same
synthetic workload (identical core + per-tier extras).

## Per-tier verdict per machine

| Tier | M5 | M3 Ultra | M1 Max | dispatch | flops | bit_equal |
|---|---|---|---|---|---|---|
| `PMinLoRA`  | ✅ | ✅ | ✅ | 3 | 856 | True everywhere |
| `PEquLoRA`  | ✅ | ✅ | ✅ | 4 | 858 | True everywhere |
| `PMaxLoRA`  | ✅ | ✅ | ✅ | 5 | 874 | True everywhere |

**Dispatch counts** (`3 / 4 / 5`) match across all three
machines bit-perfectly. **K1 FLOPs estimates** (`856 / 858 /
874`) are also identical — the FLOP estimate is a pure
function of the op shapes, no hardware-dependent factors.

## DR-4 chain inclusion

| Machine | `ops_strict_subset` | `emitters_strict_subset` | verdict |
|---|---|---|---|
| M5 | ✅ | ✅ | PASS |
| M3 Ultra | ✅ | ✅ | PASS |
| M1 Max | ✅ | ✅ | PASS |

All three machines confirm the strict-subset chain :
- `ops` (active emitters) : `PMin = {REPLAY, DOWNSCALE} ⊂
  PEqu ⊂ PMax = PMin ∪ {RESTRUCTURE, RECOMBINE}`
- emitted channel types : `{WeightUpdate} ⊂
  {WeightUpdate, TopologyDiff} ⊂ {WeightUpdate, TopologyDiff,
  LatentSample}`

## Within-machine R1 verdict

The pilot's `bit_equal` field measures *within-machine* R1 :
build dream + awake `LoRAModel(seed=42)` clones, run the
workload, call `consolidate_log()`, then compare adapter
arrays. All three machines pass on all three tiers ; the
`apply_channel_outputs` loop produces an awake-side model
bit-equal to the dream-side mutation result, every time.

This is consistent with the 2026-05-20 R1 cross-machine
finding (`docs/milestones/r1-cross-machine-m5-vs-m1-2026-05-20.md`,
`docs/proofs/r1-cross-machine.md`) :

- Within-machine R1 is **BLOCKING** and intact — this pilot
  confirms that property holds across all three machines for
  the new B-series surface.
- Cross-machine R1 is **WARN-conditional** — the pilot does
  not measure it directly (each machine compares only its
  own dream/awake pair). To probe cross-machine bit-equality
  one would have to diff the raw float bytes of the awake
  models across machines, which we don't do here.

The pilot's `total_flops`, `dispatch_count`, and qualitative
`emitted_types` set are all hardware-agnostic, so we expect
them to match across machines — and they do.

## Wall-time

Informational only (no SLA) :

| Tier | M5 | M3 Ultra | M1 Max |
|---|---|---|---|
| `PMinLoRA` | ~0.029 s | ~0.673 s* | ~0.030 s |
| `PEquLoRA` | ~0.008 s | ~0.030 s | ~0.009 s |
| `PMaxLoRA` | ~0.009 s | ~0.064 s | ~0.010 s |

*M3 Ultra's PMinLoRA wall_s is dominated by Metal kernel
warm-up on the first MLX op of the run (the JIT compile path
is cold). Subsequent tiers run in normal time. M5 and M1 Max
were warmed by earlier pytest runs ; M3 Ultra was a fresh
process.

## What this does not measure

- **Cross-machine bit-equality of awake-side tensors** —
  each machine asserts dream == awake locally ; we don't
  diff awake-tensor bytes across machines. The 2026-05-20
  R1 probe handles that for the underlying ops.
- **Benchmark accuracy** — synthetic workload only.
- **Performance SLA** — wall_s is informational.
- **No new empirical claim** — EC stays `+PARTIAL`.

## Provenance

- M5 (grosmac) : first-run audit, captured in the earlier
  single-file 2026-05-21 milestone (now renamed to the
  `-apple_m5` suffix in commit `979f1d3`).
- M3 Ultra (Studio) : run after `uv sync --all-extras` +
  `git pull --rebase` at `216700b`. Captured locally to
  `/Users/clems/dream-of-kiki-r1-probe/docs/milestones/`,
  copied here as `-apple_m3_ultra`.
- M1 Max (macM1) : same flow, captured to
  `/Users/electron/dream-of-kiki/docs/milestones/`, copied
  here as `-apple_m1_max`.
