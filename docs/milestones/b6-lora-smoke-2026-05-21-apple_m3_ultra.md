# b6-lora-smoke-2026-05-21

**Date** : 2026-05-21  
**Framework** : C-v0.23.0+PARTIAL  
**Package** : 0.21.0  
**Commit** : `979f1d3dbdcc`  
**Chip family** : `apple_m3_ultra`  
**Seed** : `42`  
**Sibling JSON** : `b6-lora-smoke-2026-05-21.json`

---

First script-level exercise of the closed awake/dream loop across the three LoRA profile tiers, post-B6c. Synthetic workload, within-machine R1 only. No new empirical claim — infra-validation milestone.

## Per-tier metrics

| tier | wall_s | dispatch | bit_equal | flops | emitted |
|---|---|---|---|---|---|
| `PMinLoRA` | 0.673490 | 3 | True | 856 | WeightUpdate |
| `PEquLoRA` | 0.030255 | 4 | True | 858 | TopologyDiff, WeightUpdate |
| `PMaxLoRA` | 0.064218 | 5 | True | 874 | LatentSample, TopologyDiff, WeightUpdate |

## DR-4 chain inclusion

- `ops_strict_subset` : **True**
- `emitters_strict_subset` : **True**
- Verdict : **PASS**

## What this does not measure

- Cross-machine R1 — bit-equality is within-machine only ; for cross-machine see `docs/milestones/r1-cross-machine-m5-vs-m1-2026-05-20.{md,json}`.
- Benchmark accuracy — synthetic workload, no held-out evaluation.
- Performance SLA — `wall_s` is informational.
- No empirical claim — EC stays `+PARTIAL`.
