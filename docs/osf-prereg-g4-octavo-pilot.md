# G4-octavo pilot pre-registration

**Date:** 2026-05-04
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**Sister pilot:** G4-septimo (H6-B confirmed, H6-C universality
**CONFIRMED** at the four-benchmark × four-CNN-or-MLP scope ceiling
— commit `c8dd268`).
**Substrate:** MLX ViT-tiny (`G4ViTTiny`, patch=4, dim=192,
depth=4, heads=3, mlp_dim=384, dropout=0.0, n_classes=20 per
task, 64×64 RGB input ; ~1.8 M params).
**Benchmark:** Split-Tiny-ImageNet (10 sequential 20-class tasks,
identical to G4-septimo Step 1 — re-uses `tiny_imagenet_dataset.
load_split_tiny_imagenet_10tasks_auto` to keep cell-level R1
parity with the CNN run).
**Compute:** N = 30 seeds/arm × 2 strategies × 4 arms = 240 cells.
Per-cell wall ≈ 50-90 s on Studio M3 Ultra (transformer attention
is heavier than CNN ; ViT-tiny is the smallest viable transformer
that still tests the question). Total ≈ 4-6 h overnight on Studio
(Path A) — accepted under §9 envelope c if rate stays ≤ 90 s.
**Lock commit:** *(filled by introducing commit hash)*
**Lock timestamp:** 2026-05-04 (pre-driver-run).

## §1 Background

G4-septimo (commit `c8dd268`, milestones
`docs/milestones/g4-septimo-{step1,aggregate}-2026-05-04.{json,md}`)
closed the H6-C universality conjunction at its full pre-registered
scope : RECOMBINE-empty across {Split-FMNIST, Split-CIFAR-10,
Split-CIFAR-100, Split-Tiny-ImageNet} × {3-layer MLP, 5-layer
MLP, small CNN, medium CNN}. The DR-4 prediction "richer ops
yield richer consolidation" is empirically refuted across the
entire CNN-and-MLP escalation ladder at ≤ 200 classes per benchmark.

The DR-4 evidence v0.6 amendment (`docs/proofs/dr4-profile-inclusion.md`)
explicitly flags transformer substrates as the largest open
empirical hole : the four-benchmark × four-substrate scope ceiling
applies only to *classification with feed-forward / convolutional*
heads. A transformer substrate could in principle restore the
predicted RECOMBINE effect, since attention-based architectures
capture long-range dependencies in a fundamentally different way
than CNNs. G4-octavo fires that follow-up at the **smallest
practical transformer scale** — ViT-tiny on the same Tiny-ImageNet
benchmark used by G4-septimo — to keep the architecture-only
contrast clean.

The Hu 2020 anchor (g = 0.29, human sleep-dependent memory
consolidation) is used here strictly as a directional reference,
not a magnitude calibrator (cf. all sister pilot pre-regs).

## §2 Hypotheses (confirmatory)

- **H7-A (transformer architectural break-through hypothesis)** —
  on Split-Tiny-ImageNet with the ViT-tiny substrate (`G4ViTTiny`,
  ~1.8 M params, n_classes = 20 per-task head, 64×64 RGB input),
  `retention(P_max with mog)` is statistically distinguishable
  from `retention(P_max with none)`. Test : Welch two-sided rejects
  H0 at α = 0.05 (single new test, no Bonferroni inheritance from
  the closed G4-{quater..septimo} cycle), with the predicted
  positive sign `mean(mog) > mean(none)` and `Hedges' g ≥ 0.5`
  (medium effect size threshold for a "real" architectural break-
  through). **Rejecting H0 with the predicted sign** is the H7-A
  positive empirical claim ; rejecting with the opposite sign
  (RECOMBINE *hurts*) is a Row 4 outcome (negative-direction
  break-through, framework's claim further weakened) ; failing to
  reject is the H7-A null which extends the CNN scope ceiling to
  the small-transformer tier.

- **H7-B (transformer scope-extension hypothesis)** — derived :
  if H7-A null (fail-to-reject), the RECOMBINE-empty universality
  flag extends from the closed CNN scope to include the small-
  transformer tier (Tiny-ImageNet 200-class with ViT-tiny). If
  H7-A confirmed (positive sign), the universality is shown to
  break at the transformer architecture and the framework's
  RECOMBINE prediction is restored at the transformer scale.

No additional Welch test for H7-B ; it is logical aggregation.

## §3 Power analysis

N = 30 seeds per arm at α = 0.05 detects |g| ≥ 0.74 at 80 % power
(Welch two-sided). The H7-A threshold of `g ≥ 0.5` is below the
detection floor at this N — so a confirmed H7-A here would only
register at ~50 % power. To formalise this, the H7-A confirmation
threshold is set conservatively : `g ≥ 0.5 AND p < 0.05` (both
conditions). Sub-threshold positive-sign effects are reported as
exploratory. Identical N to all sister pilots in the G4-{quater..
septimo} ladder for direct comparability.

## §4 Exclusion criteria

- multi-class exclusion floor : `acc_initial < 2 × random_chance =
  0.10` for n_classes = 20 — exclude cell. Identical to G4-septimo.
- `acc_final` non-finite — exclude cell.
- run_id collision with prior pilot's registry — abort + amend.

## §5 Substrate / driver paths

- Driver : `experiments/g4_octavo_test/run_step1_vit_tiny.py`
- Substrate : `experiments.g4_octavo_test.vit_tiny.G4ViTTiny` (new,
  patch=4, dim=192, depth=4, heads=3, mlp_dim=384, n_classes=20)
- Loader : re-uses
  `experiments.g4_septimo_test.tiny_imagenet_dataset.load_split_tiny_imagenet_10tasks_auto`
  (no new SHA-256 ; cell-level R1 parity with G4-septimo
  preserved).
- Aggregator : `experiments/g4_octavo_test/aggregator.py` (emits
  H7-A verdict + H7-B scope-extension flag).
- Source : same HF parquet `zh-plus/tiny-imagenet` shards already
  pinned in G4-septimo `tiny_imagenet_dataset.py`.

## §6 DualVer outcome rules

| Outcome | EC bump | FC bump |
|---|---|---|
| Row 1 — H7-A confirmed (Welch rejects, positive sign, g ≥ 0.5) | EC stays PARTIAL ; H7-B → universality breaks at transformer ; **the framework's RECOMBINE prediction is restored at the transformer architectural tier** ; DR-4 evidence v0.7 records the architectural escape from the CNN scope ceiling. | FC stays C-v0.12.0 |
| Row 2 — H7-A null (fail-to-reject) | EC stays PARTIAL ; H7-B → RECOMBINE-empty universality extends from {MLP, CNN} to {MLP, CNN, small-transformer} at ≤ 200 classes ; DR-4 evidence v0.7 logs the scope extension. | FC stays C-v0.12.0 |
| Row 3 — H7-A rejected with negative sign (RECOMBINE *hurts*) | EC stays PARTIAL ; the framework's "richer ops yield richer consolidation" claim is *further weakened* (RECOMBINE actively reduces transformer retention) ; DR-4 evidence v0.7 records a negative-direction architectural finding. | FC stays C-v0.12.0 |
| Row 4 — H7-A confirmed but `g < 0.5` (sub-threshold positive) | EC stays PARTIAL ; reported as exploratory ; H7-B not resolved ; DR-4 evidence v0.7 records the sub-threshold finding. | FC stays C-v0.12.0 |
| Row 5 — exclusion-rate > 50 % (insufficient cells) | abort and amend pre-reg with raised epochs (per §9 envelope b) ; do not commit milestone. | n/a |

EC stays PARTIAL across all rows. FC stays at v0.12.0 across all
rows (no formal-axis bump scheduled by this pilot).

## §7 Reporting commitment

Honest reporting of all observed scalars regardless of outcome.
H7-A confirmation specifically requires Welch *rejecting* H0 with
positive sign and `g ≥ 0.5` — note this **inverts** the honest-
reading clause from the G4-{quater..septimo} ladder, where
fail-to-reject was the predicted positive claim (RECOMBINE-empty).
For G4-octavo, **rejecting** is the predicted positive claim
(RECOMBINE-active at transformer scale), reflecting the directional
flip when the question shifts from "is RECOMBINE empty here too ?"
to "does the architectural change unlock RECOMBINE ?".

If H7-A is null (Row 2), DR-4 evidence v0.7 amends v0.6 with the
RECOMBINE-empty scope extension to the small-transformer tier.
If H7-A is confirmed (Row 1), DR-4 evidence v0.7 records the
architectural escape and triggers a re-analysis of the whole
escalation ladder (the framework's RECOMBINE prediction is partly
salvaged at this tier ; the CNN-only scope ceiling becomes a
substrate-bounded refutation rather than universal).

## §8 Audit trail

Cells registered via `harness/storage/run_registry.py` with
profile keys `g4-octavo/step1/<arm>/<combo>/<strategy>` and R1
bit-stable run_ids. Milestone artefacts under
`docs/milestones/g4-octavo-step1-2026-05-04.{json,md}` plus
aggregate `docs/milestones/g4-octavo-aggregate-2026-05-04.{json,md}`.

## §9 Deviations

Pre-known envelopes :

a. ViT-tiny implementation surprises (e.g., MLX attention numerical
   instability at fp16) — abort and file §9.1 amendment (fall back
   to fp32 attention or smaller dim).
b. `acc_initial < 0.10` for majority of seeds — raise epochs from
   8 to 12 (mirroring G4-sexto §9.1 / G4-septimo carry-over).
c. Per-cell wall > 90 s sustained — extrapolated total > 6 h ;
   escalate to user before committing milestone, propose N = 20
   reduced run.
d. SHA-256 mismatch (loader re-uses G4-septimo SHA, so this should
   not fire ; if it does, the issue is upstream of G4-octavo and
   blocks both pilots).

Any deviation outside the envelopes requires an amendment commit
*before* the affected cell runs, OR a post-hoc honest disclosure
in Paper 2 §7.1.12 acknowledging the deviation and its impact on
confirmatory status.

### §9.1 — TBD on first run if surprises surface

### §9.2 — Pilot ABANDONED (filed 2026-05-04)

**Trigger** : Path D launched on Studio at 09:48 (PID 54775).
Per-cell wall observed at ~4.5 min/cell during the first 25 min
(5 cells), well above the §9 envelope c threshold (90 s sustained
→ extrapolated total > 6 h → escalate). Extrapolated total run
time at this rate : ~18 h, vs the §"Compute" budget of ~4-6 h.
Root cause : `mlx.nn.MultiHeadAttention` on Studio M3 Ultra has
no flash-attn / mixed-precision optimised path at the `(seq=257,
heads=3, dim=64)` working point ; the attention forward+backward
dominates per-iter cost.

**Decision** : pilot ABANDONED before any milestone artefact was
committed. Process killed at 10:30 (5/240 cells written ; the
partial JSON in flight was never committed to git). Studio
compute reallocated to G6-Studio Path A (PID 60194 → 61031 →
63102 → 66815 → final 66815 commit `d963b24`) per user
go-instruction.

**Consequences for the empirical program** :
- The H7-A / H7-B hypotheses about transformer architectural
  break-through are **not resolved by this pilot**. The 5
  collected cells are insufficient (N too small for any test) and
  not committed. The G4-{quater..septimo} CNN/MLP H6-C
  universality scope ceiling stands unaffected.
- Future re-launches under this pre-reg require an §9.3
  amendment with either : (a) optimised MLX attention path, or
  (b) reduced compute (N=20 instead of N=30, or 4-bit quantised
  ViT), or (c) Studio side-by-side launch with a faster substrate.
- The pre-reg remains valid for any future re-attempt ; it is not
  invalidated, just consumed-but-not-resolved by this Path D
  attempt.

This §9.2 amendment closes the abandoned-pilot loop. The
pre-reg's §"Lock commit" line refers to the original lock
(`698af7d`) ; this amendment is filed at a later commit and does
NOT alter the original timestamp.
