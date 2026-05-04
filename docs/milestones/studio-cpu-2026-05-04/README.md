# Studio CPU exploratory pilots — 2026-05-04

Three small CPU pilots on Mac Studio M3 Ultra (24 P-cores, 512 GB RAM)
launched in parallel with G4-sexto Studio N=95 (GPU). Total compute :
~6 min wall.

**Status** : exploratory (no individual OSF pre-registration).
Findings are descriptive and not part of Paper 2 confirmatory claims.

## 1. K2 real-substrate validation

- Script : `k2_real_substrate.py`
- Data : 24 random `.npz` modules from SpikingKiki-V4 35B-A3B (31 070 modules)
- LIF metadata (from `lif_metadata.json`) : `T=128, threshold=0.0625, tau=1.0`
- Procedure : project Poisson input through each module's weight tensor,
  simulate LIF dynamics over T steps, compute pairwise mean-vector-length
  (MVL) of phase coupling on rate envelopes
- Pairs : 24 modules → C(24,2) = 276 pairs

| Statistic | Value |
|-----------|-------|
| MVL mean | **0.3855** |
| MVL std | 0.0901 |
| MVL median | 0.3797 |
| MVL 5–95 % | [0.2557, 0.5434] |
| Fraction in K2 band | 45.7 % |

K2 invariant range : `[0.27, 0.39]` (eLife 2025 Bayesian meta-analysis,
BF=58 for SO-spindle phase coupling).

**Verdict** : *partial empirical support*. Mean and median both inside
the K2 anchor band ; 5-95 percentile spans wider, only ~46 % of pairs
strictly within. Consistent with the K2 invariant being satisfied at
the population level on real SpikingKiki-V4 modules, with substantial
inter-pair variability — *not* a strong-form validation.

Caveats :
- Synthetic input (Poisson) rather than naturalistic context
- LIF parameters from metadata file, not retrained
- N=24 modules sampled out of 31 070 ; bootstrap CI not computed
- Phase computed via FFT, not Hilbert transform
- No null model (e.g. shuffled spike trains) baseline

## 2. R1 cross-machine verification

- Test : `tests/reproducibility/` (9 tests : 5 bit-exact + 4 contract)
- M1 Max baseline : Python 3.12, MLX 0.31.1, golden_hashes.json status
  was `pending_review` (committed 2026-05-03)
- Studio target : Python 3.14.4, MLX 0.31.1, fresh `uv sync`

Result : **9/9 PASS** with identical hashes across machines.

Test names :
- `test_r1_replay`, `test_r1_downscale`, `test_r1_restructure`,
  `test_r1_recombine`, `test_r1_full_pipeline`
- `test_r1_registry_same_tuple_same_run_id`,
  `test_r1_registry_different_seed_different_run_id`,
  `test_r1_registry_idempotent_insert`,
  `test_r1_registry_output_hash_contract`

**Action** : `golden_hashes.json` status field promoted from
`pending_review` to `validated_cross_machine_2026-05-04` for all 5
bit-exact entries.

Caveats :
- Both machines are Apple Silicon (M1 Max + M3 Ultra) — does NOT prove
  CUDA/Linux portability of MLX/Metal backend
- Python 3.12 vs 3.14.4 differs but MLX 0.31.1 identical
- Run on `8874c7b` Studio vs main M1 Max (same effective code state)

## 3. Robertson 2018 sequential-ordering test

- Script : `robertson_sequential_ordering.py`
- Reference : Robertson EM (Curr Biol 2018) — sequential structure of
  NREM→REM determines what gets retained
- Question : does the firing **order** of (REPLAY, DOWNSCALE,
  RESTRUCTURE, RECOMBINE) affect retention on a CL stream ?
- Design : 6 permutations × 5 seeds = 30 cells, 5-task synthetic
  stream, MLP 2×32 + dream-op stubs
- Metric : average task retention after full stream

| Permutation | Mean retention | Hedges' g vs canonical |
|-------------|---------------|------------------------|
| REP→DOW→RES→REC (canonical) | 0.6748 ± 0.111 | — |
| DOW→REP→RES→REC | 0.6860 ± 0.119 | +0.079 |
| REP→RES→DOW→REC | 0.6748 ± 0.111 | +0.000 |
| REC→REP→DOW→RES | 0.6860 ± 0.119 | +0.079 |
| RES→REC→REP→DOW | 0.6802 ± 0.125 | +0.037 |
| REP→REC→DOW→RES | 0.6748 ± 0.111 | +0.000 |

Max |g| across 5 alternatives = **0.079** < 0.2.

**Verdict** : H_RO-A — permutation effect SMALL ; ordering does not
materially alter retention on this synthetic stream.

This descriptively supports DR-3 (substrate-independence) extension :
not only is the substrate fungible, but the *intra-cycle ordering*
is also weakly determinative — at least in this regime.

Caveats :
- Synthetic 5-task stream ; not on real CL benchmark
- Dream-op stubs are simplified (3-step replay, weight-magnitude SHY)
- 5 seeds × 6 perms is very small ; CI not computed
- Single architecture (MLP 2×32) ; CIFAR/Tiny-IN regime untested
- Robertson's NREM/REM split is not modeled here ; we only test
  intra-set ordering of the 4 operations
