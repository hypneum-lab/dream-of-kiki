# G6-Studio Path A — real LoRA SpikingKiki-V4 × MMLU CL stream

**Date** : 2026-05-04
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `882965c2af762648fe46a8554b36cbe191802fa4`
**Cells** : 20
**Wall time** : 3160.1s
**Smoke** : False

## Pre-registered hypotheses (LOCKED)

Pre-registration : `docs/osf-prereg-g6-studio-path-a.md`.
Decision rules at α/3 = 0.0167 (Bonferroni over {H9-A, H9-B, H9-C}).

### Verdict
```
{
  "bonferroni_alpha": 0.016666666666666666,
  "h9a_classification": "INSUFFICIENT",
  "h9a_g": NaN,
  "h9a_positive_sign": false,
  "h9a_strict_large_effect": false,
  "h9a_welch_p": NaN,
  "h9a_welch_reject": false,
  "h9c_classification": "INSUFFICIENT",
  "h9c_jonckheere_p": NaN,
  "h9c_jonckheere_reject": false,
  "h9c_mean_p_equ": NaN,
  "h9c_mean_p_max": NaN,
  "h9c_mean_p_min": NaN
}
```

## Cells (R1 traceability)

| arm | seed | retention | excluded | run_id |
|-----|------|-----------|----------|--------|
| baseline | 0 | 1.0000 | True | `40ed235b38333fa5e82019dd4ad53caf` |
| baseline | 1 | 1.3333 | True | `bd16fa82cd5e081bc0d1ddd000cd45b7` |
| baseline | 2 | 1.3333 | True | `595b26bd116a04550908d4d858a3f77f` |
| baseline | 3 | 1.1667 | True | `7df66f010e70129df44fee33b3fadf0e` |
| baseline | 4 | 1.2500 | False | `7a90dd71beaf3c6a91ee6c9b01a75264` |
| P_min | 0 | 1.0000 | True | `e893981f5403f1f55a22b6b8298b4cea` |
| P_min | 1 | 0.9167 | True | `e4ff666cd4fb987c1f4d326f0bf25245` |
| P_min | 2 | 1.2500 | True | `9443390cbdeb47098db2d3063036a23c` |
| P_min | 3 | 0.9583 | True | `32fc105482d506d819d0513b2123ffbf` |
| P_min | 4 | 1.1667 | True | `be24136eba4831dd6f7d10f3aab2c873` |
| P_equ | 0 | 1.0000 | True | `716dd3b0539dc49f52484706e5270beb` |
| P_equ | 1 | 1.0000 | True | `0e0faf2c90e315177fbaa7254c2c2365` |
| P_equ | 2 | 1.1042 | True | `1bd8956c7bed10650089857ad28b28ba` |
| P_equ | 3 | 0.8750 | True | `3fcf7ff1d8d2815b241e94fec9ed2620` |
| P_equ | 4 | 1.8333 | True | `83e07c3f2f77d85d72924fe739e03a5f` |
| P_max | 0 | 1.0833 | True | `17a4f43759c0bfc4a705e24e25136e4f` |
| P_max | 1 | 1.0000 | True | `2c3f3e91de4da1fb91cb742fafb6b836` |
| P_max | 2 | 1.0167 | True | `461a3c2b83066b583b9807e9bc66c2b5` |
| P_max | 3 | 1.6042 | True | `59ed1c18430654d97313ac5ad8efe2a1` |
| P_max | 4 | 1.7083 | True | `95eb36dcf25a3a8fd4ccbda03a61fcef` |

## Honest reporting

Per pre-reg §6, EC stays PARTIAL across all H9-{A,B,C} rows. FC stays at C-v0.12.0. H9-A confirmation queues an Option-A (N >= 10) follow-up pre-reg before any STABLE bump. H9-B confirms G5-bis MLX-only artefact extends from toy E-SNN to real-LLM tier. H9-C confirms DR-4 inversion universalises at real-LLM scale.