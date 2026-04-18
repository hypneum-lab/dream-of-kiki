# G8 — P_max Functional Report

**Gate** : G8 (P_max profile fully wired and operational)
**Target week** : cycle 2 (after C2.5-C2.8)
**Status** : **LOCKED — P_max wired with 4 ops + 4 channels**

## Context

Cycle 1 left P_max as a skeleton (S16.1) with target_ops +
target_channels_out metadata only — no handlers registered, no
α-stream ring buffer, no canal-4 ATTENTION_PRIOR. Cycle 2 Phase
2 (C2.5-C2.8) wires the missing pieces :

- C2.5 α-stream raw traces ring buffer
- C2.6 recombine_full MLX VAE variant (KL divergence)
- C2.7 ATTENTION_PRIOR canal-4 + S4 invariant guard
- C2.8 P_max profile wiring (this task)

## Wiring summary

| Component | Location | Status |
|-----------|----------|--------|
| 4 op handlers | `runtime._handlers` (replay+downscale+restructure+recombine) | ✅ wired |
| α channel input | `alpha_stream` (1024-capacity FIFO ring buffer) | ✅ wired |
| canal-1 weight_delta output | via swap protocol (existing) | ✅ inherited |
| canal-2 latent_sample output | via recombine_handler_full_mlx | ✅ wired |
| canal-3 hierarchy_chg output | via restructure_handler | ✅ wired |
| canal-4 attention_prior output | `attention_prior` (budget 1.5, S4-guarded) | ✅ wired |
| target_ops metadata | preserved from cycle-1 S16.1 | ✅ kept |
| target_channels_out metadata | preserved from cycle-1 S16.1 | ✅ kept |

## Test coverage

- 5 wiring tests in `tests/unit/test_p_max_wiring.py` covering
  status flip, 4 ops registered, 4-op execution, α-stream
  presence, attention_prior presence
- 6 channel tests across `test_alpha_stream.py` (C2.5) +
  `test_attention_prior.py` (C2.7)
- 2 conformance tests : S4 invariant via
  `test_s4_attention.py`
- DR-4 chain validation : ops(P_equ) ⊆ target_ops(P_max),
  channels(P_equ) ⊆ target_channels_out(P_max) — verified by
  `test_dr4_profile_inclusion.py` (cycle 1) which now passes
  on real wiring not just metadata

## Decision

**G8 LOCKED** — P_max profile is empirically operational with
all 4 ops + 4 channels + α input + S4-guarded attention output.
Foundation for cycle 2 Phase 3 (C2.9-C2.12 cross-substrate
ablation) which now has 3 fully-wired profiles to compare
(P_min, P_equ, P_max) instead of the cycle-1 skeleton.

## Implications for Paper 1 + Paper 2

### Paper 1 (cycle 1) update

Discussion §8.3 currently lists "P_max skeleton only" as a
limitation. Cycle 2 C2.8 closes this caveat. Paper 1 v2 arXiv
update (cycle 2 C2.12) will incorporate the full P_max wiring
evidence.

### Paper 2 (cycle 2) ablation matrix

P_max real wiring enables the 3-profile ablation matrix
(baseline + P_min + P_equ + P_max) on the cross-substrate
comparison. Hypothesis H2 (P_max equivalence vs P_equ within
±5%) becomes a real comparison rather than the cycle-1 self-
equivalence smoke test.

## Cross-references

- α-stream : `kiki_oniric/dream/channels/alpha_stream.py` (C2.5)
- ATTENTION_PRIOR + S4 : `kiki_oniric/dream/channels/attention_prior.py`
  + `kiki_oniric/dream/guards/attention.py` (C2.7)
- recombine_full VAE : `kiki_oniric/dream/operations/recombine.py`
  `recombine_handler_full_mlx` (C2.6)
- Profile : `kiki_oniric/profiles/p_max.py` (this task)
