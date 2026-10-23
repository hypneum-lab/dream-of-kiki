# E1 35B matched-compute replay controls pre-registration

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI).
**Date:** 2026-10-23
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**Parent backlog item:** `EXPERIMENT-BACKLOG.md` E1 (powered TOST,
35B run pending).
**Prior evidence (not modified):**
`docs/milestones/pilot-cycle3-real-1p5b-CORRECTION-2026-06-19.md`.
It invalidates the 1.5B GO and the 1.5B↔7B scaling-law GO (mega_v2
fixture leak). It leaves a replay-specific residual
`p_min − p_ctrl_sft ≈ +0.10` (N = 20, 95 % CI [+0.085, +0.109]) as a
non-adjudicable candidate, confounded by template / format transfer.
**c_version at drafting:** C-v0.25.0+PARTIAL
**Substrate:** Qwen3.6-35B-A3B, bf16, HF transformers
(`HL_MODEL=Qwen/Qwen3.6-35B-A3B`, revision hash pinned at filing).
**Harness:** `scripts/pilot_cycle3_cuda.py` (load-once + restore) and
`scripts/run_e1_cuda_35b.sh`, extended with the three new profiles in
§5. Analysis: `scripts/analyze_e1_cuda.py`, extended with the paired
tests in §2.
**Benchmarks:** `tests/fixtures/mmlu_real.jsonl` (114 rows) and
`tests/fixtures/hellaswag_sanity.jsonl` (200 rows). There is no
`mega_v2`: the CUDA harness omits it.
**Compute:** 141 confirmatory seeds × 9 profiles = 1269 cells, plus
10 tuning seeds × the tuning grid (§5). Hardware is an open decision
(§10).
**Lock commit:** *(filled by introducing commit hash at filing)*
**Lock timestamp:** *(filled at filing, pre-run)*

## §1 Background

The 2026-06-19 correction established two design rules for any LLM-
tier framework-C claim:

1. score the effect as replay **minus a matched control**, not
   post − pre;
2. control for template / format transfer, because row-disjoint items
   that share a template still transfer.

Two 2026 results raise the bar further:

- Chen et al. (2026), *On-Policy Replay for Continual Supervised
  Fine-Tuning* (arXiv:2605.29495). Replaying the current checkpoint's
  own reward-filtered outputs on past prompts cuts forgetting by
  42-46 % versus **tuned vanilla replay** on TRACE (7-8B).
  Low-score on-policy replay is worse than vanilla. A surviving
  dream-replay residual must therefore be compared against tuned
  vanilla replay and against on-policy replay (OPR) at matched
  compute, not only against neutral SGD.
- Wang et al. (2026), *Denser ≠ Better: Limits of On-Policy
  Self-Distillation for Continual Post-Training* (arXiv:2607.01763).
  On-policy self-distillation can forget more than RL and can
  **collapse**, with drift amplified by the teacher–student loop.
  Any self-generated replay arm, OPR included, needs a pre-specified
  drift / collapse diagnostic (§6).

In this harness `p_min` **is** SGD replay of held-out facts at the
default hyperparameters (`HL_LR = 1e-4`, `HL_INNER = 5`). It is
vanilla replay, untuned. The framework-specific part is what
`p_equ` (replay + DOWNSCALE) and `p_max` (+ RESTRUCTURE) add. In the
CUDA harness RECOMBINE is a weight-neutral no-op.

## §2 Hypotheses (confirmatory)

Unit = seed. The same seed gives the same replay / eval split for
every profile (`_cell_split`), so all contrasts are **paired by
seed**. Per-cell outcome: Δ = post − pre on the composite (mean of
the two benchmarks). Effect size: Hedges' g_z on the paired
differences. SESOI = **g = 0.29**, the pre-registered floor already
used by the E1 TOST harness. Family: 3 tests, Bonferroni
α = 0.05 / 3 = **0.0167** (same α as the E1 harness).

- **H14-A (content specificity over format)** — `p_min` >
  `p_ctrl_template`. Paired one-sided t-test at α = 0.0167.
  Confirmed iff H0 is rejected **and** g_z ≥ 0.29. This adjudicates
  the 2026-06-19 candidate: replaying the actual fact content must
  beat replaying the same template with uninformative answers.
- **H14-B (dream ops beyond tuned replay)** — `p_equ` >
  `p_ctrl_vanilla_replay_tuned`. Same test and floor. This asks
  whether DOWNSCALE adds anything to an HP-tuned vanilla replay at
  matched compute.
- **H14-C (non-inferiority to on-policy replay)** — `p_equ` ≥
  `p_ctrl_onpolicy_replay` − SESOI. This is the lower one-sided test
  of a TOST with margin g_z = 0.29, at α = 0.0167. Confirmed iff the
  lower test rejects. The full two-sided TOST
  (`kiki_oniric.eval.statistics.tost_equivalence`, ε = 0.29 SD of
  the paired differences) is reported next to it.

Secondary (reported, outside the family): every dream profile
(`p_min`, `p_equ`, `p_max`) against every control. This keeps the
`analyze_e1_cuda.py` defensibility signature:
`dream > p_ctrl_sft > p_ctrl_shuffle ≈ p_noise`.

## §3 Power analysis

n = 141 paired seeds is the E1 harness's powered count. At
α = 0.0167 and SESOI g_z = 0.29:

- one-sided superiority (H14-A, H14-B) detects g_z ≥ 0.25 at 80 %
  power;
- the non-inferiority test (H14-C) has ≈ 0.90 power when the true
  difference is 0;
- the full two-sided TOST has ≈ 0.80 power when the true difference
  is 0.

This meets the multi-seed-first-class rule (≥ 5) by a wide margin.

## §4 Exclusion criteria

- Any BLOCKING invariant violation, or a non-finite pre/post score:
  exclude the cell and log it.
- `p_ctrl_onpolicy_replay` cells where the pre-dream checkpoint
  answers **zero** replay prompts correctly (empty OPR corpus) are
  excluded **from H14-C only**, and the count is reported.
- Seeds are excluded as whole pairs. If a seed loses a cell needed
  for a contrast, that seed is dropped from that contrast only.
- If, after exclusions, a contrast has n < 141, it is reported with
  its achieved n and the hypothesis is downgraded to exploratory
  (§8).

## §5 Profiles and matched compute

**Token budget.** For each seed, T = (tokens in the `p_min` replay
corpus) × `inner`. Every replaying profile truncates or cycles its
own corpus to reach T gradient-update tokens (±5 %). This is a
change from the current harness, where `p_ctrl_sft` is matched on
**record count**, not tokens. It must ship before filing.

Existing profiles, unchanged except for token matching: `p_min`,
`p_equ`, `p_max`, `p_ctrl_sft`, `p_ctrl_shuffle`, `p_noise`.

New profiles:
- **`p_ctrl_template`** — the same replay items with the same
  `prompt + choice` template, but the answer is a choice drawn
  uniformly at random among the 4 (seeded by cell seed). Format is
  identical; fact content is uninformative on average. This is the
  template-matched control requested by the 2026-06-19 correction
  (item c).
- **`p_ctrl_vanilla_replay_tuned`** — `p_min`'s corpus with LR and
  `inner` selected on a tuning grid (below).
- **`p_ctrl_onpolicy_replay`** (OPR, multiple-choice form) — for each
  replay prompt, the **pre-dream** checkpoint scores the choices with
  `_score_choice`. Reward = 1 iff its argmax equals the gold answer.
  The corpus is `prompt + model's own argmax choice` for reward-1
  items only (high-score on-policy replay, per Chen et al.), cycled
  to budget T.

**Tuning (equal budget for every tuned arm).** Grid
LR ∈ {3e-5, 1e-4, 3e-4} × `inner` ∈ {2, 5}, evaluated on tuning seeds
1000-1009, which are disjoint from the confirmatory seeds 0-140. The
selection metric is the mean composite Δ. The **same grid and
selection** are applied to `p_ctrl_vanilla_replay_tuned`, `p_equ`
and `p_ctrl_onpolicy_replay`. The selected configurations are frozen
in a dated amendment **before** the first confirmatory seed runs.
`HL_SHRINK = 0.99` (DOWNSCALE factor) is not tuned.

**Fixed run knobs:** `HL_NSAMPLES = 50`, `HL_NREPLAY = 20` (the
`run_e1_cuda_35b.sh` defaults), seeds 0-140. The per-seed replay and
eval items are row-disjoint (`_cell_split`).

## §6 Drift and collapse diagnostics (reported, not tested)

Wang et al. (arXiv:2607.01763) motivate these diagnostics:
- per cell: relative weight drift ‖w_post − w_pre‖ / ‖w_pre‖;
- per cell: **collapse flag** = post composite < pre composite − 0.10,
  **or** post accuracy below chance (0.25) on either benchmark;
- per profile: collapse rate over seeds.

Collapse is an outcome, not an exclusion. If the OPR collapse rate
exceeds 20 %, H14-C is reported as "OPR baseline unstable at this
scale", and its verdict is not used to rank dream replay.

## §7 Outcome rules

| Outcome | Reading | EC / FC |
|---|---|---|
| Row 1 — H14-A ∧ H14-B ∧ H14-C | First non-artefact LLM-tier framework-C signal: content-specific, beyond tuned replay, competitive with OPR. Candidate evidence that needs replication before any EC promotion. | EC stays PARTIAL ; FC unchanged |
| Row 2 — H14-A ∧ ¬H14-B ∧ H14-C | Replay content matters, but DOWNSCALE adds nothing over tuned vanilla replay. Consistent with a REPLAY-only active core. No DR-4 support at the LLM tier. | unchanged |
| Row 3 — H14-A ∧ ¬H14-C | Dream replay is inferior to on-policy replay by more than the SESOI. It is not a competitive replay method at this scale. | unchanged |
| Row 4 — ¬H14-A | The 2026-06-19 residual is attributed to template / format transfer. **Clean null**: close the candidate with a dated milestone. | unchanged |
| Row 5 — OPR collapse rate > 20 % | H14-C is uninterpretable (§6). H14-A and H14-B are read as above. | unchanged |

## §8 Reporting commitment and stopping rule

- **Fixed N = 141 confirmatory seeds, no optional stopping, no
  interim tests.** The run is resumable (JSONL cell cache). Partial
  caches are not analysed for the confirmatory family.
- If compute ends before 141 complete seeds, the achieved N is
  reported and **all** H14 hypotheses become exploratory. The run is
  not re-labelled confirmatory after the fact.
- The CUDA backend is a different backend from the MLX runs. Numbers
  are never pooled across backends (R1 note in
  `pilot_cycle3_cuda.py`).
- All per-profile, per-benchmark scalars are reported regardless of
  outcome.

## §9 Audit trail and deviations

- Milestones: `docs/milestones/e1-35b-replay-controls-<date>.{json,md}`,
  including the frozen tuning amendment and the HF revision hash.
- Deviations: an amendment commit before the affected seeds run, or
  post-hoc disclosure marking the affected hypotheses exploratory
  (per `docs/osf-preregistration-draft.md` §5).

## §10 Open decisions for the PI (resolve before filing)

1. **Compute accounting for OPR:** charge only gradient-update tokens
   (default, above), or also charge the OPR scoring forward passes
   against the budget T.
2. **OPR form:** multiple-choice argmax (default; it keeps the harness
   and fixtures) or free-generation OPR closer to Chen et al. Free
   generation needs a generation step and a correctness judge.
3. **Primary dream arm:** `p_equ` (default; canonical P_equ as in the
   parent H1) or `p_max`.
4. **Hardware and model revision:** multi-H200 node vs Mac Studio,
   and the exact HF revision hash of Qwen3.6-35B-A3B.
5. **Fixture size:** keep the 114-row MMLU fixture (row-disjoint at
   `n_replay = 20`, `n_samples = 50`) or ship a larger
   independent-fact corpus first, as the correction note recommends.
