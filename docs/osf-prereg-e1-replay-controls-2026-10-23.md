# E1 35B matched-compute replay controls pre-registration

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI).
**Date:** 2026-10-23 (revision 2 after independent review)
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
`scripts/run_e1_cuda_35b.sh`, extended as listed in §9 (required
pre-filing code tasks). Analysis: `scripts/analyze_e1_cuda.py`,
extended with the paired procedures of §2.
**Benchmarks:** `tests/fixtures/mmlu_real.jsonl` (114 rows, SHA-256
`4d881827…a09dd`) and `tests/fixtures/hellaswag_sanity.jsonl`
(200 rows, SHA-256 `2ab581dc…0a979`), split into item-disjoint
tuning and confirmatory partitions (§5). There is no `mega_v2`.
**Compute:** 141 confirmatory seeds × 9 profiles = 1,269 cells, plus
10 tuning seeds × 6 configurations × 2 tuned arms = 120 tuning
cells, plus 3 smoke seeds × 9 profiles = 27 smoke cells;
**1,416 cells** in total (§5, §9).
**Power script:** `scripts/prereg_power_2026_10.py` (§3).
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
  own reward-filtered **generated** responses on past prompts cuts
  forgetting by 42-46 % versus **tuned vanilla replay** on TRACE
  (7-8B). Low-score on-policy replay is worse than vanilla.
- Wang et al. (2026), *Denser ≠ Better: Limits of On-Policy
  Self-Distillation for Continual Post-Training* (arXiv:2607.01763).
  On-policy self-distillation can **collapse**, with drift amplified
  by the teacher–student loop. Any self-generated replay arm needs a
  pre-specified drift / collapse diagnostic (§6).

In this harness `p_min` **is** SGD replay at default hyperparameters
(`HL_LR = 1e-4`, `HL_INNER = 5`): vanilla, untuned replay. The
framework-specific part is what `p_equ` (replay + DOWNSCALE) and
`p_max` (+ RESTRUCTURE) add. RECOMBINE is a weight-neutral no-op in
the CUDA harness.

**What "content" means here.** Replay items are row-disjoint from
the items each seed is scored on (`_cell_split`), and in this
revision also come from the same item-disjoint partition (§5). Any
eval gain from replay is therefore transfer from **other** items: it
can come from answer-format adaptation, domain text, or learning to
emit correct answers in general. H14-A compares answer-containing
replay with prompt-only replay. It does **not** separate answer
format from answer content (the prompt-only control sees no
completions at all), and it does not test recall of the scored facts.

**Scope.** All conclusions are about this experiment: Qwen3.6-35B-A3B
on the partitioned MMLU / HellaSwag fixtures. The 2026-06-19 residual
was measured on hand-authored mega_v2 facts with a 1.5B model and a
different endpoint. This study motivates, but cannot adjudicate, the
mechanism of that historical residual.

## §2 Hypotheses (confirmatory)

**Unit and pairing.** Unit = seed. Within a seed every profile uses
the same replay / eval split, the same eval items for pre and post
(except `p_noise`, whose post-evaluation uses the disjoint draw
`_cell_split(seed + 9973)` by design, to measure item-sampling noise;
`p_noise` enters no confirmatory contrast), and starts from the same
restored base weights. All contrasts are
**paired by seed**: for contrast (X, Y) the analysed quantity is
`D_s = Δ_s(X) − Δ_s(Y)`, with `Δ = post − pre` on the composite.

**Composite.** The unweighted mean of the two benchmark accuracies,
`(acc_MMLU + acc_HellaSwag) / 2`, each over that seed's 50 eval items
(not pooled over 100 items; the two are equal here only because both
eval sets have 50 items).

**SESOI and margins.** δ = **0.02** composite-accuracy points, a
fixed raw margin (one item out of 50 on each benchmark; one fifth of
the 1.5B residual of +0.10). It is used as the superiority floor, the
bound for "no effect of practical size", and the non-inferiority
margin. It is not rescaled by any observed SD. For reference, it
equals the old 0.29-SD convention only if SD(D) = 0.069 (§3).

**Paired procedures (exact).** For a contrast with retained n:
`D̄ = mean(D)`, `s = sd(D, ddof=1)`, `se = s / √n`,
`q = t_{1−α, n−1}`, `L = D̄ − q·se`, `U = D̄ + q·se`.
- Superiority: reject iff `t = D̄ / se > q`
  (`ttest_rel(X, Y, alternative="greater")`), i.e. `L > 0`.
- Non-inferiority (margin δ): reject iff `(D̄ + δ) / se > q`, i.e.
  `L > −δ`.
- Paired TOST (bounds ±δ): equivalence iff `L > −δ` and `U < δ`.
- Effect sizes reported: raw D̄ with the (1 − 2α) interval [L, U],
  and `g_z = D̄ / s × (1 − 3/(4(n − 1) − 1))`.

`kiki_oniric.eval.statistics.tost_equivalence` is **not** used: it
computes an independent-samples standard error with
`df = n₁ + n₂ − 2`, which is wrong for paired data. Implementing a
paired helper with tests is a required pre-filing task (§9).

**Family.** Three hypotheses, Bonferroni α = 0.05 / 3 = **1/60
≈ 0.0167** (same α as the E1 harness).

- **H14-A (answer-containing vs prompt-only replay)** —
  `p_min` vs `p_ctrl_template` (prompt-only control, §5).
  Primary claim: superiority.
- **H14-B (DOWNSCALE on top of tuned replay)** — `p_equ` vs
  `p_ctrl_vanilla_replay_tuned`. Both use the **same** tuned replay
  configuration and the identical update sequence (§5); the only
  difference is the DOWNSCALE step after replay. Primary claim:
  superiority.
- **H14-C (non-inferiority to on-policy replay)** — `p_equ` vs
  `p_ctrl_onpolicy_replay`. Primary claim: non-inferiority at δ.

**Per-hypothesis states** (evaluated in order; first match; the list
is exhaustive):

For H14-A and H14-B:

| State | Rule |
|---|---|
| UNINTERPRETABLE | a validity gate of §4 / §6 fails for this contrast |
| SUPERIOR | `L > 0` and `D̄ ≥ δ` |
| POSITIVE-BELOW-SESOI | `L > 0` and `D̄ < δ` |
| BOUNDED-NO-GAIN | `U < δ` (gain below δ; includes negative estimates) |
| INCONCLUSIVE | none of the above |

For H14-C:

| State | Rule |
|---|---|
| UNINTERPRETABLE | a validity gate of §4 / §6 fails for this contrast |
| NON-INFERIOR | `L > −δ` |
| INFERIOR (exploratory) | `U < −δ` (worse than OPR by more than δ) |
| INCONCLUSIVE | neither |

**What SUPERIOR means.** `L > 0` establishes a positive population
difference at level α; `D̄ ≥ δ` filters the **point estimate**.
SUPERIOR reads "a positive effect whose estimate is at least δ",
never "the population effect is at least δ".

**Error accounting.** Each hypothesis spends α once. For H14-A/B the
confirmatory conclusions are SUPERIOR (false only if the true μ ≤ 0)
and BOUNDED-NO-GAIN (false only if μ ≥ δ); as δ > 0, at most one can
be false for any μ. For H14-C only NON-INFERIOR is confirmatory;
INFERIOR is exploratory and spends no α (both can be false at
μ = −δ exactly). The family bound is 3α = 0.05.

`U < 0` in H14-A/B, and the paired TOST in H14-C, are reported as
secondary results; no harm or equivalence conclusion is part of the
confirmatory family. A non-significant superiority test never
establishes absence: only BOUNDED-NO-GAIN does (and only below δ).
Failing to show non-inferiority is INCONCLUSIVE, not inferiority.

Secondary (reported, outside the family, nominal α = 0.05, same
state rules): every dream profile (`p_min`, `p_equ`, `p_max`)
against every control, which keeps the `analyze_e1_cuda.py`
defensibility signature `dream > p_ctrl_sft > p_ctrl_shuffle ≈
p_noise`.

## §3 Power analysis

n = 141 paired seeds; reproduced by `scripts/prereg_power_2026_10.py`
(sufficient-statistic Monte Carlo, 200 000 draws).

- Test-only superiority MDE at 80 % power, one-sided α = 1/60:
  **d_z = 0.252** (g_z ≈ 0.251).
- Standardised check (margin fixed at 0.29 population SD, true
  difference 0): NI ≈ **0.901**, paired TOST ≈ **0.802** (≈ 80 %).

The confirmatory margins are raw (δ = 0.02), so power depends on the
unknown SD of the paired composite difference, SD_D:

| SD_D | δ/SD_D | SUPERIOR at μ = 0.02 | μ = 0.03 | μ = 0.04 | BOUNDED-NO-GAIN at μ = 0 | NI at μ = 0 | TOST at μ = 0 |
|---|---|---|---|---|---|---|---|
| 0.03 | 0.67 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.05 | 0.40 | 0.50 | 0.99 | 1.00 | 1.00 | 0.99 | 0.99 |
| 0.069 | 0.29 | 0.50 | 0.96 | 1.00 | 0.90 | 0.90 | 0.80 |
| 0.10 | 0.20 | 0.50 | 0.88 | 0.99 | 0.59 | 0.59 | 0.18 |

The complete SUPERIOR rule has ≈ 0.50 power when the true effect
equals δ (the observed-mean floor), whatever SD_D is. No prior run
reports SD_D for these contrasts at 35B. The tuning-partition cells
(§5) estimate SD_D for the tuned contrasts; that estimate is
recorded in the frozen tuning amendment. **N stays 141 whatever the
estimate**: if SD_D > 0.069, the amendment states the reduced
NI/TOST power before any confirmatory seed runs.

## §4 Exclusions, replacement and retained-N inference

- **Cell-level.** A cell is excluded if a BLOCKING invariant is
  violated, or pre/post scores are non-finite. A contrast uses the
  seeds where **both** of its cells are retained (pairwise deletion).
- **No replacement.** Confirmatory seeds are exactly 0-140. Excluded
  seeds are not replaced; inference uses the retained n, and every
  exclusion is reported with its reason.
- **OPR feasibility.** If the pre-replay checkpoint answers **zero**
  of a seed's 40 replay prompts correctly, the OPR corpus is empty.
  This is a **feasibility** failure determined before any treatment
  is applied, not a selective removal of hard seeds: that seed is
  dropped from H14-C only, and the count is reported. If more than
  7 seeds (> 5 %) are dropped this way, H14-C is UNINTERPRETABLE.
  The distribution of OPR corpus sizes (1-40 items) is reported.
- **Differential-exclusion gate.** If a confirmatory contrast loses
  more than 7 of 141 seeds to cell-level exclusions, its state is
  UNINTERPRETABLE.
- **Incomplete run.** If compute ends before 141 seeds are complete,
  **all** H14 hypotheses become exploratory (§8).

## §5 Partitions, profiles and matched compute

**Item-disjoint partitions.** `scripts/e1_fixture_partition.py`
assigns every fixture row to one partition, grouping rows by MMLU
`subject` and HellaSwag `activity_label` so that related items never
straddle the split. Manifest:
`tests/fixtures/e1_partition_2026-10.json` (per-row SHA-256 and
partition; `--check` verifies it).

| Benchmark | Partition | Rows | Groups | Partition SHA-256 |
|---|---|---|---|---|
| MMLU | confirmatory | 70 | 34 | `c9c3706054352dcdcaa5287384c3f4d135092a1e773f8612f2d9a740ace449a2` |
| MMLU | tuning | 44 | 19 | `abfdd75c8d5cc7b4bb19c1d4151889172bf0ad0e46b62943d37eae05b9d8323a` |
| HellaSwag | confirmatory | 140 | 43 | `26149930b89b5fc8c930ef512bd4a73e11543b9b344520dbdec667c88f173da4` |
| HellaSwag | tuning | 60 | 19 | `cfad7e5150c1f49c362ae39cc2aad1a35ce721a2e6302cdadfe5411c527e2688` |

Confirmatory seeds (0-140) draw replay and eval items only from the
confirmatory partition: per benchmark `n_replay = 20`,
`n_eval = 50` (all 70 confirmatory MMLU rows are used by every seed;
only the replay/eval split varies). Tuning seeds (1000-1009) draw
only from the tuning partition: `n_replay = 20`, `n_eval` = the rest
(24 MMLU, 40 HellaSwag). Seed-level inference is **conditional on
these fixed item sets**; it does not generalise over items.

**Token budget T (independent of any tuned value).** For seed s,
`T_s = 5 × tok(p_min corpus of seed s)`, where `tok` counts loss
tokens (sequence length − 1, pinned tokenizer) and **5 is a fixed
constant** (the harness default `inner`), not the tuned `inner`.

**Update schedule (all replaying profiles).** Given a profile corpus
in `_cell_split` order and an inner-repeat count m (default 5, or the
tuned value), the update sequence repeats each record m times
consecutively, cycles from the start of the corpus when exhausted,
and stops at the first update after which cumulative loss tokens are
≥ T_s. Records with < 2 tokens are skipped. Each update is one
sequence (batch size 1), momentumless SGD, as in `_replay`. Achieved
tokens must lie in [T_s, 1.05 T_s]; otherwise the last update is
dropped. Tokens are matched; optimizer-step counts differ across
profiles and are reported. DOWNSCALE (`HL_SHRINK = 0.99`, untuned)
and RESTRUCTURE are applied after replay and are not charged.
Pre/post scoring passes are identical across profiles and not
charged. OPR's extra scoring passes (4 choices × 40 prompts = 160
forward passes per cell) are **not** charged against T and are
reported separately. This restricts "matched compute" to matched
gradient-update tokens.

**Ties.** `_score_choice` argmax ties resolve to the lowest choice
index (`torch.argmax`); this rule applies to eval and to OPR reward.

**Profiles** (9 per confirmatory seed):
- Existing, token-matched: `p_min`, `p_equ`, `p_max`, `p_ctrl_sft`,
  `p_ctrl_shuffle`, `p_noise` (no updates; the noise floor).
- **`p_ctrl_template` (prompt-only control)** — the same replay
  items, but each replay text is the **prompt only** (MMLU
  `question + "\nAnswer:"`; HellaSwag `ctx`), with no answer tokens.
  It trains on the identical template and question text without any
  answer completion, true or false. It is cycled to T_s. It does not
  hold completion-format exposure constant, so H14-A is read only as
  "answer-containing vs prompt-only replay" (§1). (A random-answer
  control was rejected: three times out of four it teaches a false
  answer, so beating it could reflect avoiding harmful updates.)
- **`p_ctrl_vanilla_replay_tuned`** — `p_min`'s corpus with LR and m
  set by tuning.
- **`p_ctrl_onpolicy_replay` (OPR)** — for each replay prompt, the
  **pre-replay** checkpoint scores the choices with `_score_choice`;
  reward = 1 iff its argmax equals the gold answer. The corpus is
  `prompt + model's argmax choice` for reward-1 items only, cycled to
  T_s. This is a **restricted multiple-choice adaptation** of Chen et
  al.'s OPR: no free generation, a binary exact-match reward, and a
  scoring pass instead of sampling. It is a selection of the
  checkpoint's own correct choices, not their generated-response
  method. Claims about OPR in this study are limited to this form.

**Tuning.** Grid LR ∈ {3e-5, 1e-4, 3e-4} × m ∈ {2, 5} (6
configurations), tuning seeds 1000-1009 on the tuning partition.
Two independent selections: (i) vanilla replay (`p_min`'s corpus,
no DOWNSCALE) and (ii) `p_ctrl_onpolicy_replay`. Each selects the
configuration with the highest mean composite Δ over the 10 tuning
seeds; ties (within 1e-9) go to the lower LR, then the smaller m.
10 × 6 × 2 = 120 tuning cells. **`p_equ` is not tuned separately**:
it uses selection (i), so `p_equ` and `p_ctrl_vanilla_replay_tuned`
share LR, m and the update sequence and differ only by DOWNSCALE
(this is what makes H14-B a DOWNSCALE ablation). The two selections
and the SD_D estimates are frozen in a dated amendment **before** the
first confirmatory seed runs. `p_min`, `p_max` and the other
controls keep the defaults (LR 1e-4, m = 5); `p_equ` uses selection
(i) in every confirmatory cell, including secondary contrasts.

**Fixed run knobs:** `HL_NSAMPLES = 50`, `HL_NREPLAY = 20`, seeds
0-140, tuning seeds 1000-1009, smoke seeds 2000-2002 (tuning
partition, never analysed).

## §6 Drift and collapse diagnostics

Per cell: relative weight drift ‖w_post − w_pre‖ / ‖w_pre‖ (over all
floating-point parameters), and a **collapse flag** = post composite
< pre composite − 0.10, **or** post accuracy below 0.25 (4-way
chance) on either benchmark. Per profile: collapse rate over seeds.

Collapse is an outcome, not an exclusion. **Validity gate:** if the
collapse rate of **either** profile in a confirmatory contrast
exceeds 20 %, that hypothesis's state is UNINTERPRETABLE (the
contrast is reported, but it is not used to rank methods). This gate
precedes every substantive state, so a collapsing arm can never
produce a SUPERIOR, NON-INFERIOR or BOUNDED reading.

## §7 Outcome rules

Each hypothesis is read through its state (§2), so every outcome
has a reading:

| Hypothesis | State | Reading |
|---|---|---|
| H14-A | SUPERIOR | In this experiment, answer-containing replay improves the composite over prompt-only replay at matched tokens (positive effect, estimate ≥ δ). Answer format and answer content are not separated. |
| H14-A | POSITIVE-BELOW-SESOI | Positive, with an estimate below δ; reported, no claim. |
| H14-A | BOUNDED-NO-GAIN | In this experiment, answer-containing replay adds less than δ over prompt-only replay. No statement about the 2026-06-19 mega_v2 residual. |
| H14-B | SUPERIOR | Adding DOWNSCALE to tuned vanilla replay (same configuration and updates) improves the composite (positive effect, estimate ≥ δ). |
| H14-B | POSITIVE-BELOW-SESOI | Positive, with an estimate below δ; reported, no claim. |
| H14-B | BOUNDED-NO-GAIN | DOWNSCALE adds less than δ to tuned vanilla replay in this setting. Consistent with a REPLAY-only active core at this tier. |
| H14-C | NON-INFERIOR | `p_equ` is not worse than restricted MC-OPR by δ or more. |
| H14-C | INFERIOR (exploratory) | Exploratory: `p_equ` appears worse than restricted MC-OPR by more than δ. |
| any | INCONCLUSIVE | Inconclusive at n = 141 for this contrast; no claim either way. |
| any | UNINTERPRETABLE | Validity gate failed; reported descriptively only. |

**Joint headline** (first matching row; the last row catches every
other combination):

| # | Condition | Headline | EC / FC |
|---|---|---|---|
| J1 | A = SUPERIOR ∧ B = SUPERIOR ∧ C = NON-INFERIOR | Candidate LLM-tier framework-C signal on this benchmark pair: answer-containing replay beats prompt-only replay, DOWNSCALE adds to tuned replay, and the dream arm is not worse than restricted MC-OPR. Needs replication before any EC promotion. | EC stays PARTIAL ; FC unchanged |
| J2 | A = SUPERIOR ∧ B = BOUNDED-NO-GAIN | Answer-containing replay helps; DOWNSCALE adds < δ to tuned replay. No DR-4 support at this tier in this setting. | unchanged |
| J3 | A = BOUNDED-NO-GAIN | In this setting, answer-containing replay adds < δ over prompt-only replay. The mega_v2 candidate is **not** closed by this result; closing it needs a replication on that task. | unchanged |
| J4 | any other combination | No joint headline; per-hypothesis readings only. | unchanged |

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
- All per-profile, per-benchmark scalars, token and step counts,
  drift and collapse diagnostics are reported regardless of outcome.

## §9 Feasibility, required pre-filing tasks, audit trail

**Memory (arithmetic, bf16, full-parameter SGD).** Weights ≈ 35 B ×
2 B ≈ 70 GB on GPU; gradients up to the same ≈ 70 GB (only routed
experts receive gradients with `zero_grad(set_to_none=True)`, so this
is an upper bound); momentumless SGD keeps no optimizer state;
activations for ≤ 512-token sequences at batch 1 are a few GB. Peak
GPU ≈ 145 GB, sharded with `device_map="auto"`: at least 2 × H200
(141 GB each); the turnkey script targets 4 × H200. The CPU restore
snapshot holds another ≈ 70 GB of host RAM, and every cell restores
≈ 70 GB host → device.

**Runtime (to be measured; required before filing).** Per cell:
≈ 800 scoring forward passes (pre + post, 2 × 100 items × 4 choices),
+ 160 for OPR, + U forward/backward updates (U ≈ T_s / mean record
length; ≈ 100-200 for fact corpora at m = 5, more for the short
`p_ctrl_sft` texts), + one 70 GB restore. Wall per cell and peak
memory are unmeasured at 35B. The smoke on the filing hardware
(seeds 2000-2002, all 9 profiles = 27 cells, tuning partition) must
record per-cell wall, `torch.cuda.max_memory_allocated` per device
and host RSS, achieved tokens and update counts. Because tuning-
partition seeds use fewer eval items (24 MMLU + 40 HellaSwag instead
of 50 + 50) and their own T_s, the projection to the 1,416 cells
scales each component by the actual confirmatory workload (scoring
forwards, update tokens, restores), not by cell count. The projection goes into the
filed version. If it exceeds the available budget, the PI changes
hardware or amends the design **before filing**; N is never reduced
after filing.

**Filing gate.** This draft is not filed until tasks 1-6 below are
merged at a pinned commit with passing tests, the decision code has
been run on synthetic paired data covering every §2 state and §7
row, the smoke has run, and its measurements, the hardware and the
HF revision hash are written into the filed version.

**Required pre-filing code tasks** (each with unit tests):
1. Paired NI / TOST / bound helper (e.g.
   `kiki_oniric.eval.statistics.paired_bounds`) implementing §2
   exactly, tested against a hand-computed example and against
   `scipy.stats.ttest_rel`. The existing `tost_equivalence` is left
   unchanged and unused here.
2. Partition loader: `pilot_cycle3_cuda.py` reads the §5 manifest,
   verifies its hashes, and restricts `_cell_split` to the requested
   partition.
3. Token-budget update schedule (§5), with achieved tokens and step
   counts written to each cell.
4. New profiles `p_ctrl_template` (prompt-only), tuned arms and
   restricted MC-OPR, with OPR corpus size per cell.
5. Drift (full-norm) and collapse fields per cell.
6. `analyze_e1_cuda.py`: §2 states, §7 tables, validity gates.

**Audit trail.** Milestones:
`docs/milestones/e1-35b-replay-controls-<date>.{json,md}`, including
the frozen tuning amendment, SD_D estimates and the HF revision hash.
Deviations: an amendment commit before the affected seeds run, or
post-hoc disclosure marking the affected hypotheses exploratory (per
`docs/osf-preregistration-draft.md` §5).

## §10 Open decisions for the PI (resolve before filing)

The protocol above is complete with these defaults (previous
revision's choices are resolved: scoring passes are not charged, OPR
is the restricted MC form, `p_equ` is the primary dream arm, the
current fixtures are kept and partitioned). Changing an item below
requires editing the named section before the lock commit.

1. **SESOI δ (§2).** 0.02 composite-accuracy points (default) or
   another fixed raw value.
2. **Hardware and model revision (§9).** Multi-H200 node vs Mac
   Studio, and the exact HF revision hash of Qwen3.6-35B-A3B.
3. **Larger fact corpus.** Keep the partitioned fixtures (default) or
   ship a larger independent-fact corpus first, as the correction
   note recommends; the latter replaces §5's partition table.
