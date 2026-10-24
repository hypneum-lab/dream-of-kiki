# E1 35B matched-compute replay controls pre-registration

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI).
**Date:** 2026-10-23 (revision 3: PI decisions of 2026-10-23 on δ
and the fact corpus applied; revision 4, 2026-10-24: pre-filing code
status, no protocol change; revision 5, 2026-10-24: PI decisions —
scoring passes counted in the compute budget, LR-matched replay
registered as an exploratory arm, fact-corpus source = a public fact
set pending the PI's choice of set)
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
**Study gate (PI decision, 2026-10-23):** E1 does not run until a
larger **independent fact corpus** has been built, frozen and hashed
to the specification of §5.1. That corpus supplies the tuning and
confirmatory items. The existing fixtures
`tests/fixtures/mmlu_real.jsonl` (114 rows, SHA-256
`4d881827…a09dd`) and `tests/fixtures/hellaswag_sanity.jsonl`
(200 rows, SHA-256 `2ab581dc…0a979`), with their partition manifest,
are a **pilot / smoke set only** (§5.3); no confirmatory or tuning
cell uses them. There is no `mega_v2`.
**Compute:** 141 confirmatory seeds × 10 profiles (9 registered + 1
exploratory LR-matched arm) = 1,410 cells, plus 120 tuning cells
(10 tuning seeds × 6 configurations × 2 tuned arms) and 60
LR-matching calibration cells (10 tuning seeds × (5 multipliers +
1 `p_equ` reference)), plus 3 smoke seeds × 10 profiles = 30 smoke
cells; **1,620 cells** in total (§5, §9). Without the exploratory
arm: 1,416.
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

Three further 2026 works frame the design:

- Behrouz, Hashemi, Javanmard & Mirrokni, *Language Models Need
  Sleep: Learning to Self-Modify and Consolidate Memories*
  (arXiv:2606.03979; also an ICLR 2027 submission, OpenReview
  `4BIMcGrZjq`). A "Sleep" paradigm with (1) memory consolidation by
  upward distillation (Knowledge Seeding: on-policy distillation
  combined with RL-based imitation learning) and (2) "Dreaming", an
  RL-driven generation of synthetic curricula, evaluated on
  long-horizon, continual-learning, knowledge-incorporation and
  few-shot tasks. **Differentiation:** that work proposes a
  sleep/dream method; E1 does not. E1 is a controlled,
  token-matched comparison at one fixed 35B-A3B checkpoint of the
  existing GENIAL dream arm (`p_equ`) against tuned vanilla replay,
  restricted on-policy replay and a same-checkpoint prompt-only
  control, with the DOWNSCALE contribution isolated by ablation
  (H14-B). No claim here is novel because of the words "sleep" or
  "dream", and E1 does not reproduce or test the Behrouz et al.
  method.
- Marek, Cho, Qiu, Chunara, Izmailov & Wilson, *Forgetting in
  Language Models: Capacity, Optimization, and Self-Generated
  Replay* (arXiv:2605.26097). Self-generated samples serve as
  effective replay data, nearly eliminating forgetting; low learning
  rates reduce forgetting but need substantially more training
  steps, and replay breaks that trade-off. Learning rate and step
  count therefore change the outcome directly, which is why the
  controls are tuned (§5), and it motivates the DOWNSCALE caveat in
  §7.
- Peng, Li, Ouyang, Yuan & Ding, *Diagnosing and Mitigating Thinking
  Collapse in On-Policy Self-Distillation* (arXiv:2607.10805),
  reports a collapse of native intermediate reasoning under
  on-policy self-distillation and a mitigation that anchors to a
  reference derived from the frozen base model. It motivates the
  frozen-reference diagnostic of §6. A second report of collapse,
  in latent on-policy distillation (LastOPD, arXiv:2609.28845), is
  consistent with it; no newer result removes the need for the
  diagnostic.

In this harness `p_min` **is** SGD replay at default hyperparameters
(`HL_LR = 1e-4`, `HL_INNER = 5`): vanilla, untuned replay. The
framework-specific part is what `p_equ` (replay + DOWNSCALE) and
`p_max` (+ RESTRUCTURE) add. RECOMBINE is a weight-neutral no-op in
the CUDA harness.

**What "content" means here.** Replay items are row-disjoint from
the items each seed is scored on (`_cell_split`), and in this
revision also come from the same item-disjoint partition (§5.2). Any
eval gain from replay is therefore transfer from **other** items: it
can come from answer-format adaptation, domain text, or learning to
emit correct answers in general. Because the corpus items are
independent facts (§5.1), replaying one item does not rehearse the
fact scored by another. H14-A compares answer-containing
replay with prompt-only replay. It does **not** separate answer
format from answer content (the prompt-only control sees no
completions at all), and it does not test recall of the scored facts.

**Scope.** All conclusions are about this experiment: Qwen3.6-35B-A3B
on the frozen fact corpus of §5.1. The 2026-06-19 residual
was measured on hand-authored mega_v2 facts with a 1.5B model and a
different endpoint. This study motivates, but cannot adjudicate, the
mechanism of that historical residual.

## §2 Hypotheses (confirmatory)

**Unit and pairing.** Unit = seed. Within a seed every profile uses
the same replay / eval split, the same eval items for pre and post
(except `p_noise`, whose post-evaluation uses the different draw
`_cell_split(seed + 9973)` by design, to measure item-sampling noise;
`p_noise` enters no confirmatory contrast), and starts from the same
restored base weights. All contrasts are
**paired by seed**: for contrast (X, Y) the analysed quantity is
`D_s = Δ_s(X) − Δ_s(Y)`, with `Δ = post − pre` on `acc` (Endpoint, below).

**Endpoint.** `acc` = accuracy on that seed's `n_eval` eval items of the
confirmatory partition of the fact corpus (§5.1; `n_eval` ≥ 130,
fixed by the rule of §5.2). This replaces the previous draft's
MMLU / HellaSwag composite; "accuracy" below always means `acc`.

**SESOI and margins.** δ = **0.01** accuracy points (PI decision of
2026-10-23, half the previous draft's 0.02; one tenth of the 1.5B
residual of +0.10; 1.3 items at `n_eval` = 130). It is a fixed raw
margin used as the superiority floor, the bound for "no effect of
practical size", and the non-inferiority margin. It is not rescaled
by any observed SD. For reference, it equals the old 0.29-SD
convention only if SD_D = 0.0345 (§3).

**Paired procedures (exact).** For a contrast with retained n:
`D̄ = mean(D)`, `s = sd(D, ddof=1)`, `se = s / √n`,
`q = t_{1−α, n−1}`, `L = D̄ − q·se`, `U = D̄ + q·se`.
- Superiority: reject iff `t = D̄ / se > q`
  (`ttest_rel(X, Y, alternative="greater")`), i.e. `L > 0`.
- Non-inferiority (margin δ): reject iff `(D̄ + δ) / se > q`, i.e.
  `L > −δ`.
- Paired TOST (bounds ±δ): equivalence iff `L > −δ` and `U < δ`.
- **Degenerate case.** If every retained D is identical (`sd(D) = 0`),
  the t statistic is undefined; then `L = U = mean(D)` and the state
  follows from these bounds. g_z is then reported as
  undefined.
- Effect sizes reported: raw D̄ with the (1 − 2α) interval [L, U],
  and `g_z = D̄ / s × (1 − 3/(4(n − 1) − 1))`.

`kiki_oniric.eval.statistics.tost_equivalence` is **not** used: it
computes an independent-samples standard error with
`df = n₁ + n₂ − 2`, which is wrong for paired data. The paired
procedures above are implemented as
`kiki_oniric.eval.statistics.paired_tost` and
`paired_noninferiority` (PR #42; tested against hand-computed
examples and SciPy one-sided p-values, identical differences handled
exactly), which closes task 1 of §9.

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
(sections 1, 7 and 8; superiority rules by sufficient-statistic
Monte Carlo, 200 000 draws; one-sided bound rules at μ = 0 exact).

- Test-only superiority MDE at 80 % power, one-sided α = 1/60:
  **d_z = 0.252** (g_z ≈ 0.251).
- Standardised check (margin fixed at 0.29 population SD, true
  difference 0): NI ≈ **0.901**, paired TOST ≈ **0.802** (≈ 80 %).
  This is where N = 141 comes from (backlog E1).

The confirmatory margins are raw (δ = 0.01), so power depends on the
unknown SD of the paired difference, SD_D:

| SD_D | δ/SD_D | SUPERIOR at μ = 0.01 | μ = 0.015 | μ = 0.02 | BOUNDED-NO-GAIN / NI at μ = 0 | TOST at μ = 0 |
|---|---|---|---|---|---|---|
| 0.015 | 0.67 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.02 | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.028 | 0.36 | 0.50 | 0.98 | 1.00 | 0.98 | 0.96 |
| 0.035 | 0.29 | 0.50 | 0.96 | 1.00 | 0.89 | 0.78 |
| 0.05 | 0.20 | 0.50 | 0.88 | 0.99 | 0.59 | 0.18 |
| 0.069 | 0.14 | 0.33 | 0.67 | 0.90 | 0.34 | 0.00 |

The complete SUPERIOR rule has ≈ 0.50 power when the true effect
equals δ (the observed-mean floor) whenever the test itself is well
powered; at SD_D = 0.069 the test is the binding constraint (0.33).

**Planning SD.** The only measured SD of a paired replay-minus-control
difference in this harness family is **SD_D = 0.028** (Run A of the
2026-06-19 correction: `p_min − p_ctrl_sft`, Qwen2.5-1.5B, 20 seeds,
hand-authored facts, 10 eval items). It is the planning value. It
comes from another model, endpoint and eval size, so it is a rough
anchor, not evidence for this design.

**What N = 141 can and cannot establish (stated plainly).** At the
planning SD_D = 0.028, N = 141 gives 0.98 for BOUNDED-NO-GAIN / NI
and 0.96 for the secondary TOST; 72 (BOUNDED / NI) and 94 (TOST)
seeds would suffice. N = 141 reaches 80 % only if SD_D ≤ 0.0397
(BOUNDED / NI) or ≤ 0.0345 (TOST). **If SD_D is larger, the
registered N is insufficient** for an 80 %-powered "no practical
effect" conclusion at δ = 0.01: at SD_D = 0.05 it needs 223 / 294
seeds, and at the previous draft's reference SD_D = 0.069 it needs
423 / 557. The corpus-size rule of §5.2 is set so that item sampling
alone uses at most half of the 0.0397 budget.

No prior run reports SD_D for these contrasts at 35B. The
pre-confirmatory proxy is the 10-seed tuning data: the SD over tuning
seeds of the paired difference between the two selected
configurations (vanilla vs OPR) on the tuning partition. It is a
rough, item-shifted proxy for the H14-C SD_D and none exists for
H14-A/B. It is recorded in the frozen tuning amendment for
information only. **N stays 141 whatever the proxy shows**; if the
proxy exceeds 0.0397, the amendment states the correspondingly
reduced BOUNDED / NI power before any confirmatory seed runs.

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
  UNINTERPRETABLE. Below that, the estimand is the mean paired
  difference over seeds whose two cells both complete, and the error
  rates of §2 are conditional on that set; the gate limits, but does
  not remove, selection by treatment-dependent failures.
- **Incomplete run.** If compute ends before 141 seeds are complete,
  **all** H14 hypotheses become exploratory (§8).

## §5 Corpus, partitions, profiles and matched compute

### §5.1 Fact corpus (study gate)

The confirmatory and tuning items come from a new corpus that must
exist, frozen and hashed, before tuning starts and before filing.
Required properties:

1. **Format.** Four-choice items (`question`, `choices[4]`, `answer`
   index), scored by `_score_choice` exactly like the pilot MMLU
   rows. One gold answer per item; the three distractors are of the
   same type as the answer. Question plus gold answer is at most 256
   tokens under the pinned tokenizer, which keeps the ≤ 512-token
   memory assumption of §9.
2. **Independence.** One fact per item. Items are written or sourced
   individually, never produced by a fill-in template generator.
   Masking named entities, numbers and dates in the question gives a
   *stem skeleton*; no skeleton may be shared by more than 5 items.
   Items that state, paraphrase or presuppose the same fact are one
   *fact group*.
3. **Deduplication.** Exact duplicates (after Unicode NFKC,
   lower-casing and whitespace collapse of question + gold answer)
   are removed. Near-duplicates (character 5-gram Jaccard ≥ 0.6 on
   the normalised question, or an identical gold answer string with
   Jaccard ≥ 0.4) are merged into one fact group. Each fact group is
   then reduced to one item (the one with the lowest item SHA-256),
   so every retained item is its own fact group and the row-level
   `_cell_split` is fact-disjoint within a seed. Any item that is a
   near-duplicate of a row of `mmlu_real.jsonl`,
   `hellaswag_sanity.jsonl` or the 20 hand-authored mega_v2 facts is
   removed.
4. **Provenance.** Each item records its source (citation, DOI or
   URL, or "authored"), author, licence and creation date. The
   build script and its inputs are versioned in this repository.
5. **Headroom.** Before freezing, the base checkpoint is scored on
   the tuning partition only. If its accuracy there exceeds 0.80,
   the corpus is not used (too little room for a gain of δ) and is
   extended or replaced before filing. Confirmatory items are never
   scored before the first confirmatory seed.
6. **Size.** After deduplication and reduction: confirmatory partition ≥ **300**
   items and tuning partition ≥ **120** items. The confirmatory
   minimum is `n_replay + n_eval` = 40 + 255 = 295, rounded up to
   300, where 255 is the `n_eval` the §5.2 rule requires for a
   discordance of 0.20;
   the tuning minimum is 40 replay + 80 eval items.
7. **Partitioning.** Fact groups (never single items) are assigned
   to the confirmatory or tuning partition by SHA-256 of a
   content-blind group key, in the manner of
   `scripts/e1_fixture_partition.py`; no difficulty or model output
   is used. A fact group never straddles the partitions, so the two
   partitions are item- and fact-disjoint.
8. **Hashing.** Per-item SHA-256 of the canonical JSON line, the
   corpus file SHA-256, per-partition hashes (SHA-256 of sorted item
   hashes) and a manifest with a `--check` mode, as for the pilot
   manifest. The corpus and partition hashes are written into the
   filed version; any later change to the corpus is a deviation.

### §5.2 Per-seed item counts

Confirmatory seeds (0-140) draw replay and eval items only from the
confirmatory partition: `n_replay = 40` and `n_eval` items, disjoint
within a seed (`_cell_split`). Tuning seeds (1000-1009) draw only
from the tuning partition: `n_replay = 40`, `n_eval` = the rest.
Seed-level inference is **conditional on the frozen corpus**; it
does not generalise to other item sets.

**Rule for `n_eval`.** Write SD* = 0.039661, the largest SD_D at which
N = 141 still has 80 % power for BOUNDED-NO-GAIN / NI at δ = 0.01
(§3). Item sampling contributes about `p_disc / n_eval` to the
variance of D, where `p_disc` is the fraction of eval items on which
the two post-checkpoints of a contrast differ in correctness. The
item part is allowed half of the variance budget:
`n_eval = max(130, ceil(p̂_disc / (SD*² / 2)))`, computed with the
exact SD* printed by the power script (`SD* / √2 ≈ 0.0280`), where `p̂_disc` is measured on the tuning
data (the mean over the 10 tuning seeds, for the two selected
configurations) and frozen in the tuning amendment. The planning
value `p_disc = 0.10` gives 128, hence the floor of 130; 0.20 gives
255. If the rule needs more than (confirmatory partition size − 40)
items, `n_eval` is capped at that value (the frozen corpus is never
changed after filing) and the amendment states the reduced power
before the first confirmatory seed. This is an
approximation (it ignores the finite-population correction and the
training noise inside `p_disc`); it fixes the design in advance and
uses no confirmatory data.

### §5.3 Pilot / smoke set

The pilot fixtures keep their item-disjoint partition
(`scripts/e1_fixture_partition.py`, manifest
`tests/fixtures/e1_partition_2026-10.json`, `--check` verifies it).
They are used only for the pre-filing smoke (§9) and pipeline tests.
No tuning or confirmatory cell, and no number in §3, uses them.

### §5.4 Profiles and matched compute

**Token budget T (independent of any tuned value).** For seed s,
`T_s = 5 × tok(p_min corpus of seed s)`, where `tok` counts loss
tokens (sequence length − 1, pinned tokenizer) and **5 is a fixed
constant** (the harness default `inner`), not the tuned `inner`.

**Update schedule (all replaying profiles).** Given a profile corpus
in `_cell_split` order and an inner-repeat count m (default 5, or the
tuned value), the update sequence repeats each record m times
consecutively, cycles from the start of the corpus when exhausted,
and ends with exactly T_s loss tokens. Records with < 2 tokens are
skipped. Each update is one sequence (batch size 1), momentumless
SGD, as in `_replay`. **Terminal update:** if the next full record
would take the cumulative count past T_s, with r tokens remaining
(1 ≤ r), the record is truncated to its first r + 1 input tokens, so
it contributes exactly r loss tokens; its loss is the mean over those
r positions, like every other update. Achieved loss tokens therefore
equal the profile's update budget exactly (T_s, or U_OPR for OPR; see below) for every replaying profile (a unit test covers
unequal record lengths and r = 1). Tokens are matched; optimizer-step counts differ across
profiles and are reported. DOWNSCALE (`HL_SHRINK = 0.99`, untuned)
and RESTRUCTURE are applied after replay and are not charged (they
are elementwise / permutation operations with no forward pass).
**Scoring passes are counted (PI decision 2026-10-24).** Compute is
counted in training-token equivalents: a forward-only scoring token
costs 1/3 of a training token (about 2 N vs 6 N FLOPs per token for
N active parameters). Pre/post evaluation passes are identical for
every profile, so they are counted in every profile's total and
cancel in the match. Scoring passes that only some profiles run are
charged against those profiles' update tokens: OPR's 160 scoring
passes (4 choices × 40 prompts) read S_OPR tokens (prompt + choice
per pass), so OPR's update budget is
`U_OPR = T_s − ceil(S_OPR / 3)` (floored at 0), and every other
profile keeps `U = T_s`. The schedule then ends at U instead of T_s
(same terminal-update rule). `scripts/e1_budget.py`
(`scoring_pass_tokens`, `update_token_budget`) implements this, with
tests; S_OPR, the charge and U are written to each cell. **Feasibility
(review round 3).** The match holds only if `ceil(S_OPR / 3) ≤ T_s`.
A cell where OPR's scoring alone exceeds T_s is still run (with 0
update tokens) but is classified `scoring-overbudget`
(`budget_status`): it is excluded from every matched-compute claim and
reported with its counts. A common-budget adjustment (raising every
arm's budget by OPR's scoring) was rejected because it would make
every arm's T_s depend on OPR and couple the arms. Expected margin:
160 passes of roughly 60-70 tokens give about 3 500 token-equivalents
against a T_s of about 12 000 for 40 fact records at m = 5, so the
case should not arise; the rule fixes what happens if it does. "Matched
compute" therefore means matched total compute (updates plus
profile-specific scoring), not only matched update tokens.

**Ties.** `_score_choice` argmax ties resolve to the lowest choice
index (`torch.argmax`); this rule applies to eval and to OPR reward.

**Profiles** (9 registered + 1 exploratory per confirmatory seed):
- Existing, token-matched: `p_min`, `p_equ`, `p_max`, `p_ctrl_sft`,
  `p_ctrl_shuffle`, `p_noise` (no updates; the noise floor).
- **`p_ctrl_template` (prompt-only control)** — the same replay
  items, but each replay text is the **prompt only**
  (`question + "\nAnswer:"`), with no answer tokens.
  It trains on the identical wrapper and question text without any
  answer completion, true or false. It is cycled to T_s. It does not
  hold completion-format exposure constant, so H14-A is read only as
  "answer-containing vs prompt-only replay" (§1). (A random-answer
  control was rejected: three times out of four it teaches a false
  answer, so beating it could reflect avoiding harmful updates.)
  A control for prompt artefacts is not a new idea: Zahn et al.
  (arXiv:2607.16256, v2) ran an in-context analogue, prepending
  cross-domain material in context to a different, ~671B model.
  `p_ctrl_template` is a different control: a same-checkpoint,
  question-without-answer replay arm under the same token budget and
  update schedule. No novelty is claimed for the control concept.
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
  It is also not current-policy replay in the sense of Continual
  Prompt Replay (arXiv:2608.18574), which regenerates responses with
  the model being trained: the OPR corpus is fixed once from the
  pre-replay checkpoint and never regenerated, like every replay
  corpus in this study (all arms replay frozen responses).

**Tuning.** Grid LR ∈ {3e-5, 1e-4, 3e-4} × m ∈ {2, 5} (6
configurations), tuning seeds 1000-1009 on the tuning partition.
Two independent selections: (i) vanilla replay (`p_min`'s corpus,
no DOWNSCALE) and (ii) `p_ctrl_onpolicy_replay`. Each selects the
configuration with the highest mean Δ of `acc` over the 10 tuning
seeds; ties (within 1e-9) go to the lower LR, then the smaller m.
10 × 6 × 2 = 120 tuning cells (plus the 60 LR-matching calibration
cells of the exploratory arm, which select nothing else). **`p_equ` is not tuned separately**:
it uses selection (i), so `p_equ` and `p_ctrl_vanilla_replay_tuned`
share LR, m and the update sequence and differ only by DOWNSCALE
(this is what makes H14-B a DOWNSCALE ablation). OPR's corpus
depends only on the pre-replay (base) checkpoint, so it is the same
for every configuration of a tuning seed. A tuning seed whose OPR
corpus is empty is dropped, for all configurations alike, from the
OPR selection, the SD_D proxy and `p̂_disc`, and is reported; if more
than 2 of the 10 tuning seeds are dropped, OPR keeps the default
configuration (LR 1e-4, m = 5) and the amendment says so. The two
selections, the SD_D proxy, `p̂_disc` and the resulting `n_eval` (§5.2) are
frozen in a dated amendment **before** the first confirmatory seed
runs. `p_min`, `p_max` and the other
controls keep the defaults (LR 1e-4, m = 5); `p_equ` uses selection
(i) in every confirmatory cell, including secondary contrasts.

- **`p_ctrl_lr_matched` (exploratory; PI decision 2026-10-24;
  reported only).** Replay with the learning rate scaled to match the
  norm change that DOWNSCALE produces (Marek et al.,
  arXiv:2605.26097): `p_min`'s corpus with selection (i)'s m and
  update sequence and **no DOWNSCALE**, at LR `c × LR_(i)`. c is
  calibrated once on the tuning seeds 1000-1009 from
  {1, 1.5, 2, 3, 5}: for each c, the median over tuning seeds of the
  full-parameter norm change ‖θ_post − θ_pre‖ is compared with that
  of `p_equ` (selection (i) + DOWNSCALE 0.99) on the same seeds; the
  c with the smallest |log ratio| wins (ties to the smaller c). c is
  frozen in the tuning amendment with the other selections (60
  calibration cells). Contrast `Δacc(p_equ) − Δacc(p_ctrl_lr_matched)`
  is reported with the §2 bounds at nominal α = 0.05; it is never
  confirmatory, outside the family and §7, and never changes a
  decided state.

**Fixed run knobs:** `n_replay = 40`, `n_eval` per §5.2, seeds
0-140, tuning seeds 1000-1009, smoke seeds 2000-2002 (pilot set,
§5.3, never analysed).

## §6 Drift and collapse diagnostics

Per cell: relative weight drift ‖w_post − w_pre‖ / ‖w_pre‖ (over all
floating-point parameters), and a **collapse flag** = post `acc`
< pre `acc` − 0.10, **or** post `acc` below 0.25 (4-way chance). Per profile: collapse rate over seeds.

**Frozen-reference monitoring (secondary, descriptive).** Motivated
by Peng et al. (arXiv:2607.10805) and Khamis et al., *iSDFT:
Information-Proximal Self-Distillation for Continual Learning in
LLMs* (arXiv:2609.24646), which anchor to the frozen base model.
The frozen reference is the pre-replay checkpoint of the same seed
(no extra model). Per cell, on a fixed held-out probe set drawn from
the confirmatory eval partition's prompts (no gold answers used):
(i) the mean KL divergence from the frozen reference's
next-token distribution to the post-replay model's over the 4 answer
choices, and (ii) the fraction of items the pre-replay model
answered correctly that the post-replay model answers wrongly
("previously correct, now lost"). Both are reported per profile with
means and 95 % intervals. They are **not** gates, enter no state
rule and do not alter the 20 % collapse gate below; each cell scores a single
pre/post replay phase, so collapse over repeated consolidation
cycles is not observed; this is a stated limitation.

**MoE drift (secondary, descriptive).** Qwen3.6-35B-A3B is a
mixture-of-experts model, and replay can move the router as well as
the experts (CP-MoE, arXiv:2605.20247; FINCH, arXiv:2605.20005).
Per cell, on the same probe set: (i) relative drift of the router
(gate) weights, ‖g_post − g_pre‖ / ‖g_pre‖, and (ii) the L1 distance
between the pre- and post-replay expert-usage histograms (top-k
routed token counts per expert, normalised, summed over layers);
per profile, the mean replay loss over the update sequence. These are
reported per profile with means and 95 % intervals. Like the
frozen-reference measures, they are not gates and enter no state rule.

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
| H14-A | SUPERIOR | In this experiment, answer-containing replay improves `acc` over prompt-only replay at matched tokens (positive effect, estimate ≥ δ). Answer format and answer content are not separated. |
| H14-A | POSITIVE-BELOW-SESOI | Positive, with an estimate below δ; reported, no claim. |
| H14-A | BOUNDED-NO-GAIN | In this experiment, answer-containing replay adds less than δ over prompt-only replay. No statement about the 2026-06-19 mega_v2 residual. |
| H14-B | SUPERIOR | Adding DOWNSCALE to tuned vanilla replay (same configuration and updates) improves `acc` (positive effect, estimate ≥ δ). |
| H14-B | POSITIVE-BELOW-SESOI | Positive, with an estimate below δ; reported, no claim. |
| H14-B | BOUNDED-NO-GAIN | DOWNSCALE adds less than δ to tuned vanilla replay in this setting. Consistent with a REPLAY-only active core at this tier. |
| H14-C | NON-INFERIOR | `p_equ` is not worse than restricted MC-OPR by δ or more. |
| H14-C | INFERIOR (exploratory) | Exploratory: `p_equ` appears worse than restricted MC-OPR by more than δ. |
| any | INCONCLUSIVE | Inconclusive at n = 141 for this contrast; no claim either way. |
| any | UNINTERPRETABLE | Validity gate failed; reported descriptively only. |

**DOWNSCALE caveat (H14-B SUPERIOR).** DOWNSCALE is a weight shrink
(`HL_SHRINK = 0.99`, untuned) applied after replay. Marek et al.
(arXiv:2605.26097) show that learning rate and step count directly
trade off forgetting against learning, so a shrink of the weights
may act as an **implicit learning-rate reduction or regularisation**.
An H14-B SUPERIOR reading is therefore compatible with that
mechanism and is **not** evidence of a sleep-specific one. Only the
registered contrast (tuned replay with versus without DOWNSCALE,
same configuration and updates) is tested; an LR-matched replay arm
that reproduces the weight-norm change of DOWNSCALE would separate
the two readings and is **not** registered here (see §10).

**Joint headline** (first matching row; the last row catches every
other combination):

| # | Condition | Headline | EC / FC |
|---|---|---|---|
| J1 | A = SUPERIOR ∧ B = SUPERIOR ∧ C = NON-INFERIOR | Candidate LLM-tier framework-C signal on this fact corpus: answer-containing replay beats prompt-only replay, DOWNSCALE adds to tuned replay, and the dream arm is not worse than restricted MC-OPR. Needs replication before any EC promotion. | EC stays PARTIAL ; FC unchanged |
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
- All per-profile scalars (and per-benchmark scalars of the smoke), token and step counts,
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
≈ 1,040 scoring forward passes at `n_eval` = 130 (pre + post,
2 × 130 items × 4 choices; more if §5.2 raises `n_eval`), + 160 for
OPR, + U forward/backward updates (U ≈ T_s / mean record
length; ≈ 100-200 for fact corpora at m = 5, more for the short
`p_ctrl_sft` texts), + one 70 GB restore. Wall per cell and peak
memory are unmeasured at 35B. The smoke on the filing hardware
(seeds 2000-2002, all 10 profiles = 30 cells, pilot set) must
record per-cell wall, `torch.cuda.max_memory_allocated` per device
and host RSS, achieved tokens and update counts. Because the pilot
set has different eval sizes and its own T_s, the projection to the
1,620 cells scales each component by the actual confirmatory
workload (scoring forwards at the §5.2 `n_eval`, update tokens,
restores), not by cell count. The projection goes into the
filed version. If it exceeds the available budget, the PI changes
hardware or amends the design **before filing**; N is never reduced
after filing.

**Filing gate.** This draft is not filed until the fact corpus of
§5.1 is frozen with its manifest and hashes, tasks 1-7 below are
merged at a pinned commit with passing tests, the decision code has
been run on synthetic paired data covering every §2 state and §7
row, the smoke has run, and its measurements, the corpus and
partition hashes, the hardware and the HF revision hash are written
into the filed version.

**Required pre-filing code tasks** (each with unit tests):
1. **Done (PR #42):** paired NI / TOST / bound helper
   (`kiki_oniric.eval.statistics.paired_tost`,
   `paired_noninferiority`) implementing §2 exactly, tested against
   hand-computed examples and SciPy one-sided p-values. The existing
   `tost_equivalence` is left unchanged and unused here.
2. Corpus loader: `pilot_cycle3_cuda.py` reads the fact corpus and
   its §5.1 manifest (and the pilot manifest for the smoke), verifies
   the hashes, and restricts `_cell_split` to the requested
   partition.
3. Token-budget update schedule (§5), with achieved tokens and step
   counts written to each cell; the scoring-pass charge of §5.4 is
   implemented in `scripts/e1_budget.py` (done, tested) and must be
   wired into the schedule.
4. New profiles `p_ctrl_template` (prompt-only), tuned arms,
   restricted MC-OPR (with OPR corpus size and S_OPR per cell) and
   the exploratory `p_ctrl_lr_matched` with its calibration.
5. Drift (full-norm), collapse, frozen-reference and MoE-drift
   (router drift, expert-usage histogram) fields per cell (§6).
6. `analyze_e1_cuda.py`: §2 states, §7 tables, validity gates.
7. Fact-corpus build and check script (§5.1): normalisation, exact
   and near-duplicate detection (including against the pilot
   fixtures and the mega_v2 facts), stem-skeleton count, group-level
   partition, manifest with `--check`; tests on constructed
   duplicates and skeleton collisions.

**Audit trail.** Milestones:
`docs/milestones/e1-35b-replay-controls-<date>.{json,md}`, including
the frozen tuning amendment, the SD_D proxy and the HF revision hash.
Deviations: an amendment commit before the affected seeds run, or
post-hoc disclosure marking the affected hypotheses exploratory (per
`docs/osf-preregistration-draft.md` §5).

## §10 Open decisions for the PI (resolve before filing)

The protocol above is complete with these defaults (earlier
revisions' choices are resolved: scoring passes are charged (§5.4,
PI decision 2026-10-24), OPR
is the restricted MC form, `p_equ` is the primary dream arm).
Decided by the PI on 2026-10-23 and no longer open: δ = 0.01; E1 is
gated on the independent fact corpus of §5.1, and the pilot
fixtures are smoke-only. Changing an item below requires editing the
named section before the lock commit.

1. **CUDA node and model revision (§9).** The registered backend is
   CUDA / HF transformers only; the choice is which multi-GPU node
   (at least 2 × H200-class, 4 × H200 default) and the exact HF
   revision hash of Qwen3.6-35B-A3B. Running on Apple MLX instead
   would be a different backend and requires a protocol revision
   before filing, not a hardware swap.
2. ~~LR-matched replay arm~~ — decided by the PI on 2026-10-24:
   registered as an exploratory arm (§5.4), reported only.
3. **Fact-corpus source (§5.1) — partly decided on 2026-10-24.** A
   public fact set (not mega-v2, not newly authored items); **which
   set is pending the PI's choice** among the proposed candidates.
   The §5.1 properties apply to it unchanged (in particular the
   template / stem-skeleton limit, deduplication, headroom ≤ 0.80 on
   the tuning partition, and SHA-256 pins of the source files at a
   fixed version).
