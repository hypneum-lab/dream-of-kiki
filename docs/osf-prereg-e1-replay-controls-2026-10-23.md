# E1 35B matched-compute replay controls pre-registration

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI).
**Date:** 2026-10-23 (revision 3: PI decisions of 2026-10-23 on δ
and the fact corpus applied; revision 4, 2026-10-24: pre-filing code
status, no protocol change; revision 5, 2026-10-24: PI decisions —
scoring passes counted in the compute budget, LR-matched replay
registered as an exploratory arm, fact-corpus sources = RealTime QA +
Daily Oracle + MMLU-CF (§5.1a); node and HF revision deferred until
the corpus exists; revision 6, 2026-10-24: PI confirmed the pooled
bank and the MMLU-CF cap; exact construction, hashes, quotas and
rebalancing in §5.1b after an independent review; revision 7,
2026-10-25: PI decisions — tokenizer and model revision pinned (§9,
§10 item 1), rule 3 widened to a frozen exact list of
"none of the above"-type options (§5.1a); revision 8, 2026-10-25:
PI decisions — empty-OPR tuning rule and H14-C infeasibility,
undefined LR matching, MMLU-CF source date (§5.1, §5.4); revision 9,
2026-10-25: PI decision — backend changed to Apple MLX on the Mac
Studio M3 Ultra (§9 "Backend", deviations D1-D14))
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
**Substrate:** Qwen3.6-35B-A3B, bf16 (`Qwen/Qwen3.6-35B-A3B`,
revision `995ad96eacd98c81ed38be0c5b274b04031597b0`, pinned by the PI
on 2026-10-25; §10 item 1). **Backend (PI decision 2026-10-25,
pre-filing): Apple MLX** (mlx 0.31.1, mlx-lm 0.31.2, the versions
hash-pinned in this repository's `uv.lock`) on the Mac Studio M3
Ultra (512 GB unified memory), instead of HF transformers on CUDA;
§9 "Backend" lists every difference this forces (D1-D14).
**Harness:** `scripts/e1_harness.py` (modes, cell, run identity),
backends `scripts/e1_mlx.py` (registered) and
`scripts/pilot_cycle3_cuda.py` (PyTorch, kept for the parity check
and the CUDA path); analysis `scripts/analyze_e1_cuda.py` (§2).
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
   Jaccard ≥ 0.4) are merged into one fact group (connected
   components, §5.1b). Each fact group is then reduced to one item
   (the one with the lowest source-item hash `src_hash`, §5.1b),
   so every retained item is its own fact group and the row-level
   `_cell_split` is fact-disjoint within a seed. Any item that is a
   near-duplicate of a row of `mmlu_real.jsonl`,
   `hellaswag_sanity.jsonl` or the 20 hand-authored mega_v2 facts is
   removed.
4. **Provenance.** Each item records its source (citation, DOI or
   URL, or "authored"), author, licence and creation date. The
   build script and its inputs are versioned in this repository.
   **PI decision 2026-10-25:** citation, author and licence are
   recorded per source; each manifest item carries its source,
   `source_id` and a `provenance_date`: the item's own date for the
   dated sources (`provenance_date_kind` = `item`) and, for the
   undated `mmlu_cf` items, the release date of the pinned MMLU-CF
   revision `c25b89a` (2025-01-08), labelled `source_release`, i.e. a
   source date, not an item creation date.
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
8. **Hashing.** Per-item SHA-256 of the canonical JSON line (the
   final hash `item_sha256` of §5.1b; the pre-permutation `src_hash`
   is stored in the item and drives every random choice), the
   corpus file SHA-256, per-partition hashes (SHA-256 of sorted item
   hashes) and a manifest with a `--check` mode, as for the pilot
   manifest. The corpus and partition hashes are written into the
   filed version; any later change to the corpus is a deviation.

### §5.1a Sources (PI decision 2026-10-24: all three public sets)

The PI chose to use all three proposed public fact sets and
confirmed on 2026-10-24 (17:25) the design: one pooled item bank,
with `source` recorded on every item and pre-registered as a
stratum; partitions are stratified by source; per-source results are
reported descriptively; the primary analysis (§2) is pooled. The §5.1
properties apply to every item, whatever its source.

| `source` | Set and pin | Licence | Role |
|---|---|---|---|
| `realtimeqa` | RealTime QA, `github.com/realtimeqa/realtimeqa_public` at commit `e288dcfe1d7818bf48299249ae892939637775db` (2026-10-23), files `past/2026/*_qa.jsonl`; per-file SHA-256 computed and pinned at build | none declared (questions from CNN / The Week news quizzes) | human-written, native 4-choice, dated news items |
| `daily_oracle` | Daily Oracle, HF `agentic-learning-ai-lab/daily-oracle` at revision `455b35b2a2415034b05ef91e56c80191fef4a737`, `mc_questions_2020-01-01_2026-07-18.csv` (119 302 323 B, LFS SHA-256 `c8c0b5adaa266fa5e9c1ec59b9cd7d9182c89607d39eb0ef2b269cbb12e58178`) | CC BY 4.0 | LLM-generated, 4-choice, dated, one article per item |
| `mmlu_cf` | MMLU-CF validation, HF `microsoft/MMLU-CF` at revision `c25b89a968a2062e422dd96ddf0fe3387507f0dd`, `val.csv` (2 247 391 B, SHA-256 `28816045197decd701b1a0e51923b43684d708d8c06ec62d20541f90939ac49f`; `val.parquet` LFS `a44a6bb1…77173d`) | CDLA-Permissive-2.0 | decontaminated MMLU-style items, undated |

Rules fixed for all sources (applied by the task-7 build script before
the §5.1 deduplication, and recorded per item):
1. **Date cutoff.** Dated items (`realtimeqa` `question_date`,
   `daily_oracle` `date`) are kept only if dated **2026-04-16 or
   later** (Qwen3.6-35B-A3B was released 2026-04-15), so they post-date
   the model. `mmlu_cf` items carry no date: they are kept with a
   contamination caveat (the validation set is public since
   December 2024) and rely on the headroom rule (5).
2. **Whitespace normalisation.** Question and each option are
   stripped of leading / trailing whitespace before hashing and
   prompting (in Daily Oracle the gold option never carries trailing
   spaces while distractors do, a format leak). After normalisation
   the build reports, per source, how often the gold option is the
   longest; this is descriptive.
3. **Same-type distractors.** `mmlu_cf` items with a
   "none of the above"-type option are removed (§5.1, property 1).
   **Widened by the PI on 2026-10-25 to a frozen exact list**
   (`NOTA_OPTIONS` in `scripts/e1_sources.py`). An item is removed
   iff one of its options, after rule 2 (leading / trailing
   whitespace stripped) and with no other normalisation, equals one
   of these 27 strings exactly (case-sensitive, no fuzzy match):
   `None of the other choices`; `None of the mentioned`,
   `None of the Mentioned`, `none of the mentioned`,
   `None of the above mentioned`, `none of the above mentioned`;
   `None of the above`, `None of the above.`, `None of the Above.`,
   `none of the above`, `none of the above.`, `None of above`;
   `None of these`, `None of These`, `None of these.`,
   `none of these`, `none of these.`; `None of the choices`,
   `none of the choices.`, `none of the options`;
   `None of the above is correct.`,
   `None of the above answers is correct.`,
   `None of the above are true`,
   `None of these alternatives is correct.`,
   `None of these answer choices are correct.`,
   `None of these answers is correct.`; `None is applicable`.
   The list was enumerated from every distinct option of the pinned
   `val.csv` that starts with "none" (case-insensitive) and keeps
   exactly the options that only refer to the other options. Not on
   the list (kept): bare `None` / `none` / `NONE` / `None.` (can be a
   substantive answer, e.g. a count) and options that add content
   (`None of the statements are true`, `None of these (Socrates)`,
   `None of the above is a difference.`,
   `None of the above because the temporal method is not allowed
   under FASB ASC 830.`, `None of the above accurately describe
   financial reports pre-1900.`, `None of the production and service
   departments`, `none of the major theories`). "All of the above"
   and "Both ..." options are not covered. The rule applies to
   `mmlu_cf` only; matches in the dated sources are counted in the
   build report, not removed. The build reports, per variant, the
   number of items carrying it.
4. **Date injection.** Dated items are prompted as
   `"Date: YYYY-MM-DD\n" + question` with their own date; undated
   items get no prefix. (Many news items say "this week".)
5. **Headroom per source.** The §5.1 headroom check (base accuracy
   ≤ 0.80 on the tuning partition) is applied to the pooled tuning
   partition **and to each source**; a source above 0.80 is dropped
   before freezing and the drop is recorded.
6. **Option order.** Choices are permuted per item with
   `default_rng(int(src_hash[:16], 16)).permutation(4)`, where
   `src_hash` is the **source-item hash** of §5.1b (computed before
   any permutation), and the gold index is updated; the permutation
   is part of the frozen item, whose **final hash** is computed after
   it (no circularity).
7. **Balance (confirmed by the PI on 2026-10-24).** The dated
   sources give about 545 (`realtimeqa`) and 492 (`daily_oracle`)
   items after the cutoff, while `mmlu_cf` has about 5 400 after
   rule 3. `mmlu_cf` is capped at the size of the largest dated
   source, by the exact algorithm of §5.1b. The confirmatory (≥ 300)
   and tuning (≥ 120) minima of §5.1 apply to the pool.
8. **Stratified partition.** Fixed per-source quotas (§5.1b), so each
   source contributes to the confirmatory and tuning partitions in
   the same proportion.
9. **Licences.** `realtimeqa` text is not redistributed: the repository
   and any publication carry only item IDs (source file, line, date),
   item SHA-256 and the build script; the item text is rebuilt from the
   pinned commit. `daily_oracle` (CC BY 4.0) and `mmlu_cf`
   (CDLA-Permissive-2.0) are attributed in the milestone.
10. **Reporting.** Every §2 statistic is also reported per source
    (descriptive, no state); the confirmatory states are pooled.

Only the pins and metadata above were fetched (the 2.2 MB MMLU-CF
`val.csv` to compute its SHA-256, then deleted); the corpus is built
by task 7 once the study gate is passed.

### §5.1b Construction (exact; independent review round 4)

Two hashes per item, so randomisation never depends on its own output:
- **Source-item hash** `src_hash` = SHA-256 of the canonical JSON
  (UTF-8, `sort_keys=True`, `separators=(",", ":")`,
  `ensure_ascii=False`) of `{source, source_id, date, question,
  choices, answer}` **after normalisation and before permutation**
  (`choices` in source order, `answer` the source index; `date`
  `YYYY-MM-DD` or `null`; `source_id` is a string: for `realtimeqa`
  the repository-relative path from the repository root with forward
  slashes, a colon and the 0-based line index, e.g.
  `past/2026/20260501_qa.jsonl:3`; for `daily_oracle` the 0-based
  data-row index of the pinned CSV as a decimal string, e.g. `1527`;
  for `mmlu_cf` `val.csv:` and the 0-based data-row index, e.g.
  `val.csv:42`). It drives every random or ordered choice below,
  including the fact-group representative of §5.1 property 3.
- **Final hash** `item_sha256` = SHA-256 of the canonical JSON of the
  frozen item `{source, source_id, date, prompt, choices, answer,
  src_hash}` **after** permutation (`prompt` includes the date
  prefix). It is the §5.1 per-item hash used in the manifest and the
  partition hashes.

Construction order (each step logs its in / out counts per source):

```
1  load      verify every pin (§5.1a table); map rows to source items
2  normalise strip question and options (rule 2); NFKC + lower-case +
             whitespace collapse only inside dedup / skeleton keys
3  exclude   drop: != 4 choices, empty field, invalid gold, a
             NOTA_OPTIONS option in mmlu_cf (rule 3, exact list),
             question + gold > 256 tokens (pinned tokenizer: question
             and gold encoded separately, no special tokens, summed)
4  date      dated sources: keep date >= 2026-04-16 (rule 1); a kept
             item that is a near-duplicate (§5.1 property 3) of any
             dropped pre-cutoff item of the same source is dropped
             too (recurring quiz questions)
5  dedupe    §5.1 property 3 over the pooled survivors and against
             the pilot fixtures / mega_v2 facts; each fact group keeps
             its lowest src_hash; skeletons with > 5 items keep the 5
             lowest src_hash
6  cap       n_rt, n_do, n_cf = post-dedup counts; k = max(n_rt, n_do);
             mmlu_cf: sort by src_hash ascending, keep the first
             min(n_cf, k)
7  split     per source s, items sorted by src_hash ascending;
             t_s = ceil(0.30 * n_s) lowest go to TUNING, the other
             n_s - t_s to CONFIRMATORY (each item is its own fact
             group after step 5, so the split is fact-disjoint)
8  permute   rule 6 with src_hash; add the date prefix (rule 4);
             compute item_sha256; write corpus, manifest, hashes
9  headroom  score the base checkpoint on TUNING only (§5.1
             property 5), pooled and per source
```

Dedupe (5) comes after the date filter (4) so that an item removed for
its date can never knock out a kept duplicate; step 4's second clause
covers contamination through such duplicates. Its reference set is
**every pre-cutoff row of the same source after step 2, including
rows that step 3 would exclude** (a malformed earlier copy still
leaks the fact).

**Exact definitions (reference implementation
`scripts/e1_corpus_core.py`, unit-tested on synthetic items).**
- *Keys.* `norm_key(t)` = NFKC, lower-case, whitespace collapsed to
  single spaces, trimmed. `grams(t)` = the set of character 5-grams
  of `norm_key(t)`; a non-empty key shorter than 5 characters is one
  gram; an empty key gives the empty set. `J(a, b)` = |a ∩ b| /
  |a ∪ b|, and 0 when both are empty (never a duplicate).
- *Near-duplicate relation.* Items x, y are near-duplicates iff
  `J(grams(q_x), grams(q_y)) ≥ 0.6`, or `J ≥ 0.4` and their
  normalised gold options are equal; the gold is consulted only once
  `J ≥ 0.4`. *Invalid gold (final review):* a row whose `choices` is
  not a list or whose `answer` is not an integer index into it (an
  empty list, an out-of-range or negative index, a non-integer) has
  no gold. This can only happen among the pre-cutoff reference rows,
  which keep rows that step 3 would exclude; such a row matches by
  the question rule (`J ≥ 0.6`) only, never by the equal-gold rule.
  Keys and gram sets are computed once per row.
- *Grouping.* Union-find over **all pairs** of the step-5 survivors
  plus the reference rows (pilot fixtures, mega_v2 facts): every
  near-duplicate pair is joined. A connected component that contains
  a reference row is removed entirely; every other component keeps
  its item with the lowest `src_hash`. Step 4's contamination check
  uses the same relation against the pre-cutoff reference set.
- *Skeleton* (dependency-free; after the grouping). On the NFKC
  question, in this order: (i) ISO dates `\b\d{4}-\d{2}-\d{2}\b`
  and month-name dates (`Jan`/`January` … `Dec`/`December`,
  optional `.`, a 1-2 digit day with optional `st|nd|rd|th`,
  optional `, YYYY`) → `<d>`; (ii) numbers `\d(?:[\d,.]*\d)?` →
  `<n>`; (iii) the first character is lower-cased (sentence-initial
  word); (iv) runs of capitalised tokens
  `[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*` → `<e>`; (v) lower-case and
  whitespace collapse. Example: "What did Emmanuel Macron announce
  on May 3, 2026 about 1,200 New Jobs?" → `what did <e> announce on
  <d> about <n> <e>?`. Items sharing a skeleton beyond 5 keep the 5
  with the lowest `src_hash`.

**Headroom drop and rebalancing (`headroom_rebalance`).** Every
tuning item must have been scored. Outcomes are ACCEPT or **REJECT**;
REJECT means the corpus is not used and is extended or replaced
before filing (§5.1). Items are only ever removed; quotas are never
recomputed and no tuning item moves to the confirmatory partition.
```
a  acc_s = tuning accuracy of each source; D = {s : acc_s > 0.80}
b  both dated sources in D                      -> REJECT
c  remove every source in D from both partitions
d  if mmlu_cf remains and a dated source is in D:
     k' = total size of the remaining dated source
     if |mmlu_cf tuning| > k'                   -> REJECT (cap cannot
        be met without moving scored items)
     else drop mmlu_cf CONFIRMATORY items with the highest src_hash
        until |mmlu_cf| = min(|mmlu_cf|, k')
e  pooled accuracy of the remaining tuning partition > 0.80
                                                -> REJECT
f  confirmatory < 300 or tuning < 120           -> REJECT
g  ACCEPT; record the partitions and the weights w_s
```
If only `mmlu_cf` fails, step d does nothing (k is unchanged).

**Pooled estimand and source weights.** `_cell_split` draws each
seed's replay and eval items uniformly from the pooled confirmatory
partition, without stratification. The pooled estimand therefore
weights each source s by its share of the confirmatory partition,
`w_s = c_s / Σ c`. With the cap the three shares are only roughly
equal (the sources differ in size after deduplication, and a
headroom drop changes them); the **recorded** `w_s` govern the
estimand. They are written to the manifest and the filed version;
the realised per-seed shares are reported.

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
configuration (LR 1e-4, m = 5) and the amendment says so.
**PI decision 2026-10-25:** `p̂_disc` and `n_eval` are computed on the
tuning seeds whose OPR corpus is non-empty. If all 10 are empty,
H14-C is declared **INFEASIBLE** in the amendment, before any
confirmatory seed runs: the OPR arm is not run, H14-C and the
secondary contrasts against OPR are reported as INFEASIBLE, and
H14-A and H14-B proceed unchanged (the joint headline then cannot be
J1). In that case `p̂_disc` is undefined and `n_eval` takes the §5.2
floor of 130 (confirmed by the PI on 2026-10-28). The two
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
  decided state. **PI decision 2026-10-25:** if the calibration is
  undefined (a median norm change ≤ 0, so the log ratio does not
  exist), c is recorded as null, the LR-matched arm is not run and
  the contrast is reported as "not computable"; nothing else changes.

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
same configuration and updates) is tested. The LR-matched replay arm
that reproduces the weight-norm change of DOWNSCALE
(`p_ctrl_lr_matched`, §5.4) is registered as **exploratory** only
(PI decision 2026-10-24): it is reported next to H14-B and cannot
change this reading.

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
- Each backend is a different measurement instrument: numbers are
  never pooled across backends (MLX, CUDA / transformers), and the
  backend name is part of every cell's run identity; tuning,
  amendment and confirmatory cells must all come from the registered
  backend (MLX).
- All per-profile scalars (and per-benchmark scalars of the smoke), token and step counts,
  drift and collapse diagnostics are reported regardless of outcome.

## §9 Feasibility, required pre-filing tasks, audit trail

**Backend (PI decision 2026-10-25): MLX on the Mac Studio.** The
registered protocol (N, seeds, corpus, budgets, profiles, tuning,
states and gates) is unchanged; only the instrument changes. Weights:
the HF checkpoint of the pinned revision already on the Studio, every
one of its 40 files checked against the HF tree of commit `995ad96`
(26 safetensors by LFS SHA-256, the other files by git blob id; list
in `scripts/e1_model_pins.json`, re-checked by the harness at start),
loaded directly by `mlx_lm` (no conversion, no download). Item draws:
the `_cell_split` and word-shuffle permutations are those of the torch
generators, written to a split file by `scripts/e1_splits.py` and
replayed by the MLX backend with item-hash checks (verified exact in
tests). Differences the port forces, listed for the PI (none is
changed silently):

- **D1 Instrument.** Metal kernels instead of CUDA / cuBLAS; the
  linear-attention (Gated DeltaNet) layers use `mlx_lm`'s
  differentiable op-by-op recurrence during replay and its Metal kernel
  for scoring, transformers uses the `fla` chunked kernels on CUDA.
  Numbers differ at the rounding level and are never pooled (§8);
  D11 and D12 were two places where `mlx_lm` also used a different
  formula; both are aligned since 2026-10-28.
- **D2 Parameters in scope (accepted and declared by the PI,
  2026-10-28).** `mlx_lm` loads the language model only:
  the vision tower and the multi-token-prediction head of the
  checkpoint are not loaded. "Full-parameter" SGD, DOWNSCALE and the
  drift / norm-change measures therefore cover the language model
  (embeddings, every decoder layer with all 256 routed experts, router
  and shared expert, final norm, LM head). Neither excluded part takes
  part in a text forward pass, so SGD never changes them on either
  backend, but on a transformers multimodal load DOWNSCALE would also
  shrink them and they would enter the norms.
- **D3 RMSNorm parameterisation.** `mlx_lm` stores the zero-centred
  RMSNorm weights as `1 + w` in bf16, which rounds the base model
  (e.g. 1.05 → 1.046875) and makes small SGD steps on those weights
  vanish and DOWNSCALE act on `1 + w`. The harness restores the HF form
  (parameter w, `norm(x)·(1 + w)` in float32, cast back), so SGD,
  DOWNSCALE and drift act on the same parameter as in transformers.
- **D4 Routed-expert gradients.** MLX forms a dense gradient for the
  stacked expert tensors (zero rows for experts no token was routed
  to); transformers also stores the experts as one stacked parameter.
  With momentumless SGD and no weight decay, unrouted experts are
  unchanged on both backends. No difference in the update.
- **D5 Precision.** bf16 parameters and gradients, loss and scoring
  log-softmax in float32, on both backends. The SGD update and
  DOWNSCALE are computed in float32 and rounded once to bf16 (as
  torch's float32 "opmath"): the harness does not use
  `mlx.optimizers.SGD`, which casts the learning rate to bf16 and keeps
  a zero momentum buffer the size of the model, nor a bf16 × Python
  float product, which would turn 0.99 into 0.98828125 (both tested
  bit-exactly). torch's CPU bf16 `add_` rounds differently on ~1 % of
  elements; it is not the reference. Discrete MoE routing can still
  flip a token whose top-k margin is below rounding noise to another
  expert on one backend only.
- **D6 Determinism (accepted and declared by the PI, 2026-10-28).** MLX replay is not bit-reproducible run to run
  (parity: after one fp32 cell, 68 of 76 tensors differ, by at most
  3e-8). E1 cells are therefore not bit-stable; the resumable cache
  and the run identity are unaffected, and no E1 claim relies on
  bit-stability.
- **D7 RESTRUCTURE.** Both backends swap two adjacent entries of the
  decoder-layer list. MLX picks the attention mask from each layer's
  own type, transformers from the position (`config.layer_types[i]`);
  for unpadded batch-1 sequences both give the same outputs (parity,
  asserted: identical per-item correctness, accuracies and
  RESTRUCTURE index for the swapped cells; KL to the frozen reference
  within 1e-3).
- **D8 Memory recording.** `mx.get_peak_memory` (unified memory)
  replaces `torch.cuda.max_memory_allocated`; host RSS as before.
- **D9 Restore.** MLX arrays are immutable: the snapshot keeps the
  base arrays and restoring re-binds them (exact, no 70 GB copy).
- **D11 Linear-attention q / k normalisation (aligned, PI decision
  2026-10-28).** `mlx_lm` normalises q and k with an RMS norm whose ε
  is added to the mean square (an ε larger by the head dimension, 128×,
  in sum space); the harness replaces the Gated DeltaNet forward so
  that q and k are L2-normalised as in transformers / FLA
  (`x · rsqrt(Σx² + 1e-6)`, in float32) and q is scaled by
  `d_k^-0.5`, then cast back to the activation dtype.
- **D12 Decay-gate precision (aligned, PI decision 2026-10-28).** The
  replaced forward computes `g = exp(−exp(A_log) · softplus(a +
  dt_bias))` with `a` and `dt_bias` promoted to float32 first, as
  transformers does (`mlx_lm` adds and applies softplus in bf16).
- **D13 Router arithmetic (aligned).** `mlx_lm` selects the top-k
  experts on softmax probabilities rounded to bf16, where distinct
  logits can tie; transformers selects on float32 probabilities and
  renormalises in float32. The harness replaces the MoE forward with
  the transformers arithmetic, and the expert-usage histogram counts
  the router's actual selections, summed over layers (§6).
- **D14 Expert-usage histogram scope (both backends, fixed).** The
  histogram is keyed by expert and summed over layers, as §6 states
  (an earlier draft of the code keyed it by layer, which made it
  depend on RESTRUCTURE's layer order).
- **D10 Tokenizer (PI decision S8, 2026-10-25).** Every token count
  and every token id in the harness (scoring, T_s, the replay
  schedule, OPR's S_OPR) comes from the pinned raw `tokenizer.json`
  through the `tokenizers` library without special tokens, the
  tokenizer that counted §5.1b step 3, on both backends (its SHA-256 is
  checked at load). The transformers tokenizer of the same files
  differs on 1 of 7 850 corpus prompt / prompt+choice texts (a
  Devanagari combining mark; 63 vs 66 tokens) and is not used. The
  `tokenizers` versions in use (0.22.2 in the locked environment,
  0.23.3 for the corpus build) give identical ids on all 7 850 texts.

**Memory on the Studio (arithmetic, bf16).** Language-model weights
≈ 35 B × 2 B ≈ 70 GB. During a replay step MLX holds the base
snapshot (70 GB, kept by reference), the current weights (70 GB once
they differ from the base), the gradients (70 GB) and, while the
update is evaluated, the new weights before the old ones are
released (70 GB): a nominal tensor subtotal of ≈ 280 GB, with no
optimizer state (the update is computed directly, D5). It is not a
complete transient estimate: float32 update intermediates, the
allocator's cache and activations come on top. Activations of the differentiable
linear-attention recurrence are not bounded by arithmetic here; the
single-cell smoke measures the peak (`mx.get_peak_memory`) before
any further cell runs. Budget: 512 GB of unified memory minus the OS
and other resident processes (a virtual machine of ≈ 118 GB RSS was
resident on 2026-10-25), i.e. ≈ 390 GB for E1.

**Port revalidation.** *Parity, PyTorch vs MLX* (macM1,
`scripts/e1_mlx_parity.py`, tiny Qwen3.5-MoE with the 35B's
architecture and the same weights; tolerances fixed in the script).

*First run, 2026-10-25, before the D11 / D12 alignment (float32;
accepted as is by the PI on 2026-10-28):* token ids and base-model
routing identical; choice scores max |Δ| 2.0e-4 (tol 1e-3), identical
argmax; per-token cross-entropy max |Δ| 1.005e-3 against a 1e-3
tolerance (a marginal fail, reported, not re-toleranced); one SGD step
max relative Δ 5.7e-3; 12 cells with identical decisions and budgets;
overall `ok: false`. Reports
`docs/milestones/e1-mlx-parity-{fp32,bf16}-2026-10-25.json`.

*After the D11 / D12 alignment, 2026-10-28 (code at the commit that
records this paragraph; run under the macM1 heavy-job lock).*
**float32: every check passes (`ok: true`).** Token ids and base-model
routing identical; choice scores max |Δ| 3.9e-6, identical argmax;
per-token cross-entropy max |Δ| 6.7e-6, no position above 1e-2; one
SGD step: all 76 tensors change on both, max relative Δ 1.7e-4, no
spurious change; 12 harness cells (p_min, p_equ, p_max, OPR, shuffle,
template × 2 seeds) with identical budgets, token and step counts,
OPR corpus sizes, per-item correctness, accuracies and RESTRUCTURE
index; drift / norm change rel. Δ ≤ 3.9e-7, router drift ≤ 3.0e-6,
KL ≤ 1.1e-7, usage L1 Δ 0; restore bit-exact on both backends; MLX
run to run max |Δ| 6e-8 (not bit-identical, within the 1e-6
envelope). The q / k normalisation and decay-gate formulas were the
source of the earlier float32 differences.
**bf16 agreement envelope (`--quick`: one seed, p_min / p_equ / p_max
/ OPR; accepted and declared by the PI on 2026-10-28; overall report
verdict `ok: false`):** token ids identical; base-model expert-usage
histogram not identical (near-tie routing in bf16); choice scores max
|Δ| 0.062 nat with identical argmax on the scored eval items; OPR
reward selection differed on 1 of 40 replay items (OPR corpus 9 vs
10, so one OPR cell's budget split differs; its eval decisions are
identical); per-token cross-entropy: 67 of 590 positions differ by
more than 1e-2, the others by less than 1e-2 (max 9.93e-3); one SGD step at lr 1e-2:
69 of 76 tensors change on the PyTorch side and the same on MLX (no
spurious change), max relative Δ 0.69 on a norm weight (bf16
quantisation of small updates); in the four cells, per-item
correctness pre and post, accuracies, collapse flags and RESTRUCTURE
index identical, drift / norm change rel. Δ ≤ 4.0e-3, router drift
≤ 1.8e-2, KL ≤ 5.4e-4, usage L1 Δ ≤ 1.0e-2; restore bit-exact; MLX
bf16 run to run max |Δ| 2.5e-4 (50 tensors). The PyTorch side of this
comparison is the CPU bf16 path (its own bf16 rounding, D5), not CUDA.
Reports: `docs/milestones/e1-mlx-parity-{fp32,bf16}-aligned-2026-10-28.json`.
*Single 35B cell on the Studio* (smoke mode, pilot fixtures, seed
2000, `p_min`, code `8f2173e`, record
`docs/milestones/e1-mlx-smoke-1cell-2026-10-25.json`): weights
verified against the pins, T_s = 18 335 = achieved loss tokens, 400
updates, 80 eval items, pre 0.563 → post 0.653, no collapse, no
invariant violation; **wall 1 483 s; peak MLX memory 219.8 GB**
(active 138.6 GB at the end; the host RSS high-water mark of 70.7 GB
is a different view of the same memory, not an addition). Limits: the
GPU wired limit read with `sysctl iogpu.wired_limit_mb` is 466 944 MB
(456 GiB); MLX reports a recommended working set of 489.6 GB; a
virtual machine of ≈ 120 GB RSS (another user's) was resident (read
with `ps`, not part of the record). The peak is well inside the
≈ 390 GB left for E1. Component timings (`scripts/e1_mlx_timing.py`,
same model, 20 pilot items): one scoring forward 0.114 s (≈ 27
tokens); one replay update 3.28 s at 27.2 tokens (20 updates, 544
input tokens); drift 3.8 s; restore 3 ms. The update-cost model is
fitted from two points: the timing run (3.28 s at 27.2 tokens) and
the smoke cell's replay time obtained by subtraction (1 483 s wall −
640 scoring forwards × 0.114 s − drift ≈ 1 406 s for 400 updates of
≈ 46.8 input tokens, i.e. ≈ 3.5 s), giving ≈ 3.0 s fixed per
full-parameter step + ≈ 0.011 s per token. A direct measurement on
long records replaces this two-point fit in the full smoke.
*Projection* (confirmatory corpus: fact records ≈ 40.2 tokens, prompts
≈ 35.0, choice passes ≈ 40.5, neutral filler ≈ 10.8; scoring forward
taken as 0.13 s; tuning items assumed to have the same mean lengths
as the confirmatory ones): T_s ≈ 5 × 40 × 39.2 ≈ 7 840 loss tokens.
Updates per cell: ≈ 200 (≈ 680 s) for each of the six arms that
replay fact-length records — `p_min`, `p_equ`, `p_max`,
`p_ctrl_vanilla_replay_tuned`, the exploratory `p_ctrl_lr_matched`
and `p_ctrl_shuffle` (the word shuffle can change the token count of a
record slightly; assumed equal); ≈ 231 for `p_ctrl_template`; ≈ 800
(≈ 2 500 s) for `p_ctrl_sft`; for OPR the charge is ≈ 2 160 tokens
(160 extra scoring forwards, ≈ 21 s, reading ≈ 6 480 tokens), leaving
≈ 5 680 update tokens, ≈ 145 updates (OPR corpus size assumed ≈ 20 of
40); 0 for `p_noise`. Scoring: 8 × n_eval forwards per cell (≈ 135 s
at n_eval = 130, ≈ 265 s at 255). Per confirmatory seed (10 profiles)
≈ 2.6 h at n_eval = 130 (≈ 2.9 h at 255); 141 seeds ≈ 365-416 h;
tuning (120 cells, n_eval = the 432 remaining tuning items, ≈ 450 s of
scoring per cell) ≈ 35 h; calibration (60 cells) ≈ 19 h; smoke (30
cells, rough: 30 × the measured 1 483 s; the `p_ctrl_sft` cells will
be longer) ≈ 12 h; headroom minutes. **Total ≈ 431-482 h, i.e. about
18-20 days of exclusive Studio time**, dominated by the fixed cost of
a full-parameter step on 35 B parameters. The PI
decides whether this fits the budget before filing; N is not reduced.

**Memory (CUDA path, arithmetic, bf16, full-parameter SGD; not the
registered backend since 2026-10-25).** Weights ≈ 35 B ×
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
`p_ctrl_sft` texts), + one restore (a 70 GB host → device copy on
CUDA; a re-binding of the base arrays on MLX, D9). At 35B, wall and
peak memory are measured for one MLX cell only (above); the other
profiles and the full smoke workload remain unmeasured. The smoke on the filing hardware
(seeds 2000-2002, all 10 profiles = 30 cells, pilot set; on the
Studio a single cell first) must record per-cell wall, peak memory
(`mx.get_peak_memory` on MLX, `torch.cuda.max_memory_allocated` per
device on CUDA) and host RSS, achieved tokens and update counts. Because the pilot
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
2. **Done (2026-10-25):** corpus loader `scripts/e1_corpus_io.py`,
   wired into `pilot_cycle3_cuda.py --mode`: it verifies the corpus
   file, every item hash and both partition hashes, and each mode
   reads one partition only (confirmatory: CONFIRMATORY of a FROZEN
   corpus; tuning / calibrate: TUNING of a FROZEN corpus; headroom:
   TUNING of the PENDING-HEADROOM corpus; smoke: the pilot fixtures
   through their partition manifest). Seeds are fixed per mode
   (0-140, 1000-1009, 2000-2002); the pinned model revision is
   enforced at load.
3. **Done (2026-10-25):** token-budget schedule
   `scripts/e1_schedule.py` (T_s, m-fold repeats, cycling, terminal
   truncation to exactly the budget), wired with the §5.4 scoring
   charge of `scripts/e1_budget.py`; achieved tokens, optimizer steps
   and truncations are written to each cell.
4. **Done (2026-10-25):** profile table, corpora (prompt-only
   template, restricted MC-OPR with corpus size and S_OPR per cell),
   tuning selection, empty-OPR handling, LR-matching calibration of c
   and the frozen tuning amendment (`scripts/e1_profiles.py`,
   `--mode tuning|calibrate|amend`).
5. **Done (2026-10-25):** per-cell drift (full norm), collapse flag,
   frozen-reference KL and "previously correct, now lost", router
   drift, expert-usage L1 and mean replay loss
   (`scripts/e1_diagnostics.py`; operationalisation in its
   docstring: probe set = the cell's own eval items, KL over the
   softmax of the 4 choice scores).
6. **Done (2026-10-25):** `analyze_e1_cuda.py` replaced by the §2 / §4
   / §6 / §7 decision code (states, gates, joint headline, secondary
   and exploratory contrasts, per-source description).
7. **Done (2026-10-25)** up to step 8: `scripts/e1_fetch_sources.sh`,
   `scripts/e1_sources.py` and `scripts/e1_corpus_build.py` (pins,
   row mapping, rules 1-9, the reference core, manifest without item
   text, `--check`, `--freeze` for step 9). Step 9 needs the base
   checkpoint on the CUDA node (`--mode headroom`, TUNING only).
   All code is torch-free and tested, except the harness wiring,
   whose tests run on a tiny CPU model where torch is installed.

**Corpus build, steps 1-8 (2026-10-25, PI go).** Code at `d82a512`
(build path unchanged since), host macM1 (M1, 32 GB, macOS 27,
CPU, `nice 19`), Python 3.14.4, tokenizers 0.23.3, pinned tokenizer
of §10 item 1; sources fetched on macM1 with
`scripts/e1_fetch_sources.sh` and verified against their pins
(Daily Oracle and MMLU-CF SHA-256 equal to §5.1a; RealTime QA's
41 `past/2026/*_qa.jsonl` files equal to the blobs of commit
`e288dcf`; every raw SHA-256 in the manifest). Wall time 159 s, peak
RSS 0.6 GB. `--check` rebuilt it bit-identically. Status
**PENDING-HEADROOM**: step 9 needs the base checkpoint on the CUDA
node (`--mode headroom`, TUNING only), then `--freeze`. No item was
scored.

| step (items) | `realtimeqa` | `daily_oracle` | `mmlu_cf` |
|---|---|---|---|
| 1 load | 700 | 18 557 | 10 000 |
| 3 exclude | 700 | 18 556 (1 empty field) | 4 653 (5 346 rule 3, 1 > 256 tokens) |
| 4 date | 545 | 492 | 4 653 |
| 4 contamination | 545 | 491 | 4 653 |
| 5 dedupe / skeleton cap | 543 | 484 | 4 544 |
| 6 cap (k = 543) | 543 | 484 | 543 |
| 7 tuning / confirmatory | 163 / 380 | 146 / 338 | 163 / 380 |

Pooled: **confirmatory 1 098** (≥ 300), **tuning 472** (≥ 120);
pre-headroom weights w = 0.346 / 0.308 / 0.346 (`realtimeqa` /
`daily_oracle` / `mmlu_cf`). No item fell in a component with a
reference row; the skeleton cap removed none. Rule 3 removals per
variant (items carrying it): `None of the other choices` 4 634,
`None of the mentioned` 178, `None of these` 141, `None of the above`
137, `none of the mentioned` 87, `none of these` 51,
`none of the above` 42, `None of the above.` 15,
`None of the Mentioned` 14, `none of the above.` 10, `none of these.`
7, `None of these.` 5, `None of the above is correct.` 4,
`None of These` 3, `None of above` 3, `none of the choices.` 3,
`None is applicable` 2, `None of these alternatives is correct.` 2,
and 1 each for the other nine strings (an item can carry two).
Gold option longest (descriptive, rule 2): 199 / 543, 165 / 484,
212 / 543. Corpus file SHA-256
`f8379b810353d25e3c85e4e94b9a8d02553ea8c18abdef631a0289339497c0a2`
(not committed, rule 9); confirmatory partition
`546f44c6d2fc23f7a355a9182dd3e635b47cd52082d0b1ac347abbcf4914cb3f`,
tuning `7aea63a8630eac6c9bf045d0644db898c7b779f35bfb98107114905b35afb0f7`;
manifest (no item text) committed as
`docs/milestones/e1-corpus-manifest-2026-10-25-pending.json`
(SHA-256 `95c69035…d056344`; rebuilt at `e4d85e8` to add the
provenance dates of the PI decision of 2026-10-25, corpus file
byte-identical). These hashes are pre-headroom; the filed hashes are
those of the FROZEN corpus after step 9.

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

1. ~~CUDA node (§9)~~ — **decided by the PI on 2026-10-25: the
   backend is Apple MLX on the Mac Studio M3 Ultra** (a protocol
   revision before filing, recorded in §9 "Backend" with deviations
   D1-D14; the CUDA path is kept for the parity check only).
   **Pending (PI, 2026-10-28):** the hardware for the full run — the
   Mac Studio (≈ 18-20 days of exclusive time, §9 projection) or a
   rented 2-4 × H200 node — is still open; nothing else in this item
   changes until the PI decides. **Model revision pinned by the
   PI on 2026-10-25:** `Qwen/Qwen3.6-35B-A3B` at commit
   `995ad96eacd98c81ed38be0c5b274b04031597b0` (last modified
   2026-04-24T02:53:42Z, resolved 2026-10-25). This pin fixes the
   tokenizer used by step 3 of §5.1b now and also fixes the model
   revision of the later scoring node, unless the PI changes it
   before filing (a change would require rebuilding the corpus).
   Tokenizer files downloaded at that commit (no weights), SHA-256:
   `tokenizer.json`
   `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
   (12 807 982 B, equal to its LFS oid; the only file the step-3
   counter reads), `tokenizer_config.json`
   `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b`,
   `vocab.json`
   `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003`,
   `merges.txt`
   `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d`
   (git blob ids of the last three match the pinned commit).
   Counted with the `tokenizers` package 0.23.3 (wheel
   `tokenizers-0.23.3-cp310-abi3-macosx_11_0_arm64.whl`, SHA-256
   `68649e97d5b43c44c031d8d848874a6eecae8f8fe40ea989aa777a5a83aca716`,
   installed with `pip --require-hashes --no-deps` in a separate
   environment; no BOS / EOS added).
2. ~~LR-matched replay arm~~ — decided by the PI on 2026-10-24:
   registered as an exploratory arm (§5.4), reported only.
3. ~~Fact-corpus source~~ — decided by the PI on 2026-10-24: all
   three public sets (§5.1a); the pooled-bank design with `source` as
   a stratum and the `mmlu_cf` cap were confirmed the same day.
4. ~~Harness tokenizer (D10)~~ — decided by the PI on 2026-10-25
   (S8): the raw pinned `tokenizer.json` through `tokenizers` (§9, D10).
