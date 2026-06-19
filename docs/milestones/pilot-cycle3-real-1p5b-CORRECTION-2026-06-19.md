# CORRECTION — the 1.5B framework-C "GO" is a mega_v2 fixture leak

**Date:** 2026-06-19
**Affects:** `docs/milestones/pilot-cycle3-real-1p5b.{json,md}` (verdict **GO**,
H1 rejects H0 3/3, p≈e-20) and the corresponding `STATUS.md` line
("2026-06-04 cycle-3 real 1.5B pilot … Verdict GO").
**Status of this note:** leak CONFIRMED from source + committed data + controls.
Leak-free disjoint-fold re-run DONE (2026-06-19, kxkm) — see §"Leak-free re-test".
Outcome: the GO is invalid (bulk = leak + generic SGD warmup), but a small
replay-specific signal survives leak-free and is NOT a clean null.

## Finding

The committed 1.5B pilot's positive composite delta (p_min +0.1037,
[+0.079,+0.118]) is **carried entirely by the `mega_v2` benchmark, and that
gain is train/eval recall, not consolidation.** On the shared benchmarks the
dream profiles go *negative*; the headline is a benchmark artifact.

Per-benchmark decomposition of the committed p_min cells (e.g. seed 0):

| benchmark | pre | post | Δ |
|---|---|---|---|
| mmlu | 0.64 | 0.63 | −0.01 |
| hellaswag | 0.46 | 0.45 | −0.01 |
| **mega_v2** | **0.3297** | **0.6593** | **+0.330** |
| composite | 0.4766 | 0.5798 | **+0.1032** |

## Mechanism (root cause)

1. **The fixture is missing.** `harness/real_benchmarks/mega_v2_eval.py`
   resolves `_DEFAULT_MEGA_V2_FALLBACK = tests/fixtures/mega_v2_sanity.jsonl`,
   which **does not exist** on disk. The loader therefore falls back to the
   hand-authored in-module `_mega_v2_default_fallback_records()` — **20 rows**,
   explicitly "sized to support the Phase B 8-train / 8-eval split".
2. **n_samples=100 ≫ 20 rows collapses the disjoint fold.** The pilot runs
   `n_samples_per_benchmark = 100`. In `_load_mega_v2_records`, for
   `fold ∈ {train, eval}` with a corpus smaller than `2*n_samples`, the loader
   cycles: `block = [shuffled[i % 20] for i in range(200)]`, `train = block[:100]`,
   `eval = block[100:200]`. Both blocks therefore **cover all 20 distinct rows**.
   The code's own comment (lines 384-387) admits it:
   > "If the underlying corpus is smaller than n_samples, **some raw rows
   > inevitably appear in both folds**, but their positions in the fold block
   > remain disjoint."
   The folds are disjoint in *index*, identical in *content*.
3. **Dream replays the eval content.** `replay_real` consumes `fold="train"`
   (all 20 facts, ~5× each); `evaluate_mega_v2` scores `fold="eval"` (the same
   20 facts). The model is graded on the exact strings it just SGD-memorised.
   `mega_v2 pre ≈ 0.330` matches the 20-row fallback baseline exactly.

## Confirming controls (kxkm RTX 4090, torch reimpl, 3 seeds)

A torch reimplementation with the same 20-row fallback reproduces the effect and
the controls isolate the cause:

| profile | mega_v2 pre → post |
|---|---|
| p_min (dream replay) | 0.330 → 0.61–0.70 |
| p_ctrl_sft (neutral SGD) | 0.330 → 0.43 |
| p_noise (no replay) | 0.330 → 0.330 (flat) |

The gain tracks *replaying the evaluated content*, not the dream operator set.

## Leak-free re-test (2026-06-19, kxkm RTX 4090, genuine loader)

Re-ran with the REAL `_load_mega_v2_records` at **n_samples=10** (2·10 = 20 =
corpus ⇒ NO cycling ⇒ train and eval row-ids genuinely disjoint, verified
empty intersection on all 5 seeds). Qwen2.5-1.5B, cuda, SGD lr=1e-4/3 inner,
5 seeds. mega_v2 accuracy pre→post:

| config | profile | Δacc |
|---|---|---|
| leaky n=100 (overlap 20/20) | p_min | **+0.585** |
| **leak-free n=10 (disjoint)** | **p_min** (dream replay) | **+0.198** |
| leak-free n=10 | p_ctrl_sft (neutral SGD) | +0.094 |
| leak-free n=10 | p_noise (no replay) | +0.000 (flat) |

Two corrections this forces on the original analysis:

1. **The headline was mostly leak.** The +0.585 leaky gain shrinks ~3× to
   +0.198 once train/eval content is disjoint.
2. **The pilot's pre→post delta also conflates generic SGD warmup.** A
   content-free neutral replay (`p_ctrl_sft`) lifts the held-out `exp(-nll)`
   proxy by +0.094 on its own (p_noise = exactly 0.000 confirms it is an SGD
   effect, not noise). The correct effect estimator is **replay vs matched
   neutral SGD (`p_min − p_ctrl_sft`), not post − pre.**

The replay-specific, leak-free, control-corrected effect is therefore
**Δacc ≈ +0.104** (sd 0.026, positive on all 5 seeds) — small, consistent,
on genuinely held-out content, but at N=10 held-out rows/seed and confounded by
the proxy's SGD-sensitivity.

## Impact

- The **GO verdict and the p≈e-20 H1 rejection are INVALID**: the bulk of the
  reported +0.10 is a train/eval leak, and the residual is further inflated by
  generic SGD warmup that the post−pre design does not control for. This is
  consistent with the rest of the programme (vision/symbolic ladder empty,
  Helium-2B real-LLM null).
- **BUT it is not a clean null.** A small replay-specific signal (~+0.10 acc
  over neutral SGD) survives on disjoint held-out content. It is a *candidate*,
  not a finding: N=10, single benchmark, proxy-confounded.
- **Do NOT power the leaky n=100 config** — it sharpens an artifact. A powered
  run is only meaningful on a real ≥200-row fixture with `p_min − p_ctrl_sft`
  as the effect estimator.

## Fix + path to an honest verdict

Step 2 below is DONE (see §"Leak-free re-test"); steps 1 and 3 remain to
adjudicate the residual ~+0.10 replay-specific signal.

1. **Ship a real `tests/fixtures/mega_v2_sanity.jsonl`** with **≥ 2·n_samples
   rows** (≥ 200 for n_samples=100) from the FineFab mega-v2 corpus, so the
   disjoint fold holds at content level at scale. The corpus is an internal
   artefact (no HF pin) — not present on macM1 or kxkm; it must come from the
   FineFab/KIKI-Mac_tunner artefact path.
2. ~~Leak-free directional check at n_samples=10~~ — DONE: the gain shrank ~3×
   and the proper estimator is replay − neutral-SGD, leaving ~+0.10.
3. **Powered run on the real fixture**, scoring the effect as
   `p_min − p_ctrl_sft` (replay over matched neutral SGD), NOT post − pre.
   If the ~+0.10 holds adequately-powered on a real held-out fold → the first
   non-artifact framework-C signal at the LLM tier (small but real). If it
   vanishes → clean null. Either is a legitimate, R1-defensible result.

## Also fix (code)

- Make `_load_mega_v2_records` **raise** (or loudly warn) when the corpus has
  `< 2·n_samples` rows for `fold ∈ {train, eval}`, instead of silently cycling
  into a content leak. The current behaviour (lines 380-391) is the root cause.
- Consider adding `p_ctrl_sft` (neutral-SGD) as the **default baseline** for the
  delta, so generic SGD warmup is not reported as consolidation.

## Mechanism follow-up (2026-06-19, kxkm — pipeline/mechanism only)

To test whether the residual ~+0.10 is robust or N=10 noise, two further
leak-free runs (genuine loader, content-disjoint folds verified ∅):

**Run A — real 20-row fallback, n=10, 20 seeds** (powering the 5-seed result):
`p_min − p_ctrl_sft = +0.097`, sd 0.028, 95% CI **[+0.085, +0.109]**, **20/20
seeds positive**. → the replay-specific residual is **robust, not noise** on the
hand-authored facts. (Half of p_min's raw +0.19 is generic any-SGD lift
(`p_ctrl_sft` +0.095); the contrast isolates the ~+0.10 specific to replaying
the actual fact content.)

**Run B — synthetic 500-row fixture, n=100, 5 seeds** (intended as a clean
null-control). It did **NOT** return ~0: `p_min` +0.89 vs `p_ctrl_sft` −0.003.
The lab synthetic generator emits a rigid template
("synthetic context N domain X variant Y" → "expected-NNNNN"); train and eval
rows are row-id-disjoint but share the identical template, so replay teaches the
low-entropy format and it transfers to held-out rows. **Lesson: row-id
disjointness does not guarantee a clean null when content shares a template —
the replay-vs-neutral-SGD design is itself confounded by template/format
transfer.**

**Refined verdict.** The residual ~+0.10 on real facts is robust under power, but
**cannot be attributed to genuine consolidation**: Run B shows that shared
surface structure alone produces a (large) spurious `p_min − p_ctrl_sft` gap, and
the 20 hand-authored facts share cloze structure too. A clean framework-C test
therefore needs (a) a real ≥200-row corpus of **independent** facts (no shared
template), (b) the `p_min − p_ctrl_sft` estimator, AND (c) a template-matched
control to subtract format transfer. Item (a) requires the absent FineFab corpus
— so the candidate stays **non-adjudicable** here, and the eval design needs the
extra (c) control regardless. (Synthetic/toy results above are
pipeline/mechanism only — never framework-C empirical claims.)

## Provenance

Diagnosed while attempting to power the 1.5B pilot on the kxkm RTX 4090
(torch CUDA pilot `scripts/pilot_cycle3_cuda.py`, which omits mega_v2 and so
returned the honest mmlu+hellaswag null). Source verified directly:
`harness/real_benchmarks/mega_v2_eval.py:330-409`. No committed result file was
mutated by this note.
