# G6-Studio Path C pilot pre-registration (synthetic CL convergent validation)

**Date:** 2026-05-04
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**Sister pre-regs:**
- `docs/osf-prereg-g6-studio-path-a.md` (Path A INSUFFICIENT verdict — synthetic 10/10 fixture).
- `docs/osf-prereg-g6-studio-path-a-star.md` (Path A* in flight at lock time — real MMLU 50/50 fixture, real-knowledge benchmark).
**Substrate:** unchanged — real `SpikingKiki-35B-A3B-V4` substrate via `MicroKikiSubstrate.load()` Path 2 with `DREAM_MICRO_KIKI_REAL=1`, KIKI-Mac_tunner mlx_lm fork, base model `Qwen3.6-35B-A3B` bf16.
**Adapter stack:** unchanged — `SpikingKiki-V4-adapters/` LoRA tensors mutated by 4-channel dream handlers (live-tensor coupling).
**Benchmark:** **synthetic symbolic CL stream** (5 bit-string transformation rules : `xor_shift_3`, `parity_chunks_4`, `gray_code`, `reverse`, `complement_alternating`) with `n_train = 50` + `n_eval = 50` per sub-domain ; total 500 records pinned at `tests/fixtures/symbolic_cl_g6.jsonl` SHA-256 to be filled at first commit. **Each rule is something the 35B definitely does NOT know pre-trained**, so the LoRA must actually learn it ; conversely, real forgetting can occur when the LoRA is overwritten by the next rule.
**Compute:** N = 5 seeds × 4 arms × 5 sub-domains = **100 measurements** (20 cells × 5 sub-domains). Per-cell wall ~50 min on Studio M3 Ultra (extrapolated from Path A* rate ; symbolic strings are shorter than MMLU prompts so per-step training may be faster). Total ~3-5 h overnight on Studio.
**Lock commit:** *(filled by introducing commit hash)*
**Lock timestamp:** 2026-05-04 (pre-driver-run, locked BEFORE Path A* aggregator runs to avoid contamination from Path A* verdict).

## §1 Background and convergent-validation rationale

The G6-Studio Path A pilot (commit `d963b24`) returned **INSUFFICIENT** because the synthetic 10/10 MMLU fixture was too small to push the LoRA above the underperforming-baseline floor. The G6-Studio Path A* pilot (commit `fd8bf52`, in flight at lock time) re-runs with real MMLU 50/50 budget and the identical pipeline.

Whatever Path A* returns (H9-A confirmed / H9-B confirmed / H9-C confirmed / INSUFFICIENT-2), the verdict on its own is **methodologically insufficient** for a confirmatory claim about the framework C real-LLM tier, because :

- **MMLU tests pre-trained factual recall.** Qwen-35B-A3B already "knows" anatomy / astronomy / business_ethics / clinical_knowledge / college_biology at substantial accuracy (often > 0.5 from base weights alone). LoRA fine-tuning marginally adapts response *form*, not knowledge content. Catastrophic forgetting between MMLU sub-domains is therefore *small* — there is little to forget that wasn't already in the base model.
- **A positive H9-A on MMLU could be artefactual.** Possible alternative sources of a Welch reject + g > 0 :
  1. LoRA delta correlation between sub-domains (parameter sharing, no dream needed).
  2. Eval order or prompt-format bias (P_max arm gets the dream-modified adapter which might align prompts differently).
  3. `mlx_lm.generate` letter-argmax noise at small N.
  4. Floor effect in baseline (early subdomains evaluated more times, so retention measured against more denominators).
- **A null H9-B on MMLU could be artefactual too.** If the model never had real forgetting to recover from, RECOMBINE has nothing to re-combine. The null is consistent with both "framework wash-out at real-LLM" (the H9-B hypothesis) AND "MMLU is the wrong benchmark" (no question-of-fact at all).

**Convergent-validation principle :** a positive empirical claim about the framework C real-LLM tier requires the effect to replicate on a benchmark that actually pressures the consolidation question — i.e., one where the model must learn something it doesn't already know AND where catastrophic forgetting is observable. Path C provides that benchmark.

## §2 Hypotheses (confirmatory)

- **H10-A (synthetic-CL transfer of REPLAY+DOWNSCALE coupling)** — on the 5-rule symbolic CL stream, `retention(P_max with mog)` is statistically distinguishable from `retention(P_max with none)` with predicted positive sign and `Hedges' g ≥ 0.5`. Welch two-sided rejects H0 at α = 0.05 (single new test, no Bonferroni inheritance from G6 family). Strict large-effect threshold `g ≥ 2` at N = 5 detection floor. Confirming H10-A is the predicted positive empirical claim that the framework's RECOMBINE prediction is restored at real-LLM scale on a benchmark that actually pressures the question.
- **H10-A*** — exploratory medium-effect band (0.5 ≤ g < 2).
- **H10-B (synthetic-CL real-LLM wash-out)** — fail-to-reject H0 at α = 0.05, OR rejection with `g < 0.5`. Even on a benchmark that should pressure the question, the framework C effect does not survive the substrate transfer to production-scale LLM. EC stays PARTIAL ; DR-3 evidence file gets an H10-B row.
- **H10-C (synthetic-CL negative-direction break-through)** — rejection with negative sign (RECOMBINE actively *hurts* synthetic-CL retention).
- **H10-D (joint convergent-validation conjunction)** — derived `H9-{A_or_A*}_resolution AND H10-{A_or_A*}_resolution`. **NOT a Welch test ; logical aggregation only.** Resolution states :
  - `confirmed_positive` iff both Path A* AND Path C return H9/H10-A or H9/H10-A* with consistent sign AND `min(g_path_a_star, g_path_c) ≥ 0.5`. The framework C real-LLM tier prediction is restored under convergent validation.
  - `confirmed_null` iff both return H9/H10-B (G5-bis MLX-only artefact extends to real-LLM under both benchmark types).
  - `confirmed_negative` iff both return H9/H10-C (RECOMBINE empirically harmful at real-LLM scale).
  - `divergent` iff Path A* and Path C differ on resolution category. **The divergent case is itself a positive scientific finding** : it means the apparent effect at real-LLM is benchmark-dependent, and the convergent-validation principle is exactly what surfaced it. Honest reporting in Paper 2 §7.1.13 reports both sub-results without aggregation.
  - `unresolved` iff either path returns INSUFFICIENT.

## §3 Power analysis

Identical to Path A pre-reg §3 — N = 5 detects |g| ≥ 1.85 at 80 % power. The H10-A* exploratory band captures medium effects (0.5 ≤ g < 2). The H10-D conjunction does NOT compose Welch tests (no Bonferroni adjustment beyond the per-path α).

## §4 Exclusion criteria

Identical to Path A pre-reg §4. Underperforming-baseline rule unchanged (`acc[S_1 after S_1] < UNDERPERFORM_THRESHOLD = 0.30`). Symbolic CL is designed so LoRA *will* learn each rule from 50 train examples (target acc > 0.5 on first sub-domain after own training), so exclusion-rate is expected to be low — well under the 50 % threshold.

## §5 Substrate / driver paths

- Driver : **re-uses** `experiments/g6_studio_path_a/run_g6_studio_path_a.py` (no new driver — only the fixture changes). Path C is **not** a code change, it is a fixture + pre-reg change.
- Fixture : `tests/fixtures/symbolic_cl_g6.jsonl` (built by `experiments/g6_studio_path_a/build_symbolic_cl.py`). Contains 500 records (5 rules × 100 examples each, split into 50 train / 50 eval per rule by the existing `build_subdomain_stream` loader).
- Builder : `experiments/g6_studio_path_a/build_symbolic_cl.py` (deterministic, seeded, ~80 lines). Generates 16-bit input strings, applies 5 transformation rules, formats records with the same schema as the MMLU fixture (`{"subject": rule_name, "question": ..., "choices": [...], "answer": int}`) so the existing driver consumes them without modification.
- Substrate, dream handlers, fork, Metal cache, registry profile keys : **all identical** to Path A / Path A*.
- Output milestones :
  - `docs/milestones/g6-studio-path-c-2026-05-04.{json,md}`
  - per-subdomain partial dumps `docs/milestones/g6-studio-path-c-partial-<rule>-<arm>-seed<seed>-step<NN>-2026-05-04.json`

### §5.1 Synthetic CL stream specification

5 transformation rules on 16-bit input strings :

1. **`xor_shift_3`** : `output = input XOR (input >> 3)`. Tests bit-level XOR pattern recognition.
2. **`parity_chunks_4`** : `output[i] = parity(input[i*4:i*4+4])` for i in 0..3 ; result is 4 bits, padded to 16 with zeros. Tests local-window aggregation.
3. **`gray_code`** : `output = input XOR (input >> 1)`. Standard binary-to-Gray transformation.
4. **`reverse`** : `output = reversed(input)`. Trivial position-mapping.
5. **`complement_alternating`** : `output[i] = NOT input[i] if i % 2 == 0 else input[i]`. Tests positional masking.

Each example formatted as a multiple-choice question (compatible with MMLU schema for driver re-use without modification) :
- `question` : "Input: <16 bits>\nRule: <rule_name>\nWhat is the output?"
- `choices` : 4 options : the correct answer + 3 distractors (each generated by applying a different rule to the same input).
- `answer` : index of the correct choice ∈ {0, 1, 2, 3}.

This gives the LoRA something **definitely not in pre-training** to learn (each rule is a specific symbolic operation, not natural-language knowledge), AND a clean catastrophic-forgetting signal (training on rule i overwrites rule j's adapter weights, retention drops measurably without dream consolidation).

## §6 DualVer outcome rules

| Outcome | EC bump | FC bump |
|---|---|---|
| Row 1 — H10-A confirmed (g ≥ 2 strict) | EC stays PARTIAL ; framework C 4-channel coupling validated at real-LLM-scale on a forgetting-pressuring benchmark ; H10-D conjunction with Path A* updates DR-3 evidence v0.x. | FC stays C-v0.12.0 |
| Row 1* — H10-A* exploratory (0.5 ≤ g < 2) | EC stays PARTIAL ; reported as exploratory ; needs N > 5 follow-up. | FC stays C-v0.12.0 |
| Row 2 — H10-B confirmed (null OR g < 0.5) | EC stays PARTIAL ; G5-bis MLX-only artefact verdict empirically extends from toy E-SNN to real-LLM at *both* knowledge-recall AND symbolic-CL benchmark types ; DR-3 evidence v0.x extension. | FC stays C-v0.12.0 |
| Row 3 — H10-C confirmed (negative sign) | EC stays PARTIAL ; framework's RECOMBINE prediction further weakened at real-LLM scale ; DR-3 evidence v0.x records negative-direction synthetic-CL finding. | FC stays C-v0.12.0 |
| Row 4 — H10-D `divergent` (Path A* and Path C disagree) | EC stays PARTIAL ; **positive scientific finding** : the framework's real-LLM-tier behavior is benchmark-dependent ; honest Paper 2 §7.1.13 reports both sub-results without aggregation. | FC stays C-v0.12.0 |
| Row 5 — exclusion-rate > 50 % (insufficient cells) | abort and amend pre-reg ; do NOT commit milestone. | n/a |

EC stays PARTIAL across all rows. FC stays at v0.12.0. STABLE promotion of any framework C empirical claim at real-LLM tier blocked unless Row 1 fires AND Path A* returns H9-A or H9-A* with consistent sign (i.e., H10-D `confirmed_positive`).

## §7 Reporting commitment

Honest reporting of all observed scalars regardless of outcome.

The convergent-validation principle is explicit and publicly committed : a positive H10-A on its own is reported as exploratory ; the H10-D `confirmed_positive` joint conjunction is the only path to a confirmatory claim about framework C at real-LLM tier.

If H10-D `divergent`, the divergence is reported as the headline result, not aggregated into a single verdict. This is the *correct* scientific outcome for a benchmark-dependent effect.

## §8 Audit trail

Cells registered via `harness/storage/run_registry.py` with profile keys `g6-studio-path-c/<arm>` (distinct from Path A's `g6-studio-path-a/<arm>` namespace). Milestone artefacts under `docs/milestones/g6-studio-path-c-2026-05-04.{json,md}` plus per-subdomain partial dumps for resumability.

## §9 Deviations

Pre-known envelopes :

a. Symbolic CL fixture build fails (RNG seed collision, format mismatch) — abort and file §9.1 amendment.
b. `acc_initial` < 0.30 for majority of seeds despite the symbolic design — raise epochs from 50 → 100 iters per subdomain ; document in milestone header.
c. Per-cell wall > 70 min sustained — propose N = 3 reduced run.
d. LoRA learns the rules trivially (acc → 1.0 even without dreams) — design too easy ; if observed, abort and amend with longer rules (32-bit instead of 16-bit) or more rules (8 instead of 5).

Any deviation outside the envelopes requires an amendment commit *before* the affected cell runs.

### §9.1 — TBD on first run if surprises surface

Reserved per the pattern.
