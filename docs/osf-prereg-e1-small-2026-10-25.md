# E1-small: replay controls at 4B (dense) pre-registration

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI).
**Date:** 2026-10-25 (revision 1).
**Origin:** PI decision of 2026-10-25 (14:43): run E1 first at small
scale on a Qwen 4B dense model on KXKM-AI (one RTX 4090, 24 GB) with the
existing PyTorch / CUDA harness, as a **separate** pre-registration.
The 35B study (`docs/osf-prereg-e1-replay-controls-2026-10-23.md`,
"the E1 draft" below) continues later and is not changed by this one.
**Results are never pooled across model sizes** (§4).
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**Prior evidence (not modified):**
`docs/milestones/pilot-cycle3-real-1p5b-CORRECTION-2026-06-19.md`.
**c_version at drafting:** C-v0.25.0+PARTIAL
**Substrate:** Qwen3.5-4B (dense), bf16, HF transformers,
`Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (proposed pin, §2;
**PI decision S1**).
**Harness:** the E1 harness with the study selector `HL_STUDY=e1-small`,
runner `scripts/run_e1_small_cuda.sh` (§6).
**Lock commit:** *(filled by introducing commit hash at filing)*
**Lock timestamp:** *(filled at filing, pre-run)*

## §1 Scientific role and scope

E1-small is a clean re-test, at small scale, of the evidence that the
2026-06-19 correction invalidated: the 1.5B framework-C GO and the
1.5B↔7B scaling-law GO (mega_v2 fixture leak), and the leak-free
residual `p_min − p_ctrl_sft ≈ +0.10` that the correction left as a
non-adjudicable candidate confounded by template / format transfer.
It applies the E1 design (matched-compute controls, prompt-only
control, tuned vanilla replay, restricted MC on-policy replay, the
independent fact corpus) to a model in the size range of that
evidence (1.5B-7B), on the hardware that produced it (KXKM-AI,
RTX 4090).

**What it can establish.** The states of H14-A, H14-B and H14-C (§3)
for Qwen3.5-4B on the frozen fact corpus, under the E1 protocol.

**What it cannot establish.** Nothing about the 35B model or any other
size. A result here does not predict, replace, or pre-empt the 35B
study; the 35B study's hypotheses, N and α are untouched by it, and
neither study's data enter the other's analysis. No scaling claim
(4B vs 35B) is pre-registered; any comparison of the two studies is
descriptive and outside both confirmatory families. E1-small also does
not adjudicate the historical mega_v2 residual (different task,
endpoint and model), exactly as the E1 draft §1 states for 35B.

## §2 Model choice and pin (PI decision S1)

No Qwen3.6 4B model exists on Hugging Face (2026-10-25: the Qwen3.6
releases are `Qwen3.6-35B-A3B` and `Qwen3.6-27B`, with FP8 variants).
Two dense 4B candidates were compared:

| | `Qwen/Qwen3.5-4B` (proposed) | `Qwen/Qwen3-4B-Instruct-2507` (alternative) |
|---|---|---|
| Revision (full sha) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Last modified | 2026-03-02T00:52:52Z (repo created 2026-02-27) | 2025-09-17T06:56:53Z |
| Generation | Qwen3.5, the architecture family of Qwen3.6 (35B is `Qwen3_5MoeForConditionalGeneration`) | Qwen3 (2025) |
| Text architecture | dense, 32 layers, hybrid: 3 Gated DeltaNet (linear attention) + 1 gated full-attention layer per block of 4, hidden 2560 | dense, 36 full-attention layers, hidden 2560 |
| Post-training | post-trained (hybrid thinking), like the 35B pin | instruct, non-thinking |
| `tokenizer.json` SHA-256 | `5f9e4d49…81cb42`, **identical** to the pinned 35B tokenizer | `aeb13307…92dae4`, different (151 936 vs 248 320 vocabulary) |
| Licence | Apache-2.0 | Apache-2.0 |

**Proposal: `Qwen/Qwen3.5-4B` at `851bf6e…`.** It is the closest
sibling of the 35B pin: same model generation and architecture family
(minus MoE), same post-training style, and a byte-identical tokenizer,
so the corpus build (§5.1b step 3 of the E1 draft), every loss-token
count `T_s` and every scoring pass use the same token sequences for
both studies. The alternative would change three things at once
(tokenizer, attention architecture, post-training generation). The
PI's wording was "Qwen3 4B dense"; if the PI wants the Qwen3
generation literally, the alternative's pin is above and the only
consequence is the tokenizer note of §5 below.

**Tokenizer comparison (proposed pin vs 35B pin).** At the two pinned
revisions: `tokenizer.json` SHA-256
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
(12 807 982 B) in both; `vocab.json`
`ce99b4cb…dfa003` and `merges.txt` `a9d356d7…1db8d` identical (same
SHA-256 as the E1 draft §10 item 1, same git blob ids
`0aa0ce06…` / `a494e019…`). `tokenizer_config.json` differs
(`316230d6…64ac8` vs `5186f0de…2b29b`) **only in `chat_template`**
(sorted-key JSON diff: one key); the harness never applies a chat
template, adds no BOS (`bos_token` is `None`), and scores raw cloze
prompts, so tokenisation is identical.

**What is loaded.** The checkpoint is multimodal
(`Qwen3_5ForConditionalGeneration`: 426 language-model tensors, 297
vision-tower tensors, 15 multi-token-prediction tensors).
`AutoModelForCausalLM` (transformers 5.7.0) loads the text model
`Qwen3_5ForCausalLM` only: 4 205 751 296 parameters, tied input /
output embeddings, layers at `model.layers` (32); the vision tower and
the MTP head are not instantiated, as for the 35B harness. This was
checked on KXKM-AI on CPU (no GPU) on 2026-10-25.

**Weights on the run node.** Downloaded on 2026-10-25 with
`huggingface_hub.hf_hub_download(revision=851bf6e…)` into
`/ai/saisail/e1-small/models` (HF cache layout) on KXKM-AI; only the
files below; every SHA-256 recomputed locally and equal to the Hub's
LFS SHA-256 / size:

| file | bytes | SHA-256 |
|---|---|---|
| `model.safetensors-00001-of-00002.safetensors` | 5 329 398 688 | `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61` |
| `model.safetensors-00002-of-00002.safetensors` | 3 990 429 408 | `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188` |
| `model.safetensors.index.json` | 76 196 | `cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d` |
| `config.json` | 3 161 | `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670` |
| `tokenizer.json` | 12 807 982 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `tokenizer_config.json` | 16 710 | `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8` |
| `vocab.json` | 6 722 759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |
| `merges.txt` | 3 353 259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `chat_template.jinja` | 7 756 | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `LICENSE` | 11 544 | `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a` |

Total 9 342 827 463 B ≈ 9.34 GB (weights 9 319 828 096 B). The harness
refuses any other revision of this repository and checks the SHA-256
of the weight shards and of `tokenizer.json` before loading (§6).

## §3 Protocol inherited from the E1 draft

Unless the table of §4 says otherwise, E1-small **is** the E1 draft:
§1 (background, "what content means"), §2 (hypotheses H14-A / B / C,
unit and pairing, endpoint `acc`, δ = 0.01, exact paired procedures,
family of three with Bonferroni α = 1/60, per-hypothesis states, error
accounting, secondary contrasts), §3 (power: N = 141, the same
planning SD_D and the same caveats), §4 (exclusions, no replacement,
OPR feasibility, differential-exclusion gate, incomplete run), §5
(corpus properties, sources, construction, the `n_eval` rule, the
pilot / smoke set, profiles, token budget and update schedule, scoring
charge, ties, tuning grid and selections, LR-matched exploratory arm
and the PI rules of 2026-10-25 on empty OPR and undefined
calibration), §6 (drift and collapse diagnostics, frozen-reference
monitoring, collapse validity gate), §7 (outcome rules and joint
headline), §8 (reporting and stopping rule), with "35B" / "Qwen3.6-35B-
A3B" read as "Qwen3.5-4B". The seeds are the same numbers (confirmatory
0-140, tuning 1000-1009, smoke 2000-2002), so each seed draws the same
replay / eval items in both studies; this enables no pooled test.

The E1-small family is its own family: α = 0.05 / 3 for H14-A / B / C
of E1-small, with no correction shared with the 35B family. Each study
spends its own 0.05.

**Inheritance is by pinned text (PI decision S7).** At filing, the E1
draft text that E1-small inherits is fixed by its git blob at the
E1-small lock commit and its SHA-256 (at drafting: commit `9172768`,
revision 9 of the E1 draft, SHA-256 `ae00aea5…bf71b5`). Later
revisions of the E1 draft do not change E1-small unless an E1-small
amendment adopts them before its first tuning seed. The alternative
is to file a self-contained copy of the inherited sections.

**Backend.** Revision 9 of the E1 draft moved the 35B study to Apple
MLX on the Mac Studio (its §9 "Backend", deviations D1-D10).
E1-small keeps the **CUDA / PyTorch / HF transformers** backend
(PI decision of 2026-10-25 14:43), i.e. the "CUDA path" of the E1
draft; its §8 rule "every tuning, amendment and confirmatory cell
from the registered backend" reads "CUDA / transformers" here. Of the
35B deviations, only these points carry over, restated for the 4B:
(35B D2) the transformers causal-LM load of this checkpoint is the
language model only (§2), so full-parameter SGD, DOWNSCALE and the
drift / norm measures cover the language model, exactly as `mlx_lm`
does for the 35B; (35B D3) transformers stores the RMSNorm weight `w` of
`norm(x)·(1 + w)` natively, nothing to restore; (35B D5) bf16 parameters,
gradients and SGD update, float32 loss and scoring log-softmax; (35B D6)
CUDA replay is not guaranteed bit-reproducible and no claim relies on
it; (35B D7) RESTRUCTURE swaps two adjacent entries of the decoder-layer
list (masks follow `config.layer_types` by position, which for
unpadded batch-1 sequences gives the same outputs as per-layer types);
(35B D10) the harness tokenizes with the transformers tokenizer of the
pinned files, which differs from the raw `tokenizer.json` on 1 of
7 850 corpus texts (**PI decision S8**, the same question as the 35B
draft §10). The MLX-specific 35B D4, D8 and D9 do not apply (their CUDA counterparts are rows 5, 7 and 8b of §4). Unlike the
35B draft's assumption for CUDA, the `fla` linear-attention kernels
are not installed (S6 of §9 below).

## §4 Differences from the E1 (35B) draft

| # | Item | E1 (35B) draft | E1-small | Forced by |
|---|---|---|---|---|
| 1 | Model | `Qwen/Qwen3.6-35B-A3B` @ `995ad96e…`, MoE 35B total / 3B active | `Qwen/Qwen3.5-4B` @ `851bf6e8…`, dense 4.21B (text) | PI decision |
| 2 | Tokenizer | `tokenizer.json` `5f9e4d49…` | identical file (§2) | — (no change) |
| 3 | Hardware | Mac Studio M3 Ultra, 512 GB unified memory (registered, revision 9) | 1 × RTX 4090 24 GB (KXKM-AI), whole model on one device, no device map | PI decision, size |
| 4 | Memory arithmetic | ≈ 280 GB upper bound in unified memory (snapshot by reference, weights, gradients, update) | §5: ≈ 18-21 GiB GPU peak (weights + gradients + activations) plus a 9.7 GB host snapshot | size, backend |
| 5 | Restore per cell | re-binds the immutable base arrays (35B D9, no copy) | `load_state_dict` copy of ≈ 9.7 GB host → device | backend |
| 6 | MoE drift (E1 §6) | router drift, expert-usage L1 | **not applicable** (no router): recorded as null; the full-parameter drift, collapse, frozen-reference KL, "previously correct, now lost" and mean replay loss are unchanged | dense model |
| 7 | Routed-expert gradients (35B D4) | dense stacked-expert gradient, zero rows for unrouted experts | not applicable: dense model, every language-model parameter receives a gradient | dense model |
| 8 | RESTRUCTURE (`p_max`, 35B D7) | swap of two adjacent decoder-layer entries; MLX masks by each layer's type | same swap; transformers masks by position (`config.layer_types[i]`); adjacent layers may differ in type (linear vs full attention) | backend; hybrid layers |
| 8b | Memory recording (35B D8) | `mx.get_peak_memory` (unified) | `torch.cuda.max_memory_allocated` per device, host RSS | backend |
| 8c | Weight verification | 40 files of `995ad96` checked by the MLX harness (`scripts/e1_model_pins.json`) | the 10 downloaded files of `851bf6e8` checked by the runner, snapshot must hold exactly these files (§6) | model, backend |
| 9 | Headroom (step 9, E1 §5.1 property 5) | 35B base on TUNING | 4B base on TUNING only, separately (§5) | model |
| 10 | Frozen corpus | 35B freeze of the pending corpus | its own freeze of the **same** pending corpus with the 4B headroom scores (§5); hashes filed in this document | model |
| 11 | Date cutoff rationale | 2026-04-16, after the 35B release (2026-04-15) | same cutoff; it also post-dates the 4B release (2026-02-27) | — |
| 12 | Runtime | unmeasured at 35B | estimate in §5, measured by the smoke (§7) | size, hardware |
| 13 | Entry point | `scripts/e1_mlx.py` (registered; `run_e1_cuda_35b.sh` is the CUDA parity path) | `scripts/run_e1_small_cuda.sh` → `scripts/pilot_cycle3_cuda.py` with `HL_STUDY=e1-small`; any other backend or a CPU device is refused (§6) | backend |
| 14 | Milestones | `docs/milestones/e1-35b-replay-controls-<date>` | `docs/milestones/e1-small-4b-replay-controls-<date>` | separate study |
| 15 | α family | its own 0.05 | its own 0.05 (§3) | separate study |
| 16 | Inference | 35B only | 4B only; no inference about 35B (§1) | scope |
| 17 | Backend | Apple MLX on the Mac Studio (revision 9; CUDA kept as a parity path) | CUDA, PyTorch 2.11.0, transformers 5.7.0, hash-pinned (§3, §6) | PI decision |
| 17b | Linear-attention kernels | `mlx_lm` Metal kernel / op-by-op recurrence | transformers torch fallback (no `fla`, S6) | environment |
| 18 | Cell counts | 1 620 (incl. 30 smoke) | same 1 620 | — (no change) |

Everything not listed is identical.

## §5 Memory, runtime, headroom and corpus

**Memory (arithmetic, bf16, full-parameter momentumless SGD, batch 1).**
Text parameters P = 4 205 751 296.
- Weights: 2P B = 8.41 GB (7.83 GiB) on the GPU.
- Gradients: every text parameter gets one (dense model; the tied
  embedding gets a dense gradient): another 7.83 GiB.
- Optimizer state: none (momentumless SGD, `foreach` in-place update).
- Activations for ≤ 512-token sequences (the E1 §5.1 bound): 32
  layers × hidden 2560 / MLP 9216 stored for backward, the Gated
  DeltaNet state in float32 (`mamba_ssm_dtype`), and the 248 320-way
  logits with the float cross-entropy: about 2-4 GiB.
- CUDA context and allocator slack: about 0.5-1 GiB.

Expected peak ≈ 18-21 GiB of the 23.99 GiB card. Other tenants of
KXKM-AI's GPU used about 1.4 GiB on 2026-10-25 (a shared model-swap
proxy can load more), so the runner checks free memory before
starting (§7). If the smoke shows a peak that does not fit, the
proposed fallback is activation (gradient) checkpointing, which
leaves the update rule unchanged (**PI decision S5**). It is **not
implemented** in the harness today: if the PI selects it, it is
implemented, tested and recorded in every cell (a flag in the run
identity) before the smoke is repeated and before filing, and it then
applies to every cell of the study. A change of optimizer, precision
or batch is not allowed without an amendment. The 35B arithmetic
("only routed experts receive gradients") does not apply.

**Host.** The CPU restore snapshot is the full `state_dict` copy:
9.68 GB (the tied embedding is stored under two keys). KXKM-AI has
62.6 GiB of RAM (about 42 GiB available on 2026-10-25), an Intel
i7-14700KF and 1.8 TB of disk (≈ 700 GB free).

**Runtime (estimate; the smoke measures it).** Per confirmatory cell
at `n_eval` = 130: 1 040 scoring forwards (pre + post, 4 choices), +
160 for OPR, + about 200 forward / backward updates (T_s ≈ 12 000
loss tokens at m = 5), + one 9.7 GB restore. With the transformers
torch fallback for Gated DeltaNet (the `flash-linear-attention` /
`causal-conv1d` kernels are not installed; **PI decision S6**), a
rough planning figure is 1.5-2.5 min per cell, i.e. about 40-70 h
for the 1 620 cells. The smoke (30 cells) and its projection to the
confirmatory workload (E1 §9 method) replace this estimate in the
filed version. If the projection exceeds the available KXKM-AI GPU
time, the PI amends before filing; N is never reduced after filing.

**Kernels (S6).** The run uses one attention implementation for the
whole study (all of smoke, headroom, tuning and confirmatory). Default:
the transformers torch fallback (no new dependency). Installing the
fast kernels would add two new hash-pinned packages (to be approved)
and must happen before the smoke; switching between implementations
mid-study is a deviation.

**Corpus (same items, model-independent).** E1-small uses the **same**
PENDING-HEADROOM fact corpus built on 2026-10-25 (corpus file
`f8379b81…c0a2`, confirmatory partition `546f44c6…cb3f`, tuning
`7aea63a8…b0f7`; manifest `docs/milestones/e1-corpus-manifest-2026-10-25-
pending.json`). Its construction depends on the model only through the
step-3 token bound (question + gold ≤ 256 tokens), counted with the
pinned 35B tokenizer, which is byte-identical to the 4B tokenizer
(§2); with the proposed pin it is therefore the same corpus for both
models in every respect. (With the S1 alternative, the bound would
still be counted with the 35B tokenizer, i.e. it would be a fixed,
model-external length filter; the ≤ 512-token memory bound would then
be re-checked on the 4B tokenizer during the smoke.)

**Headroom (step 9, separate for the 4B).** The 4B base checkpoint is
scored on the TUNING partition only (`HL_STUDY=e1-small`, `--mode headroom`);
confirmatory items are never scored before the first
confirmatory seed. `headroom_rebalance` (E1 §5.1b) is applied to the
4B scores exactly as specified (pooled and per-source 0.80 rule,
REJECT conditions, weights `w_s`). The result is the E1-small FROZEN
corpus and manifest, written to their own directory and hashed into
this document. The 35B study later applies its own step 9 to the same
pending corpus.

**Consequences for the PI (S2).** (i) If a source exceeds 0.80 for
one model and not the other, the two FROZEN corpora differ (different
sources and weights `w_s`); since nothing is pooled across sizes this
does not invalidate either study, but per-source and pooled estimands
then differ between the studies, which any descriptive side-by-side
must state. (ii) A 4B model is expected to be below 0.80 on all three
sources, so the expected E1-small outcome is ACCEPT with no drop.
(iii) The alternative — freezing one corpus for both studies using
the stricter of the two headroom results — would need the 35B
headroom first, which the PI has deferred; it is not proposed.

## §6 Code (harness support for the 4B)

- Study selector: one table of pinned models,
  `scripts/e1_sources.py` `STUDY_MODELS` (`e1` →
  `Qwen/Qwen3.6-35B-A3B` @ `995ad96e…`, `e1-small` →
  `Qwen/Qwen3.5-4B` @ `851bf6e8…`), read by the single
  model-selection spot of `scripts/e1_harness.py` (`MODEL_REPO`,
  `MODEL_REVISION`) according to `HL_STUDY` (default `e1`; an unknown
  value is refused). Every model check of the harness core (per-mode
  refusal, tuning / calibration provenance) and of the PyTorch backend
  (load at the pinned revision) reads the selected pin, so a cell is
  accepted only from the pinned model of its study.
- Registered instrument: for `e1-small` the harness accepts only the
  PyTorch backend with every parameter on one CUDA device (the backend
  records the actual parameter placement; CPU, offloaded or
  multi-device placements are refused) (`STUDY_BACKEND`), for the
  running backend, every tuning / calibration cell it reads (which must
  also carry `study = e1-small`) and the amendment (which records
  `study` and `backend`). The MLX backend is therefore
  refused for E1-small (its weight pins are the 35B files).
- Corpus consumer: tuning, calibration, amendment and confirmatory
  modes refuse a FROZEN manifest whose headroom record does not name
  the study's pinned model and revision, so E1-small can never run on
  a corpus frozen with the 35B headroom (and vice versa once the 35B
  freeze records its model).
- File integrity: `scripts/run_e1_small_cuda.sh` requires the pinned
  snapshot to hold exactly the ten files of §2, checks each against
  `scripts/e1_small_weights.sha256`, checks that PyTorch sees a CUDA
  device and runs offline (`HF_HUB_OFFLINE=1`).
- Provenance: each cell and the headroom file record `study`, model
  and revision (model and revision are part of the run identity).
- Corpus freeze: `scripts/e1_corpus_build.py --freeze --study e1-small`
  accepts headroom scores from the E1-small pin only; the step-3
  tokenizer pin is unchanged.
- Environment on KXKM-AI: `/ai/saisail/e1-small/venv`, CPython 3.12.3,
  installed with `pip --require-hashes --no-deps --only-binary=:all:`
  from `scripts/e1_small_cuda_requirements.txt` (57 packages: torch
  2.11.0, transformers 5.7.0, huggingface-hub 1.13.0, tokenizers
  0.22.2, safetensors 0.7.0, numpy 2.4.4, scipy 1.17.1, pytest 9.0.3,
  CUDA 13 wheels and their dependencies). Every version **and every hash** is taken from
  an existing audited lock (the dream-of-kiki `uv.lock` of this branch
  and the nerve-wml `uv.lock` at `06dd573`, which pins the torch
  stack); no package outside those two baselines was installed.

## §7 Smoke (pre-filing; after the GPU is free)

Smoke = E1 §9 smoke, unchanged: seeds 2000-2002, all 10 profiles
(30 cells), pilot fixtures only (§5.3 of the E1 draft), placeholder
amendment (defaults for the untuned arms), `--smoke-n-eval 50`, never
analysed. It records per-cell wall time, `torch.cuda.max_memory_
allocated`, host RSS, achieved tokens and update counts, as in E1 §9,
and its projection to the 1 620-cell workload goes into the filed
version. The exact command is in the PR description.

## §8 Filing gate (E1-small)

Not filed until: the PI has decided S1-S8 (§9); the 4B smoke has run
and its measurements and projection are written here; the 4B headroom
has run and the E1-small FROZEN corpus and manifest hashes are written
here; the harness code is merged at a pinned commit with passing tests;
the inherited E1 text is pinned (§3).

## §9 Open decisions for the PI

- **S1 Model.** `Qwen/Qwen3.5-4B` @ `851bf6e8…` (proposed; same
  tokenizer and family as the 35B pin) or `Qwen/Qwen3-4B-Instruct-2507`
  @ `cdbee75f…` (literal "Qwen3 4B dense instruct"; different
  tokenizer and architecture). Weights of the proposed pin only were
  downloaded.
- **S2 Headroom / freeze.** E1-small freezes its own copy of the same
  pending corpus with the 4B headroom (proposed), accepting that the two
  FROZEN corpora can differ if a source is dropped for one model only.
- **S3 Order.** E1-small is filed and run before the 35B study; the 35B
  draft is unchanged (a one-line cross-reference can be added by its
  author).
- **S4 Hardware.** KXKM-AI, one RTX 4090, shared with other studies
  (scheduling, not design).
- **S5 Memory fallback.** Activation checkpointing if the smoke does
  not fit (update rule unchanged), decided before filing.
- **S6 Kernels.** Torch fallback for Gated DeltaNet (no new
  dependency, proposed) or the fast kernels (two new hash-pinned
  packages to approve); fixed for the whole study.
- **S7 Inheritance.** By pinned reference to the E1 draft text
  (proposed) or a self-contained copy at filing.
- **S8 Harness tokenizer.** The transformers tokenizer of the pinned
  files (current harness) or the raw `tokenizer.json` (1 of 7 850
  corpus texts differs); best decided together with the 35B draft's
  same item so both studies tokenize alike.
