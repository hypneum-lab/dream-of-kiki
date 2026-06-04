# Wave 3b — Real MLX latent-diffusion substrate (implementation plan)

**Version** : v0.1-draft (2026-05-20)
**Author** : Clement Saillant (L'Electron Rare)
**Status** : Draft for user approval — NO code yet, NO branch, NO PR
**DualVer target** : C-v0.13.0+PARTIAL (MINOR — additive primitive
wiring on a new substrate file ; STABLE deferred to milestone M6)
**Supersedes** : —
**Related** :
- Wave 3a closure proof — `docs/proofs/dream2learn-dr3-separation.md`
- Framework C spec — `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §2.1, §6.2, §12
- Existing substrates — `kiki_oniric/substrates/{mlx_kiki_oniric,esnn_thalamocortical,esnn_norse,micro_kiki,wake_sleep_cl_baseline}.py`
- Substrate factory — `kiki_oniric/substrates/factory.py`
- R1 contract — `harness/storage/run_registry.py`
- Ablation harness — `scripts/ablation_cycle3.py` (real-data path),
  `scripts/ablation_cycle2.py` (synthetic-vs-real banner)
- OSF parent registration — `docs/osf-preregistration-draft.md`
  (OSF DOI `10.17605/OSF.IO/Q6JYN`)
- Amendment template — `docs/osf-amendment-bonferroni-cycle3.md`
- Memory note — `~/.claude/projects/-Users-electron/memory/project_dream2learn_dr3_resolution_2026_05_20.md`

---

## §1 Goal

### 1.1 Concrete definition of "real MLX diffusion bench"

A **real MLX latent-diffusion substrate** in this plan means a new
file `kiki_oniric/substrates/mlx_latent_diffusion.py` that :

1. Loads (or trains, see §2.2) an MLX-native latent-diffusion model
   `D = (E, U, σ)` where `E` is an encoder mapping data samples to
   a latent vector `z ∈ R^d`, `U` is a denoising MLP/UNet, and
   `σ(t)` is a fixed noise schedule. All tensors are `mx.array`,
   all randomness flows through `mx.random.key(seed)` derived keys
   per §2.3.
2. Exposes an `MLXDiffusionAdapter` implementing the
   `SubstrateAdapter` Protocol of `kiki_oniric/substrates/factory.py`
   (`execute_profile(CellRequest) → dict`, `teardown()`).
3. Wires the four dream operations (replay / downscale /
   restructure / recombine) onto the diffusion model with a
   **typed `DreamEpisode` 5-tuple** per profile, emitting a typed
   `WeightDelta` (Canal 1) plus typed `LatentSamples` (Canal 2)
   per Episode close. This is the conformance-attempt path.
4. Registers each cell in the existing `RunRegistry` via the same
   `HARNESS_VERSION` mechanism `scripts/ablation_cycle3.py` already
   uses (line 70), and exposes its `output_hash` via
   `register_output_hash` (the second half of R1 contract, see
   `STATUS.md` "R1 output-hash API landed").

### 1.2 Substrate vs baseline framing — two-track design

The plan deliberately keeps **both** possibilities open until
milestone M2 :

- **Track S (substrate)** — pursue full DR-3 Conformance Criterion :
  signature typing ∧ axiom property tests passing ∧ BLOCKING
  invariants S1/S2/S3/I1 enforced. If achieved, the substrate is
  a fourth DR-3-conformant variant joining MLX-kiki-oniric, E-SNN,
  and micro-kiki — and a genuine framework-C empirical breadth gain.
- **Track B (baseline)** — fall back to a `wake_sleep_cl_baseline`-
  style adapter (see `kiki_oniric/substrates/wake_sleep_cl_baseline.py`)
  exposing only the `evaluate_continual` comparator contract, with
  an explicit "non-conformant" banner mirroring the Paper 2 §7.7
  pattern. This is the safe-ship fallback.

The decision Track-S-vs-Track-B is made **at M2 gate review**, on
the basis of the §3.2 typing dossier and a first-pass attempt at
the property suite. Section §6 of this document lists the risks
that force the fallback.

### 1.3 Relation to Wave 3a (Dream2Learn proof)

Wave 3a closed Dream2Learn (D2L) as category-(a) citation-only by
formal proof in `docs/proofs/dream2learn-dr3-separation.md`. The
proof establishes **what a DR-3-conformant latent-diffusion variant
would need that D2L lacks** — and Wave 3b is precisely the attempt
to build that delta. The required deltas, lifted verbatim from
§3.1 and §3.2 of the proof :

1. **Typed α/β/γ/δ + Canal 1/2/3/4 surface.** D2L exposes 0/8
   typed primitives. Wave 3b targets ≥6/8 (Canal 3 hierarchy_chg
   and Canal 4 attention_prior may stay no-op for the latent
   space if `D`'s geometry remains flat — see §3 design note).
2. **Discrete DreamEpisode unit.** D2L's soft-prompt inner loop
   is unbounded ; Wave 3b wraps each diffusion-driven
   consolidation step in a `(trigger, input_slice, operation_set,
   output_delta, budget)` 5-tuple logged to `RunRegistry`.
3. **Awake/dream role partition.** D2L mutates `f_θ` in-line ;
   Wave 3b maintains `W_awake`, `W_dream`, `W_scratch` as three
   MLX weight-set copies with the `kiki_oniric.dream.swap`
   protocol at DE boundaries.
4. **DE-bounded accountability.** D2L's training is gradient-loss
   stopped ; Wave 3b enforces a per-DE compute budget (K family
   invariant) and logs DE budget exhaustion to the registry.
5. **Mutable hierarchy (Canal 3).** D2L's geometry is frozen ;
   Wave 3b's diffusion model latents are *mutable* via the
   `restructure` op (this is the genuinely novel structural
   delta over D2L — see §3.3).

A conformant Wave 3b variant therefore exists as a *non-vacuous*
research artifact : it would be the first DR-3-conformant
substrate that uses **generative replay via learned latent
geometry** (rather than buffer-based replay as in MLX-kiki-oniric,
spike-based replay as in E-SNN, or LoRA-delta merging as in
micro-kiki). It is **not** a refutation target ; it is a fourth
breadth point along the DR-3 conformance axis. Wave 3a remains
unmodified.

---

## §2 Open decisions (5 blockers)

### 2.1 Blocker D1 — Dataset / corpus

**Question.** Which evaluation grid does the diffusion substrate
plug into ? The existing matrix in `scripts/ablation_cycle3.py`
expects MMLU / HellaSwag / mega_v2 on a Qwen3.5 base. Diffusion is
naturally generative-visual.

**Recommended answer.** Re-use **the CIFAR-100 G4-sexto / Tiny-
ImageNet G4-septimo grid** that already drives `kiki_oniric.eval`
small-CNN benchmarks (STATUS.md confirms 760 cells confirmatory
at N=95). Rationale : (a) the diffusion model has a meaningful
latent encoder for natural images ; (b) the existing fixture and
evaluators are bit-stable under R1 ; (c) class-incremental Split-
CIFAR-100 maps directly onto the framework-C profile axis
(P_min/P_equ/P_max) — same metric stack (M1.a forgetting, M1.b
average accuracy, M3.b retention) ; (d) the eval-matrix.yaml
baselines block already lists CIFAR-10 for Wake-Sleep CL, so the
cross-row comparator is canonical.

**Alternatives.**
- *mega_v2_stratified Q&A* — would unify with the cycle-3
  scale-axis runner but forces a text-domain encoder choice (no
  natural MLX latent-image baseline ; would have to train a
  text-VAE — adds a side-quest).
- *New dataset (e.g. Split-FashionMNIST + diffusion-generated
  augmentations)* — cleanest theoretical fit but requires a fresh
  fixture lock + sha256 + OSF amendment ; cost outweighs benefit.

**Downstream.** CIFAR-100 keeps the OSF amendment minimal
(re-uses pre-registered G4-sexto / G4-septimo §6 §7 row scaffolds),
keeps the harness changes additive (new substrate row in
`SUBSTRATES`, no new SCALES tuple), and keeps R1 within the
existing `tests/reproducibility/` envelope for image-tensor
pipelines.

### 2.2 Blocker D2 — Training corpus size + model size

**Question.** Train `D` from scratch on M5 GrosMac (16 GB) or use
a pretrained MLX checkpoint ? UNet or pure-MLP denoiser ?

**Recommended answer.** **3-layer MLP latent denoiser** (~2 M
params) trained from scratch on Split-CIFAR-100 train shard
restricted to the first 5 tasks (50 classes × 500 images = 25 k
samples), `d_latent = 64`. Reasons : (a) fits comfortably in 16 GB
unified memory at fp32 with batch 256 ; (b) bit-exact reproducible
under MLX seeding (no external pretrained-weight hash to pin) ;
(c) MLP is intentionally weaker than a UNet — this is a *research*
substrate, not a competition entry, so the relevant question is
"does dream consolidation hold under diffusion-class replay",
not "is this SOTA generation".

**Alternatives.**
- *UNet (~10 M params)* trained on Studio M3 Ultra — gives
  better generative quality and a more honest D2L analog but
  doubles M4 wall-clock and forces Studio-only training (M5
  developers cannot iterate locally).
- *Pretrained Stable-Diffusion-tiny MLX port* — fastest path to
  visually convincing samples but pulls an external weight blob
  that needs sha256-pinning and license review ; also kills the
  R1 cross-machine story (no guarantee MLX port reproduces
  byte-for-byte across M5/M3-Ultra/M1-Max).

**Downstream.** Local M5 trainability lets the user iterate at
M2/M3 without Studio queue contention. Studio is reserved for the
M5 full bench run.

### 2.3 Blocker D3 — MLX seeding policy + R1

**Question.** Diffusion training and sampling require many
per-step random draws (noise per timestep, dropout, batch
shuffling). How does this fit the existing R1 contract — which
already fails cross-machine on 3/5 entries per STATUS.md, with
`mx.random.normal(shape, key=mx.random.key(seed))` localised as
the M1 Max-specific divergence (issue
[ml-explore/mlx#3568](https://github.com/ml-explore/mlx/issues/3568)) ?

**Recommended answer.** Adopt a **per-DE key-derivation tree** :

```
root_key       = mx.random.key(cell_seed)
de_key, …      = mx.random.split(root_key, n_episodes)
step_key, …    = mx.random.split(de_key[i], n_diffusion_steps)
noise_t        = mx.random.normal(shape, key=step_key[t])
```

Per the STATUS.md probe, `mx.random.split`-derived keys reproduce
across all three Apple Silicon variants ; the failing pattern is
direct `mx.random.key(seed)` consumption by `mx.random.normal`.
The new substrate MUST therefore consume only **split-derived**
keys, never raw keys, for any operator participating in the R1
hash. The single root key construction is allowed (it is the
seed-deterministic root).

R1 entries to add in `tests/reproducibility/golden_hashes.json` :
`test_r1_diffusion_train_step`, `test_r1_diffusion_dream_replay`,
`test_r1_diffusion_full_de`. All start `status:
"pending_review"` per the 2026-05-10 N2 rebaseline discipline ;
promotion to `accepted` requires the same mlx / numpy pin
hardening listed in `tests/reproducibility/REBASELINE_NOTE.md`.

**Alternatives.**
- *Skip R1 for the diffusion substrate and document it as a
  R1-EXEMPT substrate* — would unblock M3 faster but breaks
  the framework's "Determinism is a contract" working rule
  (CLAUDE.md §Working rules item 1). Rejected.
- *Use numpy RNG for noise instead of MLX RNG* — reproduces but
  forces a numpy↔MLX boundary per step ; benchmarks at ~3-4×
  slowdown in informal local probes. Acceptable fallback only
  if §2.3 recommended path itself fails cross-machine.

**Downstream.** The constraint propagates into §3 (the substrate
file MUST expose a `_derive_step_key(de_id, step)` helper) and
into the M3 deliverable acceptance criterion.

### 2.4 Blocker D4 — DualVer bump

**Question.** MINOR (+0.1.0) or MAJOR (+1.0.0) ? When does the
bump land — M1 PR, M3 PR, or M6 ?

**Recommended answer.** **FC MINOR : C-v0.12.0 → C-v0.13.0**, EC
state **+PARTIAL until M6 bench closure**, then **+STABLE** iff
all four conditions of framework-C §12.3 STABLE definition hold
(coverage of stratified matrix, no axiom violation, no orphan
result, conformance criterion passed *or* explicit non-conformant
banner).

Per §12.2 :
- FC-MINOR is "addition of new axiom / new optional primitive /
  new derived constraint". Wave 3b adds a new *substrate* but the
  primitive *signatures* are unchanged ; the Canal-3 / Canal-4
  no-op behaviour for a flat latent space is a *derived
  constraint* on the substrate, not on the framework — strictly
  speaking this is an FC-PATCH. **However**, M6 will likely
  produce empirical evidence bearing on DR-4 ("richer ops yield
  richer consolidation") under a third substrate class
  (generative-replay), which justifies MINOR over PATCH.
- MAJOR is rejected : no axiom signature changes, no primitive
  signature changes.

**Bump timing.** Two-phase :
- M1 PR : bump CHANGELOG draft entry `[C-v0.13.0+PARTIAL] —
  Wave 3b substrate skeleton, pending M6 closure`, but the
  effective version in code stays at C-v0.12.0+PARTIAL until M3.
- M3 PR : flip `HARNESS_VERSION` in
  `scripts/ablation_cycle3.py` to `"C-v0.13.0+PARTIAL"`, update
  `STATUS.md` "Version" line, run the stratified compat suite
  per §12.4 step 2.
- M6 PR : conditional flip to `+STABLE` if §12.3 STABLE
  conditions all hold ; otherwise stay `+PARTIAL` and document
  the deferred cell.

**Alternatives.** MINOR at M1 (premature — no shipped substrate
yet to justify a public bump). PATCH only (under-counts the DR-4
evidential weight of a third substrate class).

**Downstream.** Single CHANGELOG section to maintain across
6 milestone PRs ; the deferred-flip pattern (M1 draft → M3 flip)
matches the C3.10 precedent (cycle-3 launch bump).

### 2.5 Blocker D5 — OSF amendment

**Question.** Does Wave 3b need a new OSF pre-registration or an
amendment to the existing Q6JYN parent ? Who approves ? Timeline ?

**Recommended answer.** **Amendment to Q6JYN**, filed as an
Open-Ended Registration linked to the parent (same pattern as
`docs/osf-amendment-bonferroni-cycle3.md` which produced
`10.17605/OSF.IO/TPM5S` 2026-04-21). Amendment scope :

1. Add `mlx_latent_diffusion` to the `SUBSTRATES` enumeration in
   §2 of the parent pre-reg (axis materialization).
2. Add three R1 entries (per §2.3) to the reproducibility
   contract enumeration in §5.
3. Cite the §3 conformance dossier (whether Track S or Track B
   wins at M2) as the verbatim statement of what is empirically
   tested.

Approver chain : PI (Clement Saillant) drafts, ship-critic
agent reviews (per `~/.claude/projects/-Users-electron/memory/feedback_critic_before_ship.md`),
OSF mint via DataCite (auto). Timeline : draft at M1 ship,
filing window opens at M3 ship (the bump to
C-v0.13.0+PARTIAL is the trigger), publish at M5 (before any
bench numbers leave the local registry).

**Alternatives.** New pre-registration (justified only if a
genuinely new hypothesis grid is introduced ; Wave 3b is a
substrate-axis extension, not a new hypothesis grid). Skip
amendment (rejected — STATUS.md and CHANGELOG.md treat OSF
pre-reg as a hard contract for any new empirical result row).

**Downstream.** The draft text lives in §5 of this document
(below) ; the M1 ship deliverable includes filing-ready text.

---

## §3 Architecture sketch

### 3.1 Module layout (files to create, ~200 LoC granularity each)

```
kiki_oniric/substrates/
  mlx_latent_diffusion.py          # ~250 LoC — adapter + DE wiring
  _diffusion/
    __init__.py                    # ~10 LoC
    model.py                       # ~180 LoC — encoder E, MLP denoiser U, sched σ
    sampler.py                     # ~120 LoC — reverse process, key-derivation
    trainer.py                     # ~200 LoC — train loop, R1-clean RNG
    primitives_wiring.py           # ~150 LoC — α/β/γ/δ + Canal 1/2/3/4 typed handlers

harness/
  diffusion_eval/                  # new package, sibling of real_benchmarks
    __init__.py
    cifar100_split_loader.py       # ~120 LoC — Split-CIFAR-100 5-task loader
    diffusion_metrics.py           # ~80 LoC — M1.a / M1.b / M3.b adapters

scripts/
  ablation_cycle3_diffusion.py     # ~200 LoC — fork of ablation_cycle3, new SUBSTRATES axis member
  train_diffusion_base.py          # ~150 LoC — one-shot M5-local base training driver

tests/
  unit/test_mlx_latent_diffusion_adapter.py     # ~120 LoC
  unit/test_diffusion_sampler_keys.py           # ~90 LoC — R1 key-derivation contract
  conformance/axioms/test_dr3_diffusion_substrate.py  # ~100 LoC — Track S only
  conformance/axioms/test_dr0_diffusion_de_budget.py  # ~70 LoC
  conformance/axioms/test_dr1_diffusion_finite.py     # ~70 LoC
  reproducibility/test_r1_diffusion.py          # ~100 LoC — 3 new R1 entries
  integration/test_ablation_cycle3_diffusion.py # ~80 LoC

docs/
  proofs/
    dr3-diffusion-substrate-evidence.md         # ~400 lines, Track S only
  papers/paper2/
    results.md                                  # +§7.9 section (~200 lines)
```

Total new code budget : ≈ 1700 LoC across 14 files, with the
heaviest single file kept under 300 LoC per CLAUDE.md hygiene.

### 3.2 DR-3 conformance design — typing table

For each of the 8 primitives (signatures in `kiki_oniric/core/primitives.py`),
this is the planned Wave 3b correspondent. Mark `T` (typed,
DR-3 condition 1 satisfied) vs `N` (no-op, documented).

| Primitive | Signature (core/primitives.py) | Wave 3b correspondent | Status |
|-----------|--------------------------------|-----------------------|--------|
| α `AlphaStreamProtocol` | `append_trace(trace)` / `iter_traces()` | Awake encoder activations on real CIFAR batches ; logged as `mx.array` traces | T |
| β `BetaBufferProtocol` | `append_record` / `fetch_unconsumed` / `mark_consumed` | Curated `(z, label)` latent records sampled from α stream | T |
| γ `GammaSnapshotProtocol` | `get_checkpoint_path` / `get_checkpoint_sha256` | Awake-classifier `f_θ` MLX checkpoint at DE start, sha256 pinned | T |
| δ `DeltaLatentsProtocol` | `snapshot(...)` / `get_recent(...)` | Diffusion-denoiser `U` latent-layer activation snapshot at DE start | T |
| Canal 1 `WeightDeltaChannel` | `apply(delta)` | Merge of dream-phase `U_dream → U_awake` weight delta under invariant S1 | T |
| Canal 2 `LatentSampleChannel` | `enqueue` / `dequeue` | Reverse-process samples `x̂ ~ D(·|z)` consumed by classifier training | T |
| Canal 3 `HierarchyChangeChannel` | `apply_diff(diff)` | Diffusion latent-geometry mutation via `restructure` op | T (novel) |
| Canal 4 `AttentionPriorChannel` | `set_prior` / `get_prior` | No-op (flat latent space, no attention prior in MLP denoiser) | N |

Score : **7/8 typed, 1/8 documented no-op**. The Canal 4 no-op is
explicitly allowed by framework-C §6.2 (the criterion requires
*signature typing*, satisfied by the protocol implementation
even when the operational return is empty). This contrasts with
D2L's 0/8 typed (Wave 3a proof §3.1) — Wave 3b clears the
conformance bar D2L could not.

### 3.3 Substrate factory integration

Add a fourth member to `SubstrateName` and `SUBSTRATE_NAMES` in
`kiki_oniric/substrates/factory.py` :

```python
SubstrateName = Literal[
    "mlx_kiki_oniric",
    "esnn_thalamocortical",
    "micro_kiki",
    "mlx_latent_diffusion",     # NEW (Wave 3b)
]
```

Add the corresponding adapter class `MLXLatentDiffusionAdapter` in
`mlx_latent_diffusion.py`, implementing `SubstrateAdapter`. Add
lazy-import dispatch in the factory's getter (the same pattern
`ESNNAdapter` uses). Update `scripts/ablation_cycle3.py` `SUBSTRATES`
tuple to include `"mlx_latent_diffusion"` — this is the empirical-
axis touchpoint that triggers a DualVer EC re-evaluation per §12.4.

### 3.4 Tests required (per DR-3 criterion)

Three test families, mapped to the Conformance Criterion :

1. **Signature typing (condition 1)** — Python `runtime_checkable`
   Protocol conformance assertions in
   `test_mlx_latent_diffusion_adapter.py`. Fails if any of the
   7 typed primitives is not implemented with the correct
   signature.
2. **Axiom property tests (condition 2)** — DR-0 budget
   accountability, DR-1 finite invariants, DR-2 compositionality
   under the v0.2 weakened precondition. Each test file ≤ 100 LoC,
   uses Hypothesis property-based testing as the existing
   `tests/conformance/axioms/` corpus does.
3. **BLOCKING invariants enforceable (condition 3)** — S1 (no
   silent weight drift outside swap), S2 (finite gradients, no
   NaN/Inf in `z` or `x̂`), S3 (per-DE compute budget bounded),
   I1 (episodic record conservation). Each guard cites its
   invariant ID in the assert message per CLAUDE.md §Working
   rules item 2.

Coverage target : ≥ 90 % line coverage on the new substrate (the
project-wide gate is 91.31 % per STATUS.md and the
`pyproject.toml` `fail_under` is 40 with 51 pp headroom — the
new substrate must not drop the overall coverage).

---

## §4 Jalons (6 milestones, ≤ 1 week each, each = independent PR)

### M1 — Design dossier + DR-3 argument (1 week, ~ 600 LoC docs)
**Deliverable** : This plan finalized + `docs/proofs/dr3-diffusion-substrate-evidence.md` v0.1-draft (skeleton, structural angle filled, behavioural angle TODO) + `docs/osf-amendment-wave3b.md` draft. **Acceptance** : ship-critic GO on plan + proof skeleton ; user approves the 5 decisions in §2. **PR scope** : docs only, no code.

### M2 — Substrate skeleton + unit tests (1 week, ~ 500 LoC code, ~ 300 LoC tests)
**Deliverable** : `kiki_oniric/substrates/mlx_latent_diffusion.py` + `_diffusion/model.py` skeleton (E, U, σ classes with `mx.array` signatures, no training yet) + unit tests for adapter Protocol conformance + factory wiring. **Acceptance** : `uv run pytest tests/unit/test_mlx_latent_diffusion_adapter.py` green ; `runtime_checkable` Protocol assertions pass ; coverage on new module ≥ 90 %. **Track S/B decision gate** : at M2 review, user decides Track S (continue to M3 full conformance) or Track B (rebrand as baseline, follow Wake-Sleep pattern, skip §3.4 condition-2/3 tests).

### M3 — Training loop + R1 golden (1 week, ~ 550 LoC code, ~ 200 LoC tests)
**Deliverable** : `_diffusion/trainer.py` + `_diffusion/sampler.py` with the §2.3 key-derivation tree + `scripts/train_diffusion_base.py` + 3 new R1 entries (`test_r1_diffusion_train_step`, `test_r1_diffusion_dream_replay`, `test_r1_diffusion_full_de`) in `tests/reproducibility/golden_hashes.json` with `status: "pending_review"`. **Acceptance** : within-machine R1 9/9 + 3/3 new entries green on M5 GrosMac ; `HARNESS_VERSION` flipped to `"C-v0.13.0+PARTIAL"` ; `STATUS.md` updated ; `r1-nightly.yml` runs the new entries on macos-14 runner (cross-machine R1 expected to PARTIALLY fail on M1 Max per the issue #3568 known limitation — document, do not block).

### M4 — ablation_cycle3 integration + smoke run (1 week, ~ 300 LoC code, ~ 150 LoC tests)
**Deliverable** : `scripts/ablation_cycle3_diffusion.py` + add `"mlx_latent_diffusion"` to `SUBSTRATES` tuple + `harness/diffusion_eval/cifar100_split_loader.py` + smoke test (1 substrate × 3 profiles × 3 seeds = 9 cells, ~ 15 min wall on M5). **Acceptance** : 9/9 cells registered in `RunRegistry` with non-empty output hashes ; `ablation_cycle2.py`-style synthetic-vs-real banner correctly marks the run as real ; conformance test suite passes for Track S (or non-conformance banner present for Track B).

### M5 — Full bench run (1 week, mostly Studio wall-clock)
**Deliverable** : Full Cartesian product : 1 substrate × 3 profiles × N=30 seeds × 5 CIFAR-100 task-splits = 450 cells on Studio M3 Ultra (est. wall ~ 6-8 h). Results dump in `docs/milestones/wave3b-bench-2026-MM-DD.{json,md}`. **Acceptance** : 450/450 cells included (or excluded with rule citation per the underperforming-baseline rule, Paper 2 §6) ; H1/H2/H4 verdicts reported per profile ; cross-substrate consistency check vs MLX-kiki-oniric + E-SNN per the `ablation_cycle2.py` pattern.

### M6 — Paper 2 §7.9 + DualVer bump + ship-critic (1 week, ~ 400 LoC docs)
**Deliverable** : `docs/papers/paper2/results.md` §7.9 (new section) drafted in EN, FR mirror under `docs/papers/paper2-fr/` per CLAUDE.md EN→FR propagation rule ; `docs/proofs/dr3-diffusion-substrate-evidence.md` v1.0 (behavioural angle now backed by M5 numbers) ; CHANGELOG bump to `[C-v0.13.0+STABLE]` *iff* §12.3 STABLE conditions all hold (no orphan, no axiom violation, conformance pass), else stay `+PARTIAL` and document the deferred cell ; ship-critic agent review per `feedback_critic_before_ship.md` (mandatory). **Acceptance** : ship-critic GO ; OSF amendment Q6JYN-W3B published with DataCite DOI ; ICANN-grade external reproducibility (anyone with the repo + R1 hashes can re-run M5).

---

## §5 OSF amendment draft (ready to send)

> **Title** : Wave 3b — MLX latent-diffusion substrate, additive
> substrate-axis extension and three new R1 reproducibility
> entries.
>
> **Amendment summary**. The dreamOfkiki parent pre-registration
> ([OSF Q6JYN](https://osf.io/q6jyn), DataCite DOI
> `10.17605/OSF.IO/Q6JYN`, 2026-04-19) enumerates three
> framework-C substrate instantiations (`mlx_kiki_oniric`,
> `esnn_thalamocortical`, `micro_kiki`). This amendment adds a
> fourth substrate, `mlx_latent_diffusion`, implementing a
> latent-diffusion generative-replay variant of the framework-C
> 8-primitive contract. The new substrate exposes 7/8 typed
> primitives (α, β, γ, δ, Canals 1, 2, 3) and one documented
> no-op (Canal 4 attention-prior, justified by the flat latent
> geometry of the MLP denoiser). Conformance with the DR-3
> Conformance Criterion (signature typing + axiom property tests
> + BLOCKING invariants S1/S2/S3/I1 enforceable) is the M2 gate
> deliverable ; if the gate fails, the substrate is re-classified
> as a published-reference baseline mirroring the Paper 2 §7.7
> Wake-Sleep CL pattern, with an explicit non-conformant banner.
> The empirical grid is Split-CIFAR-100 5-tasks-buffer-500
> (same fixture as G4-sexto confirmatory N=95, see
> `docs/milestones/g4-sexto-confirmatory-n95-results.md`),
> evaluated across the three framework-C profiles
> (P\_min, P\_equ, P\_max) at N=30 seeds, with hypotheses H1, H2,
> H4 reported per profile per the parent pre-reg §2.
>
> **Reproducibility extension**. Three new R1 entries are added
> to `tests/reproducibility/golden_hashes.json` :
> `test_r1_diffusion_train_step`, `test_r1_diffusion_dream_replay`,
> `test_r1_diffusion_full_de`. All entries open at `status:
> "pending_review"` per the 2026-05-10 N2 rebaseline discipline
> documented in `tests/reproducibility/REBASELINE_NOTE.md`. The
> diffusion sampler consumes only `mx.random.split`-derived keys
> per the M1 Max divergence workaround filed upstream as
> [ml-explore/mlx#3568](https://github.com/ml-explore/mlx/issues/3568).
> Cross-machine R1 across {M5 GrosMac, M3 Ultra Studio, M1 Max
> macM1} is the M3 acceptance target ; partial failure on M1 Max
> is documented but not a blocker per the same pattern as the
> 2026-05-20 R1 cross-machine probe in `STATUS.md`. No hypothesis
> in the parent pre-reg is modified. DualVer transition :
> C-v0.12.0+PARTIAL → C-v0.13.0+PARTIAL at M3 ; conditional
> promotion to +STABLE at M6 per framework-C §12.3.

---

## §6 Risks + mitigations

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| R1 | MLX seed determinism fails cross-machine on new diffusion entries (recurrence of `mx.random.normal` + raw-key bug on M1 Max) | **HIGH** | §2.3 split-key-only discipline ; M3 acceptance gate requires `r1-nightly.yml` green on macos-14 ; if M1 Max-specific divergence appears, document under issue #25 follow-up and keep the M5/M3-Ultra pair as the canonical cross-machine R1 pair (same posture as the 2026-05-20 probe). |
| R2 | M5 GrosMac 16 GB OOM during training or inference at batch 256 | **MEDIUM** | Reduce batch to 128 + grad-accum × 2 ; if still OOM, fall back to `d_latent=32` (halves model size). Hard fallback : train on Studio, distribute checkpoint sha256-pinned for local M5 eval-only. |
| R3 | DR-3 conformance not achievable (some primitive can't be typed without contortion ; or property tests fail under §3.4 condition 2) | **MEDIUM** | Track S/B decision gate at M2 review (§4 M2). Track B fallback path is fully specified : same file structure, same factory wiring, but skip §3.4 condition-2/3 tests and rebrand the adapter as a baseline mirroring `wake_sleep_cl_baseline.py`. Paper 2 §7.9 then becomes a baseline-row contribution, not a conformant-substrate contribution. |
| R4 | Ship-critic DO-NOT-SHIP at M6 (per `feedback_critic_before_ship.md` 3 validations precedent) | **MEDIUM** | Schedule critic review *before* the DualVer flip and *before* OSF amendment filing ; budget +2 days in M6 for the critic-found-issue fix cycle. Treat the critic verdict as binding. |
| R5 | Diffusion model fails to learn anything useful on the M5-scale training corpus (D collapse, mode collapse, or trivial-denoiser baseline) | **LOW** | M3 acceptance includes a sanity check : reconstruction MSE on a held-out 500-sample shard ≤ 0.5 (rough generative-quality floor). If failed, escalate to UNet on Studio per §2.2 alternative (adds ~3 days to M3). |
| R6 | OSF amendment gets rejected or delayed at filing (parent Q6JYN is a closed registration ; amendment must be Open-Ended Registration per the Bonferroni amendment precedent) | **LOW** | Mirror the exact filing procedure used for `10.17605/OSF.IO/TPM5S` (Bonferroni amendment) ; pre-draft the DataCite metadata block ; ETA 48 h between filing and DOI mint per the 2026-04-21 precedent. |
| R7 | Confirmation bias on the conformance proof — the proof author wants Track S to succeed | **LOW** | Mandate the M2 gate review uses a different reviewer than the M2 implementation author. The two-angle structure of Wave 3a's D2L proof (structural ∧ behavioural, either sufficient) is a good template for the symmetric *positive* case (Wave 3b conformance) : require both angles to GREEN before flipping the Track S/B switch. |

---

## §7 Decision matrix (one-screen scan)

| Decision (§2) | Recommended | Alternatives | M1 blocker ? |
|---|---|---|---|
| **D1 Dataset** | Split-CIFAR-100 5-tasks (re-use G4-sexto fixture) | mega_v2 Q&A ; new Split-FMNIST + diffusion-aug | YES — locks §3.1 file paths and §5 OSF text |
| **D2 Model size** | 3-layer MLP denoiser, d_latent=64, 25k samples train, M5-local | UNet 10 M params on Studio ; Stable-Diffusion-tiny port | YES — locks §3.1 module budget and §4 M3 wall-clock |
| **D3 RNG / R1** | `mx.random.split`-only key tree, 3 new R1 entries `pending_review` | R1-EXEMPT substrate ; numpy RNG side-channel | YES — locks §3 module sampler.py contract |
| **D4 DualVer** | FC MINOR C-v0.12.0 → C-v0.13.0, EC PARTIAL→STABLE at M6 only | MAJOR ; PATCH only ; flip at M1 instead of M3 | NO (M3 blocker, not M1) — M1 PR ships CHANGELOG draft entry only |
| **D5 OSF** | Amendment to Q6JYN, Open-Ended Registration, filed at M3, published at M5 | New pre-registration ; skip amendment | NO (M3 blocker, not M1) — M1 PR ships filing-ready draft text |

**M1 approval = approve D1 + D2 + D3 + the recommended M1 deliverable scope.** D4 and D5 are M3 blockers, surfaced now so they don't ambush the schedule. Track S vs Track B is the M2 gate, deferred to actual implementation evidence.

---

**End of plan.**

Next step after user review : if approved, open milestone-M1 PR per `superpowers:writing-plans` skill discipline, scope = docs only.
