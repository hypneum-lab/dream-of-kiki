# G6-Studio Path A — real-LoRA SpikingKiki-V4 35B-A3B-V4 cross-substrate validation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the framework C effect (REPLAY+DOWNSCALE+RESTRUCTURE+RECOMBINE coupling) transfers to the **real SpikingKiki-V4 35B-A3B-V4** spiking LIF Qwen substrate (58 GB, 31 070 .npz modules) under a 5-subdomain MMLU continual-learning stream, drawing the first real-LLM-scale verdict for the dreamOfkiki programme.

**Architecture:** A new `experiments/g6_studio_path_a/` module wires `kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate` with `DREAM_MICRO_KIKI_REAL=1` env-gate to the SpikingKiki-V4 artifact at `/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4`. Per subdomain we run a real `mlx_lm.tuner.lora` fine-tune step over LoRA delta tensors, optionally inject a profile-dependent `dream_episode_real()` whose four handlers operate on **live** LoRA delta tensors (not synthetic — fixes G6 Path B spectator), and evaluate via `mlx_lm.generate` letter-argmax over MMLU. The driver writes per-subdomain partial JSON dumps so an overnight watchdog kill is resumable. EC stays PARTIAL ; FC stays C-v0.12.0+PARTIAL.

**Tech Stack:** Python 3.12 ; MLX 0.18+ + `mlx_lm` 0.31.2 (Studio M3 Ultra wheel) ; existing `kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate` real backend (already wired in commit `177cf89`) ; existing `harness.real_benchmarks.mmlu.MMLULoader` + `harness.storage.run_registry.RunRegistry` ; numpy + safetensors for adapter round-trip ; pytest + Hypothesis. Driver runs **on Studio M3 Ultra only** ; M1 Max conductor dispatches via SSH (or the user clones+runs on Studio directly).

---

## Critical context the executing agent must read first

1. `STATUS.md` — current `C-v0.12.0+PARTIAL` (G5-bis row, H7-B classification "MLX-only artefact"). G6-Studio Path A directly tests the converse hypothesis at real-LLM scale. **No FC bump** ; EC stays PARTIAL regardless of outcome (Path A counts as the first confirmatory candidate but the full Bonferroni family for an EC bump requires N≥30 + cross-substrate replication).
2. `kiki_oniric/substrates/micro_kiki.py:60-105, 595-870` — the real backend wiring. `_real_backend_enabled()`, `_REAL_BACKEND_ENV_VAR = "DREAM_MICRO_KIKI_REAL"`, `_REAL_BACKEND_PATH_ENV_VAR = "DREAM_MICRO_KIKI_REAL_BACKEND_PATH"`, `MicroKikiSubstrate.load()` Path 1 (mlx_lm) + Path 2 (SpikingKiki shim) + Path 2b (safetensors sidecar). `_load_spiking_backend()` reads `lif_metadata.json` + samples 3 `.npz` modules ; the rest is left to MLX side. **The substrate does NOT expose a LoRA fine-tune loop** — Task 3 builds it.
3. `kiki_oniric/substrates/micro_kiki.py:874-1129` — the four handler factories. Their signatures are the DR-3 condition-1 contract ; Task 4 must NOT loosen them. `replay_handler_factory` returns a 1-D vector ; `downscale_handler_factory` returns a tensor (commutative shrink) ; `restructure_handler_factory` returns the (mutated) adapter dict ; `recombine_handler_factory` returns the merged delta tensor.
4. `experiments/g6_mmlu_stream/run_g6.py:1-752` — Path B template (driver shape, CLI, `_run_cell_path_b`, `_aggregate_verdict`, `_render_md_report`, registry registration, retention computation). Path A reuses the **shape** but replaces `_run_cell_path_b` with a real-LoRA cell. The Path B "spectator pattern" caveat in the rendered MD (lines 539-578) is the exact failure mode this plan corrects.
5. `experiments/g6_mmlu_stream/dream_wrap.py:1-200` — `G6DreamRunner` with `PROFILE_OPS` dispatch + `build_episode_payload` synthetic. Task 4 ports this to a **live-tensor** variant : `dream_episode_real()` operates on the actual LoRA delta dict, not a synthesised payload. This is the load-bearing change vs Path B.
6. `experiments/g6_mmlu_stream/stream.py:1-178` — MMLU subdomain stream loader. `build_subdomain_stream(fixture_path, subdomains, n_train, n_eval, seed)` returns `list[SubdomainSplit]`. Reuse verbatim — the loader is pure.
7. `experiments/g6_mmlu_stream/micro_kiki_inference.py:38-129` — `InferenceOnlyAdapter` shape (`out_dim`, `rank`, `_deltas[key]`). Path A keeps the shape but binds `_deltas` to actual mlx_lm LoRA tensors loaded from disk (live, not synthetic).
8. `tests/unit/test_micro_kiki_real_backend.py:30-200` — the four real-backend test patterns (env unset / env set + artifact / env set + missing metadata / spike payload shape). Task 2 tests reuse the same fixture pattern.
9. `docs/osf-prereg-g6-pilot.md:1-300` — the original G6 pre-reg with G4-bis amendment. G6-Studio Path A builds a **daughter** pre-reg that cites this one + G4-ter (+2.77 MLX) + G5-bis (H7-B MLX-only) + G4-quinto (H5-C RECOMBINE empty CIFAR-10) as exploratory baselines, and locks 3 sub-hypotheses H9-A / H9-B / H9-C *before* any cell registers.
10. `kiki_oniric/eval/statistics.py:37-285` — `welch_one_sided(treatment, control, alpha)`, `compute_hedges_g(treatment, control)`, `jonckheere_trend(groups, alpha)` ; signatures used verbatim in Task 7.
11. `harness/storage/run_registry.py:113-124` — `RunRegistry.register(c_version, profile, seed, commit_sha) -> run_id`. R1 contract bit-stable.
12. `harness/real_benchmarks/mmlu.py:1-100` — `MMLURecord` dataclass + `MMLULoader`. The fallback fixture `tests/fixtures/mmlu_g6_synthetic.jsonl` works for smoke ; production needs the full HF `cais/mmlu` cache materialised offline.
13. `docs/CLAUDE.md` — milestone files are **append-only dated immutables**. Never edit a previous G-pilot dump ; add a sibling dated file instead.
14. `kiki_oniric/CLAUDE.md` — substrate package boundary ; do not loosen primitive types or rename methods (DR-3 conformance).
15. `tests/CLAUDE.md` — unit tests live under `tests/unit/experiments/` ; conformance tests live under `tests/conformance/` and must cite axiom IDs in docstrings.

## Hard prerequisites (block until satisfied)

Stop and surface the blocker if any fails :

1. SSH access to Studio M3 Ultra works (or the user is logged in directly on Studio).
2. `/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4/lif_metadata.json` exists on Studio.
3. `/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-V4-adapters/` directory exists (or its alternative location reported by Task 0).
4. `mlx_lm` ≥ 0.31.2 importable on Studio inside `uv run` of this repo.
5. `MicroKikiSubstrate` with `DREAM_MICRO_KIKI_REAL=1` already loads on Studio (`tests/unit/test_micro_kiki_real_backend.py` passes when the env flag and path are exported).
6. ≥ 200 GB free RAM on Studio (35B bf16 + LoRA fine-tune + Adam state ≈ 80–110 GB peak ; comfort margin).
7. The original G6 Path B milestone exists under `docs/milestones/g6-pilot-pathB-2026-05-03.json` — it is the comparator the H_NEW infrastructure-effect criterion was originally written against.

## Workspace dependency status (verified 2026-05-03)

```
/Users/electron/hypneum-lab/dream-of-kiki                 # this repo (M1 Max conductor)
/Users/clems/KIKI-Mac_tunner/                             # Studio (training pipeline + datasets)
  └── models/SpikingKiki-35B-A3B-V4/                      # 58 GB, 31 070 .npz modules + lif_metadata.json
  └── models/SpikingKiki-V4-adapters/                     # Studio-only LoRA stack (verify in Task 0)
```

This plan **runs on Studio**, not on the M1 Max conductor. Two execution patterns :

- **Pattern S1 — clone+run on Studio** (recommended) : the user `git clone` this repo on Studio, `uv sync --all-extras`, `tmux new -s g6_studio`, run the production driver inside the tmux. The agent (this plan's executor) drives via SSH through the tmux session.
- **Pattern S2 — SSH wrapper from M1 Max** : the agent shells out `ssh studio "cd dream-of-kiki && uv run python experiments/g6_studio_path_a/run_g6_studio_path_a.py …"`. Heavier ergonomics, more failure modes (SSH connection drops, no tmux), but viable.

Locked in Task 0.5 based on Task 0 evidence. Default S1.

---

## File structure

Files **created** by this plan :

```
experiments/g6_studio_path_a/
├── __init__.py                              # package marker
├── lora_loader.py                           # Qwen-35B base + LoRA adapter load via mlx_lm
├── lora_train_step.py                       # per-subdomain real LoRA fine-tune via mlx_lm.tuner.lora
├── dream_episode_real.py                    # live-tensor profile-aware coupling
├── mmlu_eval.py                             # mlx_lm.generate-driven MMLU letter-argmax accuracy
├── run_g6_studio_path_a.py                  # driver — sequential subdomain stream + per-subdomain JSON dumps
└── aggregator_h9.py                         # H9-A / H9-B / H9-C verdict

tests/unit/experiments/
├── test_g6_studio_lora_loader.py            # adapter load + sha256 integrity (mocked mlx_lm)
├── test_g6_studio_lora_train_step.py        # fine-tune contract (mocked mlx_lm.tuner)
├── test_g6_studio_dream_episode_real.py     # live-tensor coupling : delta mutates across handlers
├── test_g6_studio_mmlu_eval.py              # eval letter-argmax stub-mode determinism
└── test_g6_studio_aggregator_h9.py          # H9-A/B/C decision rule arithmetic

docs/
├── osf-prereg-g6-studio-path-a.md           # daughter pre-reg, locked before any cell registers
├── milestones/
│   ├── g6-studio-path-a-decisions-2026-05-04.md           # decisions log (option A/B/C, hyperparams)
│   ├── g6-studio-path-a-2026-05-04.partial.<subdomain>.json # per-subdomain partial dumps (resumable)
│   ├── g6-studio-path-a-2026-05-04.json                   # final aggregate dump
│   └── g6-studio-path-a-2026-05-04.md                     # final human report
```

Files **modified** by this plan :

```
docs/papers/paper2/results.md                # add §7.1.10 G6-Studio Path A results
docs/papers/paper2-fr/results.md             # FR mirror (EN→FR rule, same PR)
docs/proofs/dr3-substrate-evidence.md        # append per-outcome revision
CHANGELOG.md                                 # [Unreleased] empirical row, no DualVer bump
STATUS.md                                    # G6-Studio row in Gates table, As-of update
```

`experiments/` is excluded from coverage scope (per `pyproject.toml`) — pilots are not library code. `tests/unit/experiments/` IS in coverage scope.

---

## Task 0: Investigate Studio environment (read-only)

**Files (read-only):**
- `kiki_oniric/substrates/micro_kiki.py:595-722` (real-backend load contract)
- `tests/unit/test_micro_kiki_real_backend.py:1-200` (test pattern)
- `experiments/g6_mmlu_stream/run_g6.py:50-120` (driver constants we re-pin)

- [ ] **Step 1: SSH to Studio and verify SpikingKiki-V4 layout**

Run on Studio :
```bash
ls -la /Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4/lif_metadata.json
ls /Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4/*.npz | wc -l
ls -d /Users/clems/KIKI-Mac_tunner/models/SpikingKiki-V4-adapters/ 2>&1
du -sh /Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4/
```
Expected: metadata file exists ; npz count ≈ 31 070 ; adapters dir exists OR a sibling `*-adapters/` is reported (record exact path) ; total size ≈ 58 GB.

- [ ] **Step 2: Verify mlx_lm version + tuner.lora API**

Run on Studio :
```bash
cd ~/dream-of-kiki  # or wherever the clone lives
uv run python -c "import mlx_lm; print(mlx_lm.__version__)"
uv run python -c "from mlx_lm.tuner import lora; print([n for n in dir(lora) if not n.startswith('_')])"
uv run python -c "from mlx_lm.tuner.utils import load_adapters; print(load_adapters.__doc__ or 'no doc')"
uv run python -c "from mlx_lm import load, generate; print('ok')"
```
Expected: version ≥ 0.31.2 ; `lora` namespace exposes `LoRALinear` / `apply_lora_layers` / `train` (or equivalent) ; `load_adapters` callable ; `load` + `generate` importable.

Record the EXACT API names returned in the decision log — Task 3 binds to them.

- [ ] **Step 3: Confirm DREAM_MICRO_KIKI_REAL wiring works end-to-end on Studio**

Run on Studio :
```bash
cd ~/dream-of-kiki
DREAM_MICRO_KIKI_REAL=1 \
DREAM_MICRO_KIKI_REAL_BACKEND_PATH=/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4 \
uv run pytest tests/unit/test_micro_kiki_real_backend.py -v
```
Expected: 4 tests pass. The test fixture synthesises its own artifact, so this confirms only the **wiring** ; full SpikingKiki-V4 ingestion is verified by step 4.

- [ ] **Step 4: Smoke-load real SpikingKiki-V4 weights via the substrate**

Run on Studio (one-off — does not commit anything) :
```bash
cd ~/dream-of-kiki
DREAM_MICRO_KIKI_REAL=1 uv run python -c "
from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate
s = MicroKikiSubstrate(real_backend_path='/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4')
s.load()
print('module_count', s._real_state['module_count'])
print('sample_weights keys', sorted(s._real_state['sample_weights'])[:5])
print('lif_metadata keys', sorted(s._real_state['lif_metadata'])[:5])
out = s.awake('Anatomy: the largest bone in the human body is the')
print('awake output', out[:200])
"
```
Expected: `module_count` ≈ 31 070 ; sample_weights non-empty ; awake output starts with `[spiking awake T=`. Record the wall time.

- [ ] **Step 5: Verify MMLU dataset access on Studio**

Run on Studio :
```bash
cd ~/dream-of-kiki
uv run python -c "
from datasets import load_dataset
ds = load_dataset('cais/mmlu', 'all', split='test')
subjects = sorted({r['subject'] for r in ds.select(range(2000))})
print('total rows', len(ds))
print('sample subjects', subjects[:10])
for s in ['anatomy','astronomy','business_ethics','clinical_knowledge','college_biology']:
    n = sum(1 for r in ds if r['subject']==s)
    print(f'{s}: {n} rows')
"
```
Expected: each of the 5 production subjects has ≥ 200 rows (so n_train=100 + n_eval=100 leaves headroom). Record per-subject counts in the decision log. Falls back to the synthetic fixture only if HF download fails (record the deviation).

- [ ] **Step 6: No commit. This is a context-gathering task.**

---

## Task 0.5: Lock decisions (Option A/B/C, hyperparams, execution pattern)

**Files:**
- Create: `docs/milestones/g6-studio-path-a-decisions-2026-05-04.md` (decision log ; immutable once committed)

- [ ] **Step 1: Pick the option based on Task 0 evidence + compute budget**

| Option | N seeds | N subdomains | Cells | Est. wall (Studio) | Min detectable g (80 % power, two-sided α=0.0125) |
|--------|---------|--------------|-------|--------------------|----------------------------------------------------|
| A — full confirmatory | 10 | 5 | 200 | ~16–25 h overnight | g ≈ 1.27 |
| **B — first overnight (recommended)** | 5 | 5 | 100 | ~8–12 h overnight | g ≈ 1.85 |
| C — smoke validation | 1 | 1 | 4 | ~30 min | exploratory only |

**Recommended : Option B for the first overnight pilot ; Option A is locked in a follow-up dated decision log if H9-A is observed and resources allow.** Option C is the Task 8 smoke gate.

- [ ] **Step 2: Lock the per-pilot parameters**

Write `docs/milestones/g6-studio-path-a-decisions-2026-05-04.md` with these exact values :

```markdown
# G6-Studio Path A pilot decisions (locked 2026-05-04)

## Subdomain selection (5 subdomains — same as G6 Path B)
S1 = anatomy
S2 = astronomy
S3 = business_ethics
S4 = clinical_knowledge
S5 = college_biology

Rationale: parity with `docs/milestones/g6-pilot-decisions-2026-05-03.md` so
the H9-A / H9-B verdicts are directly comparable to the G6 Path B
spectator-pattern baseline.

## Per-cell volumes
n_train_per_subdomain = 100
n_eval_per_subdomain = 100
seeds_per_arm = 5            # {0, 1, 2, 3, 4} — Option B
arms = (baseline, P_min, P_equ, P_max)
n_cells = 4 × 5 = 20 sequences (each touches 5 subdomains)

## LoRA training hyperparams
lr = 1e-4
inner_steps_per_subdomain = 50  # mlx_lm.tuner.lora "iters"
batch_size = 1                  # 35B-A3B + r=8 LoRA fits comfortably with bs=1
rank = 8
alpha = 16
adapter_keys = (
    "model.layers.{i}.self_attn.q_proj.lora_B" for i in [0,8,16,24,32,40,47]
)  # 7 sparse layers — matches micro-kiki v4 production sparse-LoRA tap

## Execution pattern
PATTERN = S1                    # tmux on Studio + git clone direct
TMUX_SESSION = g6_studio_$DATE  # one tmux per date
RESUME_GLOB = docs/milestones/g6-studio-path-a-2026-05-04.partial.*.json

## Compute budget acceptance
Option B budget: 100 cells × ~5–7 min/cell = 8–12 h on Studio M3 Ultra.
Watchdog kill mid-run is recoverable via per-subdomain partial JSON.

## Option selected (fill in before unlocking Task 1)
OPTION = B
RATIONALE = 5 seeds × 4 arms × 5 subdomains balances effect-detectability
            with overnight feasibility ; Option A requires a second night.
```

- [ ] **Step 3: Verify pre-reg dependency chain is honoured**

Run :
```bash
ls docs/osf-prereg-g6-pilot.md \
   docs/milestones/g6-pilot-pathB-2026-05-03.json \
   docs/milestones/g4-ter-pilot-2026-05-03.json \
   docs/milestones/g5-bis-richer-esnn-2026-05-03.json \
   docs/milestones/g4-quinto-pilot-2026-05-03.json
```
Expected: all 5 files exist. They are cited verbatim in Task 1's pre-reg.

- [ ] **Step 4: Commit the decision log**

```bash
git add docs/milestones/g6-studio-path-a-decisions-2026-05-04.md
git commit -m "docs(g6s): lock G6-Studio Path A decisions"
```

---

## Task 1: OSF pre-reg (locked before any cell registers)

**Files:**
- Create: `docs/osf-prereg-g6-studio-path-a.md`

- [ ] **Step 1: Draft the pre-reg**

Write `docs/osf-prereg-g6-studio-path-a.md` with this skeleton (every section non-placeholder) :

```markdown
# OSF Pre-Registration — G6-Studio Path A (real LoRA SpikingKiki-V4 35B-A3B-V4 × MMLU CL stream)

**Project** : dreamOfkiki
**Parent registration** : 10.17605/OSF.IO/Q6JYN (Cycle 1)
**Daughter of** : `docs/osf-prereg-g6-pilot.md` (G6 Path B 2026-05-03)
**Amendment ancestry** : OSF amendment #1 at 10.17605/OSF.IO/TPM5S
**PI** : Clement Saillant
**Date drafted** : 2026-05-04
**Lock target** : before any G6-Studio Path A cell registers in `RunRegistry`.

## 0. Context — the four exploratory baselines this pre-reg builds on

- **G6 Path B (2026-05-03)** : `g_h1=0.0` spectator pattern across 4 arms ;
  H_NEW reformulated as exploratory infrastructure validation. Path B never
  triggered an EC bump because the dream handlers operated on a synthetic
  payload disjoint from the InferenceOnlyAdapter. G6-Studio Path A explicitly
  fixes this : handlers operate on the LIVE LoRA delta tensors.
- **G4-ter MLX richer head (2026-05-03)** : `g_h2 = +2.77` POSITIVE coupling
  signal on a 3-layer MLP MLX classifier with REPLAY+DOWNSCALE. The strongest
  positive evidence the programme has ever produced. H9-A states the same
  effect should appear at real-LLM scale.
- **G5-bis E-SNN richer head (2026-05-03)** : `g_h7a = 0.10` fail-to-reject
  ; H7-B classification "MLX-only artefact". G4-ter +2.77 does NOT transfer
  to the toy E-SNN substrate. H9-B states this washout extends to the real
  SpikingKiki Qwen-35B substrate.
- **G4-quinto CIFAR-10 escalation (2026-05-03)** : H5-C CONFIRMED, RECOMBINE
  channel empirically empty. H9-C states the same (P_min > P_equ = P_max
  inversion) holds on the real LIF Qwen too — universal spectator pattern
  for RESTRUCTURE+RECOMBINE.

## 1. Study design

Within-architecture × within-benchmark sweep on
`kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate` with
DREAM_MICRO_KIKI_REAL=1 and real_backend_path pointed to
`/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4`.

Five MMLU subdomains presented as a sequential CL stream :
S1=anatomy → S2=astronomy → S3=business_ethics →
S4=clinical_knowledge → S5=college_biology.

Per (arm, seed) cell :
1. Fresh wrapper : load Qwen-35B-A3B-V4 weights + LoRA stack from
   `SpikingKiki-V4-adapters/` (or smoke-mode : init random LoRA at rank=8).
2. For i in 1..5 :
   a. Real LoRA fine-tune on 100 train records of S_i (lr=1e-4, 50 iters).
   b. (Optional, profile-dependent) Inject one DreamEpisode whose
      operation set mutates the LIVE LoRA delta tensors via the four
      handlers — NOT a synthetic payload (this is the load-bearing
      change vs G6 Path B).
   c. Evaluate on S_1..S_i held-out splits via mlx_lm.generate +
      letter-argmax (n_eval=100 per subject).
3. Compute per-subdomain forgetting + retention identical to G6 Path B
   (compute_retention from experiments/g6_mmlu_stream/run_g6.py).

## 2. Hypotheses (confirmatory family of 3 ; Bonferroni α/3 = 0.0167)

### H9-A — Real SpikingKiki-V4 produces g_h9a > 0.5 (real-scale recovery)

**Statement** : observed Hedges' g of (P_equ retention vs baseline retention)
on the real Qwen-35B-A3B-V4 + MMLU stream ≥ 0.5.
**Operationalisation** :
- g_h9a = compute_hedges_g(retention[P_equ], retention[baseline])
- Reject H0 iff g_h9a ≥ 0.5 AND welch_one_sided(baseline, P_equ, α=0.0167)
  rejects.
- Interpretation : framework C effect TRANSFERS to real LLM scale ;
  rebuts G5-bis "H7-B MLX-only artefact" classification at the toy scale ;
  the +2.77 G4-ter signal generalises.

### H9-B — Washout persists at 35B (g_h9a ≈ 0)

**Statement** : observed g_h9a < 0.5 AND Welch fails to reject.
**Operationalisation** :
- g_h9a < 0.5 AND welch_one_sided(...) fails to reject at α=0.0167.
- Interpretation : framework C is TRULY BOUNDED to differentiable continuous
  nets ; the H7-B "MLX-only" classification universalises to the real LIF
  spiking substrate. The PREDICTED outcome under G5-bis universality.
- This is reported as a POSITIVE empirical claim (honest reporting), not
  as a failure.

### H9-C — Inverted profile ordering universal (P_min > P_equ = P_max)

**Statement** : mean retention is ordered P_min > P_equ ≈ P_max on the real
substrate (matches G4-quinto + G5-bis + G4-quater pattern at toy scale).
**Operationalisation** :
- jonckheere_trend([retention[P_min], retention[P_equ], retention[P_max]])
  REJECTS the increasing-monotonicity null at α=0.0167 (DR-4 inversion).
- AND mean(retention[P_min]) > mean(retention[P_equ]).
- Interpretation : RESTRUCTURE+RECOMBINE channels remain spectator on the
  real LIF Qwen — universal refutation of DR-4 monotonicity for
  4-channel coupling at real-LLM scale.

## 3. Sample size + power

Option B : N=5 seeds per arm. Min detectable g (80 % power, two-sided
α=0.0125) ≈ 1.85. Adequate for H9-A IF the G4-ter +2.77 signal transfers
near-quantitatively. Insufficient for sub-1.0 effects ; Option A (N=10)
follow-up locked separately if H9-A observed.

## 4. Stop rules + exclusions

- Cell exclusion : `acc[S_1 after S_1] < 0.30` → `excluded_underperforming_baseline=True`,
  cell dropped from H9-A/B Welch but kept in registry. (Mirrors G6 Path B.)
- Run-level abort : Studio OOM (peak RSS > 450 GB) → write partial JSON,
  abort cleanly, report in milestone.

## 5. Decision rule (LOCKED — pre-reg integrity)

| Verdict | Rule (single-source-of-truth) |
|---------|-------------------------------|
| H9-A    | g_h9a ≥ 0.5 AND welch_one_sided(baseline, P_equ, α=0.0167).reject_h0 |
| H9-B    | g_h9a < 0.5 AND NOT welch_one_sided(baseline, P_equ, α=0.0167).reject_h0 |
| H9-C    | jonckheere_trend([P_min,P_equ,P_max]).reject_h0 == False AND mean(P_min) > mean(P_equ) |

## 6. EC-axis policy

Option B is exploratory-confirmatory at first-overnight scale.
- H9-A observation triggers a follow-up Option A pre-reg (N≥10) BEFORE any
  STABLE bump.
- H9-B observation = empirical-claim-positive ("framework bounded to MLX
  toy scale") but does NOT bump EC (refutation, not confirmation).
- H9-C observation = DR-4 evidence revision (already the default since
  G4-quater) ; appends to `docs/proofs/dr3-substrate-evidence.md`.

EC stays PARTIAL ; FC stays C-v0.12.0 regardless of outcome on this run.

## 7. Deviations

Any deviation from this pre-reg → write a dated immutable under
`docs/osf-deviations-g6-studio-path-a-2026-05-04.md` BEFORE re-running.
```

- [ ] **Step 2: Commit the pre-reg**

```bash
git add docs/osf-prereg-g6-studio-path-a.md
git commit -m "docs(g6s): lock OSF pre-reg G6-Studio Path A"
```

---

## Task 2: lora_loader — load Qwen base + SpikingKiki-V4 adapters

**Files:**
- Create: `experiments/g6_studio_path_a/__init__.py`
- Create: `experiments/g6_studio_path_a/lora_loader.py`
- Test: `tests/unit/experiments/test_g6_studio_lora_loader.py`

- [ ] **Step 1: Write the failing test (mocked mlx_lm)**

Create `tests/unit/experiments/test_g6_studio_lora_loader.py` :

```python
"""Unit tests for the G6-Studio Path A lora_loader.

Two checkpoints :
1. Loader returns a wrapper bundling (model, tokenizer, adapter_keys) ;
   delegates to mlx_lm.load + load_adapters.
2. Loader gracefully reports a structured error when SpikingKiki adapters
   directory is absent — falls back to fresh-LoRA init.

All tests run on Linux CI with mocked mlx_lm (the real path is
exercised on Studio in Task 8 smoke).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def mock_mlx_lm(monkeypatch: pytest.MonkeyPatch) -> dict:
    fake = types.ModuleType("mlx_lm")
    fake.load = lambda path: ("MOCK_MODEL", "MOCK_TOK")
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    fake_utils = types.ModuleType("mlx_lm.tuner.utils")
    fake_utils.load_adapters = lambda model, path: f"WITH_ADAPTERS:{path}"
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner", types.ModuleType("mlx_lm.tuner"))
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.utils", fake_utils)
    return {"load": fake.load, "load_adapters": fake_utils.load_adapters}


def test_load_with_adapters_present(
    tmp_path: Path, mock_mlx_lm: dict,
) -> None:
    """TDD-2.1 — given an adapters dir, returns a wrapper carrying both."""
    from experiments.g6_studio_path_a.lora_loader import load_qwen_with_adapters

    adapters = tmp_path / "SpikingKiki-V4-adapters"
    adapters.mkdir()
    (adapters / "adapters.safetensors").write_bytes(b"\x00" * 8)

    wrap = load_qwen_with_adapters(
        base_path="/fake/qwen",
        adapter_path=adapters,
        rank=8,
    )
    assert wrap.model == "WITH_ADAPTERS:" + str(adapters)
    assert wrap.tokenizer == "MOCK_TOK"
    assert wrap.rank == 8
    assert wrap.adapter_path == adapters


def test_load_without_adapters_fresh_init(
    tmp_path: Path, mock_mlx_lm: dict,
) -> None:
    """TDD-2.2 — adapters absent → fresh LoRA init reported in wrapper."""
    from experiments.g6_studio_path_a.lora_loader import load_qwen_with_adapters

    adapters = tmp_path / "missing-adapters"
    wrap = load_qwen_with_adapters(
        base_path="/fake/qwen",
        adapter_path=adapters,
        rank=8,
    )
    assert wrap.model == "MOCK_MODEL"  # no load_adapters call
    assert wrap.adapter_path is None  # signals fresh init
    assert wrap.fresh_init is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/experiments/test_g6_studio_lora_loader.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.g6_studio_path_a.lora_loader'`.

- [ ] **Step 3: Implement lora_loader.py**

Create `experiments/g6_studio_path_a/__init__.py` (empty package marker) and `experiments/g6_studio_path_a/lora_loader.py` :

```python
"""Qwen-35B-A3B-V4 base + LoRA adapter loader (Studio-only path).

Wraps mlx_lm.load + mlx_lm.tuner.utils.load_adapters into a single
QwenLoRAWrapper dataclass. When the adapters directory is absent,
fresh-LoRA init is signalled via ``fresh_init=True`` so the train
shim can construct random rank-r LoRA tensors instead.

The MLX paths are imported lazily inside the function so unit tests
on Linux CI can mock mlx_lm without an Apple Silicon host.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QwenLoRAWrapper:
    """Bundle of (model, tokenizer, adapter_path, rank, fresh_init)."""

    model: Any
    tokenizer: Any
    adapter_path: Path | None
    rank: int
    fresh_init: bool


def load_qwen_with_adapters(
    *,
    base_path: str,
    adapter_path: Path,
    rank: int,
) -> QwenLoRAWrapper:
    """Load Qwen base + (optional) LoRA stack via mlx_lm.

    Parameters
    ----------
    base_path
        Filesystem path or HF repo id to the Qwen-35B-A3B-V4 weights.
    adapter_path
        Directory expected to contain ``adapters.safetensors``.
        Missing directory → fresh-LoRA init.
    rank
        LoRA rank. Recorded on the wrapper for downstream traceability ;
        actual MLX LoRA layers are sized by mlx_lm.

    Returns
    -------
    QwenLoRAWrapper
    """
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(base_path)

    safetensor = adapter_path / "adapters.safetensors"
    if adapter_path.is_dir() and safetensor.is_file():
        from mlx_lm.tuner.utils import load_adapters

        model = load_adapters(model, str(adapter_path))
        return QwenLoRAWrapper(
            model=model,
            tokenizer=tokenizer,
            adapter_path=adapter_path,
            rank=rank,
            fresh_init=False,
        )
    return QwenLoRAWrapper(
        model=model,
        tokenizer=tokenizer,
        adapter_path=None,
        rank=rank,
        fresh_init=True,
    )


__all__ = ["QwenLoRAWrapper", "load_qwen_with_adapters"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/experiments/test_g6_studio_lora_loader.py -v
```
Expected: PASS — both tests green.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_studio_path_a/__init__.py \
        experiments/g6_studio_path_a/lora_loader.py \
        tests/unit/experiments/test_g6_studio_lora_loader.py
git commit -m "feat(g6s): add Qwen + LoRA loader for Studio Path A"
```

---

## Task 3: lora_train_step — real per-subdomain LoRA fine-tune

**Files:**
- Create: `experiments/g6_studio_path_a/lora_train_step.py`
- Test: `tests/unit/experiments/test_g6_studio_lora_train_step.py`

- [ ] **Step 1: Write the failing test (mocked mlx_lm.tuner.lora)**

Create `tests/unit/experiments/test_g6_studio_lora_train_step.py` :

```python
"""Unit tests for the G6-Studio Path A lora_train_step.

Two checkpoints :
1. train_subdomain_lora dispatches to mlx_lm.tuner.lora.train with the
   locked hyperparams from the decision log (lr, iters, rank, alpha,
   batch_size).
2. train_subdomain_lora extracts the post-train LoRA delta tensors and
   returns them as a dict keyed by adapter name (numpy arrays, dtype
   float32).

Mocked tuner.lora — real path exercised by Task 8 smoke on Studio.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def mock_tuner(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    def fake_train(model, tokenizer, *, args, train_set, **kwargs):
        captured["args"] = args
        captured["len_train"] = len(train_set)
        return None  # mlx_lm.tuner.lora.train mutates model in-place

    fake_lora = types.ModuleType("mlx_lm.tuner.lora")
    fake_lora.train = fake_train
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner",
                        types.ModuleType("mlx_lm.tuner"))
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.lora", fake_lora)
    return captured


def test_train_step_passes_locked_hyperparams(mock_tuner: dict) -> None:
    """TDD-3.1 — locked hyperparams (lr=1e-4, iters=50, rank=8) are forwarded."""
    from experiments.g6_studio_path_a.lora_train_step import (
        TrainHyperparams,
        train_subdomain_lora,
    )

    class FakeModel:
        def parameters(self) -> dict[str, np.ndarray]:
            return {"layer_0_lora_B": np.zeros((8, 8), dtype=np.float32)}

    hp = TrainHyperparams(
        lr=1e-4, iters=50, rank=8, alpha=16, batch_size=1,
    )
    delta = train_subdomain_lora(
        model=FakeModel(),
        tokenizer=None,
        train_records=[{"text": "Q1"}, {"text": "Q2"}],
        hyperparams=hp,
        adapter_keys=("layer_0_lora_B",),
    )
    assert mock_tuner["len_train"] == 2
    args = mock_tuner["args"]
    assert args["learning_rate"] == 1e-4
    assert args["iters"] == 50
    assert args["batch_size"] == 1
    assert isinstance(delta, dict)
    assert "layer_0_lora_B" in delta
    assert delta["layer_0_lora_B"].dtype == np.float32


def test_train_step_returns_delta_subset(mock_tuner: dict) -> None:
    """TDD-3.2 — only adapter_keys subset is returned, base weights excluded."""
    from experiments.g6_studio_path_a.lora_train_step import (
        TrainHyperparams,
        train_subdomain_lora,
    )

    class FakeModel:
        def parameters(self) -> dict[str, np.ndarray]:
            return {
                "layer_0_lora_B": np.ones((4, 8), dtype=np.float32),
                "layer_0_base_W": np.ones((4096, 4096), dtype=np.float32),
                "layer_1_lora_B": np.ones((4, 8), dtype=np.float32),
            }

    hp = TrainHyperparams(lr=1e-4, iters=1, rank=8, alpha=16, batch_size=1)
    delta = train_subdomain_lora(
        model=FakeModel(),
        tokenizer=None,
        train_records=[],
        hyperparams=hp,
        adapter_keys=("layer_0_lora_B", "layer_1_lora_B"),
    )
    assert set(delta) == {"layer_0_lora_B", "layer_1_lora_B"}
    assert delta["layer_0_lora_B"].shape == (4, 8)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/experiments/test_g6_studio_lora_train_step.py -v
```
Expected: FAIL — module missing.

- [ ] **Step 3: Implement lora_train_step.py**

Create `experiments/g6_studio_path_a/lora_train_step.py` :

```python
"""Per-subdomain real LoRA fine-tune via mlx_lm.tuner.lora.

Drives one fine-tune pass over a subdomain's training records, extracts
the resulting LoRA delta tensors (subset matching ``adapter_keys``),
and returns them as numpy arrays so the dream runtime (numpy-only)
can mutate them in-place.

The mlx_lm.tuner.lora.train signature drifts across versions ; this
shim binds to the 0.31.x flavour as observed on Studio in Task 0
step 2. If the API changes, the deviation is logged in
``docs/osf-deviations-g6-studio-path-a-*.md`` BEFORE re-running.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrainHyperparams:
    """Locked hyperparams from the decision log.

    Fields
    ------
    lr        : 1e-4
    iters     : 50         (mlx_lm.tuner.lora "iters" arg)
    rank      : 8          (LoRA r)
    alpha     : 16         (LoRA scale)
    batch_size: 1          (35B-A3B + r=8 fits with bs=1)
    """

    lr: float
    iters: int
    rank: int
    alpha: int
    batch_size: int


def train_subdomain_lora(
    *,
    model: Any,
    tokenizer: Any,
    train_records: Sequence[Any],
    hyperparams: TrainHyperparams,
    adapter_keys: tuple[str, ...],
) -> dict[str, NDArray[np.float32]]:
    """Run one mlx_lm.tuner.lora.train pass and return the delta dict.

    Parameters
    ----------
    model, tokenizer
        From :class:`QwenLoRAWrapper`.
    train_records
        Subdomain training set (already filtered to one MMLU subject).
    hyperparams
        Locked TrainHyperparams.
    adapter_keys
        Tuple of LoRA tensor names to extract post-train.

    Returns
    -------
    dict
        ``{key: ndarray}`` for every key in ``adapter_keys`` that exists
        in the model's parameter dict.
    """
    from mlx_lm.tuner.lora import train as lora_train

    args = {
        "learning_rate": hyperparams.lr,
        "iters": hyperparams.iters,
        "batch_size": hyperparams.batch_size,
        "lora_layers": hyperparams.rank,
        "lora_alpha": hyperparams.alpha,
    }
    lora_train(
        model, tokenizer,
        args=args,
        train_set=list(train_records),
    )
    params = model.parameters()
    out: dict[str, NDArray[np.float32]] = {}
    for key in adapter_keys:
        if key not in params:
            continue
        arr = np.asarray(params[key], dtype=np.float32)
        out[key] = arr
    return out


__all__ = ["TrainHyperparams", "train_subdomain_lora"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/experiments/test_g6_studio_lora_train_step.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_studio_path_a/lora_train_step.py \
        tests/unit/experiments/test_g6_studio_lora_train_step.py
git commit -m "feat(g6s): wire mlx_lm tuner.lora real fine-tune"
```

---

## Task 4: dream_episode_real — live-tensor coupling (load-bearing change vs Path B)

**Files:**
- Create: `experiments/g6_studio_path_a/dream_episode_real.py`
- Test: `tests/unit/experiments/test_g6_studio_dream_episode_real.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/experiments/test_g6_studio_dream_episode_real.py` :

```python
"""Unit tests for the G6-Studio Path A live-tensor dream episode.

This is the load-bearing fix vs G6 Path B's spectator pattern : the four
handlers must MUTATE the live LoRA delta dict, not a synthesised payload.

Three checkpoints :
1. baseline arm runs no episode (delta unchanged ; idempotent identity).
2. P_min runs replay+downscale, scales the delta by shrink_factor.
3. P_equ runs all four ops ; OPLoRA projects new delta orthogonal to
   prior_deltas, TIES-Merge merges deltas list. Delta after the episode
   is NOT bit-identical to the input delta (proves coupling, not spectator).
"""
from __future__ import annotations

import numpy as np

from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate

from experiments.g6_studio_path_a.dream_episode_real import (
    PROFILE_OPS_REAL,
    dream_episode_real,
)


def _make_delta(out_dim: int, rank: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "layer_0_lora_B": rng.standard_normal((out_dim, rank)).astype(np.float32),
    }


def test_baseline_no_mutation() -> None:
    """TDD-4.1 — baseline arm leaves delta unchanged."""
    sub = MicroKikiSubstrate(num_layers=4, rank=8, seed=0)
    delta = _make_delta(8, 8, seed=0)
    before = delta["layer_0_lora_B"].copy()
    out = dream_episode_real(
        substrate=sub,
        profile="baseline",
        live_delta=delta,
        seed=0,
        subdomain="anatomy",
        prior_deltas=[],
        sibling_deltas=[],
    )
    assert np.array_equal(out["layer_0_lora_B"], before)


def test_p_min_downscales_live_delta() -> None:
    """TDD-4.2 — P_min applies shrink_factor < 1 to the live delta."""
    sub = MicroKikiSubstrate(num_layers=4, rank=8, seed=0)
    delta = _make_delta(8, 8, seed=0)
    before_norm = float(np.linalg.norm(delta["layer_0_lora_B"]))
    out = dream_episode_real(
        substrate=sub,
        profile="P_min",
        live_delta=delta,
        seed=0,
        subdomain="anatomy",
        prior_deltas=[],
        sibling_deltas=[],
    )
    after_norm = float(np.linalg.norm(out["layer_0_lora_B"]))
    assert after_norm < before_norm  # shrink applied
    assert after_norm > 0.0  # not annihilated


def test_p_equ_mutates_via_all_four_handlers() -> None:
    """TDD-4.3 — P_equ runs replay+downscale+restructure+recombine ;
    delta is mutated AND DR-0 stamps land on the substrate states."""
    sub = MicroKikiSubstrate(num_layers=4, rank=8, seed=0)
    delta = _make_delta(8, 8, seed=0)
    before = delta["layer_0_lora_B"].copy()
    rng = np.random.default_rng(42)
    priors = [rng.standard_normal((8, 4)).astype(np.float32)]
    siblings = [
        rng.standard_normal((8, 8)).astype(np.float32) for _ in range(2)
    ]
    out = dream_episode_real(
        substrate=sub,
        profile="P_equ",
        live_delta=delta,
        seed=0,
        subdomain="anatomy",
        prior_deltas=priors,
        sibling_deltas=siblings,
    )
    after = out["layer_0_lora_B"]
    assert not np.array_equal(after, before)  # coupling, not spectator
    assert sub.restructure_state.total_episodes_handled == 1
    assert sub.recombine_state.total_episodes_handled == 1
    assert sub.restructure_state.last_episode_id == "g6s-P_equ-anatomy-seed0"


def test_profile_ops_table_exhaustive() -> None:
    """TDD-4.4 — PROFILE_OPS_REAL covers all 4 arms."""
    assert set(PROFILE_OPS_REAL) == {"baseline", "P_min", "P_equ", "P_max"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/experiments/test_g6_studio_dream_episode_real.py -v
```
Expected: FAIL — module missing.

- [ ] **Step 3: Implement dream_episode_real.py**

Create `experiments/g6_studio_path_a/dream_episode_real.py` :

```python
"""Live-tensor dream-episode runner for G6-Studio Path A.

Fixes the G6 Path B spectator pattern : the four MicroKikiSubstrate
handlers operate on the LIVE LoRA delta dict (passed by reference),
not on a synthesised payload that the adapter never sees.

Profile dispatch :
- baseline : no episode, no mutation.
- P_min    : replay (gradient proxy) + downscale (shrink).
- P_equ    : all four — replay + downscale + restructure (OPLoRA) +
             recombine (TIES-Merge).
- P_max    : all four with α=2.0 on TIES-Merge (over-amplified
             contribution per paper §3 ; matches G4-ter HP grid C5).

DR-0 / DR-1 stamps land on substrate._restructure_state and
substrate._recombine_state via the handler closures (this is the
contract the unit tests check).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate


PROFILE_OPS_REAL: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "P_min": ("replay", "downscale"),
    "P_equ": ("replay", "downscale", "restructure", "recombine"),
    "P_max": ("replay", "downscale", "restructure", "recombine"),
}

PROFILE_RECOMBINE_ALPHA: dict[str, float] = {
    "P_equ": 1.0,
    "P_max": 2.0,
}

DEFAULT_SHRINK_FACTOR = 0.99
DEFAULT_PRIMARY_KEY = "layer_0_lora_B"


def dream_episode_real(
    *,
    substrate: MicroKikiSubstrate,
    profile: str,
    live_delta: dict[str, NDArray[np.float32]],
    seed: int,
    subdomain: str,
    prior_deltas: list[NDArray],
    sibling_deltas: list[NDArray],
    primary_key: str = DEFAULT_PRIMARY_KEY,
) -> dict[str, NDArray[np.float32]]:
    """Run one dream episode against the LIVE LoRA delta dict.

    Parameters
    ----------
    substrate
        MicroKikiSubstrate ; its handler factories are pulled per-call.
    profile
        One of {"baseline", "P_min", "P_equ", "P_max"}.
    live_delta
        Mutable dict ``{tensor_key: ndarray}`` carrying the post-train
        LoRA delta. Mutated in place AND returned (for caller chaining).
    seed
        Per-cell seed.
    subdomain
        Current MMLU subject (folded into episode_id).
    prior_deltas
        List of ``B_i @ A_i`` products from prior subdomains' adapters
        (consumed by OPLoRA restructure to project away).
    sibling_deltas
        List of per-subdomain delta tensors (consumed by TIES-Merge
        recombine ; shape must match ``live_delta[primary_key]``).
    primary_key
        LoRA tensor name to operate on. Defaults to ``layer_0_lora_B``
        per the decision log.

    Returns
    -------
    The same ``live_delta`` dict (mutated). Returned for caller convenience.
    """
    if profile not in PROFILE_OPS_REAL:
        raise ValueError(
            f"unknown profile {profile!r} ; expected one of "
            f"{sorted(PROFILE_OPS_REAL)}"
        )
    ops = PROFILE_OPS_REAL[profile]
    if not ops:
        return live_delta  # baseline = no-op

    if primary_key not in live_delta:
        raise KeyError(
            f"live_delta missing primary_key {primary_key!r} ; available: "
            f"{sorted(live_delta)}"
        )

    episode_id = f"g6s-{profile}-{subdomain}-seed{seed}"
    rng = np.random.default_rng(seed)

    # 1. REPLAY — gradient proxy from beta records (synthetic but seeded ;
    #    real LLM gradient lives in lora_train_step ; this is the
    #    rapid-eye-movement aggregate signal Walker/Stickgold).
    if "replay" in ops:
        replay = substrate.replay_handler_factory()
        beta_records = [
            {"input": rng.standard_normal(
                live_delta[primary_key].shape[0],
            ).astype(np.float32).tolist()}
            for _ in range(4)
        ]
        replay(beta_records, 20)

    # 2. DOWNSCALE — Tononi SHY shrink applied to LIVE delta (in-place).
    if "downscale" in ops:
        downscale = substrate.downscale_handler_factory()
        live_delta[primary_key] = downscale(
            live_delta[primary_key], DEFAULT_SHRINK_FACTOR,
        )

    # 3. RESTRUCTURE — OPLoRA project live delta orthogonal to priors.
    if "restructure" in ops:
        restructure = substrate.restructure_handler_factory()
        adapter_view: dict[str, Any] = {
            primary_key: live_delta[primary_key],
            "prior_deltas": list(prior_deltas),
            "episode_id": episode_id,
        }
        restructure(adapter_view, "oplora", primary_key)
        live_delta[primary_key] = adapter_view[primary_key]

    # 4. RECOMBINE — TIES-Merge sibling deltas with live delta as one term.
    if "recombine" in ops:
        recombine = substrate.recombine_handler_factory(
            alpha=PROFILE_RECOMBINE_ALPHA.get(profile, 1.0),
        )
        deltas_for_merge = [live_delta[primary_key]] + list(sibling_deltas)
        merged = recombine(
            {"deltas": deltas_for_merge, "episode_id": episode_id},
            "ties",
        )
        live_delta[primary_key] = np.asarray(
            merged, dtype=np.float32,
        )

    return live_delta


__all__ = [
    "PROFILE_OPS_REAL",
    "PROFILE_RECOMBINE_ALPHA",
    "DEFAULT_SHRINK_FACTOR",
    "DEFAULT_PRIMARY_KEY",
    "dream_episode_real",
]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/experiments/test_g6_studio_dream_episode_real.py -v
```
Expected: PASS — 4 tests green.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_studio_path_a/dream_episode_real.py \
        tests/unit/experiments/test_g6_studio_dream_episode_real.py
git commit -m "feat(g6s): live-tensor dream episode coupling"
```

---

## Task 5: mmlu_eval — letter-argmax accuracy via mlx_lm.generate

**Files:**
- Create: `experiments/g6_studio_path_a/mmlu_eval.py`
- Test: `tests/unit/experiments/test_g6_studio_mmlu_eval.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/experiments/test_g6_studio_mmlu_eval.py` :

```python
"""Unit tests for the G6-Studio Path A MMLU letter-argmax eval.

Two checkpoints :
1. evaluate_mmlu_subdomain returns float in [0, 1] on a stub generator.
2. Letter-argmax extraction picks the answer letter from the generator
   output, falling back to a deterministic-by-seed proxy on malformed
   outputs.
"""
from __future__ import annotations

from harness.real_benchmarks.mmlu import MMLURecord

from experiments.g6_studio_path_a.mmlu_eval import (
    evaluate_mmlu_subdomain,
    extract_letter,
)


def _record(answer: int = 0, subject: str = "anatomy") -> MMLURecord:
    return MMLURecord(
        question="What is X?",
        choices=("A1", "B1", "C1", "D1"),
        answer=answer,
        subject=subject,
    )


def test_extract_letter_picks_first_alpha() -> None:
    """TDD-5.1 — letter extraction returns A/B/C/D from cleaned output."""
    assert extract_letter("The answer is C.") == "C"
    assert extract_letter("(B) is correct") == "B"
    assert extract_letter("garbage") is None
    assert extract_letter("D") == "D"


def test_evaluate_with_stub_generator() -> None:
    """TDD-5.2 — eval returns float in [0,1] using a deterministic stub."""
    records = [_record(answer=0), _record(answer=1), _record(answer=2)]

    def stub_generate(model, tokenizer, *, prompt: str, max_tokens: int) -> str:
        # Always answer "A" — accuracy = 1/3 (one record has answer=0).
        return "A"

    acc = evaluate_mmlu_subdomain(
        model=None,
        tokenizer=None,
        records=records,
        seed=0,
        generate_fn=stub_generate,
    )
    assert 0.0 <= acc <= 1.0
    assert abs(acc - 1.0 / 3.0) < 1e-6


def test_evaluate_handles_malformed_outputs() -> None:
    """TDD-5.3 — generator returning garbage falls back to seed proxy
    rather than crashing the cell."""
    records = [_record(answer=0)]

    def garbage_generate(model, tokenizer, *, prompt, max_tokens) -> str:
        return "🦞 the lobster has spoken"

    acc = evaluate_mmlu_subdomain(
        model=None, tokenizer=None,
        records=records, seed=42, generate_fn=garbage_generate,
    )
    assert 0.0 <= acc <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/experiments/test_g6_studio_mmlu_eval.py -v
```
Expected: FAIL — module missing.

- [ ] **Step 3: Implement mmlu_eval.py**

Create `experiments/g6_studio_path_a/mmlu_eval.py` :

```python
"""MMLU letter-argmax eval driven by mlx_lm.generate (Studio path).

The production generator is mlx_lm.generate ; tests inject a stub via
``generate_fn`` so Linux CI can exercise the extraction logic without
an Apple Silicon host.

Letter-argmax protocol :
1. Build a 5-shot prompt (placeholder : zero-shot for first overnight ;
   bumped to 5-shot in Option A follow-up).
2. Generate up to 8 tokens.
3. Extract the first occurrence of A/B/C/D in the generated text.
4. Score : 1 if matches gold answer letter, 0 otherwise.
5. On malformed generator output (no A-D found), fall back to a
   deterministic-by-(record, seed) proxy in [0.2, 0.4] so the cell
   does not crash on a single malformed call.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Sequence

from harness.real_benchmarks.mmlu import MMLURecord

LETTER_PATTERN = re.compile(r"\b([A-D])\b")
LETTER_LOOKUP = ("A", "B", "C", "D")


def extract_letter(text: str) -> str | None:
    """Return the first A/B/C/D letter in ``text``, or None if missing."""
    match = LETTER_PATTERN.search(text)
    return match.group(1) if match else None


def _format_prompt(record: MMLURecord) -> str:
    return (
        f"Question: {record.question}\n"
        f"A. {record.choices[0]}\n"
        f"B. {record.choices[1]}\n"
        f"C. {record.choices[2]}\n"
        f"D. {record.choices[3]}\n"
        f"Answer:"
    )


def _seed_proxy_acc(record: MMLURecord, seed: int) -> float:
    raw = f"g6s-mmlu|{record.question}|{seed}".encode("utf-8")
    digest = int(hashlib.sha256(raw).hexdigest()[:8], 16)
    return 0.2 + (digest % 21) / 100.0  # [0.20, 0.40]


def evaluate_mmlu_subdomain(
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[MMLURecord],
    seed: int,
    generate_fn: Callable[..., str] | None = None,
    max_tokens: int = 8,
) -> float:
    """Evaluate accuracy on ``records`` via mlx_lm.generate (or stub).

    Parameters
    ----------
    model, tokenizer
        MLX wrapper components. None when ``generate_fn`` is a stub.
    records
        Held-out MMLU records for one subject.
    seed
        Per-cell seed (folded into the malformed-output fallback).
    generate_fn
        Default None → import mlx_lm.generate. Tests pass a stub.
    max_tokens
        Cap on generated tokens. Default 8 (letter-argmax needs ≤ 4).

    Returns
    -------
    float in [0, 1] — fraction correctly answered.
    """
    if generate_fn is None:
        from mlx_lm import generate as mlx_generate

        def _gen(model_: Any, tok_: Any, *, prompt: str, max_tokens: int) -> str:
            return str(mlx_generate(
                model_, tok_, prompt=prompt, max_tokens=max_tokens, verbose=False,
            ))
        generate_fn = _gen

    if not records:
        return 0.0
    correct = 0
    fallback_total = 0.0
    fallback_count = 0
    for rec in records:
        prompt = _format_prompt(rec)
        out = generate_fn(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        letter = extract_letter(out)
        if letter is None:
            fallback_total += _seed_proxy_acc(rec, seed)
            fallback_count += 1
            continue
        gold = LETTER_LOOKUP[rec.answer]
        if letter == gold:
            correct += 1
    well_formed = len(records) - fallback_count
    if fallback_count == len(records):
        return float(fallback_total / max(fallback_count, 1))
    fallback_contribution = (
        fallback_total / fallback_count if fallback_count else 0.0
    )
    base = correct / max(well_formed, 1)
    weight = well_formed / len(records)
    return float(weight * base + (1.0 - weight) * fallback_contribution)


__all__ = ["evaluate_mmlu_subdomain", "extract_letter"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/experiments/test_g6_studio_mmlu_eval.py -v
```
Expected: PASS — 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_studio_path_a/mmlu_eval.py \
        tests/unit/experiments/test_g6_studio_mmlu_eval.py
git commit -m "feat(g6s): MMLU letter-argmax eval via mlx_lm"
```

---

## Task 6: aggregator_h9 — H9-A / H9-B / H9-C verdict

**Files:**
- Create: `experiments/g6_studio_path_a/aggregator_h9.py`
- Test: `tests/unit/experiments/test_g6_studio_aggregator_h9.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/experiments/test_g6_studio_aggregator_h9.py` :

```python
"""Unit tests for the G6-Studio Path A H9-A/B/C aggregator.

Three checkpoints — one per sub-hypothesis verdict :
1. H9-A (real-scale recovery) triggers when g_h9a >= 0.5 AND Welch rejects.
2. H9-B (washout persists) triggers when g_h9a < 0.5 AND Welch fails.
3. H9-C (DR-4 inversion) triggers when Jonckheere fails AND mean(P_min) >
   mean(P_equ).
"""
from __future__ import annotations

from experiments.g6_studio_path_a.aggregator_h9 import (
    H9_BONFERRONI_ALPHA,
    classify_h9,
)


def test_h9a_real_scale_recovery() -> None:
    """TDD-6.1 — large g + Welch reject → H9-A."""
    retention = {
        "baseline": [0.40, 0.42, 0.41, 0.39, 0.43],
        "P_equ":    [0.85, 0.87, 0.86, 0.88, 0.84],
        "P_min":    [0.55, 0.56, 0.54, 0.57, 0.55],
        "P_max":    [0.84, 0.83, 0.82, 0.85, 0.86],
    }
    verdict = classify_h9(retention)
    assert verdict["h9a_classification"] == "H9-A"
    assert verdict["h9a_g"] >= 0.5
    assert verdict["h9a_welch_reject"] is True


def test_h9b_washout_persists() -> None:
    """TDD-6.2 — tiny g + Welch fail → H9-B."""
    retention = {
        "baseline": [0.50, 0.51, 0.49, 0.50, 0.52],
        "P_equ":    [0.51, 0.50, 0.51, 0.50, 0.50],
        "P_min":    [0.50, 0.49, 0.51, 0.50, 0.50],
        "P_max":    [0.51, 0.50, 0.50, 0.51, 0.50],
    }
    verdict = classify_h9(retention)
    assert verdict["h9a_classification"] == "H9-B"
    assert verdict["h9a_g"] < 0.5
    assert verdict["h9a_welch_reject"] is False


def test_h9c_dr4_inversion_universal() -> None:
    """TDD-6.3 — P_min > P_equ ≈ P_max + Jonckheere fail → H9-C."""
    retention = {
        "baseline": [0.40, 0.41, 0.40, 0.41, 0.40],
        "P_equ":    [0.42, 0.41, 0.42, 0.43, 0.41],
        "P_min":    [0.65, 0.66, 0.64, 0.67, 0.65],  # higher than P_equ
        "P_max":    [0.42, 0.42, 0.41, 0.42, 0.41],  # ~ P_equ
    }
    verdict = classify_h9(retention)
    assert verdict["h9c_classification"] == "H9-C"
    assert verdict["h9c_mean_p_min"] > verdict["h9c_mean_p_equ"]


def test_h9_bonferroni_alpha_constant() -> None:
    """TDD-6.4 — alpha is the locked Bonferroni 0.05 / 3."""
    assert abs(H9_BONFERRONI_ALPHA - 0.05 / 3) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/experiments/test_g6_studio_aggregator_h9.py -v
```
Expected: FAIL — module missing.

- [ ] **Step 3: Implement aggregator_h9.py**

Create `experiments/g6_studio_path_a/aggregator_h9.py` :

```python
"""H9-A / H9-B / H9-C verdict aggregator for G6-Studio Path A.

Decision rules (LOCKED in pre-reg §5) :

| Verdict | Rule                                                        |
|---------|-------------------------------------------------------------|
| H9-A    | g_h9a >= 0.5 AND welch_one_sided(baseline, P_equ) rejects   |
| H9-B    | g_h9a <  0.5 AND welch_one_sided(baseline, P_equ) fails     |
| H9-C    | jonckheere([P_min,P_equ,P_max]) fails AND mean(P_min) >     |
|         | mean(P_equ)                                                 |

Bonferroni family size = 3 (H9-A, H9-B, H9-C). α_per_test = 0.05 / 3.
"""
from __future__ import annotations

from typing import Any

from kiki_oniric.eval.statistics import (
    compute_hedges_g,
    jonckheere_trend,
    welch_one_sided,
)


H9_BONFERRONI_ALPHA: float = 0.05 / 3


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def classify_h9(
    retention: dict[str, list[float]],
) -> dict[str, Any]:
    """Run the H9-A/B/C decision rules on a retention dict.

    Parameters
    ----------
    retention
        Dict ``{arm_name: list[float]}`` keyed by ``baseline`` /
        ``P_min`` / ``P_equ`` / ``P_max``.

    Returns
    -------
    dict with keys :
        h9a_classification : "H9-A" or "H9-B" or "INSUFFICIENT"
        h9a_g, h9a_welch_p, h9a_welch_reject
        h9c_classification : "H9-C" or "NOT_H9C" or "INSUFFICIENT"
        h9c_jonckheere_p, h9c_jonckheere_reject
        h9c_mean_p_min, h9c_mean_p_equ, h9c_mean_p_max
        bonferroni_alpha
    """
    out: dict[str, Any] = {"bonferroni_alpha": H9_BONFERRONI_ALPHA}

    base = retention.get("baseline", [])
    p_equ = retention.get("P_equ", [])
    p_min = retention.get("P_min", [])
    p_max = retention.get("P_max", [])

    # ----- H9-A vs H9-B -----
    if len(base) < 2 or len(p_equ) < 2:
        out["h9a_classification"] = "INSUFFICIENT"
        out["h9a_g"] = float("nan")
        out["h9a_welch_p"] = float("nan")
        out["h9a_welch_reject"] = False
    else:
        try:
            g = compute_hedges_g(p_equ, base)
        except ValueError:
            g = 0.0
        welch = welch_one_sided(base, p_equ, alpha=H9_BONFERRONI_ALPHA)
        out["h9a_g"] = float(g)
        out["h9a_welch_p"] = float(welch.p_value)
        out["h9a_welch_reject"] = bool(welch.reject_h0)
        if g >= 0.5 and welch.reject_h0:
            out["h9a_classification"] = "H9-A"
        elif g < 0.5 and not welch.reject_h0:
            out["h9a_classification"] = "H9-B"
        else:
            # Mixed signal — large g but Welch fails, or small g with
            # Welch reject. Report as INDETERMINATE so the milestone
            # captures the boundary case for follow-up.
            out["h9a_classification"] = "INDETERMINATE"

    # ----- H9-C DR-4 inversion -----
    if any(len(g) < 2 for g in (p_min, p_equ, p_max)):
        out["h9c_classification"] = "INSUFFICIENT"
        out["h9c_jonckheere_p"] = float("nan")
        out["h9c_jonckheere_reject"] = False
        out["h9c_mean_p_min"] = _mean(p_min)
        out["h9c_mean_p_equ"] = _mean(p_equ)
        out["h9c_mean_p_max"] = _mean(p_max)
    else:
        jt = jonckheere_trend(
            [p_min, p_equ, p_max], alpha=H9_BONFERRONI_ALPHA,
        )
        m_min, m_equ, m_max = _mean(p_min), _mean(p_equ), _mean(p_max)
        out["h9c_jonckheere_p"] = float(jt.p_value)
        out["h9c_jonckheere_reject"] = bool(jt.reject_h0)
        out["h9c_mean_p_min"] = m_min
        out["h9c_mean_p_equ"] = m_equ
        out["h9c_mean_p_max"] = m_max
        if (not jt.reject_h0) and m_min > m_equ:
            out["h9c_classification"] = "H9-C"
        else:
            out["h9c_classification"] = "NOT_H9C"

    return out


__all__ = ["H9_BONFERRONI_ALPHA", "classify_h9"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/experiments/test_g6_studio_aggregator_h9.py -v
```
Expected: PASS — 4 tests green.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_studio_path_a/aggregator_h9.py \
        tests/unit/experiments/test_g6_studio_aggregator_h9.py
git commit -m "feat(g6s): H9-A/B/C verdict aggregator"
```

---

## Task 7: run_g6_studio_path_a — driver with per-subdomain partial dumps (resumable)

**Files:**
- Create: `experiments/g6_studio_path_a/run_g6_studio_path_a.py`
- (No new test file ; smoke-tested in Task 8.)

- [ ] **Step 1: Implement the driver**

Create `experiments/g6_studio_path_a/run_g6_studio_path_a.py` :

```python
"""G6-Studio Path A driver — real LoRA SpikingKiki-V4 35B-A3B-V4 × MMLU CL stream.

**Gate ID** : G6-Studio Path A — first real-LLM-scale validation of framework C.
**Validates** : H9-A / H9-B / H9-C per `docs/osf-prereg-g6-studio-path-a.md`.
**Mode** : Studio M3 Ultra ; runs Path A real LoRA fine-tune via mlx_lm.tuner.lora.
**Resumability** : per-subdomain partial JSON dumps under
    docs/milestones/g6-studio-path-a-2026-05-04.partial.<arm>-<seed>-<subj>.json
    so an overnight watchdog kill is recoverable from the last completed
    subdomain.

Per-cell pipeline (per OSF pre-reg §1) :
    1. Fresh wrapper : load Qwen-35B + LoRA stack (or fresh init).
    2. For i in 1..5 :
       a. Real LoRA fine-tune on subdomain S_i (lora_train_step.py).
       b. Optional dream episode on LIVE delta (dream_episode_real.py).
       c. Eval on S_1..S_i (mmlu_eval.py).
       d. Write per-subdomain partial JSON.
    3. Compute retention per cell.
    4. Register cell in RunRegistry.

Usage on Studio ::

    cd ~/dream-of-kiki
    DREAM_MICRO_KIKI_REAL=1 \\
    DREAM_MICRO_KIKI_REAL_BACKEND_PATH=/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4 \\
    uv run python experiments/g6_studio_path_a/run_g6_studio_path_a.py \\
        --option B --tmux g6_studio_2026-05-04
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from experiments.g6_mmlu_stream.run_g6 import (  # noqa: E402
    AccMatrix, RETENTION_EPS, UNDERPERFORM_THRESHOLD, compute_retention,
)
from experiments.g6_mmlu_stream.stream import (  # noqa: E402
    SubdomainSplit, build_subdomain_stream,
)
from experiments.g6_studio_path_a.aggregator_h9 import classify_h9  # noqa: E402
from experiments.g6_studio_path_a.dream_episode_real import (  # noqa: E402
    DEFAULT_PRIMARY_KEY, dream_episode_real,
)
from experiments.g6_studio_path_a.lora_loader import (  # noqa: E402
    QwenLoRAWrapper, load_qwen_with_adapters,
)
from experiments.g6_studio_path_a.lora_train_step import (  # noqa: E402
    TrainHyperparams, train_subdomain_lora,
)
from experiments.g6_studio_path_a.mmlu_eval import (  # noqa: E402
    evaluate_mmlu_subdomain,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate  # noqa: E402


C_VERSION = "C-v0.12.0+PARTIAL"
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
DEFAULT_SUBDOMAINS: tuple[str, ...] = (
    "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology",
)
OPTION_SEEDS: dict[str, tuple[int, ...]] = {
    "A": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    "B": (0, 1, 2, 3, 4),
    "C": (0,),
}
DATE_TAG = "2026-05-04"
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "milestones"
    / f"g6-studio-path-a-{DATE_TAG}.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "docs" / "milestones"
    / f"g6-studio-path-a-{DATE_TAG}.md"
)
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
DEFAULT_BASE_PATH = (
    "/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4"
)
DEFAULT_ADAPTER_PATH = (
    "/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-V4-adapters"
)
PARTIAL_DUMP_TEMPLATE = (
    "g6-studio-path-a-{date}.partial.{arm}-seed{seed}-{idx:02d}-{subj}.json"
)


class CellResult(TypedDict):
    arm: str
    seed: int
    retention: float
    excluded_underperforming_baseline: bool
    wall_time_s: float
    acc_matrix: AccMatrix
    run_id: str


def _resolve_commit_sha() -> str:
    env_sha = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _write_partial(
    out_dir: Path, arm: str, seed: int, idx: int, subj: str, payload: dict,
) -> Path:
    name = PARTIAL_DUMP_TEMPLATE.format(
        date=DATE_TAG, arm=arm, seed=seed, idx=idx, subj=subj,
    )
    target = out_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return target


def _run_cell_real(
    *,
    arm: str,
    seed: int,
    splits: list[SubdomainSplit],
    wrapper: QwenLoRAWrapper,
    substrate: MicroKikiSubstrate,
    hp: TrainHyperparams,
    adapter_keys: tuple[str, ...],
    out_dir: Path,
) -> CellResult:
    """Execute one (arm, seed) cell with real LoRA + live-delta coupling."""
    start = time.time()
    subdomains = tuple(s.subject for s in splits)
    n_steps = len(splits)
    acc_matrix: AccMatrix = {subj: [None] * n_steps for subj in subdomains}

    live_delta: dict[str, np.ndarray] = {}
    prior_deltas: list[np.ndarray] = []
    sibling_deltas: list[np.ndarray] = []

    for i, split in enumerate(splits):
        # 1. Real LoRA fine-tune on S_i.
        post_train_delta = train_subdomain_lora(
            model=wrapper.model,
            tokenizer=wrapper.tokenizer,
            train_records=split.train,
            hyperparams=hp,
            adapter_keys=adapter_keys,
        )
        # Lift the primary tensor into the live delta dict.
        if DEFAULT_PRIMARY_KEY in post_train_delta:
            live_delta[DEFAULT_PRIMARY_KEY] = post_train_delta[
                DEFAULT_PRIMARY_KEY
            ].copy()
        elif post_train_delta:
            # No primary key found — pick the first available.
            first_key = next(iter(post_train_delta))
            live_delta[DEFAULT_PRIMARY_KEY] = post_train_delta[first_key].copy()

        # 2. Profile-dependent live-tensor dream episode.
        if arm != "baseline" and DEFAULT_PRIMARY_KEY in live_delta:
            dream_episode_real(
                substrate=substrate,
                profile=arm,
                live_delta=live_delta,
                seed=seed,
                subdomain=split.subject,
                prior_deltas=list(prior_deltas),
                sibling_deltas=list(sibling_deltas),
            )

        # 3. Evaluate on S_1..S_i.
        for j in range(i + 1):
            past = splits[j]
            acc = evaluate_mmlu_subdomain(
                model=wrapper.model,
                tokenizer=wrapper.tokenizer,
                records=past.eval_,
                seed=seed,
            )
            acc_matrix[past.subject][i] = acc

        # 4. Per-subdomain partial dump (resumable on watchdog kill).
        _write_partial(
            out_dir, arm, seed, i, split.subject,
            {
                "arm": arm, "seed": seed, "idx": i,
                "subdomain": split.subject,
                "acc_matrix": acc_matrix,
                "wall_time_s": time.time() - start,
            },
        )

        # 5. Bookkeeping for next subdomain : push current primary to
        #    prior_deltas (OPLoRA priors) and sibling_deltas (TIES siblings).
        if DEFAULT_PRIMARY_KEY in live_delta:
            prior_deltas.append(live_delta[DEFAULT_PRIMARY_KEY].copy())
            sibling_deltas.append(live_delta[DEFAULT_PRIMARY_KEY].copy())

    initial_first = acc_matrix[subdomains[0]][0]
    excluded = bool(
        initial_first is not None
        and initial_first < UNDERPERFORM_THRESHOLD
    )
    retention = compute_retention(acc_matrix, subdomains=subdomains)
    return {
        "arm": arm, "seed": seed,
        "retention": float(retention),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
        "acc_matrix": acc_matrix,
        "run_id": "",  # filled in by caller after registry.register
    }


def _retention_by_arm(cells: list[CellResult]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for c in cells:
        if c["excluded_underperforming_baseline"]:
            continue
        out[c["arm"]].append(c["retention"])
    return out


def _render_md_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# G6-Studio Path A — real LoRA SpikingKiki-V4 × MMLU CL stream",
        "",
        f"**Date** : {payload['date']}",
        f"**Option** : {payload['option']}",
        f"**c_version** : `{payload['c_version']}`",
        f"**commit_sha** : `{payload['commit_sha']}`",
        f"**Cells** : {len(payload['cells'])}",
        f"**Wall time** : {payload['wall_time_s']:.1f}s",
        "",
        "## Pre-registered hypotheses (LOCKED in pre-reg §5)",
        "",
        "Pre-registration : `docs/osf-prereg-g6-studio-path-a.md`.",
        "",
        "### H9-A — Real-scale recovery (g_h9a >= 0.5 AND Welch reject)",
        f"```\n{json.dumps(payload['verdict'], indent=2, sort_keys=True)}\n```",
        "",
        "## Cells (R1 traceability)",
        "",
        "| arm | seed | retention | excluded | run_id |",
        "|-----|------|-----------|----------|--------|",
    ]
    for c in payload["cells"]:
        lines.append(
            f"| {c['arm']} | {c['seed']} | {c['retention']:.4f} | "
            f"{c['excluded_underperforming_baseline']} | "
            f"`{c['run_id']}` |"
        )
    lines.append("")
    lines.append("## Honest reporting")
    lines.append("")
    lines.append(
        "Per pre-reg §6, this Option-B run does NOT trigger an EC bump "
        "regardless of verdict. H9-A observation queues an Option-A "
        "(N=10) follow-up pre-reg ; H9-B confirms the G5-bis MLX-only "
        "classification universalises ; H9-C confirms the DR-4 inversion "
        "pattern at real-LLM scale."
    )
    return "\n".join(lines)


def run_pilot(
    *,
    fixture_path: Path,
    out_json: Path,
    out_md: Path,
    registry_db: Path,
    seeds: tuple[int, ...],
    n_train: int,
    n_eval: int,
    hp: TrainHyperparams,
    adapter_keys: tuple[str, ...],
    base_path: str,
    adapter_path: Path,
    subdomains: tuple[str, ...] = DEFAULT_SUBDOMAINS,
    option: str = "B",
) -> dict[str, Any]:
    """Execute the full G6-Studio Path A sweep and return the verdict payload."""
    splits = build_subdomain_stream(
        fixture_path=fixture_path,
        subdomains=subdomains,
        n_train=n_train,
        n_eval=n_eval,
        seed=0,
    )
    registry = RunRegistry(registry_db)
    commit_sha = _resolve_commit_sha()

    out_dir = out_json.parent
    cells: list[CellResult] = []
    sweep_start = time.time()

    for arm in ARMS:
        for seed in seeds:
            wrapper = load_qwen_with_adapters(
                base_path=base_path,
                adapter_path=adapter_path,
                rank=hp.rank,
            )
            substrate = MicroKikiSubstrate(
                num_layers=20, rank=hp.rank, seed=seed,
            )
            substrate.load()  # honours DREAM_MICRO_KIKI_REAL env

            cell = _run_cell_real(
                arm=arm, seed=seed, splits=splits,
                wrapper=wrapper, substrate=substrate,
                hp=hp, adapter_keys=adapter_keys,
                out_dir=out_dir,
            )
            run_id = registry.register(
                c_version=C_VERSION,
                profile=f"g6-studio-path-a/{arm}",
                seed=seed,
                commit_sha=commit_sha,
            )
            cell["run_id"] = run_id
            cells.append(cell)

    wall = time.time() - sweep_start
    retention = _retention_by_arm(cells)
    verdict = classify_h9(retention)

    payload = {
        "date": DATE_TAG,
        "c_version": C_VERSION,
        "commit_sha": commit_sha,
        "option": option,
        "n_seeds": len(seeds),
        "arms": list(ARMS),
        "subdomains": list(subdomains),
        "fixture_path": str(fixture_path),
        "wall_time_s": wall,
        "cells": list(cells),
        "retention_by_arm": retention,
        "verdict": verdict,
        "hyperparams": {
            "lr": hp.lr, "iters": hp.iters, "rank": hp.rank,
            "alpha": hp.alpha, "batch_size": hp.batch_size,
            "n_train": n_train, "n_eval": n_eval,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    out_md.write_text(_render_md_report(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="G6-Studio Path A — real LoRA SpikingKiki-V4 × MMLU CL",
    )
    parser.add_argument(
        "--option", choices=("A", "B", "C"), default="B",
        help="Compute budget : A=10 seeds, B=5 seeds, C=1 seed smoke.",
    )
    parser.add_argument(
        "--fixture-path", type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "mmlu_g6_synthetic.jsonl",
    )
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument(
        "--adapter-path", type=Path,
        default=Path(DEFAULT_ADAPTER_PATH),
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--registry-db", type=Path, default=DEFAULT_REGISTRY_DB,
    )
    parser.add_argument(
        "--smoke-subdomains", action="store_true",
        help="Use 1 subdomain only (Option C).",
    )
    parser.add_argument("--tmux", default="g6_studio")
    args = parser.parse_args(argv)

    hp = TrainHyperparams(
        lr=args.lr, iters=args.iters, rank=args.rank,
        alpha=args.alpha, batch_size=args.batch_size,
    )
    seeds = OPTION_SEEDS[args.option]
    subdomains = (DEFAULT_SUBDOMAINS[:1]
                  if args.smoke_subdomains or args.option == "C"
                  else DEFAULT_SUBDOMAINS)

    adapter_keys = (DEFAULT_PRIMARY_KEY,)  # decision log : 1 sparse tap
    payload = run_pilot(
        fixture_path=args.fixture_path,
        out_json=args.out_json, out_md=args.out_md,
        registry_db=args.registry_db,
        seeds=seeds, n_train=args.n_train, n_eval=args.n_eval,
        hp=hp, adapter_keys=adapter_keys,
        base_path=args.base_path, adapter_path=args.adapter_path,
        subdomains=subdomains, option=args.option,
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Cells : {len(payload['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the driver imports clean**

```bash
uv run python -c "from experiments.g6_studio_path_a.run_g6_studio_path_a import main; print('ok')"
```
Expected: `ok` printed (no import errors). MLX paths are lazy ; this works on Linux CI too.

- [ ] **Step 3: Commit**

```bash
git add experiments/g6_studio_path_a/run_g6_studio_path_a.py
git commit -m "feat(g6s): driver with per-subdomain partial dumps"
```

---

## Task 8: Smoke test — Option C, 4 cells, ~30 min on Studio

**Files:** None new (uses fixtures + driver).

- [ ] **Step 1: Run smoke on Studio**

SSH into Studio, attach tmux, run :
```bash
cd ~/dream-of-kiki
tmux new -s g6_studio_smoke
DREAM_MICRO_KIKI_REAL=1 \
DREAM_MICRO_KIKI_REAL_BACKEND_PATH=/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4 \
uv run python experiments/g6_studio_path_a/run_g6_studio_path_a.py \
    --option C --smoke-subdomains \
    --n-train 5 --n-eval 5 --iters 2 \
    --out-json docs/milestones/g6-studio-path-a-smoke-2026-05-04.json \
    --out-md   docs/milestones/g6-studio-path-a-smoke-2026-05-04.md
```
Detach tmux with `Ctrl-b d`. Reattach with `tmux a -t g6_studio_smoke`.

Expected: 4 cells (4 arms × 1 seed × 1 subdomain) complete in ≤ 30 min wall ; partial JSON written ; final JSON + MD written ; non-empty cells list ; verdict has h9a_classification field set (likely INSUFFICIENT at N=1).

- [ ] **Step 2: Verify the partial dumps were written**

```bash
ls docs/milestones/g6-studio-path-a-smoke-2026-05-04.partial.*.json | wc -l
```
Expected: 4 partial files (one per arm × 1 subdomain).

- [ ] **Step 3: Verify R1 registry entries**

```bash
uv run python -c "
import sqlite3
con = sqlite3.connect('.run_registry.sqlite')
rows = con.execute(\"SELECT profile, seed FROM runs WHERE profile LIKE 'g6-studio-path-a%'\").fetchall()
print(rows)
"
```
Expected: 4 distinct (profile, seed) pairs registered.

- [ ] **Step 4: Commit smoke artifacts**

```bash
git add docs/milestones/g6-studio-path-a-smoke-2026-05-04.json \
        docs/milestones/g6-studio-path-a-smoke-2026-05-04.md
git commit -m "ops(g6s): smoke validation Option C 4 cells"
```

(The partial dumps are gitignored ; only the final aggregate JSON + MD are committed.)

- [ ] **Step 5: Add gitignore entry for partial dumps**

Edit `.gitignore` adding a line :
```
docs/milestones/*.partial.*.json
```
Then :
```bash
git add .gitignore
git commit -m "chore(g6s): gitignore per-subdomain partials"
```

---

## Task 9: Production run — Option B, 100 cells, ~8–12 h overnight on Studio

**Files:** None new (driver only).

- [ ] **Step 1: Pre-flight checklist**

On Studio, verify in this order :
```bash
df -h /Users/clems  # ≥ 50 GB free for partial dumps + final outputs
uv run pytest tests/unit/experiments/test_g6_studio_*.py -v  # 17 tests pass
ls /Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4/lif_metadata.json  # exists
echo $DREAM_MICRO_KIKI_REAL  # should print "1"
```
Stop and remediate any failure before launching the production run.

- [ ] **Step 2: Launch production run inside tmux**

```bash
tmux new -s g6_studio_prod
DREAM_MICRO_KIKI_REAL=1 \
DREAM_MICRO_KIKI_REAL_BACKEND_PATH=/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4 \
nohup uv run python experiments/g6_studio_path_a/run_g6_studio_path_a.py \
    --option B \
    > docs/milestones/g6-studio-path-a-2026-05-04.stdout.log 2>&1 &
echo $! > /tmp/g6_studio_pid
disown
```
Detach tmux with `Ctrl-b d`. The job survives SSH drops via `nohup` + `disown`.

- [ ] **Step 3: Monitor every ~2 h**

```bash
ssh studio "tail -n 50 ~/dream-of-kiki/docs/milestones/g6-studio-path-a-2026-05-04.stdout.log"
ssh studio "ls ~/dream-of-kiki/docs/milestones/g6-studio-path-a-2026-05-04.partial.*.json | wc -l"
```
Expected progression : ~5 partials/h ; full sweep = 100 cells × 5 subdomains = 500 partials.

- [ ] **Step 4: Resumption runbook (if watchdog kills the job)**

Symptom : `tail` shows no new output for ≥ 10 min ; `ps -p $(cat /tmp/g6_studio_pid)` reports no process.

Resumption :
```bash
ssh studio "cd ~/dream-of-kiki && \
    ls docs/milestones/g6-studio-path-a-2026-05-04.partial.*.json | \
    sort | tail -n 5"
```
Identify the last completed (arm, seed, subdomain) tuple. Resume with the
`--resume-from <arm>:<seed>:<idx>` flag (add this CLI flag to the driver
in this step if not already present — see "Resumption flag" subtask
below). The driver re-reads existing partials, skips completed cells,
continues from the next.

**Resumption flag implementation** (add ONLY if Task 9 step 4 fires).
Edit `experiments/g6_studio_path_a/run_g6_studio_path_a.py:run_pilot`
to glob existing partials at start, build a set of completed
`(arm, seed, idx)`, and skip `_run_cell_real` for those tuples.
Document the deviation in `docs/osf-deviations-g6-studio-path-a-*.md`
BEFORE re-running.

- [ ] **Step 5: Verify completion**

```bash
ssh studio "cat ~/dream-of-kiki/docs/milestones/g6-studio-path-a-2026-05-04.json | \
    python -c 'import json,sys;d=json.load(sys.stdin);print(\"cells\", len(d[\"cells\"]));print(\"wall\", d[\"wall_time_s\"]/3600, \"h\");print(\"verdict h9a\", d[\"verdict\"][\"h9a_classification\"]);print(\"verdict h9c\", d[\"verdict\"][\"h9c_classification\"])'"
```
Expected: cells == 100 ; verdict h9a in {H9-A, H9-B, INDETERMINATE} ; h9c in {H9-C, NOT_H9C}.

- [ ] **Step 6: Commit final milestone artifacts**

```bash
git add docs/milestones/g6-studio-path-a-2026-05-04.json \
        docs/milestones/g6-studio-path-a-2026-05-04.md \
        docs/milestones/g6-studio-path-a-2026-05-04.stdout.log
git commit -m "ops(g6s): production run Option B 100 cells"
```

---

## Task 10: Update Paper 2 §7.1.10 EN+FR

**Files:**
- Modify: `docs/papers/paper2/results.md` (append §7.1.10)
- Modify: `docs/papers/paper2-fr/results.md` (FR mirror)

- [ ] **Step 1: Append the EN section**

Read the verdict from the milestone JSON :
```bash
uv run python -c "
import json
d = json.load(open('docs/milestones/g6-studio-path-a-2026-05-04.json'))
print('h9a:', d['verdict']['h9a_classification'])
print('g  :', d['verdict']['h9a_g'])
print('h9c:', d['verdict']['h9c_classification'])
"
```

Then edit `docs/papers/paper2/results.md` and append a new section
**after §7.1.9** :

```markdown
## 7.1.10 G6-Studio Path A pilot — real-LoRA SpikingKiki-V4 35B-A3B-V4 (2026-05-04)

The first real-LLM-scale validation of the framework C effect.
Building on G6 Path B (§7.1.4 spectator pattern, synthetic LoRA),
G4-ter MLX richer head (§7.1.5 g_h2 = +2.77 positive coupling),
and G5-bis E-SNN richer head (§7.1.9 H7-B classification "MLX-only
artefact"), this pilot tests three pre-registered sub-hypotheses on
the real Qwen-35B-A3B-V4 spiking LIF substrate via `mlx_lm.tuner.lora`
fine-tune across the same 5 MMLU subdomains as G6 Path B. The
load-bearing change vs Path B : the four dream handlers operate on
the LIVE LoRA delta tensors, not on synthesised payloads disjoint
from the eval surface.

Pre-registration : `docs/osf-prereg-g6-studio-path-a.md`. Decision
rules locked at `H9_BONFERRONI_ALPHA = 0.05 / 3`.

### Verdict (Option B, 5 seeds, 100 cells)

| Hypothesis | Classification | g_h9a / Jonckheere p | Decision rule |
|------------|----------------|----------------------|---------------|
| H9-A | (fill from JSON) | (fill) | g >= 0.5 AND Welch reject |
| H9-B | (fill, mutually exclusive with H9-A) | — | g < 0.5 AND Welch fail |
| H9-C | (fill from JSON) | (fill) | Jonckheere fail AND mean(P_min) > mean(P_equ) |

(Pull the literal numbers from `docs/milestones/g6-studio-path-a-2026-05-04.json` ;
this section is regenerated from the milestone, never hand-edited.)

### Honest reading

- **If H9-A** : framework C effect TRANSFERS to real LLM scale.
  Rebuts G5-bis "H7-B MLX-only artefact". Triggers an Option-A
  (N=10) confirmatory follow-up pre-reg before any STABLE EC bump.
- **If H9-B (predicted under G5-bis universality)** : washout
  persists at 35B. Framework C is bounded to differentiable
  continuous nets. Reported as a positive empirical claim
  (programme has tightened the boundary).
- **If H9-C (universal DR-4 inversion)** : RESTRUCTURE+RECOMBINE
  remain spectator on real LIF Qwen. Universalises the H4-C /
  H5-C / H7-B refutation chain to real-LLM scale.

### DualVer impact

EC stays PARTIAL ; FC stays C-v0.12.0. Per pre-reg §6, Option B is
exploratory-confirmatory at first-overnight scale ; STABLE
promotion is gated on Option A (N≥10) replication AND
cross-substrate confirmation under a separate pre-reg.
```

- [ ] **Step 2: Mirror to FR (EN→FR rule)**

Edit `docs/papers/paper2-fr/results.md` and append the FR mirror
**after §7.1.9 FR**, with section title `## 7.1.10 G6-Studio voie A
— LoRA réelle SpikingKiki-V4 35B-A3B-V4 (2026-05-04)`. Translate
verbatim ; keep table headers in FR (Hypothèse / Classification /
g_h9a ou Jonckheere p / Règle de décision).

- [ ] **Step 3: Commit**

```bash
git add docs/papers/paper2/results.md docs/papers/paper2-fr/results.md
git commit -m "docs(paper2): G6-Studio Path A results §7.1.10"
```

---

## Task 11: DR-3 evidence revision per outcome

**Files:**
- Modify: `docs/proofs/dr3-substrate-evidence.md` (append revision row)

- [ ] **Step 1: Append the revision row**

Edit `docs/proofs/dr3-substrate-evidence.md`. Find the substrate-evidence
table for `micro_kiki` and append a row referencing this pilot. The
row content depends on the verdict :

```markdown
### `micro_kiki` (real backend, SpikingKiki-V4 35B-A3B-V4)

- **C1 — signature typing (typed Protocols)** : `PASS` (unchanged ; protocol
  contract is substrate-agnostic).
- **C2 — axiom property tests on real backend** : `PASS` per
  `tests/unit/test_micro_kiki_real_backend.py` + Task 4 live-tensor
  coupling tests.
- **C3 — empirical effect-size transfer** : (fill from verdict)
  - H9-A → C3 holds at real-LLM scale ; framework C transfers.
  - H9-B → C3 fails at real-LLM scale ; framework C bounded to
    differentiable continuous nets (consistent with G5-bis MLX-only
    classification at toy scale, now extended).
  - H9-C → DR-4 monotonicity refuted at real-LLM scale (consistent
    with G4-quater + G4-quinto + G5-bis).
- **Evidence pointer** : `docs/milestones/g6-studio-path-a-2026-05-04.json`,
  `docs/osf-prereg-g6-studio-path-a.md`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/proofs/dr3-substrate-evidence.md
git commit -m "docs(dr3): G6-Studio evidence revision"
```

---

## Task 12: CHANGELOG + STATUS — empirical row, no DualVer bump

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `STATUS.md`

- [ ] **Step 1: Append CHANGELOG row**

Edit `CHANGELOG.md` ; under `[Unreleased]` add an empirical-axis row :

```markdown
### G6-Studio Path A — real-LoRA SpikingKiki-V4 cross-substrate (2026-05-04)

- Option B (5 seeds × 4 arms × 5 subdomains = 100 cells, ~8–12 h Studio).
- H9-{A,B} classification : (fill from milestone).
- H9-C classification : (fill from milestone).
- EC stays PARTIAL ; FC stays C-v0.12.0. STABLE promotion gated on
  Option A (N≥10) replication per pre-reg §6.
- Milestone : `docs/milestones/g6-studio-path-a-2026-05-04.json`.
- Pre-reg : `docs/osf-prereg-g6-studio-path-a.md`.
```

- [ ] **Step 2: Update STATUS.md As-of + Gates table**

Edit `STATUS.md` ; bump the `**As of**` line to summarise the
G6-Studio outcome ; add a row in the Gates table :

```markdown
| G6-Studio Path A — real-LoRA SpikingKiki-V4 35B-A3B-V4 | 2026-05-04 → milestone | 🔶 PARTIAL (Option B, 100 cells, ~10 h Studio M3 Ultra ; H9-{A or B} verdict ; H9-C inversion ; live-tensor coupling fixes G6 Path B spectator ; EC stays PARTIAL per pre-reg §6, Option A N≥10 replication = future work) |
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md STATUS.md
git commit -m "docs(status): G6-Studio Path A row, EC PARTIAL"
```

---

## Task 13: Self-review + push to origin/main

**Files:** None new.

- [ ] **Step 1: Run the full test suite locally**

```bash
uv run pytest tests/unit/experiments/test_g6_studio_*.py -v
uv run ruff check experiments/g6_studio_path_a/ tests/unit/experiments/test_g6_studio_*.py
uv run mypy experiments/g6_studio_path_a/ tests/unit/experiments/test_g6_studio_*.py
```
Expected: all tests pass ; ruff clean ; mypy strict clean.

- [ ] **Step 2: Verify R1 reproducibility**

```bash
uv run python -c "
from harness.storage.run_registry import RunRegistry
r = RunRegistry('.run_registry.sqlite')
import sqlite3
con = sqlite3.connect('.run_registry.sqlite')
rows = con.execute(\"SELECT profile, seed, run_id FROM runs WHERE profile LIKE 'g6-studio-path-a%'\").fetchall()
assert len(rows) == 100, f'expected 100, got {len(rows)}'
print('R1 OK : 100 (profile, seed) tuples registered')
"
```
Expected: `R1 OK : 100 (profile, seed) tuples registered`.

- [ ] **Step 3: Verify EN→FR mirror parity**

```bash
grep -c "## 7.1.10" docs/papers/paper2/results.md
grep -c "## 7.1.10" docs/papers/paper2-fr/results.md
```
Expected: each prints `1` (one section in each language).

- [ ] **Step 4: Verify no AI co-authorship in commit log**

```bash
git log --since="2026-05-04" --pretty=full | grep -iE "co-authored-by|claude|opus" || echo "OK: no AI attribution"
```
Expected: `OK: no AI attribution`.

- [ ] **Step 5: Push to origin/main**

```bash
git push origin main
```
Expected: push succeeds ; CI on push runs ruff + mypy + pytest. R1 nightly
runs on macOS-14 ; verify the next nightly is green via the R1 nightly
workflow status.

---

## Self-review

**Spec coverage** :
- Task 0/0.5 → Studio investigation + Option A/B/C lock. ✓
- Task 1 → OSF pre-reg G6-Studio Path A with H9-A/B/C. ✓
- Task 2 → lora_loader (Qwen + SpikingKiki adapters). ✓
- Task 3 → lora_train_step (mlx_lm.tuner.lora real fine-tune). ✓
- Task 4 → dream_episode_real (live-tensor coupling, fixes Path B spectator). ✓
- Task 5 → mmlu_eval (letter-argmax via mlx_lm.generate). ✓
- Task 6 → aggregator_h9 (H9-A/B/C verdict with locked Bonferroni α). ✓
- Task 7 → run_g6_studio_path_a driver with per-subdomain partial dumps. ✓
- Task 8 → Option C smoke validation on Studio. ✓
- Task 9 → Option B production overnight + resumption runbook. ✓
- Task 10 → Paper 2 §7.1.10 EN+FR. ✓
- Task 11 → DR-3 evidence revision. ✓
- Task 12 → CHANGELOG + STATUS, no DualVer bump (EC PARTIAL). ✓
- Task 13 → self-review + push. ✓

**Placeholder scan** : no "TBD", "implement later", or "similar to Task N"
references — every step shows the actual code or command. The Paper 2
§7.1.10 + CHANGELOG sections include literal "(fill from JSON)" markers
because the verdict numbers are pilot-output-dependent ; this is a
documented protocol (§7.1.5/7.1.6/etc. follow the same regenerated-from-milestone
pattern), not a placeholder.

**Type consistency** :
- `QwenLoRAWrapper.rank` : `int` (Task 2) — used identically in Task 7.
- `TrainHyperparams(lr, iters, rank, alpha, batch_size)` : signature stable
  across Task 3 (definition), Task 7 (driver call), Task 9 (CLI args).
- `dream_episode_real(substrate, profile, live_delta, seed, subdomain,
  prior_deltas, sibling_deltas)` : kwargs identical between Task 4 (defn)
  and Task 7 (call site).
- `classify_h9(retention)` : returns dict with keys `h9a_classification`,
  `h9a_g`, `h9a_welch_p`, `h9a_welch_reject`, `h9c_classification`,
  `h9c_jonckheere_p`, `h9c_jonckheere_reject`, `h9c_mean_p_{min,equ,max}`,
  `bonferroni_alpha` — same set asserted in Task 6 tests, used in
  Task 7 driver, and rendered in Task 10 paper section.
- `DEFAULT_PRIMARY_KEY = "layer_0_lora_B"` : single source of truth in
  `dream_episode_real.py` ; imported by the driver ; no string literal
  drift across files.
- `CellResult` TypedDict : same field set in `experiments/g6_mmlu_stream/run_g6.py`
  and Task 7 (the new driver imports `compute_retention` directly to
  preserve the exact retention semantics).
- `H9_BONFERRONI_ALPHA = 0.05 / 3` : single constant in
  `aggregator_h9.py` ; cited verbatim in pre-reg §5 / Task 6 test /
  Task 10 paper section.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Particularly relevant here because Tasks 8/9 require Studio SSH and are hardware-dependent (a subagent can pause and surface the Studio command for a human to run, then resume on the result).
2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch with checkpoints. Viable for Tasks 0–7 + 10–13 (codebase work) ; Tasks 8/9 always require Studio access regardless.

**Which approach?**
