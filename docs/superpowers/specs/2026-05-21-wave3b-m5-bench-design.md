# Wave 3b M5 — Full bench run spec

> **Parent plan** : `docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md` §4 M5.
> **OSF amendment** : Q6JYN-W3B (draft `docs/osf-amendment-wave3b.md`, not yet filed).
> **Status** : spec written 2026-05-21, awaiting user review. Plan + launch in
> future session. **No code today.**

## 1. Goal

Ship the full ablation bench for the `mlx_latent_diffusion` substrate
on Studio M3 Ultra : **1 substrate × 3 profiles × N=30 seeds × 5
CIFAR-100 task-splits = 450 cells**, registered in `RunRegistry`,
hashed for R1, and summarised in a deterministic milestone dump.

Acceptance (verbatim from plan §4 M5) :

- 450/450 cells included (or excluded with rule citation per the
  underperforming-baseline rule, Paper 2 §6).
- H1 / H2 / H4 verdicts reported per profile.
- Cross-substrate consistency check vs `mlx_kiki_oniric` + `esnn_norse`
  per the `ablation_cycle2.py` pattern.

## 2. What is missing today (audited 2026-05-21)

Two production blockers in the M4 deliverable :

1. **Loader prod path raises.** `harness/diffusion_eval/cifar100_split_loader.py:248`
   raises `FileNotFoundError("…not implemented in M4")` whenever
   `smoke=False`. Synthetic latents (smoke path) are the *only*
   working path today.
2. **Runner exits before per-cell execution.** `scripts/ablation_cycle3_diffusion.py`
   (lines 364-378) registers the envelope, enumerates the grid,
   and exits with `"[plan-only] per-cell execution deferred to M5"`.

Plus one *minor reconciliation* :

3. **Seed count drift.** Plan §4 M5 says N=30 ; script
   `DEFAULT_SEEDS = tuple(range(60))` says 60. Resolution :
   **N=30 is authoritative** (plan §4 acceptance criterion), launch
   passes `--num-seeds 30` and the default constant gets a comment
   pointing at the plan. No code change required if the CLI flag is
   wired ; if not, patch the default to 30 in M5 Task 1.

## 3. Decisions (no clarifying questions per user directive)

### D1 — Seed count

- **N = 30** seeds. Plan §4 is authoritative.
- `DEFAULT_SEEDS = tuple(range(30))` ; remove the `range(60)` over-spec.
- CLI flag `--num-seeds 30` remains supported for ad-hoc reduction
  during smoke / debugging (≤ N=30 only ; > 30 is unspecified).

### D2 — Loader prod path source

- **HuggingFace `datasets`** library, repo
  `uoft-cs/cifar100` (the canonical mirror). Cached under
  `HF_HOME` (env override supported).
- Materialisation : load once into a numpy uint8 array (50 000 × 32
  × 32 × 3 train + 10 000 test), then partition by fine-label index
  modulo `N_TASKS=5` into 5 disjoint 20-class windows.
- Output : `SplitCifar100Batch` instances with
  `features: mx.array[float32, (B, RAW_FEATURE_DIM=3072)]`,
  `labels: mx.array[int32, (B,)]` mapped to the task-local 0..19
  range, `task_idx: int` ∈ 0..4.
- Train / val split : per-task `PROD_N_TRAIN_PER_TASK = 5_000`
  (already a constant) drawn from the CIFAR-100 train split ;
  validation drawn from the CIFAR-100 test split, sliced per task.
- Determinism : an inner-key derivation `_derive_task_keys(task,
  seed)` already exists ; the prod path adds a **batch-order
  permutation** keyed off `(task, seed, "perm")` so the same seed
  reproduces the same batch ordering byte-for-byte.
- Offline fallback : if `HF_HUB_OFFLINE=1` and the dataset is not
  cached, raise `FileNotFoundError` with the exact `huggingface-cli
  download uoft-cs/cifar100` command in the message. No silent
  torchvision fallback (would diverge from the bench README).

### D3 — Runner prod execution branch

- Drop the `[plan-only]` early-exit. Per-cell execution path :

  1. `loader = load_split_cifar100(task_idx, batch_size=…, seed=seed,
     smoke=False)` (prod path from D2).
  2. `cell_request = _CellRequest(profile, seed, task_idx, …)`.
  3. `output = substrate.execute_profile(profile, loader, seed=seed,
     task_idx=task_idx)`.
  4. `output_hash = sha256(canonical_bytes(output))` — canonical
     bytes follow the `mlx_kiki_oniric` precedent
     (`ablation_cycle2.py` `_canonicalize_output`).
  5. `RunRegistry.register_run(run_id, …, output_hash=output_hash,
     metrics=output.metrics)`.
- **Resume support** : `--resume` flag (already partially scaffolded
  in `RunRegistry`). On startup, skip any `(c_version, profile, seed,
  commit_sha)` already present in the registry with non-null
  `output_hash`. Crash-safe over 6-8 h wall.
- **Memory cap** : each cell drops the loader iterable + the
  substrate's per-cell tensors before moving to the next cell. Peak
  RSS budget per cell ≤ 1 GB (target ≤ 600 MB ; see §5 wall-clock
  budget).

### D4 — Studio launch strategy

- **tmux session** `wave3b-m5-bench` on Studio M3 Ultra.
- Inside the session : `nohup uv run python
  scripts/ablation_cycle3_diffusion.py --num-seeds 30 --resume
  --output docs/milestones/wave3b-bench-2026-05-22.json 2>&1 |
  tee logs/wave3b-m5-bench.log`.
- Watchdog : `tail -F logs/wave3b-m5-bench.log` from the controller
  laptop via `ssh studio` ; per-cell progress line emitted after
  each `RunRegistry.register_run`.
- Studio prerequisites (verify *before* launch — see §6
  pre-flight) : `HF_HOME` set, CIFAR-100 cached, `uv sync
  --all-extras` ran cleanly, `git status` clean, current commit
  pinned in milestone metadata.

### D5 — Per-cell wall-clock budget

- Target : **50 s / cell average**, ⇒ 450 × 50 s ≈ **6.25 h** wall.
- Upper bound : 80 s / cell ⇒ 10 h wall (8 h plan §4 estimate + 25 %
  buffer). If a single cell exceeds 120 s, register a `slow_cell`
  flag and continue — do not abort.
- Memory peak : **≤ 1 GB RSS / cell** (Studio M3 Ultra has 512 GB ;
  this is for the substrate's tensor footprint, not the host).

### D6 — What to dump per cell

Per-cell record (one row per cell in the RunRegistry, plus
aggregated in the JSON milestone) :

| Field | Type | Source |
|---|---|---|
| `run_id` | str (32-hex) | `sha256(c_version|profile|seed|commit_sha)[:32]` |
| `c_version` | str | `"C-v0.14.0+PARTIAL"` (substrate-internal) |
| `profile` | str | `"P_min" | "P_equ" | "P_max"` |
| `seed` | int | 0..29 |
| `task_idx` | int | 0..4 |
| `output_hash` | str (sha256) | canonical-bytes(output) |
| `H1_replay_distinctness` | float | per ablation_cycle2 pattern |
| `H2_downscale_rank_decay` | float | per ablation_cycle2 pattern |
| `H4_recombine_novelty` | float | per ablation_cycle2 pattern |
| `wall_s` | float | per-cell wall time |
| `peak_rss_mb` | int | resource.getrusage |
| `slow_cell` | bool | wall_s > 120 |
| `n_dispatch` | int | runtime.log size before reset |

Aggregated milestone JSON (`docs/milestones/wave3b-bench-2026-MM-DD.json`) :

- Per-profile H1 / H2 / H4 summary statistics (mean, std, 95 % CI
  bootstrap over the 30 × 5 = 150 cells / profile).
- Cross-substrate consistency table : `mlx_latent_diffusion` vs
  `mlx_kiki_oniric` per H1 / H2 / H4 (registered runs reused from
  the `ablation_cycle2.py` artefacts). Delta + sign per profile.
- Underperforming-baseline rule check (Paper 2 §6) : if any profile
  loses to `mlx_kiki_oniric` by > δ on H1 *and* H2, that profile's
  150 cells are *excluded* from the bench and the JSON records the
  exclusion with rule citation.

### D7 — Acceptance criteria (machine-checkable)

1. **Cell count** : `len(RunRegistry.query(c_version="C-v0.14.0+PARTIAL",
   substrate="mlx_latent_diffusion")) == 450` OR == 300 if one
   profile is excluded under §6 rule.
2. **Hash completeness** : every selected run has a non-null
   `output_hash`.
3. **R1 nightly check** : same `(profile, seed=0, task_idx=0)` cell
   re-run on Studio reproduces its registered `output_hash`
   byte-for-byte. (Single-machine R1, per the cross-machine R1
   posture documented in `docs/proofs/r1-cross-machine.md`.)
4. **Milestone determinism** : two consecutive runs of the
   aggregation step from the same registry produce identical
   `wave3b-bench-2026-MM-DD.json` bytes.
5. **Consistency table** : `mlx_kiki_oniric` reference cells exist
   and the delta column is populated for all 3 profiles × 3
   hypotheses.

## 4. File touch map (M5 implementation plan)

The actual implementation lives in a future `writing-plans`
deliverable. For this spec, the touch map is :

| File | Action | LOC est. |
|---|---|---|
| `harness/diffusion_eval/cifar100_split_loader.py` | Add prod branch (lines 244-252 currently raise) | +120 |
| `harness/diffusion_eval/__init__.py` | Re-export prod loader if not already | +1 |
| `scripts/ablation_cycle3_diffusion.py` | Drop early-exit lines 364-378, add execute loop, `--resume`, `--output` | +90 −15 |
| `scripts/ablation_cycle3_diffusion.py` | Patch `DEFAULT_SEEDS` to `range(30)` per D1 | ±1 |
| `harness/storage/run_registry.py` | Confirm `output_hash` column + `--resume` query (likely already there) | +0..20 |
| `kiki_oniric/substrates/mlx_latent_diffusion/` | Wire `execute_profile(loader, …)` if not already | +20..40 |
| `tests/unit/diffusion_eval/test_cifar100_prod_loader.py` | New : 1 happy + 1 fallback + 1 determinism test | +120 |
| `tests/unit/scripts/test_ablation_cycle3_resume.py` | New : 1 resume test | +60 |
| `docs/milestones/wave3b-bench-2026-MM-DD.{json,md}` | Generated by the run | (output) |
| `CHANGELOG.md` | `[Unreleased]` Empirical bullet | +3 |
| `pyproject.toml` | SemVer bump 0.22.0 → 0.22.1 (no FC bump, EC bench only) | ±1 |

No framework-C spec change (`docs/specs/2026-04-17-…`) — M5 is
purely empirical-axis, no axiom touched.

## 5. Pre-flight checklist (Studio M3 Ultra)

To run *immediately before* launching the bench (future session) :

- [ ] `cd ~/Documents/Projets/dream-of-kiki && git status` is clean.
- [ ] `git log -1 --format="%H"` matches the run-registry
      `commit_sha` field.
- [ ] `uv sync --all-extras` exits 0.
- [ ] `uv run pytest tests/unit/diffusion_eval -v` green.
- [ ] `uv run pytest tests/reproducibility/ -v --no-cov` green
      within-machine on Studio.
- [ ] `HF_HOME` set, `huggingface-cli scan-cache | grep cifar100`
      shows the dataset locally cached.
- [ ] Smoke run : `uv run python scripts/ablation_cycle3_diffusion.py
      --num-seeds 1 --task-idx 0` exits 0 in < 5 min (9-cell smoke).
- [ ] Disk free ≥ 50 GB on `/` and on the HF cache filesystem.
- [ ] tmux session `wave3b-m5-bench` created, but not yet running
      the bench.

## 6. Out of scope for M5

- M6 paper §7.9 drafting (separate milestone).
- DualVer flip to +STABLE (gated on §12.3, decided at M6).
- OSF amendment Q6JYN-W3B filing (M6).
- Cross-machine R1 for the new diffusion R1 entries (per `docs/
  proofs/r1-cross-machine.md` posture, deferred to M6 nightly).
- Adding new R1 golden hashes — M3 already shipped those ; M5 does
  not touch `golden_hashes.json`.

## 7. Risks (M5-specific extracts from plan §5)

| Ref | Risk | Mitigation in M5 |
|---|---|---|
| R1 (cross-machine R1 cracks on new entries) | `mx.random.normal` raw-key M1 Max divergence | Document, do not block (#3568). M5 is single-machine on Studio. |
| R2 (M5 GrosMac 16 GB OOM) | n/a in M5 (Studio only) | — |
| R4 (ship-critic DO-NOT-SHIP) | M5 does not ship to +STABLE | — (M6 concern) |
| R5 (D collapse) | Diffusion model fails to learn | M3 already gate-passed sanity. If H1/H2/H4 are degenerate → §6 underperforming-baseline rule triggers per-profile exclusion. |

## 8. Self-review

- **Placeholder scan** : no TBD, no TODO, no "implement later" left
  in the spec. The "future session" boundary is explicit (§1, §5).
- **Internal consistency** : D1 (N=30) ↔ §1 (450 cells) ↔ D7 acceptance
  ↔ §4 touch map (`range(30)` patch) all align.
- **Scope check** : single substrate, single milestone, single
  deliverable JSON. Fits one writing-plans plan.
- **Ambiguity check** : D6 cell record schema, D7 acceptance, §4
  touch map are explicit. The only deliberate ambiguity is the
  `mlx_kiki_oniric` reference cells in D6 — they are assumed to
  exist in the registry from `ablation_cycle2.py` artefacts. M5
  pre-flight should `RunRegistry.query` to verify before launch ;
  if missing, M5 budget needs +1 h to re-run the reference cells.
