# Wave 3b M5 — Full bench run spec

> **Parent plan** : `docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md` §4 M5.
> **OSF amendment** : Q6JYN-W3B (draft `docs/osf-amendment-wave3b.md`, not yet filed).
> **Status** : spec written 2026-05-21 ; **revised 2026-05-21 post
> codebase audit** (see §9 — corrects RunRegistry API, canonical-hash
> precedent, `execute_profile` signature, CLI parser, version bump).
> Plan + launch in future session. **No code today.**

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

- Drop the `[plan-only]` early-exit (lines 364-378). Per-cell
  execution path, using the **actual** APIs (see §9 audit) :

  1. `loader = load_split_cifar100(task_idx, batch_size=…, seed=seed,
     smoke=False)` (prod path from D2) — returns a single-use
     iterable of `SplitCifar100Batch`.
  2. Extend `_CellRequest` with a `loader_batches` field
     (materialised `tuple[SplitCifar100Batch, ...]`) so the substrate
     can consume it. `_CellRequest` stays a plain `@dataclass`.
  3. `metrics = substrate.execute_profile(cell_request)` — the
     existing signature is `execute_profile(self, request)`. M5 work
     (§4 touch map) makes the substrate **consume** `request.
     loader_batches` instead of building its own synthetic latents.
  4. `output_hash = _r1_hash_metrics(metrics)` — reuse the existing
     canonical-bytes helper in `ablation_cycle3_diffusion.py`
     (`json.dumps(..., sort_keys=True, default=repr)` + sha256,
     `wall_time_s` excluded). There is **no** `_canonicalize_output`
     in `ablation_cycle2.py`.
  5. `run_id = registry.register(c_version, profile, seed,
     commit_sha)` then `registry.register_output_hash(run_id,
     output_hash)`. `RunRegistry` has **no** `register_run` /
     `query` method ; `register` is `INSERT OR IGNORE` (idempotent)
     and returns the run_id.
- **Resume support** : `--resume` is currently parsed by `_parse_cli`
  but **never consumed** (dead flag). M5 wires it : before executing
  a cell, compute `run_id` via the registry's `_compute_run_id`
  tuple, call `registry.get_output_hash(run_id)` and **skip the cell
  on success** ; a `KeyError` means not-yet-done → execute. Crash-safe
  over 6-8 h wall.
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

**Two-level dump.** Per-cell records carry the *raw substrate
metrics* ; H1 / H2 / H4 are *aggregate stat-test verdicts* computed
at the milestone step over the 150 cells/profile — they are **not**
per-cell floats (the `ablation_cycle2.py` H1-H4 are
`welch`/`tost`/`threshold` verdicts run once over accuracy arrays).

Per-cell record — `run_id` + `output_hash` in `RunRegistry`, the
full row appended to a per-cell JSONL sidecar
(`docs/milestones/wave3b-bench-2026-MM-DD.cells.jsonl`) :

| Field | Type | Source |
|---|---|---|
| `run_id` | str (32-hex) | `sha256(c_version\|profile\|seed\|commit_sha)[:32]` |
| `c_version` | str | `"C-v0.14.0+PARTIAL"` (substrate-internal) |
| `profile` | str | `"p_min" \| "p_equ" \| "p_max"` (registry tag form) |
| `seed` | int | 0..29 |
| `task_idx` | int | 0..4 |
| `output_hash` | str (sha256) | `_r1_hash_metrics(metrics)` |
| `replay_rate` | float | substrate `metrics["replay_rate"]` |
| `downscale_norm` | float | substrate `metrics["downscale_norm"]` |
| `restructure_sum` | float | substrate `metrics["restructure_sum"]` |
| `recombine_rate` | float | substrate `metrics["recombine_rate"]` |
| `delta_acc` | float | substrate `metrics["delta_acc"]` |
| `wall_s` | float | per-cell wall time |
| `peak_rss_mb` | int | `resource.getrusage` |
| `slow_cell` | bool | `wall_s > 120` |

Aggregated milestone JSON (`docs/milestones/wave3b-bench-2026-MM-DD.json`) :

- Per-profile descriptive stats (mean, std, 95 % CI bootstrap) on
  each raw metric over the 30 × 5 = 150 cells / profile.
- H1 / H2 / H4 verdicts per profile : reuse the `welch_one_sided` /
  `tost_equivalence` / `one_sample_threshold` primitives from
  `kiki_oniric.eval.statistics` (Bonferroni α = 0.0125), fed the
  real per-cell metric arrays — replacing the synthetic stand-ins
  (`p_max_smoke`, hard-coded `energy_ratios`) used in
  `ablation_cycle2._run_h1_h4`.
- Cross-substrate consistency table : `mlx_latent_diffusion` vs
  `mlx_kiki_oniric` per H1 / H2 / H4, following the
  `ablation_cycle2._cross_substrate_consistency` verdict-agreement
  pattern (`agree = len(set(verdicts)) == 1`).
- Underperforming-baseline rule check (Paper 2 §6) : if any profile
  loses to `mlx_kiki_oniric` by > δ on H1 *and* H2, that profile's
  150 cells are *excluded* from the bench and the JSON records the
  exclusion with rule citation.

### D7 — Acceptance criteria (machine-checkable)

1. **Cell count** : iterating the 450-cell grid and calling
   `registry.get_output_hash(run_id)` returns a non-`KeyError` hash
   for all 450 cells (OR 300 if one profile is excluded under the
   §6 rule). `RunRegistry` has no `query` method — the count is the
   number of grid cells whose `run_id` resolves to a stored hash.
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
| `harness/diffusion_eval/cifar100_split_loader.py` | Replace the prod-path `FileNotFoundError` (lines 244-252) with the HF `datasets` materialisation + per-task partition + batch-perm | +120 |
| `scripts/ablation_cycle3_diffusion.py` | Patch `DEFAULT_SEEDS` to `tuple(range(30))` per D1 | ±1 |
| `scripts/ablation_cycle3_diffusion.py` | Extend `_parse_cli` with `--num-seeds`, `--output`, `--task-idx` (currently only `--smoke/--resume/--dry-run/--max-runs`) | +25 |
| `scripts/ablation_cycle3_diffusion.py` | Extend `_CellRequest` with a `loader_batches` field | +3 |
| `scripts/ablation_cycle3_diffusion.py` | Replace the `[plan-only]` block (lines 364-378) with the prod execute loop : materialise loader → `execute_profile` → `_r1_hash_metrics` → `register` + `register_output_hash` → JSONL sidecar append ; wire `--resume` skip via `get_output_hash` | +90 −15 |
| `kiki_oniric/substrates/mlx_latent_diffusion.py` | `execute_profile` consumes `request.loader_batches` instead of building synthetic latents (single file, **not** a package dir) | +20..40 |
| `harness/storage/run_registry.py` | No change — `register` + `register_output_hash` + `get_output_hash` already cover the M5 needs | 0 |
| `tests/unit/diffusion_eval/__init__.py` | New (directory does not exist yet) | +0 |
| `tests/unit/diffusion_eval/test_cifar100_prod_loader.py` | New : 1 happy + 1 offline-fallback + 1 determinism test | +120 |
| `tests/unit/scripts/test_ablation_cycle3_resume.py` | New : 1 resume-skip test (`tests/unit/scripts/` already exists) | +60 |
| `docs/milestones/wave3b-bench-2026-MM-DD.{json,md,cells.jsonl}` | Generated by the run | (output) |
| `CHANGELOG.md` | `[Unreleased]` Empirical bullet | +3 |
| `pyproject.toml` | SemVer bump **0.22.1 → 0.22.2** (disk is already at 0.22.1 ; no FC bump, EC bench only) | ±1 |

No framework-C spec change (`docs/specs/2026-04-17-…`) — M5 is
purely empirical-axis, no axiom touched. The MLX-only test paths
(`tests/unit/diffusion_eval/`) must be added to the root
`conftest.py` `collect_ignore_glob` so Linux CI skips them
(per CLAUDE.md PR #28 convention).

## 9. Audit corrections (2026-05-21, post codebase read)

The first draft of this spec assumed APIs that do not exist. The
following were corrected against the disk state :

1. **`RunRegistry`** has no `query` / `register_run`. The real API
   is `register(c_version, profile, seed, commit_sha) -> run_id`
   (idempotent `INSERT OR IGNORE`), `register_output_hash`,
   `get_output_hash` (raises `KeyError` if absent). Resume + cell
   count are built on the `KeyError` existence check. D3, D7 fixed.
2. **Canonical hashing** : there is no `_canonicalize_output` in
   `ablation_cycle2.py` (that file does no output hashing). The
   precedent is `_r1_hash_metrics` already in
   `ablation_cycle3_diffusion.py`. D3, D6 fixed.
3. **`execute_profile`** signature is `execute_profile(self,
   request)` and is synthetic-driven — it ignores any loader. M5
   couples it via a new `_CellRequest.loader_batches` field. D3,
   §4 fixed.
4. **CLI** : the runner uses a hand-rolled `_parse_cli`, not
   argparse. `--num-seeds`, `--output`, `--task-idx` do not exist
   and must be added ; `--resume` is parsed but never consumed.
   D1, D3, D4, §4 fixed.
5. **`pyproject.toml`** is already at `0.22.1` on disk (CLAUDE.md
   text is stale). M5 bumps to `0.22.2`. §4 fixed.
6. **`mlx_latent_diffusion`** is a single module file, not a
   package directory. §4 fixed.
7. **H1 / H2 / H4** in `ablation_cycle2._run_h1_h4` are aggregate
   stat-test verdicts over accuracy arrays (with synthetic
   stand-ins for H2/H4), not per-cell floats. D6 split into a
   per-cell raw-metric record + an aggregate verdict step.

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
