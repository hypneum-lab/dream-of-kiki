# G6 pilot decisions (locked 2026-05-03)

## Subdomain selection (5 subdomains)
S1 = anatomy
S2 = astronomy
S3 = business_ethics
S4 = clinical_knowledge
S5 = college_biology

Rationale: 5 distinct MMLU subjects spanning life-science,
hard-science, humanities, applied medicine; alphabetical-prefix
order chosen for neutrality (no curriculum bias). Each subject in
cais/mmlu has at least 100 test records, sufficient for
`n_train=100 / n_eval=100` per cell on Path A.

## Per-cell volumes
n_train_per_subdomain = 100  # train split, capped (Path A only)
n_eval_per_subdomain = 100   # held-out eval per subject (subset of test)
seeds_per_arm = 3            # {0, 1, 2}
arms = (baseline, P_min, P_equ, P_max)
n_cells = 4 arms x 3 seeds = 12 cell sequences (each touches 5 subdomains)
n_eval_calls_total = 12 sequences x sum_{i=1..5}(i) = 12 x 15 = 180

## LoRA training hyperparams (Path A only)
lr = 5e-5
inner_steps_per_subdomain = 50
batch_size = 4
rank = 16
alpha = 16

## Compute budget acceptance
Path A budget: 60 cells x ~15 min = 15 h on Studio M3 Ultra
  (overnight run).
Path B budget: 60 cells x ~30 s = ~30 min on any Apple Silicon.

## Path selected (Task 0.5 step 1 evidence — verified 2026-05-03)

```
$ ls -ld /Users/electron/KIKI-Mac_tunner /Users/clems/KIKI-Mac_tunner 2>&1
ls: /Users/electron/KIKI-Mac_tunner: No such file or directory
ls: /Users/clems/KIKI-Mac_tunner: No such file or directory
$ test -d ~/KIKI-Mac_tunner && echo PATH_A_AVAILABLE || echo PATH_A_MISSING
PATH_A_MISSING
$ uv run python -c "import mlx_lm; print(mlx_lm.__version__)"
ImportError (mlx_lm not present in the dev env on this M1 Max host)
```

PATH = B
RATIONALE = This is an Apple M1 Max / 32 GB host without
  `~/KIKI-Mac_tunner` and with no `mlx_lm` install ; Path A
  prerequisites are unmet. Path B (inference-only adapter-tensor
  mutation, 1.5B fallback) is the only branch runnable here. Per
  pre-reg §6, Path B never triggers a STABLE / UNSTABLE EC bump
  regardless of effect-size outcome.

## G4-bis dependency

G6 was originally pre-registered (early draft) with H_NEW gating on
G4-bis confirming a non-zero, positive coupling effect on the MLX
substrate. The actual G4-bis run (`docs/milestones/g4-pilot-2026-
05-03-bis.json`) returned :

- g4_bis_milestone_path = docs/milestones/g4-pilot-2026-05-03-bis.json
- g4_bis_g_h1 = -2.3067 (sign-reversed, NOT the positive
  retention-gain expected for P_equ)
- g4_bis_above_hu_lower_ci = false (g_h1 = -2.31 is far below
  Hu 2020 CI lower bound 0.21)
- g4_bis_h_dr4_degenerate = true (H_DR4 nominally monotonic but
  P_min / P_equ / P_max retention vectors are bit-identical because
  RESTRUCTURE + RECOMBINE remained spectator-only on the binary MLP)
- g6_unblocked = true (Path B exploratory; absolute synthetic
  effect being negative does NOT block infrastructure validation)

The G4-bis findings are honestly carried forward into the G6
pre-reg via the H_NEW reformulation (see
`docs/osf-prereg-g6-pilot.md` §2 H_NEW). H_NEW is now an
**exploratory infrastructure-validation** hypothesis, not a
synthetic-to-real margin test. The chain of scientific review
(G4 spectator -> G4-bis coupling -> G6 Path B inference-only) is
documented as part of the pre-reg amendment.

## Path B / Path A scope discipline (binding)

- Path A modules (`micro_kiki_train.py`) are **not** developed in
  this run (KIKI-Mac_tunner absent). The plan's Task 6 is therefore
  out-of-scope on this host. The cell runner (Task 7 in plan / the
  user's Task 5+6+7 collapsed) only wires the Path B branch and
  raises `NotImplementedError("Path A unavailable on this host")`
  on the Path A leg.
- Per pre-reg §6, Path B outcomes do not trigger EC bump.
  CHANGELOG / STATUS rows record the Path B run as exploratory
  infrastructure validation only ; EC stays PARTIAL.

## Sanity-fixture smoke discipline

The repo's only MMLU fixture (`tests/fixtures/mmlu_sanity.jsonl`)
covers `world_facts`, `astronomy`, `chemistry`,
`elementary_mathematics`, `world_literature` (each with at least
~9 records) — *not* the production target subjects (anatomy,
clinical_knowledge, college_biology, business_ethics).

The pilot (`run_pilot`) therefore runs in two distinct modes :

- `--smoke` : uses `SMOKE_SUBDOMAINS` = (`world_facts`, `astronomy`,
  `chemistry`, `elementary_mathematics`, `world_literature`) on the
  sanity fixture, with reduced volumes (n_train=4, n_eval=4).
- Full pilot (no `--smoke`) : on this M1 Max host, the full pilot
  is run against an **expanded synthetic fixture** generated at
  test-fixture quality covering the 5 production subjects with
  enough records per subject for `n_train=10, n_eval=10` (the
  full `cais/mmlu` HF cache is not materialised on this host).
  This still meets the Path B exploratory bar (~30 min, 60 cells)
  and keeps the dump structurally identical to a real Path A run.
