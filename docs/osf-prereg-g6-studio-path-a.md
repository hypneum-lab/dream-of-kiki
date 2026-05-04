# G6-Studio Path A pilot pre-registration (daughter of G6 Path B)

**Date:** 2026-05-04
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**Sister pre-reg:** `docs/osf-prereg-g6-pilot.md` (Path B = synthetic spectator wrapper, milestone `docs/milestones/g6-pilot-pathB-2026-05-03.json`).
**Substrate:** real `SpikingKiki-35B-A3B-V4` spiking LIF Qwen substrate (58 GB, 31 070 `.npz` modules) at `/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4`. Loaded via `kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate.load()` Path 2 with `DREAM_MICRO_KIKI_REAL=1` env-gate (commit `177cf89`).
**Adapter stack:** `SpikingKiki-V4-adapters/` (real LoRA tensors). Per-subdomain `mlx_lm.tuner.lora` fine-tune step over the LoRA delta tensors ; dream operations mutate the **live** delta tensors (this fixes the G6 Path B "spectator wrapper" caveat).
**Benchmark:** MMLU 5-subdomain continual-learning stream (anatomy → astronomy → college_biology → college_chemistry → college_physics, 200 records each ; identical fixture to the existing Path B pilot for direct comparability).
**Compute:** N = 5 seeds × 4 arms × 5 subdomains = **100 cells**, target ~8-12 h overnight on Studio M3 Ultra. Path A run lives **only on Studio** (M1 Max OOM at 35B bf16). Driver writes per-subdomain partial JSON dumps so an overnight watchdog kill is resumable.
**Lock commit:** *(filled by introducing commit hash)*
**Lock timestamp:** 2026-05-04 (pre-driver-run).

## §1 Background

The G6 Path B pilot (commit `855` per session memory) wired the dream protocol to the real SpikingKiki-V4 substrate but **only as a spectator** — the four dream handlers ran on synthetic LoRA-shaped tensors and never touched the live adapter delta. The Path B pre-reg explicitly flagged this caveat. Path B reported a small g_h9 ≈ -0.06 with no signal (consistent with the spectator architecture).

Path A removes the spectator caveat : per subdomain, the four dream handlers operate on the **live** LoRA delta tensors emitted by `mlx_lm.tuner.lora`, then `mlx_lm.generate` evaluates letter-argmax MMLU on the dream-modified adapter. This is the first real-LLM-scale test of the framework C 4-channel coupling at full 35B bf16 scale.

The G4-ter §7.1.5 finding (g_h2 = + 2.77 on MLX richer head) is the only positive empirical claim of the empirical program at the time of locking ; G5-bis classified it as "MLX-only artefact" at toy E-SNN scale (H7-B confirmed). G6-Studio Path A tests the **converse** at real-LLM scale : does the positive REPLAY+DOWNSCALE effect re-emerge when the substrate is itself a real spiking LLM ?

The G4-{quater..septimo} ladder (commit `c8dd268`) closed the H6-C universality at the four-benchmark × four-CNN-or-MLP scope ceiling : RECOMBINE-empty across all classification escalation cells. That ceiling is **substrate-bounded** (DR-4 evidence v0.6) ; G6-Studio Path A is the first test outside the CNN scope at real-LLM tier.

The Hu 2020 anchor (g = 0.29) is used here strictly as a directional reference, not a magnitude calibrator (cross-class biological-vs-LLM magnitude calibration is a category error).

## §2 Hypotheses (confirmatory)

- **H9-A (real-LLM transfer of REPLAY+DOWNSCALE coupling)** — on the 5-subdomain MMLU CL stream with the real SpikingKiki-V4 35B-A3B-V4 substrate, `retention(P_max with mog)` is statistically distinguishable from `retention(P_max with none)` with predicted positive sign and `Hedges' g ≥ 0.5`. Test : Welch two-sided rejects H0 at α = 0.05 (single new test, no Bonferroni inheritance from the closed G4 ladder). **Rejecting H0 with the predicted sign and g ≥ 0.5** is the H9-A positive empirical claim — the framework's RECOMBINE prediction is restored at real-LLM scale.

- **H9-B (real-LLM wash-out / spectator pattern)** — fail-to-reject H0 at α = 0.05, OR rejection with `g < 0.5`. The G5-bis "MLX-only artefact" verdict extends from toy E-SNN to real-LLM ; the framework C effect does not survive the substrate transfer at production scale. EC stays PARTIAL ; DR-3 evidence file gets an H9-B row.

- **H9-C (real-LLM negative-direction break-through)** — rejection with negative sign (RECOMBINE/RESTRUCTURE actively *hurts* real LoRA retention). Framework C's RECOMBINE prediction is *further weakened* at real-LLM scale ; DR-3 evidence v0.x records the negative-direction finding.

No additional Welch test for the conjunction ; H9-{A,B,C} are mutually exclusive resolution states.

## §3 Power analysis

N = 5 seeds per arm at α = 0.05 detects |g| ≥ 1.85 at 80 % power (Welch two-sided, very small N — large effect floor only). The H9-A threshold of `g ≥ 0.5` is well below the detection floor at this N — so a confirmed H9-A here would only register if the real-LLM effect is *very large* (g ≥ 2). Sub-large effects (0.5 ≤ g < 2) are reported as **exploratory** under H9-A* with a separate flag. This is a deliberate design choice : real-LLM compute (~8-12 h Studio for N=5) cannot afford N=30 ; the pilot's role is to detect large effects or rule them out, not to power detection of medium effects.

## §4 Exclusion criteria

- per-subdomain MMLU `acc_initial < 0.20` (= 1 / 5 = random chance at multiple-choice with 5 choices, defensive widening) — exclude cell.
- `acc_final` non-finite — exclude cell.
- `mlx_lm.tuner.lora` fine-tune fails (e.g., OOM, NaN gradient) — abort cell, log via §9 envelope a.
- run_id collision with prior pilot's registry — abort + amend.

## §5 Substrate / driver paths

- Driver : `experiments/g6_studio_path_a/run_g6_studio_path_a.py`
- Substrate loader : `kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate.load()` Path 2 (real backend)
- LoRA loader : `experiments.g6_studio_path_a.lora_loader`
- LoRA train step : `experiments.g6_studio_path_a.lora_train_step` (wraps `mlx_lm.tuner.lora`)
- Live-tensor dream operations : `experiments.g6_studio_path_a.dream_episode_real`
- MMLU eval : `experiments.g6_studio_path_a.mmlu_eval` (wraps `mlx_lm.generate`)
- Aggregator : `experiments.g6_studio_path_a.aggregator_h9` (emits H9-A/B/C verdict)
- Sources :
  - SpikingKiki-V4 35B-A3B-V4 : `/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4`
  - V4 adapters : `/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-V4-adapters`
  - MMLU fixture : same as G6 Path B (`tests/fixtures/mmlu_g6_synthetic.jsonl` or production MMLU subdomain split, locked at first run)

## §6 DualVer outcome rules

| Outcome | EC bump | FC bump |
|---|---|---|
| Row 1 — H9-A confirmed (g ≥ 0.5, p < 0.05, positive sign, large effect ≥ 2) | EC stays PARTIAL ; **framework C 4-channel coupling validated at real-LLM scale** ; DR-3 evidence v0.x logs the architectural escape from G5-bis toy-E-SNN scope ; cross-substrate claim restored. | FC stays C-v0.12.0 |
| Row 1* — H9-A* exploratory (g ≥ 0.5, p < 0.05, positive sign, sub-large effect 0.5 ≤ g < 2) | EC stays PARTIAL ; reported as exploratory ; DR-3 evidence flags as suggestive ; needs N>5 follow-up. | FC stays C-v0.12.0 |
| Row 2 — H9-B confirmed (fail-to-reject, OR g < 0.5) | EC stays PARTIAL ; G5-bis MLX-only artefact verdict extends from toy-E-SNN to real-LLM ; DR-3 evidence v0.x extension. | FC stays C-v0.12.0 |
| Row 3 — H9-C confirmed (rejection negative sign) | EC stays PARTIAL ; framework's RECOMBINE prediction further weakened at real-LLM ; DR-3 evidence v0.x records negative-direction finding. | FC stays C-v0.12.0 |
| Row 4 — exclusion-rate > 50 % (insufficient cells) | abort and amend pre-reg with raised epochs OR larger N ; do not commit milestone. | n/a |

EC stays PARTIAL across all rows. FC stays at v0.12.0 across all rows.

## §7 Reporting commitment

Honest reporting of all observed scalars regardless of outcome.
H9-A confirmation specifically requires Welch *rejecting* H0 with positive sign and `g ≥ 0.5` (medium effect threshold) AND `g ≥ 2` (large effect threshold for the strict confirmation row, per §3 N=5 detection floor). H9-A* exploratory captures the medium-but-sub-large band.

If H9-B confirmed, the G5-bis MLX-only-artefact verdict is empirically extended from toy E-SNN to real-LLM tier ; DR-3 evidence amends with the real-LLM scope row.
If H9-A confirmed, the framework's RECOMBINE prediction is restored at real-LLM scale and the CNN-tier scope ceiling becomes substrate-bounded rather than universal.
If H9-C confirmed, the framework's claim is further weakened ; DR-3 evidence records the negative-direction real-LLM finding.

## §8 Audit trail

Cells registered via `harness/storage/run_registry.py` with profile keys `g6-studio-path-a/<arm>/<subdomain>/<seed>` and R1 bit-stable run_ids. Milestone artefacts under `docs/milestones/g6-studio-path-a-2026-05-04.{json,md}` (canonical) plus per-subdomain partial dumps `docs/milestones/g6-studio-path-a-partial-<subdomain>-2026-05-04.json` for resumability.

## §9 Deviations

Pre-known envelopes :

a. `mlx_lm.tuner.lora` OOM or NaN gradient — abort affected cell, log under §9.1 amendment ; if > 2 cells fail, abort pilot and amend pre-reg (consider reducing dim or batch).
b. SpikingKiki-V4 module load fails (`.npz` file missing or SHA-256 mismatch) — abort and file §9.1 amendment ; do not proceed with cells.
c. Per-cell wall > 30 min sustained — extrapolated total > 50 h ; escalate to user before committing milestone, propose N = 3 reduced run or subdomain reduction (5 → 3).
d. `mlx_lm.generate` letter-argmax non-deterministic across re-runs — log seed-stability check ; if non-deterministic, file §9.1 amendment with logit-argmax fallback.

Any deviation outside the envelopes requires an amendment commit *before* the affected cell runs, OR a post-hoc honest disclosure in Paper 2 §7.1.13 acknowledging the deviation and its impact on confirmatory status.

### §9.1 — TBD on first run if surprises surface

Reserved per the G4-quinto §9.1 / G4-sexto §9.1 / G4-septimo §9.1 pattern.
