# G6-Studio Path A* pilot pre-registration (daughter of Path A)

**Date:** 2026-05-04
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**Sister pre-reg:** `docs/osf-prereg-g6-studio-path-a.md` (Path A INSUFFICIENT verdict — 19/20 cells excluded by underperforming-baseline rule on 10-train/10-eval synthetic fixture, milestone `docs/milestones/g6-studio-path-a-2026-05-04.{json,md}` commit `d963b24`).
**Substrate:** unchanged from Path A — real `SpikingKiki-35B-A3B-V4` substrate via `MicroKikiSubstrate.load()` Path 2 with `DREAM_MICRO_KIKI_REAL=1`, KIKI-Mac_tunner mlx_lm fork at `/Users/clems/KIKI-Mac_tunner/lib/mlx_lm_fork/`, base model `Qwen3.6-35B-A3B` bf16.
**Adapter stack:** unchanged — `SpikingKiki-V4-adapters/` LoRA tensors mutated by 4-channel dream handlers (live-tensor coupling, NOT spectator).
**Benchmark:** **REAL** MMLU 5-subdomain CL stream from HuggingFace `cais/mmlu` (vs the 200-record synthetic fixture used by Path A). Subjects identical : anatomy (135 records), astronomy (152), business_ethics (100), clinical_knowledge (265), college_biology (144). Total 796 records pinned at `tests/fixtures/mmlu_g6_real.jsonl` SHA-256 to be filled at first commit.
**Compute:** N = 5 seeds × 4 arms × 5 subdomains = **100 measurements** (20 cells × 5 subdomains). Per-cell wall ~50 min on Studio M3 Ultra extrapolated from Path A actual rate. Larger train/eval budget (50/50 vs 10/10) means ~5× more LoRA iterations per subdomain, so total ~4-5 h overnight on Studio.
**Lock commit:** *(filled by introducing commit hash)*
**Lock timestamp:** 2026-05-04 (pre-driver-run, post Path A INSUFFICIENT debrief).

## §1 Background

The G6-Studio Path A pilot (commit `d963b24`) verified the real-LLM pipeline end-to-end : 35B bf16 + LoRA fine-tune (loss 0.74 → 0.07 typical) + 4-channel dream coupling on **live** delta tensors + `mlx_lm.generate` letter-argmax MMLU eval, all under the Metal memory budget (peak 139 GB / 210 GB cache budget) in 52 min wall on Studio M3 Ultra. The G6 Path B "spectator wrapper" caveat is empirically corrected.

However, the Path A H9-A/B/C verdicts came back as **INSUFFICIENT** because the synthetic 10-train / 10-eval per-subdomain fixture was too small : 19 of 20 (arm, seed) cells were excluded by the underperforming-baseline rule (`acc[S_1 after S_1] < UNDERPERFORM_THRESHOLD = 0.30` per `experiments/g6_mmlu_stream/run_g6.py:119`). With only 10 training records per subdomain, 50 LoRA iters on Qwen-35B-A3B do not push the first-subdomain eval accuracy above the 0.30 floor.

Path A* re-runs the identical pipeline with **real MMLU shards** (downloaded from HuggingFace `cais/mmlu` test split) and a **5× larger train/eval budget** (50/50 per subdomain). All other components — substrate, dream handlers, fork, Metal cache config, registry profile keys, exclusion rules — are unchanged from Path A. The hypothesis decision rules are unchanged from Path A pre-reg §2.

## §2 Hypotheses (confirmatory)

Identical to Path A pre-reg §2. H9-A/B/C decision rules unchanged.

- **H9-A** : `retention(P_max with mog)` is statistically distinguishable from `retention(P_max with none)` with predicted positive sign and `Hedges' g ≥ 0.5` ; strict large-effect threshold `g ≥ 2` at α = 0.05 (single test, no Bonferroni inheritance).
- **H9-A*** : exploratory medium-effect band (0.5 ≤ g < 2).
- **H9-B** : real-LLM wash-out (fail-to-reject OR g < 0.5) ; G5-bis MLX-only artefact verdict extends from toy-E-SNN to real-LLM tier.
- **H9-C** : real-LLM negative-direction break-through (Welch rejects with negative sign).

The Path A INSUFFICIENT outcome is *not* a confirmation or refutation of any H9-{A,B,C}. Path A* is a methodological re-run, not a re-test of a different hypothesis.

## §3 Power analysis

Identical to Path A pre-reg §3. N = 5 detects |g| ≥ 1.85 at 80 % power. The H9-A* exploratory band captures medium effects (0.5 ≤ g < 2) ; the H9-A strict confirmation requires g ≥ 2.

## §4 Exclusion criteria

Identical to Path A pre-reg §4. The underperforming-baseline rule (`acc[S_1 after S_1] < 0.30`) is unchanged. Path A* aims to reduce the exclusion rate from 95 % → < 50 % by giving the model 5× more train data per subdomain.

## §5 Substrate / driver paths

Identical to Path A pre-reg §5. Only differences : `--fixture-path tests/fixtures/mmlu_g6_real.jsonl`, `--n-train 50`, `--n-eval 50`. Driver, substrate, loader, aggregator, registry profile keys all identical. Output milestones :
- `docs/milestones/g6-studio-path-a-star-2026-05-04.{json,md}`
- per-subdomain partial dumps `docs/milestones/g6-studio-path-a-star-partial-<subdomain>-<arm>-seed<seed>-step<NN>-2026-05-04.json`

## §6 DualVer outcome rules

Identical to Path A pre-reg §6. EC stays PARTIAL across all rows. FC stays at C-v0.12.0+PARTIAL. Path A* INSUFFICIENT (if exclusion rate stays > 50 %) escalates to a Path A** with even larger compute (N=10 or n_train=100/n_eval=100).

## §7 Reporting commitment

Identical to Path A pre-reg §7. Honest reporting of all observed scalars regardless of outcome. The H9-A "rejecting is the predicted positive claim" inversion vs the G4-{quater..septimo} ladder is preserved.

## §8 Audit trail

Cells registered via `harness/storage/run_registry.py` with profile keys `g6-studio-path-a/<arm>` (note : same as Path A — the seed disambiguator in the R1 tuple `(c_version, profile, seed, commit_sha)` uniquely identifies each run, so Path A* uses the same profile-key namespace as Path A but a different commit_sha distinguishes them at registry level).

## §9 Deviations

Pre-known envelopes :

a. Real MMLU download fails (HF rate limit, network) — abort and file §9.1 amendment ; the fixture has already been built at `tests/fixtures/mmlu_g6_real.jsonl` for this pilot, so this envelope is consumed by the existing fixture.
b. Per-cell wall > 60 min sustained — extrapolated total > 20 h ; escalate to user before committing milestone, propose N = 3 reduced run.
c. `acc_initial` at first subdomain < 0.30 for majority of seeds again — declare H9 INSUFFICIENT a *second time* and file an honest §9.1 amendment ; do NOT escalate to Path A** without explicit user approval.
d. Out-of-memory on the larger training set — reduce `--n-train` from 50 → 30 and amend.

Any deviation outside the envelopes requires an amendment commit *before* the affected cell runs, OR a post-hoc honest disclosure in Paper 2 §7.1.13 acknowledging the deviation.

### §9.1 — TBD on first run if surprises surface

Reserved per the pattern.

### §9.2 — Partial-dump filename collision across Path A / A* / D (filed 2026-05-04)

**Trigger** : the driver's `PARTIAL_TEMPLATE` constant in
`experiments/g6_studio_path_a/run_g6_studio_path_a.py` is
hardcoded :

    PARTIAL_TEMPLATE = (
        "g6-studio-path-a-partial-{subdomain}-{arm}-seed{seed}-"
        "step{idx:02d}-{DATE_TAG}.json"
    )

It does NOT include a per-pilot label. Path A* (this pilot) and
Path D (Helium-2B on M1 Max, parallel run) write per-subdomain
partial dumps to the same filename namespace as Path A and Path
C, all under `g6-studio-path-a-partial-...-2026-05-04.json`.
Same date + same template → the partials of the latest-running
pilot **silently overwrite** earlier pilots' partials on disk.

**Impact assessment** :
- Final milestone JSONs use distinct names per pilot
  (`g6-studio-path-a-2026-05-04.json` vs
  `g6-studio-path-a-star-2026-05-04.json` vs
  `g6-m1max-path-d-mmlu-2026-05-04.json` etc.) and ARE distinct on
  disk + in git. The full data record for each pilot is preserved
  in its committed final JSON.
- Per-subdomain partial dumps are NOT committed to git (gitignored
  per plan §"File structure"). Their role is **purely operational**
  (resumability under watchdog kill within a single pilot's run).
  Their overwriting across pilots does NOT impair scientific
  reproducibility — every cell measurement is recorded in the
  final milestone JSON.
- If a pilot is killed mid-run AND a sibling pilot is started
  concurrently with overlapping partial namespace, resume logic
  could read the wrong pilot's partials. **This did not happen in
  this run** (Path A's process was already finished and committed
  before Path A* started ; Path D started after Path A's termination
  too).

**Documentation** : this collision is **noted but not corrected**
in this pre-reg amendment. The fix is scheduled for any future
G6 family pilot that runs concurrently with another, via either :
(a) parameterising `PARTIAL_TEMPLATE` to embed a pilot label
(`g6-studio-path-a-star-partial-...`), or (b) relocating partial
dumps to a per-pilot subdirectory (`docs/milestones/<pilot>/`).
The existing pre-reg §6 outcome rules and milestone-provenance
clauses are unaffected.

**Honest reporting clause for Paper 2 §7.1.13** : reviewers
should be informed that per-subdomain partial dumps shared a
filename namespace across the G6 family ; final JSONs and paper
verdicts are derived only from the committed per-pilot final
milestone files, not from the operational partial dumps.

### §9.3 — Path A* ABANDONED at user request (filed 2026-05-04)

**Trigger** : at 17:25 (~3 h 52 min into the run, ~30/100
partial dumps written, ETA ~7 h remaining), the user explicitly
requested termination of all dreamOfkiki compute on Studio and
M1 Max in order to reallocate hardware to other projects.
PID 80542 (parent `uv`) and PID 80543 (child python) were sent
SIGTERM and confirmed dead.

**State at termination** :
- ~30 (arm, seed, subdomain) measurements completed
  out of 100 planned (5 seeds × 4 arms × 5 subdomains)
- Final milestone JSON
  `docs/milestones/g6-studio-path-a-star-2026-05-04.{json,md}`
  was NOT written (driver writes only at end of full sweep)
- Per-subdomain partial dumps survive on Studio at
  `/Users/clems/Projets/dream-of-kiki/docs/milestones/g6-studio-
  path-a-partial-...-2026-05-04.json` (gitignored, not
  committed) — these can in principle be aggregated post-hoc
  into a partial verdict, but **this pre-reg amendment elects
  NOT to do so** : the H9-{A,B,C} decision rules locked in §2
  require the full N=5 × 4 arm × 5 subdomain sweep to compute
  the Welch tests at full power, and a partial-N verdict would
  conflate "INSUFFICIENT" with "ABANDONED".

**Decision** : pilot ABANDONED, no milestone artefact committed.
The H9-{A,B,C} hypotheses are NOT resolved by this pilot. The
Path A* verdict is recorded as **ABANDONED** in Paper 2 §7.1.13
and CHANGELOG.

**Consequences for the convergent-validation matrix** :
- The {Qwen-35B, real MMLU} cell remains EMPTY
  (no verdict, neither H9-A nor H9-B nor H9-C nor INSUFFICIENT).
- Path C ({Qwen-35B, symbolic CL}) was scaffolded but NEVER
  launched — also empty.
- The convergent-validation matrix at programme close has only
  2 of 4 cells filled :
    - Helium-2B + MMLU = H9-B (null)
    - Helium-2B + symbolic CL = INSUFFICIENT
  The H10-D / H11-D conjunctions cannot fire ; the framework C
  real-LLM tier verdict is provisionally read as : *no positive
  empirical claim has fired across any pre-registered scope
  ceiling tested ; the Helium-tier null is the only confirmed-
  cell finding ; Qwen-tier paths are deferred indefinitely.*
- The pre-reg remains valid for any future re-attempt ; it is
  not invalidated, just consumed-but-not-resolved by this
  Path A* attempt.

This §9.3 amendment closes the abandoned-pilot loop. The
pre-reg's §"Lock commit" line refers to the original lock
(`fd8bf52`) ; this amendment is filed at a later commit and
does NOT alter the original timestamp.
