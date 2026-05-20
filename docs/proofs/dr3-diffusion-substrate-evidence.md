# DR-3 — MLX latent-diffusion substrate conformance evidence (v0.1-draft)

**Version** : v0.1-draft (2026-05-20)
**Status** : WIP — structural angle filled, behavioural angle deferred to M6
**Target promotion** : v1.0 at Wave 3b M6 (bench closure) iff M5 numbers back
the behavioural angle ; otherwise downgrade to category-b reference baseline.
**Supersedes** : — (first issue, sibling to `dr3-substrate-evidence.md` which
covers MLX-kiki-oniric + E-SNN)
**Amendment pointer** : none. This is a *positive* conformance argument for
a candidate fourth substrate ; the DR-3 statement (axioms spec §6.2) is
unchanged. Mirror discipline of `dream2learn-dr3-separation.md` v0.1
(two-angle structure, structural ∧ behavioural).
**Target venue** : Paper 2 §7.9 (Wave 3b row).
**Executable counterpart** : pending — `tests/conformance/axioms/test_dr3_diffusion_substrate.py`
is a Wave 3b M2 deliverable (per plan §3.1 module layout). This document
sketches what that test will assert.
**Plan source of truth** : `docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md`
§3.2 (typing table) + §3.4 (test families).
**Related** :
- DR-3 axiom statement — `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §6.2
- DR-3 two-substrate evidence (MLX + E-SNN) — `dr3-substrate-evidence.md`
- Dream2Learn separation (negative case template) — `dream2learn-dr3-separation.md`
- Framework-C 8-primitive contract — `kiki_oniric/core/primitives.py`

---

## §1 Goal

Establish whether the **MLX latent-diffusion substrate** specified in
Wave 3b plan §3.1 (`kiki_oniric/substrates/mlx_latent_diffusion.py`,
3-layer MLP denoiser, `d_latent = 64`, M5-local training on Split-CIFAR-100
5-tasks, ~2 M parameters — see plan §2.2 Decision D2) satisfies the **DR-3
Conformance Criterion** (signature typing ∧ axiom property tests
∧ BLOCKING invariants S1/S2/S3/I1 enforceable).

The argument is **structural** (this document, M1) ∧ **behavioural** (M6,
once empirical evidence from the M5 bench run is available). Both angles
must conclude GREEN for the substrate to be promoted to a fourth
DR-3-conformant variant joining MLX-kiki-oniric, E-SNN, and micro-kiki.

The mirror posture against Wave 3a's Dream2Learn negative case is
deliberate : Wave 3a established *what a DR-3-conformant latent-diffusion
variant would need that D2L lacks* ; Wave 3b is the construction attempt
for that delta, and this document is the dossier the Track-S-vs-Track-B
decision at the M2 gate reads from (plan §1.2).

---

## §2 DR-3 statement (verbatim)

From `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §6.2,
axiom **DR-3 (substrate-agnosticism)** :

> The framework C 8-primitive contract (α, β, γ, δ ; Canals 1 / 2 / 3 / 4)
> admits more than one substrate instantiation. A substrate `S` is said to
> **conform** to DR-3 iff all three conditions hold :
>
> 1. **Signature typing.** Each of the 8 primitives is implemented by `S`
>    with a Python `runtime_checkable` Protocol that type-matches the
>    canonical signature in `kiki_oniric/core/primitives.py`. A documented
>    no-op return (e.g. empty list, zero-tensor) is permitted iff it is
>    type-correct ; un-implemented primitives are not.
> 2. **Axiom property tests.** The substrate passes the DR-0
>    (accountability), DR-1 (episodic conservation), and DR-2 (weakened
>    compositionality, precondition v0.2) Hypothesis-based property suites
>    under at least one configured `RunRegistry` seed.
> 3. **BLOCKING invariants enforceable.** The S-family invariants S1
>    (no silent weight drift outside the dream-swap protocol), S2 (finite
>    gradients, no NaN/Inf), S3 (per-DE compute budget bounded), and the
>    I-family invariant I1 (episodic record conservation) are enforced
>    by the substrate's guard layer and cite their invariant ID in the
>    assertion message (CLAUDE.md §Working rules item 2).
>
> A substrate that satisfies conditions 1+2+3 is **DR-3-conformant**.
> A substrate that satisfies only condition 1 is a **typed baseline**
> (mirror Paper 2 §7.7 Wake-Sleep CL pattern, non-conformant banner).

(The DR-3 statement is reproduced verbatim above ; if the spec changes,
update both this proof and `dr3-substrate-evidence.md` in the same PR per
the `docs/proofs/CLAUDE.md` "Proof revision ↔ axiom test" coupling rule.)

---

## §3 Structural angle — typed primitives present ?

Map each of the 8 framework-C primitives (signatures in
`kiki_oniric/core/primitives.py`) to its planned Wave 3b
correspondent. This table is the M1 dossier the M2 gate review consumes.
It mirrors `dream2learn-dr3-separation.md` §3.1 structural table (which
scored 0/8) and Wave 3b plan §3.2 (which scored 7/8).

| # | Primitive | Canonical signature | Wave 3b correspondent | Typed (T / N / X) |
|---|-----------|---------------------|-----------------------|-------------------|
| 1 | α `AlphaStreamProtocol` | `append_trace(trace) / iter_traces()` | Awake encoder `E` activations on real CIFAR-100 batches, logged as `mx.array` traces under per-DE key (plan §2.3 key tree) | **T** |
| 2 | β `BetaBufferProtocol` | `append_record / fetch_unconsumed / mark_consumed` | Curated `(z, label)` latent records sampled from α stream via fixed-budget reservoir | **T** |
| 3 | γ `GammaSnapshotProtocol` | `get_checkpoint_path / get_checkpoint_sha256` | Awake-classifier `f_θ` MLX checkpoint at DE start, sha256 pinned via `register_output_hash` (plan §1.1 condition 4) | **T** |
| 4 | δ `DeltaLatentsProtocol` | `snapshot(...) / get_recent(...)` | Diffusion denoiser `U` latent-layer activation snapshot at DE start, `mx.array` of shape `(batch, d_latent=64)` | **T** |
| 5 | Canal 1 `WeightDeltaChannel` | `apply(delta)` | Merge of dream-phase `U_dream → U_awake` weight delta under invariant S1 (no silent drift outside swap) | **T** |
| 6 | Canal 2 `LatentSampleChannel` | `enqueue / dequeue` | Reverse-process samples `x̂ ~ D(·\|z)` consumed by classifier training in awake phase | **T** |
| 7 | Canal 3 `HierarchyChangeChannel` | `apply_diff(diff)` | Diffusion latent-geometry mutation via the `restructure` op — the novel delta over Dream2Learn whose latents are frozen (separation §3.1) | **T** (novel) |
| 8 | Canal 4 `AttentionPriorChannel` | `set_prior / get_prior` | **No-op** : flat MLP denoiser geometry, no attention mechanism in U ; type-correct empty return per DR-3 condition 1 escape clause | **N** |

**Structural score : 7/8 typed, 1/8 documented no-op.** This places the
Wave 3b candidate at the **conformance bar** : the 7/8 floor is plan
§1.3 condition 1 (the verbatim delta over D2L's 0/8). Below the bar
(≤ 6/8) would force Track B fallback at M2.

### §3.1 The 8th gap — Canal 4 attention prior, explicitly identified

The single non-typed primitive is **Canal 4 `AttentionPriorChannel`**.
The framework-C §6.2 DR-3 condition 1 admits a documented no-op iff it
is *type-correct* — meaning the substrate must still implement the
Protocol, but `set_prior` may store the prior in a discarded slot and
`get_prior` may return a zero-tensor of the canonical shape.

**Why this is acceptable here.** The MLP denoiser `U` operates on a
flat latent vector `z ∈ R^64` with no attention mechanism. There is no
operator inside `U` that *could* consume a non-trivial attention prior
without changing `U`'s architecture (which would void the M5-local
trainability target — see plan §2.2 D2 alternative "UNet" rejected for
this reason). The no-op is therefore *honest* : the substrate has no
internal place to *use* an attention prior, so it advertises the
absence via a type-correct empty return rather than faking a usage path.

**Why this remains a gap nonetheless.** Canal 4 is not vacuous in the
framework — `mlx_kiki_oniric` and `esnn_thalamocortical` both consume
non-trivial attention priors per their substrate-specific channel
implementations (cited in `dr3-substrate-evidence.md` §2 and §3). The
Wave 3b substrate therefore covers a strictly smaller operational
surface than the two existing DR-3-conformant substrates. **This is
the 7/8-not-8/8 caveat that the plan §3 design note flagged as
"just enough conformance to be dangerous"** — the substrate clears
the typing bar but does not exercise the full channel set.

**Consequence for the M6 verdict.** If the M5 bench shows that the
no-op Canal 4 substantively hurts the substrate's continual-learning
metrics (M1.a forgetting, M1.b average accuracy, M3.b retention)
relative to MLX-kiki-oniric and E-SNN, the M6 proof closure must
**either** (a) qualify the conformance verdict as PARTIAL (citing the
no-op Canal 4 as the candidate cause), **or** (b) escalate to a UNet
denoiser variant that does carry an attention mechanism (plan §2.2
alternative path), reopening D2 at M6.

This document **does not** prejudge that path at M1. The 7/8 typing
score is sufficient to keep Track S open at the M2 gate ; the
behavioural angle (§4) is what closes or reopens the question at M6.

---

## §4 Behavioural angle — STATUS : DEFERRED TO M6

The behavioural angle requires empirical evidence from the M5 bench
run (plan §4 M5 deliverable : 1 substrate × 3 profiles × N=30 seeds
× 5 CIFAR-100 task-splits = 450 cells on Studio M3 Ultra).
At M1, no such evidence exists ; this section is a **template** for
the M6 fill-in.

### §4.1 What M6 must report

For each of the three configured profiles (P_min, P_equ, P_max) :

1. **DR-0 (accountability)** — DE budget exhaustion logged in
   `RunRegistry` for ≥ 95 % of cells ; no orphan DE without a
   `budget_consumed` field.
2. **DR-1 (episodic conservation)** — `|episode_records_in| ==
   |episode_records_out|` invariant holds per DE across all 450
   cells (Hypothesis-property suite, `test_dr1_diffusion_finite.py`).
3. **DR-2 (weakened compositionality, v0.2 precondition)** — under
   the `¬(∃ i<j : π_i=RESTRUCTURE ∧ π_j=REPLAY)` precondition,
   compositionality closure holds. Witnesses for the precondition
   failure path documented per `dr2-compositionality.md` v0.2
   §4.3 pattern.
4. **S1 / S2 / S3 / I1 invariant guards** — assertion-message
   audit on all 450 cells confirms each guard cites its invariant
   ID per CLAUDE.md §Working rules item 2.

### §4.2 Threshold for conformance pass at M6

The substrate is **DR-3-conformant** iff §4.1 items 1+2+3+4 all
GREEN. **Any** item RED forces the verdict to PARTIAL conformance
(§5) and triggers the framework-C §12.3 "no axiom violation"
condition to fail for the +STABLE EC flip — meaning the M6 DualVer
bump stays at `C-v0.13.0+PARTIAL` (plan §2.4 D4 conditional flip).

### §4.3 Threshold for non-conformance downgrade at M6

The substrate is **re-classified as Track B baseline** (Wake-Sleep
CL pattern, Paper 2 §7.7 non-conformant banner) iff DR-2 fails
even under the v0.2 precondition (i.e. the compositionality
weakening cannot rescue the diffusion substrate's
operator-chaining behaviour). This is the same downgrade path the
plan §1.2 Track S/B gate at M2 would have taken, applied post-hoc
at M6 if the M5 evidence forces it.

---

## §5 Provisional verdict (v0.1)

**PARTIAL conformance candidate, pending behavioural angle at M6.**

The structural angle (§3) closes 7/8 typed primitives + 1 type-correct
no-op, clearing the DR-3 Conformance Criterion condition 1 bar. The
behavioural angle (§4) is uncomputable at M1 because the substrate is
not yet implemented (Wave 3b M2 deliverable) and the bench is not yet
run (Wave 3b M5 deliverable).

This verdict is **not** a refutation : the substrate remains a
candidate fourth DR-3-conformant variant. The verdict is **not** a
confirmation either : the 7/8-not-8/8 caveat (§3.1) and the
deferred behavioural angle (§4) leave open both the +STABLE promotion
path and the Track B downgrade path.

The verdict **is** : Track S remains open going into M2 ; the M2 gate
should read this document plus the M2 implementation artifact, not
this document alone.

Parallel to `dream2learn-dr3-separation.md` v0.1's *negative* two-angle
structure (structural ∨ behavioural sufficient for separation), this
*positive* dossier requires **both** angles GREEN for full conformance
— mirroring the Wave 3b plan §6 R7 risk mitigation (avoid
confirmation bias by requiring both angles before flipping the
Track S/B switch).

---

## §6 What M6 must measure to close

| Measurement | Source | Threshold |
|-------------|--------|-----------|
| DR-0 accountability audit | RunRegistry dump of all 450 cells | ≥ 95 % cells have `budget_consumed` field, 0 orphan DE |
| DR-1 episodic conservation | `test_dr1_diffusion_finite.py` Hypothesis run on 450 cells | 100 % cells satisfy `\|records_in\| == \|records_out\|` |
| DR-2 weakened compositionality | `test_dr2_*` adapted for diffusion substrate | precondition-respecting runs : 100 % closure ; precondition-violating runs : documented xfail witnesses |
| S1 silent-drift guard | substrate code grep + runtime assertion log | every weight mutation cites S1 in assert message |
| S2 finite-gradient guard | runtime assertion log on 450 cells | 0 NaN/Inf in `z` or `x̂` ; any breach cites S2 |
| S3 per-DE compute budget | RunRegistry budget field | every DE has bounded budget recorded ; 0 unbounded |
| I1 episodic record conservation | `test_dr1_*` cross-checks I1 | 100 % cells |
| Canal 4 no-op impact (caveat §3.1) | M5 metrics comparison vs MLX-kiki-oniric | if M1.a forgetting / M1.b avg accuracy / M3.b retention show statistically significant gap (Welch + TOST per Paper 2 §6 ladder), document gap and qualify verdict in §5 |

A future v1.0 of this document, filed at M6 ship, fills the right-hand
column with the M5 bench numbers and either (a) promotes the verdict
in §5 to "FULL DR-3-conformant" (the four substrates row gains a
fourth member), or (b) demotes to "Track B baseline" with the M5
numbers backing the demotion. **No claim of either kind is made at
M1.** This is consistent with CLAUDE.md §Working rules item 3 (no
synthetic-pipeline numbers reported as empirical claims) — at M1, the
behavioural angle is *empty*, not *synthetic*.

---

## §7 References

- **Plan source of truth** :
  `docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md`
  §1 (goal), §2 (decisions D1–D5), §3.2 (typing table), §3.4 (test
  families), §4 M1/M2/M6, §6 R7 (confirmation-bias mitigation), §7
  (decision matrix).
- **Wave 3a negative case (template)** :
  `docs/proofs/dream2learn-dr3-separation.md` v0.1 — the two-angle
  (structural ∧ behavioural) structure is mirrored here in the
  positive direction.
- **DR-3 axiom (source of truth)** :
  `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §6.2.
- **DR-3 existing two-substrate evidence** :
  `docs/proofs/dr3-substrate-evidence.md` (MLX-kiki-oniric + E-SNN,
  C2.10 status) — the M6 closure must place this Wave 3b substrate
  *alongside* this dossier, not replace it.
- **Framework-C 8-primitive contract (signatures)** :
  `kiki_oniric/core/primitives.py`.
- **R1 reproducibility coupling** :
  `tests/reproducibility/REBASELINE_NOTE.md` (pending_review →
  accepted discipline) + `STATUS.md` 2026-05-20 cross-machine
  probe.
- **OSF parent registration** :
  `docs/osf-preregistration-draft.md` (Q6JYN) + amendment template
  `docs/osf-amendment-wave3b.md` (draft, M1 deliverable sibling).

---

**End of v0.1-draft.** Next revision : v1.0 at Wave 3b M6 ship,
behavioural angle filled with M5 bench numbers, verdict §5 promoted or
demoted accordingly, table §6 filled column by column.
