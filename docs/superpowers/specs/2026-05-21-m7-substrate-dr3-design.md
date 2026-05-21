# M7 — Substrate DR-3 conformance fix design

> **Closes** : issue [#36](https://github.com/hypneum-lab/dream-of-kiki/issues/36).
> **Parent decision** : `2026-05-21-m6-m7-sequencing.md` — M7 first.
> **DualVer impact** : FC MINOR `C-v0.14.0 → C-v0.15.0`, EC stays `+PARTIAL`
> until the post-M7 bench v3 + ship-critic.

## 1. Goal

Make `kiki_oniric/substrates/mlx_latent_diffusion.execute_profile`
honour the framework-C profile contract : each profile activates a
*distinct subset* of the 4 dream-side primitives, and each activated
primitive emits its own **real** per-op metric (not a proxy of the
denoiser loss).

After M7, the 450-cell ablation bench produces *non-trivial* H1 / H2 /
H4 verdicts — `reject_h0 == True` exactly where the framework predicts,
and `False` exactly where it predicts the empirical-emptiness pattern
(DR-4 v0.6 amendment, paper 2 §6).

## 2. The contract (framework C §3, kiki_oniric/profiles/)

The three profiles activate the dream-side primitives as follows
(verbatim from `kiki_oniric/profiles/p_min.py`, `p_equ.py`,
`p_max.py` — confirm against disk at plan time) :

| Profile | Replay (α) | Downscale (β) | Restructure (γ) | Recombine (δ) |
|---|---|---|---|---|
| `p_min` | — | — | — | ✓ |
| `p_equ` | ✓ | ✓ | — | ✓ |
| `p_max` | ✓ | ✓ | ✓ | ✓ |

(Mapping is illustrative — M7 plan reads the real profile dataclasses
from `kiki_oniric/profiles/` and binds the substrate to them ; if the
disk shape diverges from this table, the disk wins.)

`mlx_latent_diffusion` today runs `Trainer(denoiser).fit + Sampler.sample`
unconditionally and attaches the profile string to the output dict. M7
replaces this with a per-profile pipeline that calls the real handler
for each activated primitive.

## 3. Decisions

### D1 — Handler boundary

- **Reuse** the existing `_real.py` handlers in
  `kiki_oniric/dream/operations/{replay,downscale,restructure,recombine}_real.py`.
  They already implement the K1 FLOP-tag contract and the S2 / S3
  guards. Do **not** inline op logic into `mlx_latent_diffusion`.
- The substrate gives each handler the diffusion-substrate-specific
  state object it needs (a small adapter dataclass holding the
  denoiser / encoder / sampler handles), and reads back the
  state's emitted metric.

### D2 — Per-op metrics

Replace the M5-era proxies with op-native quantities :

| Op | Today (M5 proxy) | M7 (real) |
|---|---|---|
| replay_rate | `loss_last` (single trainer.fit loss) | distinct-replay rate per `replay_real` (count of unique latent samples emitted / total batches) |
| downscale_norm | `sample_norm` (single sampler.sample L2) | rank-decay measure from `downscale_real` (S-Vd: top-k singular-value retained ratio post-shrink) |
| restructure_sum | hard-coded `0.0` | `restructure_real` topology-diff size (Add / Remove / Reroute event counts, S3-guarded) |
| recombine_rate | `float(len(losses))` | novel-sample rate per `recombine_real` (LatentSample provenance count, ep-tag distinct) |
| delta_acc | `loss_first - loss_last` | downstream MLP classifier delta accuracy on the task's val split (real CL signal) |
| wall_time_s | per-cell wall | unchanged |

`delta_acc` becomes a real continual-learning measurement : train a
tiny one-layer classifier head on the substrate's latents over the
task, evaluate on the task's val split. This is what makes H1
(forgetting / replay benefit) actually measurable.

### D3 — Per-profile dispatch

Inside `execute_profile` :

1. Read the activation set from `request.profile_obj` (extend
   `_CellRequest` to carry the profile dataclass, not just the
   string tag — keeps lookups out of the substrate).
2. For each activated op, in the canonical chain order
   `(replay → downscale → restructure) ∥ recombine` (per
   `kiki_oniric/dream/operations/CLAUDE.md`), invoke the
   `_real.py` handler with the substrate state.
3. Collect the metrics dict, emit it as-is to the harness row.

The synthetic-only branch (no `loader_batches`) is removed in M7 —
once the substrate respects the profile contract, the synthetic
path becomes redundant with the conformance tests. The R1 hashes
for the synthetic path are regenerated under FC MINOR.

### D4 — FC MINOR justification

Framework-C §12 bump rules : a substrate change that ships a *new*
primitive-activation surface is FC MINOR. M7 ships :

- `mlx_latent_diffusion` ↔ profile-contract binding (new behaviour).
- 4 new `_real.py` ↔ substrate adapters (mechanical).
- A new conformance test in `tests/conformance/axioms/test_dr3_diffusion_profile.py`
  proving the substrate's `execute_profile` activates exactly the
  primitives the profile declares (no more, no less).

EC stays `+PARTIAL` until the M7 re-bench result is reviewed by
ship-critic at M6.

### D5 — R1 regeneration

The synthetic R1 entries (`test_r1_diffusion_train_step`,
`test_r1_diffusion_dream_replay`, `test_r1_diffusion_full_de`) all
change under M7 because the substrate pipeline changes shape. They
are regenerated under the M7 FC bump with `status: "pending_review"`
per `REBASELINE_NOTE.md` discipline. A new entry
`test_r1_diffusion_profile_activation` is added : same `(seed,
task_idx)` cell across the 3 profiles produces 3 distinct hashes
that are *each* byte-stable within a machine.

### D6 — Acceptance for the eventual bench v3

The M7 work is *complete* (and unlocks M6) when :

1. All 4 conformance tests pass : `test_dr3_diffusion_profile.py`
   (new) + existing `test_dr{0,1,3}_diffusion*.py`.
2. The 450-cell bench v3 on macM1 produces 450 unique hashes AND
   per-profile metric distributions are visibly different (e.g.
   `restructure_sum > 0` for `p_max` only).
3. At least one of {H1, H2, H4} returns `reject_h0 == True` for
   at least one profile-vs-baseline pair (positive signal proves
   the substrate is no longer a degenerate uniform-output).
4. R1 within-machine byte-stable on macM1 (cold re-run produces
   identical hashes across all 450 cells).

If #3 fires `False` everywhere (i.e. profile contract honoured but
no positive verdict), the result is **still** acceptance-worthy : it
becomes a stronger empirical-emptiness claim, citable in paper 2 §6
alongside G4-sexto / G4-septimo. M7 ships either way.

## 4. File touch map (M7 implementation plan target)

| File | Action | LoC est. |
|---|---|---|
| `kiki_oniric/substrates/mlx_latent_diffusion.py` | Replace `execute_profile` body with per-profile dispatch | +120 −80 |
| `kiki_oniric/substrates/_diffusion/dream_ops_adapter.py` | New : 4 substrate→`_real.py` adapter shims | +200 |
| `scripts/ablation_cycle3_diffusion.py` | Extend `_CellRequest` with `profile_obj`, materialise it from the profile string | +30 −5 |
| `kiki_oniric/profiles/p_{min,equ,max}.py` | Read-only — confirm activation maps match D2 table | 0 |
| `tests/conformance/axioms/test_dr3_diffusion_profile.py` | New : prove profile→activation surface | +90 |
| `tests/reproducibility/golden_hashes_apple_*.json` | Regenerate per chip family under FC MINOR | (regen) |
| `tests/reproducibility/REBASELINE_NOTE.md` | Append M7 rebaseline entry | +10 |
| `kiki_oniric/substrates/mlx_latent_diffusion.py` (header) | Bump `MLX_LATENT_DIFFUSION_SUBSTRATE_VERSION` to `C-v0.15.0+PARTIAL` | ±1 |
| `pyproject.toml` | SemVer 0.22.2 → 0.23.0 (FC MINOR) | ±1 |
| `CHANGELOG.md` | New section `[C-v0.15.0+PARTIAL] — 2026-05-2X` documenting M7 | +15 |
| `docs/proofs/dr3-diffusion-substrate-evidence.md` | Bump to v1.0 with M7 + bench-v3 evidence | +60 |
| `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` | §3.X amendment if profile→activation table is normatively pinned | +20 (or 0 if §3 already pins) |

EN→FR propagation : the framework-C amendment (if any) propagates
to `docs/specs-fr/`.

## 5. Out of scope for M7

- Paper 2 §7.9 drafting (M6).
- OSF amendment filing (M6).
- DualVer flip to `+STABLE` (M6, gated on ship-critic).
- Cross-machine R1 reseeding (deferred to nightly per existing
  `docs/proofs/r1-cross-machine.md` posture).
- Adding new dream-side primitives (M7 only binds the existing 4).

## 6. Risks

| Ref | Risk | Mitigation |
|---|---|---|
| R1 | The diffusion-substrate adapters are non-trivial (e.g. `restructure_real` topology-diff doesn't map onto a denoiser MLP without a topology stand-in) | Plan task : decide the diffusion's "topology" surface explicitly (the MLP layer stack is the natural choice). If the mapping is genuinely impossible for restructure on this substrate, document and emit a documented no-op (mirror Canal 4 attention-prior treatment) — that is still a DR-3 conformance fix because the *contract* (substrate declares which ops it supports) is now honoured. |
| R2 | FC MINOR bump triggers a paper 1 / framework-C spec mirror update in `docs/specs-fr/` | Plan tasks the FR mirror as one task. Per `CONTRIBUTING.md`, EN→FR propagation is enforced in the same PR. |
| R3 | The 4 R1 hashes change → existing nightly `r1-nightly.yml` fails on first push | Plan ships the regenerated golden hashes in the same PR as the substrate change. The nightly fails for the duration of the PR review window — that is expected per `REBASELINE_NOTE.md` `pending_review` discipline. |
| R4 | M7 takes longer than 5 days | Fall back to the sequencing-spec §4 recovery : ship M6 as `+PARTIAL` with the existing 2026-05-21 milestone, file the hedged OSF amendment, M7 → Wave 3c. |

## 7. Self-review

- **Placeholder scan** : no TBD ; `2026-05-2X` is the deliberately
  unresolved re-bench date.
- **Internal consistency** : D2 metric table ↔ D6 acceptance ↔ §4
  touch map all reference the same 4 ops.
- **Scope check** : single substrate, single conformance gap, single
  FC MINOR bump. Fits one writing-plans plan.
- **Ambiguity** : the activation map in §2 is *illustrative* —
  the implementation plan will read the real dataclasses. Flagged
  explicitly so the plan task starts with a read of
  `kiki_oniric/profiles/p_{min,equ,max}.py`.
