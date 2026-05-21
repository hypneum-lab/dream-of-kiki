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

## 2. The contract (framework C §3.1, NORMATIVE)

**Corrected post-audit 2026-05-21.** The activation table is fixed by
`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §3.1 as a
formal definition with the monotonic inclusion constraint of §3.2 :
`ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)`. The substrate **must** honour
this exact shape — the table is normative, not illustrative.

| Profile | replay | downscale | restructure | recombine |
|---|---|---|---|---|
| `p_min` | ✓ | ✓ | — | — |
| `p_equ` | ✓ | ✓ | ✓ | ✓ (light) |
| `p_max` | ✓ | ✓ | ✓ | ✓ (full) |

(Plus channel-out sets : P_min → {WeightDelta}, P_equ → {WeightDelta,
HierarchyChange, AttentionPrior}, P_max → all 4. M7 honours the ops
column ; channels are surfaced through the existing channel infra and
are out of scope for the substrate adapter.)

`PMinProfile` / `PEquProfile` / `PMaxProfile` already encode this map
**in their `__post_init__`** — each registers exactly the listed
handlers on `self.runtime: DreamRuntime`. The handler signatures
(audited 2026-05-21) :

- `replay_real_handler(state: ReplayRealState, *, model, lr=0.01)`
  → `Callable[[DreamEpisode], None]`. Reads `input_slice["beta_records"]`.
  Mutates `model.parameters()` in-place ; touches K1
  `state.last_compute_flops` / `total_compute_flops`.
- `downscale_real_handler(state: DownscaleRealState, *, model)` →
  `Callable[[DreamEpisode], None]`. Reads `input_slice["shrink_factor"]`
  (must be in `(0, 1]`). Mutates layer weights + bias in-place ; S2
  finite guard on the mutated tensors. K1 = `_param_count(model)`.
- `restructure_real_handler(state: RestructureRealState, *, model)`
  → `Callable[[DreamEpisode], None]`. *Legacy cycle-3* — only
  `topo_op == "reroute"` is supported in the **real** variant
  (the `_lora` variant has the full `{add, remove, reroute}` vocab).
  Reads `input_slice["topo_op"]` and `input_slice["swap_indices"]`.
- `recombine_real_handler(state: RecombineRealState, *, encoder, decoder, seed)`
  → `Callable[[DreamEpisode], LatentSample | None]`. Reads
  `input_slice["delta_latents"]` (required, non-empty — raises
  `I3` on empty). The substrate must supply VAE-shape `encoder` /
  `decoder` ; the diffusion `MLPDenoiser` is **not** a VAE — see §3
  decision D7 below for how the substrate maps onto this.

`mlx_latent_diffusion` today runs `Trainer(denoiser).fit + Sampler.sample`
unconditionally, attaches the profile string to the output dict, and
**never instantiates `PMinProfile / PEquProfile / PMaxProfile`** —
the `DreamRuntime` handler-dispatch mechanism is entirely bypassed
(`scripts/ablation_cycle3_diffusion.py` carries only the profile
**string** in `_CellRequest`, never the dataclass). M7 replaces this
with a per-profile pipeline that *instantiates* the right profile
dataclass, drives a small `DreamEpisode` stream through its `runtime`,
and reads the per-op metrics off the profile's state fields.

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

### D2 — Per-op metrics (audited)

Read the metrics off the **state objects the handlers already mutate**.
The four state classes from the audit (verbatim) :

```python
@dataclass class ReplayRealState:
    total_records_consumed: int = 0
    last_loss: float | None = None
    last_compute_flops: int = 0
    total_compute_flops: int = 0

@dataclass class DownscaleRealState:
    compound_factor: float = 1.0
    last_compute_flops: int = 0

@dataclass class RestructureRealState:
    diff_history: list[str] = field(default_factory=list)
    last_compute_flops: int = 0
    adds_this_episode: int = 0
    total_adds: int = 0
    total_removes: int = 0
    total_reroutes: int = 0

@dataclass class RecombineRealState:
    last_sample: list | None = None
    last_compute_flops: int = 0
    _episode_count: int = 0
```

Substrate row schema after M7 (replaces the M5 proxies — the field
names stay so milestone-aggregator code does not break, but the
**source** of each field changes) :

| Substrate row field | Source after M7 |
|---|---|
| `replay_rate` | `replay_state.last_loss` (drops to `0.0` if replay inactive) |
| `downscale_norm` | `downscale_state.compound_factor` (stays `1.0` if inactive — that *is* "no downscaling") |
| `restructure_sum` | `restructure_state.total_reroutes` (legacy real-handler only emits `reroute`, see audit ; stays `0` if inactive — matching the M5 behaviour for `p_min` honestly) |
| `recombine_rate` | `recombine_state._episode_count` (number of recombine events ; `0` if inactive) |
| `delta_acc` | a tiny 1-layer classifier head trained on the substrate's latents over the task, eval on the task's val split — task-local CL accuracy delta (`acc_after_dream − acc_before_dream`). This is the real H1 signal. |
| `wall_time_s` | per-cell wall, unchanged |
| `op_flops_total` | new : sum of `last_compute_flops` across the activated ops (K1 audit trail) |

`delta_acc` is the load-bearing new measurement. The classifier head
is a `mx.nn.Linear(d_latent, N_CLASSES_PER_TASK)` trained for a
fixed micro-budget (e.g. 50 steps, batch 64) on the encoded latents
before and after the dream cycle ; the difference is per-cell real
CL signal. Hyper-parameters pinned in the plan so the bench is
deterministic.

### D3 — Per-profile dispatch via existing `DreamRuntime`

Inside `execute_profile` :

1. **Instantiate the profile dataclass** matching `request.profile`
   (`{"p_min": PMinProfile, "p_equ": PEquProfile, "p_max": PMaxProfile}[…]`).
   Each profile's `__post_init__` already registers exactly the right
   handlers on its `runtime: DreamRuntime` per framework-C §3.1 — we
   do **not** re-derive the activation surface, we *use* it.
2. The substrate provides : the `model` (denoiser MLP), and for the
   `recombine` adapter an `encoder` / `decoder` pair (see D7) and a
   `seed`. These are passed to the handler factories via a small
   **substrate-wiring adapter** that re-registers the existing
   handlers on the profile's `runtime` with the substrate-supplied
   `model` argument (the default `model=None` in profile fields
   needs to be replaced for the real path).
3. Drive a **canonical `DreamEpisode` stream** : one `DreamEpisode`
   per loader batch, with `input_slice` populated for *all* possible
   ops (the runtime ignores keys whose op isn't registered, so a
   single shared episode shape works) :
   - `"beta_records"` : encoded `(x, y)` pairs from the batch
   - `"shrink_factor"` : a fixed value (e.g. `0.95`) so the
     downscale handler does measurable but bounded work
   - `"topo_op"` : `"reroute"` ; `"swap_indices"` : a deterministic
     pair per (seed, batch_idx)
   - `"delta_latents"` : the encoder output for the batch
   - `"species"` : `"diffusion"`
4. Read the metrics dict by snapshotting the 4 state dataclasses
   after the run. Inactive ops keep their default values (legitimate
   zeros, **not** hard-coded — the field defaults *are* the framework-C
   no-op semantics).
5. Compute `delta_acc` separately (before/after head-eval, see D2).

The synthetic-only branch (no `loader_batches`) **stays** in M7 to
preserve the M3 R1 test contract — the synthetic path remains
profile-aware in the same way, but with synthetic latents instead
of CIFAR. The synthetic-path R1 hashes regenerate under FC MINOR
(D5).

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

### D7 — VAE-shape adapter for the recombine handler

The `recombine_real_handler` audit reveals that it expects an
`encoder: VAEEncoder` and a `decoder: VAEDecoder` per the existing
Protocols at `kiki_oniric/dream/operations/recombine_real.py` lines
55-77 (audit-confirmed). The diffusion substrate has an encoder
(`self._cifar_encoder : 3072 → d_latent`) but **no decoder** — the
`MLPDenoiser` predicts noise, it does not reconstruct features.

For the recombine contract to be honoured on the diffusion substrate
(active for `p_equ` and `p_max` per framework-C §3.1), M7 adds a
small **`_diffusion_decoder` MLP** mirroring the existing `Encoder`
shape : `d_latent → RAW_FEATURE_DIM=3072`. Construction lives in
the substrate's `__init__` next to `_cifar_encoder` ; the two are
passed as `encoder` / `decoder` arguments to `recombine_real_handler`
at handler-build time.

This is **not** a learned-decoder claim — it is a random-init MLP
that gives the recombine handler a Protocol-valid surface so the
deterministic LatentSample generation is well-defined. The decoder
weights are stable across the bench (init once, reuse 450 cells per
seed family — actually re-init per cell to match the per-cell
fresh-substrate posture from M5).

`p_min` is unaffected — recombine is inactive there.

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
| `kiki_oniric/substrates/_diffusion/dream_ops_adapter.py` | New : profile → substrate-wiring adapter that supplies `model` / `encoder` / `decoder` / `seed` to the existing `_real.py` factories and re-registers them on the profile's `runtime` | +150 |
| `kiki_oniric/substrates/_diffusion/decoder.py` | New : random-init MLP `d_latent → 3072` for the recombine VAE-shape contract (D7) | +40 |
| `kiki_oniric/substrates/_diffusion/__init__.py` | Re-export the new `Decoder` next to `Encoder` | +2 |
| `scripts/ablation_cycle3_diffusion.py` | No `profile_obj` field needed — the substrate instantiates `PMinProfile/PEquProfile/PMaxProfile` from the string at `execute_profile` entry. Keep `_CellRequest` shape stable. | 0 |
| `kiki_oniric/profiles/p_{min,equ,max}.py` | Read-only — the existing `__post_init__` handler registrations *are* the activation surface ; M7 honours them via D3 | 0 |
| `kiki_oniric/substrates/_diffusion/cl_eval_head.py` | New : 1-layer classifier head for `delta_acc` measurement (D2) | +60 |
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
| R1 | The diffusion-substrate adapters are non-trivial (e.g. `restructure_real` only supports `reroute` per audit — the *legacy* real handler is feature-restricted vs. the LoRA variant) | The denoiser MLP `n_layers` stack *is* the topology surface for `reroute` (swap two layer indices). For the `add` and `remove` topo_ops the handler raises today — that is a substrate-honest limitation, not an M7 blocker. The activation surface honours what each profile actually drives ; the diffusion's restructure surface is `{reroute}` and the M7 conformance test asserts exactly that. |
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
