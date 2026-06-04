# OSF Pre-registration Amendment — Wave 3b (DRAFT, ready-to-file)

**Target parent registration** : OSF Q6JYN
(DataCite DOI `10.17605/OSF.IO/Q6JYN`, parent registered 2026-04-19).
**Amendment type** : MINOR (additive substrate-axis extension + three
new R1 reproducibility entries ; no hypothesis modification).
**Status** : **DRAFT — awaiting approver confirmation before send.**
This file is *ready-to-file* (the §3 body below can be pasted into the
OSF Open-Ended Registration amendment form), but **no send action is
taken from this PR**. The send is a separate manual step gated on
§7 approval.
**Filing target** : end of Wave 3b M3 (after `HARNESS_VERSION` is
flipped to `C-v0.13.0+PARTIAL`), per plan §2.5 Decision D5 timeline.
**Publish target** : Wave 3b M5 (before any bench numbers leave the
local registry), per plan §4 M5 acceptance.
**Plan source of truth** :
`docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md` §2.5
(D5), §4 (M3 + M5), §5 (canonical amendment text).
**Sibling proof** : `docs/proofs/dr3-diffusion-substrate-evidence.md`
v0.1-draft — attach as supplementary at filing time.
**Filing-procedure precedent** : Bonferroni amendment
`docs/osf-amendment-bonferroni-cycle3.md` (Open-Ended Registration linked
to Q6JYN parent, minted as `10.17605/OSF.IO/TPM5S` on 2026-04-21,
48 h DOI mint ETA).

---

## §1 What changed (one-paragraph summary)

The Q6JYN parent pre-registration enumerates three framework-C
substrate instantiations in its `SUBSTRATES` axis : `mlx_kiki_oniric`,
`esnn_thalamocortical`, `micro_kiki`. **This amendment adds a fourth
substrate**, `mlx_latent_diffusion`, implementing an MLX-native
latent-diffusion generative-replay variant of the framework-C
8-primitive contract per the dossier
`docs/proofs/dr3-diffusion-substrate-evidence.md` v0.1-draft
(structural angle : 7/8 typed primitives + 1 documented type-correct
no-op for Canal 4 attention prior, justified by the flat geometry of
the 3-layer MLP denoiser). The empirical grid is **Split-CIFAR-100
5-tasks** (re-use of the G4-sexto / G4-septimo fixture already locked
under R1 — same SHA-256, no new fixture). The N=30 seed grid is
unchanged. Three new R1 entries are added to the reproducibility
contract per §4.

## §2 Why the pre-registered protocol still holds

The Q6JYN parent registration registered four hypotheses (H1, H2,
H3, H4) on the substrate × profile × seed grid plus the Bonferroni
correction structure (amendment `10.17605/OSF.IO/TPM5S` of
2026-04-21). **None of H1–H4 is modified** by this amendment. The
substrate axis gains one row, but the per-cell test ladder
(Jonckheere profile-chain + Welch + TOST per Paper 2 §6 ladder) is
unchanged. The per-cell Bonferroni denominator stays at the size set
by the parent amendment, because the new substrate row enlarges the
*number* of cells in the Cartesian grid, not the *family size per
cell*. The cross-cell families H3 / H6 (Bonferroni amendment table)
gain `mlx_latent_diffusion` as an additional cell each ; that
enlargement is the expected scaling of those families under
substrate-axis additions and is consistent with the amendment's
"Cross-cell (size 2)" pattern interpreted as "size 2 *per substrate
pair under comparison*".

## §3 New secondary hypothesis

**H7 (secondary, exploratory)** : Substrate `mlx_latent_diffusion`
satisfies the DR-3 Conformance Criterion (framework-C spec §6.2 :
signature typing ∧ axiom property tests ∧ BLOCKING invariants S1 /
S2 / S3 / I1 enforceable).

**Status at filing** : PARTIAL conformance pending Wave 3b M6
behavioural-angle closure (see sibling proof §5 verdict). The H7
report at M6 is *exploratory* : a PARTIAL or FAIL outcome does not
falsify H1–H4 of the parent registration ; it only re-classifies the
substrate row as Track B baseline (mirror Paper 2 §7.7 Wake-Sleep CL
non-conformant banner pattern). The plan §1.2 Track S / Track B
fork is the operational mechanism for that re-classification.

H7 is added here for OSF transparency only ; it carries no Bonferroni
correction (single hypothesis, exploratory, secondary).

## §4 Statistical plan delta

**None.** The per-cell test ladder (Jonckheere + Welch + TOST) is
identical to the cycle-3 protocol. The Bonferroni family structure
inherits from `10.17605/OSF.IO/TPM5S` unchanged. The N=30 seed grid
is unchanged. The substrate-axis enlargement from 3 to 4 substrates
multiplies the total cell count by 4/3 but does not change the
per-cell α. The H7 exploratory hypothesis is *not* added to any
Bonferroni family per §3.

## §5 Compute-budget delta

- **Wave 3b M3 (training)** : ~25 000 samples (Split-CIFAR-100
  5-tasks × 500 imgs/class × 50 classes) trained on M5 GrosMac
  (16 GB), `d_latent = 64`, batch 256, fits in unified memory. Local
  iteration ; no Studio queue time. Estimated wall ~ 2 h per training
  run.
- **Wave 3b M5 (bench)** : 1 substrate × 3 profiles × N=30 seeds
  × 5 CIFAR-100 task-splits = **450 cells** on Studio M3 Ultra.
  Estimated wall **6–8 h** end-to-end (Studio reserved for this
  window per plan §2.2 D2 downstream).
- **Wave 3b M2 / M4 (smoke + integration)** : 9 cells (1 × 3 × 3
  seeds) at ~ 15 min wall on M5 (plan §4 M4 acceptance) — negligible.

Total marginal compute over the Q6JYN parent budget is therefore
~ 2 h M5-local + ~ 8 h Studio M5 milestone + ~ 15 min M5 smoke.
This sits well inside the Q6JYN parent's reserved Studio bench
window pattern (cf. G4-octavo / G4-septimo Studio runs).

## §6 Timeline

- **DRAFT** (this file) : Wave 3b M1 ship (2026-05-20).
- **APPROVER REVIEW** : after M1 PR merge, before M3 ship — user
  reviews this draft + sibling proof + plan, decides approver chain
  per §7.
- **FILE** : end of Wave 3b M3, immediately after the M3 PR flips
  `HARNESS_VERSION` to `C-v0.13.0+PARTIAL` and the R1 entries land
  green on the macos-14 nightly runner.
- **MINT** : ~ 48 h after filing (DataCite ETA per
  `10.17605/OSF.IO/TPM5S` precedent of 2026-04-21).
- **PUBLISH** : Wave 3b M5 ship (the DOI must be live before any bench
  result row leaves the local `RunRegistry`).

## §7 Approver

**Approver** : `OSF approver TBD — confirm with PI before send.`

The Bonferroni amendment of 2026-04-21 was filed by Clement Saillant
as PI ; the same chain is the obvious default for this amendment.
**The user (PI) must confirm this explicitly before the send action
in §8 step 3 is taken.** This file deliberately does not pre-name
the approver to avoid impersonation in a draft that may sit between
M1 and M3 ship (several weeks).

## §8 Send instructions (do NOT execute from this PR)

1. Verify the M3 PR has merged ; `STATUS.md` reflects
   `C-v0.13.0+PARTIAL` ; `r1-nightly.yml` macos-14 run green for
   the three new R1 entries (see §4 R1 list below).
2. Confirm approver per §7 in writing.
3. On OSF (osf.io/q6jyn), open *Add registration → Open-Ended
   Registration linked to parent Q6JYN*. Copy-paste §1–§5 of this
   file verbatim into the *Project description* field (the
   §1-paragraph + §2 + §3 + §4 + §5 blocks).
4. Attach **two supplementary files** :
   - `docs/proofs/dr3-diffusion-substrate-evidence.md` (current
     version at filing time, expected v0.1-draft or later).
   - `docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md`
     (current version at filing time).
5. Mirror the DataCite metadata block from
   `docs/osf-amendment-bonferroni-cycle3.md` (substituting Wave 3b
   title and §1 abstract).
6. Submit. Wait ~ 48 h for DOI mint.
7. On mint, update **three locations** in the repo (all in the same
   follow-up commit) :
   - This file : flip status to PUBLISHED, add minted DOI under
     frontmatter as `Amendment #2 OSF DOI`.
   - `STATUS.md` : add Wave 3b amendment row to the OSF table.
   - `CHANGELOG.md` : append an entry under the Wave 3b M5 line.

## §9 New R1 reproducibility entries (preview)

Three entries land in `tests/reproducibility/golden_hashes.json` at
M3 ship per plan §2.3 (D3) :

| Entry | Asserts | Status at filing |
|-------|---------|------------------|
| `test_r1_diffusion_train_step` | One denoiser train step bit-exact across runs given fixed `mx.random.split`-derived per-step key | `pending_review` (per 2026-05-10 N2 rebaseline discipline) |
| `test_r1_diffusion_dream_replay` | One dream-phase replay invocation bit-exact across runs given fixed per-DE key | `pending_review` |
| `test_r1_diffusion_full_de` | One complete DreamEpisode (DE) bit-exact across runs given fixed root key + DE index | `pending_review` |

All three consume only **split-derived** keys, never raw
`mx.random.key(seed)` results consumed directly by `mx.random.normal`
— this is the M1 Max workaround for upstream issue
[ml-explore/mlx#3568](https://github.com/ml-explore/mlx/issues/3568)
per plan §2.3.

Cross-machine R1 across {M5 GrosMac, M3 Ultra Studio, M1 Max macM1}
is the M3 acceptance target ; partial failure on M1 Max is documented
but not a blocker, per the same posture as the 2026-05-20 cross-machine
probe in `STATUS.md`.

---

## §10 Verbatim paste-block for OSF form (consolidates §1–§5)

> **Title** : Wave 3b — MLX latent-diffusion substrate, additive
> substrate-axis extension and three new R1 reproducibility entries.
>
> **Amendment summary** (see plan §5 text — kept synchronised
> verbatim with `docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md`
> §5) :
>
> The dreamOfkiki parent pre-registration ([OSF Q6JYN](https://osf.io/q6jyn),
> DataCite DOI `10.17605/OSF.IO/Q6JYN`, 2026-04-19) enumerates three
> framework-C substrate instantiations (`mlx_kiki_oniric`,
> `esnn_thalamocortical`, `micro_kiki`). This amendment adds a
> fourth substrate, `mlx_latent_diffusion`, implementing a
> latent-diffusion generative-replay variant of the framework-C
> 8-primitive contract. The new substrate exposes 7/8 typed
> primitives (α, β, γ, δ, Canals 1, 2, 3) and one documented
> no-op (Canal 4 attention-prior, justified by the flat latent
> geometry of the MLP denoiser). Conformance with the DR-3
> Conformance Criterion (signature typing + axiom property tests
> + BLOCKING invariants S1/S2/S3/I1 enforceable) is the Wave 3b
> M2 gate deliverable ; if the gate fails, the substrate is
> re-classified as a published-reference baseline mirroring the
> Paper 2 §7.7 Wake-Sleep CL pattern, with an explicit
> non-conformant banner. The empirical grid is Split-CIFAR-100
> 5-tasks-buffer-500 (same fixture as G4-sexto confirmatory N=95,
> see `docs/milestones/g4-sexto-confirmatory-n95-results.md`),
> evaluated across the three framework-C profiles (P\_min, P\_equ,
> P\_max) at N=30 seeds, with hypotheses H1, H2, H4 reported per
> profile per the parent pre-reg §2. A secondary exploratory
> hypothesis H7 (DR-3 conformance of the new substrate) is added
> for transparency ; H7 carries no Bonferroni correction
> (single, exploratory).
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

**End of DRAFT.** Awaiting §7 approver confirmation. Do not file
from this PR.
