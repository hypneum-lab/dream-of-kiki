# Amendment: DR-0 as a budget, selective DR-1, `downscale` to `rescale`, "axiom" to "framework hypothesis"

- **Date**: 2026-11-08
- **Status**: **L0 DRAFT** (not frozen, no tag, no SWHID; see section 7)
- **Target**: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
  §6.1 (Op), §6.2 (DR-0, DR-1), §4 pillar B (`downscale`, line 265);
  `docs/axioms/AXIOMS.md`; `kiki_oniric/axioms/__init__.py`
- **Issue**: hypneum-lab/dream-of-kiki#48 (milestone M1, due 2026-11-29)
- **Baseline**: `main` at `ef1c399`, `C-v0.25.0+PARTIAL`
- **Proposed bump**: formal axis only, decided by the PI (decision D8);
  no bump is applied by this document
- **Author**: agent draft; every normative choice below is a proposal
  until the PI closes the decisions of section 9

This amendment is **additive**. It does not edit the frozen spec text,
the registered preregistrations, or any dated result. Where it says
"old text", the quoted text is the one currently on `main`.

## 1. Summary

| Item | Old | New (proposed) |
|---|---|---|
| DR-0 | every dream output traces to a DE with `budget(DE) < ∞`; the DE is a bounded off-line slot | DR-0 is an explicit **budget** (bits erased, traffic, duration; joules measured on a separate axis) that **selects the operator**; a DE may be continuous or nested, bounded by its budget, not by a clock |
| DR-1 | every buffered episode is offered to a DE within `τ_max`, then purged | **selective**: only episodes above a declared weight threshold are owed a chance; **no mandatory purge**; raw retention is allowed |
| Pillar B | `downscale`, factor in (0, 1] (SHY) | `rescale(f)`, factor of indefinite sign (WISE): the optimal f may lie on either side of 1 depending on the wake/sleep rate contrast; `downscale` is the special case f ≤ 1 |
| Vocabulary | "axiom" | "framework hypothesis" (a posed hypothesis whose consequences are derived, with a delimited empirical scope) |

## 2. Why: sources

The four changes come from the 2026-10-24 hypotheses review of
`hypneum-lab/.github` (commit `00110de`), directory
`docs/research/2026-10-24-revue-hypotheses/`. All paths below are
relative to it. `R1` and `R9` below abbreviate
`revues/R1-consolidation-reve-sommeil.md` and
`revues/R9-thermodynamique-apprentissage.md`.

> Source-path note: the issue text points at the 2026-11-01
> "analyse profonde" README for §2.1, §3, §7.2 and §7.6. The passages
> quoted there (§2.1 dream-of-kiki, §3 reformulations 1-2, §7.2
> point 3, §7.6 "Posture") are in fact in the 2026-10-24 review
> README at the lines below; the 2026-11-01 README has a different
> section layout and contains none of these passages. The 2026-11-06
> catalogue v3 (`docs/research/2026-11-06-refresh-hypotheses/catalogue-v3.md`)
> rows `DOK-DR0` and `DOK-DR1` (lines 62-63) carry the decision status.

| Change | Source (file:line) |
|---|---|
| Four convergent reformulations: DR-0 budget, DR-1 selective, `downscale` to `rescale` | `README.md:78` (§2.1, "Ce que la littérature change") |
| DR-0 = budget that selects the operator; a TTT module is a conforming substrate | `README.md:146` (§3, reformulation 1; Titans/Hope, Titan-Llama 2610.19210, SCoL 2605.07076, MemoryBench, Kang 2607.17545, Thumann & Soloveichik 2610.24435) |
| DR-1 selective, no mandatory purge | `README.md:147` (§3 table, row 2) (Denis 2026, PLoS Biol; minority of SWR replayed; Zhang 2605.12978); `revues/R1-consolidation-reve-sommeil.md:131-134` (§3.2, verdict: weighted conservation, optional purge) |
| `downscale` to `rescale`, indefinite sign | `README.md:150` (§3, row 5; WISE, Kinoshita 2025); `revues/R1-consolidation-reve-sommeil.md:45` (WISE contradicts "factor in (0, 1]"), `:214` (T5), `:231` (H-R1.5) |
| Action item | `README.md:273` (§7.2 point 3: "DR-0 to operational budget (writes, traffic, duration; joules apart; R9 H1 revised); DR-1 selective; `downscale` to `rescale` (R1 T5)") |
| Do not state DR-0 in joules; instantiate by bits erased | `revues/R9-thermodynamique-apprentissage.md:10` (substrate-agnosticism), `:254` (H1), `:282` (C1) |
| Bits alone are insufficient: reversible replay has B_eff = 0 yet changes outputs, hence a traffic term | `revues/R9-thermodynamique-apprentissage.md:272` (H10) |
| "Axiom" vs "framework hypothesis" | `README.md:304` (§7.6 "Posture": "« axiome » est légitime pour une hypothèse de cadre posée dont on démontre les conséquences, à condition d'en délimiter la portée empirique"; "invariant" reserved for what passed a calibrated null) |
| Schedule of the work | `ROADMAP.md:130` (DR-0/DR-1 deferred "après M3", status "utile, sous verrou"), `ROADMAP.md:163` (Q2, 8-21 Feb: Paper 1 reformulated, "axiomes" kept as framework hypotheses) in `hypneum-lab/.github` |

Local evidence already in this repository (all dated, none edited here):

- `docs/results/2026-10-25-nh03-ttt.md:93-118`: NH-03 primary contrast
  (TTT, 24 writes) satisfies the preregistered rule "continuous >
  bounded", but is a recency effect, "not an advantage specific to TTT
  fast weights" (:96); the evidence supports "a write budget whose
  schedule is a free parameter" (:113-115).
- `docs/results/2026-10-27-nh03-schedule-ablation.md:51-55`: verdict
  `recency`; a DR-0 amendment "should read write budget + placement
  relative to the next demand".
- `docs/results/2026-10-30-nh03-deferred-eval.md:61,73`: verdict
  `recency_survives`; "still no TTT-specific advantage".
- `docs/milestones/2026-10-25-nh03-ttt-vs-episodes.md:175-179`: the
  preregistered reading of each NH-03 outcome ("formal-axis bump to be
  proposed separately, with the DR-0 amendment issue", :178-179).

## 3. What changes: old text to new text

The "new text" blocks are **proposals** (decisions D3-D7 may change
them). They are written so that the old statements are special cases.

### 3.1 DR-0 (spec §6.2; `kiki_oniric/axioms/__init__.py` `DR0`)

Old (verbatim, `docs/axioms/AXIOMS.md` "DR-0"):

```
∀ δ ∈ dream_output_channels,
∃ DE ∈ History : budget(DE) < ∞ ∧ δ ∈ outputs(DE)
```

with the interpretation "no ambient dreaming: every dream contribution
is traceable to a bounded DE", and `BudgetCap(flops, wall_time_s,
energy_j)` as the budget unit (`kiki_oniric/dream/episode.py:35-40`).

New (proposed, DR-0'):

```
budget(DE) = (B_bits, B_traffic, T)           # all finite, all recorded
∀ δ ∈ dream_output_channels,
∃ DE ∈ History : δ ∈ outputs(DE) ∧ budget(DE) is finite and recorded
ops(DE) = select(budget(DE), state)           # the budget chooses the operator set
```

- `B_bits`: bits logically erased by the episode, a substrate-agnostic
  lower bound (Landauer); `B_traffic`: writes or bytes moved, times
  activity; `T`: duration. Joules are **measured and reported on a
  separate axis**, not part of the statement (R9:10).
- A DE is bounded by its budget, **not by a clock**: a continuous
  fast-weight update (TTT) is a DE whose budget accrues over wake time;
  a nested DE is a DE whose budget is spent in sub-episodes.
- The schedule (placement of the writes relative to the next demand) is
  a free parameter of the DE, not part of the axiom (NH-03 results).
- `select` is a declared policy, an input of the framework, not a
  theorem. The amendment claims no optimality for any selection rule.
- "No ambient dreaming" is kept: every output channel still traces to a
  recorded finite-budget DE.

Not implied by the old statement: today's `BudgetCap` has no
`bits` or `traffic` field, so the **new** form is not yet checked by
the code or by `tests/conformance/axioms/test_dr0_accountability.py`
(follow-up code PR; the old form keeps passing).

### 3.2 DR-1 (spec §6.2, line 436; `DR1`)

Old (verbatim):

```
∀ e ∈ β_t, ∃ t' ∈ [t, t + τ_max] : e ∈ inputs(DE_{t'})
```

"every buffered episode gets a chance at consolidation", enforced by
the I1 runtime check and a "β purge gate".

New (proposed, DR-1'):

```
∀ e ∈ β_t, weight(e) > θ  ⇒  ∃ t' ∈ [t, t + τ_max] : e ∈ inputs(DE_{t'})
purge(e) is optional; raw retention of e is permitted
```

- `weight(e)` and `θ` are declared per profile and fixed in a
  preregistration before use (decision D6); the amendment does not
  choose them.
- With `weight ≡ 1` and `θ < 1` the old statement is recovered.
- Consequence for invariants: I1 ("episodic conservation until
  consolidation", BLOCKING, `docs/invariants/registry.md:8`) and its
  "β purge gate" must be restated (e.g. "episodic primacy": no
  episode is destroyed before it is retained raw or consolidated).
  This is a registry change and belongs to the follow-up spec PR.

### 3.3 Pillar B: `downscale` to `rescale` (spec line 265, §6.1 `Op`)

Old (spec line 265): "apply homeostatic regularizer (SHY-style),
shrink weights toward prior, reduce noise"; handlers reject
`shrink_factor` outside (0, 1]
(`kiki_oniric/dream/operations/downscale.py:42-45`,
`downscale_real.py:87-90,162-166`,
`kiki_oniric/substrates/ttt_fast_weights.py:246`).

New (proposed): `rescale(f)` multiplies the targeted weights (or rates)
by f, where the sign of the homeostatic change is **not fixed**: the
WISE framework gives potentiation under classical Hebb/STDP sleep-like
firing and depression (SHY) only under anti-Hebb or when wake firing is
much faster (R1:45). The optimal f is therefore an empirical function
of the wake/sleep rate contrast; the hypothesis that it crosses 1 is
NH-05 (catalogue v3 line 253; R1 H-R1.5), **not tested here**.

- `downscale` becomes the special case f ≤ 1 and stays valid.
- Domain of f and the weight-magnitude guard (S2, `I-Wmag`,
  `kiki_oniric/dream/operations/CLAUDE.md:45`) are decision D4.
- Algebra: scalar rescales still commute; non-idempotence still holds
  for f ≠ 1 (`docs/papers/paper1/full-draft.md:228` states it for
  `downscale`); what changes is that a state-dependent f makes
  "commutative and non-idempotent" no longer the only relevant
  property (`revues/R1-consolidation-reve-sommeil.md:138`).

### 3.4 "Axiom" to "framework hypothesis"

DR-0..DR-4 and DR-2' keep their identifiers. In prose they become
"framework hypotheses (DR-n)": hypotheses posed by the framework, whose
consequences are derived, with a stated empirical scope. "Invariant"
stays reserved for what passed a calibrated null (README:304 of the
review). Code identifiers (`Axiom`, `kiki_oniric.axioms`, test directory
`tests/conformance/axioms/`) are untouched by this PR (decision D7).

## 4. Consequences

- **NH-03 / TTT as a C+ substrate.** `ttt_fast_weights` already passes
  the C+ suite under the old DR-0
  (`tests/conformance/axioms/test_dr3_ttt_substrate.py:1-12`; its write
  ledger caps writes, `kiki_oniric/substrates/ttt_fast_weights.py:85`).
  The amendment changes what that conformance *means* (a continuous
  write stream is one DE under a write budget, not a sequence of
  bounded slots), not whether it holds. It does **not** establish that
  TTT consolidates better than discrete episodes: the three NH-03
  results exclude a TTT-specific advantage on this toy benchmark
  (G4-ter MLP, Split-FMNIST, N=30). The amendment must be cited as
  "budget-defined episode" evidence only, never as "continuous
  consolidation wins".
- **Paper 1.** Reformulation is the ROADMAP Q2 deliverable
  (`ROADMAP.md:163`, 8-21 Feb 2027). Passages touched:
  `docs/papers/paper1/full-draft.md:215-228` (pillar B / SHY),
  `:520` (DR-0 property test as conformance), `:1063` (SHY row of the
  related-work table). No paper text is edited here.
- **DR-3 / C+.** `conforms(S)` (spec §6.2) keeps its three conditions;
  condition 2 now tests the amended DR-0/DR-1 once the tests exist.
  Until then C+ results are labelled "pre-amendment" (section 6).
- **DR-2 / DR-2'.** Unchanged in statement. Canonical order text
  "replay < downscale < restructure < recombine" is renamed with the
  code, not now.
- **DR-4.** Out of scope (the review's inter-domain reopening of DR-4
  is a separate item, `README.md:150`).
- **Roadmap conflict.** `ROADMAP.md:130` defers DR-0/DR-1 to after M3
  ("utile, sous verrou") while issue #48 sits in M1 (2026-11-29), and
  principle 6 forbids a freeze tag without a reading note,
  an adversarial reviewer and three passed tests
  (`ROADMAP.md:14`; quoted in `docs/adr/0001-kxkm-derogation-e1-small-controls-p8c.md`).
  See decision D1.

## 5. What this PR does and does not do

Does: add this amendment, its inventory companion, ADR-0002, and
one-line "amended by (L0 draft)" pointers in the spec (EN and FR mirror)
and `docs/axioms/AXIOMS.md`.
Does not: rename any identifier, edit any frozen text, change tests,
run anything, create a tag, or touch OSF.

## 6. Runs and results excluded or re-labelled

Search done on `ef1c399`: every use of `downscale` (section 8), the
budget unit (`kiki_oniric/dream/episode.py:35-40`), and the dated
results under `docs/milestones/`, `docs/results/`, `docs/osf-*`.

**Invalidated: none. Excluded from their own confirmatory analysis:
none.** The old statements are special cases of the new ones, every
registered run used f ≤ 1 (the handlers reject f > 1, so no run could
have used another sign) and finite-budget bounded DEs. What changes is
the **label** and what they may be cited for.

### 6.1 Re-labelled "pre-amendment" (fixed-sign `downscale`, f in (0, 1], DEs scheduled as bounded episodes)

They remain valid under their own frozen preregistration and tag. They
may not be cited as evidence for sign-indefinite `rescale`, for the
bits/traffic form of DR-0, or for selective DR-1.

| Family | Evidence (file:line) |
|---|---|
| E1-small (tag `prereg-e1-small-20261028`) incl. H14-B DOWNSCALE ablation | `docs/osf-prereg-e1-small-2026-10-25.md:219-222,704,874-876`; `docs/osf-amendment-e1-small-2026-10-30.md:151-154` (SHRINK = 0.99); same-norm control `docs/osf-declaration-h14b-same-norm-control-2026-11-02.md`; controls addendum `docs/osf-addendum-e1-small-controls-2026-11-06.md`; run records `docs/milestones/e1-small/` |
| NH-03 (tags `prereg-nh03-20261025`, `prereg-nh03-sched-20261027`, `prereg-nh03-deferred-20261030`) | arm (a) = replay + downscale, one bounded DE (`docs/milestones/2026-10-25-nh03-ttt-vs-episodes.md:81`); TTT forgetting gate rejects f > 1 (`experiments/nh03_ttt_vs_episodes/ttt_classifier.py:149`, `kiki_oniric/substrates/ttt_fast_weights.py:246`); results `docs/results/2026-10-25-nh03-ttt.md`, `2026-10-27-nh03-schedule-ablation.md`, `2026-10-30-nh03-deferred-eval.md` |
| G2/G4/G5/G6 pilots and cycle-3 pilots (G4-ter replay + downscale g = +2.77, etc.) | `docs/milestones/g2-*`, `g4-*`, `g5-*`, `g6-*`, `pilot-cycle3-*`; G4-ter C5 factor 0.95 (`docs/milestones/2026-10-25-nh03-ttt-vs-episodes.md:68`) |
| Conformance matrix and DR-0/DR-1 property tests (old form) | `docs/milestones/conformance-matrix.md`, `tests/conformance/axioms/test_dr0_accountability.py`, `test_dr1_episodic_conservation.py` (fake β buffer, `docs/proofs/dr-1.md`) |

Not individually audited: I did not open each G-series runner to check
how its DEs are scheduled; the statement that they are bounded-slot
DEs rests on the spec text (§6.2 DR-0) and on the NH-03 arm (a) record.

### 6.2 Excluded from supporting a specific claim

| Run or result | Excluded from supporting | Reason |
|---|---|---|
| NH-03 primary contrast (`ttt_uniform − slow_early_block`, +4.7 / +6.9) | "TTT beats discrete episodes"; "continuous consolidation beats episodes" | recency effect, schedule and substrate confounded (`docs/results/2026-10-25-nh03-ttt.md:93-98`; `2026-10-30-nh03-deferred-eval.md:73-77`) |
| every run to date | "optimal f crosses 1" (NH-05) | no run used f > 1 |
| every run to date | "B_eff / traffic is finite and monotone in the budget" (R9 H1, H10) | no bits or traffic counter exists; `BudgetCap` carries flops, wall time, joules only |

### 6.3 Rule for the future

Any run started before the L1 tag of this amendment is pilot or
exploratory for the amended hypotheses (hypneum-prereg: tag-before-data
check). Currently running campaigns are not touched.

## 7. L1 procedure (not executed)

1. PI closes decisions D1-D8 (section 9); open "TBD by PI" items block
   the freeze.
2. Reading-lock items cleared (D1): reading note, adversarial review,
   three tests (`ROADMAP.md:14`).
3. This PR (or its successor with the chosen wording) is merged on
   **green CI** (`gh pr checks --watch`).
4. Manifest: the amendment, ADR-0002, the inventory TSV, and the spec
   pointer files, each with SHA-256; its own SHA-256 recorded in the tag
   message.
5. Annotated tag on the merge commit, pushed to `github`:
   ```bash
   git tag -a prereg-amendment-dr0-dr1-rescale-YYYYMMDD \
     -m "Amendment DR-0/DR-1/rescale: manifest sha256 <...>"
   git push github prereg-amendment-dr0-dr1-rescale-YYYYMMDD
   ```
   `YYYYMMDD` is the merge date (existing tags use the compact form,
   e.g. `prereg-nh03-20261025`). Proposed name if merged on the day:
   `prereg-amendment-dr0-dr1-rescale-20261108`.
6. Software Heritage save of
   `https://github.com/hypneum-lab/dream-of-kiki/`, poll until
   `succeeded`, record the **revision** SWHID (`swh:1:rev:...`, never
   `swh:1:rel:`). Tag and SWHID are recorded below at freeze time.
7. A tag that expires (wording changed after the tag) is replaced by a
   new dated tag, never moved.

| Field | Value |
|---|---|
| Tag | _not created_ |
| Tagger date | _not created_ |
| SWHID (revision) | _not recorded_ |
| Manifest SHA-256 | _not computed_ |

L2 (e-Soleau / restricted Zenodo) and L3 (OSF) are PI-only and not
required for a framework-wording amendment unless the PI decides
otherwise (D2).

## 8. Inventory: every occurrence of `downscale`

Command: `git grep -n downscale -- '*.py' '*.md' '*.toml'` on `ef1c399`
(case-sensitive, as requested), extended with `-i` to also catch
`DOWNSCALE` (enum member `Operation.DOWNSCALE`) and "downscaling".
No `.toml` file matches. The complete line-level listing (1,981 lines
case-insensitive) is in
`docs/specs/amendments/2026-11-08-downscale-inventory.tsv`; the
per-file tables are below for code, tests, scripts and normative docs.
Dated documents (classes D1-D3) are listed by count only: they are
never edited.

Rules: (1) code and tests are renamed in a **follow-up code PR** with a
backward-compatible alias (`rescale` primary; `downscale`,
`Operation.DOWNSCALE` and the serialised op-name string kept working,
so frozen scripts, R1 golden hashes and run logs stay valid);
(2) files under a freeze tag are never renamed in place; (3) normative
docs are renamed in a follow-up docs PR after the L1 tag, because the
spec text is frozen and docs saying `rescale` while the enum still says
`DOWNSCALE` would be inconsistent; the only docs that state the
pillar's meaning and are touched now are the spec, FR spec and
`AXIOMS.md`, by **pointer only**.

### 8.1 Summary

| Class | Kind | Files | `downscale` lines (case-sensitive) | All-case lines | Treatment |
|---|---|---|---|---|---|
| C1 | code | 39 | 159 | 200 | Follow-up code PR: introduce `rescale` names; keep `downscale`/`DOWNSCALE` as backward-compatible aliases |
| C2 | test | 59 | 256 | 367 | Follow-up code PR: rename tests last, after the alias exists; add alias-equivalence tests; keep R1 golden hashes unchanged |
| C3 | frozen script/experiment | 51 | 157 | 271 | Rename only in the follow-up code PR, never a file listed in a freeze manifest (check against tags `prereg-*`); until then runs keep working through the alias |
| D1 | pre-registration (frozen) | 16 | 17 | 102 | Never edit; amendments are additive |
| D2 | dated result (append-only) | 28 | 33 | 52 | Never edit; re-label through the amendment (section 6) |
| D3 | historical plan / changelog | 50 | 574 | 727 | Never edit; history |
| N1 | normative spec (frozen text) | 6 | 43 | 45 | Pointer only in this PR; new wording lives in this amendment; fold in at the next FC bump |
| N2 | paper source | 28 | 128 | 159 | Follow-up docs PR: Paper 1 reformulation (ROADMAP Q2); rename where the pillar meaning is stated |
| N3 | proof / evidence | 5 | 48 | 51 | Follow-up docs PR after L1 tag; add dated note, do not rewrite the proof |
| N4 | other doc | 4 | 7 | 7 | Follow-up docs PR after L1 tag; rename where the pillar meaning is stated |
| | **Total** | **286** | **1422** | **1981** | |

### 8.2 Code, tests, scripts, experiments (C1-C3)

| Class | File | `downscale` | all-case |
|---|---|---|---|
| C1 | `harness/diffusion_eval/milestone.py` | 4 | 4 |
| C1 | `harness/real_models/base_model_registry.py` | 1 | 1 |
| C1 | `harness/real_models/qwen_mlx_fp16.py` | 4 | 4 |
| C1 | `kiki_oniric/CLAUDE.md` | 1 | 1 |
| C1 | `kiki_oniric/axioms/__init__.py` | 1 | 2 |
| C1 | `kiki_oniric/consolidate.py` | 1 | 4 |
| C1 | `kiki_oniric/dream/episode.py` | 1 | 1 |
| C1 | `kiki_oniric/dream/operations/CLAUDE.md` | 4 | 4 |
| C1 | `kiki_oniric/dream/operations/downscale.py` | 8 | 12 |
| C1 | `kiki_oniric/dream/operations/downscale_real.py` | 11 | 15 |
| C1 | `kiki_oniric/dream/operations/downscale_snn.py` | 8 | 11 |
| C1 | `kiki_oniric/dream/operations/restructure.py` | 1 | 1 |
| C1 | `kiki_oniric/dream/runtime.py` | 1 | 1 |
| C1 | `kiki_oniric/eval/state_alignment.py` | 1 | 1 |
| C1 | `kiki_oniric/profiles/p_equ.py` | 5 | 7 |
| C1 | `kiki_oniric/profiles/p_equ_lora.py` | 8 | 11 |
| C1 | `kiki_oniric/profiles/p_max.py` | 5 | 8 |
| C1 | `kiki_oniric/profiles/p_max_lora.py` | 8 | 11 |
| C1 | `kiki_oniric/profiles/p_min.py` | 6 | 8 |
| C1 | `kiki_oniric/profiles/p_min_lora.py` | 7 | 11 |
| C1 | `kiki_oniric/substrates/_adversarial/cat_a_trivial.py` | 6 | 6 |
| C1 | `kiki_oniric/substrates/_adversarial/cat_b_adversarial.py` | 7 | 7 |
| C1 | `kiki_oniric/substrates/_adversarial/cat_c_statistical.py` | 5 | 5 |
| C1 | `kiki_oniric/substrates/_adversarial/cat_d_modular.py` | 4 | 4 |
| C1 | `kiki_oniric/substrates/_adversarial/cat_e_replay_loops.py` | 4 | 4 |
| C1 | `kiki_oniric/substrates/_adversarial/cat_f_asymmetric.py` | 2 | 2 |
| C1 | `kiki_oniric/substrates/_diffusion/__init__.py` | 2 | 2 |
| C1 | `kiki_oniric/substrates/_diffusion/denoiser_weight_channel.py` | 1 | 1 |
| C1 | `kiki_oniric/substrates/_diffusion/dream_ops_adapter.py` | 5 | 8 |
| C1 | `kiki_oniric/substrates/_diffusion/handlers_emit.py` | 7 | 8 |
| C1 | `kiki_oniric/substrates/esnn_norse.py` | 2 | 2 |
| C1 | `kiki_oniric/substrates/esnn_thalamocortical.py` | 3 | 3 |
| C1 | `kiki_oniric/substrates/factory.py` | 3 | 3 |
| C1 | `kiki_oniric/substrates/micro_kiki/handlers.py` | 3 | 3 |
| C1 | `kiki_oniric/substrates/micro_kiki/substrate.py` | 4 | 4 |
| C1 | `kiki_oniric/substrates/mlx_kiki_oniric.py` | 1 | 1 |
| C1 | `kiki_oniric/substrates/mlx_latent_diffusion.py` | 9 | 11 |
| C1 | `kiki_oniric/substrates/ttt_fast_weights.py` | 4 | 7 |
| C1 | `kiki_oniric/substrates/wake_sleep_cl_baseline.py` | 1 | 1 |
| C2 | `tests/conformance/axioms/_dsl.py` | 11 | 14 |
| C2 | `tests/conformance/axioms/test_dr2_compositionality_empirical.py` | 10 | 12 |
| C2 | `tests/conformance/axioms/test_dr2_prime_canonical_order.py` | 1 | 3 |
| C2 | `tests/conformance/axioms/test_dr3_diffusion_profile.py` | 0 | 3 |
| C2 | `tests/conformance/axioms/test_dr3_esnn_substrate.py` | 7 | 7 |
| C2 | `tests/conformance/axioms/test_dr3_g4_deeper_substrate.py` | 1 | 2 |
| C2 | `tests/conformance/axioms/test_dr3_g4_hierarchical_substrate.py` | 2 | 3 |
| C2 | `tests/conformance/axioms/test_dr3_micro_kiki_substrate.py` | 2 | 2 |
| C2 | `tests/conformance/axioms/test_dr3_ttt_substrate.py` | 9 | 11 |
| C2 | `tests/conformance/test_baseline_registration.py` | 1 | 1 |
| C2 | `tests/conformance/test_snn_dr3.py` | 15 | 20 |
| C2 | `tests/reproducibility/REBASELINE_NOTE.md` | 3 | 3 |
| C2 | `tests/reproducibility/__init__.py` | 1 | 1 |
| C2 | `tests/reproducibility/test_r1_bit_exact.py` | 11 | 20 |
| C2 | `tests/unit/diffusion_eval/test_milestone.py` | 1 | 1 |
| C2 | `tests/unit/experiments/test_g4_dream_wrap.py` | 8 | 13 |
| C2 | `tests/unit/experiments/test_g4_ter_hier.py` | 5 | 6 |
| C2 | `tests/unit/experiments/test_g4_ter_hp_grid.py` | 1 | 1 |
| C2 | `tests/unit/experiments/test_g5_bis_esnn_dream_wrap_hier.py` | 2 | 3 |
| C2 | `tests/unit/experiments/test_g5_esnn_dream_wrap.py` | 0 | 1 |
| C2 | `tests/unit/experiments/test_g5_ter_dream_wrap_cnn.py` | 8 | 8 |
| C2 | `tests/unit/experiments/test_g6_dream_wrap.py` | 2 | 2 |
| C2 | `tests/unit/experiments/test_g6_micro_kiki_inference.py` | 1 | 1 |
| C2 | `tests/unit/experiments/test_g6_studio_dream_episode_real.py` | 2 | 2 |
| C2 | `tests/unit/experiments/test_nh03_driver.py` | 1 | 1 |
| C2 | `tests/unit/profiles/test_p_equ_lora.py` | 5 | 7 |
| C2 | `tests/unit/profiles/test_p_max_lora.py` | 4 | 6 |
| C2 | `tests/unit/profiles/test_p_min_lora.py` | 5 | 7 |
| C2 | `tests/unit/scripts/test_e1_controls.py` | 7 | 8 |
| C2 | `tests/unit/scripts/test_e1_profiles.py` | 5 | 5 |
| C2 | `tests/unit/scripts/test_e1_small_torch.py` | 1 | 1 |
| C2 | `tests/unit/scripts/test_pilot_cycle3_cuda.py` | 3 | 3 |
| C2 | `tests/unit/test_apply_channel_outputs.py` | 0 | 1 |
| C2 | `tests/unit/test_axioms_api.py` | 0 | 10 |
| C2 | `tests/unit/test_diffusion_handlers_emit.py` | 4 | 7 |
| C2 | `tests/unit/test_downscale_lora.py` | 16 | 25 |
| C2 | `tests/unit/test_downscale_op.py` | 16 | 21 |
| C2 | `tests/unit/test_downscale_op_mlx.py` | 12 | 16 |
| C2 | `tests/unit/test_dream_ops_adapter.py` | 1 | 3 |
| C2 | `tests/unit/test_episode.py` | 1 | 1 |
| C2 | `tests/unit/test_esnn_norse.py` | 7 | 7 |
| C2 | `tests/unit/test_esnn_operations.py` | 2 | 3 |
| C2 | `tests/unit/test_esnn_substrate.py` | 2 | 2 |
| C2 | `tests/unit/test_g4_quinto_cifar_mlp.py` | 3 | 3 |
| C2 | `tests/unit/test_g4_quinto_small_cnn.py` | 4 | 4 |
| C2 | `tests/unit/test_micro_kiki_substrate.py` | 5 | 5 |
| C2 | `tests/unit/test_mlx_latent_diffusion_adapter.py` | 7 | 7 |
| C2 | `tests/unit/test_p_equ_wiring.py` | 1 | 5 |
| C2 | `tests/unit/test_p_max.py` | 0 | 2 |
| C2 | `tests/unit/test_p_max_wiring.py` | 1 | 4 |
| C2 | `tests/unit/test_p_min.py` | 8 | 10 |
| C2 | `tests/unit/test_p_min_e2e.py` | 1 | 1 |
| C2 | `tests/unit/test_qwen_mlx_fp16.py` | 1 | 1 |
| C2 | `tests/unit/test_real_ops.py` | 26 | 41 |
| C2 | `tests/unit/test_runtime.py` | 0 | 2 |
| C2 | `tests/unit/test_runtime_channel_outputs.py` | 0 | 5 |
| C2 | `tests/unit/test_substrate_abstraction.py` | 1 | 1 |
| C2 | `tests/unit/test_substrate_factory.py` | 1 | 1 |
| C2 | `tests/unit/test_wake_sleep_baseline_adapter.py` | 1 | 1 |
| C3 | `docs/milestones/studio-cpu-2026-05-04/robertson_sequential_ordering.py` | 2 | 10 |
| C3 | `experiments/g4_nono_test/run_cross_domain.py` | 1 | 3 |
| C3 | `experiments/g4_octavo_test/run_step1_vit_tiny.py` | 1 | 5 |
| C3 | `experiments/g4_octavo_test/vit_tiny.py` | 1 | 2 |
| C3 | `experiments/g4_quater_test/__init__.py` | 0 | 1 |
| C3 | `experiments/g4_quater_test/aggregator.py` | 0 | 1 |
| C3 | `experiments/g4_quater_test/deeper_classifier.py` | 1 | 1 |
| C3 | `experiments/g4_quater_test/recombine_strategies.py` | 0 | 1 |
| C3 | `experiments/g4_quater_test/run_step1_deeper.py` | 1 | 6 |
| C3 | `experiments/g4_quater_test/run_step2_restructure_sweep.py` | 1 | 1 |
| C3 | `experiments/g4_quater_test/run_step3_recombine_strategies.py` | 2 | 6 |
| C3 | `experiments/g4_quinto_test/aggregator.py` | 0 | 1 |
| C3 | `experiments/g4_quinto_test/cifar_mlp_classifier.py` | 1 | 1 |
| C3 | `experiments/g4_quinto_test/run_step1_mlp_cifar.py` | 1 | 6 |
| C3 | `experiments/g4_quinto_test/run_step2_cnn_cifar.py` | 1 | 6 |
| C3 | `experiments/g4_quinto_test/run_step3_cnn_recombine.py` | 1 | 6 |
| C3 | `experiments/g4_quinto_test/small_cnn.py` | 1 | 2 |
| C3 | `experiments/g4_septimo_test/aggregator.py` | 0 | 1 |
| C3 | `experiments/g4_septimo_test/medium_cnn.py` | 1 | 2 |
| C3 | `experiments/g4_septimo_test/run_step1_tiny_imagenet.py` | 1 | 6 |
| C3 | `experiments/g4_sexto_test/aggregator.py` | 0 | 2 |
| C3 | `experiments/g4_sexto_test/run_step1_cifar100.py` | 1 | 6 |
| C3 | `experiments/g4_split_fmnist/dream_wrap.py` | 7 | 11 |
| C3 | `experiments/g4_ter_hp_sweep/dream_wrap_hier.py` | 6 | 9 |
| C3 | `experiments/g4_ter_hp_sweep/hp_grid.py` | 1 | 1 |
| C3 | `experiments/g4_ter_hp_sweep/run_g4_ter.py` | 2 | 2 |
| C3 | `experiments/g5_bis_richer_esnn/esnn_dream_wrap_hier.py` | 5 | 10 |
| C3 | `experiments/g5_bis_richer_esnn/run_g5_bis.py` | 1 | 2 |
| C3 | `experiments/g5_cross_substrate/esnn_dream_wrap.py` | 4 | 6 |
| C3 | `experiments/g5_ter_spiking_cnn/dream_wrap_cnn.py` | 5 | 13 |
| C3 | `experiments/g5_ter_spiking_cnn/run_g5_ter.py` | 1 | 2 |
| C3 | `experiments/g6_mmlu_stream/dream_wrap.py` | 12 | 12 |
| C3 | `experiments/g6_mmlu_stream/micro_kiki_inference.py` | 1 | 1 |
| C3 | `experiments/g6_studio_path_a/__init__.py` | 0 | 1 |
| C3 | `experiments/g6_studio_path_a/aggregator_h9.py` | 0 | 1 |
| C3 | `experiments/g6_studio_path_a/dream_episode_real.py` | 6 | 10 |
| C3 | `experiments/h1_split_mnist/run_h1.py` | 1 | 1 |
| C3 | `experiments/nh03_ttt_vs_episodes/run_nh03.py` | 6 | 10 |
| C3 | `experiments/nh03_ttt_vs_episodes/ttt_classifier.py` | 4 | 6 |
| C3 | `scripts/ablation_cycle3_diffusion.py` | 1 | 1 |
| C3 | `scripts/conformance_matrix.py` | 3 | 3 |
| C3 | `scripts/e1_harness.py` | 13 | 17 |
| C3 | `scripts/e1_profiles.py` | 2 | 7 |
| C3 | `scripts/e1_small_determinism_check.py` | 0 | 1 |
| C3 | `scripts/e1_small_null_supplement.py` | 1 | 1 |
| C3 | `scripts/pilot_b6_lora_smoke.py` | 6 | 7 |
| C3 | `scripts/pilot_cycle3_cuda.py` | 4 | 4 |
| C3 | `scripts/pilot_cycle3_real.py` | 18 | 21 |
| C3 | `scripts/pilot_cycle3_sanity.py` | 11 | 14 |
| C3 | `scripts/pilot_phase2b_neuromorph.py` | 17 | 20 |
| C3 | `scripts/run_q2_conformance_audit.py` | 1 | 1 |

### 8.3 Normative docs (N1-N4)

| Class | File | `downscale` | all-case |
|---|---|---|---|
| N1 | `docs/axioms/AXIOMS.md` | 5 | 6 |
| N1 | `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` | 17 | 17 |
| N1 | `docs/specs-fr/2026-04-17-dreamofkiki-master-design.md` | 2 | 2 |
| N1 | `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` | 17 | 17 |
| N1 | `docs/specs/2026-04-17-dreamofkiki-master-design.md` | 2 | 2 |
| N1 | `docs/specs/amendments/2026-04-21-dr2-empirical-falsification.md` | 0 | 1 |
| N2 | `docs/papers/paper1-fr/abstract.md` | 1 | 1 |
| N2 | `docs/papers/paper1-fr/background.md` | 7 | 7 |
| N2 | `docs/papers/paper1-fr/blog.md` | 8 | 8 |
| N2 | `docs/papers/paper1-fr/discussion.md` | 2 | 2 |
| N2 | `docs/papers/paper1-fr/full-draft.md` | 21 | 22 |
| N2 | `docs/papers/paper1-fr/outline.md` | 2 | 2 |
| N2 | `docs/papers/paper1-fr/results-section.md` | 2 | 2 |
| N2 | `docs/papers/paper1/abstract.md` | 1 | 1 |
| N2 | `docs/papers/paper1/background.md` | 7 | 7 |
| N2 | `docs/papers/paper1/discussion.md` | 2 | 2 |
| N2 | `docs/papers/paper1/full-draft-fr.md` | 13 | 14 |
| N2 | `docs/papers/paper1/full-draft.md` | 24 | 25 |
| N2 | `docs/papers/paper1/outline.md` | 2 | 2 |
| N2 | `docs/papers/paper1/results-section.md` | 2 | 2 |
| N2 | `docs/papers/paper2-fr/architecture.md` | 5 | 5 |
| N2 | `docs/papers/paper2-fr/background.md` | 2 | 2 |
| N2 | `docs/papers/paper2-fr/conformance-section.md` | 1 | 1 |
| N2 | `docs/papers/paper2-fr/full-draft.md` | 3 | 3 |
| N2 | `docs/papers/paper2-fr/introduction.md` | 1 | 1 |
| N2 | `docs/papers/paper2-fr/outline.md` | 1 | 1 |
| N2 | `docs/papers/paper2-fr/results.md` | 3 | 17 |
| N2 | `docs/papers/paper2/architecture.md` | 5 | 5 |
| N2 | `docs/papers/paper2/background.md` | 2 | 2 |
| N2 | `docs/papers/paper2/conformance-section.md` | 1 | 1 |
| N2 | `docs/papers/paper2/full-draft.md` | 5 | 5 |
| N2 | `docs/papers/paper2/introduction.md` | 1 | 1 |
| N2 | `docs/papers/paper2/outline.md` | 1 | 1 |
| N2 | `docs/papers/paper2/results.md` | 3 | 17 |
| N3 | `docs/proofs/dr-0.md` | 1 | 1 |
| N3 | `docs/proofs/dr2-compositionality.md` | 17 | 19 |
| N3 | `docs/proofs/dr4-profile-inclusion.md` | 7 | 8 |
| N3 | `docs/proofs/op-pair-analysis.md` | 22 | 22 |
| N3 | `docs/proofs/r1-cross-machine.md` | 1 | 1 |
| N4 | `STATUS.md` | 2 | 2 |
| N4 | `docs/invariants/registry.md` | 1 | 1 |
| N4 | `docs/research/2026-05-02-sleep-consolidation-state-of-the-art.md` | 1 | 1 |
| N4 | `ops/formal-reviewer-email-template.md` | 3 | 3 |

### 8.4 Dated and historical documents (D1-D3)

Counts in 8.1: 94 files (pre-registrations and filing copies, dated
milestones and results, superpowers plans/specs, CHANGELOG). Never
edited; the amendment re-labels through section 6.

## 9. Decisions left to the PI

Recommended option first. Each decision is closed with date and value
before the freeze.

**D1. Timing against the reading lock and the roadmap.**
(a) Merge as L0 draft now; tag L1 only after the reading lock items are
cleared, aligned with `ROADMAP.md:130,163` (Paper 1 reformulation, Q2).
Issue #48 stays open past M1. (b) Tag by 2026-11-29 under an explicit
dated derogation to principle 6 (ADR on the model of ADR-0001).
(c) Tag at merge without derogation (violates principle 6; not
recommended).

**D2. What the L1 tag freezes.**
(a) The text of the reformulated hypotheses and their stated
predictions only; no data, no new protocol. (b) Also the protocols of
the tests that would check them (R9 H1 bits instrumentation, H10
traffic, NH-05 sign cross-over), as separate preregistrations tagged
together. (c) Vocabulary and renames only (no new content).

**D3. DR-0 budget components and the selection clause.**
(a) Vector (B_bits, B_traffic, T); joules measured, outside the
statement; `select` a declared policy parameter. (b) Write count only
(the existing `WriteLedger`), simplest and matching the NH-03 evidence.
(c) Bits only (Landauer): contradicted by R9 H10. (d) Include joules:
breaks substrate-agnosticism (R9:10).

**D4. Domain of the rescale factor and the magnitude guard.**
(a) f in (0, ∞), "sign" meaning the sign of log f; a per-profile cap
[f_min, f_max] keeps S2 / `I-Wmag` enforceable. (b) Keep (0, 1] as the
default profile value and allow f > 1 only through an explicit flag in
a preregistered run. (c) f in ℝ (negative factors): no source in the
review supports it.

**D5. Naming and compatibility.**
(a) `rescale` primary; `downscale` and `Operation.DOWNSCALE` kept as
permanent aliases for f ≤ 1, serialised op-name string unchanged.
(b) Hard rename with a deprecation window: breaks frozen scripts and
log hashes. (c) Keep `downscale` and add `rescale` as a fifth
operation: changes the monoid, DR-2 and DR-4 op sets; not recommended.

**D6. DR-1 weight and threshold.**
(a) `weight(e)` and `θ` declared per profile and preregistered; purge
optional; I1 restated as "episodic primacy". (b) Selective DR-1 but I1
and its purge gate unchanged for non-selected episodes. (c) Keep DR-1
as is and add DR-1' as a separate hypothesis.

**D7. Scope of the vocabulary change.**
(a) Prose and docs say "framework hypothesis"; identifiers (`Axiom`,
`kiki_oniric.axioms`, DR-n) unchanged, optional alias
`FrameworkHypothesis`. (b) Full rename including module and test
directory paths. (c) Documentation only, no alias.

**D8. DualVer bump.**
(a) Formal axis MINOR (C-v0.26.0), applied by the follow-up spec PR:
statements generalise, old ones are special cases. (b) Formal axis
MAJOR: DR-0 and DR-1 change meaning. (c) PATCH, as for the DR-2
weakening (C-v0.7.1): not recommended since a statement is rewritten
rather than weakened.

## 10. Not verified

- Whether all G-series runners schedule DEs as bounded slots (section 6.1).
- The SWHID, tag date and manifest hash (nothing frozen).
- Primary literature (WISE, Titans, Denis 2026, Landauer-based claims):
  taken from the review files above, not re-read here.
- Whether `docs/proofs/dr-0.md` / `dr-1.md` need a dated note or a
  rewrite (follow-up docs PR).
