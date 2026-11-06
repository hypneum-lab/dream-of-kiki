# ADR-0001: dated derogation to roadmap principles 4 and 6 for the E1-small controls run and its concurrency with P8c on KXKM (2026-11-06)

> Status: **Proposed — awaiting PI signature** (section 9). PI decision
> 2026-11-06, QCM ~20:50 CET (question Q4, option (a), of the
> 2026-11-06 hypotheses refresh, hypneum-lab/.github PR #30, report §7
> Q4, findings N-13 and N-16). Drafted by an agent; not signed.
>
> Model: nerve-wml ADR-0003, "reading-gate exception for P1bis and P8c"
> (`docs/adr/0003-reading-gate-exception-p1bis-p8c.md`, hypneum-lab/nerve-wml
> PR #118, head `2d2bf0f`, open and **unsigned** at the time of writing).
>
> Roadmap cited at hypneum-lab/.github `docs/ROADMAP.md` v2.1, commit
> `c8967a2`, as `RM:<line>`.
>
> This ADR was drafted from process records only (PR bodies, the launch
> record of the addendum, process listings already reported). No score,
> verdict file or results report of either campaign was opened to write
> it.

This is the first ADR of this repository. It follows the ADR convention
of nerve-wml (`docs/adr/NNNN-*.md`) because the decision is a laboratory
process decision, not an OSF amendment: it changes no registered state of
any preregistration and is not meant for an OSF amendment field.

## 1. Context

### 1.1 The rules met (quoted verbatim)

- `RM:11`, principle 3: « **Une campagne, un verdict.** Pas de nouveau
  **gel** tant que la précédente campagne n'a pas rendu son verdict ; la
  lecture n'est pas un gel (spec:140). Exception inchangée : un addendum
  de contrôle sur une campagne en cours. »
- `RM:12`, principle 4: « **Le calcul suit les machines.** Une seule
  campagne lourde à la fois. Aucune nouvelle campagne GPU avant S6. CPU
  léger sur electron-server, CPU lourd sur KXKM en `nice 19`, jamais sur
  GrosMac. »
- `RM:14`, principle 6: « **Lire avant de geler.** Aucun tag de gel sans
  note de lecture, contradicteur et trois tests passés (saturation,
  recoupement, couverture ; spec §4.2). Le verrou s'applique à toute
  campagne lourde (GPU, ou > 6 h CPU, ou > 1 semaine calendaire) et à
  toute préreg L3. Les campagnes déjà gelées (E1-small, N10-C run 2, Me6,
  GB300 g7) ne sont pas re-soumises. »
- `RM:57`, PI decision of 2026-11-06 13:58: the two missing E1-small GPU
  controls « sont lancés sur KXKM dès le 06/11 par **addendum de
  contrôle** (principe 3, exception « addendum sur une campagne en
  cours »), sans gel nouveau. »
- `RM:190` (out of scope this quarter): « … pas plus d'une campagne
  lourde à la fois. »
- `RM:198` (KPI table): « Tags de gel posés sans verrou | 0 ».

S6 starts on 2026-11-30 (`RM:103`).

### 1.2 Chronology (CET unless marked Z)

| When | Event | Record |
|---|---|---|
| 2026-11-01 20:20 | E1-small run stops on error after 1 271 of 1 410 cells; the 9 registered profiles are complete | `RM:42` |
| 2026-11-01 21:32 | Roadmap v2 "under the reading gate": principles 4 and 6 in force | .github `8014889` (per ADR-0003 §1.2) |
| 2026-11-04 18:22:10 | nerve-wml tag `prereg-hopfield-p1bis-p8c-freeze-20261104` | nerve-wml ADR-0003 §1.2 |
| 2026-11-05 19:33:59Z | nerve-wml tag `prereg-hopfield-p8c-relaunch-20261105` | nerve-wml ADR-0003 §1.2 |
| 2026-11-05 19:34:32Z | P8c-nocap N=16384 relaunch on the KXKM-AI CPU: 8 workers, `p8c_nocap --confirmatory`, wall cap 120 h (2026-11-10 19:34Z) | nerve-wml ADR-0003 §1.2, §6 |
| 2026-11-06 13:58 | PI decision: launch the two E1-small GPU controls on KXKM as a controls addendum | `RM:57`; PR #92 addendum, header |
| 2026-11-06 14:17:45 | Launch of the controls run on KXKM-AI GPU (tmux `e1s-controls`, commit `8c1d57a`); P8c running on the CPU at the same time | PR #92, `docs/osf-addendum-e1-small-controls-2026-11-06.md` §7 |
| 2026-11-06 14:26 | `controls` stage starts (564 cells); ETA ≈ 28-49 h, end between 2026-11-07 ~18:30 and 2026-11-08 ~16:00 | same, §7 "ETA" |
| 2026-11-06 17:12 | PI decision recorded in ADR-0003 for P1bis/P8c; ADR-0003 left unsigned | nerve-wml PR #118 |
| 2026-11-06 18:57 | Read-only state: KXKM GPU runs `pilot_cycle3_cuda.py` (PR #92, 21.8 / 24.6 GB); KXKM CPU runs 8 `p8c_nocap --confirmatory` workers | refresh inventory, compute status (.github PR #30) |
| 2026-11-06 ~20:50 | PI decision Q4 (a): this ADR | .github PR #30, report §7 Q4 |

### 1.3 Facts

- F1. PR #92 ("docs(e1-small): controls addendum and launch record") is
  **open, unmerged** (head `9333898`). Its addendum declares itself « un
  addendum de contrôle d'une campagne déjà gelée (principe 3 …) » and
  « pas un nouveau gel » (header).
- F2. The run of PR #92 is a GPU run on KXKM-AI, launched on 2026-11-06,
  before S6 (2026-11-30), with an ETA of 28-49 h (§1.2).
- F3. The E1-small campaign had stopped on 2026-11-01 (`RM:42`); its
  registered analysis was read and merged on 2026-11-06 (PR #91,
  `RM:57`).
- F4. No freeze tag was posed for the PR #92 run (no tag of this
  repository matches E1-small controls at the time of writing).
- F5. No reading note, no blind contradictor and no dated watch round
  exist for the PR #92 run.
- F6. From 2026-11-06 14:17:45 onwards, two runs share KXKM-AI: the PR
  #92 GPU run and the P8c-nocap CPU relaunch (8 workers, `--confirmatory`
  flag, wall cap 2026-11-10 19:34Z). The PR #92 launch record notes
  contention: seed-0 cells took 248-404 s against a 177 s mean in
  E1-small, « la contention CPU de P8c (4 cœurs pleins) est la cause
  probable, non isolée » (§7).
- F7. nerve-wml ADR-0003 records the PI decision of 2026-11-06 17:12 for
  P1bis/P8c, excludes confirmatory language for both studies, and its
  signature box (section 8) is empty.

### 1.4 Inferences (not facts)

- I1. The exception of principle 3 bears on **freezes** (`RM:11`: « Pas
  de nouveau gel … Exception … »). It does not cover the rule the PR #92
  run meets, which is principle 4, « Aucune nouvelle campagne GPU avant
  S6 », stated without exception (`RM:12`). `RM:57` therefore invokes an
  exception that does not cover that rule (report §4 C-6).
- I2. The PR #92 run is a heavy campaign in the sense of `RM:14` (GPU,
  and by its ETA more than 6 h). Whether a controls addendum of an exempt
  campaign (E1-small is in the closed list) needs the reading gate is not
  decided by the roadmap text; this ADR does not decide it either and
  derogates so that either reading is covered.
- I3. P8c-nocap is a heavy campaign on any reading of `RM:14` (8 CPU
  workers up to 120 h; ADR-0003 §1.3). With the PR #92 GPU run, two
  heavy campaigns run at once, against « Une seule campagne lourde à la
  fois » (`RM:12`) and `RM:190`.
- I4. One of the two concurrent runs carries a `--confirmatory` flag
  while its exception (ADR-0003) is unsigned; a confirmatory reading of
  it is not receivable before that signature (report N-16).

## 2. Decision

1. A **dated derogation** to principles 4 and 6 of the roadmap is
   recorded for:
   - (a) the PR #92 run (E1-small controls: H14-B same-norm arm and
     "options only" control, KXKM-AI GPU, launched 2026-11-06 14:17:45,
     ETA 28-49 h), against « Aucune nouvelle campagne GPU avant S6 »
     (principle 4) and against the reading gate of principle 6;
   - (b) the **concurrency** of that run with the P8c-nocap relaunch
     (`p8c_nocap --confirmatory`, 8 CPU workers on KXKM-AI), against
     « Une seule campagne lourde à la fois » (principle 4, `RM:190`), for
     as long as both runs are up.
2. The P8c half of (b) rests on nerve-wml ADR-0003, which this ADR does
   not replace: the gate exception for P8c itself stays in ADR-0003.
3. Both ADRs are to be signed by the PI: this one (section 9) and
   nerve-wml ADR-0003 (its section 8).
4. The derogation is dated, names its two objects, and is **not a
   precedent**. It does not cover anything else (section 4).

## 3. Consequences

1. **No reading before both signatures.** No value produced by the PR
   #92 run (`pre`, `post`, `Δ`, score, accuracy, verdict) and no value
   of P8c-nocap is read before both this ADR and nerve-wml ADR-0003 are
   signed. Progress and integrity fields (cell counts, `rc`, memory,
   invariant checks, hashes) may be read, as in the PR #92 launch record.
2. **No "confirmatory" reading of P8c** before both signatures; after
   them, P8c results keep the ADR-0003 label « frozen prereg, gate not
   passed » and its claim limits (ADR-0003 §3). The `--confirmatory`
   flag of the command line does not make a result confirmatory.
3. **The running campaigns are not stopped.** Neither run is stopped,
   reniced or changed because of this ADR; no frozen parameter of
   E1-small, of the PR #92 addendum or of P8c-nocap changes.
4. **PR #92 results stay exploratory and informed**, as its addendum
   already states (§1); this ADR adds the reporting line « run under
   dated derogation to roadmap principles 4 and 6 (dream-of-kiki
   ADR-0001) » to any result, table or note drawn from it.
5. **Contention is a declared caveat.** Any timing, throughput or
   numerical-determinism remark on either run mentions the concurrency
   (F6); no claim is made here that the concurrency changes any score.
6. **KPI.** The PI decision asks that the KPI « campagnes sans verrou »
   be incremented. The roadmap has no KPI by that name; its closest line
   is « Tags de gel posés sans verrou » (`RM:198`), which counts tags,
   and the PR #92 run posed none (F4). Proposed for the next
   resume-point session, in hypneum-lab/.github and not in this
   repository: add the line « Campagnes lourdes lancées sans verrou ou
   par dérogation » with **1 for S2** (the PR #92 run), and leave the
   tag KPI at the value set by ADR-0003 (2 for S2). The PI chooses the
   wording at that session.
7. **`RM:57`** is to be amended at the same session so that it cites
   this derogation instead of the principle 3 exception.

## 4. What is not covered

- Any other run, on KXKM or elsewhere, including any relaunch, extension
  or new seed of the PR #92 controls after this run ends or stops.
- The P1bis/P8c gate exception itself (ADR-0003), and any new P8c arm.
- The N10-C-bis gate A+ development runs of S3-S5 (their own dated
  derogation to principle 4, `RM:56`, nerve-wml PR #114), and any
  overlap of those runs with the two runs named here.
- Any freeze. This ADR authorises no freeze tag and does not pass the
  reading gate for anything.
- Merging PR #92: its own review stays as stated in that PR.

## 5. Alternatives considered

- **Q4 (b): ratify `RM:57` as a dated derogation and sign ADR-0003 only
  before any P8c reading.** Not retained (PI choice): it leaves the
  concurrency without a written record and does not bar a reading of the
  PR #92 values before the signatures.
- **Treat the run as covered by the principle 3 exception.** Excluded by
  the text (I1; report §7 Q4, last bullet, relecture C-M5).
- **Stop one of the two runs.** Not proposed by the PI; the QCM options
  had no effect on the running campaigns (report §7 Q4 header).

## 6. State of the runs at drafting (process fields only)

As reported read-only on 2026-11-06 18:57 CET (.github PR #30, compute
status): KXKM GPU runs `pilot_cycle3_cuda.py` (PR #92), 21.8 / 24.6 GB,
end expected 2026-11-07/08; KXKM CPU runs 8 `p8c_nocap --confirmatory`
workers, wall cap 2026-11-10 19:34Z. Not re-read for this ADR. No score
was read.

## 7. References

- hypneum-lab/.github `docs/ROADMAP.md` v2.1, commit `c8967a2`, lines
  11, 12, 14, 42, 56, 57, 103, 190, 198.
- hypneum-lab/.github PR #30 (hypotheses refresh 2026-11-06), report §2
  findings N-13 and N-16, §4 C-6, §7 Q4.
- hypneum-lab/nerve-wml PR #118, `docs/adr/0003-reading-gate-exception-p1bis-p8c.md`
  (head `2d2bf0f`), §1.2, §1.3, §3, §6, §8.
- This repository, PR #92 (head `9333898`),
  `docs/osf-addendum-e1-small-controls-2026-11-06.md`, header, §1, §7;
  PR #91 (registered analysis, merged).

## 8. Change log

- 2026-11-06: drafted (agent), status Proposed.

## 9. PI signature

- [ ] PI signature (Clément Saillant)
- Date and time (CET):
