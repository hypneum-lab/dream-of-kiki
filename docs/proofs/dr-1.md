# DR-1 Episodic Conservation — operational evidence

**Version** : v0.1-draft (2026-05-02)
**Supersedes** : none (initial issue ; closes the symmetry gap with
DR-2 / DR-3 / DR-4 evidence files)
**Amendment pointer** : none (DR-1 statement unchanged since
framework-C v0.7.0 ; this is documentation only, no FC bump)
**Target venue** : PLOS Computational Biology (Paper 1 v0.2)
**Executable counterpart** :
`tests/conformance/axioms/test_dr1_episodic_conservation.py` (full
suite, S5.3 fake β skeleton ; real β substitution at S7+).

---

**Status** : operational evidence (S5.3 fake-buffer skeleton ;
real β implementation lands S7+ alongside the swap protocol).
**Axiom** : DR-1 (framework spec §6.2). **Invariant** : I1
(framework spec §5.1).

DR-1 is the **formal contract** stating that every curated
episodic record gets a chance at consolidation within a finite
horizon. I1 is its **runtime enforcement assertion** — a query
periodically evaluated against the β buffer's `consumed_by`
column. This document records the evidence linking the two.

---

## 1. Axiom statement (spec §6.2)

```
DR-1 (Episodic conservation, formalizes I1)
  ∀ e ∈ β_t, ∃ t' ∈ [t, t + τ_max] : e ∈ inputs(DE_{t'})
```

**Interpretation** : every record `e` that enters the curated
episodic buffer `β` at time `t` must be consumed as input by
some `DreamEpisode` no later than `t + τ_max`. The axiom
formalises invariant I1 (§5.1) and is one of the three
enforcement legs of DR-0 (§6.2 : "Enforcement : I1 + I2 + K1
combined").

## 2. Operationalisation (axiom → invariant I1)

The axiom is a contract over an unbounded ω-trace ; the runtime
checks a finite witness via I1 :

```
I1 : SELECT COUNT(*) FROM β
     WHERE consumed_by IS NULL
       AND created_at < now() − τ_max
     == 0
```

Three structural facts make the axiom decidable from I1 :

1. **Bounded buffer.** `BetaBufferProtocol` (see
   `kiki_oniric/core/primitives.py` lines 53–71) exposes
   `append_record / fetch_unconsumed / mark_consumed` with a
   single mutable column `consumed_by`. The buffer is finite at
   any wall-clock instant.
2. **Finite horizon.** `τ_max` is a pinned scalar (configurable
   per C-version, default in §5.1). It is **not** unbounded
   under any profile.
3. **Stable consumption witness.** Once `mark_consumed(rec_ids,
   de_id)` writes a non-null `consumed_by`, the record is
   permanently witnessed by `de_id` ; the witness cannot be
   revoked because β rows are append-log semantics
   (`AppendLog<EpisodicRecord>`, §3.2).

Consequently `I1 == 0` over a sliding window of width `τ_max`
on the β table is equivalent to DR-1 holding on the trace
restricted to that window. Cron periodicity (≤ τ_max / 2)
turns the equivalence into total coverage.

## 3. Evidence

### 3.1 FIFO-fairness of consumption

`fetch_unconsumed(limit)` (primitive contract) returns a
prefix of unconsumed records ordered by insertion, and the
S5.3 skeleton (`FakeBetaBuffer.consume`,
`tests/conformance/axioms/test_dr1_episodic_conservation.py`
lines 32–36) consumes that prefix without skipping. No
unconsumed record can therefore be perpetually overtaken by
later arrivals : every record is consumed within at most
`⌈|β_unconsumed| / batch_size⌉` DEs.

### 3.2 τ_max bound from buffer + budget arithmetic

Let `R` be the rate of valid β appends (saillance > threshold)
and `C` the per-DE consumption batch (`mark_consumed` cardinality).
DR-0 + K1 guarantee a bounded inter-DE wall time `Δ_DE`. The
per-record latency is therefore at most :

```
latency(e) ≤ ⌈|β_unconsumed_at_t(e)| / C⌉ · Δ_DE
```

For DR-1 to hold it suffices that the runtime sizes `C` and
schedule `Δ_DE` such that the right-hand side is dominated by
`τ_max`. This is the operational duty of T-Ops (cron + DE
scheduler) ; the axiom does not constrain `R, C, Δ_DE`
individually, only their joint product.

### 3.3 Purge gate

`Enforcement : I1 runtime check + β purge gate` (spec §6.2
DR-1). A record may only be evicted from β once
`consumed_by IS NOT NULL`. If I1 returns a non-zero count, the
purge gate refuses and the record stays in β past `τ_max`,
surfacing as a BLOCKING invariant violation rather than a
silent loss. This converts any DR-1 failure into an audit
event (cf. parallel pattern with `aborted-swaps/` for S1/S2).

## 4. Test correspondence

`tests/conformance/axioms/test_dr1_episodic_conservation.py`
exercises the operational core of the axiom on a fake β :

| Clause of the evidence | Test assertion |
|------------------------|----------------|
| §3.1 FIFO-fairness, no record skipped | `assert all(r.consumed_by is not None for r in buf.records)` (line 59) |
| §3.2 latency bounded by `⌈n/C⌉` | `assert de_counter <= expected_max_des` (line 62) |
| Hypothesis-driven exploration of `(record_count, batch_size) ∈ [1,100]×[1,50]` | `@given(...) @settings(max_examples=30)` (lines 39–43) |

The test uses `FakeBetaBuffer` rather than the real SQLite β ;
this is the single sanctioned skeleton fake (cf.
`tests/conformance/axioms/CLAUDE.md`, "S5.3 `FakeBetaBuffer` is
the single tolerated exception"). The real β substitution
arrives with the swap protocol in S7+.

## 5. Limits

- **Skeleton fake.** DR-1 is currently witnessed against an
  in-memory list, not the SQLite β specified in §3.2. The
  proof-test mapping above is therefore evidence for DR-1
  *given* a buffer that honours the `BetaBufferProtocol`
  contract ; the real-substrate evidence will be reissued at
  S7+ after the swap protocol lands.
- **No empirical τ_max calibration.** §3.2 reduces DR-1 to a
  scheduling bound but does not pin the constants `R, C,
  Δ_DE` ; calibration against pilot runs is deferred to G2/G4.
- **No formal proof.** Unlike DR-2 (`dr2-compositionality.md`
  v0.2) and DR-4 (`dr4-profile-inclusion.md`), DR-1 is
  recorded here as operational evidence, not as a structural
  proof. A mechanised version is deferred to cycle 3 if
  reviewers request it (cf. `proofs/CLAUDE.md`).

## 6. Cross-references

- Spec §5.1 (I1) and §6.2 (DR-1) : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
- Invariant registry entry : `docs/invariants/registry.md` I1
- Primitives contract : `kiki_oniric/core/primitives.py` (`BetaBufferProtocol`)
- Runtime / DE log : `kiki_oniric/dream/runtime.py`, `kiki_oniric/dream/episode.py`
- Conformance test : `tests/conformance/axioms/test_dr1_episodic_conservation.py`
- Proofs index : `docs/proofs/__init__.md`
