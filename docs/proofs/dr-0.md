# DR-0 Accountability — operational evidence

**Version** : v0.1-draft (2026-05-02)
**Supersedes** : none (initial issue ; closes the symmetry gap with
DR-2 / DR-3 / DR-4 evidence files)
**Amendment pointer** : none (DR-0 statement unchanged since
framework-C v0.7.0 ; this is documentation only, no FC bump)
**Target venue** : PLOS Computational Biology (Paper 1 v0.2)
**Executable counterpart** :
`tests/conformance/axioms/test_dr0_accountability.py` (full suite)

---

**Status** : operational evidence (constructive), not a formal proof.

DR-0 (spec §6.2) is a **design axiom**, not a theorem : it constrains
the runtime API surface so that no dream-output escapes into the
awake state without leaving a bounded, finite-budget trace. This
document records the **constructive evidence** that the kiki-oniric
substrate satisfies DR-0 by construction of its API, plus the
property checks that exercise the guarantee. It is the symmetric
counterpart to `dr3-substrate-evidence.md` for runtime-invariant
axioms : DR-0 has no closed-form proof obligation (unlike DR-2 and
DR-4) — its content is "the only path from a dream computation to
an awake-state mutation goes through a logged DreamEpisode with a
finite BudgetCap".

## Axiom statement (spec §6.2)

```
DR-0 (Accountability)
∀ δ ∈ dream_output_channels,
∃ DE ∈ History : budget(DE) < ∞ ∧ δ ∈ outputs(DE)
```

**Interpretation** (spec §6.2) : no "ambient dreaming" — every dream
contribution is traceable to a bounded DE. **Enforcement** : I1 + I2
+ K1 combined.

## Operationalisation

DR-0 splits into two operational obligations on the substrate :

1. **Traceability (∃ DE ∈ History)** — every output emitted on a
   channel 1..4 must originate from a DreamEpisode recorded in the
   runtime log.
2. **Finiteness (budget(DE) < ∞)** — the DreamEpisode's budget must
   be a triple of non-negative, finite resource counters
   `(flops, wall_time_s, energy_j)`.

Discharged separately : traceability by `DreamRuntime.execute()`'s
log-on-`finally` discipline, finiteness by `BudgetCap.__post_init__`'s
rejection of `NaN`/`Inf`/negative values.

## Evidence (constructive)

### E1 — Traceability by API construction (DR-0 ① )

`kiki_oniric/dream/runtime.py:60-98` defines the **only** call path
through which a DreamEpisode produces awake-state side-effects :

- The runtime exposes a single mutating entry point, `execute(ep)`.
- `execute()` wraps the per-operation handler dispatch in a
  `try/except/finally` block whose `finally` clause unconditionally
  appends an `EpisodeLogEntry` to `self._log`.
- The append is **unconditional** : success path
  (`completed=True`), handler-exception path (`completed=False`,
  `error` populated, exception re-raised), and partial-execution
  paths (only ops attempted up to the failing one are recorded).
- Configuration errors (no handler registered) raise
  `NotImplementedError` *before* the `finally` scope is entered and
  do not mutate the runtime — symmetric contract : no log entry ⇒
  no handler ran ⇒ no awake-state mutation.

DR-0 ① is therefore **structurally satisfied** : channel-1..4 outputs
are only produced by registered handlers, handlers run exclusively
inside the logged scope, and the `finally` clause guarantees the
log entry exists whenever a handler ran.

### E2 — Finiteness by dataclass invariant (DR-0 ② )

`kiki_oniric/dream/episode.py:34-60` defines `BudgetCap` as a frozen
dataclass whose `__post_init__` rejects :

- negative `flops` (must be ≥ 0),
- negative `wall_time_s` / `energy_j` (must be ≥ 0),
- non-finite `wall_time_s` / `energy_j` (`math.isfinite` check
  rejects `NaN` and `±Inf`).

A `DreamEpisode` cannot be instantiated without a `BudgetCap`, and
the cap cannot be mutated post-construction (frozen dataclass).
Therefore every `DreamEpisode` reaching the runtime carries a
finite, non-negative budget triple — DR-0 ② is **enforced at the
type boundary**, not at execution time.

### E3 — Episode identity for History indexing

`DreamEpisode.episode_id: str` is a required field
(`kiki_oniric/dream/episode.py:74`). The runtime preserves it
verbatim into the corresponding `EpisodeLogEntry.episode_id`
(`kiki_oniric/dream/runtime.py:93`). The `History` set referenced by
the axiom is concretely instantiated as `DreamRuntime.log` and is
indexable by `episode_id` — the existential `∃ DE ∈ History` is thus
operationalised as a containment check against the log.

### E4 — Public axiom registry

`kiki_oniric/axioms/__init__.py:111-124` exposes DR-0 as a frozen
`Axiom` dataclass with `predicate=None` (no closed-form predicate,
by design) and `test_references` pinned at the canonical property
suite. Downstream code (`harness/`, `nerve-wml`, `micro-kiki`)
queries DR-0 via this stable surface ; the axiom statement is never
inlined.

## Test correspondence

The conformance suite
`tests/conformance/axioms/test_dr0_accountability.py` covers each
clause of the axiom :

| Test | Covers | Notes |
|------|--------|-------|
| `test_dr0_every_executed_de_has_log_entry` | DR-0 ① traceability | Hypothesis property, 50 examples : registers a no-op handler, calls `runtime.execute(ep)`, asserts `ep.episode_id` appears in `runtime.log`. Witnesses the `finally`-append discipline of E1. |
| `test_dr0_budget_is_finite` | DR-0 ② finiteness (positive) | Hypothesis property, 50 examples : asserts every constructible `BudgetCap` has non-negative + finite components. Witnesses E2 on the success path. |
| `test_budget_cap_rejects_non_finite_values` | DR-0 ② finiteness (negative) | Direct construction with `math.nan` / `math.inf` for `wall_time_s` is rejected by `__post_init__`. Witnesses E2 on the rejection path — guarantees no `BudgetCap(NaN, ·, ·)` can ever exist. |

The Hypothesis strategy `dream_episodes_with_replay_only` is
intentionally narrow (replay-only, channel-1 only) : the DR-0
obligation is independent of the operation set — the `finally`-
append in `execute()` fires for every registered op. Broader op
strategies belong to DR-2 and DR-3 suites.

## Limitations

- **Skeleton runtime (S5.2).** The single-threaded scheduler will
  be superseded by a concurrent swap runtime in S7+. The DR-0
  contract on the new runtime must preserve the `finally`-append
  discipline ; any move to async dispatch must keep the log
  append in a `finally` observing the per-task scope.
- **No real handlers yet.** `noop_handler` is a placeholder — the
  constructive evidence does not depend on handler bodies (the
  `finally` fires regardless), but DR-0 empirical breadth will
  widen once `replay`/`downscale`/`restructure`/`recombine` land
  (S5.4+).
- **E-SNN synthetic substitute.** The E-SNN substrate shares the
  runtime + log via the typed-Protocol registry ; DR-0 evidence on
  E-SNN is inherited from the same `DreamRuntime` scaffolding
  (cf. `dr3-substrate-evidence.md`). A native LIF scheduler (cycle
  3) will require re-stating E1 against its logging discipline.

## Cross-references

- Spec §6.2 (axiom statement) :
  `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
- Public axiom : `kiki_oniric/axioms/__init__.py` (`DR0`)
- Runtime (E1) : `kiki_oniric/dream/runtime.py`
- BudgetCap / DreamEpisode (E2, E3) : `kiki_oniric/dream/episode.py`
- Conformance suite :
  `tests/conformance/axioms/test_dr0_accountability.py`
- Sibling evidence document : `docs/proofs/dr3-substrate-evidence.md`
- Companion proof for DR-4 : `docs/proofs/dr4-profile-inclusion.md`
