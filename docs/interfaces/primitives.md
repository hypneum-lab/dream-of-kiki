# Primitives Interface Contract

**Version** : C-v0.3.1+STABLE (draft, bump to C-v0.5.0 at S4.2)
**Owners** : Track C (design) + Track A (implementation)
**Locked** : S2 → S4 (evolves with framework)

Any substrate instantiating the dreamOfkiki framework MUST implement
these 8 typed Protocols. Reference: `kiki_oniric/core/primitives.py`.

---

## Awake → Dream primitives

### α — AlphaStreamProtocol (P_max only)

Raw forward-pass traces firehose.

**Python Protocol**: `kiki_oniric.core.primitives.AlphaStreamProtocol`

**Methods**:
- `append_trace(tokens, activations, attention, errors) -> None`
- `iter_traces() -> Iterator[dict[str, NDArray]]`

**Storage contract**: ring buffer, LIFO rotation, mmap-backed, cap
10 GB (configurable per profile).

**Activation**: P_max only. Calling in P_min/P_equ profile is a
logic error (caller must check profile).

### β — BetaBufferProtocol (all profiles)

Curated episodic buffer with saillance-gated append.

**Python Protocol**: `kiki_oniric.core.primitives.BetaBufferProtocol`

**Methods**:
- `append_record(context, outcome, saillance, ts) -> record_id`
- `fetch_unconsumed(limit) -> list[dict]`
- `mark_consumed(record_ids, de_id) -> None`

**Storage contract**: SQLite append-log with index on
`(saillance DESC, consumed_by NULLS FIRST)`.

**Invariant I1 contract**: every record must have
`consumed_by_DE_id` set before purge. Enforcement: hourly cron
(T-Ops.6) + purge policy.

### γ — GammaSnapshotProtocol (fallback)

Weights-only snapshot pointer.

**Python Protocol**: `kiki_oniric.core.primitives.GammaSnapshotProtocol`

**Methods**:
- `get_checkpoint_path() -> Path`
- `get_checkpoint_sha256() -> str`

**Usage**: fallback / diagnostic. P_min/P_equ/P_max rely on β+δ
primarily.

### δ — DeltaLatentsProtocol (P_equ, P_max)

Hierarchical latent snapshots across ortho species.

**Python Protocol**: `kiki_oniric.core.primitives.DeltaLatentsProtocol`

**Methods**:
- `snapshot(species_activations) -> snapshot_id`
- `get_recent(n) -> list[dict[str, NDArray]]`

**Storage contract**: ring buffer N=256 snapshots.

---

## Dream → Awake channels

### Canal 1 — WeightDeltaChannel

Parametric consolidation output, applied via swap.

**Python Protocol**: `kiki_oniric.core.primitives.WeightDeltaChannel`

**Methods**:
- `apply(lora_delta, fisher_bump=None) -> None`

**Constraint**: must satisfy invariants S1 (retained non-regression)
+ S2 (finite values). Enforcement: swap protocol guards.

### Canal 2 — LatentSampleChannel

Generative replay / data augmentation queue.

**Python Protocol**: `kiki_oniric.core.primitives.LatentSampleChannel`

**Methods**:
- `enqueue(species, latent_vector, provenance) -> None`
- `dequeue() -> dict | None`

**Constraint**: must satisfy invariant I3 (distributional drift
bounded, KL ≤ ε_drift).

### Canal 3 — HierarchyChangeChannel

Topology diff applied atomically at swap time.

**Python Protocol**: `kiki_oniric.core.primitives.HierarchyChangeChannel`

**Methods**:
- `apply_diff(diff: list[tuple[str, dict]]) -> None`

**Constraint**: must satisfy invariant S3 (validate_topology
passes on post-state).

### Canal 4 — AttentionPriorChannel

Meta-cognitive attention guidance.

**Python Protocol**: `kiki_oniric.core.primitives.AttentionPriorChannel`

**Methods**:
- `set_prior(prior: NDArray) -> None`
- `get_prior() -> NDArray | None`

**Constraint**: must satisfy invariant S4 (each component in [0,1],
sum ≤ budget_attention).

---

## Contract test suite

Contract tests are in `tests/conformance/axioms/` and
`tests/conformance/invariants/`. Each protocol must pass:

1. **Signature check** (`test_dr3_substrate.py`): runtime-checkable,
   8 protocols declared.
2. **Axiom check** (S5+): DR-0..DR-4 property tests pass for concrete
   implementations.
3. **Invariant check** (S5+): I1/I2/I3/S1/S2/S3/S4/K1/K3/K4
   enforceable on runtime.

Failing any contract test blocks the substrate from claiming
framework conformance (DR-3 Conformance Criterion, framework §6.2).
