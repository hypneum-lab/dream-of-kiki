# R1 — cross-machine reproducibility qualification

**Version** : v0.1 (2026-05-20)
**Supersedes** : none — first dated qualification of R1's
cross-machine clause.
**Amendment pointer** : none — this is an evidence note that
refines wording without changing the formal axis. No
`docs/specs/amendments/` entry required (no FC bump).
**Target venue** : framework-C spec §8.3 wording refinement
(future revision); not paper-cited.
**Executable counterpart** :
`tests/reproducibility/test_r1_bit_exact.py`,
`tests/reproducibility/_r1_helpers.py:_chip_family`,
plus per-family golden files
`tests/reproducibility/golden_hashes_<family>.json`.

> **Scope note** : R1 is an *invariant* (Family R in
> `docs/invariants/registry.md`), not an axiom (DR-0..DR-4).
> This document lives under `docs/proofs/` as an evidence stub
> alongside the axiom proofs, matching the K2-coupling pattern
> (`docs/proofs/k2-coupling-evidence.md`) where an empirical CI
> refines an invariant's claim.

**Trigger** : 3-machine R1 probe at commit `0a8ec29` (Apple M5,
M3 Ultra, M1 Max) and matching MLX primitive probe.
**Source data** :
[`docs/milestones/r1-cross-machine-m5-vs-m1-2026-05-20.{md,json}`](../milestones/r1-cross-machine-m5-vs-m1-2026-05-20.md).
**Upstream** :
[ml-explore/mlx#3568](https://github.com/ml-explore/mlx/issues/3568).

---

## 1. Restated R1

Framework-C spec §8.3 defines R1 as:

> Every `MetricResult` is bit-identical reproducible from
> `(c_version, profile, seed, run_id, commit_sha,
> benchmark_version)` **for the metrics whose external
> dependencies are SHA-pinned**.

This document refines that statement with two qualifications
shown empirically on 2026-05-20.

## 2. The empirical findings

### 2.1 Within-machine bit-stability holds

On each of the three Apple Silicon machines tested (M5, M3 Ultra,
M1 Max), running `uv run pytest tests/reproducibility/` produces
identical hashes on repeated runs : 9 / 9 R1 tests PASS, no
within-run drift, no within-machine churn beyond the normal
`commit` metadata field tracking HEAD.

### 2.2 Cross-machine bit-stability is *conditional*

At the same commit (`0a8ec29`), same MLX 0.31.1, same Python
3.14.x, same code :

- **Ops that do not touch `mx.random.normal`** (`test_r1_downscale`,
  `test_r1_restructure`) → **bit-identical hashes across all
  three machines**.
- **Ops that route through `mx.random.normal` with a directly-
  constructed key** (`test_r1_replay`, `test_r1_recombine`,
  `test_r1_full_pipeline`) → **hashes diverge in a non-monotone
  cluster pattern**.

The MLX primitive probe confirms that the divergence is
isolated to `mx.random.normal` with a raw `mx.random.key(seed)`
input, on Apple M1 Max specifically. The PRNG key tensor, the
output of `mx.random.split`, and the output of `mx.random.uniform`
match across all three machines bit-for-bit.

## 3. Qualified R1 contract

R1 is therefore amended to:

> **R1 (within-machine bit-stable)** — every `MetricResult` is
> bit-identical reproducible from `(c_version, profile, seed,
> run_id, commit_sha, benchmark_version)` **on the same machine**,
> for the metrics whose external dependencies are SHA-pinned.
> Repeated runs on the same machine produce identical hashes.
>
> **R1 cross-machine (conditional)** — bit-identical hashes
> across two machines additionally requires :
>
> - same MLX version,
> - same chip family (Apple Silicon generation, as detected via
>   `sysctl machdep.cpu.brand_string`),
> - and the metrics' code paths do not route through MLX kernels
>   known to diverge per hardware family.
>
> As of MLX 0.31.1, the empirically-observed divergent kernel is
> `mx.random.normal` when the input key is the raw output of
> `mx.random.key(seed)` on Apple M1 Max specifically. `mx.random
> .uniform`, `mx.random.split`, `mx.random.key`, and `mx.random
> .normal` with a `mx.random.split`-derived key match across
> tested machines (M5, M3 Ultra, M1 Max).

## 4. Operational implication — per-family golden files

The `tests/reproducibility/golden_hashes.json` single-file
baseline was replaced (commit `deeb1e8`) with per-chip-family
files :

- `golden_hashes_apple_m5.json` (committed from grosmac, M5)
- `golden_hashes_apple_m3_ultra.json` (committed from Studio,
  M3 Ultra)
- `golden_hashes_apple_m1_max.json` (committed from macM1,
  M1 Max)

The helper `_chip_family()` in
`tests/reproducibility/_r1_helpers.py` reads
`sysctl machdep.cpu.brand_string` to pick the file for the
current machine. Cross-machine comparison is an **explicit
milestone exercise**, not a CI gate.

## 5. What this means for the formal axis

- **No FC bump.** R1 is an EC-axis property in the published
  framework spec ; the within-machine contract is intact. The
  cross-machine caveat is a refinement of the existing wording,
  not a new axiom or invariant.
- **Severity** : R1-within-machine is **BLOCKING** (a within-
  machine drift would fail a pre-registered run and block any
  EC promotion). R1-cross-machine is **WARN** (the cross-machine
  property is documented as conditional ; a divergence triggers
  a milestone-class probe and an upstream MLX issue, not a CI
  failure).
- **Severity escalation path** : if a future MLX release
  republishes a `mx.random.normal` kernel that *does* match
  across all three families, the cross-machine clause can be
  promoted to BLOCKING with no spec change (the wording
  "conditional on … MLX version" already permits it).

## 6. Evidence pointers

- Triple-machine R1 hash table : milestone `.json` §
  `r1_hash_comparison`.
- 8-probe MLX primitive table : milestone `.json` §
  `mlx_primitive_probe`.
- Upstream issue : ml-explore/mlx#3568 (filed 2026-05-20).
- Tracking issue : hypneum-lab/dream-of-kiki#25
  (3/4 follow-ups closed in the same session ; this proof
  document closes follow-up #2).

## 7. Future falsification gates

This qualification is itself falsifiable :

1. **A 4th Apple Silicon family** (M2, M4, A19, …) that produces
   *yet another* `test_r1_replay` cluster invalidates the
   "non-monotone" finding by adding a row to the matrix. The
   per-family file scheme accommodates this without further
   spec changes — adding `golden_hashes_apple_m2.json` is
   sufficient.
2. **An MLX point release** that closes ml-explore/mlx#3568
   should restore three-way M5 = M3 Ultra = M1 Max parity on
   `mx.random.normal` with a raw key. Re-running the milestone
   probe at the new MLX version confirms or refutes ; the proof
   document gets a new dated section, not a rewrite.
3. **Non-Apple Silicon backends** (CUDA, CPU-only) are *outside*
   the current R1 scope per spec §8.3 (substrate is "kiki-oniric
   on Apple Silicon"). The contract above applies only to MLX
   on Apple Silicon ; other backends inherit the within-machine
   clause but the cross-machine refinement is undefined for
   them.
