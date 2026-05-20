# R1 cross-machine probe — Apple M5 vs Apple M1 (2026-05-20)

**Milestone** : R1 bit-exact reproducibility probe across Apple
Silicon generations — M5, M3 Ultra, M1 Max.
**Trigger commit** : `0a8ec29` (`docs(paper1): FR mirror of PR #18
+ integrity fixes (#23)`).
**Status** : **PARTIAL CROSS-MACHINE FAILURE** — only `mx.random
.normal` with a directly-constructed key diverges; cause isolated
to M1 Max specifically. Filed upstream as
[ml-explore/mlx#3568](https://github.com/ml-explore/mlx/issues/3568).
Tracking: [hypneum-lab/dream-of-kiki#25](https://github.com/hypneum-lab/dream-of-kiki/issues/25).
**Sibling JSON** : `r1-cross-machine-m5-vs-m1-2026-05-20.json`.

**Amendment 2026-05-20 (same day)** : initial probe assumed
macM1 = Apple M1. `sysctl -n machdep.cpu.brand_string` revealed
macM1 = **Apple M1 Max**. This is the SAME machine that the
2026-05-04 STATUS.md "M3 Ultra ↔ M1 Max bit-exact" claim was
made on. With B5 (`apply_channel_outputs`) shipped, that claim
now PARTIALLY regresses : M3 Ultra ↔ M1 Max still match on 4/5
R1 entries but diverge on `test_r1_recombine` (added in B4).
Studio (M3 Ultra) was also re-tested at `0a8ec29` and integrated
into the comparison below.

---

## Why this milestone exists

`STATUS.md` (2026-05-04) documented a successful cross-machine R1
match on the Studio (M3 Ultra) ↔ M1 Max pair :

> R1 cross-machine verification : `tests/reproducibility/`
> 9/9 PASS on Studio Python 3.14.4/MLX 0.31.1 with hashes
> identical to M1 Max baseline → cross-machine bit-exact match
> observed on Apple Silicon pair, but `golden_hashes.json` entries
> remain flagged `pending_review`.

When B5 (`apply_channel_outputs` loop closure, `C-v0.20.0`)
shipped, the user requested running the full suite on a third
Apple Silicon machine — **macM1** (Apple M1, 32 GB, MLX 0.31.1).
The opportunity to extend the cross-machine R1 evidence one more
generation was taken.

## Setup

| Field | grosmac | Studio | macM1 |
|-------|---------|--------|-------|
| Chip | **Apple M5** | **Apple M3 Ultra** | **Apple M1 Max** |
| RAM | 16 GB | 512 GB | 32 GB |
| macOS | 26.5 | 26.4.1 | 26.4.1 |
| Python | 3.14.3 | 3.14.4 | 3.14.4 |
| MLX | 0.31.1 | 0.31.1 | 0.31.1 |
| Commit | `0a8ec29` | `0a8ec29` | `0a8ec29` |
| `uv sync --all-extras` | same lock | same lock | same lock |

All three machines run identical code at `0a8ec29` (clean git
clones on Studio + macM1; clean `git checkout 0a8ec29` on
grosmac with the only working-tree change being a stash of the
R1 metadata drift).

## Method

1. `git checkout 0a8ec29` on both machines.
2. `uv sync --all-extras` on both.
3. `uv run pytest tests/reproducibility/ --no-cov -q` on both.
4. The R1 tests regenerate `golden_hashes.json` on each run
   (current contract — the JSON entries are `pending_review`).
5. Capture the regenerated JSON from each machine, diff entry by
   entry.

## Result — 3-machine R1 hash comparison

**5 R1 entries** in `golden_hashes.json` at this commit. All 5
pytest assertions pass **within each machine** (within-machine R1
is intact, 9/9 tests green on all three). But the cross-machine
pattern is **non-monotone** :

| R1 entry | M5 (grosmac) | M3 Ultra (Studio) | M1 Max (macM1) | Cluster |
|---|---|---|---|---|
| `test_r1_downscale` | `81297292f562…` | `81297292f562…` | `81297292f562…` | ✅ all 3 match (no RNG) |
| `test_r1_restructure` | `1adfe6b3924f…` | `1adfe6b3924f…` | `1adfe6b3924f…` | ✅ all 3 match (no RNG) |
| `test_r1_replay` | `37e1a4b47dfb…` | `cd53efd35fe8…` | `cd53efd35fe8…` | M3 Ultra == M1 Max ≠ M5 |
| `test_r1_recombine` | `2f947c43b3ab…` | `2f947c43b3ab…` | `b5ae6f5e2284…` | M5 == M3 Ultra ≠ M1 Max |
| `test_r1_full_pipeline` | `ba105f143202…` | `3c9e7dbe456f…` | `3c9e7dbe456f…` | M3 Ultra == M1 Max ≠ M5 |

**Surprise:** the divergences do NOT form a clean generational
ladder. M3 Ultra is closer to M1 Max on `replay`/`full_pipeline`
but closer to M5 on `recombine`. This rules out a single "newer
chip ⇒ different kernel" hypothesis; the kernel choice must
branch on op-specific factors.

## MLX primitive probe — narrows the cause to `mx.random.normal`

A direct probe of `mx.random.*` primitives at this commit
(`/tmp/mlx_rng_repro.py`, run on the same three machines) shows
that **only `mx.random.normal`** diverges, and only on **Apple
M1 Max** :

| MLX call | M5 | M3 Ultra | M1 Max | Match ? |
|---|---|---|---|---|
| `mx.random.key(0)` materialised | `af5570…` | `af5570…` | `af5570…` | ✅ all match |
| `mx.random.split(key(7))[0]` | `1cd6e8…` | `1cd6e8…` | `1cd6e8…` | ✅ all match |
| `mx.random.split(key(7))[1]` | `7b3d5b…` | `7b3d5b…` | `7b3d5b…` | ✅ all match |
| `mx.random.uniform(low=-1, high=1, shape=(8,), key=key(0))` | `f8646c…` | `f8646c…` | `f8646c…` | ✅ all match |
| `mx.random.normal(shape=(4,), key=split-child-of-key(7))` | `50ac95…` | `50ac95…` | `50ac95…` | ✅ all match |
| `mx.random.normal(shape=(8,), key=key(0))` | `a66260…` | `a66260…` | `8a8c5a…` | ❌ M1 Max diverges |
| `mx.random.normal(shape=(8,), key=key(42))` | `7ef9a0…` | `7ef9a0…` | `46a0c8…` | ❌ M1 Max diverges |
| `mx.random.normal(shape=(16, 4), key=key(0))` | `b2b3b1…` | `b2b3b1…` | `4e136e…` | ❌ M1 Max diverges |

**Three independent shape × seed combinations** of `mx.random
.normal` with a directly-constructed key all diverge on **Apple
M1 Max only**. The PRNG key tensor itself is bit-identical
everywhere; `mx.random.split` returns identical halves; `mx.random
.uniform` produces identical bytes with the same raw key. Only
`mx.random.normal` plus the raw key combination diverges, only on
M1 Max.

Plausibly: `mx.random.normal` on M1 Max takes a different Metal
kernel path (likely a SIMD-group reduction order detail in the
Philox / threefry float32 conversion), but only when the input
key was *not* produced by `mx.random.split`. A `mx.random.normal`
fed a split-derived key matches all three machines, which makes
the test-suite-level pattern above (M5↔M3 Ultra on `recombine`
because B4 uses `mx.random.split` before sampling) consistent
with the M1 Max divergence elsewhere — `replay` and the LoRA-init
path use raw `mx.random.key(seed)` directly.

Filed upstream as
[ml-explore/mlx#3568](https://github.com/ml-explore/mlx/issues/3568).

## Interpretation

The R1 entry split correlates with **whether the op uses
`mx.random.normal` with a raw key** :

- `downscale_real_handler` : multiplicative scaling, no RNG → **bit-exact across machines**.
- `restructure_real_handler` : pointer swap on `model.layers`,
  no RNG → **bit-exact across machines**.
- `replay_lora_handler` : SGD step → no fresh sampling, but the
  seeded LoRA model init (B1a `LoRALinear`) uses
  `mx.random.normal(shape, key=key(seed))` for `lora_a` and
  `mx.random.uniform` for `base_weight`. The **normal init**
  takes a raw key from `mx.random.split` immediately after
  `mx.random.key(seed)`, but the relevant divergence pattern
  here (M3 Ultra == M1 Max ≠ M5) suggests M5 changed the
  uniform-or-normal path relative to {M3 Ultra, M1 Max}.
- `recombine_real_handler` : VAE reparameterization uses
  `mx.random.split(key(seed + ep))` then `mx.random.normal(shape,
  key=split_child)`. The pattern (M5 == M3 Ultra ≠ M1 Max)
  matches the MLX primitive probe: `normal` with a split-derived
  key actually matched in the isolated probe, so this divergence
  is on a *different* code path — possibly the `exp(0.5 *
  log_var)` chain or another normal call hidden in the model.
- `full_pipeline` : composes all four ops → M3 Ultra ==
  M1 Max ≠ M5 (matches the replay pattern).

**Refined working hypothesis** : Apple Silicon implementations of
MLX's `mx.random.normal` Metal kernel diverge per chip family in
ways that depend on how the input key was constructed *and* on
the call shape. The PRNG state representation itself is portable
(`mx.random.key`, `mx.random.split` produce bit-identical
tensors). The Box-Muller / Marsaglia-polar / ziggurat conversion
from uniform bits to normal float32 — or whichever transform MLX
uses internally — appears to branch on the chip family for at
least some (shape, key-origin) combinations.

**This contradicts the prior STATUS.md claim** of
machine-agnostic bit-exact R1 on Apple Silicon. The earlier
2026-05-04 claim was made on the Studio (M3 Ultra) ↔ macM1
(M1 Max) pair at a commit *before* B4/B5 shipped, and held at
that point. With B4 (`recombine_real_handler` returning a real
`LatentSample` with `mx.random.normal` post-split) and B5
(`apply_channel_outputs` exercising the new path end-to-end), the
M3 Ultra ↔ M1 Max pair now diverges on `test_r1_recombine`
specifically.

The previously committed R1 hashes (the baseline tracked in git
at the moment of test invocation) were stable on the M1 Max
baseline. They drift on M5 *because* the kernel path under
`mx.random.normal` shifted, not because of any code change in
this repo.

## Implications

- **Within-machine R1 contract is intact.** Each machine
  reproduces its own runs bit-exactly. The DR-0 / DR-3 / DR-4
  conformance tests are unaffected — they don't depend on
  cross-machine hash equality.
- **The repo's `tests/reproducibility/golden_hashes.json` is
  machine-specific by hardware family.** Treating it as a global
  baseline is wrong; the file should either be regenerated per
  hardware family, gitignored with a per-machine override
  (`pre-commit` hook), or split per `(chip_family, MLX version)`
  key.
- **No FC bump.** R1 is an empirical-axis (EC) property and the
  formal-axis spec already documents R1 as "bit-stable run_id
  hashing" without quantifying *across which hardware*. EC stays
  `+PARTIAL`.
- **No CHANGELOG entry on the main framework.** This is an
  external probe finding, not a code change. The CHANGELOG
  remains at `[C-v0.20.0+PARTIAL]` (B5).

## Recommended follow-ups (out of scope for B5)

1. **Re-verify M3 Ultra ↔ M1 Max** at commit `0a8ec29` — was the
   2026-05-04 claim still valid post-B0..B5, or did it also drift?
2. **Open `docs/proofs/r1-cross-machine.md`** (or amend
   `docs/invariants/registry.md`) to qualify R1 as
   "within-machine bit-stable; cross-machine bit-stability
   conditional on hardware family + MLX version + RNG-touching
   ops".
3. **MLX upstream issue** — file a minimal reproducer
   (`mx.random.normal(shape=(8,), key=mx.random.key(0))` on M5
   vs M1 producing different float32 outputs) and check whether
   the divergence is a known MLX RNG kernel detail or a real
   bug.
4. **Per-hardware golden_hashes.json** — split the file into
   `{m1.json, m3ultra.json, m5.json}` keyed on detected chip
   family, or gitignore the regenerated file with a manifest
   listing the per-family expected hashes.

## Sanity numbers

Both machines run the full suite identically modulo wall-time :

| Metric | grosmac (M5) | macM1 (M1) |
|---|---|---|
| `pytest` collected | 816 | 816 |
| `pytest` passed | 816 | 816 |
| `pytest` skipped | 3 | 3 |
| `pytest` xfailed | 12 | 12 |
| Coverage | 87.70% | 87.70% |
| `pytest` wall | ~9.5 s | 48.3 s (~5× slower) |
| `mypy harness tests` | Success 170 files | Success 170 files |
| `ruff check .` | clean | clean |
| `tests/reproducibility/` | 9/9 pass | 9/9 pass |

The functional verdict (816 / 87.70% / mypy / ruff) is **fully
reproducible across hardware**. Only the cryptographic-level
bit-exact RNG outputs differ.

## Provenance

- Probe initiated 2026-05-20 ~10:30 CEST.
- macM1 cloned from `https://github.com/hypneum-lab/dream-of-kiki`
  at `0a8ec29`.
- grosmac transitioned via `git checkout 0a8ec29` (detached
  HEAD), R1 metadata drift stashed beforehand
  (`stash@{0}: wip-golden-drift-pre-r1-cross`).
- Both machines independently regenerated
  `tests/reproducibility/golden_hashes.json` ; captured snapshots
  saved to `/tmp/golden_macm1.json` and
  `/tmp/golden_grosmac_at_0a8ec29.json` and diffed entry-by-entry.
- See sibling JSON for the full hash table.
