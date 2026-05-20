# R1 cross-machine probe — Apple M5 vs Apple M1 (2026-05-20)

**Milestone** : R1 bit-exact reproducibility probe across Apple
Silicon generations (M5 ↔ M1).
**Trigger commit** : `0a8ec29` (`docs(paper1): FR mirror of PR #18
+ integrity fixes (#23)`).
**Status** : **PARTIAL CROSS-MACHINE FAILURE** — RNG-dependent ops
diverge between M5 and M1; non-RNG ops match bit-exactly.
**Sibling JSON** : `r1-cross-machine-m5-vs-m1-2026-05-20.json`.

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

| Field | grosmac | macM1 |
|-------|---------|-------|
| Chip | **Apple M5** | **Apple M1** |
| RAM | 16 GB | 32 GB |
| macOS | 26.3.1 | 25.4.0 |
| Python | 3.14.4 | 3.14.4 |
| MLX | 0.31.1 | 0.31.1 |
| Commit | `0a8ec29` | `0a8ec29` |
| `uv sync --all-extras` | same lock | same lock |

Both machines run identical code (clean git clone on macM1; clean
`git checkout 0a8ec29` on grosmac with the only working-tree
change being a stash of the R1 metadata drift).

## Method

1. `git checkout 0a8ec29` on both machines.
2. `uv sync --all-extras` on both.
3. `uv run pytest tests/reproducibility/ --no-cov -q` on both.
4. The R1 tests regenerate `golden_hashes.json` on each run
   (current contract — the JSON entries are `pending_review`).
5. Capture the regenerated JSON from each machine, diff entry by
   entry.

## Result

**5 R1 entries** in `golden_hashes.json` at this commit. All 5
pytest assertions pass **within each machine** (within-machine R1
is intact, 9/9 tests green on both). But **3 of 5 hashes diverge
across the two machines** :

| R1 entry | grosmac (M5) | macM1 (M1) | Verdict |
|---|---|---|---|
| `test_r1_downscale` | `81297292f562…` | `81297292f562…` | ✅ **MATCH** |
| `test_r1_restructure` | `1adfe6b3924f…` | `1adfe6b3924f…` | ✅ **MATCH** |
| `test_r1_replay` | `37e1a4b47dfb…` | `cd53efd35fe8…` | ❌ DIVERGE |
| `test_r1_recombine` | `2f947c43b3ab…` | `b5ae6f5e2284…` | ❌ DIVERGE |
| `test_r1_full_pipeline` | `ba105f143202…` | `3c9e7dbe456f…` | ❌ DIVERGE |

**2/5 match, 3/5 diverge.**

## Interpretation

The split is not random — it correlates with **whether the op
samples from MLX's PRNG** :

- `downscale_real_handler` : multiplicative scaling, no RNG → **bit-exact across machines**.
- `restructure_real_handler` : pointer swap on `model.layers`,
  no RNG → **bit-exact across machines**.
- `replay_lora_handler` : SGD step → does *not* itself sample,
  but the seeded LoRA model init (B1a) uses
  `mx.random.uniform` + `mx.random.normal` → **DIVERGES**.
- `recombine_real_handler` : VAE reparameterization
  (`mx.random.normal(shape=mu.shape, key=sample_key)`) → **DIVERGES**.
- `full_pipeline` : composes all four ops → DIVERGES because two
  of its constituents diverge.

**Working hypothesis** : `mx.random.normal` (and possibly
`mx.random.uniform`) with the same PRNG key produce different
output tensors on Apple **M1** versus Apple **M5**. The
underlying Metal kernel path likely changed between hardware
generations (e.g. SIMD-group reduction order, threadgroup tile
size, or precision details of the underlying Philox / threefry
implementation).

This **contradicts the prior STATUS.md claim** of
machine-agnostic bit-exact R1 on Apple Silicon. The earlier claim
was made on the M3 Ultra ↔ M1 Max pair, where the result held. It
does not generalise to the M5 ↔ M1 pair.

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
