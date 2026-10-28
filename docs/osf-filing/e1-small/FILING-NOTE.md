# OSF filing note — E1-small (4B dense) replay controls

**Status:** prepared 2026-10-28 for the PI, **not submitted**, and **not
yet fileable**: two conditions of the draft's filing gate are open (see
"Filing-gate status"). The PI
files on OSF himself. The KXKM-AI run waits for the PI's confirmation
that the pre-registration is filed.

## What is filed

| item | value |
|---|---|
| Title | E1-small: replay controls at 4B (dense) pre-registration |
| Authors | Clément Saillant (L'Electron Rare, Grandris, France), as in `CITATION.cff` |
| Parent OSF project | 10.17605/OSF.IO/Q6JYN (`https://osf.io/q6jyn`) |
| Source draft | `docs/osf-prereg-e1-small-2026-10-25.md` |
| Frozen at commit | `0fb1e68db39ca09082d7f7d7105ac864a92a135b` (2026-10-27T16:01:42Z), the last commit that changed the draft |
| Draft SHA-256 | `7c613a66161b6263611fa969f3f95d5448f3feb8a11b87598cd50467bf082c5a` |
| Frozen copy | `docs/osf-filing/e1-small/osf-prereg-e1-small-2026-10-25.frozen.md` (byte-identical to the draft at that commit, same SHA-256) |
| PDF rendering | `docs/osf-filing/e1-small/osf-prereg-e1-small-2026-10-25.frozen.pdf` (24 pages; SHA-256 in `SHA256SUMS`) |
| Rendering | `render.sh` + `render.lua` in the same folder: pandoc 3.12.1, xelatex (TeX Live 2026), layout only (column widths, line breaks in code); the text is the frozen Markdown. The PDF is not byte-reproducible (embedded dates); the Markdown and its SHA-256 are the reference. |
| Pull request | `https://github.com/hypneum-lab/dream-of-kiki/pull/68` (draft, branch `docs/osf-prereg-e1-small`) |
| Commit | `https://github.com/hypneum-lab/dream-of-kiki/commit/0fb1e68db39ca09082d7f7d7105ac864a92a135b` |
| Sibling study | 35B draft `docs/osf-prereg-e1-replay-controls-2026-10-23.md` (PR #41); E1-small reproduces its revision 9 (`9172768`, SHA-256 `ae00aea5…bf71b5`) as §1-§8. Results are never pooled across sizes, studies or instruments. |

At filing, record on OSF: the lock commit (the commit of this package,
or a later one that leaves the draft byte-identical) and the lock
timestamp; the draft's own "Lock commit / Lock timestamp" fields are
left to be filled at filing, as for every pre-registration in this
repository.

## PI decisions S1-S11 (draft §10)

- **S1 Model:** `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (tokenizer byte-identical to the 35B pin).
- **S2 Headroom / freeze:** E1-small freezes its own copy of the pending corpus with the 4B headroom, TUNING only.
- **S3 Order:** E1-small is filed and run before the 35B study; the 35B draft is not edited.
- **S4 Hardware:** KXKM-AI, one RTX 4090 24 GB, shared; used only when released.
- **S5 Memory fallback:** activation checkpointing only if memory does not fit; not triggered (peak 20.08 GB).
- **S6 Kernels:** fla-core / causal-conv1d evaluated; parity check failed; by the pre-fixed fp32 rule the transformers torch fallback is registered and the kernels are dropped.
- **S7 Text:** self-contained (§1-§8 copy revision 9 of the 35B draft; edits listed in §11).
- **S8 Tokenizer:** raw pinned `tokenizer.json` through `tokenizers`, no special tokens.
- **S9 macM1 pipeline smoke:** in addition to the KXKM measurement smoke, on the `torch-mps` instrument (first attempt stopped on memory pressure; not rerun, macM1 unreachable).
- **S10 Determinism:** registered for every CUDA cell, enforced.
- **S11 Allocator:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, registered, enforced.

## Run plan on KXKM-AI (after the PI confirms the filing)

Registered instrument: PyTorch 2.11.0 + transformers 5.7.0, one RTX 4090,
torch fallback for Gated DeltaNet, raw tokenizer, RAM snapshot,
`torch.optim.SGD`, deterministic mode, expandable-segments allocator;
runner `scripts/run_e1_small_cuda.sh` (`HL_STUDY=e1-small`), one tmux
session per step, GPU only when no other job is on it.

| step | mode | cells | measured projection (§9.7d) |
|---|---|---|---|
| 1 | `headroom` (TUNING only) then `e1_corpus_build.py --freeze --study e1-small` | — | about 3-4 min (≈ 1 900 scoring forwards) |
| 2 | `tuning` (seeds 1000-1009) | 120 | 11.6 h |
| 3 | `calibrate` | 60 | 5.6 h |
| 4 | `amend` (frozen tuning amendment, no GPU) | — | — |
| 5 | `confirmatory` (seeds 0-140) | 1,410 | 63.2 h at `n_eval` 130; 95.0 h at 255 |
| — | smoke (done, deterministic, §9.7d) | 30 | 2.1 h (measured) |
| | **total, 1,620 cells** | | **82.4 h** at `n_eval` 130; **114.3 h** at `n_eval` 255 |

`n_eval` is fixed by the §5.2 rule in the frozen tuning amendment (130
for p_disc ≤ 0.10, 255 at 0.20, more for larger p_disc, capped at the
confirmatory partition size − 40).

## Filing-gate status (draft §9.8) — for the PI before filing

| condition | status |
|---|---|
| Measurement smoke on the registered instrument | **done** (§9.7d: 30/30 cells, deterministic, 82.4 h / 114.3 h) |
| 4B headroom run and E1-small FROZEN corpus / manifest hashes written into the draft | **not done** — needs about 4 min of KXKM-AI GPU (TUNING items only) |
| Decision code exercised on synthetic paired data covering every §2 state and §7 row | covered by the unit tests of `scripts/analyze_e1_cuda.py` (`tests/unit/scripts/test_analyze_e1_cuda.py`); no separate recorded run |
| Harness code merged at a pinned commit with passing tests | **not done** — PR #68 is a draft into `docs/osf-prereg-2026-10`; CI runs only on PRs to `main`; the full unit suite last passed on macM1 before the MPS changes, the E1 test files pass on KXKM CPU at the current code; mypy was not run on the latest commits |

**Conflict to resolve.** The draft's own gate (§5.1 property 5, §9.8;
inherited from the 35B draft) requires the corpus to be frozen with the
headroom scores **before filing**, while the current instruction is that
the KXKM run waits for the filing; §9.8 also requires the harness to
be merged at a pinned commit with passing tests. **This snapshot
therefore does not satisfy the draft's filing gate and should not be
filed as is.** Path that keeps the protocol as written: (1) authorise
the short TUNING-only headroom run and the freeze; (2) write the
FROZEN corpus and manifest hashes into the draft; (3) merge PR #68 (or
pin the harness commit with a passing CI run); (4) refresh this package
(new frozen draft, PDF, SHA-256) and file. Filing earlier would need an
explicit protocol amendment by the PI that changes §9.8 (and the §5.1
property 5 timing) before filing; a post-hoc deviation note would not
satisfy the gate.

## Package contents (`docs/osf-filing/e1-small/`)

- `FILING-NOTE.md` — this note
- `osf-prereg-e1-small-2026-10-25.frozen.md` — frozen draft
- `osf-prereg-e1-small-2026-10-25.frozen.pdf` — PDF rendering
- `render.sh`, `render.lua` — how the PDF was made
- `SHA256SUMS` — hashes of the files above
