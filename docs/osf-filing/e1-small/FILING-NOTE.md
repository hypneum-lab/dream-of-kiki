# OSF filing note — E1-small (4B dense) replay controls

**Status:** refreshed 2026-10-28 for the PI, **not submitted**. One
condition of the draft's filing gate remains: the harness must be
**merged** (PR #72, CI green at the pinned commit below); after the merge
this package is re-pinned to the merge commit. The PI files on OSF
himself. The KXKM-AI run waits for the PI's confirmation that the
pre-registration is filed.

## What is filed

| item | value |
|---|---|
| Title | E1-small: replay controls at 4B (dense) pre-registration |
| Authors | Clément Saillant (L'Electron Rare, Grandris, France), as in `CITATION.cff` |
| Parent OSF project | 10.17605/OSF.IO/Q6JYN (`https://osf.io/q6jyn`) |
| Source draft | `docs/osf-prereg-e1-small-2026-10-25.md` |
| Pinned commit | `82dc277df209708d7ed08f6808e9926667d2f891`, head of PR #72 to `main` with CI green (workflow run 40714009612); the commit that adds this package on top changes only `docs/osf-filing/`; the draft is byte-identical there (last changed in `0f3104389590797ac0936d70468718c3d2c7bba7` on that branch, same content as `cc0bc66` on PR #68) |
| Draft SHA-256 | `6d47d8b71829d884d3a18dda84cd21aa080f1090101d62a4bafe6bc493208463` |
| Frozen copy | `docs/osf-filing/e1-small/osf-prereg-e1-small-2026-10-25.frozen.md` (byte-identical, same SHA-256) |
| PDF rendering | `docs/osf-filing/e1-small/osf-prereg-e1-small-2026-10-25.frozen.pdf` (SHA-256 in `SHA256SUMS`) |
| Rendering | `render.sh` + `render.lua` in the same folder: pandoc 3.12.1, xelatex (TeX Live 2026), layout only (column widths, line breaks in code); the text is the frozen Markdown. The PDF is not byte-reproducible (embedded dates); the Markdown and its SHA-256 are the reference. |
| Merge PR | `https://github.com/hypneum-lab/dream-of-kiki/pull/72` (E1 core + E1-small to `main`) |
| Working PR | `https://github.com/hypneum-lab/dream-of-kiki/pull/68` (history of the E1-small work) |
| Commit | `https://github.com/hypneum-lab/dream-of-kiki/commit/82dc277df209708d7ed08f6808e9926667d2f891` |
| Corpus | E1-small FROZEN corpus: file `f8379b81…c0a2`, confirmatory `546f44c6…cb3f`, tuning `7aea63a8…b0f7`; manifest `docs/milestones/e1-small/corpus-manifest-frozen-2026-10-28.json`, SHA-256 `093fdf61…7202ab` (draft §9.3) |
| Sibling study | 35B draft `docs/osf-prereg-e1-replay-controls-2026-10-23.md` (PR #41, not yet on `main`); E1-small reproduces its revision 9 (`9172768`, SHA-256 `ae00aea5…bf71b5`) as §1-§8. Results are never pooled across sizes, studies or instruments. |

At filing, record on OSF the lock commit (the merge commit, to which
this package is re-pinned) and the lock timestamp; the draft's own
"Lock commit / Lock timestamp" fields are left to be filled at filing,
as for every pre-registration in this repository.

## PI decisions S1-S11 (draft §10)

- **S1 Model:** `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (tokenizer byte-identical to the 35B pin).
- **S2 Headroom / freeze:** E1-small freezes its own copy of the pending corpus with the 4B headroom, TUNING only — done 2026-10-28: ACCEPT, no source dropped.
- **S3 Order:** E1-small is filed and run before the 35B study; the 35B draft is not edited.
- **S4 Hardware:** KXKM-AI, one RTX 4090 24 GB, shared; used only when released.
- **S5 Memory fallback:** activation checkpointing only if memory does not fit; not triggered (peak 20.08 GB).
- **S6 Kernels:** fla-core / causal-conv1d evaluated; parity check failed; by the pre-fixed fp32 rule the transformers torch fallback is registered and the kernels are dropped.
- **S7 Text:** self-contained (§1-§8 copy revision 9 of the 35B draft; edits listed in §11).
- **S8 Tokenizer:** raw pinned `tokenizer.json` through `tokenizers`, no special tokens.
- **S9 macM1 pipeline smoke:** in addition to the KXKM measurement smoke, on the `torch-mps` instrument (first attempt stopped on memory pressure; not rerun).
- **S10 Determinism:** registered for every CUDA cell, enforced.
- **S11 Allocator:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, registered, enforced.

## Run plan on KXKM-AI (after the PI confirms the filing)

Registered instrument: PyTorch 2.11.0 + transformers 5.7.0, one RTX 4090,
torch fallback for Gated DeltaNet, raw tokenizer, RAM snapshot,
`torch.optim.SGD`, deterministic mode, expandable-segments allocator;
runner `scripts/run_e1_small_cuda.sh` (`HL_STUDY=e1-small`), the FROZEN
corpus and manifest above, one tmux session per step, GPU only when no
other job is on it.

| step | mode | cells | measured projection (§9.7d) |
|---|---|---|---|
| — | headroom (TUNING only) and freeze | — | done 2026-10-28 (§9.3) |
| — | smoke (deterministic, §9.7d) | 30 | done, 2.1 h measured |
| 1 | `tuning` (seeds 1000-1009) | 120 | 11.6 h |
| 2 | `calibrate` | 60 | 5.6 h |
| 3 | `amend` (frozen tuning amendment, no GPU) | — | — |
| 4 | `confirmatory` (seeds 0-140) | 1,410 | 63.2 h at `n_eval` 130; 95.0 h at 255 |
| | **total, 1,620 cells** (smoke included) | | **82.4 h** at `n_eval` 130; **114.3 h** at `n_eval` 255 |

`n_eval` is fixed by the §5.2 rule in the frozen tuning amendment (130
for p_disc ≤ 0.10, 255 at 0.20, more for larger p_disc, capped at the
confirmatory partition size − 40).

## Filing-gate status (draft §9.8)

| condition | status |
|---|---|
| Measurement smoke on the registered instrument | **done** (§9.7d: 30/30 cells, deterministic, 82.4 h / 114.3 h) |
| 4B headroom run and E1-small FROZEN corpus / manifest hashes written into the draft | **done** (§9.3; headroom 2026-10-28, freeze ACCEPT) |
| Decision code exercised on synthetic paired data covering every §2 state and §7 row | covered by the unit tests of `scripts/analyze_e1_cuda.py` (`tests/unit/scripts/test_analyze_e1_cuda.py`), run by CI on PR #72; no separate recorded run |
| Harness code merged at a pinned commit with passing tests | **pending the PI's merge** — PR #72, CI green at `82dc277`; torch tests also pass on macM1 (55) and KXKM CPU |

Remaining steps: (1) the PI merges PR #72; (2) this package is
re-pinned to the merge commit (draft and frozen copy unchanged unless
the merge changes them; PDF regenerated with the new commit in its
subtitle; SHA-256 refreshed); (3) the PI files on OSF and records the
lock commit and timestamp; (4) the KXKM-AI run starts after his
confirmation.

## Package contents (`docs/osf-filing/e1-small/`)

- `FILING-NOTE.md` — this note
- `osf-prereg-e1-small-2026-10-25.frozen.md` — frozen draft
- `osf-prereg-e1-small-2026-10-25.frozen.pdf` — PDF rendering
- `render.sh`, `render.lua` — how the PDF was made
- `SHA256SUMS` — hashes of the files above
