# E1-small deviation: run started before the OSF filing (2026-10-29)

**Written:** 2026-10-29 (about 05:15 UTC), **before** the E1-small run
is launched on KXKM-AI. **No E1-small tuning, calibration or
confirmatory result exists at the time of writing**; the only E1-small
GPU outputs so far are the pre-run records already on `main` in
`docs/milestones/e1-small/` (kernel parity, fp32 diagnostic, determinism
check, both smokes on the pilot fixtures, the TUNING-only headroom).

## Deviation

By PI decision of 2026-10-29 06:13 (CET), the registered E1-small run
(tuning 120 cells → calibration 60 → frozen tuning amendment →
confirmatory 1,410) starts on KXKM-AI **before** the pre-registration is
filed on OSF. The protocol says it is filed first (status header of the
draft; filing note, "Remaining steps").

## What locks the protocol meanwhile

- **Lock commit:** `963bb245db6bf1d5437527675eb9a64e66c9f7aa`, the merge
  of PR #72 into `main` of `github.com/hypneum-lab/dream-of-kiki`,
  created on GitHub at **2026-10-28T16:57:07Z** (GitHub API: PR #72
  `mergedAt`, the commit's committer date, and the repository's
  `PullRequestEvent` "merged" `created_at` all give this time; the commit
  is GitHub-verified). CI on that commit passed (run 40730983815).
- **Frozen draft:** `docs/osf-prereg-e1-small-2026-10-25.md` at that
  commit, SHA-256
  `6d47d8b71829d884d3a18dda84cd21aa080f1090101d62a4bafe6bc493208463`
  (identical to the frozen copy in `docs/osf-filing/e1-small/`).
- **Frozen corpus (draft §9.3):** file `f8379b81…c0a2`, confirmatory
  partition `546f44c6…cb3f`, tuning partition `7aea63a8…b0f7`; FROZEN
  manifest `docs/milestones/e1-small/corpus-manifest-frozen-2026-10-28.json`
  (SHA-256 `093fdf61…7202ab`).
- **Code:** the E1 harness and PyTorch backend at that commit; later
  commits on `main` up to this note (`a33611b`) change no E1 code, the
  E1-small draft or the frozen corpus manifest (only the filing package
  re-pin, G4 / DR-3 files and the 35B GO-rule note).

Neither the frozen draft nor its filing package is edited by this
deviation.

## Consequences

- Every E1-small cell is run from a clean checkout of `main` containing
  `963bb24`, on the registered instrument (deterministic mode,
  expandable-segments allocator, torch fallback kernels, raw tokenizer,
  FROZEN 4B corpus); the harness refuses any other instrument.
- The frozen tuning amendment is written and committed **before** the
  first confirmatory cell, as the protocol requires.
- The OSF filing follows and cites this deviation: it records that the
  run started before filing, at the time given in the run log, under
  the lock commit above.
