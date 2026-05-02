# Canonical Glossary (authoritative)

Any term appearing in specs, code, papers MUST match this glossary.
Rename only via DualVer MINOR bump of the framework spec.

## Core entities

- **dreamOfkiki** — program name, logical identifier (camelCase for
  branding, prose, figures)
- **dream-of-kiki** — filesystem-safe technical name (kebab-case for
  repo paths when needed)
- **kiki-oniric** — dedicated fork of kiki-flow-core for Track A
  experiments (placeholder name; Python package is `kiki_oniric`)
- **kiki-flow-core** — source repo (not touched directly, jalonné
  rebase S1/S8/S18)

## Processes

- **awake process** — kiki real-time inference/training running on
  Studio (MLX)
- **dream process** — asynchronous offline consolidation running per
  profile topology
- **DE (dream-episode)** — atomic dream unit: 5-tuple
  `(trigger, input_slice, operation_set, output_delta, budget)`

## Data flows

- **α (alpha)** — raw traces firehose (P_max only)
- **β (beta)** — curated episodic buffer (all profiles)
- **γ (gamma)** — weights-only snapshot (fallback)
- **δ (delta)** — hierarchical latent snapshots (P_equ, P_max)
- **Canal 1** — weight_delta (dream → awake)
- **Canal 2** — latent_samples (dream → awake)
- **Canal 3** — hierarchy_chg (dream → awake)
- **Canal 4** — attention_prior (dream → awake)

## State copies (swap worktree)

- **W_awake** — active weights, read+write by awake
- **W_dream** — frozen snapshot, read-only by dream
- **W_scratch** — working copy modified by dream, becomes W_awake at
  swap

## Profiles

- **P_min** — β → 1 (minimal publishable)
- **P_equ** — β+δ → 1+3+4 (balanced, canonical)
- **P_max** — α+β+δ → 1+2+3+4 (maximalist)

## Versioning (DualVer)

- **DualVer** — format
  `C-v<FC-MAJOR>.<FC-MINOR>.<FC-PATCH>+<EC-STATE>`
- **FC** — formal consistency (SemVer-like)
- **EC** — empirical consistency ∈ {STABLE, DIRTY, INVALIDATED}

## Gates

- **G1** S2 — T-Col fallback locked
- **G2** S8 — P_min viable
- **G3** S8 — DR-2 preuve peer-reviewed (+ G3-draft S6 circulé)
- **G4** S12 — P_equ fonctionnel
- **G5** S18 — PUBLICATION-READY
- **G6** S28 — amorce cycle 2 décision

## Maturity modes

- **RED** — ≥1 BLOCKING violé
- **GREEN** — BLOCKING respectés, WARN sous seuil
- **PUBLICATION-READY** — GREEN + critères supplémentaires §9
  framework spec

## Metrics (E3 cognitive + E4 engineering)

- **M1.a** forgetting rate
- **M1.b** average accuracy cross-tasks
- **M2.b** RSA fMRI alignment
- **M3.a** FLOPs ratio dream/awake
- **M3.b** offline gain (pivot metric, shared E3/E4)
- **M3.c** energy per episode
- **M4.a** recombination quality (teacher scorer gelé)
- **M4.b** structure discovery (permutation test)

## Baselines

- **Wake-Sleep CL** — Wake-Sleep Consolidated Learning, Alfarano et al.
  2024 [IEEE TNNLS, arXiv 2401.08623] — closest published NREM/REM
  dual-phase analog ; Paper 2 §5.8 baseline (variant-c published
  reference, FC-MINOR `baselines:` block landed C-v0.12.0).
