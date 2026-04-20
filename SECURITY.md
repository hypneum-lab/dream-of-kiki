# Security policy — dream-of-kiki

dreamOfkiki is primarily a **research code base**, not a production
system. Correctness and reproducibility are our top priorities;
security is handled on a best-effort basis.

## Scope

This policy applies to :

- The code in this repository (`c-geni-al/dream-of-kiki`)
- The downstream `harness`, `scripts`, and `kiki_oniric` modules
- The documentation in `docs/` (inaccurate or misleading information
  qualifies as a security/integrity concern)

It does *not* apply to dependencies; please report those upstream.

## Reporting a vulnerability

If you discover a security issue that could compromise :

- the reproducibility contract R1 (e.g., a run with the same
  `(c_version, profile, seed, commit_sha)` producing different
  `run_id`)
- the bit-exact behaviour of core framework primitives
- the integrity of the pre-registered H1-H4 hypotheses analysis
  pipeline
- the authenticity of any signed artifact (BibTeX, SHA-256 digests,
  registered benchmarks)

please report it **privately** via one of these channels :

1. Email : `clement@saillant.cc` — subject starting with `[SECURITY]`
2. GitHub Private Vulnerability Reporting :
   https://github.com/c-geni-al/dream-of-kiki/security/advisories/new

Please include :
- a description of the issue,
- steps to reproduce (ideally with a deterministic seed),
- the `run_id` and `commit_sha` of the affected run if applicable,
- your preferred disclosure timeline.

## Our commitment

- Initial acknowledgement within 72 h.
- Initial assessment within 1-2 weeks.
- Public disclosure and fix coordinated with the reporter.
- Credit to the reporter (unless anonymity is requested).

## Non-security issues

For research methodology, reproducibility bugs, or framework
correctness questions that are *not* security-sensitive, open a
public issue on GitHub — we prefer open discussion when it is safe
to do so.

## Supply-chain notes

- All external benchmarks ship with `.sha256` digests in the
  `harness/` registry.
- The teacher scorer (Qwen3.5-9B Q4_K_M) is pinned by SHA-256.
- Python dependencies are pinned in `uv.lock`.
- No network calls are made during experimental runs; cached
  artifacts are verified by digest before use.

## Scope limitations

This policy does **not** cover :

- Dependency vulnerabilities in `uv.lock` — please report to the
  upstream package maintainer and notify us if a fix requires a
  version bump here.
- Hardware-specific side channels on Apple Silicon MLX or Linux
  CPU fallback.
- Issues in private sibling repositories (`micro-kiki-quantum`) —
  those are triaged separately.
