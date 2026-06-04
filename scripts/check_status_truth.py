#!/usr/bin/env python3
"""Validate STATUS.md / CHANGELOG.md claims against measurable truth.

Run as part of CI to fail fast when documentation drifts from
the underlying code/test state. Each check exits PASS / FAIL
and the script returns non-zero if any FAIL.

N4 Task 5 (2026-05-10) — created in response to N1-N3 audits
that surfaced systematic drift between docs and reality.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
results: list[tuple[str, bool, str]] = []


def report(name: str, passed: bool, message: str) -> None:
    results.append((name, passed, message))
    icon = "PASS" if passed else "FAIL"
    print(f"[{icon}] {name}: {message}")


def check_test_count() -> None:
    """STATUS.md test count vs `pytest --collect-only` reality."""
    status_path = REPO_ROOT / "STATUS.md"
    if not status_path.exists():
        report("test_count", True, "no STATUS.md (skip)")
        return
    status = status_path.read_text()
    m = re.search(r"(\d{2,5})\s+tests?\s+collected", status)
    if not m:
        report("test_count", False, "no 'N tests collected' claim found in STATUS.md")
        return
    claimed = int(m.group(1))

    out = subprocess.run(
        ["uv", "run", "pytest", "--collect-only", "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    actual_m = re.search(r"(\d+)\s+tests?\s+collected", out.stdout + out.stderr)
    if not actual_m:
        report("test_count", False, f"could not parse pytest collect output (claimed: {claimed})")
        return
    actual = int(actual_m.group(1))

    # Allow +/-5% tolerance for test-suite churn between commits
    drift = abs(actual - claimed) / max(claimed, 1)
    if drift > 0.05:
        report("test_count", False,
               f"STATUS claims {claimed} tests, actual {actual} ({(actual-claimed)/claimed*100:+.1f}% drift)")
    else:
        report("test_count", True, f"{claimed} claimed ~ {actual} actual")


def check_pyproject_changelog_version() -> None:
    """pyproject.toml version matches CHANGELOG.md top non-Unreleased entry."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    declared = pyproject["project"]["version"]

    changelog_path = REPO_ROOT / "CHANGELOG.md"
    if not changelog_path.exists():
        report("version_consistency", True, f"no CHANGELOG.md (only checked pyproject={declared})")
        return

    changelog = changelog_path.read_text()
    # Match `## [X.Y.Z] - DATE` with em-dash, en-dash or hyphen separator
    pattern = r"^##\s+\[(\d+\.\d+\.\d+)\][\s—–\-]+\d{4}-\d{2}-\d{2}"
    match = re.search(pattern, changelog, re.MULTILINE)
    if not match:
        report("version_consistency", True,
               f"no semver-tagged dated release header in CHANGELOG.md (pyproject={declared})")
        return
    top_release = match.group(1)
    if declared == top_release:
        report("version_consistency", True, f"pyproject={declared} = CHANGELOG top {top_release}")
    else:
        report("version_consistency", False,
               f"pyproject={declared} != CHANGELOG top {top_release}")


def check_citation_version() -> None:
    """CITATION.cff top-level `version:` matches pyproject.toml version."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    declared = pyproject["project"]["version"]
    cff_path = REPO_ROOT / "CITATION.cff"
    if not cff_path.exists():
        report("citation_version", True, f"no CITATION.cff (skip; pyproject={declared})")
        return
    cff = cff_path.read_text()
    m = re.search(r'^version:\s*["\']?([0-9][^\s"\']+)["\']?\s*$',
                  cff, re.MULTILINE)
    if not m:
        report("citation_version", True,
               f"no top-level `version:` in CITATION.cff (skip; pyproject={declared})")
        return
    cff_version = m.group(1)
    if cff_version == declared:
        report("citation_version", True,
               f"CITATION.cff version={cff_version} = pyproject={declared}")
    else:
        report("citation_version", False,
               f"CITATION.cff version={cff_version} != pyproject={declared}")


def check_version_py_consistency() -> None:
    """pyproject.toml version matches src/<package>/_version.py.

    N6 Task 2 (2026-05-10): caught by N5 Task 6 — bouba_sens had
    pyproject 0.5.9 but _version.py 0.3.0 (4-minor stale).
    doc-truth now covers this 4th version source. Defensive: skips
    silently if no _version.py present.
    """
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    declared = pyproject["project"]["version"]

    candidates = list(REPO_ROOT.glob("src/**/_version.py")) + list(REPO_ROOT.glob("*/_version.py"))
    candidates = [c for c in candidates if ".venv" not in c.parts and "build" not in c.parts]
    if not candidates:
        report("version_py_consistency", True, "no _version.py (skip)")
        return

    version_file = candidates[0]
    text = version_file.read_text()
    m = re.search(r'__version__\s*=\s*["\']([\d.]+)["\']', text)
    if not m:
        report("version_py_consistency", False,
               f"no __version__ literal in {version_file.relative_to(REPO_ROOT)}")
        return
    file_ver = m.group(1)

    if declared == file_ver:
        report("version_py_consistency", True,
               f"pyproject={declared} = _version.py {file_ver}")
    else:
        report("version_py_consistency", False,
               f"pyproject={declared} != _version.py {file_ver} "
               f"({version_file.relative_to(REPO_ROOT)})")


def check_r1_hashes_status() -> None:
    """STATUS.md narrative about R1 hashes matches golden_hashes.json status fields.

    Detects the failure mode caught in N1: STATUS claiming promotion
    to validated_cross_machine while JSON entries remain pending_review.
    """
    hashes_path = REPO_ROOT / "tests" / "reproducibility" / "golden_hashes.json"
    if not hashes_path.exists():
        report("r1_status", True, "no golden_hashes.json (skip)")
        return

    data = json.loads(hashes_path.read_text())
    if not isinstance(data, dict):
        report("r1_status", True, "golden_hashes.json is not dict (skip)")
        return

    statuses = {k: v.get("status", "?") for k, v in data.items() if isinstance(v, dict)}
    promoted = [k for k, s in statuses.items() if s.startswith("validated")]
    pending = [k for k, s in statuses.items() if s == "pending_review"]

    status_path = REPO_ROOT / "STATUS.md"
    if not status_path.exists():
        report("r1_status", True, "no STATUS.md (skip)")
        return
    status = status_path.read_text()

    # Claim of present-tense promotion (not retracted/historical)
    # Look for active claim "promoted to validated_cross_machine_..." NOT
    # qualified by "was" / "previously" / "never propagated" / "corrected".
    promotion_re = re.compile(
        r"(?:promoted to|now)\s+`?validated_cross_machine[_\w\-]*`?", re.IGNORECASE)
    retraction_re = re.compile(
        r"(was a STATUS narrative|never propagated|previously[\s\-]claimed|"
        r"previously claimed promotion|corrected\s+\d{4}-\d{2}-\d{2})",
        re.IGNORECASE)

    claims_promoted = bool(promotion_re.search(status)) and not retraction_re.search(status)

    if promoted and not claims_promoted and not pending:
        report("r1_status", False,
               f"JSON has {len(promoted)} promoted entries but STATUS.md doesn't mention promotion")
    elif claims_promoted and pending:
        report("r1_status", False,
               f"STATUS.md actively claims promotion but {len(pending)} entries still pending_review in JSON")
    else:
        report("r1_status", True,
               f"narrative-truth aligned ({len(pending)} pending, {len(promoted)} promoted)")


def main() -> int:
    print("=" * 60)
    print(f"check_status_truth.py -- {REPO_ROOT.name}")
    print("=" * 60)
    check_test_count()
    check_pyproject_changelog_version()
    check_citation_version()
    check_version_py_consistency()
    check_r1_hashes_status()
    print("=" * 60)
    failed = [n for n, p, _ in results if not p]
    if failed:
        print(f"FAIL: {len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print(f"PASS: {len(results)} check(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
