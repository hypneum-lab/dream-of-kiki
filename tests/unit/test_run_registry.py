"""Unit tests for run registry (SQLite-backed)."""
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from harness.storage.run_registry import RunRegistry


@pytest.fixture
def tmp_registry(tmp_path: Path) -> RunRegistry:
    db_path = tmp_path / "runs.db"
    return RunRegistry(db_path=db_path)


def test_register_run_creates_entry(tmp_registry: RunRegistry) -> None:
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    assert run_id is not None
    assert tmp_registry.get(run_id)["profile"] == "P_min"


def test_register_run_is_idempotent_for_same_inputs(tmp_registry: RunRegistry) -> None:
    run_id_1 = tmp_registry.register(
        c_version="C-v0.1.0+STABLE", profile="P_equ", seed=1,
        commit_sha="def",
    )
    run_id_2 = tmp_registry.register(
        c_version="C-v0.1.0+STABLE", profile="P_equ", seed=1,
        commit_sha="def",
    )
    assert run_id_1 == run_id_2  # Deterministic run_id for repro contract R1


def test_run_id_has_128_bit_entropy(tmp_registry: RunRegistry) -> None:
    # 128 bits = 32 hex chars — collision negligible at any scale
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_equ",
        seed=1,
        commit_sha="xyz",
    )
    assert len(run_id) == 32
    assert all(c in "0123456789abcdef" for c in run_id)


# --------------------------------------------------------------------------
# Output-hash API — second half of R1 (recorded output is stable)
# --------------------------------------------------------------------------


def test_register_output_hash_stores_and_roundtrips(
    tmp_registry: RunRegistry,
) -> None:
    """``register_output_hash`` + ``get_output_hash`` round-trip."""
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    output_hash = "a" * 64
    tmp_registry.register_output_hash(run_id, output_hash)
    assert tmp_registry.get_output_hash(run_id) == output_hash


def test_register_output_hash_idempotent_on_exact_match(
    tmp_registry: RunRegistry,
) -> None:
    """Registering the same (run_id, hash) twice is a silent no-op."""
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    output_hash = "b" * 64
    tmp_registry.register_output_hash(run_id, output_hash)
    # Second call must not raise and must not mutate the stored value.
    tmp_registry.register_output_hash(run_id, output_hash)
    assert tmp_registry.get_output_hash(run_id) == output_hash


def test_register_output_hash_raises_on_conflict(
    tmp_registry: RunRegistry,
) -> None:
    """Conflicting hash for an existing run_id → R1-labelled ValueError."""
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    tmp_registry.register_output_hash(run_id, "c" * 64)
    with pytest.raises(ValueError) as excinfo:
        tmp_registry.register_output_hash(run_id, "d" * 64)
    message = str(excinfo.value)
    assert "R1" in message
    assert run_id in message


def test_register_output_hash_raises_on_unknown_run_id(
    tmp_registry: RunRegistry,
) -> None:
    """Unknown run_id must not silently acquire an output hash."""
    with pytest.raises(KeyError):
        tmp_registry.register_output_hash("deadbeef" * 4, "e" * 64)


def test_get_output_hash_raises_when_missing(
    tmp_registry: RunRegistry,
) -> None:
    """Registered run without a recorded hash → KeyError."""
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    with pytest.raises(KeyError):
        tmp_registry.get_output_hash(run_id)


# --------------------------------------------------------------------------
# Multi-artifact output-hash API — issue #2 (Paper 2 support)
# --------------------------------------------------------------------------


def test_backcompat_register_output_hash_without_artifact_name(
    tmp_registry: RunRegistry,
) -> None:
    """Pre-existing callers (no kwargs) roundtrip under 'canonical'."""
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    output_hash = "f" * 64
    tmp_registry.register_output_hash(run_id, output_hash)
    assert tmp_registry.get_output_hash(run_id) == output_hash
    assert tmp_registry.list_output_hashes(run_id) == {"canonical": output_hash}


def test_register_two_artifacts_same_run_coexist(
    tmp_registry: RunRegistry,
) -> None:
    """Two distinct artifact_names for the same run_id live side-by-side."""
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    hash_a = "1" * 64
    hash_b = "2" * 64
    tmp_registry.register_output_hash(run_id, hash_a, artifact_name="ckpt-step-100")
    tmp_registry.register_output_hash(run_id, hash_b, artifact_name="metrics")
    listed = tmp_registry.list_output_hashes(run_id)
    assert listed == {"ckpt-step-100": hash_a, "metrics": hash_b}
    assert tmp_registry.get_output_hash(run_id, artifact_name="ckpt-step-100") == hash_a
    assert tmp_registry.get_output_hash(run_id, artifact_name="metrics") == hash_b


def test_register_same_run_same_artifact_idempotent_exact(
    tmp_registry: RunRegistry,
) -> None:
    """Same (run, artifact, hash, type) twice is a silent no-op."""
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    h = "3" * 64
    tmp_registry.register_output_hash(run_id, h, artifact_name="ckpt-step-100")
    tmp_registry.register_output_hash(run_id, h, artifact_name="ckpt-step-100")
    assert (
        tmp_registry.get_output_hash(run_id, artifact_name="ckpt-step-100") == h
    )


def test_register_same_run_same_artifact_conflict_raises(
    tmp_registry: RunRegistry,
) -> None:
    """Same (run, artifact), different hash → R1 ValueError."""
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    tmp_registry.register_output_hash(run_id, "4" * 64, artifact_name="ckpt-step-100")
    with pytest.raises(ValueError) as excinfo:
        tmp_registry.register_output_hash(
            run_id, "5" * 64, artifact_name="ckpt-step-100"
        )
    message = str(excinfo.value)
    assert "R1" in message
    assert run_id in message
    assert "ckpt-step-100" in message


def test_register_different_hash_type_on_same_artifact_raises(
    tmp_registry: RunRegistry,
) -> None:
    """Same (run, artifact), different hash_type → R1 ValueError."""
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    h = "6" * 64
    tmp_registry.register_output_hash(run_id, h, artifact_name="metrics")
    with pytest.raises(ValueError) as excinfo:
        tmp_registry.register_output_hash(
            run_id, h, artifact_name="metrics", hash_type="blake2b"
        )
    message = str(excinfo.value)
    assert "R1" in message
    assert run_id in message
    assert "metrics" in message


def test_list_output_hashes_empty_for_registered_run_no_artifacts(
    tmp_registry: RunRegistry,
) -> None:
    """A registered run without any hashes yields an empty dict."""
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    assert tmp_registry.list_output_hashes(run_id) == {}


def test_list_output_hashes_unknown_run_returns_empty(
    tmp_registry: RunRegistry,
) -> None:
    """Unknown run_id → empty dict (documented symmetry, no raise)."""
    assert tmp_registry.list_output_hashes("deadbeef" * 4) == {}


# --------------------------------------------------------------------------
# Schema-migration from the v1 single-hash-per-run layout
# --------------------------------------------------------------------------


_V1_SCHEMA_SQL = """
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    c_version TEXT NOT NULL,
    profile TEXT NOT NULL,
    seed INTEGER NOT NULL,
    commit_sha TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE run_output_hashes (
    run_id TEXT PRIMARY KEY,
    output_hash TEXT NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""


def test_migration_from_v1_schema_preserves_canonical_hashes(
    tmp_path: Path,
) -> None:
    """Old schema opens cleanly; rows migrate to artifact_name='canonical'."""
    db_path = tmp_path / "runs.db"
    # Build a DB with the legacy schema + 2 pre-populated rows.
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.executescript(_V1_SCHEMA_SQL)
        conn.execute(
            "INSERT INTO runs (run_id, c_version, profile, seed, commit_sha) "
            "VALUES (?, ?, ?, ?, ?)",
            ("run_a" * 6 + "ab", "C-v0.1.0+STABLE", "P_min", 1, "sha_a"),
        )
        conn.execute(
            "INSERT INTO runs (run_id, c_version, profile, seed, commit_sha) "
            "VALUES (?, ?, ?, ?, ?)",
            ("run_b" * 6 + "cd", "C-v0.1.0+STABLE", "P_equ", 2, "sha_b"),
        )
        conn.execute(
            "INSERT INTO run_output_hashes (run_id, output_hash) VALUES (?, ?)",
            ("run_a" * 6 + "ab", "a" * 64),
        )
        conn.execute(
            "INSERT INTO run_output_hashes (run_id, output_hash) VALUES (?, ?)",
            ("run_b" * 6 + "cd", "b" * 64),
        )

    # Open via RunRegistry — migration runs in _ensure_schema.
    registry = RunRegistry(db_path=db_path)

    rid_a = "run_a" * 6 + "ab"
    rid_b = "run_b" * 6 + "cd"
    # Backward-compat accessors work without kwargs.
    assert registry.get_output_hash(rid_a) == "a" * 64
    assert registry.get_output_hash(rid_b) == "b" * 64
    assert registry.list_output_hashes(rid_a) == {"canonical": "a" * 64}
    assert registry.list_output_hashes(rid_b) == {"canonical": "b" * 64}
    # The legacy-named backup table must be gone.
    with closing(sqlite3.connect(db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "run_output_hashes_v1" not in tables
    assert "run_output_hashes" in tables


def test_migration_is_idempotent_on_new_schema(tmp_path: Path) -> None:
    """Re-instantiating over an already-migrated DB is a no-op."""
    db_path = tmp_path / "runs.db"
    first = RunRegistry(db_path=db_path)
    run_id = first.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    first.register_output_hash(run_id, "9" * 64, artifact_name="ckpt")
    first.register_output_hash(run_id, "8" * 64, artifact_name="metrics")

    # Re-instantiate → _ensure_schema runs again over new-schema DB.
    second = RunRegistry(db_path=db_path)
    assert second.list_output_hashes(run_id) == {
        "ckpt": "9" * 64,
        "metrics": "8" * 64,
    }
