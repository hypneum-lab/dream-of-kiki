"""Run registry — SQLite-backed, reproducibility contract R1."""
import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


class RunRegistry:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    c_version TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    commit_sha TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

    def _compute_run_id(
        self, c_version: str, profile: str, seed: int, commit_sha: str
    ) -> str:
        key = f"{c_version}|{profile}|{seed}|{commit_sha}".encode()
        return hashlib.sha256(key).hexdigest()[:16]

    def register(
        self, c_version: str, profile: str, seed: int, commit_sha: str
    ) -> str:
        run_id = self._compute_run_id(c_version, profile, seed, commit_sha)
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                "INSERT OR IGNORE INTO runs "
                "(run_id, c_version, profile, seed, commit_sha) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, c_version, profile, seed, commit_sha),
            )
        return run_id

    def get(self, run_id: str) -> dict[str, Any]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"run_id not found: {run_id}")
            return dict(row)
