"""SQLite persistence for SpectroGuard.

Two small tables:
* ``users``      - registered accounts (email, salted PBKDF2 password hash).
* ``anon_usage`` - how many free checks each anonymous client has used.

SQLite keeps the prototype dependency-free (stdlib ``sqlite3``) while still
giving real persistence. Note: on ephemeral deployments (Docker/HF Spaces) the
database resets with the container - acceptable for a prototype.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .utils import ensure_dir

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "spectroguard.db"

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    ensure_dir(DATA_DIR)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if missing (idempotent; keeps the DB self-healing)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id            TEXT PRIMARY KEY,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt          TEXT NOT NULL,
            created_at    REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS anon_usage (
            client_id TEXT PRIMARY KEY,
            used      INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def _execute(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur
        finally:
            conn.close()


def _query_one(sql: str, params: tuple = ()) -> sqlite3.Row | None:
    with _lock:
        conn = _connect()
        try:
            return conn.execute(sql, params).fetchone()
        finally:
            conn.close()


def init_db() -> None:
    """Create tables on first run (called at application startup)."""
    conn = _connect()
    conn.close()


# --------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------
def create_user(user_id: str, email: str, password_hash: str, salt: str) -> None:
    _execute(
        "INSERT INTO users (id, email, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, email, password_hash, salt, time_now()),
    )


def get_user_by_email(email: str) -> sqlite3.Row | None:
    return _query_one("SELECT * FROM users WHERE email = ?", (email,))


def get_user_by_id(user_id: str) -> sqlite3.Row | None:
    return _query_one("SELECT * FROM users WHERE id = ?", (user_id,))


# --------------------------------------------------------------------------
# Anonymous free-check usage
# --------------------------------------------------------------------------
def get_anon_used(client_id: str) -> int:
    row = _query_one("SELECT used FROM anon_usage WHERE client_id = ?", (client_id,))
    return row["used"] if row else 0


def increment_anon_used(client_id: str) -> int:
    """Increment the free-check counter and return the new value."""
    current = get_anon_used(client_id)
    if current == 0:
        _execute(
            "INSERT INTO anon_usage (client_id, used) VALUES (?, ?)",
            (client_id, 1),
        )
    else:
        _execute(
            "UPDATE anon_usage SET used = used + 1 WHERE client_id = ?",
            (client_id,),
        )
    return current + 1


def time_now() -> float:
    import time

    return time.time()
