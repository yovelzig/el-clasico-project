"""
SQLite database module for persistent chat history.
Stores sessions, messages, and timestamps.
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "database/chat_history.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    conn.close()


def ensure_session(session_id: str):
    conn = get_conn()
    exists = conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not exists:
        conn.execute("INSERT INTO sessions VALUES (?,?)", (session_id, datetime.utcnow().isoformat()))
        conn.commit()
    conn.close()


def save_message(session_id: str, role: str, content: str):
    ensure_session(session_id)
    conn = get_conn()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
        (session_id, role, content, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(session_id: str, limit: int = 20) -> list:
    """Return last N messages as list of dicts."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content, timestamp FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def get_all_sessions() -> list:
    conn = get_conn()
    rows = conn.execute("SELECT id, created_at FROM sessions ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]
