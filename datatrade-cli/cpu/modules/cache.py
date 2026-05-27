"""
modules/cache.py — SQLite-backed persistent cache.

Survives restarts. Same API as MockRedis so all upstream code works unchanged.
"""

import pickle
import time
import fnmatch
import sqlite3
import threading
from typing import Any, Optional

from config import CACHE_DB


class SqliteCache:
    def __init__(self, db_path: str = CACHE_DB):
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self):
        self._conn().execute("""
            CREATE TABLE IF NOT EXISTS kv (
                key   TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                exp   REAL
            )
        """)
        self._conn().commit()

    # ── Core ops ──────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        row = self._conn().execute(
            "SELECT value, exp FROM kv WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        value_blob, exp = row
        if exp is not None and time.time() > exp:
            self._conn().execute("DELETE FROM kv WHERE key = ?", (key,))
            self._conn().commit()
            return None
        return pickle.loads(value_blob)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        exp = (time.time() + ttl) if ttl is not None else None
        self._conn().execute(
            "INSERT OR REPLACE INTO kv (key, value, exp) VALUES (?, ?, ?)",
            (key, pickle.dumps(value), exp),
        )
        self._conn().commit()

    def delete(self, key: str) -> bool:
        cur = self._conn().execute("DELETE FROM kv WHERE key = ?", (key,))
        self._conn().commit()
        return cur.rowcount > 0

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    # ── Inspection ───────────────────────────────────────────────────────────

    def keys(self, pattern: str = "*") -> list[str]:
        now = time.time()
        rows = self._conn().execute(
            "SELECT key, exp FROM kv"
        ).fetchall()
        live = [k for k, exp in rows if exp is None or exp > now]
        return [k for k in live if fnmatch.fnmatch(k, pattern)]

    def ttl(self, key: str) -> Optional[float]:
        row = self._conn().execute(
            "SELECT exp FROM kv WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return -1
        exp = row[0]
        if exp is None:
            return None
        remaining = exp - time.time()
        return remaining if remaining > 0 else -1

    def info(self) -> list[dict]:
        now = time.time()
        rows = self._conn().execute("SELECT key, value, exp FROM kv").fetchall()
        result = []
        for k, v, exp in rows:
            if exp is not None and exp <= now:
                continue
            result.append({
                "key": k,
                "ttl": f"{exp - now:.0f}s" if exp else "∞",
                "size": f"~{len(v)} bytes",
            })
        return sorted(result, key=lambda r: r["key"])

    def flush(self) -> None:
        self._conn().execute("DELETE FROM kv")
        self._conn().commit()


# ── Singleton used by all modules ─────────────────────────────────────────────
cache = SqliteCache()
