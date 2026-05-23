"""
modules/cache.py — Mock Redis using an in-memory dict + TTL.
Drop-in replacement; swap for redis-py later with no upstream changes needed.

Usage:
    from modules.cache import cache     # singleton
    cache.set("query:abc", result, ttl=3600)
    val = cache.get("query:abc")        # None if expired or missing
"""

import time
import fnmatch
import threading
from typing import Any, Optional


class MockRedis:
    """Thread-safe in-memory store that mimics the Redis API subset we need."""

    def __init__(self):
        # key → (value, expires_at_unix | None)
        self._store: dict[str, tuple[Any, Optional[float]]] = {}
        self._lock = threading.Lock()

    # ── Core ops ──────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, exp = entry
            if exp is not None and time.time() > exp:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        ttl: seconds until expiry. None = permanent.
        """
        exp = (time.time() + ttl) if ttl is not None else None
        with self._lock:
            self._store[key] = (value, exp)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    # ── Inspection ───────────────────────────────────────────────────────────

    def keys(self, pattern: str = "*") -> list[str]:
        """Glob-style pattern, e.g. 'faiss:*' or 'query:*'."""
        now = time.time()
        with self._lock:
            live = [
                k for k, (_, exp) in self._store.items()
                if exp is None or exp > now
            ]
        return [k for k in live if fnmatch.fnmatch(k, pattern)]

    def ttl(self, key: str) -> Optional[float]:
        """Remaining seconds, or None if permanent, or -1 if missing/expired."""
        with self._lock:
            entry = self._store.get(key)
        if entry is None:
            return -1
        _, exp = entry
        if exp is None:
            return None
        remaining = exp - time.time()
        return remaining if remaining > 0 else -1

    def info(self) -> list[dict]:
        """Human-readable summary of all live keys (for /cache command)."""
        now = time.time()
        rows = []
        with self._lock:
            for k, (v, exp) in self._store.items():
                if exp is not None and exp <= now:
                    continue
                size_hint = len(str(v)) if not hasattr(v, "__len__") else len(v)
                rows.append({
                    "key": k,
                    "ttl": f"{exp - now:.0f}s" if exp else "∞",
                    "size": f"~{size_hint} chars",
                })
        return sorted(rows, key=lambda r: r["key"])

    def flush(self) -> None:
        with self._lock:
            self._store.clear()


# ── Singleton used by all modules ─────────────────────────────────────────────
cache = MockRedis()
