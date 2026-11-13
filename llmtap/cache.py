"""Response cache: identical requests are served from SQLite instead of hitting the upstream."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from .models import CachedResponse

VOLATILE_KEYS: frozenset[str] = frozenset({"user", "metadata", "request_id"})


def _canonical_body(req_body: bytes) -> bytes:
    """Canonical JSON (sorted keys, compact) minus volatile top-level keys; raw bytes if not JSON."""
    try:
        obj = json.loads(req_body.decode("utf-8")) if req_body else None
    except (UnicodeDecodeError, ValueError):
        return bytes(req_body)
    if obj is None and not req_body:
        return b""
    if isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if k not in VOLATILE_KEYS}
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def cache_key(provider: str, path: str, req_body: bytes) -> str:
    """sha256 hex over provider, path (no query) and the canonicalized request body."""
    bare_path = path.split("?", 1)[0]
    h = hashlib.sha256()
    h.update(provider.encode("utf-8"))
    h.update(b"\n")
    h.update(bare_path.encode("utf-8"))
    h.update(b"\n")
    h.update(_canonical_body(req_body))
    return h.hexdigest()


class ResponseCache:
    """Thread-safe SQLite-backed cache of 2xx responses (table ``cache``)."""

    def __init__(self, path: str, ttl_s: int = 0) -> None:
        self.path = path
        self.ttl_s = int(ttl_s or 0)
        self._lock = threading.Lock()
        self._hits = 0
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        with self._lock:
            if path != ":memory:":
                try:
                    self._conn.execute("PRAGMA journal_mode=WAL")
                except sqlite3.DatabaseError:
                    pass
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                " key TEXT PRIMARY KEY,"
                " status INTEGER NOT NULL,"
                " headers TEXT NOT NULL,"
                " body TEXT NOT NULL,"
                " streamed INTEGER NOT NULL,"
                " cost_usd REAL NOT NULL DEFAULT 0,"
                " created_at REAL NOT NULL)"
            )
            self._conn.commit()

    def _expired(self, created_at: float) -> bool:
        return self.ttl_s > 0 and (time.time() - created_at) > self.ttl_s

    def get(self, key: str) -> CachedResponse | None:
        """Return the cached response for ``key`` or None (missing or older than ttl)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT status, headers, body, streamed, cost_usd, created_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            status, headers, body, streamed, cost, created_at = row
            if self._expired(float(created_at)):
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._conn.commit()
                return None
            try:
                hdrs = json.loads(headers) or {}
            except ValueError:
                hdrs = {}
            self._hits += 1
            return CachedResponse(
                status=int(status), headers=dict(hdrs), body=body, streamed=bool(streamed), cost_usd=float(cost or 0.0)
            )

    def put(self, key: str, resp: CachedResponse) -> None:
        """Store (or replace) a response. Callers should only pass 2xx responses."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, status, headers, body, streamed, cost_usd, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    int(resp.status),
                    json.dumps(dict(resp.headers or {}), sort_keys=True),
                    resp.body,
                    1 if resp.streamed else 0,
                    float(resp.cost_usd or 0.0),
                    time.time(),
                ),
            )
            self._conn.commit()

    def stats(self) -> dict:
        """{"entries": rows in table, "hits": hits served by this process}."""
        with self._lock:
            (n,) = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            return {"entries": int(n), "hits": self._hits}

    def clear(self) -> None:
        """Delete every cached entry and reset the hit counter."""
        with self._lock:
            self._conn.execute("DELETE FROM cache")
            self._conn.commit()
            self._hits = 0

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()
