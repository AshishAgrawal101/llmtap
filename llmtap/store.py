"""SQLite persistence for exchanges (thread-safe, stdlib only)."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import Exchange, Usage

SCHEMA_VERSION = 1

# (column name, SQL type/default). Order matters only for readability.
# To add a column in the future: append it here (with a DEFAULT) and bump SCHEMA_VERSION;
# `_migrate` adds any missing columns to existing databases via ALTER TABLE ADD COLUMN.
COLUMNS: list[tuple[str, str]] = [
    ("id", "TEXT PRIMARY KEY"),
    ("ts", "REAL NOT NULL DEFAULT 0"),
    ("provider", "TEXT NOT NULL DEFAULT 'unknown'"),
    ("style", "TEXT NOT NULL DEFAULT 'openai'"),
    ("method", "TEXT NOT NULL DEFAULT 'POST'"),
    ("path", "TEXT NOT NULL DEFAULT ''"),
    ("model", "TEXT"),
    ("request_body", "TEXT NOT NULL DEFAULT ''"),
    ("status", "INTEGER NOT NULL DEFAULT 0"),
    ("response_body", "TEXT NOT NULL DEFAULT ''"),
    ("streamed", "INTEGER NOT NULL DEFAULT 0"),
    ("latency_ms", "REAL NOT NULL DEFAULT 0"),
    ("ttft_ms", "REAL"),
    ("input_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("output_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("cache_read_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("cache_write_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("usage_estimated", "INTEGER NOT NULL DEFAULT 0"),
    ("cost_usd", "REAL NOT NULL DEFAULT 0"),
    ("saved_usd", "REAL NOT NULL DEFAULT 0"),
    ("source", "TEXT NOT NULL DEFAULT 'upstream'"),
    ("tags", "TEXT NOT NULL DEFAULT '{}'"),
    ("guard_findings", "TEXT NOT NULL DEFAULT '[]'"),
    ("error", "TEXT"),
    ("output_text", "TEXT NOT NULL DEFAULT ''"),
]
_COL_NAMES = [c for c, _ in COLUMNS]
_SELECT = ", ".join(_COL_NAMES)

_GROUP_EXPR = {
    "model": "model",
    "provider": "provider",
    "source": "source",
    "day": "strftime('%Y-%m-%d', ts, 'unixepoch', 'localtime')",
    "hour": "strftime('%Y-%m-%d %H:00', ts, 'unixepoch', 'localtime')",
}

_CACHE_SOURCES = ("cache", "cassette")
_BLOCK_SOURCES = ("blocked", "budget", "loop")


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class Store:
    """Thread-safe SQLite store of Exchange records."""

    def __init__(self, path: str) -> None:
        self.path = path
        is_file = path != ":memory:" and not path.startswith("file::memory:")
        if is_file:
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            path = str(Path(path).expanduser())
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            path, check_same_thread=False, timeout=30.0, isolation_level=None
        )
        with self._lock:
            c = self._conn
            if is_file:
                try:
                    c.execute("PRAGMA journal_mode=WAL")
                except sqlite3.DatabaseError:
                    pass
                c.execute("PRAGMA synchronous=NORMAL")
            self._migrate(c)

    # ------------------------------------------------------------------ schema
    @staticmethod
    def _migrate(c: sqlite3.Connection) -> None:
        cols = ",\n  ".join(f"{n} {t}" for n, t in COLUMNS)
        c.execute(f"CREATE TABLE IF NOT EXISTS exchanges (\n  {cols}\n)")
        existing = {row[1] for row in c.execute("PRAGMA table_info(exchanges)")}
        for name, typ in COLUMNS:
            if name not in existing:
                # ADD COLUMN can't take PRIMARY KEY; strip constraints that ALTER disallows.
                c.execute(f"ALTER TABLE exchanges ADD COLUMN {name} {typ.replace('PRIMARY KEY', '')}")
        c.execute("CREATE INDEX IF NOT EXISTS idx_exchanges_ts ON exchanges(ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_exchanges_source_ts ON exchanges(source, ts)")
        version = c.execute("PRAGMA user_version").fetchone()[0]
        if version < SCHEMA_VERSION:
            c.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def schema_version(self) -> int:
        """Return the on-disk schema version (PRAGMA user_version)."""
        with self._lock:
            return int(self._c().execute("PRAGMA user_version").fetchone()[0])

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            raise sqlite3.ProgrammingError("Store is closed")
        return self._conn

    # ------------------------------------------------------------------ (de)serialization
    @staticmethod
    def _to_row(ex: Exchange) -> tuple[Any, ...]:
        u = ex.usage or Usage()
        return (
            ex.id, float(ex.ts), ex.provider, ex.style, ex.method, ex.path, ex.model,
            ex.request_body or "", int(ex.status), ex.response_body or "", 1 if ex.streamed else 0,
            float(ex.latency_ms), None if ex.ttft_ms is None else float(ex.ttft_ms),
            int(u.input_tokens), int(u.output_tokens), int(u.cache_read_tokens), int(u.cache_write_tokens),
            1 if u.estimated else 0, float(ex.cost_usd), float(ex.saved_usd), ex.source,
            json.dumps(ex.tags or {}, ensure_ascii=False),
            json.dumps(ex.guard_findings or [], ensure_ascii=False, default=str),
            ex.error, ex.output_text or "",
        )

    @staticmethod
    def _from_row(row: tuple[Any, ...]) -> Exchange:
        r = dict(zip(_COL_NAMES, row))
        try:
            tags = json.loads(r["tags"] or "{}")
        except (TypeError, ValueError):
            tags = {}
        try:
            findings = json.loads(r["guard_findings"] or "[]")
        except (TypeError, ValueError):
            findings = []
        usage = Usage(
            input_tokens=int(r["input_tokens"] or 0),
            output_tokens=int(r["output_tokens"] or 0),
            cache_read_tokens=int(r["cache_read_tokens"] or 0),
            cache_write_tokens=int(r["cache_write_tokens"] or 0),
            estimated=bool(r["usage_estimated"]),
        )
        return Exchange.from_dict({
            "id": r["id"], "ts": float(r["ts"]), "provider": r["provider"], "style": r["style"],
            "method": r["method"], "path": r["path"], "model": r["model"],
            "request_body": r["request_body"], "status": int(r["status"]),
            "response_body": r["response_body"], "streamed": bool(r["streamed"]),
            "latency_ms": float(r["latency_ms"]),
            "ttft_ms": None if r["ttft_ms"] is None else float(r["ttft_ms"]),
            "usage": usage.__dict__, "cost_usd": float(r["cost_usd"]), "saved_usd": float(r["saved_usd"]),
            "source": r["source"], "tags": tags if isinstance(tags, dict) else {},
            "guard_findings": findings if isinstance(findings, list) else [],
            "error": r["error"], "output_text": r["output_text"],
        })

    # ------------------------------------------------------------------ writes
    def record(self, ex: Exchange) -> None:
        """Insert an exchange, replacing any existing row with the same id."""
        row = self._to_row(ex)
        placeholders = ", ".join("?" for _ in _COL_NAMES)
        with self._lock:
            self._c().execute(f"INSERT OR REPLACE INTO exchanges ({_SELECT}) VALUES ({placeholders})", row)

    def clear(self) -> None:
        """Delete all exchanges."""
        with self._lock:
            self._c().execute("DELETE FROM exchanges")

    def close(self) -> None:
        """Close the connection (idempotent)."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ------------------------------------------------------------------ reads
    def get(self, id: str) -> Exchange | None:
        """Fetch by exact id, or by a unique id prefix of at least 4 chars."""
        if not id:
            return None
        with self._lock:
            c = self._c()
            row = c.execute(f"SELECT {_SELECT} FROM exchanges WHERE id = ?", (id,)).fetchone()
            if row is None and len(id) >= 4:
                rows = c.execute(
                    f"SELECT {_SELECT} FROM exchanges WHERE id LIKE ? ESCAPE '\\' LIMIT 2",
                    (_like_escape(id) + "%",),
                ).fetchall()
                row = rows[0] if len(rows) == 1 else None
        return self._from_row(row) if row is not None else None

    def list(self, limit: int = 50, offset: int = 0, q: str | None = None, since: float | None = None,
             provider: str | None = None, model: str | None = None, source: str | None = None) -> list[Exchange]:
        """Newest-first exchanges matching the filters. `q` = substring over request/output/model/path."""
        where: list[str] = []
        args: list[Any] = []
        if since is not None:
            where.append("ts >= ?")
            args.append(float(since))
        if provider:
            where.append("provider = ?")
            args.append(provider)
        if model:
            where.append("model = ?")
            args.append(model)
        if source:
            where.append("source = ?")
            args.append(source)
        if q:
            pat = "%" + _like_escape(q) + "%"
            where.append(
                "(request_body LIKE ? ESCAPE '\\' OR output_text LIKE ? ESCAPE '\\' "
                "OR COALESCE(model,'') LIKE ? ESCAPE '\\' OR path LIKE ? ESCAPE '\\')"
            )
            args.extend([pat, pat, pat, pat])
        sql = f"SELECT {_SELECT} FROM exchanges"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC, rowid DESC LIMIT ? OFFSET ?"
        args.extend([int(limit) if limit is not None and limit >= 0 else -1, max(0, int(offset or 0))])
        with self._lock:
            rows = self._c().execute(sql, args).fetchall()
        return [self._from_row(r) for r in rows]

    def count(self) -> int:
        """Total number of stored exchanges."""
        with self._lock:
            return int(self._c().execute("SELECT COUNT(*) FROM exchanges").fetchone()[0])

    def stats(self, since: float | None = None, group_by: str | None = None) -> dict:
        """Aggregate totals, optionally broken down into groups."""
        where = ""
        args: list[Any] = []
        if since is not None:
            where = " WHERE ts >= ?"
            args.append(float(since))
        cs = ",".join(f"'{s}'" for s in _CACHE_SOURCES)
        bs = ",".join(f"'{s}'" for s in _BLOCK_SOURCES)
        totals_sql = (
            "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), "
            "COALESCE(SUM(cost_usd),0), COALESCE(SUM(saved_usd),0), COALESCE(AVG(CASE WHEN source = 'upstream' THEN latency_ms END),0), "
            f"COALESCE(SUM(CASE WHEN (status >= 400 OR source = 'error') AND source NOT IN ({bs}) THEN 1 ELSE 0 END),0), "
            f"COALESCE(SUM(CASE WHEN source IN ({cs}) THEN 1 ELSE 0 END),0), "
            f"COALESCE(SUM(CASE WHEN source IN ({bs}) THEN 1 ELSE 0 END),0), "
            "COALESCE(SUM(CASE WHEN json_valid(guard_findings) AND json_type(guard_findings) = 'array' "
            "THEN json_array_length(guard_findings) ELSE 0 END),0) "
            f"FROM exchanges{where}"
        )
        group_sql: str | None = None
        gargs: list[Any] = list(args)
        if group_by:
            if group_by in _GROUP_EXPR:
                expr = _GROUP_EXPR[group_by]
            elif group_by.startswith("tag:") and len(group_by) > 4:
                name = group_by[4:]
                expr = "(CASE WHEN json_valid(tags) THEN json_extract(tags, ?) END)"
                gargs = ['$."' + name.replace("\\", "\\\\").replace('"', '\\"') + '"'] + gargs
            else:
                raise ValueError(f"unsupported group_by: {group_by!r}")
            order = "key ASC" if group_by in ("day", "hour") else "cost_usd DESC, requests DESC, key ASC"
            group_sql = (
                f"SELECT {expr} AS key, COUNT(*) AS requests, COALESCE(SUM(cost_usd),0) AS cost_usd, "
                "COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), "
                "COALESCE(SUM(saved_usd),0), COALESCE(AVG(latency_ms),0) "
                f"FROM exchanges{where} GROUP BY key ORDER BY {order}"
            )
        with self._lock:
            c = self._c()
            t = c.execute(totals_sql, args).fetchone()
            grows = c.execute(group_sql, gargs).fetchall() if group_sql else []
        groups = [
            {"key": g[0], "requests": int(g[1]), "cost_usd": float(g[2]), "input_tokens": int(g[3]),
             "output_tokens": int(g[4]), "saved_usd": float(g[5]), "avg_latency_ms": float(g[6])}
            for g in grows
        ]
        return {
            "requests": int(t[0]),
            "input_tokens": int(t[1]),
            "output_tokens": int(t[2]),
            "cost_usd": float(t[3]),
            "saved_usd": float(t[4]),
            "avg_latency_ms": float(t[5]),
            "errors": int(t[6]),
            "cache_hits": int(t[7]),
            "blocked": int(t[8]),
            "guard_findings": int(t[9]),
            "groups": groups,
        }

    def spent_since(self, since: float) -> float:
        """Sum of cost_usd for upstream (actually paid) calls since `since`."""
        with self._lock:
            row = self._c().execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM exchanges WHERE source = 'upstream' AND ts >= ?",
                (float(since),),
            ).fetchone()
        return float(row[0])

    def timeseries(self, since: float, bucket_s: int = 3600) -> list[dict]:
        """Ascending buckets: [{"t": bucket_start, "requests", "cost_usd", "tokens"}]."""
        b = max(1, int(bucket_s))
        with self._lock:
            rows = self._c().execute(
                "SELECT CAST(ts / ? AS INTEGER) * ? AS t, COUNT(*), COALESCE(SUM(cost_usd),0), "
                "COALESCE(SUM(input_tokens + output_tokens),0) "
                "FROM exchanges WHERE ts >= ? GROUP BY t ORDER BY t ASC",
                (b, b, float(since if since is not None else 0)),
            ).fetchall()
        return [{"t": int(r[0]), "requests": int(r[1]), "cost_usd": float(r[2]), "tokens": int(r[3])} for r in rows]

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
