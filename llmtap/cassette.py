"""Cassettes: VCR-style JSONL recordings of LLM exchanges for deterministic, keyless replay."""
from __future__ import annotations

import json
import re
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import CachedResponse

MODES: frozenset[str] = frozenset({"record", "replay", "auto"})

# sk-ant-..., sk-proj-..., sk-... (not inside a longer word, with a reasonably long tail)
_KEY_RE = re.compile(r"(?<![A-Za-z0-9_\-])sk-(?:ant-[A-Za-z0-9]*-?|proj-)?[A-Za-z0-9_\-]{16,}")
_SENSITIVE_HEADER_PARTS = ("key", "auth", "token", "cookie", "secret", "password")


class CassetteMiss(Exception):
    """Raised (by the proxy) when replay mode has no recording for a request."""


def _redact_str(s: str) -> str:
    def sub(m: re.Match[str]) -> str:
        return "[REDACTED:anthropic_key]" if m.group(0).startswith("sk-ant-") else "[REDACTED:api_key]"

    return _KEY_RE.sub(sub, s)


def _redact(obj: Any) -> Any:
    if isinstance(obj, str):
        return _redact_str(obj)
    if isinstance(obj, dict):
        return {k: _redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def redact_request_for_cassette(req: dict | str) -> dict | str:
    """Return a copy of ``req`` with API-key-looking strings (sk-..., sk-ant-...) replaced."""
    return _redact(req)


def strip_sensitive_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """Drop set-cookie and any header whose name mentions key/auth/token/cookie/secret/password."""
    out: dict[str, str] = {}
    for k, v in (headers or {}).items():
        lk = str(k).lower()
        if any(p in lk for p in _SENSITIVE_HEADER_PARTS):
            continue
        out[k] = v
    return out


class Cassette:
    """A JSONL cassette file. Thread-safe; ordered replay of duplicate keys."""

    def __init__(self, path: str, mode: str = "auto") -> None:
        if mode not in MODES:
            raise ValueError(f"invalid cassette mode {mode!r}; expected one of record, replay, auto")
        self.path = path
        self.mode = mode
        self._lock = threading.Lock()
        self._entries: dict[str, list[CachedResponse]] = {}
        self._cursor: dict[str, int] = {}
        self._count = 0
        p = Path(path)
        # record mode truncates lazily on the first write, so merely starting the proxy
        # in record mode never wipes an existing cassette.
        self._truncate_pending = mode == "record"
        if mode != "record" and p.is_file():
            self._load(p)
            if mode == "auto":
                self._ensure_trailing_newline(p)

    @staticmethod
    def _ensure_trailing_newline(p: Path) -> None:
        """So appended lines never glue onto a truncated final line."""
        with p.open("rb+") as f:
            f.seek(0, 2)
            if f.tell() == 0:
                return
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")

    def _load(self, p: Path) -> None:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    key = d["key"]
                    r = d["response"]
                    resp = CachedResponse(
                        status=int(r["status"]),
                        headers=dict(r.get("headers") or {}),
                        body=r.get("body", ""),
                        streamed=bool(r.get("streamed", False)),
                        cost_usd=float(r.get("cost_usd", 0.0) or 0.0),
                    )
                    if not isinstance(key, str) or not isinstance(resp.body, str):
                        raise TypeError("bad field types")
                except (ValueError, KeyError, TypeError, AttributeError) as e:
                    warnings.warn(f"llmtap cassette {self.path}: skipping corrupted line {lineno}: {e}", stacklevel=3)
                    continue
                self._entries.setdefault(key, []).append(resp)
                self._count += 1

    def lookup(self, key: str) -> CachedResponse | None:
        """Next recorded response for ``key`` (in order; last one repeats). Always None in record mode."""
        if self.mode == "record":
            return None
        with self._lock:
            recs = self._entries.get(key)
            if not recs:
                return None
            i = self._cursor.get(key, 0)
            self._cursor[key] = i + 1
            return recs[min(i, len(recs) - 1)]

    def should_record(self) -> bool:
        """True in record and auto modes."""
        return self.mode in ("record", "auto")

    def record(self, key: str, request: dict | str, resp: CachedResponse) -> None:
        """Append one JSON line (sorted keys) and flush. No-op in replay mode."""
        if self.mode == "replay":
            return
        clean = CachedResponse(
            status=int(resp.status),
            headers=strip_sensitive_headers(resp.headers),
            body=resp.body,
            streamed=bool(resp.streamed),
            cost_usd=float(resp.cost_usd or 0.0),
        )
        line = json.dumps(
            {
                "key": key,
                "request": redact_request_for_cassette(request),
                "response": {
                    "status": clean.status,
                    "headers": clean.headers,
                    "body": clean.body,
                    "streamed": clean.streamed,
                    "cost_usd": clean.cost_usd,
                },
                "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        with self._lock:
            p = Path(self.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            open_mode = "w" if self._truncate_pending else "a"
            self._truncate_pending = False
            with p.open(open_mode, encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
            self._entries.setdefault(key, []).append(clean)
            self._count += 1

    def __len__(self) -> int:
        with self._lock:
            return self._count
