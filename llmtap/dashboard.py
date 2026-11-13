"""Live web dashboard + JSON API, served by the proxy under ``/_llmtap/``.

Everything here talks to the proxy through a tiny surface: ``tap.store``, ``tap.cache``,
``tap.cassette``, ``tap.cfg``, ``tap.budget`` and ``tap.bus`` — plus the ``Responder`` protocol
(``start``/``write``/``end``), so it is independent from ``http.server``.
"""
from __future__ import annotations

import json
import queue
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol
from urllib.parse import parse_qs, unquote, urlsplit

from . import __version__
from .models import Exchange

if TYPE_CHECKING:  # pragma: no cover
    from .proxy import LLMTap

PREFIX = "/_llmtap"
UI_PATH = Path(__file__).parent / "ui" / "index.html"
HEARTBEAT_S: float = 15.0          # SSE keep-alive interval (": ping")
PREVIEW_CHARS = 200
MAX_LIMIT = 1000


class Responder(Protocol):
    def start(self, status: int, headers: dict[str, str]) -> None: ...
    def write(self, chunk: bytes) -> None: ...
    def end(self) -> None: ...


# --------------------------------------------------------------------------------------------
# summaries / previews
# --------------------------------------------------------------------------------------------

def _clip(text: str, n: int = PREVIEW_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _content_text(content: Any) -> str:
    """Text of a message ``content``: a string, or a list of blocks (anthropic / openai / responses)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    fallback: list[str] = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict):
            btype = block.get("type")
            if isinstance(block.get("text"), str) and btype in (None, "text", "input_text", "output_text"):
                texts.append(block["text"])
            elif btype == "tool_result":
                fallback.append(_content_text(block.get("content")))
            elif btype in ("image", "image_url", "input_image"):
                fallback.append("[image]")
            elif isinstance(block.get("text"), str):
                texts.append(block["text"])
    return "\n".join(t for t in texts if t) or "\n".join(t for t in fallback if t)


def _last_user_text(req: Any) -> str:
    """Last user message text from an anthropic / openai chat / responses / embeddings / completions body."""
    if not isinstance(req, dict):
        return ""
    msgs = req.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                text = _content_text(m.get("content"))
                if text:
                    return text
        for m in reversed(msgs):  # no user text: fall back to any message text
            if isinstance(m, dict):
                text = _content_text(m.get("content"))
                if text:
                    return text
    inp = req.get("input")
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list):
        for item in reversed(inp):
            if isinstance(item, str):
                return item
            if isinstance(item, dict) and item.get("role", "user") == "user":
                text = _content_text(item.get("content"))
                if text:
                    return text
    prompt = req.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt and isinstance(prompt[-1], str):
        return prompt[-1]
    return ""


def request_preview(request_body: str) -> str:
    """Preview of the last user message in a raw request body (<= 200 chars)."""
    if not request_body:
        return ""
    try:
        req = json.loads(request_body)
    except (ValueError, TypeError):
        return _clip(request_body)
    if not isinstance(req, dict):
        return _clip(request_body)
    return _clip(_last_user_text(req))


def summary(ex: Exchange | Mapping[str, Any]) -> dict[str, Any]:
    """Exchange dict minus request/response bodies, plus ``preview`` and ``output_preview``."""
    d = ex.to_dict() if isinstance(ex, Exchange) else dict(ex)
    req = d.pop("request_body", "") or ""
    d.pop("response_body", None)
    if "preview" not in d:
        d["preview"] = request_preview(req)
    if "output_preview" not in d:
        out = d.get("output_text") or ""
        d["output_preview"] = _clip(out or (d.get("error") or ""))
    d.pop("output_text", None)
    return d


exchange_summary = summary  # alias


# --------------------------------------------------------------------------------------------
# response helpers
# --------------------------------------------------------------------------------------------

def _send(respond: Responder, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
    headers = {
        "content-type": ctype,
        "content-length": str(len(body)),
        "cache-control": "no-store",
        "x-content-type-options": "nosniff",
    }
    if extra:
        headers.update(extra)
    respond.start(status, headers)
    if body:
        respond.write(body)
    respond.end()


def _json(respond: Responder, obj: Any, status: int = 200) -> None:
    body = json.dumps(obj, default=str, separators=(",", ":")).encode("utf-8")
    _send(respond, status, body, "application/json; charset=utf-8")


def _error(respond: Responder, status: int, message: str) -> None:
    _json(respond, {"error": message}, status)


def _first(qs: dict[str, list[str]], name: str) -> str | None:
    vals = qs.get(name)
    if not vals:
        return None
    v = vals[0].strip()
    return v or None


def _int(qs: dict[str, list[str]], name: str, default: int, lo: int, hi: int) -> int:
    raw = _first(qs, name)
    if raw is None:
        return default
    try:
        return max(lo, min(hi, int(float(raw))))
    except ValueError:
        return default


def _since(qs: dict[str, list[str]], default: float | None = None) -> float | None:
    """``since`` is an epoch timestamp, or (if < 1e9) a number of seconds ago. ``all``/``0`` = no cutoff."""
    raw = _first(qs, "since")
    if raw is None:
        return default
    if raw.lower() in ("all", "0"):
        return None
    try:
        v = float(raw)
    except ValueError:
        return default
    if v <= 0:
        return None
    return v if v >= 1e9 else time.time() - v


def _local_midnight() -> float:
    t = time.localtime()
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


def _header(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    try:
        v = headers.get(name)
        if v is None:
            v = headers.get(name.title())
        if v is not None:
            return v
        for k in headers.keys():
            if k.lower() == name:
                return headers[k]
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------------------------

def _stats(tap: "LLMTap", qs: dict[str, list[str]]) -> dict[str, Any]:
    group_by = _first(qs, "group_by")
    since = _since(qs)
    out: dict[str, Any] = dict(tap.store.stats(since=since, group_by=group_by))
    out["since"] = since
    try:
        out["session_spent"] = float(tap.budget.session_spent())
    except Exception:
        out["session_spent"] = 0.0
    try:
        out["spent_today"] = float(tap.store.spent_since(_local_midnight()))
    except Exception:
        pass
    cache = getattr(tap, "cache", None)
    try:
        out["cache"] = cache.stats() if cache is not None else None
    except Exception:
        out["cache"] = {"entries": 0, "hits": 0}
    cas = getattr(tap, "cassette", None)
    if cas is not None:
        try:
            entries = len(cas)
        except Exception:
            entries = 0
        out["cassette"] = {"path": str(getattr(cas, "path", "") or getattr(tap.cfg, "cassette", "")),
                           "mode": getattr(cas, "mode", None) or getattr(tap.cfg, "mode", "auto"),
                           "entries": entries}
    else:
        out["cassette"] = None
    cfg = tap.cfg
    out["config"] = {
        "guard": cfg.guard,
        "guard_pii": cfg.guard_pii,
        "cache": bool(cfg.cache),
        "cache_ttl_s": cfg.cache_ttl_s,
        "budget_daily_usd": cfg.budget_daily_usd,
        "budget_session_usd": cfg.budget_session_usd,
        "loop_threshold": cfg.loop_threshold,
        "loop_window_s": cfg.loop_window_s,
        "mode": cfg.mode,
        "port": getattr(tap, "port", cfg.port),
    }
    out["version"] = __version__
    return out


def _exchanges(tap: "LLMTap", qs: dict[str, list[str]]) -> dict[str, Any]:
    limit = _int(qs, "limit", 50, 1, MAX_LIMIT)
    offset = _int(qs, "offset", 0, 0, 10**9)
    items = tap.store.list(limit=limit, offset=offset, q=_first(qs, "q"), since=_since(qs),
                           provider=_first(qs, "provider"), model=_first(qs, "model"),
                           source=_first(qs, "source"))
    return {"items": [summary(ex) for ex in items], "limit": limit, "offset": offset}


def _timeseries(tap: "LLMTap", qs: dict[str, list[str]]) -> dict[str, Any]:
    since = _since(qs, default=time.time() - 86400) or 0.0
    bucket = _int(qs, "bucket", 3600, 1, 86400 * 31)
    return {"since": since, "bucket": bucket, "points": tap.store.timeseries(since, bucket)}


def _event_payload(event: Any) -> dict[str, Any]:
    """Normalize a bus event: make sure any embedded exchange is a lightweight summary dict."""
    if isinstance(event, Exchange):
        return {"type": "exchange", "exchange": summary(event)}
    if not isinstance(event, dict):
        return {"type": "message", "data": event}
    ex = event.get("exchange")
    if isinstance(ex, Exchange) or (isinstance(ex, dict) and ("request_body" in ex or "response_body" in ex)):
        event = dict(event)
        event["exchange"] = summary(ex)
    return event


def _events(tap: "LLMTap", respond: Responder) -> None:
    """Server-sent events: one ``data:`` frame per bus event, ``: ping`` heartbeats."""
    bus = tap.bus
    q = bus.subscribe()
    try:
        respond.start(200, {
            "content-type": "text/event-stream; charset=utf-8",
            "cache-control": "no-cache, no-store",
            "transfer-encoding": "chunked",
            "x-accel-buffering": "no",
        })
        respond.write(b"retry: 2000\n: connected\n\n")
        while True:
            try:
                event = q.get(timeout=HEARTBEAT_S)
            except queue.Empty:
                respond.write(b": ping\n\n")
                continue
            if event is None:  # sentinel: shut down this stream
                break
            data = json.dumps(_event_payload(event), default=str, separators=(",", ":"))
            respond.write(f"data: {data}\n\n".encode("utf-8"))
    except Exception:  # client went away (BrokenPipe / ConnectionReset / ...)
        pass
    finally:
        try:
            bus.unsubscribe(q)
        except Exception:
            pass
        try:
            respond.end()
        except Exception:
            pass


def _same_origin(headers: Any) -> bool:
    """Reject cross-site POSTs (a random web page must not be able to wipe your history)."""
    origin = _header(headers, "origin")
    if not origin or origin == "null":
        return origin is None
    host = _header(headers, "host") or ""
    return urlsplit(origin).netloc.lower() == host.lower()


def handle_internal(tap: "LLMTap", method: str, path: str, headers: Any, body: bytes, respond: Responder) -> bool:
    """Serve ``/_llmtap/*``. Returns True if the request was handled (always, for that prefix)."""
    parts = urlsplit(path)
    route = unquote(parts.path)
    if not (route == PREFIX or route.startswith(PREFIX + "/")):
        return False
    qs = parse_qs(parts.query, keep_blank_values=False)
    sub = route[len(PREFIX):] or ""
    method = method.upper()

    try:
        if sub == "":
            _send(respond, 302, b"", "text/plain", {"location": PREFIX + "/"})
            return True

        if method in ("GET", "HEAD"):
            if sub in ("/", "/index.html"):
                try:
                    html = UI_PATH.read_bytes()
                except OSError:
                    html = b"<!doctype html><title>llmtap</title><p>llmtap UI missing (llmtap/ui/index.html)."
                _send(respond, 200, b"" if method == "HEAD" else html, "text/html; charset=utf-8")
                return True
            if sub == "/health":
                _json(respond, {"ok": True, "version": __version__})
                return True
            if sub == "/api/stats":
                _json(respond, _stats(tap, qs))
                return True
            if sub == "/api/exchanges":
                _json(respond, _exchanges(tap, qs))
                return True
            if sub.startswith("/api/exchanges/"):
                ex_id = sub[len("/api/exchanges/"):].strip("/")
                ex = tap.store.get(ex_id) if ex_id else None
                if ex is None:
                    _error(respond, 404, f"exchange not found: {ex_id}")
                    return True
                d = ex.to_dict()
                d["preview"] = request_preview(ex.request_body)
                _json(respond, d)
                return True
            if sub == "/api/timeseries":
                _json(respond, _timeseries(tap, qs))
                return True
            if sub == "/api/events":
                _events(tap, respond)
                return True
        elif method == "POST":
            if sub == "/api/clear":
                if not _same_origin(headers):
                    _error(respond, 403, "cross-origin request rejected")
                    return True
                tap.store.clear()
                _json(respond, {"ok": True})
                return True
        _error(respond, 404, f"no route for {method} {route}")
        return True
    except Exception as e:  # never let the dashboard take down the proxy thread
        try:
            _error(respond, 500, f"dashboard error: {type(e).__name__}: {e}")
        except Exception:
            pass
        return True
