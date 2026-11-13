"""The core: an HTTP proxy that records, prices, caches, replays, guards and budgets LLM calls.

``LLMTap.handle`` implements the whole per-request pipeline against an abstract ``Responder`` so
the dashboard and tests don't care about ``http.server``. ``_Handler`` adapts
``BaseHTTPRequestHandler`` (HTTP/1.1, keep-alive, Content-Length / chunked framing) to it.
"""
from __future__ import annotations

import gzip
import http.client
import json
import queue
import socket
import socketserver
import ssl
import threading
import time
import zlib
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol
from urllib.parse import urlsplit

from . import __version__
from .budget import Budget
from .cache import ResponseCache, cache_key
from .cassette import Cassette
from .config import Config
from .guard import apply_guard
from .models import CachedResponse, Exchange, Route, Usage
from .pricing import cost_usd
from .providers import (
    error_body,
    estimate_usage,
    extract_text,
    extract_usage,
    forward_headers,
    is_stream_request,
    parse_json,
    parse_sse,
    request_model,
    resolve_route,
    response_model,
    tags_from_headers,
)
from .store import Store

__all__ = ["LLMTap", "EventBus", "Responder", "summarize"]

# Response headers never relayed to the client (we do our own framing).
_RESP_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "content-length",
})
# Headers never stored for replay (body is stored decoded, framing is recomputed).
_STORE_DROP = _RESP_HOP | {"content-encoding", "date", "server", "x-llmtap-id", "x-llmtap-source",
                           "access-control-allow-origin", "access-control-expose-headers"}
_CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
_CORS_BASE = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Expose-Headers": "x-llmtap-id, x-llmtap-source, request-id, x-request-id",
}
_INTERNAL = "/_llmtap"


class Responder(Protocol):
    """Minimal response sink used by the pipeline and the dashboard."""

    def start(self, status: int, headers: dict[str, str]) -> None: ...
    def write(self, chunk: bytes) -> None: ...   # chunk frame when started with transfer-encoding: chunked
    def end(self) -> None: ...


class ClientGone(Exception):
    """The downstream client disconnected while we were writing to it."""


# ------------------------------------------------------------------------------------------------
# event bus
# ------------------------------------------------------------------------------------------------
class EventBus:
    """Thread-safe pub/sub fan-out; each subscriber gets its own bounded Queue (slow ones drop)."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._lock = threading.Lock()
        self._subs: list[queue.Queue] = []
        self._maxsize = maxsize

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)


def summarize(ex: Exchange) -> dict[str, Any]:
    """Exchange dict minus bodies (dashboard's richer summary when available)."""
    try:
        from .dashboard import summary  # lazy: dashboard is optional
        return summary(ex)
    except Exception:
        d = ex.to_dict()
        d.pop("request_body", None)
        d.pop("response_body", None)
        return d


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
class _Tracked:
    """Wraps a Responder: remembers whether it started, injects headers, maps disconnects."""

    def __init__(self, inner: Responder, extra: dict[str, str] | None = None) -> None:
        self.inner = inner
        self.extra = extra or {}
        self.started = False
        self.ended = False

    def start(self, status: int, headers: dict[str, str]) -> None:
        h = dict(headers)
        low = {k.lower() for k in h}
        for k, v in self.extra.items():
            if k.lower() not in low:
                h[k] = v
        self.started = True
        try:
            self.inner.start(status, h)
        except _CLIENT_GONE as e:
            raise ClientGone(str(e)) from e

    def write(self, chunk: bytes) -> None:
        try:
            self.inner.write(chunk)
        except _CLIENT_GONE as e:
            raise ClientGone(str(e)) from e

    def end(self) -> None:
        if self.ended:
            return
        self.ended = True
        try:
            self.inner.end()
        except _CLIENT_GONE as e:
            raise ClientGone(str(e)) from e

    def abort(self) -> None:
        """Drop the connection without completing the body (so the client sees truncation)."""
        self.ended = True
        fn = getattr(self.inner, "abort", None)
        if fn is not None:
            try:
                fn()
            except Exception:
                pass


def _lower(headers: Mapping[str, str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for k, v in (headers or {}).items():
            out[str(k).lower()] = str(v)
    except Exception:
        pass
    return out


def _decode_body(raw: bytes, encoding: str) -> bytes:
    """Undo content-encoding for parsing; returns raw on failure / unknown encodings."""
    enc = (encoding or "").strip().lower()
    if not raw or enc in ("", "identity"):
        return raw
    try:
        if enc in ("gzip", "x-gzip"):
            return gzip.decompress(raw)
        if enc == "deflate":
            try:
                return zlib.decompress(raw)
            except zlib.error:
                return zlib.decompress(raw, -zlib.MAX_WBITS)
    except Exception:
        return raw
    return raw


def _storable_headers(headers: list[tuple[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in headers:
        lk = k.lower()
        if lk in _STORE_DROP or lk.startswith("x-llmtap-") or lk == "set-cookie":
            continue
        out[lk] = v
    return out


def _split_sse(body: str) -> list[str]:
    """Split a stored SSE transcript into whole events (each ending with a blank line)."""
    text = body.replace("\r\n", "\n")
    parts = [p for p in text.split("\n\n") if p.strip()]
    return [p + "\n\n" for p in parts]


def _upstream_target(route: Route) -> tuple[str, str, int | None, str]:
    """(scheme, host, port, full_path) for a route, honoring base paths like https://x/api."""
    u = urlsplit(route.upstream_base)
    scheme = (u.scheme or "http").lower()
    base_path = (u.path or "").rstrip("/")
    up = route.upstream_path or "/"
    if not up.startswith("/"):
        up = "/" + up
    return scheme, u.hostname or "localhost", u.port, base_path + up


def _is_streaming_response(ctype: str, req_stream: bool, has_length: bool) -> bool:
    c = ctype.lower()
    if "text/event-stream" in c or "ndjson" in c or "jsonl" in c:
        return True
    return req_stream and not has_length and "json" not in c


# ------------------------------------------------------------------------------------------------
# the proxy
# ------------------------------------------------------------------------------------------------
class LLMTap:
    """The llmtap proxy: shared state + per-request pipeline + server lifecycle."""

    def __init__(self, cfg: Config, store: Store | None = None) -> None:
        self.cfg = cfg
        self.store: Store = store if store is not None else Store(cfg.resolved_db_path())
        self.cache: ResponseCache | None = (
            ResponseCache(cfg.resolved_db_path(), cfg.cache_ttl_s) if cfg.cache else None
        )
        self.cassette: Cassette | None = Cassette(cfg.cassette, cfg.mode) if cfg.cassette else None
        self.budget: Budget = Budget(cfg, self.store.spent_since)
        self.bus: EventBus = EventBus()
        self.port: int = cfg.port
        self.started_at: float = time.time()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None
        self._serving = threading.Event()
        self._closed = False

    # ---------------------------------------------------------------- lifecycle
    @property
    def base_url(self) -> str:
        host = self.cfg.host
        if host in ("", "0.0.0.0", "::"):
            host = "127.0.0.1"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    def _bind(self) -> "_Server":
        if self._server is None:
            self._server = _Server((self.cfg.host, self.cfg.port), _Handler, self)
            self.port = int(self._server.server_address[1])
        return self._server

    def serve_forever(self) -> None:
        """Serve in the calling thread until shutdown() (or KeyboardInterrupt propagates)."""
        srv = self._bind()
        self._serving.set()
        try:
            srv.serve_forever(poll_interval=0.25)
        finally:
            self._serving.clear()

    def start_background(self) -> "LLMTap":
        """Bind now (so .port/.base_url are valid on return) and serve in a daemon thread."""
        self._bind()
        self._thread = threading.Thread(target=self.serve_forever, name="llmtap-proxy", daemon=True)
        self._thread.start()
        self._serving.wait(5)
        return self

    def shutdown(self) -> None:
        """Stop serving and close the server socket, store and cache. Idempotent."""
        if self._closed:
            return
        self._closed = True
        srv = self._server
        if srv is not None:
            if self._serving.is_set():
                try:
                    srv.shutdown()
                except Exception:
                    pass
            try:
                srv.server_close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        for obj in (self.cache, self.store):
            close = getattr(obj, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass

    def __enter__(self) -> "LLMTap":
        return self.start_background()

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    # ---------------------------------------------------------------- pipeline
    def handle(self, method: str, path: str, headers: Mapping[str, str], body: bytes,
               respond: Responder) -> Exchange:
        """Process one request end-to-end and return the (recorded) Exchange."""
        method = (method or "GET").upper()
        path = path or "/"
        hdrs = _lower(headers)
        body = body or b""
        bare = path.split("?", 1)[0]

        # -- internal routes (not recorded) -------------------------------------------------
        if bare == _INTERNAL or bare.startswith(_INTERNAL + "/"):
            ex = Exchange(method=method, path=path, provider="llmtap", source="internal")
            self._internal(method, path, headers, body, respond, ex)
            return ex
        if bare == "/" and method in ("GET", "HEAD"):
            ex = Exchange(method=method, path=path, provider="llmtap", source="internal", status=302)
            self._simple(respond, 302, b"", {"Location": _INTERNAL + "/"})
            return ex
        if bare == "/favicon.ico":
            ex = Exchange(method=method, path=path, provider="llmtap", source="internal", status=204)
            self._simple(respond, 204, b"", {})
            return ex
        if method == "OPTIONS" and "access-control-request-method" in hdrs:
            ex = Exchange(method=method, path=path, provider="llmtap", source="internal", status=204)
            self._simple(respond, 204, b"", {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": hdrs.get("access-control-request-headers") or "*",
                "Access-Control-Max-Age": "86400",
            })
            return ex

        # -- proxied LLM call ---------------------------------------------------------------
        ex = Exchange(method=method, path=path)
        t0 = time.perf_counter()
        out = _Tracked(respond, dict(_CORS_BASE))
        style = "openai"
        try:
            route = resolve_route(path, hdrs, self.cfg)
            style = route.style
            ex.provider, ex.style = route.provider, route.style
            self._pipeline(ex, route, method, hdrs, body, out, t0)
        except ClientGone:
            ex.error = ex.error or "client disconnected"
        except Exception as e:  # never let a bug kill the connection silently
            ex.error = f"llmtap internal error: {type(e).__name__}: {e}"
            if ex.source == "upstream":
                ex.source = "error"
            if not out.started:
                ex.status = 500
                self._error(out, ex, style, 500, ex.error, "llmtap_internal_error")
            else:
                out.abort()
        ex.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self._finish(ex)
        return ex

    def _pipeline(self, ex: Exchange, route: Route, method: str, hdrs: dict[str, str], body: bytes,
                  out: _Tracked, t0: float) -> None:
        cfg = self.cfg
        req = parse_json(body)
        ex.model = request_model(req)
        stream_req = is_stream_request(req)
        ex.tags = tags_from_headers(hdrs)

        # guard
        send_body = body
        if body and (cfg.guard or "off") != "off":
            send_body, findings, blocked = apply_guard(body, cfg.guard, cfg.guard_pii)
            ex.guard_findings = [f.to_dict() if hasattr(f, "to_dict") else dict(f) for f in findings]
            if blocked:
                ex.request_body = body.decode("utf-8", "replace")
                kinds = ", ".join(sorted({str(f.get("kind", "?")) for f in ex.guard_findings}))
                ex.source = "blocked"
                msg = (f"llmtap guard blocked this request: detected {kinds} in the prompt. "
                       "Remove the secret, or run llmtap with --guard redact / warn.")
                self._error(out, ex, route.style, 400, msg, "llmtap_guard_blocked")
                return
            if send_body is not body:
                req = parse_json(send_body) if req is not None else req
        ex.request_body = send_body.decode("utf-8", "replace")

        cacheable = method == "POST"
        key = cache_key(route.provider, route.upstream_path, send_body) if cacheable else ""

        # budget / loop breaker
        if cacheable:
            verdict = self.budget.check(key)
            if not verdict.allowed:
                ex.source = verdict.kind or "budget"
                err_type = "llmtap_loop_detected" if ex.source == "loop" else "llmtap_budget_exceeded"
                self._error(out, ex, route.style, 429, verdict.reason or "llmtap budget exceeded", err_type)
                return

        # cassette / cache
        if cacheable:
            stored: CachedResponse | None = None
            if self.cassette is not None:
                stored = self.cassette.lookup(key)
                if stored is not None:
                    ex.source = "cassette"
            if stored is None and self.cache is not None:
                stored = self.cache.get(key)
                if stored is not None:
                    ex.source = "cache"
            if stored is not None:
                self._replay(ex, route, req, stored, out, t0)
                return
            if self.cassette is not None and self.cassette.mode == "replay" and not cfg.allow_miss:
                ex.source = "error"
                msg = ("llmtap cassette miss (replay mode): no recording for this request. "
                       "Re-record with --mode record")
                self._error(out, ex, route.style, 409, msg, "llmtap_cassette_miss")
                return

        self._forward(ex, route, method, hdrs, send_body, req, stream_req, key, out, t0)

    # ---------------------------------------------------------------- upstream
    def _forward(self, ex: Exchange, route: Route, method: str, hdrs: dict[str, str], body: bytes,
                 req: dict | None, stream_req: bool, key: str, out: _Tracked, t0: float) -> None:
        scheme, host, port, full_path = _upstream_target(route)
        timeout = float(self.cfg.upstream_timeout_s or 600.0)
        conn: http.client.HTTPConnection
        if scheme == "https":
            conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        fwd = forward_headers(hdrs)
        try:
            try:
                conn.request(method, full_path, body=body if (body or method in ("POST", "PUT", "PATCH")) else None,
                             headers=fwd)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                ex.source = "error"
                msg = f"llmtap: upstream {route.upstream_base} unreachable: {type(e).__name__}: {e}"
                status = 504 if isinstance(e, (socket.timeout, TimeoutError)) else 502
                self._error(out, ex, route.style, status, msg, "llmtap_upstream_error")
                return

            ex.status = resp.status
            up_headers = resp.getheaders()
            low = {k.lower(): v for k, v in up_headers}
            ctype = low.get("content-type", "")
            encoding = low.get("content-encoding", "")
            relay = [(k, v) for k, v in up_headers if k.lower() not in _RESP_HOP]
            client_h = _merge_headers(relay, {"x-llmtap-id": ex.id, "x-llmtap-source": "upstream"})
            streaming = _is_streaming_response(ctype, stream_req, "content-length" in low)
            ex.streamed = streaming
            raw = bytearray()
            client_gone = False
            upstream_err: str | None = None

            if streaming:
                client_h["Transfer-Encoding"] = "chunked"
                out.start(resp.status, client_h)
                while True:
                    try:
                        chunk = resp.read1(65536)
                    except (OSError, http.client.HTTPException) as e:
                        upstream_err = f"upstream read error: {type(e).__name__}: {e}"
                        break
                    if not chunk:
                        break
                    if ex.ttft_ms is None:
                        ex.ttft_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                    raw += chunk
                    try:
                        out.write(chunk)
                    except ClientGone:
                        client_gone = True
                        break
                if client_gone or upstream_err:
                    out.abort()
                else:
                    try:
                        out.end()
                    except ClientGone:
                        client_gone = True
            else:
                try:
                    data = resp.read()
                except (OSError, http.client.HTTPException) as e:
                    ex.source = "error"
                    msg = f"llmtap: upstream read failed: {type(e).__name__}: {e}"
                    self._error(out, ex, route.style, 502, msg, "llmtap_upstream_error")
                    return
                raw += data
                ex.ttft_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                client_h["Content-Length"] = str(len(raw))
                try:
                    out.start(resp.status, client_h)
                    if raw and method != "HEAD":
                        out.write(bytes(raw))
                    out.end()
                except ClientGone:
                    client_gone = True
                    out.abort()
        finally:
            try:
                conn.close()
            except Exception:
                pass

        text = _decode_body(bytes(raw), encoding).decode("utf-8", "replace")
        ex.response_body = text
        if client_gone:
            ex.error = "client disconnected"
        elif upstream_err:
            ex.error = upstream_err
        if resp.status >= 400 and not ex.error:
            ex.error = _upstream_error_message(text) or f"upstream HTTP {resp.status}"

        self._account(ex, route, req, text, streaming)
        ex.cost_usd = self._price(ex) if ex.usage.total() else 0.0
        try:
            self.budget.add_spend(ex.cost_usd)
        except Exception:
            pass

        if 200 <= ex.status < 300 and not client_gone and not upstream_err and key:
            cached = CachedResponse(status=ex.status, headers=_storable_headers(up_headers), body=text,
                                    streamed=streaming, cost_usd=ex.cost_usd)
            if self.cache is not None:
                try:
                    self.cache.put(key, cached)
                except Exception as e:
                    ex.error = ex.error or f"cache write failed: {e}"
            if self.cassette is not None and self.cassette.should_record():
                try:
                    self.cassette.record(key, req if req is not None else ex.request_body, cached)
                except Exception as e:
                    ex.error = ex.error or f"cassette write failed: {e}"

    # ---------------------------------------------------------------- replay
    def _replay(self, ex: Exchange, route: Route, req: dict | None, stored: CachedResponse,
                out: _Tracked, t0: float) -> None:
        ex.status = int(stored.status)
        ex.streamed = bool(stored.streamed)
        ex.response_body = stored.body
        ex.saved_usd = float(stored.cost_usd or 0.0)
        ex.cost_usd = 0.0
        h: dict[str, str] = {}
        for k, v in (stored.headers or {}).items():
            lk = str(k).lower()
            if lk in _STORE_DROP or lk.startswith("x-llmtap-") or lk == "host":
                continue
            h[lk] = str(v)
        h["x-llmtap-id"] = ex.id
        h["x-llmtap-source"] = ex.source
        data = stored.body.encode("utf-8")
        try:
            if ex.streamed:
                h["content-type"] = h.get("content-type") or "text/event-stream"
                if "event-stream" not in h["content-type"] and "ndjson" not in h["content-type"]:
                    h["content-type"] = "text/event-stream"
                h["cache-control"] = "no-cache"
                h["Transfer-Encoding"] = "chunked"
                out.start(ex.status, h)
                events = _split_sse(stored.body) if "event-stream" in h["content-type"] else (
                    [ln + "\n" for ln in stored.body.splitlines() if ln.strip()])
                for i, evt in enumerate(events):
                    out.write(evt.encode("utf-8"))
                    if i == 0:
                        ex.ttft_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                out.end()
            else:
                h.setdefault("content-type", "application/json")
                h["Content-Length"] = str(len(data))
                out.start(ex.status, h)
                if data:
                    out.write(data)
                out.end()
                ex.ttft_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        except ClientGone:
            ex.error = "client disconnected"
            out.abort()
        self._account(ex, route, req, stored.body, ex.streamed)

    # ---------------------------------------------------------------- accounting
    def _account(self, ex: Exchange, route: Route, req: dict | None, text: str, streamed: bool) -> None:
        """Fill usage / output_text / model from a response body (never raises)."""
        usage, out_text, model = Usage(), "", None
        try:
            if streamed:
                usage, out_text, model = parse_sse(route.style, text)
            else:
                j = parse_json(text.encode("utf-8"))
                if j is not None:
                    usage = extract_usage(route.style, j)
                    out_text = extract_text(route.style, j)
                    model = response_model(route.style, j)
        except Exception:
            pass
        if 200 <= ex.status < 300 and usage.input_tokens == 0 and usage.output_tokens == 0:
            try:
                usage = estimate_usage(req, out_text)
            except Exception:
                pass
        ex.usage = usage
        ex.output_text = out_text or ""
        if model:
            ex.model = model

    def _price(self, ex: Exchange) -> float:
        try:
            return float(cost_usd(ex.model, ex.usage, self.cfg.pricing_overrides) or 0.0)
        except Exception:
            return 0.0

    # ---------------------------------------------------------------- responses
    def _simple(self, respond: Responder, status: int, data: bytes, headers: dict[str, str]) -> None:
        h = dict(headers)
        h["Content-Length"] = str(len(data))
        try:
            respond.start(status, h)
            if data:
                respond.write(data)
            respond.end()
        except _CLIENT_GONE:
            pass

    def _error(self, out: _Tracked, ex: Exchange, style: str, status: int, message: str, err_type: str) -> None:
        """Send a provider-shaped JSON error and fill the exchange accordingly."""
        data = error_body(style, status, message, err_type)
        ex.status = status
        ex.error = ex.error or message
        ex.response_body = data.decode("utf-8", "replace")
        if out.started:
            return
        try:
            hdrs = {"Content-Type": "application/json", "Content-Length": str(len(data)),
                    "x-llmtap-id": ex.id, "x-llmtap-source": ex.source}
            if status in (400, 409, 429):
                # deliberate llmtap blocks (guard/cassette miss/budget/loop): tell official SDKs not to retry
                hdrs["x-should-retry"] = "false"
            out.start(status, hdrs)
            out.write(data)
            out.end()
        except ClientGone:
            pass

    def _internal(self, method: str, path: str, headers: Mapping[str, str], body: bytes,
                  respond: Responder, ex: Exchange) -> None:
        out = _Tracked(respond)
        try:
            from .dashboard import handle_internal
        except Exception:
            handle_internal = None  # type: ignore[assignment]
        try:
            handled = bool(handle_internal(self, method, path, headers, body, out)) if handle_internal else False
        except ClientGone:
            return
        except Exception as e:
            ex.error = f"dashboard error: {type(e).__name__}: {e}"
            if not out.started:
                data = json.dumps({"error": ex.error}).encode()
                ex.status = 500
                self._simple(respond, 500, data, {"Content-Type": "application/json"})
            return
        if not handled and not out.started:
            ex.status = 404
            data = json.dumps({"error": f"not found: {path}", "version": __version__}).encode()
            self._simple(respond, 404, data, {"Content-Type": "application/json"})

    # ---------------------------------------------------------------- finish
    def _finish(self, ex: Exchange) -> None:
        """Persist, publish and log; each step isolated so none can break the request."""
        try:
            self.store.record(ex)
        except Exception as e:
            ex.error = ex.error or f"store write failed: {e}"
        try:
            self.bus.publish({"type": "exchange", "exchange": summarize(ex)})
        except Exception:
            pass
        if not self.cfg.quiet:
            try:
                from .term import log_exchange
                log_exchange(ex)
            except Exception:
                pass


def _merge_headers(relay: list[tuple[str, str]], extra: dict[str, str]) -> dict[str, str]:
    """Relay headers as a dict (duplicate names joined with ', ', except set-cookie last-wins)."""
    out: dict[str, str] = {}
    lower_map: dict[str, str] = {}
    for k, v in relay:
        lk = k.lower()
        if lk in lower_map and lk != "set-cookie":
            out[lower_map[lk]] = out[lower_map[lk]] + ", " + v
        else:
            lower_map[lk] = k
            out[k] = v
    for k, v in extra.items():
        out[k] = v
    return out


def _upstream_error_message(text: str) -> str | None:
    j = parse_json(text.encode("utf-8")) if text else None
    if not j:
        return None
    err = j.get("error")
    if isinstance(err, dict) and isinstance(err.get("message"), str):
        return err["message"][:500]
    if isinstance(err, str):
        return err[:500]
    if isinstance(j.get("message"), str):
        return j["message"][:500]
    return None


# ------------------------------------------------------------------------------------------------
# http.server adapter
# ------------------------------------------------------------------------------------------------
class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    block_on_close = False
    request_queue_size = 128

    def __init__(self, addr: tuple[str, int], handler: type, tap: LLMTap) -> None:
        self.tap = tap
        if ":" in (addr[0] or ""):
            self.address_family = socket.AF_INET6
        super().__init__(addr, handler)

    def server_bind(self) -> None:
        # Skip HTTPServer.server_bind's socket.getfqdn(): it can stall for seconds on some hosts.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)

    def handle_error(self, request: Any, client_address: Any) -> None:  # quiet on client resets
        pass


class _HandlerResponder:
    """Responder over a BaseHTTPRequestHandler with Content-Length or chunked framing."""

    def __init__(self, handler: "_Handler") -> None:
        self.h = handler
        self.started = False
        self.ended = False
        self.chunked = False
        self.no_body = False

    def start(self, status: int, headers: dict[str, str]) -> None:
        if self.started:
            raise RuntimeError("response already started")
        h = self.h
        self.started = True
        self.no_body = h.command == "HEAD" or status in (204, 304) or 100 <= status < 200
        want_chunked = False
        has_length = False
        h.send_response(status)
        for k, v in headers.items():
            lk = k.lower()
            if lk == "transfer-encoding":
                want_chunked = "chunked" in str(v).lower()
                continue
            if lk == "content-length":
                has_length = True
            if lk == "connection":
                continue
            h.send_header(k, str(v))
        if self.no_body:
            if not has_length and status not in (204, 304) and not (100 <= status < 200):
                h.send_header("Content-Length", "0")
        elif want_chunked or not has_length:
            if h.request_version == "HTTP/1.0":
                h.close_connection = True
            else:
                self.chunked = True
                h.send_header("Transfer-Encoding", "chunked")
        if h.close_connection:
            h.send_header("Connection", "close")
        h.end_headers()
        h.wfile.flush()

    def write(self, chunk: bytes) -> None:
        if not self.started:
            raise RuntimeError("write() before start()")
        if not chunk or self.no_body:
            return
        w = self.h.wfile
        if self.chunked:
            w.write(b"%x\r\n" % len(chunk) + bytes(chunk) + b"\r\n")
        else:
            w.write(chunk)
        w.flush()

    def end(self) -> None:
        if self.ended:
            return
        self.ended = True
        if self.chunked and not self.no_body:
            self.h.wfile.write(b"0\r\n\r\n")
            self.h.wfile.flush()

    def abort(self) -> None:
        """Mark done without the terminating frame and close the connection afterwards."""
        self.ended = True
        self.h.close_connection = True


class _BadRequest(Exception):
    pass


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llmtap/" + __version__
    sys_version = ""
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - silence default logging
        pass

    def _read_body(self) -> bytes:
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            buf = bytearray()
            while True:
                line = self.rfile.readline(65537)
                if not line:
                    raise _BadRequest("truncated chunked body")
                size_s = line.split(b";", 1)[0].strip()
                try:
                    size = int(size_s, 16)
                except ValueError:
                    raise _BadRequest("bad chunk size") from None
                if size == 0:
                    while True:  # trailers
                        t = self.rfile.readline(65537)
                        if t in (b"\r\n", b"\n", b""):
                            break
                    return bytes(buf)
                data = self.rfile.read(size)
                if len(data) < size:
                    raise _BadRequest("truncated chunk")
                buf += data
                self.rfile.readline(3)  # CRLF after chunk
        cl = self.headers.get("Content-Length")
        if cl:
            try:
                n = int(cl)
            except ValueError:
                raise _BadRequest("bad Content-Length") from None
            if n < 0:
                raise _BadRequest("bad Content-Length")
            data = self.rfile.read(n)
            if len(data) < n:
                raise _BadRequest("truncated body")
            return data
        return b""

    def _dispatch(self) -> None:
        r = _HandlerResponder(self)
        try:
            try:
                body = self._read_body()
            except _BadRequest as e:
                self.close_connection = True
                msg = json.dumps({"error": {"message": f"llmtap: bad request: {e}", "type": "llmtap_bad_request"}})
                data = msg.encode()
                r.start(400, {"Content-Type": "application/json", "Content-Length": str(len(data))})
                r.write(data)
                r.end()
                return
            headers = {k: v for k, v in self.headers.items()}
            self.server.tap.handle(self.command, self.path, headers, body, r)
            if not r.started:
                data = b'{"error":{"message":"llmtap: no response produced","type":"llmtap_error"}}'
                r.start(500, {"Content-Type": "application/json", "Content-Length": str(len(data))})
                r.write(data)
                r.end()
            elif not r.ended:
                r.end()
        except _CLIENT_GONE:
            self.close_connection = True
        except Exception:
            self.close_connection = True
            if not r.started:
                try:
                    data = b'{"error":{"message":"llmtap internal error","type":"llmtap_error"}}'
                    r.start(500, {"Content-Type": "application/json", "Content-Length": str(len(data))})
                    r.write(data)
                    r.end()
                except Exception:
                    pass

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = _dispatch
