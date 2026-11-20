"""The core: an HTTP proxy that records, prices, caches, replays, guards and budgets LLM calls.

``LLMTap.handle`` implements the whole per-request pipeline against an abstract ``Responder`` so
the dashboard and tests don't care about ``http.server``. ``_Handler`` adapts
``BaseHTTPRequestHandler`` (HTTP/1.1, keep-alive, Content-Length / chunked framing) to it.
"""
from __future__ import annotations

import dataclasses
import gzip
import http.client
import ipaddress
import json
import queue
import re
import socket
import socketserver
import ssl
import sys
import threading
import time
import zlib
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol
from urllib.parse import parse_qsl, quote, urlsplit

from . import __version__
from .budget import Budget
from .cache import ResponseCache, cache_key
from .cassette import Cassette
from .config import Config
from .guard import apply_guard
from .models import CachedResponse, Exchange, Route, Usage
from .pricing import cost_usd
from .vault import (TAG as VAULT_TAG, Refusal, base_allows, body_models, caller_refusal, endpoint_refusal,
                    find_placeholder, request_shape_refusal, substitute, _local_midnight)
from .providers import (
    EVENTSTREAM_CTYPE,
    bedrock_header_usage,
    bedrock_route_problem,
    body_error,
    error_body,
    eventstream_to_text,
    is_bedrock_call,
    is_sigv4,
    parse_bedrock_stream,
    text_to_eventstream,
    estimate_usage,
    extract_text,
    extract_usage,
    forward_headers,
    is_stream_request,
    parse_json,
    parse_sse,
    model_from_path,
    request_model,
    resolve_route,
    response_model,
    sse_error,
    tags_from_headers,
)
from .store import Store

__all__ = ["LLMTap", "EventBus", "Responder", "summarize"]

# Response headers never relayed to the client (we do our own framing).
_RESP_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "content-length",
    "date", "server",  # send_response() adds its own; relaying the upstream's duplicated them
})
# Headers never stored for replay (body is stored decoded, framing is recomputed).
_STORE_DROP = _RESP_HOP | {"content-encoding", "date", "server", "x-llmtap-id", "x-llmtap-source",
                           "access-control-allow-origin", "access-control-expose-headers"}
_CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)
# Per-exchange cap (characters) for what is *stored* (db / cache / cassette). Relayed bytes are never capped.
MAX_STORED_CHARS = 2 * 1024 * 1024
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

    def close(self) -> None:
        """Send every subscriber the ``None`` sentinel (ends live dashboard streams), even when its queue is full."""
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            for _ in range(2):
                try:
                    q.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
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


def _try_decode(raw: bytes, enc: str) -> bytes | None:
    """Undo content-encoding; None when we can't (br, zstd, compress, corrupt gzip...)."""
    try:
        if enc in ("gzip", "x-gzip"):
            return gzip.decompress(raw)
        if enc == "deflate":
            try:
                return zlib.decompress(raw)
            except zlib.error:
                return zlib.decompress(raw, -zlib.MAX_WBITS)
    except Exception:
        return None
    return None


_BINARY_PREFIXES = ("audio/", "image/", "video/", "font/")
_BINARY_TYPES = frozenset({"application/octet-stream", "application/zip", "application/pdf", "application/gzip",
                           "application/x-tar", "application/x-zip-compressed", "application/x-protobuf"})


def _decode_response(raw: bytes, encoding: str, ctype: str) -> tuple[str, bool]:
    """(text, parseable). Binary / undecodable bodies yield a short placeholder and parseable=False
    so we never dump megabytes of mojibake into the db or cache (the client still gets the raw bytes)."""
    if not raw:
        return "", True
    enc = (encoding or "").strip().lower()
    data = raw
    if enc not in ("", "identity"):
        dec = _try_decode(raw, enc)
        if dec is None:
            return (f"[response body not stored: content-encoding '{enc}' is not decodable by llmtap, "
                    f"{len(raw)} bytes]"), False
        data = dec
    main = (ctype or "").split(";")[0].strip().lower()
    if main == EVENTSTREAM_CTYPE:                 # Bedrock streams: binary frames around JSON payloads
        return eventstream_to_text(data), True
    if main.startswith(_BINARY_PREFIXES) or main in _BINARY_TYPES:
        return f"[binary response body not stored: {main}, {len(data)} bytes]", False
    try:
        return data.decode("utf-8"), True
    except UnicodeDecodeError:
        if any(t in main for t in ("json", "text", "event-stream", "xml")):
            return data.decode("utf-8", "replace"), True
        return f"[binary response body not stored: {main or 'unknown content-type'}, {len(data)} bytes]", False


def _decode_request(body: bytes, ctype: str) -> tuple[str, bool]:
    """(text to store, truncated). Multipart / binary request bodies are stored as a short placeholder."""
    main = (ctype or "").split(";")[0].strip().lower()
    if main.startswith("multipart/") or main in _BINARY_TYPES or main.startswith(_BINARY_PREFIXES):
        return f"[{main} request body not stored: {len(body)} bytes]", False
    try:
        text = body[:MAX_STORED_CHARS * 4].decode("utf-8")
    except UnicodeDecodeError as e:
        if e.start > MAX_STORED_CHARS * 4 - 8 or len(body) > MAX_STORED_CHARS * 4:
            text = body[:MAX_STORED_CHARS * 4].decode("utf-8", "ignore")
        else:
            return f"[binary request body not stored: {main or 'unknown content-type'}, {len(body)} bytes]", False
    if len(text) > MAX_STORED_CHARS or len(body) > MAX_STORED_CHARS * 4:
        return text[:MAX_STORED_CHARS] + f"\n[llmtap: truncated, {len(body)} bytes in total]", True
    return text, False


_SENSITIVE_QS = frozenset({"key", "api_key", "apikey", "api-key", "token", "access_token", "x-api-key",
                           "signature", "sig", "auth", "password", "secret", "client_secret"})
_SENSITIVE_QS_RE = re.compile(r"([?&](?:" + "|".join(sorted(re.escape(k) for k in _SENSITIVE_QS)) + r")=)[^&#]*",
                              re.IGNORECASE)
_CRED_HEADERS = ("authorization", "proxy-authorization", "x-api-key", "api-key", "x-goog-api-key", "cookie",
                 "x-auth-token", "openai-api-key", "anthropic-auth-token", "x-amz-security-token")
REDACTED = "[REDACTED]"


def _mask_path(path: str) -> str:
    """Mask secret-looking query parameters (``?key=AIza...``) before the path is ever stored."""
    return _SENSITIVE_QS_RE.sub(lambda m: m.group(1) + REDACTED, path) if "?" in path else path


def _credentials(hdrs: Mapping[str, str], path: str) -> list[str]:
    """Credential strings of this request (>= 8 chars), longest first, so they can be scrubbed from anything stored."""
    vals: set[str] = set()
    for name in _CRED_HEADERS:
        v = (hdrs.get(name) or "").strip()
        if not v:
            continue
        vals.add(v)
        if name == "cookie":
            for part in v.split(";"):
                vals.add(part.strip().partition("=")[2].strip())
        else:
            vals.add(v.split(None, 1)[-1])      # "Bearer abc" -> "abc"
    for k, v in parse_qsl(path.partition("?")[2], keep_blank_values=True):
        if k.lower() in _SENSITIVE_QS:
            vals.add(v)
    return sorted({v for v in vals if len(v) >= 8}, key=len, reverse=True)


def _scrub(text: str, secrets: list[str]) -> str:
    for sec in secrets:
        if sec in text:
            text = text.replace(sec, REDACTED)
    return text


def _scrub_bytes(data: bytes, secrets: list[bytes]) -> bytes:
    for sec in secrets:
        if sec and sec in data:
            data = data.replace(sec, REDACTED.encode())
    return data


class _StreamScrub:
    """Removes secrets from a byte stream even when one is split across chunks. Only a tail that could be the
    start of a secret is held back (usually nothing), so streaming stays chunk-by-chunk."""

    def __init__(self, secrets: list[bytes]) -> None:
        self.secrets = [s for s in secrets if s]
        self.pending = b""

    def feed(self, chunk: bytes) -> bytes:
        buf = _scrub_bytes(self.pending + chunk, self.secrets)
        hold = 0
        for sec in self.secrets:
            for n in range(min(len(sec) - 1, len(buf)), 0, -1):
                if buf.endswith(sec[:n]):
                    hold = max(hold, n)
                    break
        self.pending = buf[len(buf) - hold:] if hold else b""
        return buf[: len(buf) - hold] if hold else buf

    def flush(self) -> bytes:
        out, self.pending = self.pending, b""
        return out


def _storable_headers(headers: list[tuple[str, str]], secrets: list[str] | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in headers:
        lk = k.lower()
        if lk in _STORE_DROP or lk.startswith("x-llmtap-") or lk == "set-cookie" or lk in _CRED_HEADERS:
            continue
        if secrets and any(sec in v for sec in secrets):     # upstream echoed a credential back
            continue
        out[lk] = v
    return out


def _note(ex: Exchange, text: str) -> None:
    """Attach a non-fatal note (e.g. truncation); folded into ``ex.error`` when the exchange is sealed."""
    ex.__dict__.setdefault("_notes", []).append(text)


def _seal(ex: Exchange) -> None:
    """Final pass before an exchange is persisted: fold notes into ``error``, scrub credentials."""
    notes = ex.__dict__.pop("_notes", None)
    if notes:
        ex.error = "; ".join(x for x in [ex.error, *notes] if x)
    secrets = ex.__dict__.pop("_secrets", None)
    if secrets:
        for f in ("path", "request_body", "response_body", "error", "output_text"):
            v = getattr(ex, f, None)
            if v:
                setattr(ex, f, _scrub(v, secrets))


_LLM_TAIL = re.compile(r"(?:/(?:chat/completions|completions|embeddings)|:(?:stream)?[gG]enerateContent|(?:^|/)v\d+[a-z0-9]*/(?:messages|complete|responses))$")


def _is_llm_call(method: str, upstream_path: str) -> bool:
    """True for billable generation endpoints (chat / messages / responses / embeddings / completions).
    Everything else (models, files, batches, count_tokens, ...) is forwarded untouched: recorded, but
    never cached, budget-checked, loop-checked or priced."""
    if method != "POST":
        return False
    return bool(_LLM_TAIL.search(upstream_path.split("?", 1)[0].rstrip("/"))) or is_bedrock_call(upstream_path)


def _is_json_doc(body: bytes) -> bool:
    """Cheap check that a (non-object) body is a JSON document, e.g. an array."""
    head = body.lstrip()[:1]
    if head not in (b"[", b'"'):
        return False
    try:
        json.loads(body.decode("utf-8-sig"))
        return True
    except Exception:
        return False


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


class _FinalResponse(http.client.HTTPResponse):
    """http.client only skips ``100 Continue``; also skip other interim responses (102, 103 Early Hints...)
    so they are never relayed as if they were the final answer."""

    def _read_status(self) -> tuple[str, int, str]:
        while True:
            version, status, reason = super()._read_status()  # type: ignore[misc]
            if not (101 < status < 200):
                return version, status, reason
            http.client.parse_headers(self.fp)  # type: ignore[arg-type]


def _is_streaming_response(ctype: str, req_stream: bool, has_length: bool) -> bool:
    c = ctype.lower()

    if (
        "text/event-stream" in c
        or "application/vnd.amazon.eventstream" in c
        or "ndjson" in c
        or "jsonl" in c
    ):
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
        self.otel: Any = None
        if cfg.otel_endpoint:
            from .otel import OtelExporter
            self.otel = OtelExporter(cfg.otel_endpoint)
        self._vault: Any = None             # key vault, opened on first use (a placeholder or the Keys API)
        self._vault_lock = threading.Lock()
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

    @property
    def vault(self) -> Any:
        """The key vault (``~/.llmtap/keys.json`` + the OS credential store), opened lazily."""
        if self._vault is None:
            with self._vault_lock:
                if self._vault is None:
                    from .vault import Vault
                    self._vault = Vault()
        return self._vault

    def _vault_lookup(self, name: str) -> tuple[Any, str | None, str | None]:
        """(KeyInfo, secret, problem) for a placeholder's key name. Messages never contain a secret."""
        try:
            v = self.vault
            info = v.get(name)
            if info is None:
                return None, None, (f"llmtap: no stored API key named {name!r}. Add it in the llmtap dashboard "
                                    f"(Keys) or with `llmtap keys add {name}`.")
            secret = v.secret(name)
            if not secret:
                return info, None, (f"llmtap: the stored key {name!r} has no secret in the {v.backend.name} store. "
                                    "Add it again in the dashboard (Keys) or with `llmtap keys add`.")
            return info, secret, None
        except Exception as e:
            return None, None, f"llmtap: the key vault is unavailable: {e}"

    @property
    def is_exposed(self) -> bool:
        """True when the listen address is reachable beyond this machine (0.0.0.0, ::, a LAN address...)."""
        return _host_is_exposed(self.cfg.host)

    def _bind(self) -> "_Server":
        if self._server is None:
            self._server = _Server((self.cfg.host, self.cfg.port), _Handler, self)
            self.port = int(self._server.server_address[1])
            if self.is_exposed:
                self._warn_exposed()
        return self._server

    def _warn_exposed(self) -> None:
        """Prominent stderr warning: prompts, responses and the dashboard are served without authentication."""
        try:
            from .term import color_enabled, paint
            msg = (f"WARNING: llmtap is listening on {self.cfg.host or '0.0.0.0'}:{self.port} and is exposed to "
                   "your network. The dashboard and every recorded prompt/response are readable by anyone who "
                   "can reach this port, and nothing is authenticated. Bind to 127.0.0.1 unless you mean this.")
            sys.stderr.write(paint(msg, "bold", "yellow", color=color_enabled(sys.stderr)) + "\n")
            sys.stderr.flush()
        except Exception:
            pass

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
        try:
            self.bus.close()   # live SSE feeds would otherwise keep their handler threads (and clients) forever
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.otel is not None:
            try:
                self.otel.close()
            except Exception:
                pass
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
               respond: Responder, client: str | None = None) -> Exchange:
        """Process one request end-to-end and return the (recorded) Exchange. ``client`` is the peer address."""
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
        ex = Exchange(method=method, path=_mask_path(path))
        # Key vault: a `llmtap-key-<name>` placeholder is replaced by the stored key before routing, so the
        # request is routed (and forwarded) exactly as if the client had sent the real key itself.
        found = find_placeholder(hdrs, path)
        vault_info: Any = None
        vault_refusal: Refusal | None = None
        vault_secrets: list[str] = []
        if found is not None:
            from .dashboard import _host_allowed
            vault_refusal = caller_refusal(hdrs, client, _host_allowed(self, hdrs))
            if vault_refusal is None and any(ord(c) < 33 or ord(c) == 127 for c in path):
                vault_refusal = Refusal(400, "llmtap: the request URL contains spaces or control characters.",
                                        "llmtap_bad_request")
            if vault_refusal is None:
                vault_refusal = request_shape_refusal(method, path, hdrs, body)
            if vault_refusal is None:
                vault_info, secret, problem = self._vault_lookup(found.name)
                if problem:
                    vault_refusal = Refusal(401, problem, "llmtap_key_not_found")
                elif secret:
                    hdrs, path = substitute(hdrs, path, found, secret)
                    ex.__dict__["_vault"] = vault_info
                    vault_secrets = sorted({secret, quote(secret, safe="")}, key=len, reverse=True)
                    ex.__dict__["_vault_secrets"] = vault_secrets
        ex.__dict__["_secrets"] = sorted(set(_credentials(hdrs, path)) | set(vault_secrets), key=len, reverse=True)
        t0 = time.perf_counter()
        # no `Access-Control-Allow-Origin: *` on anything involving a stored key: a web page must not read it
        out = _Tracked(respond, {} if found is not None else dict(_CORS_BASE))
        style = "openai"
        try:
            route = resolve_route(path, hdrs, self.cfg)
            style = route.style
            ex.provider, ex.style = route.provider, route.style
            problem = bedrock_route_problem(route, path, hdrs)
            if vault_refusal is not None:
                ex.source = "blocked"
                self._error(out, ex, route.style, vault_refusal.status, vault_refusal.message, vault_refusal.err_type)
            elif vault_info is not None and vault_info.provider != route.provider:
                ex.source = "blocked"
                self._error(out, ex, route.style, 403,
                            f"llmtap: the stored key {vault_info.name!r} is for {vault_info.provider}, but this "
                            f"request goes to {route.provider}. Stored keys are only ever sent to their own "
                            "provider.", "llmtap_key_wrong_provider")
            elif vault_info is not None and not base_allows(vault_info, route.upstream_base):
                ex.source = "blocked"
                self._error(out, ex, route.style, 403,
                            f"llmtap: the stored key {vault_info.name!r} may only be sent to "
                            f"{vault_info.base or 'its provider'}, but the {route.provider} upstream is configured "
                            f"as {route.upstream_base}. If that is intended, store the key again for that URL "
                            f"(`llmtap keys add {vault_info.name} --base {route.upstream_base}`).",
                            "llmtap_key_wrong_upstream")
            elif problem:
                ex.source = "error"
                self._error(out, ex, route.style, 400, problem, "llmtap_bedrock_route")
            else:
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
        path_model = model_from_path(route.upstream_path)
        # Bedrock and Gemini name the model in the URL, and that is the model the upstream runs (a "model" field in
        # a Gemini body is not what gets billed), so the URL wins there
        url_model_wins = is_bedrock_call(route.upstream_path) or route.style == "gemini"
        ex.model = (path_model if url_model_wins else None) or request_model(req) or path_model
        stream_req = is_stream_request(req)
        ex.tags = tags_from_headers(hdrs)
        ex.tags.pop(VAULT_TAG, None)          # llmtap's own attribution tag: a client header must not fake it
        vault_info = ex.__dict__.get("_vault")
        if vault_info is not None:
            ex.tags[VAULT_TAG] = vault_info.name

        llm = _is_llm_call(method, route.upstream_path)
        ctype = hdrs.get("content-type", "")

        # guard. Only JSON documents may ever be rewritten: multipart / binary / unparseable bodies are
        # scanned as text (warn / block) but forwarded byte-for-byte even in redact mode.
        send_body = body
        mode = cfg.guard or "off"
        if body and mode != "off":
            if mode == "redact" and req is None and not _is_json_doc(body):
                mode = "warn"
            send_body, findings, blocked = apply_guard(body, mode, cfg.guard_pii)
            ex.guard_findings = [f.to_dict() if hasattr(f, "to_dict") else dict(f) for f in findings]
            # A SigV4 signature covers the body (and often its length): a redacted body would be rejected by
            # AWS, or hang waiting for the bytes a signed Content-Length promised. Refuse it instead.
            signed_rewrite = send_body is not body and is_sigv4(hdrs)
            if blocked or signed_rewrite:
                # Never persist the secret we just refused to send: store the redacted body instead.
                stored_body, _, _ = apply_guard(body, "redact", cfg.guard_pii)
                ex.request_body, trunc = _decode_request(stored_body, ctype)
                if trunc:
                    _note(ex, f"request body truncated to {MAX_STORED_CHARS} characters in llmtap storage")
                kinds = ", ".join(sorted({str(f.get("kind", "?")) for f in ex.guard_findings}))
                ex.source = "blocked"
                msg = (f"llmtap guard blocked this request: detected {kinds} in the prompt. "
                       "Remove the secret, or run llmtap with --guard redact / warn.")
                if signed_rewrite and not blocked:
                    msg = (f"llmtap guard blocked this request: detected {kinds} in the prompt. It is signed with "
                           "AWS SigV4, so redacting it would break the signature. Remove the secret, use a Bedrock "
                           "API key (Bearer) instead of SigV4, or run llmtap with --guard warn.")
                self._error(out, ex, route.style, 400, msg, "llmtap_guard_blocked")
                return
            if send_body is not body:
                req = parse_json(send_body) if req is not None else req
        ex.request_body, trunc = _decode_request(send_body, ctype)
        if trunc:
            _note(ex, f"request body truncated to {MAX_STORED_CHARS} characters in llmtap storage (forwarded in full)")

        cacheable = llm
        key = cache_key(route.provider, route.upstream_path, send_body) if cacheable else ""
        ex.request_key = key if key and send_body else None

        # loop breaker: counts every identical generation call, even ones the cache will answer (a stuck
        # agent replaying a cached request would otherwise loop forever)
        if cacheable:
            verdict = self.budget.check_loop(key)
            if not verdict.allowed:
                ex.source = "loop"
                self._error(out, ex, route.style, 429, verdict.reason or "llmtap loop detected", "llmtap_loop_detected")
                return

        # cassette / cache next: replays are free, so spend caps never block them.
        if cacheable:
            stored: CachedResponse | None = None
            recording = self.cassette is not None and self.cassette.mode == "record"
            if self.cassette is not None:
                stored = self.cassette.lookup(key)
                if stored is not None:
                    ex.source = "cassette"
            # record mode means "call the real API": a cache hit would leave the cassette without this request
            if stored is None and self.cache is not None and not recording:
                stored = self.cache.get(key)
                if stored is not None:
                    ex.source = "cache"
                    # auto mode: the cassette missed, so keep it complete with the cache-served response
                    if self.cassette is not None and self.cassette.should_record():
                        try:
                            self.cassette.record(key, req if req is not None else ex.request_body, stored)
                        except Exception as e:
                            ex.error = ex.error or f"cassette write failed: {e}"
            if stored is not None:
                self._replay(ex, route, req, stored, out, t0)
                return
            if self.cassette is not None and self.cassette.mode == "replay" and not cfg.allow_miss:
                ex.source = "error"
                msg = ("llmtap cassette miss (replay mode): no recording for this request. "
                       "Re-record with --mode record")
                self._error(out, ex, route.style, 409, msg, "llmtap_cassette_miss")
                return

        # spend caps: only for generation calls that will actually reach the upstream
        reservation = None
        estimate = self._estimate_cost(ex, route, req, send_body) if cacheable else 0.0
        if cacheable:
            verdict = self.budget.check_spend(estimate)
            if not verdict.allowed:
                ex.source = "budget"
                self._error(out, ex, route.style, 429, verdict.reason or "llmtap budget exceeded",
                            "llmtap_budget_exceeded")
                return
            reservation = verdict.reservation

        # stored-key limits (paused, expiry, allowed models, daily / total spend) for calls that use the key
        vault_reserved: float | None = None
        if vault_info is not None:
            def spent(k: Any) -> tuple[float, float]:
                return (self.store.spent_by_tag(VAULT_TAG, k.name, _local_midnight()) if k.daily_usd else 0.0,
                        self.store.spent_by_tag(VAULT_TAG, k.name, k.created) if k.total_usd else 0.0)
            try:
                info = self.vault.get(vault_info.name) or vault_info      # pick up limit changes made meanwhile
                refusal = endpoint_refusal(info, method, route.upstream_path, cacheable)
                if refusal is None:
                    named = [ex.model, request_model(req), path_model]  # every model the request names...
                    if info.models:
                        more, dup = body_models(send_body)          # ...including tools[].model and friends
                        named += more
                        if dup:
                            refusal = Refusal(400, "llmtap: the request names \"model\" twice in one object; "
                                                   "stored keys with a model list refuse that.", "llmtap_bad_request")
                    if refusal is None:
                        refusal = self.vault.admit(info, named, estimate, cacheable, spent)
            except Exception as e:      # fail closed: limits that can't be checked must not be skipped
                refusal = Refusal(503, f"llmtap: could not check the limits of the stored key "
                                       f"{vault_info.name!r}: {type(e).__name__}: {e}", "llmtap_key_check_failed")
            if refusal is not None:
                self.budget.release(reservation)
                ex.source = "budget" if refusal.status == 429 else "blocked"
                self._error(out, ex, route.style, refusal.status, refusal.message, refusal.err_type)
                return
            vault_reserved = estimate if cacheable else None
            if vault_reserved is not None:
                # held until the exchange is stored (see _finish): per-key spend is read from the database, and the
                # client can get its response (and send the next call) before the row is written
                ex.__dict__["_vault_hold"] = (vault_info.name, vault_reserved)

        try:
            self._forward(ex, route, method, hdrs, send_body, req, stream_req, key, out, t0)
        finally:
            self.budget.release(reservation)

    def _estimate_cost(self, ex: Exchange, route: Route, req: dict | None, body: bytes) -> float:
        """Worst-case cost guess used to reserve budget for an in-flight call (0.0 when unknown)."""
        try:
            model = ex.model or (request_model(req) if req is not None else None)
            if not model:
                return 0.0
            out_tokens = 4096
            if isinstance(req, dict):
                gc = req.get("generationConfig") if isinstance(req.get("generationConfig"), dict) else {}
                ic = req.get("inferenceConfig") if isinstance(req.get("inferenceConfig"), dict) else {}
                for v in (req.get("max_tokens"), req.get("max_completion_tokens"), req.get("max_output_tokens"),
                          gc.get("maxOutputTokens"), ic.get("maxTokens"), ic.get("max_new_tokens"),
                          req.get("max_gen_len")):
                    if isinstance(v, int) and v > 0:
                        out_tokens = v
                        break
                n = req.get("n", gc.get("candidateCount"))           # several completions bill several outputs
                if isinstance(n, int) and not isinstance(n, bool) and 1 < n <= 128:
                    out_tokens *= n
            usage = Usage(input_tokens=max(1, len(body) // 4), output_tokens=out_tokens)
            return float(cost_usd(model, usage, self.cfg.pricing_overrides, style=route.style) or 0.0)
        except Exception:
            return 0.0

    # ---------------------------------------------------------------- upstream
    def _forward(self, ex: Exchange, route: Route, method: str, hdrs: dict[str, str], body: bytes,
                 req: dict | None, stream_req: bool, key: str, out: _Tracked, t0: float) -> None:
        llm = _is_llm_call(method, route.upstream_path)
        scheme, host, port, full_path = _upstream_target(route)
        timeout = float(self.cfg.upstream_timeout_s or 600.0)
        conn: http.client.HTTPConnection
        if scheme == "https":
            conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.response_class = _FinalResponse
        fwd = forward_headers(
            hdrs,
            preserve_signed=route.style == "bedrock",
        )
        # A signed Content-Length is kept, but it must describe the body actually sent: the client's claim may
        # not match how its body was framed (chunked) and would let the upstream read a second, unaccounted
        # request out of it, or wait for bytes that never come.
        for k in [k for k in fwd if k.lower() == "content-length"]:
            fwd[k] = str(len(body))
        # stored key: anything the upstream echoes back is scrubbed before the client (or the history) sees it,
        # which needs an uncompressed response
        vsec = [s.encode() for s in (ex.__dict__.get("_vault_secrets") or [])]
        if vsec:
            for k in [k for k in fwd if k.lower() == "accept-encoding"]:
                del fwd[k]
            fwd["Accept-Encoding"] = "identity"
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
            if vsec:
                vstr = [s.decode() for s in vsec]
                for k in list(client_h):
                    client_h[k] = _scrub(client_h[k], vstr)
            scrubber = _StreamScrub(vsec) if vsec else None
            bodyless = method == "HEAD" or resp.status in (204, 304) or resp.status < 200
            if bodyless and resp.status != 204 and "content-length" in low:
                client_h["Content-Length"] = low["content-length"]   # HEAD / 304 describe the entity: keep its length
            streaming = not bodyless and _is_streaming_response(ctype, stream_req, "content-length" in low)
            ex.streamed = streaming
            raw = bytearray()
            client_gone = False
            finalized = False
            upstream_err: str | None = None
            compressed = encoding.strip().lower() not in ("", "identity")
            if vsec and compressed and streaming:
                # the upstream ignored `Accept-Encoding: identity`: a compressed reply has to be decoded (in full)
                # before an echoed key can be scrubbed from it
                streaming = False
                ex.streamed = False

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
                    piece = scrubber.feed(chunk) if scrubber else chunk
                    try:
                        if piece:
                            out.write(piece)
                    except ClientGone:
                        client_gone = True
                        break
                if not (client_gone or upstream_err) and scrubber:
                    try:
                        tail = scrubber.flush()
                        if tail:
                            out.write(tail)
                    except ClientGone:
                        client_gone = True
                if vsec:
                    raw = bytearray(_scrub_bytes(bytes(raw), vsec))
                if client_gone or upstream_err:
                    out.abort()
                else:
                    self._finalize(ex, route, req, key, llm, bytes(raw), encoding, ctype, streaming, up_headers)
                    finalized = True
                    try:
                        out.end()
                    except ClientGone:
                        client_gone = True
            else:
                try:
                    data = resp.read() if not bodyless else b""
                except (OSError, http.client.HTTPException) as e:
                    ex.source = "error"
                    msg = f"llmtap: upstream read failed: {type(e).__name__}: {e}"
                    status = 504 if isinstance(e, (socket.timeout, TimeoutError)) else 502
                    self._error(out, ex, route.style, status, msg, "llmtap_upstream_error")
                    return
                if vsec and compressed and data:
                    plain = _try_decode(data, encoding.strip().lower())
                    if plain is None:
                        ex.source = "error"
                        self._error(out, ex, route.style, 502,
                                    f"llmtap: the upstream sent a {encoding}-compressed response that llmtap can't "
                                    "decode, so it can't make sure the stored key isn't echoed in it.",
                                    "llmtap_upstream_error")
                        return
                    data, encoding = plain, ""
                    for k in [k for k in client_h if k.lower() == "content-encoding"]:
                        del client_h[k]
                if vsec:
                    data = _scrub_bytes(data, vsec)
                raw += data
                ex.ttft_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                if not bodyless:
                    client_h["Content-Length"] = str(len(raw))
                self._finalize(ex, route, req, key, llm, bytes(raw), encoding, ctype, streaming, up_headers)
                finalized = True
                try:
                    out.start(resp.status, client_h)
                    if raw and not bodyless:
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

        if not finalized:
            self._finalize(ex, route, req, key, llm, bytes(raw), encoding, ctype, streaming, up_headers,
                           client_gone, upstream_err)
        elif client_gone:
            ex.error = "client disconnected"

    def _finalize(self, ex: Exchange, route: Route, req: dict | None, key: str, llm: bool, raw: bytes,
                  encoding: str, ctype: str, streaming: bool, up_headers: list[tuple[str, str]],
                  client_gone: bool = False, upstream_err: str | None = None) -> None:
        """Usage, cost, budget, cache and cassette for a finished upstream response. Runs *before* the last
        byte reaches the client (when possible) so a client that immediately sends its next request already
        sees the updated budget / cache."""
        raw = bytes(raw)
        try:
            text, parseable = _decode_response(raw, encoding, ctype)
        except Exception as e:   # never let storage decoding cut off a response that was already relayed
            text, parseable = f"[response body not stored: {type(e).__name__}, {len(raw)} bytes]", False
        ex.response_body = text
        if client_gone:
            ex.error = "client disconnected"
        elif upstream_err:
            ex.error = upstream_err
        if ex.status >= 400 and not ex.error:
            ex.error = _upstream_error_message(text) or f"upstream HTTP {ex.status}"

        in_band_err = self._account(ex, route, req, text, streaming, llm=llm, parseable=parseable,
                                    headers=dict((k.lower(), v) for k, v in up_headers))
        ex.cost_usd = self._price(ex) if llm and ex.usage.total() else 0.0  # GETs of stored objects etc. are free
        try:
            self.budget.add_spend(ex.cost_usd)
        except Exception:
            pass

        oversize = len(text) > MAX_STORED_CHARS or len(ex.output_text) > MAX_STORED_CHARS
        if oversize:
            ex.response_body = text[:MAX_STORED_CHARS] + f"\n[llmtap: truncated, {len(raw)} bytes in total]"
            ex.output_text = ex.output_text[:MAX_STORED_CHARS]
            _note(ex, f"response body truncated to {MAX_STORED_CHARS} characters in llmtap storage "
                      "(relayed to the client in full; not cached)")

        if (200 <= ex.status < 300 and not (streaming and client_gone) and not upstream_err and key and parseable
                and not oversize and not in_band_err):
            secrets = ex.__dict__.get("_secrets") or []
            cached = CachedResponse(status=ex.status, headers=_storable_headers(up_headers, secrets),
                                    body=_scrub(text, secrets), streamed=streaming, cost_usd=ex.cost_usd)
            if self.cache is not None:
                try:
                    self.cache.put(key, cached)
                except Exception as e:
                    ex.error = ex.error or f"cache write failed: {e}"
            if self.cassette is not None and self.cassette.should_record():
                try:
                    rec_req: Any = req if req is not None else ex.request_body
                    if secrets and isinstance(rec_req, dict):
                        rec_req = json.loads(_scrub(json.dumps(rec_req, ensure_ascii=False), secrets))
                    self.cassette.record(key, rec_req, cached)
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
            if ex.streamed and h.get("content-type", "").split(";")[0].strip().lower() == EVENTSTREAM_CTYPE:
                # a recorded Bedrock stream: rebuild the binary EventStream frames from the stored JSON lines
                h["Transfer-Encoding"] = "chunked"
                out.start(ex.status, h)
                for i, frame in enumerate(text_to_eventstream(stored.body)):
                    out.write(frame)
                    if i == 0:
                        ex.ttft_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                out.end()
            elif ex.streamed:
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
        self._account(ex, route, req, stored.body, ex.streamed, headers=h)

    # ---------------------------------------------------------------- accounting
    def _account(self, ex: Exchange, route: Route, req: dict | None, text: str, streamed: bool,
                 llm: bool = True, parseable: bool = True, headers: Mapping[str, str] | None = None) -> str | None:
        """Fill usage / output_text / model from a response body (never raises).

        Returns an in-band error message (stream ``error`` event, or HTTP 200 with an ``error`` body) if any.
        Usage is only *estimated* for successful generation calls whose body carried no usage and no error.
        """
        usage, out_text, model, err = Usage(), "", None, None
        if parseable:
            try:
                ctype = next((str(v) for k, v in (headers or {}).items() if str(k).lower() == "content-type"), "")
                if streamed and ctype.split(";")[0].strip().lower() == EVENTSTREAM_CTYPE:
                    usage, out_text, model, err = parse_bedrock_stream(text)
                elif streamed:
                    usage, out_text, model = parse_sse(route.style, text)
                    err = sse_error(text)
                else:
                    j = parse_json(text.encode("utf-8"))
                    if j is not None:
                        usage = extract_usage(route.style, j)
                        out_text = extract_text(route.style, j)
                        model = response_model(route.style, j)
                        err = body_error(j)
                    elif text.lstrip().startswith("["):
                        # Gemini streamGenerateContent without ?alt=sse: a JSON array of chunks. Usage is
                        # cumulative, so the last chunk that has it wins; the text is all chunks joined.
                        chunks = [c for c in json.loads(text) if isinstance(c, dict)]
                        for c in chunks:
                            u = extract_usage(route.style, c)
                            if u.total():
                                usage = u
                            model = response_model(route.style, c) or model
                            err = err or body_error(c)
                        out_text = "".join(extract_text(route.style, c) or "" for c in chunks)
            except Exception:
                pass
        if route.style == "bedrock" and not usage.total():
            usage = bedrock_header_usage(headers)        # InvokeModel reports token counts in headers
        ok = 200 <= ex.status < 300
        if err and ok:
            ex.error = ex.error or err
        if ok and llm and not err and usage.input_tokens == 0 and usage.output_tokens == 0:
            try:
                usage = estimate_usage(req, out_text)
            except Exception:
                pass
        ex.usage = usage
        ex.output_text = out_text or ""
        if model:
            ex.model = model
        return err if ok else None

    def _price(self, ex: Exchange) -> float:
        try:
            usage = ex.usage
            if usage.service_tier and ex.provider != "openai":     # only OpenAI's flex tier is half price
                usage = dataclasses.replace(usage, service_tier="")
            return float(cost_usd(ex.model, usage, self.cfg.pricing_overrides, style=ex.style) or 0.0)
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
        secrets = ex.__dict__.get("_secrets")
        if secrets:     # e.g. an http.client error quoting the URL, which may carry a swapped-in ?key=
            message = _scrub(message, secrets)
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
            _seal(ex)
        except Exception:
            ex.__dict__.pop("_secrets", None)
        hold = ex.__dict__.pop("_vault_hold", None)
        try:
            self.store.record(ex)
        except Exception as e:
            ex.error = ex.error or f"store write failed: {e}"
        finally:
            if hold is not None:
                try:
                    self.vault.release(*hold)
                except Exception:
                    pass
        try:
            self.bus.publish({"type": "exchange", "exchange": summarize(ex)})
        except Exception:
            pass
        if self.otel is not None and ex.cost_usd > 0:
            try:
                self.otel.submit(ex)    # background thread: never waits on the collector
            except Exception:
                pass
        if not self.cfg.quiet:
            try:
                from .term import log_exchange
                log_exchange(ex)
            except Exception:
                pass


def _host_is_exposed(host: str) -> bool:
    h = (host or "").strip().strip("[]").lower()
    if h == "localhost":
        return False
    try:
        return not ipaddress.ip_address(h).is_loopback
    except ValueError:
        return True   # "" (all interfaces) or a hostname we can't prove is loopback


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


_HEX = re.compile(rb"[0-9A-Fa-f]{1,16}")   # chunk size: no sign, 0x prefix or underscores (int() would take them)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llmtap/" + __version__
    sys_version = ""
    timeout = 900          # reap idle keep-alive connections / stalled clients instead of leaking threads
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - silence default logging
        pass

    def _read_exactly(self, n: int, what: str) -> bytes:
        # In pieces: rfile.read(n) allocates n bytes up front, so a lying Content-Length would cost memory.
        buf = bytearray()
        while len(buf) < n:
            data = self.rfile.read(min(n - len(buf), 1 << 20))
            if not data:
                raise _BadRequest(what)
            buf += data
        return bytes(buf)

    def _read_body(self) -> bytes:
        tes = self.headers.get_all("Transfer-Encoding") or []
        cls = {v.strip() for v in (self.headers.get_all("Content-Length") or [])}
        if tes:
            if [t.strip().lower() for t in ",".join(tes).split(",")] != ["chunked"]:
                raise _BadRequest("unsupported Transfer-Encoding")
            if cls:
                self.close_connection = True   # both framings: chunked wins, but never reuse the connection
            buf = bytearray()
            while True:
                line = self.rfile.readline(65537)
                if not line:
                    raise _BadRequest("truncated chunked body")
                size_s = line.split(b";", 1)[0].strip()
                if not _HEX.fullmatch(size_s):
                    raise _BadRequest("bad chunk size")
                size = int(size_s, 16)
                if size == 0:
                    while True:  # trailers
                        t = self.rfile.readline(65537)
                        if t in (b"\r\n", b"\n", b""):
                            break
                    return bytes(buf)
                buf += self._read_exactly(size, "truncated chunk")
                if self.rfile.readline(3) not in (b"\r\n", b"\n"):
                    raise _BadRequest("bad chunk framing")
        if cls:
            if len(cls) > 1:
                raise _BadRequest("conflicting Content-Length headers")
            cl = cls.pop()
            if not cl.isascii() or not cl.isdigit():
                raise _BadRequest("bad Content-Length")
            return self._read_exactly(int(cl), "truncated body")
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
            self.server.tap.handle(self.command, self.path, headers, body, r, client=str(self.client_address[0]))
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
