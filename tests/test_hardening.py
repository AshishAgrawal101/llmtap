"""Adversarial hardening tests: real-world SDK / HTTP / security behavior, against local fake upstreams."""
from __future__ import annotations

import http.client
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from llmtap.config import Config
from llmtap.models import Usage
from llmtap.pricing import cost_usd
from llmtap.proxy import LLMTap

ANTH_H = {"x-api-key": "sk-ant-test-0000", "anthropic-version": "2023-06-01"}


# ------------------------------------------------------------------------------------------------
# programmable fake upstream
# ------------------------------------------------------------------------------------------------
class Up:
    """Local HTTP/1.1 server; ``fn(handler, rec)`` writes the response (default: JSON 200 echo)."""

    def __init__(self, fn: Callable[[BaseHTTPRequestHandler, dict], None] | None = None) -> None:
        self.fn = fn or (lambda h, rec: send(h, 200, {"Content-Type": "application/json"}, b'{"ok":true}'))
        self.requests: list[dict] = []
        self.live = 0
        self.max_live = 0
        self.lock = threading.Lock()
        up = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def _go(self) -> None:
                if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                    buf = b""
                    while True:
                        size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                        if size == 0:
                            self.rfile.readline()
                            break
                        buf += self.rfile.read(size)
                        self.rfile.readline()
                    raw = buf
                else:
                    raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                rec = {"method": self.command, "path": self.path, "raw": raw,
                       "headers": {k.lower(): v for k, v in self.headers.items()}}
                with up.lock:
                    up.requests.append(rec)
                    up.live += 1
                    up.max_live = max(up.max_live, up.live)
                try:
                    up.fn(self, rec)
                except (OSError, ValueError):
                    self.close_connection = True
                finally:
                    with up.lock:
                        up.live -= 1

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _go

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.server.request_queue_size = 128
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def send(h: BaseHTTPRequestHandler, status: int, headers: dict[str, str], body: bytes,
         length: bool = True) -> None:
    h.send_response(status)
    for k, v in headers.items():
        h.send_header(k, v)
    if length:
        h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    if body and h.command != "HEAD":
        h.wfile.write(body)
        h.wfile.flush()


def sse(h: BaseHTTPRequestHandler, frames: list[bytes], delay: float = 0.0) -> None:
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.send_header("Transfer-Encoding", "chunked")
    h.end_headers()
    for f in frames:
        h.wfile.write(b"%x\r\n%s\r\n" % (len(f), f))
        h.wfile.flush()
        if delay:
            time.sleep(delay)
    h.wfile.write(b"0\r\n\r\n")
    h.wfile.flush()


@pytest.fixture
def mk():
    """Factory ``mk(up=None, **cfg) -> (tap, up)``."""
    taps: list[LLMTap] = []
    ups: list[Up] = []

    def make(up: Up | None = None, **kw: Any) -> tuple[LLMTap, Up]:
        if up is None:
            up = Up()
        ups.append(up)
        kw.setdefault("guard", "off")
        kw.setdefault("db_path", ":memory:")
        cfg = Config(port=0, quiet=True, **kw)
        cfg.upstreams["anthropic"] = {"base": up.base_url, "style": "anthropic"}
        cfg.upstreams["openai"] = {"base": up.base_url, "style": "openai"}
        cfg.upstream_timeout_s = kw.get("upstream_timeout_s", 10)
        t = LLMTap(cfg).start_background()
        taps.append(t)
        return t, up

    yield make
    for t in taps:
        t.shutdown()
    for u in ups:
        u.stop()


def call(tap: LLMTap, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None,
         timeout: float = 20) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", tap.port, timeout=timeout)
    h = dict(headers or {})
    if isinstance(body, (dict, list)):
        data: bytes | None = json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    else:
        data = body
    try:
        conn.request(method, path, body=data, headers=h)
        r = conn.getresponse()
        out = r.read()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, out
    finally:
        conn.close()


def last(tap: LLMTap, n: int = 1, timeout: float = 5.0) -> Any:
    """Wait until ``n`` exchanges are stored and return the newest."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if tap.store.count() >= n:
            return tap.store.list(limit=1)[0]
        time.sleep(0.02)
    raise AssertionError("exchange never recorded")


def wait_ex(tap: LLMTap, ex_id: str, timeout: float = 5.0) -> Any:
    deadline = time.time() + timeout
    while time.time() < deadline:
        ex = tap.store.get(ex_id)
        if ex is not None:
            return ex
        time.sleep(0.02)
    raise AssertionError("exchange never recorded")


def raw_http(port: int, payload: bytes, timeout: float = 10, host: str = "127.0.0.1") -> bytes:
    """Send raw bytes, read until the server closes the connection."""
    s = socket.create_connection((host, port), timeout=timeout)
    s.sendall(payload)
    buf = b""
    try:
        while True:
            c = s.recv(65536)
            if not c:
                break
            buf += c
    except socket.timeout:
        pass
    finally:
        s.close()
    return buf


def msg(text: str = "hi", model: str = "claude-sonnet-4-5", **extra: Any) -> dict:
    return {"model": model, "max_tokens": 20, "messages": [{"role": "user", "content": text}], **extra}


def anth_json(text: str = "ok", usage: dict | None = None, model: str = "claude-sonnet-4-5") -> bytes:
    return json.dumps({"id": "msg_1", "type": "message", "role": "assistant", "model": model,
                       "content": [{"type": "text", "text": text}],
                       "usage": usage or {"input_tokens": 10, "output_tokens": 5}}).encode()


def llm_up(usage: dict | None = None) -> Up:
    return Up(lambda h, rec: send(h, 200, {"Content-Type": "application/json"}, anth_json(usage=usage)))


# ================================================================================================
# A. passthrough fidelity
# ================================================================================================

def _open_fds() -> int:
    """Number of open file descriptors (Linux: /proc/self/fd, macOS/BSD: /dev/fd)."""
    for d in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(d):
            return len(os.listdir(d))
    pytest.skip("cannot count open fds on this platform")


def test_a_headers_and_query_forwarded(mk):
    tap, up = mk()
    hdrs = {**ANTH_H, "anthropic-beta": "prompt-caching-2024-07-31,files-api-2025-04-14",
            "openai-organization": "org-x", "openai-beta": "assistants=v2",
            "Authorization": "Bearer abc123", "User-Agent": "Anthropic/Python 0.40",
            "X-Stainless-Lang": "python", "x-stainless-retry-count": "0",
            "X-LLMTap-Tag-Feature": "f", "Connection": "keep-alive"}
    st, _, _ = call(tap, "POST", "/v1/messages?beta=true&x=1", msg(), hdrs)
    assert st == 200
    rec = up.requests[0]
    assert rec["path"] == "/v1/messages?beta=true&x=1"
    h = rec["headers"]
    for k in ("anthropic-beta", "anthropic-version", "x-api-key", "openai-organization", "openai-beta",
              "authorization", "user-agent", "x-stainless-lang", "x-stainless-retry-count"):
        assert k in h, k
    assert h["anthropic-beta"] == "prompt-caching-2024-07-31,files-api-2025-04-14"
    assert h["user-agent"] == "Anthropic/Python 0.40"
    assert not any(k.startswith("x-llmtap") for k in h)
    assert h["accept-encoding"] == "identity"
    ex = last(tap)
    assert ex.path == "/v1/messages?beta=true&x=1" and ex.tags == {"feature": "f"}


@pytest.mark.parametrize("method,path,body", [
    ("GET", "/v1/models", None),
    ("GET", "/v1/models?limit=5", None),
    ("DELETE", "/v1/files/file_abc", None),
    ("PUT", "/v1/things/1", b'{"a":1}'),
    ("PATCH", "/v1/things/1", b'{"a":1}'),
    ("POST", "/v1/messages/count_tokens", {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "x"}]}),
    ("POST", "/v1/messages/batches", {"requests": [{"custom_id": "a", "params": msg()}]}),
    ("POST", "/v1/files", b"--b\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--b--\r\n"),
    ("GET", "/v1/messages/batches/msgbatch_1/results", None),
])
def test_a_non_chat_endpoints_forwarded_untouched(mk, method, path, body):
    tap, up = mk(cache=True, loop_threshold=2, budget_session_usd=0.01)
    tap.budget.add_spend(5.0)  # session budget long exhausted: chat would be 429, non-chat must pass
    hdrs = dict(ANTH_H)
    if isinstance(body, bytes) and path == "/v1/files":
        hdrs["Content-Type"] = "multipart/form-data; boundary=b"
    elif isinstance(body, bytes):
        hdrs["Content-Type"] = "application/json"
    for _ in range(4):  # loop threshold 2: must never trip; cache on: must never short-circuit
        st, h, out = call(tap, method, path, body, hdrs)
        assert st == 200, (st, out)
        assert h["x-llmtap-source"] == "upstream"
    assert len(up.requests) == 4
    assert all(r["method"] == method and r["path"] == path for r in up.requests)
    if body is not None:
        want = body if isinstance(body, bytes) else json.dumps(body).encode()
        assert up.requests[0]["raw"] == want
    ex = last(tap)
    assert ex.source == "upstream" and ex.cost_usd == 0.0 and ex.usage.total() == 0
    assert tap.cache.stats()["entries"] == 0


def test_a_chat_still_budgeted_after_non_chat_exemption(mk):
    tap, up = mk(budget_session_usd=0.01)
    tap.budget.add_spend(5.0)
    st, _, _ = call(tap, "POST", "/v1/messages", msg(), ANTH_H)
    assert st == 429


def test_a_routing_anthropic_headers_and_oauth_paths():
    from llmtap.providers import resolve_route
    cfg = Config(port=0)
    # Claude Code with an OAuth bearer token: only anthropic-* headers identify the provider
    r = resolve_route("/api/oauth/profile", {"Authorization": "Bearer x", "anthropic-beta": "oauth-2025-04-20"}, cfg)
    assert r.provider == "anthropic" and r.upstream_path == "/api/oauth/profile"
    r = resolve_route("/v1/models", {"X-API-KEY": "k"}, cfg)
    assert r.provider == "anthropic"
    r = resolve_route("/api/hello", {"anthropic-dangerous-direct-browser-access": "true"}, cfg)
    assert r.provider == "anthropic"
    r = resolve_route("/v1/messages/count_tokens?beta=true", {}, cfg)
    assert r.provider == "anthropic" and r.upstream_path == "/v1/messages/count_tokens?beta=true"


# ================================================================================================
# B. binary / large bodies
# ================================================================================================
def _multipart(secret: str) -> bytes:
    return (b"--XYZ\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.bin\"\r\n"
            b"Content-Type: application/octet-stream\r\n\r\n"
            + bytes(range(256)) * 4 + b"\xff\xfe\x00 key=" + secret.encode() + b" \x00\x80\r\n"
            b"--XYZ\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nassistants\r\n--XYZ--\r\n")


@pytest.mark.parametrize("mode", ["off", "warn", "redact"])
def test_b_multipart_forwarded_byte_for_byte(mk, mode):
    tap, up = mk(guard=mode)
    body = _multipart("AKIAIOSFODNN7EXAMPLE")
    st, _, _ = call(tap, "POST", "/v1/files", body,
                    {"Content-Type": "multipart/form-data; boundary=XYZ", "Authorization": "Bearer k"})
    assert st == 200
    assert up.requests[0]["raw"] == body
    ex = last(tap)
    assert len(ex.request_body) < 2000          # binary payload is not dumped into the DB
    assert "multipart" in ex.request_body.lower()
    if mode != "off":
        assert any(f["kind"] == "aws_access_key" for f in ex.guard_findings)  # still *scanned*


def test_b_binary_audio_style_body_untouched_with_redact(mk):
    tap, up = mk(guard="redact")
    body = os.urandom(5000) + b"AKIAIOSFODNN7EXAMPLE" + os.urandom(100)
    st, _, _ = call(tap, "POST", "/v1/audio/transcriptions", body, {"Content-Type": "application/octet-stream"})
    assert st == 200 and up.requests[0]["raw"] == body


def test_b_large_request_10mb(mk):
    tap, up = mk(guard="warn")
    big = "lorem ipsum dolor " * (10 * 1024 * 1024 // 18)
    t0 = time.time()
    st, _, _ = call(tap, "POST", "/v1/messages", msg(big), ANTH_H, timeout=60)
    assert st == 200
    assert time.time() - t0 < 20
    assert len(up.requests[0]["raw"]) > 10_000_000
    ex = last(tap)
    assert len(ex.request_body) <= 2 * 1024 * 1024 + 2048      # stored copy is capped
    assert "truncated" in (ex.error or "").lower()


def test_b_large_response_relayed_fully_but_storage_capped(mk, tmp_path):
    payload = "x" * (5 * 1024 * 1024)
    body = json.dumps({"id": "m", "type": "message", "model": "claude-sonnet-4-5",
                       "content": [{"type": "text", "text": payload}],
                       "usage": {"input_tokens": 3, "output_tokens": 4}}).encode()
    up = Up(lambda h, rec: send(h, 200, {"Content-Type": "application/json"}, body))
    cas = tmp_path / "c.jsonl"
    tap, _ = mk(up, cache=True, cassette=str(cas), mode="record")
    st, _, out = call(tap, "POST", "/v1/messages", msg(), ANTH_H, timeout=60)
    assert st == 200 and out == body                     # relayed byte-for-byte
    ex = last(tap)
    assert len(ex.response_body) <= 2 * 1024 * 1024 + 2048
    assert "truncated" in (ex.error or "").lower()
    assert ex.usage.input_tokens == 3 and ex.usage.output_tokens == 4   # usage taken from the full body
    assert tap.cache.stats()["entries"] == 0             # never cache a truncated/oversized body
    assert not cas.exists() or cas.stat().st_size < 3 * 1024 * 1024


def test_b_binary_response_not_stored_as_garbage(mk):
    blob = bytes(range(256)) * 2000
    up = Up(lambda h, rec: send(h, 200, {"Content-Type": "audio/mpeg"}, blob))
    tap, _ = mk(up, cache=True)
    st, h, out = call(tap, "POST", "/v1/audio/speech", {"model": "tts-1", "input": "hello"})
    assert st == 200 and out == blob and h["content-type"] == "audio/mpeg"
    ex = last(tap)
    assert len(ex.response_body) < 500 and "audio/mpeg" in ex.response_body
    assert ex.usage.total() == 0 or ex.usage.estimated is True


# ================================================================================================
# C. upstream response quirks
# ================================================================================================
def test_c_undecodable_encodings_relayed_untouched(mk, tmp_path):
    payload = b"\x1b\x00\x01BROTLI-ish bytes \xff\xfe" + os.urandom(200)

    def fn(h, rec):
        send(h, 200, {"Content-Type": "application/json", "Content-Encoding": "br"}, payload)
    tap, up = mk(Up(fn), cache=True, db_path=str(tmp_path / "d.db"))
    st, h, out = call(tap, "POST", "/v1/messages", msg(), ANTH_H)
    assert st == 200 and out == payload and h["content-encoding"] == "br"
    ex = last(tap)
    assert "br" in ex.response_body and len(ex.response_body) < 300      # placeholder, not mojibake
    assert tap.cache.stats()["entries"] == 0                              # no poisoned cache entry
    st, h2, out2 = call(tap, "POST", "/v1/messages", msg(), ANTH_H)       # second call goes upstream again
    assert out2 == payload and h2["x-llmtap-source"] == "upstream"


def test_c_corrupt_gzip_does_not_crash(mk):
    def fn(h, rec):
        send(h, 200, {"Content-Type": "application/json", "Content-Encoding": "gzip"}, b"not gzip at all")
    tap, _ = mk(Up(fn), cache=True)
    st, h, out = call(tap, "POST", "/v1/messages", msg(), ANTH_H)
    assert st == 200 and out == b"not gzip at all"
    assert tap.cache.stats()["entries"] == 0


def test_c_head_keeps_upstream_content_length(mk):
    def fn(h, rec):
        h.send_response(200)
        h.send_header("Content-Type", "application/octet-stream")
        h.send_header("Content-Length", "12345")
        h.end_headers()
    tap, _ = mk(Up(fn))
    st, h, out = call(tap, "HEAD", "/v1/files/f1/content", None, ANTH_H)
    assert st == 200 and out == b"" and h["content-length"] == "12345"


@pytest.mark.parametrize("status", [204, 304])
def test_c_no_body_statuses(mk, status):
    def fn(h, rec):
        h.send_response(status)
        h.send_header("ETag", '"abc"')
        h.end_headers()
    tap, _ = mk(Up(fn))
    s = raw_http(tap.port, b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
    assert s.startswith(f"HTTP/1.1 {status}".encode())
    assert b'etag: "abc"' in s.lower()
    assert s.endswith(b"\r\n\r\n")                  # no body bytes, no chunk terminator


def test_c_redirect_not_followed(mk):
    def fn(h, rec):
        send(h, 302, {"Location": "https://elsewhere.example/x"}, b"")
    tap, up = mk(Up(fn))
    st, h, _ = call(tap, "GET", "/v1/files/f/content", None, ANTH_H)
    assert st == 302 and h["location"] == "https://elsewhere.example/x"
    assert len(up.requests) == 1


def test_c_rate_limit_headers_relayed(mk):
    def fn(h, rec):
        send(h, 429, {"Content-Type": "application/json", "Retry-After": "17",
                      "anthropic-ratelimit-requests-remaining": "0",
                      "x-ratelimit-reset-tokens": "6m0s", "x-should-retry": "true"},
             b'{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}')
    tap, _ = mk(Up(fn))
    st, h, out = call(tap, "POST", "/v1/messages", msg(), ANTH_H)
    assert st == 429 and h["retry-after"] == "17" and h["x-ratelimit-reset-tokens"] == "6m0s"
    assert h["anthropic-ratelimit-requests-remaining"] == "0"
    ex = last(tap)
    assert ex.cost_usd == 0.0 and ex.usage.total() == 0 and ex.error == "slow down"


def test_c_chunked_json_without_content_length(mk):
    def fn(h, rec):
        h.send_response(200)
        h.send_header("Content-Type", "application/json")
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        data = anth_json("chunky")
        for i in range(0, len(data), 7):
            part = data[i:i + 7]
            h.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
        h.wfile.write(b"0\r\n\r\n")
    tap, _ = mk(Up(fn))
    st, h, out = call(tap, "POST", "/v1/messages", msg(), ANTH_H)
    assert st == 200 and json.loads(out)["content"][0]["text"] == "chunky"
    assert last(tap).output_text == "chunky"


def test_c_close_delimited_http10_upstream(mk):
    data = anth_json("closed")
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)

    def serve():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            c.settimeout(2)
            buf = b""
            try:
                while b"\r\n\r\n" not in buf:
                    buf += c.recv(65536)
                n = int(next((ln.split(b":")[1] for ln in buf.split(b"\r\n") if ln.lower().startswith(b"content-length")), b"0"))
                while len(buf.split(b"\r\n\r\n", 1)[1]) < n:
                    buf += c.recv(65536)
                c.sendall(b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n" + data)
            except OSError:
                pass
            finally:
                c.close()
    threading.Thread(target=serve, daemon=True).start()
    try:
        tap, _ = mk(Up())
        tap.cfg.upstreams["anthropic"]["base"] = f"http://127.0.0.1:{srv.getsockname()[1]}"
        st, h, out = call(tap, "POST", "/v1/messages", msg(), ANTH_H)
        assert st == 200 and out == data
        assert h["content-length"] == str(len(data))
    finally:
        srv.close()


def test_c_upstream_reset_mid_body_is_provider_shaped_502(mk):
    def fn(h, rec):
        h.send_response(200)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", "1000")
        h.end_headers()
        h.wfile.write(b'{"partial":')
        h.wfile.flush()
        h.close_connection = True
        h.connection.shutdown(socket.SHUT_RDWR)
    tap, _ = mk(Up(fn))
    st, _, out = call(tap, "POST", "/v1/messages", msg(), ANTH_H)
    assert st == 502
    j = json.loads(out)
    assert j["type"] == "error" and "message" in j["error"]
    ex = last(tap)
    assert ex.source == "error" and ex.cost_usd == 0.0


def test_c_upstream_timeout_is_504_provider_shaped(mk):
    def fn(h, rec):
        time.sleep(1.5)
        send(h, 200, {"Content-Type": "application/json"}, b"{}")
    tap, _ = mk(Up(fn), upstream_timeout_s=0.3)
    st, _, out = call(tap, "POST", "/v1/chat/completions", {"model": "gpt-4o", "messages": []})
    assert st == 504
    assert json.loads(out)["error"]["type"] == "llmtap_upstream_error"
    assert last(tap).source == "error"


def test_c_read_timeout_after_headers_is_504(mk):
    def fn(h, rec):
        h.send_response(200)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", "100")
        h.end_headers()
        h.wfile.flush()
        time.sleep(1.5)
    tap, _ = mk(Up(fn), upstream_timeout_s=0.3)
    st, _, _ = call(tap, "POST", "/v1/messages", msg(), ANTH_H)
    assert st == 504


def test_c_expect_100_continue(mk):
    tap, up = mk(llm_up())
    body = json.dumps(msg()).encode()
    s = socket.create_connection(("127.0.0.1", tap.port), timeout=10)
    s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nx-api-key: k\r\nanthropic-version: 1\r\n"
              b"Content-Type: application/json\r\nExpect: 100-continue\r\nConnection: close\r\n"
              + b"Content-Length: %d\r\n\r\n" % len(body))
    first = s.recv(100)
    assert first.startswith(b"HTTP/1.1 100")
    s.sendall(body)
    buf = b""
    while True:
        c = s.recv(65536)
        if not c:
            break
        buf += c
    s.close()
    assert b"200 OK" in buf and b'"ok"' in buf
    assert "expect" not in up.requests[0]["headers"]


def test_c_http10_client_and_connection_close(mk):
    tap, _ = mk(llm_up())
    body = json.dumps(msg()).encode()
    req = (b"POST /v1/messages HTTP/1.0\r\nx-api-key: k\r\nContent-Type: application/json\r\n"
           + b"Content-Length: %d\r\n\r\n" % len(body) + body)
    resp = raw_http(tap.port, req)
    head, _, payload = resp.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.")
    assert json.loads(payload)["content"][0]["text"] == "ok"      # no chunk framing for 1.0
    assert b"transfer-encoding" not in head.lower()


def test_c_http10_client_streaming(mk):
    frames = [b"event: message_start\ndata: " + json.dumps({"type": "message_start", "message": {
        "model": "claude-sonnet-4-5", "usage": {"input_tokens": 4, "output_tokens": 1}}}).encode() + b"\n\n",
              b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"]
    tap, _ = mk(Up(lambda h, rec: sse(h, frames)))
    body = json.dumps(msg(stream=True)).encode()
    resp = raw_http(tap.port, b"POST /v1/messages HTTP/1.0\r\nx-api-key: k\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
    head, _, payload = resp.partition(b"\r\n\r\n")
    assert b"transfer-encoding" not in head.lower()
    assert payload == b"".join(frames)


# ================================================================================================
# D. streaming
# ================================================================================================
def _ev(name: str, obj: dict, crlf: bool = False) -> bytes:
    nl = "\r\n" if crlf else "\n"
    return f"event: {name}{nl}data: {json.dumps(obj)}{nl}{nl}".encode()


def test_d_anthropic_stream_crlf_ping_comments_and_cache_usage(mk):
    frames = [
        b": comment\r\n\r\n",
        _ev("message_start", {"type": "message_start", "message": {"id": "m", "model": "claude-sonnet-4-5", "usage": {
            "input_tokens": 100, "cache_read_input_tokens": 400, "cache_creation_input_tokens": 50,
            "output_tokens": 1, "server_tool_use": {"web_search_requests": 1}}}}, True),
        _ev("ping", {"type": "ping"}, True),
        _ev("content_block_start", {"type": "content_block_start", "index": 0,
                                    "content_block": {"type": "tool_use", "id": "t", "name": "f", "input": {}}}, True),
        _ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                    "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}}, True),
        _ev("content_block_delta", {"type": "content_block_delta", "index": 1,
                                    "delta": {"type": "text_delta", "text": "Hello"}}, True),
        b": ping\r\n\r\n",
        _ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                              "usage": {"output_tokens": 77}}, True),
        _ev("message_stop", {"type": "message_stop"}, True),
    ]
    tap, _ = mk(Up(lambda h, rec: sse(h, frames)))
    st, h, out = call(tap, "POST", "/v1/messages", msg(stream=True), ANTH_H)
    assert st == 200 and out == b"".join(frames)
    ex = last(tap)
    u = ex.usage
    assert (u.input_tokens, u.cache_read_tokens, u.cache_write_tokens, u.output_tokens) == (100, 400, 50, 77)
    assert not u.estimated and ex.output_text == "Hello"
    assert abs(ex.cost_usd - cost_usd("claude-sonnet-4-5", Usage(100, 77, 400, 50, web_search_requests=1))) < 1e-9
    assert abs(ex.cost_usd - cost_usd("claude-sonnet-4-5", Usage(100, 77, 400, 50)) - 0.01) < 1e-9


def test_d_midstream_error_event_flagged_not_priced_not_cached(mk):
    frames = [
        _ev("message_start", {"type": "message_start", "message": {"model": "claude-sonnet-4-5",
                                                                   "usage": {"input_tokens": 0, "output_tokens": 0}}}),
        _ev("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
    ]
    tap, up = mk(Up(lambda h, rec: sse(h, frames)), cache=True)
    st, _, out = call(tap, "POST", "/v1/messages", msg(stream=True), ANTH_H)
    assert st == 200 and out == b"".join(frames)
    ex = last(tap)
    assert "Overloaded" in (ex.error or "")
    assert ex.cost_usd == 0.0 and not ex.usage.estimated
    assert tap.cache.stats()["entries"] == 0


def test_d_openai_chat_stream_without_done_and_tool_calls(mk):
    ch = lambda d: b"data: " + json.dumps(d).encode() + b"\n\n"  # noqa: E731
    frames = [
        ch({"model": "gpt-4o", "choices": [{"index": 0, "delta": {"role": "assistant", "content": None, "tool_calls": [
            {"index": 0, "id": "c1", "type": "function", "function": {"name": "f", "arguments": ""}}]}}]}),
        ch({"model": "gpt-4o", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "{\"a\":1}"}}]}}]}),
        ch({"model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        ch({"model": "gpt-4o", "choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 9,
                                                        "prompt_tokens_details": {"cached_tokens": 20}}}),
    ]  # note: no [DONE]
    tap, _ = mk(Up(lambda h, rec: sse(h, frames)))
    st, _, out = call(tap, "POST", "/v1/chat/completions", {"model": "gpt-4o", "stream": True, "messages": []})
    assert st == 200 and out == b"".join(frames)
    u = last(tap).usage
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens) == (50, 9, 20) and not u.estimated


def test_d_openai_stream_error_chunk(mk):
    frames = [b'data: {"error": {"message": "The server had an error", "type": "server_error"}}\n\n']
    tap, _ = mk(Up(lambda h, rec: sse(h, frames)), cache=True)
    call(tap, "POST", "/v1/chat/completions", {"model": "gpt-4o", "stream": True, "messages": []})
    ex = last(tap)
    assert "server had an error" in (ex.error or "") and ex.cost_usd == 0.0
    assert tap.cache.stats()["entries"] == 0


def test_d_responses_api_stream(mk):
    ev = lambda name, obj: f"event: {name}\ndata: {json.dumps({'type': name, **obj})}\n\n".encode()  # noqa: E731
    item = {"type": "message", "content": [{"type": "output_text", "text": "Hi there"}]}
    frames = [
        ev("response.created", {"response": {"model": "gpt-5", "status": "in_progress"}}),
        ev("response.output_text.delta", {"delta": "Hi "}),
        ev("response.output_text.delta", {"delta": "there"}),
        ev("response.completed", {"response": {"model": "gpt-5", "status": "completed", "output": [item], "usage": {
            "input_tokens": 1000, "input_tokens_details": {"cached_tokens": 600},
            "output_tokens": 300, "output_tokens_details": {"reasoning_tokens": 250}, "total_tokens": 1300}}}),
    ]
    tap, _ = mk(Up(lambda h, rec: sse(h, frames)))
    call(tap, "POST", "/v1/responses", {"model": "gpt-5", "input": "x", "stream": True})
    ex = last(tap)
    assert ex.output_text == "Hi there"
    u = ex.usage
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens) == (1000, 300, 600)
    assert abs(ex.cost_usd - cost_usd("gpt-5", Usage(1000, 300, 600, 0))) < 1e-12   # reasoning not added twice


def test_d_huge_single_event(mk):
    big = "y" * (3 * 1024 * 1024)
    frames = [_ev("message_start", {"type": "message_start", "message": {"model": "claude-sonnet-4-5",
                                                                         "usage": {"input_tokens": 5}}}),
              _ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                          "delta": {"type": "text_delta", "text": big}}),
              _ev("message_delta", {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 9}})]
    tap, _ = mk(Up(lambda h, rec: sse(h, frames)))
    st, _, out = call(tap, "POST", "/v1/messages", msg(stream=True), ANTH_H, timeout=60)
    assert out == b"".join(frames)
    ex = last(tap)
    assert ex.usage.output_tokens == 9 and ex.usage.input_tokens == 5
    assert len(ex.response_body) <= 2 * 1024 * 1024 + 2048


def test_d_aborted_streams_do_not_leak(mk):
    ev0 = _ev("message_start", {"type": "message_start", "message": {"model": "claude-sonnet-4-5",
                                                                     "usage": {"input_tokens": 3}}})

    def fn(h, rec):
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        for _ in range(2000):
            f = ev0
            h.wfile.write(b"%x\r\n%s\r\n" % (len(f), f))
            h.wfile.flush()
            time.sleep(0.02)
    up = Up(fn)
    tap, _ = mk(up)
    base_threads = threading.active_count()
    base_fds = _open_fds()
    body = json.dumps(msg(stream=True)).encode()
    for _ in range(50):
        s = socket.create_connection(("127.0.0.1", tap.port), timeout=5)
        s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nx-api-key: k\r\nContent-Type: application/json\r\n"
                  + b"Content-Length: %d\r\n\r\n" % len(body) + body)
        s.recv(200)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")  # RST on close
        s.close()
    deadline = time.time() + 15
    while time.time() < deadline and (up.live > 0 or tap.store.count() < 50):
        time.sleep(0.05)
    assert up.live == 0, "upstream requests still being served: proxy did not drop upstream connections"
    assert tap.store.count() == 50
    assert all("disconnect" in (e.error or "") for e in tap.store.list(limit=50))
    time.sleep(0.3)
    assert threading.active_count() <= base_threads + 3
    assert _open_fds() <= base_fds + 6


# ================================================================================================
# E. usage / cost accuracy
# ================================================================================================
def test_e_openai_reasoning_not_double_counted(mk):
    body = json.dumps({"id": "c", "object": "chat.completion", "model": "gpt-5", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "A"}}],
        "usage": {"prompt_tokens": 2000, "completion_tokens": 1500, "total_tokens": 3500,
                  "prompt_tokens_details": {"cached_tokens": 1200},
                  "completion_tokens_details": {"reasoning_tokens": 1000}}}).encode()
    tap, _ = mk(Up(lambda h, rec: send(h, 200, {"Content-Type": "application/json"}, body)))
    call(tap, "POST", "/v1/chat/completions", {"model": "gpt-5", "messages": [{"role": "user", "content": "x"}]})
    ex = last(tap)
    assert ex.usage.output_tokens == 1500 and ex.usage.cache_read_tokens == 1200
    assert abs(ex.cost_usd - cost_usd("gpt-5", Usage(2000, 1500, 1200, 0))) < 1e-12


def test_e_responses_non_stream_cached_tokens(mk):
    body = json.dumps({"id": "r", "object": "response", "model": "gpt-4o", "output": [], "usage": {
        "input_tokens": 100, "input_tokens_details": {"cached_tokens": 64}, "output_tokens": 10}}).encode()
    tap, _ = mk(Up(lambda h, rec: send(h, 200, {"Content-Type": "application/json"}, body)))
    call(tap, "POST", "/v1/responses", {"model": "gpt-4o", "input": "x"})
    u = last(tap).usage
    assert (u.input_tokens, u.cache_read_tokens, u.output_tokens) == (100, 64, 10)


def test_e_embeddings_usage(mk):
    body = json.dumps({"object": "list", "model": "text-embedding-3-small", "data": [],
                       "usage": {"prompt_tokens": 800, "total_tokens": 800}}).encode()
    tap, _ = mk(Up(lambda h, rec: send(h, 200, {"Content-Type": "application/json"}, body)))
    call(tap, "POST", "/v1/embeddings", {"model": "text-embedding-3-small", "input": "x"})
    ex = last(tap)
    assert ex.usage.input_tokens == 800 and ex.usage.output_tokens == 0 and not ex.usage.estimated
    assert ex.cost_usd > 0


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503, 529])
def test_e_failed_calls_cost_nothing(mk, status):
    tap, _ = mk(Up(lambda h, rec: send(h, status, {"Content-Type": "application/json"},
                                       b'{"type":"error","error":{"type":"x","message":"nope"}}')))
    big = "word " * 5000
    call(tap, "POST", "/v1/messages", msg(big), ANTH_H)
    ex = last(tap)
    assert ex.cost_usd == 0.0 and ex.usage.total() == 0 and not ex.usage.estimated
    assert tap.budget.session_spent() == 0.0


def test_e_http200_with_error_body_costs_nothing(mk):
    tap, _ = mk(Up(lambda h, rec: send(h, 200, {"Content-Type": "application/json"},
                                       b'{"error":{"message":"provider returned error","code":502}}')))
    call(tap, "POST", "/v1/chat/completions", {"model": "gpt-4o", "messages": [{"role": "user", "content": "x " * 999}]})
    ex = last(tap)
    assert ex.cost_usd == 0.0 and not ex.usage.estimated
    assert "provider returned error" in (ex.error or "")


def test_e_count_tokens_not_priced_as_chat(mk):
    tap, _ = mk(Up(lambda h, rec: send(h, 200, {"Content-Type": "application/json"}, b'{"input_tokens": 1234}')))
    call(tap, "POST", "/v1/messages/count_tokens", msg("hello " * 400), ANTH_H)
    ex = last(tap)
    assert ex.cost_usd == 0.0 and ex.usage.total() == 0


def test_e_missing_usage_is_estimated_for_chat(mk):
    tap, _ = mk(Up(lambda h, rec: send(h, 200, {"Content-Type": "application/json"},
                                       json.dumps({"model": "gpt-4o", "choices": [
                                           {"index": 0, "message": {"role": "assistant", "content": "z" * 400}}]}).encode())))
    call(tap, "POST", "/v1/chat/completions", {"model": "gpt-4o", "messages": [{"role": "user", "content": "q" * 400}]})
    ex = last(tap)
    assert ex.usage.estimated and ex.usage.output_tokens == 100 and ex.cost_usd > 0


# ================================================================================================
# F. security
# ================================================================================================
SECRETS = ["sk-ant-api03-SUPERSECRETKEY123456", "Bearer tok_ABCDEFGH12345678", "azure-key-ZZZZZZZZZZ999",
           "session=cookiecookie123456", "AIzaSyDUMMYDUMMYDUMMYDUMMYDUMMY12345"]


def _all_file_bytes(d: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file())


def test_f_credentials_never_persisted(mk, tmp_path):
    def fn(h, rec):
        # hostile / sloppy upstream that echoes the credentials back in headers AND body
        hd = rec["headers"]
        echo = " ".join([hd.get("x-api-key", ""), hd.get("authorization", ""), hd.get("api-key", "")])
        send(h, 200, {"Content-Type": "application/json", "x-echo-auth": hd.get("authorization", ""),
                      "x-echo-key": hd.get("x-api-key", "")},
             anth_json(f"you sent {echo}"))
    cas = tmp_path / "cas" / "c.jsonl"
    tap, _ = mk(Up(fn), cache=True, cassette=str(cas), db_path=str(tmp_path / "db" / "t.db"),
                guard="warn")
    hdrs = {"x-api-key": SECRETS[0], "Authorization": SECRETS[1], "api-key": SECRETS[2], "Cookie": SECRETS[3],
            "anthropic-version": "2023-06-01"}
    st, _, _ = call(tap, "POST", f"/v1/messages?key={SECRETS[4]}&api_version=1", msg(), hdrs)
    assert st == 200
    last(tap)
    tap.store.close()
    tap.cache.close()
    blob = _all_file_bytes(tmp_path)
    assert len(blob) > 500
    for s in SECRETS:
        raw = s.split(" ", 1)[-1].replace("Bearer ", "").replace("session=", "")
        assert raw.encode() not in blob, f"credential leaked to disk: {s}"


def test_f_dashboard_host_header_allowlist(mk):
    tap, _ = mk()
    for bad in ("evil.example.com", "evil.example.com:4141", "localhost.evil.com", "127.0.0.1.evil.com:80"):
        for path in ("/_llmtap/", "/_llmtap/api/stats", "/_llmtap/health", "/_llmtap/api/exchanges"):
            st, _, out = call(tap, "GET", path, None, {"Host": bad})
            assert st == 403, (bad, path, st)
    for good in (f"127.0.0.1:{tap.port}", f"localhost:{tap.port}", "localhost", f"[::1]:{tap.port}", "LOCALHOST:1"):
        st, _, _ = call(tap, "GET", "/_llmtap/health", None, {"Host": good})
        assert st == 200, good
    # a rebinding attacker's host must not be able to reach proxied non-internal routes' dashboard either
    st, _, _ = call(tap, "GET", "/_llmtap/api/events", None, {"Host": "evil.example.com"})
    assert st == 403


def test_f_dashboard_host_allows_configured_host(mk):
    tap, _ = mk()
    tap.cfg.host = "myhost.lan"
    st, _, _ = call(tap, "GET", "/_llmtap/health", None, {"Host": "myhost.lan:4141"})
    assert st == 200


def test_f_clear_requires_same_origin(mk):
    tap, up = mk(llm_up())
    call(tap, "POST", "/v1/messages", msg(), ANTH_H)
    last(tap)
    assert tap.store.count() == 1
    me = f"127.0.0.1:{tap.port}"
    for origin in ("http://evil.example.com", "https://localhost:9999", "null", f"http://127.0.0.1:{tap.port + 1}"):
        st, _, _ = call(tap, "POST", "/_llmtap/api/clear", b"", {"Origin": origin, "Host": me})
        assert st == 403, origin
    st, _, _ = call(tap, "POST", "/_llmtap/api/clear", b"", {"Sec-Fetch-Site": "cross-site"})
    assert st == 403
    assert tap.store.count() == 1
    st, _, _ = call(tap, "POST", "/_llmtap/api/clear", b"", {"Origin": f"http://{me}"})
    assert st == 200 and tap.store.count() == 0
    call(tap, "POST", "/v1/messages", msg("again"), ANTH_H)
    last(tap)
    st, _, _ = call(tap, "POST", "/_llmtap/api/clear", b"")     # curl: no Origin
    assert st == 200


def test_f_no_cors_on_internal_routes(mk):
    tap, _ = mk()
    for path in ("/_llmtap/health", "/_llmtap/api/stats", "/_llmtap/", "/_llmtap/api/exchanges", "/_llmtap/nope"):
        st, h, _ = call(tap, "GET", path)
        assert not any(k.startswith("access-control-") for k in h), (path, h)
    st, h, _ = call(tap, "OPTIONS", "/_llmtap/api/clear", None,
                    {"Origin": "http://evil.example.com", "Access-Control-Request-Method": "POST"})
    assert not any(k.startswith("access-control-allow") for k in h)


def test_f_is_exposed():
    from llmtap.proxy import LLMTap as T
    assert T(Config(host="127.0.0.1", port=0, db_path=":memory:")).is_exposed is False
    assert T(Config(host="localhost", port=0, db_path=":memory:")).is_exposed is False
    assert T(Config(host="::1", port=0, db_path=":memory:")).is_exposed is False
    assert T(Config(host="0.0.0.0", port=0, db_path=":memory:")).is_exposed is True
    assert T(Config(host="::", port=0, db_path=":memory:")).is_exposed is True
    assert T(Config(host="192.168.1.20", port=0, db_path=":memory:")).is_exposed is True


def test_f_exposed_bind_works_and_warns(capfd):
    cfg = Config(host="0.0.0.0", port=0, db_path=":memory:", quiet=True)
    t = LLMTap(cfg).start_background()
    try:
        st, _, _ = call(t, "GET", "/_llmtap/health")
        assert st == 200
        # an IP-literal Host (what a LAN user types) works on an exposed bind; hostnames still need to match
        st, _, _ = call(t, "GET", "/_llmtap/health", None, {"Host": f"192.168.1.5:{t.port}"})
        assert st == 200
    finally:
        t.shutdown()
    err = capfd.readouterr().err.lower()
    assert "0.0.0.0" in err and ("exposed" in err or "warning" in err)


# ================================================================================================
# G. concurrency
# ================================================================================================
def test_g_100_concurrent_mixed(mock_upstream, tmp_path):
    cfg = Config(port=0, quiet=True, db_path=str(tmp_path / "c.db"), cache=True, guard="warn",
                 cassette=str(tmp_path / "c.jsonl"), mode="record")
    cfg.upstreams["anthropic"] = {"base": mock_upstream.base_url, "style": "anthropic"}
    cfg.upstreams["openai"] = {"base": mock_upstream.base_url, "style": "openai"}
    tap = LLMTap(cfg).start_background()
    results: list[tuple[int, str, str]] = [(0, "", "")] * 100
    errors: list[str] = []

    def worker(i: int) -> None:
        try:
            stream = i % 2 == 0
            if i % 4 < 2:
                st, _, out = call(tap, "POST", "/v1/messages", msg(f"prompt {i}", stream=stream), ANTH_H, timeout=60)
            else:
                st, _, out = call(tap, "POST", "/v1/chat/completions",
                                  {"model": "gpt-4o-mini", "stream": stream,
                                   "messages": [{"role": "user", "content": f"prompt {i}"}]}, None, timeout=60)
            results[i] = (st, f"prompt {i}", out.decode())
        except Exception as e:  # noqa: BLE001
            errors.append(f"{i}: {e!r}")

    try:
        ts = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        [t.start() for t in ts]
        [t.join(60) for t in ts]
        assert not errors, errors[:3]
        for i, (st, want, out) in enumerate(results):
            assert st == 200, (i, st, out[:200])
            if i % 2:   # non-stream: whole reply in one JSON body; stream: must be a complete, terminated stream
                assert f"Echo: {want}" in out
            else:
                assert "message_stop" in out or "[DONE]" in out
        deadline = time.time() + 10
        while tap.store.count() < 100 and time.time() < deadline:
            time.sleep(0.05)
        assert tap.store.count() == 100
        assert all(e.status == 200 and not (e.error or "") for e in tap.store.list(limit=100))
        assert tap.cache.stats()["entries"] == 100
    finally:
        tap.shutdown()


def test_g_cache_stampede_is_consistent(mk):
    def fn(h, rec):
        time.sleep(0.2)
        send(h, 200, {"Content-Type": "application/json"}, anth_json("same"))
    tap, up = mk(Up(fn), cache=True)
    outs: list[bytes] = []

    def w() -> None:
        outs.append(call(tap, "POST", "/v1/messages", msg("stampede"), ANTH_H)[2])
    ts = [threading.Thread(target=w) for _ in range(20)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert len(outs) == 20 and all(json.loads(o)["content"][0]["text"] == "same" for o in outs)
    assert tap.cache.stats()["entries"] == 1            # documented: concurrent identical misses all hit upstream
    st, h, _ = call(tap, "POST", "/v1/messages", msg("stampede"), ANTH_H)
    assert h["x-llmtap-source"] == "cache"


# ================================================================================================
# H. misc
# ================================================================================================
def test_h_ipv6_loopback_bind():
    try:
        s = socket.socket(socket.AF_INET6)
        s.bind(("::1", 0))
        s.close()
    except OSError:
        pytest.skip("no IPv6 loopback in this environment")
    up = Up()
    cfg = Config(host="::1", port=0, db_path=":memory:", quiet=True)
    cfg.upstreams["openai"] = {"base": up.base_url, "style": "openai"}
    t = LLMTap(cfg).start_background()
    try:
        assert t.base_url.startswith("http://[::1]:")
        c = http.client.HTTPConnection("::1", t.port, timeout=5)
        c.request("GET", "/_llmtap/health", headers={"Host": f"[::1]:{t.port}"})
        assert c.getresponse().status == 200
        assert t.is_exposed is False
    finally:
        t.shutdown()
        up.stop()


def test_h_unknown_paths_forwarded_openai_style(mk):
    tap, up = mk()
    call(tap, "GET", "/models")
    call(tap, "POST", "/some/thing", {"a": 1})
    assert up.requests[0]["path"] == "/v1/models"
    assert up.requests[1]["path"] == "/v1/some/thing"
    last(tap, 2)
    assert tap.store.count() == 2


def test_h_invalid_json_with_stream_still_works(mk):
    tap, up = mk(cache=True)
    bad = b'{"model": "gpt-4o", "stream": true, '
    st, _, _ = call(tap, "POST", "/v1/chat/completions", bad, {"Content-Type": "application/json",
                                                              "Accept": "text/event-stream"})
    assert st == 200 and up.requests[0]["raw"] == bad
    assert last(tap).request_body == bad.decode()


def test_h_header_casing(mk):
    tap, up = mk()
    st, _, _ = call(tap, "POST", "/v1/anything", b"{}", {"X-API-KEY": "k", "X-LLMTAP-TAG-Team": "a",
                                                         "x-LlMtAp-SeSsIoN": "S"})
    assert st == 200
    ex = last(tap)
    assert ex.provider == "anthropic" and ex.tags == {"team": "a", "session": "S"}


def test_h_head_root_and_options(mk):
    tap, up = mk()
    st, h, _ = call(tap, "HEAD", "/")
    assert st == 302 and h["location"] == "/_llmtap/"
    st, h, _ = call(tap, "OPTIONS", "/v1/messages", None, {"Origin": "http://app.example", "Access-Control-Request-Method": "POST",
                                                          "Access-Control-Request-Headers": "x-api-key"})
    assert st == 204 and h["access-control-allow-origin"] == "*"
    assert not up.requests
    st, _, _ = call(tap, "OPTIONS", "/v1/messages")          # plain OPTIONS is forwarded
    assert st == 200 and up.requests[-1]["method"] == "OPTIONS"
