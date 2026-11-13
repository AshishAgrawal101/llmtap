"""Tests for llmtap.proxy against a self-contained fake upstream (no network, no llmtap.mock)."""
from __future__ import annotations

import gzip
import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from llmtap.config import Config
from llmtap.proxy import EventBus, LLMTap

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"


# ------------------------------------------------------------------------------------------------
# fake upstream
# ------------------------------------------------------------------------------------------------
class FakeUpstream:
    """Tiny Anthropic + OpenAI emulator. Magic words in the prompt: GATED, GZIP, MANY, FAIL500."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.gate = threading.Event()
        self.first_sent = threading.Event()
        self.aborted = threading.Event()
        up = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n)
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                up.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                    "body": body, "raw": raw})
                text = json.dumps(body)
                if "FAIL500" in text:
                    return self._json(500, {"error": {"message": "boom", "type": "server_error"}})
                if self.path.endswith("/v1/messages"):
                    return self._anthropic(body, text)
                if self.path.endswith("/chat/completions"):
                    return self._openai(body, text)
                self._json(404, {"error": {"message": "nope"}})

            def _json(self, status: int, obj: Any, gz: bool = False) -> None:
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("request-id", "req_fake")
                if gz:
                    data = gzip.compress(data)
                    self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _sse(self, events: list[str], text: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    for i, e in enumerate(events):
                        b = e.encode()
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
                        self.wfile.flush()
                        if i == 0:
                            up.first_sent.set()
                            if "GATED" in text:
                                up.gate.wait(10)
                        if "MANY" in text:
                            time.sleep(0.01)
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError:
                    up.aborted.set()
                    self.close_connection = True

            def _anthropic(self, body: dict, text: str) -> None:
                model = body.get("model", "claude-sonnet-4-5")
                if not body.get("stream"):
                    return self._json(200, {
                        "id": "msg_1", "type": "message", "role": "assistant", "model": model,
                        "content": [{"type": "text", "text": "Hello from fake"}],
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    }, gz="GZIP" in text)
                ev = lambda name, obj: f"event: {name}\ndata: {json.dumps(obj)}\n\n"  # noqa: E731
                events = [ev("message_start", {"type": "message_start", "message": {
                    "id": "msg_2", "model": model, "usage": {"input_tokens": 12, "output_tokens": 1}}})]
                words = ["Hel", "lo", " stream"] * (100 if "MANY" in text else 1)
                for w in words:
                    events.append(ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                             "delta": {"type": "text_delta", "text": w}}))
                events.append(ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                                                   "usage": {"output_tokens": 7}}))
                events.append(ev("message_stop", {"type": "message_stop"}))
                self._sse(events, text)

            def _openai(self, body: dict, text: str) -> None:
                model = body.get("model", "gpt-4o-mini")
                if not body.get("stream"):
                    return self._json(200, {
                        "id": "chatcmpl-1", "object": "chat.completion", "model": model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi from openai"},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
                    }, gz="GZIP" in text)
                chunk = lambda d: "data: " + json.dumps(d) + "\n\n"  # noqa: E731
                events = [chunk({"model": model, "choices": [{"index": 0, "delta": {"content": p}}]})
                          for p in ["Hi", " there"]]
                events.append(chunk({"model": model, "choices": [],
                                     "usage": {"prompt_tokens": 9, "completion_tokens": 2}}))
                events.append("data: [DONE]\n\n")
                self._sse(events, text)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ------------------------------------------------------------------------------------------------
# fixtures + helpers
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def upstream():
    up = FakeUpstream()
    yield up
    up.gate.set()
    up.stop()


def make_cfg(base: str, **kw: Any) -> Config:
    kw.setdefault("guard", "off")
    cfg = Config(port=0, db_path=":memory:", quiet=True, **kw)
    cfg.upstreams["anthropic"]["base"] = base
    cfg.upstreams["openai"]["base"] = base
    cfg.upstreams["custom"] = {"base": base + "/api", "style": "openai"}
    cfg.upstream_timeout_s = 10
    return cfg


@pytest.fixture
def tapf(upstream):
    taps: list[LLMTap] = []

    def make(**kw: Any) -> LLMTap:
        tap = LLMTap(make_cfg(upstream.base_url, **kw)).start_background()
        taps.append(tap)
        return tap

    yield make
    for t in taps:
        t.shutdown()


def call(tap: LLMTap, path: str, obj: Any = None, headers: dict[str, str] | None = None,
         method: str = "POST") -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", tap.port, timeout=10)
    body = json.dumps(obj).encode() if obj is not None else None
    h = {"Content-Type": "application/json", **(headers or {})}
    conn.request(method, path, body=body, headers=h)
    r = conn.getresponse()
    data = r.read()
    hdrs = {k.lower(): v for k, v in r.getheaders()}
    conn.close()
    return r.status, hdrs, data


def wait_for(tap: LLMTap, ex_id: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        ex = tap.store.get(ex_id)
        if ex is not None:
            return ex
        time.sleep(0.02)
    raise AssertionError(f"exchange {ex_id} never recorded")


def anth(text: str = "hello", stream: bool = False, model: str = "claude-sonnet-4-5") -> dict:
    d: dict[str, Any] = {"model": model, "max_tokens": 50, "messages": [{"role": "user", "content": text}]}
    if stream:
        d["stream"] = True
    return d


def oai(text: str = "hello", stream: bool = False, model: str = "gpt-4o-mini") -> dict:
    d: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": text}]}
    if stream:
        d["stream"] = True
        d["stream_options"] = {"include_usage": True}
    return d


ANTH_H = {"x-api-key": "test", "anthropic-version": "2023-06-01"}


# ------------------------------------------------------------------------------------------------
# tests
# ------------------------------------------------------------------------------------------------
def test_anthropic_non_stream_recorded(tapf, upstream):
    tap = tapf()
    st, h, body = call(tap, "/v1/messages", anth(), {**ANTH_H, "X-LLMTap-Session": "s1"})
    assert st == 200
    assert json.loads(body)["content"][0]["text"] == "Hello from fake"
    assert h["x-llmtap-source"] == "upstream"
    assert h["access-control-allow-origin"] == "*"
    assert h["request-id"] == "req_fake"
    ex = wait_for(tap, h["x-llmtap-id"])
    assert ex.provider == "anthropic" and ex.source == "upstream" and ex.status == 200
    assert ex.usage.input_tokens == 10 and ex.usage.output_tokens == 5 and not ex.usage.estimated
    assert ex.cost_usd > 0
    assert ex.output_text == "Hello from fake"
    assert ex.model == "claude-sonnet-4-5"
    assert ex.tags == {"session": "s1"}
    assert ex.latency_ms > 0
    # forwarded headers: no x-llmtap-*, identity encoding, api key kept
    fh = upstream.requests[0]["headers"]
    assert fh["accept-encoding"] == "identity"
    assert fh["x-api-key"] == "test"
    assert not any(k.startswith("x-llmtap") for k in fh)


def test_openai_non_stream_recorded(tapf, upstream):
    tap = tapf()
    st, h, body = call(tap, "/v1/chat/completions", oai(), {"Authorization": "Bearer sk-test"})
    assert st == 200
    ex = wait_for(tap, h["x-llmtap-id"])
    assert ex.provider == "openai"
    assert ex.usage.input_tokens == 20 and ex.usage.output_tokens == 4
    assert ex.cost_usd > 0 and ex.output_text == "Hi from openai"
    assert upstream.requests[0]["path"] == "/v1/chat/completions"


def test_path_without_v1_and_base_path(tapf, upstream):
    tap = tapf()
    assert call(tap, "/chat/completions", oai())[0] == 200
    assert upstream.requests[-1]["path"] == "/v1/chat/completions"
    st, h, _ = call(tap, "/custom/v1/chat/completions", oai())
    assert st == 200
    assert upstream.requests[-1]["path"] == "/api/v1/chat/completions"
    ex = wait_for(tap, h["x-llmtap-id"])
    assert ex.provider == "custom" and ex.status == 200 and ex.output_text == "Hi from openai"
    st, h, _ = call(tap, "/v1/embeddings", {"model": "x", "input": "y"})
    assert st == 404  # upstream status passed through verbatim
    assert wait_for(tap, h["x-llmtap-id"]).error == "nope"


def _read_event(resp: http.client.HTTPResponse) -> str:
    lines = []
    while True:
        line = resp.readline()
        if not line:
            break
        lines.append(line.decode())
        if line in (b"\n", b"\r\n"):
            break
    return "".join(lines)


def test_anthropic_stream_is_incremental(tapf, upstream):
    tap = tapf()
    conn = http.client.HTTPConnection("127.0.0.1", tap.port, timeout=10)
    conn.request("POST", "/v1/messages", body=json.dumps(anth("GATED", stream=True)), headers=ANTH_H)
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("content-type") == "text/event-stream"
    assert resp.getheader("transfer-encoding") == "chunked"
    first = _read_event(resp)  # must arrive while upstream is still blocked on the gate
    assert "message_start" in first
    assert not upstream.gate.is_set()
    upstream.gate.set()
    rest = resp.read().decode()
    assert "message_stop" in rest
    ex_id = resp.getheader("x-llmtap-id")
    conn.close()
    ex = wait_for(tap, ex_id)
    assert ex.streamed and ex.source == "upstream"
    assert ex.output_text == "Hello stream"
    assert ex.usage.input_tokens == 12 and ex.usage.output_tokens == 7
    assert ex.ttft_ms is not None and ex.ttft_ms <= ex.latency_ms
    assert ex.cost_usd > 0
    assert "message_start" in ex.response_body and "message_stop" in ex.response_body


def test_openai_stream_parsed(tapf, upstream):
    tap = tapf()
    st, h, body = call(tap, "/v1/chat/completions", oai(stream=True))
    assert st == 200 and b"[DONE]" in body
    ex = wait_for(tap, h["x-llmtap-id"])
    assert ex.streamed and ex.output_text == "Hi there"
    assert ex.usage.input_tokens == 9 and ex.usage.output_tokens == 2


def test_client_disconnect_mid_stream(tapf, upstream):
    tap = tapf()
    q = tap.bus.subscribe()
    s = socket.create_connection(("127.0.0.1", tap.port), timeout=10)
    body = json.dumps(anth("GATED MANY", stream=True)).encode()
    s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"x-api-key: k\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
    got = b""
    while b"message_start" not in got:
        got += s.recv(65536)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
    s.close()  # RST
    upstream.gate.set()
    ev = q.get(timeout=10)
    ex = ev["exchange"]
    assert ex["error"] == "client disconnected"
    assert ex["streamed"] is True and ex["status"] == 200
    assert "request_body" not in ex and "response_body" not in ex
    assert upstream.aborted.wait(5)  # proxy stopped reading upstream -> upstream sees broken pipe


def test_cache_hit_second_time(tapf, upstream):
    tap = tapf(cache=True)
    st1, h1, b1 = call(tap, "/v1/messages", anth("cache me"), ANTH_H)
    st2, h2, b2 = call(tap, "/v1/messages", anth("cache me"), ANTH_H)
    assert st1 == st2 == 200 and b1 == b2
    assert h1["x-llmtap-source"] == "upstream" and h2["x-llmtap-source"] == "cache"
    assert len(upstream.requests) == 1
    ex1, ex2 = wait_for(tap, h1["x-llmtap-id"]), wait_for(tap, h2["x-llmtap-id"])
    assert ex2.source == "cache" and ex2.cost_usd == 0 and ex2.saved_usd == pytest.approx(ex1.cost_usd)
    assert ex2.usage.input_tokens == 10 and ex2.output_text == "Hello from fake"


def test_cache_streamed_replay(tapf, upstream):
    tap = tapf(cache=True)
    _, h1, b1 = call(tap, "/v1/messages", anth("s", stream=True), ANTH_H)
    st, h2, b2 = call(tap, "/v1/messages", anth("s", stream=True), ANTH_H)
    assert st == 200 and h2["x-llmtap-source"] == "cache"
    assert h2["content-type"].startswith("text/event-stream")
    assert h2["transfer-encoding"] == "chunked"
    assert b1.replace(b"\r\n", b"\n") == b2.replace(b"\r\n", b"\n")
    assert len(upstream.requests) == 1
    ex = wait_for(tap, h2["x-llmtap-id"])
    assert ex.streamed and ex.output_text == "Hello stream"


def test_cassette_record_then_replay_offline(tmp_path, upstream):
    path = str(tmp_path / "c.jsonl")
    cfg = make_cfg(upstream.base_url, cassette=path, mode="record")
    tap = LLMTap(cfg).start_background()
    try:
        _, _, b1 = call(tap, "/v1/messages", anth("record me"), ANTH_H)
        _, _, s1 = call(tap, "/v1/chat/completions", oai("stream me", stream=True))
    finally:
        tap.shutdown()
    upstream.stop()

    dead = f"http://127.0.0.1:{_closed_port()}"
    tap2 = LLMTap(make_cfg(dead, cassette=path, mode="replay")).start_background()
    try:
        st, h, b2 = call(tap2, "/v1/messages", anth("record me"), ANTH_H)
        assert st == 200 and b2 == b1 and h["x-llmtap-source"] == "cassette"
        st, h, s2 = call(tap2, "/v1/chat/completions", oai("stream me", stream=True))
        assert st == 200 and h["x-llmtap-source"] == "cassette" and b"[DONE]" in s2
        ex = wait_for(tap2, h["x-llmtap-id"])
        assert ex.source == "cassette" and ex.output_text == "Hi there" and ex.saved_usd > 0
        # miss -> 409
        st, h, b = call(tap2, "/v1/messages", anth("never recorded"), ANTH_H)
        assert st == 409
        err = json.loads(b)
        assert err["type"] == "error" and "cassette miss" in err["error"]["message"]
    finally:
        tap2.shutdown()
    upstream.stop = lambda: None  # type: ignore[method-assign]  # already stopped


def test_guard_block(tapf, upstream):
    tap = tapf(guard="block")
    st, h, b = call(tap, "/v1/messages", anth(f"my key is {AWS_KEY}"), ANTH_H)
    assert st == 400
    assert json.loads(b)["error"]["type"] == "llmtap_guard_blocked"
    assert h["x-llmtap-source"] == "blocked"
    assert upstream.requests == []
    ex = wait_for(tap, h["x-llmtap-id"])
    assert ex.source == "blocked" and ex.guard_findings and ex.guard_findings[0]["kind"] == "aws_access_key"


def test_guard_redact(tapf, upstream):
    tap = tapf(guard="redact")
    st, h, _ = call(tap, "/v1/chat/completions", oai(f"my key is {AWS_KEY}"))
    assert st == 200
    raw = upstream.requests[0]["raw"].decode()
    assert AWS_KEY not in raw and "[REDACTED:aws_access_key]" in raw
    ex = wait_for(tap, h["x-llmtap-id"])
    assert AWS_KEY not in ex.request_body and ex.guard_findings


def test_loop_breaker_429(tapf, upstream):
    tap = tapf(loop_threshold=2)
    codes = [call(tap, "/v1/chat/completions", oai("again"))[0] for _ in range(3)]
    assert codes == [200, 200, 429]
    st, h, b = call(tap, "/v1/chat/completions", oai("again"))
    assert st == 429 and h["x-llmtap-source"] == "loop"
    assert json.loads(b)["error"]["code"] == "llmtap_loop_detected"
    assert len(upstream.requests) == 2


def test_session_budget_429(tapf, upstream):
    tap = tapf(budget_session_usd=0.0000001)
    assert call(tap, "/v1/messages", anth("one"), ANTH_H)[0] == 200
    st, h, _ = call(tap, "/v1/messages", anth("two"), ANTH_H)
    assert st == 429 and h["x-llmtap-source"] == "budget"
    assert wait_for(tap, h["x-llmtap-id"]).source == "budget"


def test_upstream_down_502():
    cfg = make_cfg(f"http://127.0.0.1:{_closed_port()}")
    tap = LLMTap(cfg).start_background()
    try:
        st, h, b = call(tap, "/v1/chat/completions", oai())
        assert st == 502 and h["x-llmtap-source"] == "error"
        assert json.loads(b)["error"]["type"] == "llmtap_upstream_error"
        ex = wait_for(tap, h["x-llmtap-id"])
        assert ex.source == "error" and ex.status == 502 and ex.error
    finally:
        tap.shutdown()


def test_upstream_error_status_passthrough(tapf, upstream):
    tap = tapf(cache=True)
    st, h, b = call(tap, "/v1/messages", anth("FAIL500"), ANTH_H)
    assert st == 500 and json.loads(b)["error"]["message"] == "boom"
    ex = wait_for(tap, h["x-llmtap-id"])
    assert ex.error == "boom" and ex.cost_usd == 0
    call(tap, "/v1/messages", anth("FAIL500"), ANTH_H)
    assert len(upstream.requests) == 2  # errors are never cached


def test_gzip_upstream_relayed_and_parsed(tapf, upstream):
    tap = tapf()
    conn = http.client.HTTPConnection("127.0.0.1", tap.port, timeout=10)
    conn.request("POST", "/v1/chat/completions", body=json.dumps(oai("GZIP")))
    r = conn.getresponse()
    raw = r.read()
    assert r.getheader("content-encoding") == "gzip"
    assert json.loads(gzip.decompress(raw))["choices"][0]["message"]["content"] == "Hi from openai"
    ex = wait_for(tap, r.getheader("x-llmtap-id"))
    assert ex.output_text == "Hi from openai" and ex.usage.input_tokens == 20


def test_chunked_request_body_and_keepalive(tapf, upstream):
    tap = tapf()
    conn = http.client.HTTPConnection("127.0.0.1", tap.port, timeout=10)
    payload = json.dumps(oai("chunked body")).encode()
    conn.request("POST", "/v1/chat/completions", body=iter([payload[:10], payload[10:]]),
                 headers={"Transfer-Encoding": "chunked"}, encode_chunked=True)
    r = conn.getresponse()
    assert r.status == 200
    r.read()
    assert upstream.requests[-1]["body"]["messages"][0]["content"] == "chunked body"
    # same connection again (keep-alive), streaming this time
    conn.request("POST", "/v1/chat/completions", body=json.dumps(oai("again", stream=True)))
    r2 = conn.getresponse()
    assert r2.status == 200 and b"[DONE]" in r2.read()
    conn.request("POST", "/v1/messages", body=json.dumps(anth("third")), headers=ANTH_H)
    assert conn.getresponse().status == 200
    conn.close()


def test_options_preflight_and_root_redirect(tapf, upstream):
    tap = tapf()
    st, h, _ = call(tap, "/v1/messages", None, {"Origin": "http://x", "Access-Control-Request-Method": "POST",
                                                 "Access-Control-Request-Headers": "x-api-key, content-type"},
                    method="OPTIONS")
    assert st == 204
    assert h["access-control-allow-origin"] == "*"
    assert "POST" in h["access-control-allow-methods"]
    assert "x-api-key" in h["access-control-allow-headers"]
    assert upstream.requests == []
    st, h, _ = call(tap, "/", None, method="GET")
    assert st == 302 and h["location"] == "/_llmtap/"
    assert tap.store.count() == 0


def test_internal_paths_delegated(tapf, upstream):
    pytest.importorskip("llmtap.dashboard")
    tap = tapf()
    st, h, b = call(tap, "/_llmtap/health", None, method="GET")
    assert st == 200 and json.loads(b)["ok"] is True
    st, _, _ = call(tap, "/_llmtap/definitely-not-a-route", None, method="GET")
    assert st == 404
    assert upstream.requests == [] and tap.store.count() == 0


def test_bus_and_handle_direct(upstream):
    """handle() works with any Responder (no http.server involved)."""
    tap = LLMTap(make_cfg(upstream.base_url))

    class Rec:
        def __init__(self) -> None:
            self.status = 0
            self.headers: dict[str, str] = {}
            self.chunks: list[bytes] = []
            self.ended = False

        def start(self, status: int, headers: dict[str, str]) -> None:
            self.status, self.headers = status, headers

        def write(self, chunk: bytes) -> None:
            self.chunks.append(chunk)

        def end(self) -> None:
            self.ended = True

    q = tap.bus.subscribe()
    r = Rec()
    ex = tap.handle("POST", "/v1/messages", ANTH_H, json.dumps(anth("direct", stream=True)).encode(), r)
    assert r.status == 200 and r.ended and len(r.chunks) >= 1
    assert r.headers["x-llmtap-id"] == ex.id
    assert ex.output_text == "Hello stream"
    evt = q.get(timeout=2)
    assert evt["type"] == "exchange" and evt["exchange"]["id"] == ex.id
    tap.bus.unsubscribe(q)
    tap.shutdown()


def test_event_bus_fanout():
    bus = EventBus()
    a, b = bus.subscribe(), bus.subscribe()
    bus.publish({"x": 1})
    assert a.get_nowait() == {"x": 1} and b.get_nowait() == {"x": 1}
    bus.unsubscribe(a)
    bus.unsubscribe(a)  # idempotent
    bus.publish({"x": 2})
    assert a.empty() and b.get_nowait() == {"x": 2}


def test_start_background_port_and_concurrency(tapf, upstream):
    tap = tapf()
    assert tap.port > 0 and tap.base_url == f"http://127.0.0.1:{tap.port}"
    results: list[int] = []

    def worker(i: int) -> None:
        results.append(call(tap, "/v1/chat/completions", oai(f"c{i}"))[0])

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert results == [200] * 12
    deadline = time.time() + 5
    while tap.store.count() < 12 and time.time() < deadline:
        time.sleep(0.02)
    assert tap.store.count() == 12
