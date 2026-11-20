"""Regressions for an independent security review of the key vault: each test is an attack that used to work."""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from llmtap.proxy import _StreamScrub
from llmtap.vault import FileBackend, MemoryBackend, Vault, VaultError, base_allows
from test_vault import ANTHROPIC_KEY, GEMINI_KEY, OPENAI_KEY, _call, _everything_stored, _wait_rows


def _raw(t: Any, data: bytes) -> bytes:
    s = socket.create_connection(("127.0.0.1", t.port), timeout=5)
    s.sendall(data)
    out = b""
    try:
        while True:
            c = s.recv(65536)
            if not c:
                break
            out += c
    except socket.timeout:
        pass
    s.close()
    return out


class _Resp:
    def __init__(self) -> None:
        self.status, self.headers, self.body, self.started = 0, {}, b"", False

    def start(self, status: int, headers: dict) -> None:
        self.status, self.headers, self.started = status, headers, True

    def write(self, chunk: bytes) -> None:
        self.body += chunk

    def end(self) -> None:
        pass


# 1. a web page on another site must not be able to spend (or read answers from) a stored key
def test_cross_origin_page_cannot_use_placeholder(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    t = vtap()
    c = http.client.HTTPConnection("127.0.0.1", t.port, timeout=10)
    c.request("POST", "/v1/chat/completions",
              body=json.dumps({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}),
              headers={"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site",
                       "Authorization": "Bearer llmtap-key-openai", "Content-Type": "application/json"})
    r = c.getresponse()
    body = r.read()
    assert r.status == 403 and b"llmtap_key_cross_origin" in body
    assert r.getheader("access-control-allow-origin") is None
    assert not mock_upstream.requests


@pytest.mark.parametrize("headers", [
    {"Origin": "https://evil.example"},
    {"Origin": "null"},
    {"Sec-Fetch-Site": "same-site"},
    {"Host": "attacker.example:4141"},
])
def test_browserish_or_rebinding_requests_refused(vtap: Any, vault: Vault, mock_upstream: Any,
                                                  headers: dict) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    t = vtap()
    status, _ = _call(t, "/v1/chat/completions",
                      {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]},
                      {"Authorization": "Bearer llmtap-key-openai", **headers})
    assert status == 403 and not mock_upstream.requests


def test_remote_client_refused(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    t = vtap()
    r = _Resp()
    body = json.dumps({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}).encode()
    t.handle("POST", "/v1/chat/completions", {"Authorization": "Bearer llmtap-key-openai",
                                              "Content-Type": "application/json"}, body, r, client="192.168.1.50")
    assert r.status == 403 and b"llmtap_key_remote_client" in r.body and not mock_upstream.requests
    r = _Resp()
    t.handle("POST", "/v1/chat/completions", {"Authorization": "Bearer llmtap-key-openai",
                                              "Content-Type": "application/json"}, body, r, client="::1")
    assert r.status == 200


def test_normal_requests_keep_cors(vtap: Any) -> None:
    t = vtap()
    c = http.client.HTTPConnection("127.0.0.1", t.port, timeout=10)
    c.request("POST", "/v1/chat/completions",
              body=json.dumps({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]}),
              headers={"Authorization": "Bearer " + OPENAI_KEY, "Content-Type": "application/json"})
    r = c.getresponse()
    r.read()
    assert r.status == 200 and r.getheader("access-control-allow-origin") == "*"


# 2. an http.client error quoting the URL must not hand the swapped-in Gemini key to the client
def test_control_characters_in_url_do_not_leak_gemini_key(vtap: Any, vault: Vault) -> None:
    vault.add("gem", "gemini", GEMINI_KEY)
    t = vtap()
    body = b'{"contents":[{"parts":[{"text":"hi"}]}]}'
    resp = _raw(t, b"POST /v1beta/models/gemini-2.5-flash:generateContent?key=llmtap-key-gem&x=\x7f HTTP/1.1\r\n"
                   b"Host: 127.0.0.1\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                   + str(len(body)).encode() + b"\r\n\r\n" + body)
    assert GEMINI_KEY.encode() not in resp
    assert b" 400 " in resp.split(b"\r\n", 1)[0]


def test_error_messages_are_scrubbed(vtap: Any, vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    t = vtap()

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("upstream said: Bearer " + OPENAI_KEY)
    monkeypatch.setattr(t, "_forward", boom)
    status, body = _call(t, "/v1/chat/completions", {"model": "gpt-4o-mini", "messages": []},
                         {"Authorization": "Bearer llmtap-key-openai"})
    assert status == 500 and OPENAI_KEY not in json.dumps(body)


# 3. remove + re-add from another process must never serve the old (cached) secret
def test_no_stale_secret_after_remove_and_readd(tmp_path: Path) -> None:
    home = tmp_path / "h"
    proxy_side = Vault(home=home, backend=FileBackend(home / "s.json"))
    cli_side = Vault(home=home, backend=FileBackend(home / "s.json"))
    cli_side.add("work", "openai", OPENAI_KEY)
    assert proxy_side.secret("work") == OPENAI_KEY
    cli_side.remove("work")
    cli_side.add("work", "anthropic", ANTHROPIC_KEY)
    info = proxy_side.get("work")
    assert info.provider == "anthropic" and proxy_side.secret("work") == ANTHROPIC_KEY


# 4. an upstream echoing the key must not pass it to the client (headers, body, streams)
class _Echo(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length") or 0))
        a = self.headers.get("authorization", "")
        assert self.headers.get("accept-encoding") == "identity"
        b = json.dumps({"error": {"message": "Incorrect API key provided: " + a}}).encode()
        self.send_response(401)
        self.send_header("content-type", "application/json")
        self.send_header("x-echo", a)
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a: Any) -> None:
        pass


@pytest.fixture()
def echo() -> Iterator[str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def test_upstream_echo_is_scrubbed(vtap: Any, vault: Vault, echo: str) -> None:
    vault.add("groq", "groq", OPENAI_KEY, base=echo)
    t = vtap(upstreams={"gemini": {"base": echo, "style": "gemini"}, "groq": {"base": echo, "style": "openai"}})
    c = http.client.HTTPConnection("127.0.0.1", t.port, timeout=10)
    c.request("POST", "/groq/v1/chat/completions", body=json.dumps({"model": "llama", "messages": []}),
              headers={"Authorization": "Bearer llmtap-key-groq", "Content-Type": "application/json"})
    r = c.getresponse()
    b = r.read()
    assert r.status == 401
    assert OPENAI_KEY.encode() not in b and OPENAI_KEY not in (r.getheader("x-echo") or "")
    assert b"[REDACTED]" in b
    _wait_rows(t, 1)
    assert OPENAI_KEY not in _everything_stored(t)


def test_stream_scrubber_handles_split_secrets() -> None:
    s = b"SECRETKEY123"
    sc = _StreamScrub([s])
    data = b"aaa SECRET" + b"KEY123 bbb SECR" + b"ETKEY123" + b" end SEC"
    out = b"".join(sc.feed(data[i:i + 3]) for i in range(0, len(data), 3)) + sc.flush()
    assert out == b"aaa [REDACTED] bbb [REDACTED] end SEC"
    sc = _StreamScrub([s])
    assert sc.feed(b"data: hello\n\n") == b"data: hello\n\n"      # nothing held back when no prefix matches


# 5. a key with limits must not reach endpoints llmtap can't check
def test_batches_refused_for_limited_key(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("anthropic", "anthropic", ANTHROPIC_KEY, models=["claude-haiku*"], daily_usd=1)
    t = vtap()
    s, b = _call(t, "/v1/messages/batches",
                 {"requests": [{"custom_id": "a", "params": {"model": "claude-opus-4-1", "max_tokens": 4096,
                                                             "messages": []}}]},
                 {"x-api-key": "llmtap-key-anthropic", "anthropic-version": "2023-06-01"})
    assert s == 403 and "llmtap_key_endpoint_not_allowed" in json.dumps(b)
    assert not mock_upstream.requests


def test_unlimited_key_reaches_any_endpoint(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("anthropic", "anthropic", ANTHROPIC_KEY)
    t = vtap()
    _call(t, "/v1/messages/batches", {"requests": []},
          {"x-api-key": "llmtap-key-anthropic", "anthropic-version": "2023-06-01"})
    assert mock_upstream.requests[-1]["headers"]["x-api-key"] == ANTHROPIC_KEY


def test_count_tokens_allowed_for_limited_key(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("anthropic", "anthropic", ANTHROPIC_KEY, models=["claude-haiku*"])
    t = vtap()
    _call(t, "/v1/messages/count_tokens", {"model": "claude-haiku-4-5", "messages": []},
          {"x-api-key": "llmtap-key-anthropic", "anthropic-version": "2023-06-01"})
    assert mock_upstream.requests and mock_upstream.requests[-1]["path"] == "/v1/messages/count_tokens"


# 6. Gemini: the URL names the model that runs; a "model" field in the body must not satisfy the allowlist
def test_gemini_body_model_cannot_spoof_allowlist(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("gem", "gemini", GEMINI_KEY, models=["gemini-2.5-flash"])
    t = vtap()
    s, _ = _call(t, "/v1beta/models/gemini-2.5-pro:generateContent?key=llmtap-key-gem",
                 {"model": "gemini-2.5-flash", "contents": [{"parts": [{"text": "hi"}]}]}, {})
    assert s == 403 and not mock_upstream.requests


# 7. a spend cap can't count models without a price: refuse them
def test_unpriced_model_refused_when_key_has_spend_cap(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("openai", "openai", OPENAI_KEY, daily_usd=5)
    t = vtap()
    s, b = _call(t, "/v1/chat/completions", {"model": "my-unknown-model", "messages": [{"role": "user", "content": "x"}]},
                 {"Authorization": "Bearer llmtap-key-openai"})
    assert s == 403 and "llmtap_key_unpriced_model" in json.dumps(b) and not mock_upstream.requests


# 8. a project's llmtap.toml redirecting a provider must not receive stored keys
def test_redirected_upstream_does_not_get_the_key(vtap: Any, vault: Vault, echo: str) -> None:
    vault.add("openai", "openai", OPENAI_KEY)               # bound to the mock (the "official" URL here)
    t = vtap(upstreams={"gemini": {"base": echo, "style": "gemini"}, "openai": {"base": echo, "style": "openai"}})
    s, b = _call(t, "/v1/chat/completions", {"model": "gpt-4o-mini", "messages": []},
                 {"Authorization": "Bearer llmtap-key-openai"})
    assert s == 403 and "llmtap_key_wrong_upstream" in json.dumps(b) and OPENAI_KEY not in json.dumps(b)


def test_builtin_providers_bind_to_official_api(tmp_path: Path) -> None:
    v = Vault(home=tmp_path, backend=MemoryBackend())
    assert v.add("openai", "openai", OPENAI_KEY).base == "https://api.openai.com"
    with pytest.raises(VaultError):
        v.add("gw", "my-gateway", OPENAI_KEY)                # custom upstream: URL required
    assert v.add("gw", "my-gateway", OPENAI_KEY, base="https://GW.example/api/").base == "https://gw.example"
    br = v.add("br", "bedrock", "ABSK" + "x" * 30)
    assert base_allows(br, "https://bedrock-runtime.eu-west-1.amazonaws.com")
    assert not base_allows(br, "https://bedrock-runtime.eu-west-1.amazonaws.com.evil.example")


# low-severity items
def test_readd_with_some_limits_keeps_the_others(vault: Vault) -> None:
    vault.add("openai", "openai", OPENAI_KEY, models=["gpt-4o*"], paused=True, expires=time.time() + 999)
    vault.add("openai", "openai", OPENAI_KEY + "2", daily_usd=3)
    k = vault.get("openai")
    assert k.daily_usd == 3 and k.models == ["gpt-4o*"] and k.paused and k.expires


def test_non_ascii_secret_rejected(vault: Vault) -> None:
    with pytest.raises(VaultError):
        vault.add("openai", "openai", "sk-tést0123456789abcdef")


def test_file_backend_leaves_no_temp_files(tmp_path: Path) -> None:
    be = FileBackend(tmp_path / "s.json")
    for i in range(3):
        be.set(f"k{i}", OPENAI_KEY)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.json"]
