"""Regressions for an independent review of the 0.4.0 egress checks: each test is an attack or false positive
that the first version got wrong."""
from __future__ import annotations

import base64
import gzip
import http.client
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from llmtap import alerts
from llmtap.egress import Tripwire, parse_env, secretish, server_tools
from llmtap.models import Exchange, Usage
from llmtap.statusline import statusline
from llmtap.store import Store
from llmtap.vault import Vault
from test_vault import OPENAI_KEY, _call, _everything_stored, _wait_rows

AWS_LIKE = "wJalrXUtnFEMI/K7MDENG+bPxRfiCY" + "zq9Rk2LmNpQ"         # has "/" and "+"
STRIPE = "rk_" + "test_" + "51Habc0123456789XYZqwerty"
ANTH = "sk-ant-" + "test" + "0123456789abcdefANTH"


def _msg(text: str) -> dict:
    return {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": text}]}


def _tw(tmp_path: Path, env: str) -> Tripwire:
    f = tmp_path / ".env"
    f.write_text(env)
    return Tripwire([f], home=tmp_path)


# 1. an agent that can reach llmtap's port must be able to be kept from changing keys or wiping history
def test_dashboard_writes_off(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("anthropic", "anthropic", ANTH)
    t = vtap(dashboard_writes=False)
    hdrs = {"x-llmtap-request": "dashboard", "Content-Type": "application/json"}
    c = http.client.HTTPConnection("127.0.0.1", t.port, timeout=10)
    c.request("POST", "/_llmtap/api/keys/anthropic", body=json.dumps({"allow_server_tools": True}), headers=hdrs)
    r = c.getresponse()
    assert r.status == 403 and b"dashboard_writes" in r.read()
    c.request("POST", "/_llmtap/api/keys", body=json.dumps({"name": "evil", "provider": "anthropic",
                                                             "secret": "sk-ant-attacker0123456789"}), headers=hdrs)
    r = c.getresponse()
    assert r.status == 403
    r.read()
    c.request("POST", "/_llmtap/api/clear", body=b"{}", headers=hdrs)
    r = c.getresponse()
    assert r.status == 403
    r.read()
    assert not vault.get("anthropic").allow_server_tools and vault.get("evil") is None


# 2. a file name an agent chose must never become code in the Windows notification
def test_notify_passes_text_out_of_band(monkeypatch: Any) -> None:
    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(alerts.sys, "platform", "win32")
    monkeypatch.setattr(alerts.shutil, "which", lambda name: "powershell.exe" if name == "powershell" else None)
    monkeypatch.setattr(alerts.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))

    class _Now:
        def __init__(self, target: Any, **kw: Any) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

    monkeypatch.setattr(alerts.threading, "Thread", _Now)
    evil = "STRIPE from .env.x’; Set-Content pwned.txt pwned; ’"
    alerts.notify("llmtap", evil)
    cmd, kw = calls[0]
    assert "pwned" not in " ".join(cmd) and "Set-Content" not in " ".join(cmd)
    assert kw["env"]["LLMTAP_NOTIFY_TEXT"].isascii() and "pwned" in kw["env"]["LLMTAP_NOTIFY_TEXT"]


# 3. a compressed body would skip every check
def test_compressed_body_refused(tap: Any, tmp_path: Path, mock_upstream: Any) -> None:
    env = tmp_path / ".env"
    env.write_text(f"STRIPE_SECRET={STRIPE}\n")
    t = tap(leak_files=str(env), max_request_kb=4)
    body = gzip.compress(json.dumps(_msg(STRIPE + "x" * 200_000)).encode())
    c = http.client.HTTPConnection("127.0.0.1", t.port, timeout=10)
    c.request("POST", "/v1/chat/completions", body=body,
              headers={"Content-Type": "application/json", "Content-Encoding": "gzip", "Authorization": "Bearer sk-x"})
    r = c.getresponse()
    assert r.status == 415 and b"llmtap_encoded_body" in r.read() and not mock_upstream.requests


# 4 + 11. bodies llmtap can't parse (BOM, bad UTF-8, multipart uploads) can't carry tools past a stored key
@pytest.mark.parametrize("raw", [
    b"\xef\xbb\xbf" + json.dumps({"model": "m", "tools": [{"type": "web_fetch_20260209", "name": "web_fetch"}]}).encode(),
    b'{"model":"m","messages":[{"role":"user","content":"\xff"}],"tools":[{"type":"web_search_20260209"}]}',
    # absurd nesting: unreadable on some platforms (refused), parsed on others (and the buried tool is seen)
    b'{"a":' * 50_000 + b'{"tools":[{"type":"web_search_20260209"}]}' + b"}" * 50_000,
])
def test_unreadable_bodies(raw: bytes) -> None:
    found = server_tools(raw)
    assert found is None or found                    # either unreadable (refused) or the tool is seen


def test_stored_key_refuses_multipart_upload(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    t = vtap()
    line = json.dumps({"custom_id": "1", "method": "POST", "url": "/v1/responses",
                       "body": {"model": "gpt-5", "tools": [{"type": "web_search"}], "input": "x"}})
    body = (b"--b\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--b\r\nContent-Disposition: "
            b"form-data; name=\"file\"; filename=\"x.jsonl\"\r\n\r\n" + line.encode() + b"\r\n--b--\r\n")
    c = http.client.HTTPConnection("127.0.0.1", t.port, timeout=10)
    c.request("POST", "/v1/files", body=body, headers={"Content-Type": "multipart/form-data; boundary=b",
                                                        "Authorization": "Bearer llmtap-key-openai"})
    r = c.getresponse()
    assert r.status == 403 and b"llmtap_server_tool_blocked" in r.read() and not mock_upstream.requests


# 5. encodings an agent would plausibly use
def test_more_encodings(tmp_path: Path) -> None:
    tw = _tw(tmp_path, f"AWS_SECRET_ACCESS_KEY={AWS_LIKE}\nSTRIPE_SECRET={STRIPE}\n")
    wrapped = base64.encodebytes(("A=1\nAWS_SECRET_ACCESS_KEY=" + AWS_LIKE + "\n" + "B=2\n" * 30).encode()).decode()
    assert "\n" in wrapped.strip()
    cases = [
        wrapped,                                                             # GNU base64, 76 columns
        ":".join(f"{b:02x}" for b in STRIPE.encode()),                       # xxd / colon hex
        "".join(f"\\x{b:02x}" for b in STRIPE.encode()),                     # \xNN escapes
        quote(AWS_LIKE),                                                     # urllib default safe='/'
        re.sub(r"%[0-9A-F]{2}", lambda m: m.group(0).lower(), quote(AWS_LIKE, safe="")),   # lowercase %2f
        base64.b64encode(STRIPE.encode("utf-16-le")).decode(),               # PowerShell
    ]
    for text in cases:
        assert tw.scan(json.dumps(_msg(f"data: {text}")).encode()), text[:60]


# 6. ordinary .env values must not be fingerprinted (users would turn the tripwire off)
def test_ordinary_env_values_ignored() -> None:
    env = """APP_NAME=Laravel
SESSION_DRIVER=database
QUEUE_CONNECTION=database
DB_PASSWORD=password
AUTH_PROVIDER=keycloak
SESSION_SECRET=keyboard cat
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=myapp.firebaseapp.com
DATABASE_URL=postgres://postgres:postgres@localhost:5432/app
MAIL_FROM_ADDRESS=hello@example.org
"""
    assert [k for k, v in parse_env(env) if secretish(k, v)] == []
    real = "SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T000/B000/XXXXabcd1234\nSENTRY_DSN=https://" \
           "0123456789abcdef0123@o1.ingest.sentry.io/1\nPGPASS=$2b$12$abcdefghijklmnopqrstuv\n"
    assert {k for k, v in parse_env(real) if secretish(k, v)} == {"SLACK_WEBHOOK_URL", "SENTRY_DSN", "PGPASS"}


def test_postgres_word_doesnt_block(tap: Any, tmp_path: Path, mock_upstream: Any) -> None:
    env = tmp_path / ".env"
    env.write_text("DATABASE_URL=postgres://postgres:postgres@localhost/app\nDB_PASSWORD=password\n")
    t = tap(leak_files=str(env))
    assert _call(t, "/v1/chat/completions", _msg("set up a postgres database, password auth"),
                 {"Authorization": "Bearer sk-x"})[0] == 200


# 7. a \u-escaped secret is found, and must not be stored either
def test_escaped_secret_not_stored(tap: Any, tmp_path: Path, mock_upstream: Any) -> None:
    env = tmp_path / ".env"
    env.write_text(f"STRIPE_SECRET={STRIPE}\n")
    t = tap(leak_files=str(env))
    raw = json.dumps(_msg("k=" + STRIPE)).replace("rk_", "\\u0072k_").encode()
    c = http.client.HTTPConnection("127.0.0.1", t.port, timeout=10)
    c.request("POST", "/v1/chat/completions", body=raw, headers={"Content-Type": "application/json",
                                                                 "Authorization": "Bearer sk-x"})
    r = c.getresponse()
    r.read()
    assert r.status == 403
    _wait_rows(t, 1)
    stored = _everything_stored(t)
    assert STRIPE not in json.loads(json.dumps(stored)) and "u0072k_test" not in stored


# 8. warn mode: a cassette written from a cache hit must not contain the secret
def test_cassette_from_cache_is_scrubbed(tap: Any, tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(f"STRIPE_SECRET={STRIPE}\n")
    from llmtap.cassette import Cassette
    t = tap(leak_files=str(env), leak_check="warn", cache=True)
    assert _call(t, "/v1/chat/completions", _msg("k " + STRIPE), {"Authorization": "Bearer sk-x"})[0] == 200
    tape = tmp_path / "tape.jsonl"
    t.cassette = Cassette(str(tape), "auto")          # cassette misses, the cache hits: recorded from the cache
    assert _call(t, "/v1/chat/completions", _msg("k " + STRIPE), {"Authorization": "Bearer sk-x"})[0] == 200
    assert tape.exists() and STRIPE not in tape.read_text(encoding="utf-8")


# 9. secrets in the URL or in headers
def test_secret_in_url_or_header(tap: Any, tmp_path: Path, mock_upstream: Any) -> None:
    env = tmp_path / ".env"
    env.write_text(f"STRIPE_SECRET={STRIPE}\n")
    t = tap(leak_files=str(env))
    assert _call(t, f"/v1/chat/completions?x={STRIPE}", _msg("hi"), {"Authorization": "Bearer sk-x"})[0] == 403
    assert _call(t, "/v1/chat/completions", _msg("hi"),
                 {"Authorization": "Bearer sk-x", "X-LLMTap-Tag-Note": STRIPE})[0] == 403
    assert not mock_upstream.requests
    _wait_rows(t, 2)
    assert STRIPE not in _everything_stored(t)


# 10. provider-side search on the default upstreams
@pytest.mark.parametrize("body", [
    {"model": "openai/gpt-oss-20b", "tools": [{"type": "browser_search"}], "messages": []},       # Groq
    {"model": "groq/compound", "messages": []},
    {"model": "openai/gpt-4o", "plugins": [{"id": "web"}], "messages": []},                       # OpenRouter
    {"model": "openai/gpt-4o:online", "messages": []},
    {"model": "gpt-4o-search-preview", "messages": []},
    {"contents": [], "tools": [{"googleMaps": {}}]},
    {"model": "grok-4", "search_parameters": {"mode": "on"}, "messages": []},
    {"model": "m", "tools": [{"type": "some_new_server_tool_20270101", "name": "x"}]},           # unknown type
    {"messages": [], "toolConfig": {"tools": [{"systemTool": {"name": "nova_grounding"}}]}},     # Bedrock
    {"agent": "deep-research-pro-preview-12-2025", "input": "x"},                               # Gemini agents
])
def test_more_server_tools(body: dict) -> None:
    assert server_tools(json.dumps(body).encode())


def test_gemini_function_response_is_data() -> None:
    body = {"contents": [{"role": "user", "parts": [{"functionResponse": {"name": "f", "response": {
        "image_url": "https://x.example/a.png"}}}]}], "tools": [{"functionDeclarations": [{"name": "f"}]}]}
    assert server_tools(json.dumps(body).encode()) == []


def test_tricky_urls_are_remote() -> None:
    for url in ("https://evil.example\\@api.anthropic.com/x.png", "gs://attacker-bucket/data?x=1"):
        body = {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": url}}]}]}
        body2 = {"contents": [{"parts": [{"fileData": {"fileUri": url}}]}]}
        assert server_tools(json.dumps(body).encode()) or server_tools(json.dumps(body2).encode())


# 12. keys-only: presigned SigV4 query auth, and a placeholder that is only part of a credential
@pytest.mark.parametrize("path,headers", [
    ("/v1/chat/completions?X-Amz-Credential=AKIAEXAMPLE%2F20261124&X-Amz-Signature=abc123", {}),
    ("/v1/chat/completions", {"Authorization": "Bearer sk-attacker0123456789 llmtap-key-openai"}),
    ("/v1/chat/completions?oauth_token=ya29.attacker", {}),
])
def test_keys_only_more_credentials(vtap: Any, vault: Vault, mock_upstream: Any, path: str, headers: dict) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    t = vtap(keys_only=True)
    assert _call(t, path, _msg("hi"), headers)[0] == 403 and not mock_upstream.requests


# credential-minting endpoints would let a stored key escape llmtap entirely
@pytest.mark.parametrize("path", ["/v1/realtime/client_secrets", "/v1/realtime/sessions"])
def test_stored_key_cannot_mint_credentials(vtap: Any, vault: Vault, mock_upstream: Any, path: str) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    t = vtap()
    status, body = _call(t, path, {"session": {"type": "realtime", "model": "gpt-realtime"}},
                         {"Authorization": "Bearer llmtap-key-openai"})
    assert status == 403 and body["error"]["type"] == "llmtap_key_mint_refused" and not mock_upstream.requests


# 13. small things
def test_env_parsing_details() -> None:
    pem = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\nabcdEFGH12345678\n-----END PRIVATE KEY-----"
    env = f'A="quoted1234567" # comment\nPRIVATE_KEY="{pem}"\nB=2\n'
    pairs = dict(parse_env(env))
    assert pairs["A"] == "quoted1234567" and pairs["PRIVATE_KEY"] == pem and pairs["B"] == "2"
    assert secretish("PRIVATE_KEY", pem)


def test_statusline_odd_path(tmp_path: Path) -> None:
    d = tmp_path / "we#ird %20 dir"
    d.mkdir()
    s = Store(str(d / "x.db"))
    s.record(Exchange(ts=time.time(), status=200, cost_usd=0.5, usage=Usage(input_tokens=10)))
    s.close()
    assert statusline(str(d / "x.db"), color=False).startswith("llmtap $0.50 today")


def test_budget_warn_accepts_100() -> None:
    assert alerts.parse_thresholds("50,80,100") == [50.0, 80.0, 100.0]
