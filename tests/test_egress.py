"""What may leave: leak tripwire, canaries, keys-only mode, provider-side tool blocking, the request size limit."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from llmtap.egress import (Tripwire, add_canary, default_env_files, load_canaries, parse_env, remove_canary,
                           secretish, server_tools, variants)
from llmtap.vault import Vault
from test_vault import OPENAI_KEY, _anthropic, _call, _everything_stored, _openai, _wait_rows

DB_PASS = "pa55" + "w0rd-Xq8Zr2"
STRIPE = "rk_" + "test_" + "51Habc0123456789XYZqwerty"
ENV = f"""# app settings
DEBUG=true
PORT=8080
ANTHROPIC_BASE_URL=http://127.0.0.1:4141
OPENAI_API_KEY=llmtap-key-openai
STRIPE_SECRET={STRIPE}
DATABASE_URL="postgres://app:{DB_PASS}@db.internal:5432/app"
GREETING=hello world
SESSION_SIGNING=short
"""


def _msg(text: str) -> dict:
    return {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": text}]}


# ------------------------------------------------------------------------------------------------- .env parsing

def test_parse_env_and_secretish() -> None:
    pairs = dict(parse_env(ENV + "export TOKEN='abc def ghi'\nNOEQ\n  # c\nINLINE=value123456 # comment\n"))
    assert pairs["DATABASE_URL"].startswith("postgres://") and pairs["TOKEN"] == "abc def ghi"
    assert pairs["INLINE"] == "value123456"
    picked = {k for k, v in parse_env(ENV) if secretish(k, v)}
    assert picked == {"STRIPE_SECRET", "DATABASE_URL"}      # not DEBUG/PORT/base URL/placeholder/short/plain words
    assert secretish("RANDOM_THING", "Zx81kQ0pLm2Nv7Rt5Yb3Wc9") is True    # long and random, any name
    assert secretish("API_KEY", "your-api-key-here") is False
    assert secretish("HOME_DIR", "/home/someone/projects") is False


def test_default_env_files_skips_templates(tmp_path: Path) -> None:
    for n in (".env", ".env.local", ".env.example", ".env.sample", "env.txt"):
        (tmp_path / n).write_text("A=1\n")
    assert [p.name for p in default_env_files(tmp_path)] == [".env", ".env.local"]


# ---------------------------------------------------------------------------------------------------- variants

@pytest.mark.parametrize("prefix", [b"", b"x", b"xy", b"xyz", b"ENV=\n"])
def test_base64_found_at_any_offset(tmp_path: Path, prefix: bytes) -> None:
    env = tmp_path / ".env"
    env.write_text(f"STRIPE_SECRET={STRIPE}\n")
    tw = Tripwire([env], home=tmp_path)
    blob = base64.b64encode(prefix + STRIPE.encode() + b"\nOTHER=1\n").decode()
    leaks = tw.scan(json.dumps(_msg(f"here is the file: {blob}")).encode())
    assert [(lk.label, lk.encoding) for lk in leaks] == [("STRIPE_SECRET from .env", "base64")]
    urlsafe = base64.urlsafe_b64encode(prefix + STRIPE.encode()).decode()
    assert tw.scan(json.dumps(_msg(urlsafe)).encode())


def test_other_encodings(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(ENV)
    tw = Tripwire([env], home=tmp_path)
    for text, enc in [(STRIPE, "plain"), (STRIPE.encode().hex(), "hex"), (STRIPE.encode().hex().upper(), "hex")]:
        leaks = tw.scan(json.dumps(_msg(f"value: {text}")).encode())
        assert leaks and leaks[0].encoding == enc
    # the password inside a connection string, on its own and URL-encoded
    assert tw.scan(json.dumps(_msg(f"pw is {DB_PASS}")).encode())
    # JSON \\u escapes hide nothing
    escaped = json.dumps(_msg(STRIPE), ensure_ascii=True).replace("rk_", "\\u0072k_")
    assert tw.scan(escaped.encode())
    assert tw.scan(json.dumps(_msg("nothing secret, port 8080, localhost, hello world")).encode()) == []


def test_variants_are_unique_and_long_enough() -> None:
    v = variants("ab/cd+ef\"gh12345")
    texts = [t for _, t in v]
    assert len(texts) == len(set(texts))
    assert ("json-escaped", 'ab/cd+ef\\"gh12345') in v and any(e == "url-encoded" for e, _ in v)
    assert all(len(t) >= 12 for e, t in v if e != "plain")


def test_redact_removes_every_form(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(ENV)
    tw = Tripwire([env], home=tmp_path)
    body = json.dumps(_msg(f"{STRIPE} and {base64.b64encode(STRIPE.encode()).decode()}")).encode()
    red = tw.redact(body)
    assert STRIPE.encode() not in red and b"LEAK-REDACTED" in red
    assert tw.scan(red) == []


def test_tripwire_picks_up_changes(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("A=1\n")
    tw = Tripwire([env], home=tmp_path, refresh_s=0)
    assert tw.count() == 0
    env.write_text(f"STRIPE_SECRET={STRIPE}\n")
    assert tw.count() == 1


# ---------------------------------------------------------------------------------------------------- canaries

def test_canary_add_list_remove(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("A=1")                                     # no trailing newline
    rec = add_canary(env, "ADMIN_API_TOKEN", home=tmp_path)
    text = env.read_text()
    assert text == f"A=1\nADMIN_API_TOKEN={rec['value']}\n"
    assert [c["name"] for c in load_canaries(tmp_path)] == ["ADMIN_API_TOKEN"]
    with pytest.raises(ValueError):
        add_canary(env, "ADMIN_API_TOKEN", home=tmp_path)    # already there
    with pytest.raises(ValueError):
        add_canary(env, "bad name", home=tmp_path)
    tw = Tripwire([env], home=tmp_path)
    leaks = tw.scan(json.dumps(_msg(f"cat .env -> {text}")).encode())
    assert len(leaks) == 1 and leaks[0].canary and leaks[0].label.startswith("canary ADMIN_API_TOKEN")
    assert remove_canary("ADMIN_API_TOKEN", home=tmp_path) == 1 and load_canaries(tmp_path) == []


# ------------------------------------------------------------------------------------------- through the proxy

def test_proxy_blocks_env_value_and_never_stores_it(tap: Any, tmp_path: Path, mock_upstream: Any) -> None:
    env = tmp_path / ".env"
    env.write_text(ENV)
    t = tap(leak_files=str(env))
    status, body = _call(t, "/v1/chat/completions", _msg(f"my config: {base64.b64encode(ENV.encode()).decode()}"),
                         {"Authorization": "Bearer sk-whatever"})
    assert status == 403 and "STRIPE_SECRET from .env (base64)" in json.dumps(body)
    assert not mock_upstream.requests
    rows = _wait_rows(t, 1)
    assert rows[0].source == "blocked" and any(f["kind"] == "leak" for f in rows[0].guard_findings)
    stored = _everything_stored(t)
    assert STRIPE not in stored and DB_PASS not in stored
    assert not any(form in stored for _, form in variants(STRIPE))


def test_warn_mode_forwards_and_notes(tap: Any, tmp_path: Path, mock_upstream: Any) -> None:
    env = tmp_path / ".env"
    env.write_text(ENV)
    t = tap(leak_files=str(env), leak_check="warn")
    status, _ = _call(t, "/v1/chat/completions", _msg(f"key {STRIPE}"), {"Authorization": "Bearer sk-whatever"})
    assert status == 200 and len(mock_upstream.requests) == 1
    rows = _wait_rows(t, 1)
    assert "leak tripwire" in (rows[0].error or "") and STRIPE not in _everything_stored(t)


def test_off_mode_and_clean_requests_pass(tap: Any, tmp_path: Path, mock_upstream: Any) -> None:
    env = tmp_path / ".env"
    env.write_text(ENV)
    t = tap(leak_files=str(env), leak_check="off")
    assert _call(t, "/v1/chat/completions", _msg(STRIPE), {"Authorization": "Bearer sk-x"})[0] == 200
    t2 = tap(leak_files=str(env))
    assert _call(t2, "/v1/chat/completions", _msg("hello world on port 8080"), {"Authorization": "Bearer sk-x"})[0] == 200


def test_stored_key_in_prompt_is_blocked(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    t = vtap(leak_files="none-such-file")
    status, body = _call(t, "/v1/chat/completions", _msg(f"the key is {OPENAI_KEY}"),
                         {"Authorization": "Bearer llmtap-key-openai"})
    assert status == 403 and "stored key 'openai'" in json.dumps(body) and not mock_upstream.requests
    assert OPENAI_KEY not in _everything_stored(t)
    assert _openai(t)[0] == 200                               # the normal placeholder swap still works


# --------------------------------------------------------------------------------------------------- keys-only

def test_keys_only_refuses_real_keys(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("openai", "openai", OPENAI_KEY)
    vault.add("anthropic", "anthropic", "sk-ant-" + "test" + "0123456789abcdefANTH")
    t = vtap(keys_only=True)
    status, body = _openai(t, key="sk-attacker" + "0123456789abcdef")
    assert status == 403 and body["error"]["type"] == "llmtap_keys_only"
    status, _ = _call(t, "/v1/messages", {"model": "claude-sonnet-4-5", "max_tokens": 5, "messages": []},
                      {"x-api-key": "llmtap-key-anthropic", "authorization": "Bearer sk-attacker0123456789"})
    assert status == 403                                     # a real key next to a placeholder
    assert not mock_upstream.requests
    assert _openai(t)[0] == 200 and _anthropic(t)[0] == 200
    # no credentials at all (a local model) is fine
    assert _call(t, "/v1/chat/completions", _msg("hi"), {})[0] == 200


# ------------------------------------------------------------------------------------------ provider-side tools

@pytest.mark.parametrize("body,expect", [
    ({"model": "m", "tools": [{"type": "web_search_20260209", "name": "web_search"}]}, "web_search_20260209"),
    ({"model": "m", "tools": [{"type": "web_fetch_20260209", "name": "web_fetch"}]}, "web_fetch"),
    ({"model": "m", "tools": [{"type": "code_execution_20250825", "name": "code_execution"}]}, "code_execution"),
    ({"model": "m", "mcp_servers": [{"type": "url", "url": "https://mcp.example/sse", "name": "x"}]}, "MCP"),
    ({"model": "m", "input": "hi", "tools": [{"type": "mcp", "server_url": "https://m.example"}]}, "MCP"),
    ({"model": "m", "input": "hi", "tools": [{"type": "web_search_preview"}]}, "web_search"),
    ({"model": "m", "web_search_options": {}, "messages": []}, "web search"),
    ({"contents": [], "tools": [{"googleSearch": {}}]}, "Google Search"),
    ({"contents": [], "tools": [{"url_context": {}}]}, "URL context"),
    ({"contents": [], "tools": [{"codeExecution": {}}]}, "code execution"),
    ({"model": "m", "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "url", "url": "https://evil.example/x.png?d=SECRET"}}]}]}, "URL"),
    ({"model": "m", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://evil.example/?q=1"}}]}]}, "URL"),
    ({"contents": [{"parts": [{"fileData": {"fileUri": "https://evil.example/doc.pdf"}}]}]}, "URL"),
])
def test_server_tools_detected(body: dict, expect: str) -> None:
    found = server_tools(json.dumps(body).encode())
    assert found and any(expect.lower() in f.lower() for f in found), found


@pytest.mark.parametrize("body", [
    {"model": "m", "messages": [{"role": "user", "content": "use web_search please, type: mcp"}]},
    {"model": "m", "tools": [{"name": "fetch", "input_schema": {"type": "object", "properties": {
        "url": {"type": "string"}, "image_url": {"type": "string"}, "container": {"type": "string"}}}}]},
    {"model": "m", "messages": [{"role": "assistant", "content": [
        {"type": "tool_use", "id": "1", "name": "fetch", "input": {"image_url": "https://x.example/a.png"}}]}]},
    {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]},
    {"contents": [{"parts": [{"fileData": {"fileUri": "https://generativelanguage.googleapis.com/v1beta/files/x"}}]}]},
    {"model": "m", "tools": [{"type": "bash_20250124", "name": "bash"}, {"type": "text_editor_20250728"}]},
])
def test_server_tools_not_false_positives(body: dict) -> None:
    assert server_tools(json.dumps(body).encode()) == []


def test_duplicate_keys_cannot_hide_a_tool() -> None:
    raw = b'{"model":"m","tools":[{"type":"web_search_20260209","type":"custom","name":"x"}]}'
    assert server_tools(raw)


def test_stored_key_refuses_server_tools_unless_allowed(vtap: Any, vault: Vault, mock_upstream: Any) -> None:
    vault.add("anthropic", "anthropic", "sk-ant-" + "test" + "0123456789abcdefANTH")
    t = vtap()
    body = {"model": "claude-sonnet-4-5", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_fetch_20260209", "name": "web_fetch"}]}
    hdrs = {"x-api-key": "llmtap-key-anthropic", "anthropic-version": "2023-06-01"}
    status, resp = _call(t, "/v1/messages", body, hdrs)
    assert status == 403 and resp["error"]["type"] == "llmtap_server_tool_blocked" and not mock_upstream.requests
    # a real key is refused while one is stored; when allowed it's the user's own business (not checked)
    own = {"x-api-key": "sk-ant-own0123456789", "anthropic-version": "2023-06-01"}
    assert _call(t, "/v1/messages", body, own)[1]["error"]["type"] == "llmtap_real_key_refused"
    assert _call(vtap(allow_real_keys=True), "/v1/messages", body, own)[0] == 200
    vault.update("anthropic", allow_server_tools=True)
    assert _call(t, "/v1/messages", body, hdrs)[0] == 200


# ------------------------------------------------------------------------------------------------ size limit

def test_max_request_kb(tap: Any, mock_upstream: Any) -> None:
    t = tap(max_request_kb=4)
    status, body = _call(t, "/v1/chat/completions", _msg("x" * 5000), {"Authorization": "Bearer sk-x"})
    assert status == 413 and body["error"]["type"] == "llmtap_request_too_large" and not mock_upstream.requests
    assert _call(t, "/v1/chat/completions", _msg("x" * 100), {"Authorization": "Bearer sk-x"})[0] == 200
    rows = _wait_rows(t, 2)
    assert any("not stored" in r.request_body for r in rows)


def test_cli_canary_and_server_tools_flag(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    from llmtap.cli import main
    monkeypatch.setenv("LLMTAP_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LLMTAP_VAULT_BACKEND", "memory")
    env = tmp_path / ".env"
    assert main(["canary", "add", "--file", str(env), "--name", "OLD_TOKEN"]) == 0
    assert "OLD_TOKEN=tk_" in env.read_text()
    assert main(["canary", "list"]) == 0 and "OLD_TOKEN" in capsys.readouterr().out
    assert main(["canary", "remove", "OLD_TOKEN"]) == 0
    assert main(["canary", "remove", "OLD_TOKEN"]) == 1
