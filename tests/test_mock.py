"""Tests for llmtap.mock.MockUpstream (no proxy involved)."""
from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from llmtap.mock import MockUpstream, count_tokens, embedding_for, split_chunks


@pytest.fixture()
def mock() -> MockUpstream:
    m = MockUpstream().start()
    yield m
    m.stop()


def post(base: str, path: str, body: dict, headers: dict | None = None) -> tuple[int, dict, bytes]:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()


def sse_events(raw: bytes) -> list[tuple[str | None, str]]:
    out = []
    for block in raw.decode().split("\n\n"):
        if not block.strip():
            continue
        name, data = None, []
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        out.append((name, "\n".join(data)))
    return out


def test_split_chunks_roundtrip() -> None:
    for t in ["Echo: hello world", "  leading", "trailing  ", "one", "", "a\nb  c"]:
        assert "".join(split_chunks(t)) == t
    assert len(split_chunks("Echo: one two three")) == 4


def test_embedding_deterministic() -> None:
    v = embedding_for("hi")
    assert v == embedding_for("hi") and len(v) == 8 and v != embedding_for("ho")


def test_anthropic_non_stream(mock: MockUpstream) -> None:
    body = {"model": "claude-sonnet-4-5", "max_tokens": 100, "system": "Be terse.",
            "messages": [{"role": "user", "content": "Hello there"}]}
    status, headers, raw = post(mock.base_url, "/v1/messages", body,
                                {"x-api-key": "k", "anthropic-version": "2023-06-01"})
    assert status == 200
    assert headers["content-type"].startswith("application/json")
    resp = json.loads(raw)
    assert resp["id"].startswith("msg_mock_")
    assert resp["type"] == "message" and resp["role"] == "assistant"
    assert resp["content"] == [{"type": "text", "text": "Echo: Hello there"}]
    assert resp["model"] == "claude-sonnet-4-5"
    assert resp["stop_reason"] == "end_turn"
    u = resp["usage"]
    assert u["input_tokens"] == count_tokens("Be terse.\nHello there")
    assert u["output_tokens"] == count_tokens("Echo: Hello there")
    assert u["cache_read_input_tokens"] == 0 and u["cache_creation_input_tokens"] == 0
    rec = mock.requests[-1]
    assert rec["method"] == "POST" and rec["path"] == "/v1/messages"
    assert rec["headers"]["x-api-key"] == "k"
    assert rec["body"] == body


def test_anthropic_stream_event_sequence(mock: MockUpstream) -> None:
    body = {"model": "claude-haiku-4-5", "max_tokens": 50, "stream": True,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "stream me please"}]}]}
    status, headers, raw = post(mock.base_url, "/v1/messages", body)
    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert headers.get("transfer-encoding") == "chunked"
    events = sse_events(raw)
    names = [n for n, _ in events]
    assert names[0] == "message_start" and names[1] == "content_block_start"
    assert names[-3:] == ["content_block_stop", "message_delta", "message_stop"]
    datas = [json.loads(d) for _, d in events]
    assert all(d["type"] == n for (n, _), d in zip(events, datas))
    assert datas[0]["message"]["model"] == "claude-haiku-4-5"
    assert datas[0]["message"]["id"].startswith("msg_mock_")
    text = "".join(d["delta"]["text"] for d in datas if d["type"] == "content_block_delta")
    assert text == "Echo: stream me please"
    assert sum(1 for d in datas if d["type"] == "content_block_delta") == 4
    md = next(d for d in datas if d["type"] == "message_delta")
    assert md["usage"]["output_tokens"] == count_tokens(text)
    assert datas[0]["message"]["usage"]["input_tokens"] == count_tokens("stream me please")


def test_stream_chunks_are_flushed_incrementally() -> None:
    m = MockUpstream(token_delay_s=0.15).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", m.port, timeout=10)
        body = json.dumps({"model": "claude-sonnet-4-5", "stream": True,
                           "messages": [{"role": "user", "content": "a b c d"}]})
        t0 = time.monotonic()
        conn.request("POST", "/v1/messages", body=body, headers={"content-type": "application/json"})
        resp = conn.getresponse()
        first = resp.read1(65536)
        t_first = time.monotonic() - t0
        rest = resp.read()
        t_all = time.monotonic() - t0
        assert b"message_start" in first
        assert t_first < 0.3 < t_all  # first frames arrive before the slow token deltas
        assert b"message_stop" in rest
        conn.close()
    finally:
        m.stop()


def test_openai_chat_non_stream(mock: MockUpstream) -> None:
    body = {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "sys"},
                                                 {"role": "user", "content": "Hi"},
                                                 {"role": "assistant", "content": "Hello"},
                                                 {"role": "user", "content": "What is 2+2?"}]}
    status, _, raw = post(mock.base_url, "/v1/chat/completions", body, {"authorization": "Bearer x"})
    assert status == 200
    resp = json.loads(raw)
    assert resp["object"] == "chat.completion" and resp["model"] == "gpt-4o-mini"
    assert resp["choices"][0]["message"] == {"role": "assistant", "content": "Echo: What is 2+2?", "refusal": None}
    assert resp["choices"][0]["finish_reason"] == "stop"
    assert resp["usage"]["prompt_tokens"] == count_tokens("sys\nHi\nHello\nWhat is 2+2?")
    assert resp["usage"]["completion_tokens"] == count_tokens("Echo: What is 2+2?")
    assert resp["usage"]["prompt_tokens_details"]["cached_tokens"] == 0


@pytest.mark.parametrize("include_usage", [True, False])
def test_openai_chat_stream(mock: MockUpstream, include_usage: bool) -> None:
    body: dict = {"model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "count to three"}]}
    if include_usage:
        body["stream_options"] = {"include_usage": True}
    status, headers, raw = post(mock.base_url, "/v1/chat/completions", body)
    assert status == 200 and headers["content-type"].startswith("text/event-stream")
    events = sse_events(raw)
    assert all(n is None for n, _ in events)
    assert events[-1][1] == "[DONE]"
    chunks = [json.loads(d) for _, d in events[:-1]]
    text = "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c["choices"])
    assert text == "Echo: count to three"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    usage_chunks = [c for c in chunks if c.get("usage")]
    if include_usage:
        assert len(usage_chunks) == 1 and usage_chunks[0]["choices"] == []
        assert usage_chunks[0]["usage"]["completion_tokens"] == count_tokens(text)
    else:
        assert usage_chunks == []
    assert any(c["choices"] and c["choices"][0]["finish_reason"] == "stop" for c in chunks)


def test_responses_api(mock: MockUpstream) -> None:
    status, _, raw = post(mock.base_url, "/v1/responses", {"model": "gpt-5", "input": "Write a haiku"})
    assert status == 200
    resp = json.loads(raw)
    assert resp["object"] == "response" and resp["status"] == "completed"
    assert resp["output"][0]["content"][0]["text"] == "Echo: Write a haiku"
    assert resp["usage"]["input_tokens"] == count_tokens("Write a haiku")

    status, _, raw = post(mock.base_url, "/v1/responses",
                          {"model": "gpt-5", "stream": True, "instructions": "Be poetic",
                           "input": [{"role": "user", "content": [{"type": "input_text", "text": "Write a haiku"}]}]})
    assert status == 200
    events = sse_events(raw)
    datas = [json.loads(d) for _, d in events]
    assert events[0][0] == "response.created" and events[-1][0] == "response.completed"
    assert "".join(d["delta"] for d in datas if d["type"] == "response.output_text.delta") == "Echo: Write a haiku"
    assert datas[-1]["response"]["usage"]["input_tokens"] == count_tokens("Be poetic\nWrite a haiku")


def test_embeddings_and_models(mock: MockUpstream) -> None:
    status, _, raw = post(mock.base_url, "/v1/embeddings", {"model": "text-embedding-3-small", "input": ["a", "bb"]})
    resp = json.loads(raw)
    assert status == 200 and len(resp["data"]) == 2
    assert resp["data"][1]["embedding"] == embedding_for("bb") and len(resp["data"][0]["embedding"]) == 8
    assert resp["usage"]["prompt_tokens"] == 2
    with urllib.request.urlopen(mock.base_url + "/v1/models", timeout=5) as r:
        models = json.loads(r.read())
    assert models["object"] == "list" and any(m["id"] == "gpt-4o" for m in models["data"])
    assert mock.requests[-1]["method"] == "GET" and mock.requests[-1]["body"] is None


def test_magic_words(mock: MockUpstream) -> None:
    status, _, raw = post(mock.base_url, "/v1/messages",
                          {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "x MOCK_ERROR_500"}]})
    assert status == 500 and json.loads(raw)["type"] == "error"
    status, headers, raw = post(mock.base_url, "/v1/chat/completions",
                                {"model": "gpt-4o", "messages": [{"role": "user", "content": "MOCK_ERROR_429"}]})
    assert status == 429 and "error" in json.loads(raw) and headers.get("retry-after") == "1"
    t0 = time.monotonic()
    status, _, _ = post(mock.base_url, "/v1/chat/completions",
                        {"model": "gpt-4o", "messages": [{"role": "user", "content": "MOCK_SLOW"}]})
    assert status == 200 and time.monotonic() - t0 >= 1.0


def test_reply_truncated_and_unknown_route(mock: MockUpstream) -> None:
    status, _, raw = post(mock.base_url, "/v1/chat/completions",
                          {"model": "gpt-4o", "messages": [{"role": "user", "content": "z" * 500}]})
    assert len(json.loads(raw)["choices"][0]["message"]["content"]) == 200
    status, _, _ = post(mock.base_url, "/v1/nope", {"a": 1})
    assert status == 404


def test_keep_alive_and_concurrency(mock: MockUpstream) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", mock.port, timeout=10)
    for i in range(3):  # several requests (incl. a chunked stream) on one HTTP/1.1 connection
        body = {"model": "gpt-4o", "stream": i == 1, "messages": [{"role": "user", "content": f"n{i}"}]}
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"content-type": "application/json"})
        r = conn.getresponse()
        assert r.status == 200 and f"n{i}".encode() in r.read()
    conn.close()

    errors: list[Exception] = []

    def worker(n: int) -> None:
        try:
            post(mock.base_url, "/v1/messages", {"model": "claude-haiku-4-5", "stream": n % 2 == 0,
                                                 "messages": [{"role": "user", "content": str(n)}]})
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(mock.requests) == 23


def test_stop_is_idempotent() -> None:
    m = MockUpstream().start()
    port = m.port
    assert port > 0 and m.base_url == f"http://127.0.0.1:{port}"
    m.stop()
    m.stop()
    with pytest.raises(OSError):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
