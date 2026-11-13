"""Full-stack scenarios: plain HTTP client -> real LLMTap proxy -> MockUpstream."""
from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import pytest

from llmtap.demo import FAKE_AWS_KEY, generate_traffic
from llmtap.mock import MockUpstream, count_tokens
from llmtap.pricing import cost_usd
from llmtap.models import Usage

ANTHROPIC_HEADERS = {"x-api-key": "test-key", "anthropic-version": "2023-06-01"}
OPENAI_HEADERS = {"authorization": "Bearer test-key"}


# --------------------------------------------------------------------------------------- helpers
def call(base: str, path: str, body: dict | bytes | None = None, headers: dict | None = None,
         method: str | None = None) -> tuple[int, dict[str, str], bytes]:
    """HTTP request via urllib; returns (status, lowercased headers, body) and never raises on HTTP errors."""
    data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method or ("POST" if data is not None else "GET"),
                                 headers={"content-type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()


def anthropic(base: str, text: str, stream: bool = False, model: str = "claude-sonnet-4-5",
              headers: dict | None = None) -> tuple[int, dict[str, str], bytes]:
    body: dict[str, Any] = {"model": model, "max_tokens": 256, "messages": [{"role": "user", "content": text}]}
    if stream:
        body["stream"] = True
    return call(base, "/v1/messages", body, {**ANTHROPIC_HEADERS, **(headers or {})})


def openai(base: str, text: str, stream: bool = False, model: str = "gpt-4o-mini",
           headers: dict | None = None) -> tuple[int, dict[str, str], bytes]:
    body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": text}]}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return call(base, "/v1/chat/completions", body, {**OPENAI_HEADERS, **(headers or {})})


def wait_for(t: Any, n: int, timeout: float = 5.0) -> list:
    """Wait until the tap's store holds >= n exchanges (recording happens after the response ends)."""
    deadline = time.monotonic() + timeout
    while True:
        items = t.store.list(limit=1000)
        if len(items) >= n or time.monotonic() > deadline:
            return items
        time.sleep(0.02)


def by_id(t: Any, ex_id: str, timeout: float = 5.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ex = t.store.get(ex_id)
        if ex is not None:
            return ex
        time.sleep(0.02)
    raise AssertionError(f"exchange {ex_id} never recorded")


def sse_text(raw: bytes, style: str) -> str:
    out = []
    for line in raw.decode().splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if style == "anthropic" and d.get("type") == "content_block_delta":
            out.append(d["delta"]["text"])
        elif style == "openai" and d.get("choices"):
            out.append(d["choices"][0]["delta"].get("content") or "")
    return "".join(out)


TapFactory = Callable[..., Any]


# --------------------------------------------------------------------------------------- basics
def test_anthropic_non_stream_recorded_and_priced(tap: TapFactory, mock_upstream: MockUpstream) -> None:
    t = tap()
    status, headers, raw = anthropic(t.base_url, "Review this function for bugs")
    assert status == 200
    resp = json.loads(raw)
    assert resp["content"][0]["text"] == "Echo: Review this function for bugs"
    assert headers["x-llmtap-source"] == "upstream"
    ex = by_id(t, headers["x-llmtap-id"])
    assert ex.provider == "anthropic" and ex.style == "anthropic" and ex.model == "claude-sonnet-4-5"
    assert ex.status == 200 and ex.source == "upstream" and not ex.streamed
    assert ex.usage.input_tokens == resp["usage"]["input_tokens"]
    assert ex.usage.output_tokens == resp["usage"]["output_tokens"]
    assert not ex.usage.estimated
    assert ex.cost_usd > 0
    assert ex.cost_usd == pytest.approx(cost_usd("claude-sonnet-4-5", Usage(ex.usage.input_tokens, ex.usage.output_tokens)))
    assert ex.output_text == "Echo: Review this function for bugs"
    # the mock saw the request with the client's auth header forwarded
    rec = mock_upstream.requests[-1]
    assert rec["path"] == "/v1/messages" and rec["headers"]["x-api-key"] == "test-key"
    assert rec["body"]["messages"][0]["content"] == "Review this function for bugs"


def test_anthropic_stream_relayed_and_recorded(tap: TapFactory) -> None:
    t = tap()
    status, headers, raw = anthropic(t.base_url, "stream a reply please", stream=True, model="claude-haiku-4-5")
    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert sse_text(raw, "anthropic") == "Echo: stream a reply please"
    assert b"event: message_stop" in raw
    ex = by_id(t, headers["x-llmtap-id"])
    assert ex.streamed and ex.status == 200 and ex.model == "claude-haiku-4-5"
    assert ex.usage.input_tokens == count_tokens("stream a reply please")
    assert ex.usage.output_tokens == count_tokens("Echo: stream a reply please")
    assert ex.cost_usd > 0
    assert ex.output_text == "Echo: stream a reply please"
    assert ex.response_body.replace("\r\n", "\n") == raw.decode().replace("\r\n", "\n")
    assert ex.ttft_ms is not None


def test_openai_non_stream_and_stream(tap: TapFactory) -> None:
    t = tap()
    status, h1, raw = openai(t.base_url, "What is 2+2?", model="gpt-4o")
    assert status == 200
    resp = json.loads(raw)
    assert resp["choices"][0]["message"]["content"] == "Echo: What is 2+2?"
    ex1 = by_id(t, h1["x-llmtap-id"])
    assert ex1.provider == "openai" and ex1.model == "gpt-4o"
    assert ex1.usage.input_tokens == resp["usage"]["prompt_tokens"]
    assert ex1.usage.output_tokens == resp["usage"]["completion_tokens"]
    assert ex1.cost_usd > 0

    status, h2, raw = openai(t.base_url, "Count to five", stream=True)
    assert status == 200 and h2["content-type"].startswith("text/event-stream")
    assert sse_text(raw, "openai") == "Echo: Count to five"
    assert raw.rstrip().endswith(b"data: [DONE]")
    ex2 = by_id(t, h2["x-llmtap-id"])
    assert ex2.streamed and ex2.model == "gpt-4o-mini"
    assert ex2.usage.input_tokens == count_tokens("Count to five")
    assert ex2.usage.output_tokens == count_tokens("Echo: Count to five")
    assert not ex2.usage.estimated and ex2.cost_usd > 0


def test_openai_stream_without_usage_is_estimated(tap: TapFactory) -> None:
    t = tap()
    body = {"model": "gpt-4o-mini", "stream": True, "messages": [{"role": "user", "content": "no usage chunk here"}]}
    status, h, raw = call(t.base_url, "/v1/chat/completions", body, OPENAI_HEADERS)
    assert status == 200 and sse_text(raw, "openai") == "Echo: no usage chunk here"
    ex = by_id(t, h["x-llmtap-id"])
    assert ex.usage.estimated and ex.usage.output_tokens > 0 and ex.cost_usd > 0


def test_stream_is_relayed_incrementally(tap: TapFactory, mock_upstream: MockUpstream) -> None:
    mock_upstream.token_delay_s = 0.15
    t = tap()
    conn = http.client.HTTPConnection("127.0.0.1", t.port, timeout=15)
    body = json.dumps({"model": "claude-sonnet-4-5", "max_tokens": 10, "stream": True,
                       "messages": [{"role": "user", "content": "one two three four"}]})
    t0 = time.monotonic()
    conn.request("POST", "/v1/messages", body=body, headers={**ANTHROPIC_HEADERS, "content-type": "application/json"})
    resp = conn.getresponse()
    first = resp.read1(65536)
    t_first = time.monotonic() - t0
    rest = resp.read()
    t_all = time.monotonic() - t0
    conn.close()
    assert b"message_start" in first
    assert t_first < 0.5 < t_all, (t_first, t_all)  # proxy didn't buffer the whole stream
    assert b"message_stop" in rest


def test_upstream_errors_pass_through(tap: TapFactory) -> None:
    t = tap()
    status, h, raw = openai(t.base_url, "please MOCK_ERROR_500")
    assert status == 500 and "error" in json.loads(raw)
    status, h2, raw = anthropic(t.base_url, "please MOCK_ERROR_429")
    assert status == 429 and json.loads(raw)["type"] == "error"
    ex = by_id(t, h["x-llmtap-id"])
    assert ex.status == 500 and ex.cost_usd == 0


def test_upstream_down_gives_502(tap: TapFactory) -> None:
    dead = MockUpstream().start()
    url = dead.base_url
    dead.stop()
    t = tap(upstream_url=url)
    status, h, raw = anthropic(t.base_url, "anyone there?")
    assert status == 502
    assert json.loads(raw)["type"] == "error"
    ex = by_id(t, h["x-llmtap-id"]) if "x-llmtap-id" in h else wait_for(t, 1)[0]
    assert ex.source == "error"


def test_embeddings_and_tags(tap: TapFactory) -> None:
    t = tap()
    status, h, raw = call(t.base_url, "/v1/embeddings", {"model": "text-embedding-3-small", "input": ["a", "b"]},
                          {**OPENAI_HEADERS, "X-LLMTap-Session": "s-42", "X-LLMTap-Tag-Feature": "rag"})
    assert status == 200 and len(json.loads(raw)["data"]) == 2
    ex = by_id(t, h["x-llmtap-id"])
    assert ex.tags == {"session": "s-42", "feature": "rag"}
    assert ex.model == "text-embedding-3-small"


# --------------------------------------------------------------------------------------- cache
@pytest.mark.parametrize("stream", [False, True])
def test_cache_saves_money(tap: TapFactory, mock_upstream: MockUpstream, stream: bool) -> None:
    t = tap(cache=True)
    s1, h1, r1 = anthropic(t.base_url, "Summarize this incident report", stream=stream)
    s2, h2, r2 = anthropic(t.base_url, "Summarize this incident report", stream=stream)
    assert s1 == s2 == 200
    assert h1["x-llmtap-source"] == "upstream" and h2["x-llmtap-source"] == "cache"
    assert r1 == r2
    assert len(mock_upstream.requests) == 1
    ex1, ex2 = by_id(t, h1["x-llmtap-id"]), by_id(t, h2["x-llmtap-id"])
    assert ex2.source == "cache" and ex2.saved_usd > 0 and ex2.cost_usd == 0
    assert ex2.saved_usd == pytest.approx(ex1.cost_usd)
    if stream:
        assert h2["content-type"].startswith("text/event-stream")
    stats = t.store.stats()
    assert stats["cache_hits"] == 1 and stats["saved_usd"] > 0


def test_cache_ignores_volatile_keys_but_not_content(tap: TapFactory, mock_upstream: MockUpstream) -> None:
    t = tap(cache=True)
    base = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    call(t.base_url, "/v1/chat/completions", {**base, "user": "alice"}, OPENAI_HEADERS)
    _, h, _ = call(t.base_url, "/v1/chat/completions", {**base, "user": "bob"}, OPENAI_HEADERS)
    assert h["x-llmtap-source"] == "cache"
    _, h, _ = call(t.base_url, "/v1/chat/completions",
                   {**base, "messages": [{"role": "user", "content": "hi!"}]}, OPENAI_HEADERS)
    assert h["x-llmtap-source"] == "upstream"
    assert len(mock_upstream.requests) == 2


# --------------------------------------------------------------------------------------- cassette
def test_cassette_record_then_replay_offline(tap: TapFactory, mock_upstream: MockUpstream, tmp_path: Path) -> None:
    cassette = str(tmp_path / "llm.jsonl")
    rec = tap(cassette=cassette, mode="record")
    recorded = [
        anthropic(rec.base_url, "Write a SQL query for top customers"),
        anthropic(rec.base_url, "Stream a haiku", stream=True),
        openai(rec.base_url, "Translate to Japanese: hello"),
        openai(rec.base_url, "Stream a limerick", stream=True, model="gpt-4o"),
    ]
    assert all(s == 200 for s, _, _ in recorded)
    wait_for(rec, 4)
    rec.shutdown()
    lines = Path(cassette).read_text().splitlines()
    assert len(lines) == 4 and all("key" in json.loads(ln) for ln in lines)

    mock_upstream.stop()  # no upstream at all: replay must be fully offline
    n_seen = len(mock_upstream.requests)
    rep = tap(cassette=cassette, mode="replay")
    replayed = [
        anthropic(rep.base_url, "Write a SQL query for top customers"),
        anthropic(rep.base_url, "Stream a haiku", stream=True),
        openai(rep.base_url, "Translate to Japanese: hello"),
        openai(rep.base_url, "Stream a limerick", stream=True, model="gpt-4o"),
    ]
    for (s1, _, b1), (s2, h2, b2) in zip(recorded, replayed):
        assert s2 == s1 == 200
        assert h2["x-llmtap-source"] == "cassette"
        assert b2 == b1  # byte-for-byte (ids / timestamps from the mock are random, so this proves replay)
    assert len(mock_upstream.requests) == n_seen
    exs = wait_for(rep, 4)
    assert all(e.source == "cassette" and e.saved_usd > 0 for e in exs)

    # a request that was never recorded -> 409 in replay mode
    status, _, raw = anthropic(rep.base_url, "never recorded")
    assert status == 409
    assert "cassette miss" in json.loads(raw)["error"]["message"]


# --------------------------------------------------------------------------------------- guard
def test_guard_redact_scrubs_before_upstream(tap: TapFactory, mock_upstream: MockUpstream) -> None:
    t = tap(guard="redact")
    prompt = f"Why does this fail? export AWS_ACCESS_KEY_ID={FAKE_AWS_KEY}"
    status, h, _ = anthropic(t.base_url, prompt)
    assert status == 200
    sent = mock_upstream.requests[-1]["body"]["messages"][0]["content"]
    assert "[REDACTED:aws_access_key]" in sent
    assert FAKE_AWS_KEY not in json.dumps(mock_upstream.requests[-1]["body"])
    ex = by_id(t, h["x-llmtap-id"])
    assert any(f["kind"] == "aws_access_key" for f in ex.guard_findings)
    assert FAKE_AWS_KEY not in ex.request_body


def test_guard_block_returns_400(tap: TapFactory, mock_upstream: MockUpstream) -> None:
    t = tap(guard="block")
    status, h, raw = openai(t.base_url, f"key={FAKE_AWS_KEY}")
    assert status == 400 and "error" in json.loads(raw)
    assert mock_upstream.requests == []
    ex = by_id(t, h["x-llmtap-id"]) if "x-llmtap-id" in h else wait_for(t, 1)[0]
    assert ex.source == "blocked"


def test_guard_warn_forwards_unchanged(tap: TapFactory, mock_upstream: MockUpstream) -> None:
    t = tap(guard="warn")
    status, h, _ = openai(t.base_url, f"key={FAKE_AWS_KEY}")
    assert status == 200
    assert FAKE_AWS_KEY in mock_upstream.requests[-1]["body"]["messages"][0]["content"]
    assert by_id(t, h["x-llmtap-id"]).guard_findings


# --------------------------------------------------------------------------------------- budget/loop
def test_loop_breaker_blocks_after_threshold(tap: TapFactory, mock_upstream: MockUpstream) -> None:
    threshold = 3
    t = tap(loop_threshold=threshold, loop_window_s=60)
    statuses = [openai(t.base_url, "fix the failing test and try again")[0] for _ in range(threshold)]
    assert statuses == [200] * threshold
    status, h, raw = openai(t.base_url, "fix the failing test and try again")
    assert status == 429
    assert "loop" in json.loads(raw)["error"]["message"].lower()
    assert len(mock_upstream.requests) == threshold
    # a different request is still fine
    assert openai(t.base_url, "something else entirely")[0] == 200
    exs = wait_for(t, threshold + 2)
    assert sum(1 for e in exs if e.source == "loop") == 1


def test_daily_budget_blocks(tap: TapFactory, mock_upstream: MockUpstream) -> None:
    t = tap(budget_daily_usd=0.0001)
    big = "Summarize this design doc. " * 200   # ~1.3k tokens on opus -> well over $0.0001
    status, h, _ = anthropic(t.base_url, big, model="claude-opus-4-1")
    assert status == 200
    assert by_id(t, h["x-llmtap-id"]).cost_usd > 0.0001
    status, h, raw = anthropic(t.base_url, "a different, small request")
    assert status == 429
    body = json.loads(raw)
    assert body["type"] == "error" and "budget" in body["error"]["message"].lower()
    assert len(mock_upstream.requests) == 1
    ex = by_id(t, h["x-llmtap-id"]) if "x-llmtap-id" in h else wait_for(t, 2)[0]
    assert ex.source == "budget"


# --------------------------------------------------------------------------------------- dashboard
def test_dashboard_api_reflects_traffic(tap: TapFactory) -> None:
    t = tap(cache=True)
    anthropic(t.base_url, "dashboard one")
    anthropic(t.base_url, "dashboard one")
    openai(t.base_url, "dashboard two", stream=True)
    wait_for(t, 3)
    status, _, raw = call(t.base_url, "/_llmtap/api/stats")
    assert status == 200
    stats = json.loads(raw)
    assert stats["requests"] == 3 and stats["cache_hits"] == 1
    assert stats["cost_usd"] > 0 and stats["saved_usd"] > 0
    assert stats["input_tokens"] > 0 and stats["output_tokens"] > 0
    status, _, raw = call(t.base_url, "/_llmtap/api/stats?group_by=model")
    groups = {g["key"]: g for g in json.loads(raw)["groups"]}
    assert set(groups) >= {"claude-sonnet-4-5", "gpt-4o-mini"}

    status, _, raw = call(t.base_url, "/_llmtap/api/exchanges?limit=10")
    assert status == 200
    items = json.loads(raw)["items"]
    assert len(items) == 3
    assert "request_body" not in items[0] and "response_body" not in items[0]
    assert items[0]["preview"] == "dashboard two"  # newest first
    assert items[0]["output_preview"] == "Echo: dashboard two"
    status, _, raw = call(t.base_url, f"/_llmtap/api/exchanges/{items[0]['id']}")
    assert status == 200 and "request_body" in json.loads(raw)
    status, _, raw = call(t.base_url, "/_llmtap/api/exchanges?source=cache")
    assert [i["source"] for i in json.loads(raw)["items"]] == ["cache"]
    status, _, raw = call(t.base_url, "/_llmtap/health")
    assert status == 200 and json.loads(raw)["ok"] is True
    status, h, raw = call(t.base_url, "/_llmtap/")
    assert status == 200 and h["content-type"].startswith("text/html") and b"llmtap" in raw.lower()


def test_dashboard_event_stream_publishes_exchanges(tap: TapFactory) -> None:
    t = tap()
    conn = http.client.HTTPConnection("127.0.0.1", t.port, timeout=10)
    conn.request("GET", "/_llmtap/api/events")
    resp = conn.getresponse()
    assert resp.status == 200 and resp.getheader("content-type", "").startswith("text/event-stream")
    got: list[bytes] = []

    def reader() -> None:
        buf = b""
        while b"exchange" not in buf:
            chunk = resp.read1(4096)
            if not chunk:
                break
            buf += chunk
        got.append(buf)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    deadline = time.monotonic() + 5
    while th.is_alive() and time.monotonic() < deadline:
        openai(t.base_url, "live event please")
        th.join(0.3)
    conn.close()
    assert got and b"exchange" in got[0]


# --------------------------------------------------------------------------------------- demo
def test_generate_traffic_runs_clean(tap: TapFactory, mock_upstream: MockUpstream) -> None:
    t = tap(cache=True, guard="redact", loop_threshold=4)
    generate_traffic(t.base_url, count=15, delay_s=0)
    exs = wait_for(t, 15)
    assert len(exs) == 15
    sources = {e.source for e in exs}
    assert {"upstream", "cache"} <= sources
    assert {e.provider for e in exs} >= {"anthropic", "openai"}
    assert len({e.model for e in exs}) >= 5
    assert any(e.streamed for e in exs) and any(not e.streamed for e in exs)
    assert any(e.guard_findings for e in exs)
    assert all(FAKE_AWS_KEY not in json.dumps(r["body"]) for r in mock_upstream.requests)
    assert any(e.tags.get("feature") for e in exs)
    assert sum(e.cost_usd for e in exs) > 0
    assert not [e for e in exs if e.source == "error"]


def test_generate_traffic_full_run_and_stop(tap: TapFactory) -> None:
    t = tap(cache=True, guard="redact", loop_threshold=4)
    generate_traffic(t.base_url, count=40, delay_s=0)
    exs = wait_for(t, 40)
    assert len(exs) == 40
    sources = {e.source for e in exs}
    assert {"upstream", "cache", "loop"} <= sources
    assert any(e.status == 500 for e in exs)  # the simulated outage
    # stop event honoured immediately
    ev = threading.Event()
    ev.set()
    before = t.store.count()
    generate_traffic(t.base_url, count=10, delay_s=0, stop=ev)
    assert t.store.count() == before


def test_generate_traffic_tolerates_dead_proxy() -> None:
    dead = MockUpstream().start()
    url = dead.base_url
    dead.stop()
    generate_traffic(url, count=3, delay_s=0)  # must not raise


# --------------------------------------------------------------------------------------- SDKs (optional)
def test_anthropic_sdk_through_proxy(tap: TapFactory) -> None:
    anthropic_sdk = pytest.importorskip("anthropic")
    t = tap()
    client = anthropic_sdk.Anthropic(base_url=t.base_url, api_key="test-key", max_retries=0)
    msg = client.messages.create(model="claude-sonnet-4-5", max_tokens=50,
                                 messages=[{"role": "user", "content": "hello sdk"}])
    assert msg.content[0].text == "Echo: hello sdk"
    with client.messages.stream(model="claude-sonnet-4-5", max_tokens=50,
                                messages=[{"role": "user", "content": "hello stream"}]) as s:
        assert "".join(s.text_stream) == "Echo: hello stream"
    exs = wait_for(t, 2)
    assert all(e.cost_usd > 0 for e in exs)


def test_openai_sdk_through_proxy(tap: TapFactory) -> None:
    openai_sdk = pytest.importorskip("openai")
    t = tap()
    client = openai_sdk.OpenAI(base_url=t.base_url + "/v1", api_key="test-key", max_retries=0)
    r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hello sdk"}])
    assert r.choices[0].message.content == "Echo: hello sdk"
    chunks = client.chat.completions.create(model="gpt-4o-mini", stream=True,
                                            stream_options={"include_usage": True},
                                            messages=[{"role": "user", "content": "hello stream"}])
    text = "".join((c.choices[0].delta.content or "") for c in chunks if c.choices)
    assert text == "Echo: hello stream"
    exs = wait_for(t, 2)
    assert all(e.cost_usd > 0 for e in exs)
