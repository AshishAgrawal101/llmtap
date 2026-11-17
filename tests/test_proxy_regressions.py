"""Regression tests for proxy / cache / cassette / budget bugs found in the v0.2.1 bug hunt."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from llmtap.cache import cache_key
from llmtap.cassette import Cassette
from llmtap.config import Config
from llmtap.models import CachedResponse, Usage
from llmtap.pricing import cost_usd
from llmtap.providers import resolve_route
from llmtap.proxy import LLMTap

from test_hardening import Up, anth_json, call, last, llm_up, mk, msg, raw_http, send  # noqa: F401

SECRET = "sk-ant-api03-" + "A" * 40


def test_blocked_request_is_stored_redacted(mk):
    tap, up = mk(guard="block")
    st, _, _ = call(tap, "POST", "/v1/messages", msg(f"my key is {SECRET}"), {"x-api-key": "k"})
    assert st == 400 and not up.requests
    ex = last(tap)
    assert ex.source == "blocked"
    assert SECRET not in ex.request_body
    assert ex.guard_findings


@pytest.mark.parametrize("path", [
    "/v1beta/models/text-embedding-004:embedContent?key=AIzaXYZ",
    "/v1beta/models/gemini-2.5-flash:countTokens?key=AIzaXYZ",
    "/v1beta/models?key=AIzaXYZ",
    "/v1/models/gemini-2.5-flash:batchEmbedContents",
])
def test_gemini_endpoints_never_route_to_openai(path):
    assert resolve_route(path, {}, Config()).provider == "gemini"


def test_openai_paths_still_route_to_openai():
    assert resolve_route("/v1/chat/completions", {"authorization": "Bearer x"}, Config()).provider == "openai"
    assert resolve_route("/v1/models", {"authorization": "Bearer x"}, Config()).provider == "openai"


def test_cached_tokens_priced_by_wire_style_not_model_name():
    u = Usage(input_tokens=1000, cache_read_tokens=800, output_tokens=100)
    via_router = cost_usd("anthropic/claude-sonnet-4-5", u, style="openai")       # cached ⊂ prompt_tokens
    native = cost_usd("claude-sonnet-4-5", u, style="anthropic")                  # cached counted separately
    assert via_router < native
    assert via_router == pytest.approx((200 * 3 + 800 * 0.3 + 100 * 15) / 1e6)
    assert cost_usd("claude-sonnet-4-5", u) == native                             # no style: by name, as before


def test_non_generation_calls_are_not_priced(mk):
    tap, up = mk(up=llm_up({"input_tokens": 1000, "output_tokens": 500}))
    st, _, _ = call(tap, "GET", "/v1/responses/resp_abc", None, {"authorization": "Bearer k"})
    assert st == 200
    assert last(tap).cost_usd == 0.0


def test_no_duplicate_date_or_server_headers(mk):
    tap, up = mk()
    raw = raw_http(tap.port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n"
                             b"Content-Type: application/json\r\nConnection: close\r\n\r\n{}")
    head = raw.split(b"\r\n\r\n", 1)[0].lower().split(b"\r\n")
    assert sum(1 for h in head if h.startswith(b"date:")) == 1
    assert sum(1 for h in head if h.startswith(b"server:")) == 1


def test_cache_key_distinguishes_gemini_alt_but_ignores_other_query():
    b = b'{"contents":[]}'
    p = "/v1beta/models/g:streamGenerateContent"
    assert cache_key("gemini", p, b) != cache_key("gemini", p + "?alt=sse", b)
    assert cache_key("gemini", p + "?alt=sse&key=A", b) == cache_key("gemini", p + "?key=B&alt=sse", b)
    assert cache_key("anthropic", "/v1/messages?beta=true", b) == cache_key("anthropic", "/v1/messages", b)


def test_budget_does_not_block_free_cache_hits(mk):
    tap, up = mk(up=llm_up(), cache=True, budget_session_usd=0.0001)   # one call costs $0.000105
    assert call(tap, "POST", "/v1/messages", msg("a"), {"x-api-key": "k"})[0] == 200
    st, h, _ = call(tap, "POST", "/v1/messages", msg("a"), {"x-api-key": "k"})
    assert (st, h.get("x-llmtap-source")) == (200, "cache")
    st, h, _ = call(tap, "POST", "/v1/messages", msg("b"), {"x-api-key": "k"})
    assert (st, h.get("x-llmtap-source")) == (429, "budget")
    assert len(up.requests) == 1


def test_loop_breaker_still_counts_cache_hits(mk):
    tap, up = mk(up=llm_up(), cache=True, loop_threshold=2)
    sts = [call(tap, "POST", "/v1/messages", msg("same"), {"x-api-key": "k"})[0] for _ in range(3)]
    assert sts == [200, 200, 429]


def test_record_mode_bypasses_cache_so_the_cassette_is_complete(mk, tmp_path):
    cas = tmp_path / "rec.jsonl"
    tap, up = mk(up=llm_up(), cache=True, cassette=str(cas), mode="record")
    for _ in range(2):
        st, h, _ = call(tap, "POST", "/v1/messages", msg("x"), {"x-api-key": "k"})
        assert (st, h.get("x-llmtap-source")) == (200, "upstream")
    assert len(cas.read_text().splitlines()) == 2


def test_auto_mode_records_cache_served_responses(tmp_path):
    db = str(tmp_path / "t.db")
    up = llm_up()
    try:
        def tap_with(**kw):
            cfg = Config(port=0, quiet=True, guard="off", db_path=db, cache=True, **kw)
            cfg.upstreams["anthropic"] = {"base": up.base_url, "style": "anthropic"}
            return LLMTap(cfg).start_background()
        t1 = tap_with()
        assert call(t1, "POST", "/v1/messages", msg("warm"), {"x-api-key": "k"})[0] == 200
        t1.shutdown()
        cas = tmp_path / "auto.jsonl"
        t2 = tap_with(cassette=str(cas), mode="auto")
        st, h, _ = call(t2, "POST", "/v1/messages", msg("warm"), {"x-api-key": "k"})
        t2.shutdown()
        assert (st, h.get("x-llmtap-source")) == (200, "cache")
        assert cas.is_file() and len(cas.read_text().splitlines()) == 1
        replay = Cassette(str(cas), mode="replay")
        assert len(replay) == 1
    finally:
        up.stop()


def test_concurrent_requests_cannot_race_past_the_budget(mk):
    def slow(h, rec):
        time.sleep(0.4)
        send(h, 200, {"Content-Type": "application/json"}, anth_json())
    tap, up = mk(up=Up(slow), budget_session_usd=0.00015)              # one call costs $0.000105
    with ThreadPoolExecutor(8) as ex:
        sts = list(ex.map(lambda i: call(tap, "POST", "/v1/messages", msg(f"q{i}"), {"x-api-key": "k"})[0], range(8)))
    assert sts.count(200) <= 2, sts                                    # used to be 8 x 200
    assert sts.count(429) >= 6
    assert tap.budget.in_flight_usd() == 0.0                           # every reservation released


def test_reservation_released_when_upstream_fails(mk):
    tap, up = mk(budget_session_usd=1.0)
    up.stop()                                                          # connection refused
    call(tap, "POST", "/v1/messages", msg("x"), {"x-api-key": "k"})
    assert tap.budget.in_flight_usd() == 0.0


def test_cassette_with_utf8_bom_keeps_first_entry(tmp_path):
    p = tmp_path / "bom.jsonl"
    c = Cassette(str(p), mode="record")
    for k in ("k1", "k2"):
        c.record(k, {"m": k}, CachedResponse(status=200, headers={}, body="{}", streamed=False))
    p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())
    r = Cassette(str(p), mode="replay")
    assert len(r) == 2 and r.lookup("k1") is not None


def test_cassette_lines_use_lf_on_every_os(tmp_path):
    p = tmp_path / "lf.jsonl"
    Cassette(str(p), mode="record").record("k", {}, CachedResponse(status=200, headers={}, body="{}", streamed=False))
    assert b"\r\n" not in p.read_bytes()


def test_concurrent_cassette_writers_lose_no_lines(tmp_path):
    p = tmp_path / "shared.jsonl"
    writers = [Cassette(str(p), mode="auto") for _ in range(4)]          # separate instances = separate locks
    def work(i):
        for j in range(60):
            writers[i].record(f"k{i}-{j}", {"i": i, "j": j}, CachedResponse(status=200, headers={}, body="x" * 2000, streamed=False))
    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 240
    assert all(json.loads(l)["key"] for l in lines)


def test_gemini_json_array_stream_gets_real_usage(mk):
    chunks = [
        {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "lo"}]}}],
         "usageMetadata": {"promptTokenCount": 500, "candidatesTokenCount": 60, "totalTokenCount": 560},
         "modelVersion": "gemini-2.5-flash"},
    ]
    tap, up = mk(up=Up(lambda h, rec: send(h, 200, {"Content-Type": "application/json"}, json.dumps(chunks).encode())))
    tap.cfg.upstreams["gemini"] = {"base": up.base_url, "style": "gemini"}
    st, _, body = call(tap, "POST", "/v1beta/models/gemini-2.5-flash:streamGenerateContent",
                       {"contents": [{"parts": [{"text": "hi"}]}]}, {"x-goog-api-key": "g"})
    assert st == 200 and json.loads(body) == chunks
    ex = last(tap)
    assert (ex.usage.input_tokens, ex.usage.output_tokens) == (500, 60)
    assert ex.output_text == "Hello"
    assert ex.cost_usd > 0
