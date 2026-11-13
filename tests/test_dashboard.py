"""Tests for the dashboard JSON API / UI routes (llmtap.dashboard.handle_internal)."""
from __future__ import annotations

import json
import queue
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from llmtap import __version__, dashboard
from llmtap.config import Config
from llmtap.models import Exchange, Usage

try:  # store.py is written by another contributor; fall back to a stub if it isn't usable
    from llmtap.store import Store as _RealStore

    _probe = _RealStore(":memory:")
    _probe.close()
    HAVE_STORE = True
except Exception:  # pragma: no cover - only when store.py is missing/broken
    HAVE_STORE = False


class StubStore:
    """Tiny in-memory store implementing the bits the dashboard uses."""

    def __init__(self) -> None:
        self.items: list[Exchange] = []

    def record(self, ex: Exchange) -> None:
        self.items = [e for e in self.items if e.id != ex.id] + [ex]

    def get(self, id: str) -> Exchange | None:
        hits = [e for e in self.items if e.id == id or (len(id) >= 4 and e.id.startswith(id))]
        return hits[0] if len(hits) == 1 else None

    def list(self, limit: int = 50, offset: int = 0, q: str | None = None, since: float | None = None,
             provider: str | None = None, model: str | None = None, source: str | None = None) -> list[Exchange]:
        out = sorted(self.items, key=lambda e: e.ts, reverse=True)
        if q:
            out = [e for e in out if q in e.request_body or q in e.output_text or q in (e.model or "") or q in e.path]
        if since is not None:
            out = [e for e in out if e.ts >= since]
        for attr, val in (("provider", provider), ("model", model), ("source", source)):
            if val:
                out = [e for e in out if getattr(e, attr) == val]
        return out[offset: offset + limit]

    def stats(self, since: float | None = None, group_by: str | None = None) -> dict:
        items = [e for e in self.items if since is None or e.ts >= since]
        groups: dict[str, dict] = {}
        if group_by == "model":
            for e in items:
                g = groups.setdefault(e.model or "", {"key": e.model, "requests": 0, "cost_usd": 0.0,
                                                      "input_tokens": 0, "output_tokens": 0})
                g["requests"] += 1
                g["cost_usd"] += e.cost_usd
        return {"requests": len(items), "input_tokens": sum(e.usage.input_tokens for e in items),
                "output_tokens": sum(e.usage.output_tokens for e in items),
                "cost_usd": sum(e.cost_usd for e in items), "saved_usd": 0.0, "avg_latency_ms": 0.0,
                "errors": 0, "cache_hits": 0, "blocked": 0, "guard_findings": 0,
                "groups": sorted(groups.values(), key=lambda g: -g["cost_usd"])}

    def spent_since(self, since: float) -> float:
        return sum(e.cost_usd for e in self.items if e.ts >= since and e.source == "upstream")

    def timeseries(self, since: float, bucket_s: int = 3600) -> list[dict]:
        b: dict[int, dict] = {}
        for e in self.items:
            if e.ts >= since:
                t = int(e.ts // bucket_s * bucket_s)
                p = b.setdefault(t, {"t": t, "requests": 0, "cost_usd": 0.0, "tokens": 0})
                p["requests"] += 1
                p["cost_usd"] += e.cost_usd
                p["tokens"] += e.usage.total()
        return [b[k] for k in sorted(b)]

    def count(self) -> int:
        return len(self.items)

    def clear(self) -> None:
        self.items = []

    def close(self) -> None:
        pass


class StubBus:
    def __init__(self) -> None:
        self.subs: list[queue.Queue] = []
        self.unsubscribed = threading.Event()

    def publish(self, event: dict) -> None:
        for q in list(self.subs):
            q.put(event)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        self.subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        if q in self.subs:
            self.subs.remove(q)
        self.unsubscribed.set()


class Recorder:
    """Records what the dashboard sends. ``fail_after`` simulates a client disconnect."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self.chunks: list[bytes] = []
        self.ended = False
        self.fail_after = fail_after
        self.got_data = threading.Event()

    def start(self, status: int, headers: dict[str, str]) -> None:
        assert self.status is None, "start() called twice"
        self.status = status
        self.headers = {k.lower(): v for k, v in headers.items()}

    def write(self, chunk: bytes) -> None:
        self.chunks.append(chunk)
        if chunk.startswith(b"data:"):
            self.got_data.set()
        if self.fail_after is not None and len(self.chunks) > self.fail_after:
            raise BrokenPipeError("client went away")

    def end(self) -> None:
        self.ended = True

    @property
    def body(self) -> bytes:
        return b"".join(self.chunks)

    def json(self) -> Any:
        return json.loads(self.body)


def anthropic_ex(**kw: Any) -> Exchange:
    req = {"model": "claude-sonnet-4-5", "max_tokens": 100, "system": "be terse",
           "messages": [{"role": "user", "content": "first question"},
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": [{"type": "text", "text": "What is   the capital\nof France?"}]}]}
    base = dict(provider="anthropic", style="anthropic", path="/v1/messages", model="claude-sonnet-4-5",
                request_body=json.dumps(req), status=200, response_body='{"content":[{"type":"text","text":"Paris"}]}',
                usage=Usage(input_tokens=20, output_tokens=3), cost_usd=0.0012, output_text="Paris", latency_ms=120.0)
    base.update(kw)
    return Exchange(**base)


def openai_ex(**kw: Any) -> Exchange:
    req = {"model": "gpt-4o", "messages": [{"role": "system", "content": "sys"},
                                           {"role": "user", "content": "Translate hello to German"}]}
    base = dict(provider="openai", style="openai", path="/v1/chat/completions", model="gpt-4o",
                request_body=json.dumps(req), status=200, response_body='{"choices":[]}',
                usage=Usage(input_tokens=10, output_tokens=2), cost_usd=0.0004, output_text="Hallo", latency_ms=80.0)
    base.update(kw)
    return Exchange(**base)


@pytest.fixture
def tap() -> SimpleNamespace:
    store = _RealStore(":memory:") if HAVE_STORE else StubStore()
    cfg = Config(db_path=":memory:", guard="redact", cache=False, budget_daily_usd=5.0, loop_threshold=4)
    t = SimpleNamespace(store=store, cache=None, cassette=None, cfg=cfg,
                        budget=SimpleNamespace(session_spent=lambda: 0.25), bus=StubBus(), port=4141)
    yield t
    store.close()


def call(tap: Any, path: str, method: str = "GET", headers: dict | None = None, body: bytes = b"") -> Recorder:
    r = Recorder()
    handled = dashboard.handle_internal(tap, method, path, headers or {}, body, r)
    assert handled is True
    assert r.ended
    return r


def test_not_internal_path_returns_false(tap: Any) -> None:
    r = Recorder()
    assert dashboard.handle_internal(tap, "GET", "/v1/messages", {}, b"", r) is False
    assert r.status is None


def test_health(tap: Any) -> None:
    r = call(tap, "/_llmtap/health")
    assert r.status == 200
    assert r.json() == {"ok": True, "version": __version__}
    assert r.headers["content-type"].startswith("application/json")
    assert r.headers["content-length"] == str(len(r.body))


def test_index_html(tap: Any) -> None:
    r = call(tap, "/_llmtap/")
    assert r.status == 200
    assert r.headers["content-type"].startswith("text/html")
    assert b"<html" in r.body and b"llmtap" in r.body
    assert int(r.headers["content-length"]) == len(r.body)
    # self-contained: no external scripts / stylesheets
    assert b"<script src" not in r.body and b"https://" not in r.body


def test_bare_prefix_redirects(tap: Any) -> None:
    r = call(tap, "/_llmtap")
    assert r.status == 302 and r.headers["location"] == "/_llmtap/"


def test_unknown_route_404(tap: Any) -> None:
    for path in ("/_llmtap/nope", "/_llmtap/api/nope", "/_llmtap/api/exchanges/doesnotexist"):
        r = call(tap, path)
        assert r.status == 404, path
        assert "error" in r.json()


def test_exchanges_list_and_filters(tap: Any) -> None:
    now = time.time()
    a = anthropic_ex(ts=now - 10)
    o = openai_ex(ts=now - 5)
    c = openai_ex(ts=now - 1, source="cache", cost_usd=0.0, saved_usd=0.0004)
    for ex in (a, o, c):
        tap.store.record(ex)
    items = call(tap, "/_llmtap/api/exchanges").json()["items"]
    assert [i["id"] for i in items] == [c.id, o.id, a.id]
    first = items[-1]
    assert "request_body" not in first and "response_body" not in first
    assert first["preview"] == "What is the capital of France?"
    assert first["output_preview"] == "Paris"
    assert first["usage"]["input_tokens"] == 20

    items = call(tap, "/_llmtap/api/exchanges?provider=anthropic").json()["items"]
    assert [i["id"] for i in items] == [a.id]
    items = call(tap, "/_llmtap/api/exchanges?source=cache").json()["items"]
    assert [i["id"] for i in items] == [c.id]
    items = call(tap, "/_llmtap/api/exchanges?q=German&limit=1").json()["items"]
    assert len(items) == 1 and items[0]["preview"] == "Translate hello to German"
    items = call(tap, "/_llmtap/api/exchanges?limit=1&offset=1").json()["items"]
    assert [i["id"] for i in items] == [o.id]
    items = call(tap, "/_llmtap/api/exchanges?limit=garbage").json()["items"]
    assert len(items) == 3


def test_exchange_detail(tap: Any) -> None:
    ex = anthropic_ex()
    tap.store.record(ex)
    d = call(tap, f"/_llmtap/api/exchanges/{ex.id}").json()
    assert d["id"] == ex.id
    assert json.loads(d["request_body"])["model"] == "claude-sonnet-4-5"
    assert d["response_body"].startswith("{")
    assert d["usage"]["output_tokens"] == 3


def test_stats(tap: Any) -> None:
    tap.store.record(anthropic_ex())
    tap.store.record(openai_ex())
    d = call(tap, "/_llmtap/api/stats?since=3600&group_by=model").json()
    assert d["requests"] == 2
    assert d["session_spent"] == 0.25
    assert d["cache"] is None and d["cassette"] is None
    assert d["config"]["guard"] == "redact"
    assert d["config"]["budget_daily_usd"] == 5.0
    assert d["config"]["loop_threshold"] == 4
    assert d["config"]["cache"] is False
    assert {g["key"] for g in d["groups"]} == {"claude-sonnet-4-5", "gpt-4o"}
    assert d["spent_today"] == pytest.approx(0.0016)
    # since as an absolute epoch in the future -> nothing
    assert call(tap, f"/_llmtap/api/stats?since={time.time() + 1000:.0f}").json()["requests"] == 0
    assert call(tap, "/_llmtap/api/stats?since=all").json()["requests"] == 2


def test_stats_with_cache_and_cassette(tap: Any) -> None:
    tap.cache = SimpleNamespace(stats=lambda: {"entries": 3, "hits": 7})

    class Cas:
        path = "ci.jsonl"
        mode = "replay"

        def __len__(self) -> int:
            return 5

    tap.cassette = Cas()
    d = call(tap, "/_llmtap/api/stats").json()
    assert d["cache"] == {"entries": 3, "hits": 7}
    assert d["cassette"] == {"path": "ci.jsonl", "mode": "replay", "entries": 5}


def test_timeseries(tap: Any) -> None:
    now = time.time()
    tap.store.record(anthropic_ex(ts=now - 30))
    tap.store.record(openai_ex(ts=now - 20))
    d = call(tap, "/_llmtap/api/timeseries?since=3600&bucket=60").json()
    assert d["bucket"] == 60
    pts = d["points"]
    assert sum(p["requests"] for p in pts) == 2
    assert sum(p["cost_usd"] for p in pts) == pytest.approx(0.0016)


def test_clear(tap: Any) -> None:
    tap.store.record(anthropic_ex())
    r = call(tap, "/_llmtap/api/clear", method="POST", headers={"Host": "127.0.0.1:4141", "Origin": "http://127.0.0.1:4141"})
    assert r.status == 200 and r.json() == {"ok": True}
    assert tap.store.count() == 0


def test_clear_rejects_cross_origin(tap: Any) -> None:
    tap.store.record(anthropic_ex())
    r = call(tap, "/_llmtap/api/clear", method="POST", headers={"Host": "127.0.0.1:4141", "Origin": "https://evil.example"})
    assert r.status == 403
    assert tap.store.count() == 1


def test_get_clear_is_404(tap: Any) -> None:
    assert call(tap, "/_llmtap/api/clear").status == 404


# --------------------------------------------------------------------------------------------
# previews
# --------------------------------------------------------------------------------------------

def test_preview_anthropic_blocks_and_tool_result() -> None:
    body = {"messages": [{"role": "user", "content": "hi"},
                         {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "42 degrees"}]}]}
    assert dashboard.request_preview(json.dumps(body)) == "42 degrees"


def test_preview_openai_list_content() -> None:
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "look at"},
                                                      {"type": "image_url", "image_url": {"url": "data:..."}}]},
                         {"role": "assistant", "content": "a cat"}]}
    assert dashboard.request_preview(json.dumps(body)) == "look at"


def test_preview_responses_api_and_embeddings() -> None:
    assert dashboard.request_preview(json.dumps({"model": "gpt-5", "input": "Write a haiku"})) == "Write a haiku"
    items = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "hello there"}]}]}
    assert dashboard.request_preview(json.dumps(items)) == "hello there"
    assert dashboard.request_preview(json.dumps({"input": ["a", "b"]})) == "b"
    assert dashboard.request_preview(json.dumps({"prompt": "complete me"})) == "complete me"


def test_preview_truncates_and_handles_garbage() -> None:
    long = "x" * 500
    p = dashboard.request_preview(json.dumps({"messages": [{"role": "user", "content": long}]}))
    assert len(p) == 200 and p.endswith("…")
    assert dashboard.request_preview("not json") == "not json"
    assert dashboard.request_preview("") == ""
    assert dashboard.request_preview("[1,2]") == "[1,2]"


def test_summary_from_exchange_and_dict() -> None:
    ex = openai_ex(error=None)
    s = dashboard.summary(ex)
    assert s["preview"] == "Translate hello to German"
    assert s["output_preview"] == "Hallo"
    assert "request_body" not in s and "response_body" not in s
    s2 = dashboard.summary(anthropic_ex(output_text="", error="upstream timeout").to_dict())
    assert s2["output_preview"] == "upstream timeout"


# --------------------------------------------------------------------------------------------
# SSE
# --------------------------------------------------------------------------------------------

def test_sse_one_event_then_disconnect(tap: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dashboard, "HEARTBEAT_S", 0.05)
    # 1st write = "connected" preamble, then pings / the event; fail on the write after the event
    r = Recorder()
    orig_write = r.write

    def write(chunk: bytes) -> None:
        orig_write(chunk)
        if chunk.startswith(b"data:"):
            raise BrokenPipeError("bye")

    r.write = write  # type: ignore[method-assign]
    th = threading.Thread(target=dashboard.handle_internal, args=(tap, "GET", "/_llmtap/api/events", {}, b"", r), daemon=True)
    th.start()
    deadline = time.time() + 3
    while not tap.bus.subs and time.time() < deadline:
        time.sleep(0.01)
    assert tap.bus.subs, "SSE handler never subscribed"
    time.sleep(0.12)  # let at least one heartbeat go out
    ex = anthropic_ex()
    tap.bus.publish({"type": "exchange", "exchange": ex.to_dict()})
    assert r.got_data.wait(3)
    th.join(3)
    assert not th.is_alive()
    assert tap.bus.unsubscribed.is_set() and not tap.bus.subs
    assert r.status == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["transfer-encoding"] == "chunked"
    assert b": ping\n\n" in r.chunks
    data = [c for c in r.chunks if c.startswith(b"data:")][0]
    assert data.endswith(b"\n\n")
    ev = json.loads(data[len(b"data: "):].decode())
    assert ev["type"] == "exchange"
    assert ev["exchange"]["id"] == ex.id
    assert ev["exchange"]["preview"] == "What is the capital of France?"
    assert "request_body" not in ev["exchange"]  # bodies stripped before streaming
